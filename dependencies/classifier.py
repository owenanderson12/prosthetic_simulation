import numpy as np
import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from collections import deque

class Classifier:
    """
    Classifier module for EEG-based motor imagery classification.
    
    Implements:
    - LDA with shrinkage for left vs. right motor imagery classification
    - Continuous probability estimation
    - Decision smoothing via state machine
    - Adaptive thresholding
    - Confidence estimation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the classifier.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Classification parameters
        self.threshold = config.get('CLASSIFIER_THRESHOLD', 0.65)
        self.min_confidence = config.get('MIN_CONFIDENCE', 0.55)
        
        # Classifier
        self.classifier = None
        self.classes = ['left', 'right']  # Class labels
        self.class_map = {0: 'left', 1: 'right'}  # Map indices to class labels
        
        # State tracking
        self.current_state = 'idle'
        self.state_history = deque(maxlen=10)  # Last 10 states for smoothing
        self.probabilities_history = deque(maxlen=10)  # Last 10 probability estimates
        
        # Adaptive threshold variables
        self.performance_history = deque(maxlen=100)
        self.adaptive_threshold = self.threshold
        self.use_adaptive_threshold = config.get('ADAPTIVE_THRESHOLD', True)
        
        # Features that were most recently used for classification
        self.last_features = None
        
        # Path for model saving/loading
        self.model_dir = config.get('MODEL_DIR', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Logging initialized state
        logging.info("Classifier initialized with threshold: %.2f", self.threshold)
    
    def train(self, 
              features_left: List[np.ndarray], 
              features_right: List[np.ndarray], 
              shrinkage: Optional[float] = None) -> float:
        """
        Train the LDA classifier on left vs. right motor imagery feature vectors.
        
        Args:
            features_left: List of feature vectors for left hand imagery
            features_right: List of feature vectors for right hand imagery
            shrinkage: Shrinkage parameter for LDA (None for automatic)
            
        Returns:
            Cross-validation accuracy score
        """
        try:
            if not features_left or not features_right:
                logging.error("Cannot train classifier: No training examples provided")
                return 0.0
                
            # Prepare training data and labels
            X_train = np.vstack([np.array(features_left), np.array(features_right)])
            y_train = np.hstack([
                np.zeros(len(features_left)),  # 0 = left
                np.ones(len(features_right))   # 1 = right
            ])
            
            # Create and train LDA classifier with shrinkage
            self.classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage)
            
            # Compute cross-validation score
            cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
            accuracy = cv_scores.mean()
            
            # Fit on all data
            self.classifier.fit(X_train, y_train)
            
            logging.info(f"Classifier trained with {len(features_left)} left and {len(features_right)} right examples")
            logging.info(f"Cross-validation accuracy: {accuracy:.3f}")
            
            # Reset state
            self.state_history.clear()
            self.probabilities_history.clear()
            self.current_state = 'idle'
            
            return accuracy
            
        except Exception as e:
            logging.exception("Error during classifier training:")
            return 0.0
    
    def classify(self, features: Dict) -> Dict:
        """
        Classify a feature vector as left hand, right hand, or idle.
        
        Args:
            features: Dictionary of features from signal processor
            
        Returns:
            Classification results dictionary
        """
        try:
            self.last_features = features
            
            # Check for valid classifier
            if self.classifier is None:
                return {
                    'class': 'idle',
                    'probability': 0.5,
                    'confidence': 0.0,
                    'valid': False,
                    'message': 'Classifier not trained'
                }
            
            # Check if we have CSP features
            if features.get('csp_features') is not None:
                feature_vector = features['csp_features']
            else:
                # Fallback to basic features
                # Construct feature vector from ERD/ERS values
                mu_erd = features.get('erd_mu', [0])
                beta_erd = features.get('erd_beta', [0])
                
                # Focus on motor imagery channels
                feature_vector = np.hstack([
                    mu_erd,
                    beta_erd
                ])
            
            # Reshape for sklearn
            feature_vector = feature_vector.reshape(1, -1)
            
            # Get classification and probability
            class_idx = self.classifier.predict(feature_vector)[0]
            probabilities = self.classifier.predict_proba(feature_vector)[0]
            
            # Get probability for the predicted class
            class_probability = probabilities[int(class_idx)]
            
            # Store probability in history
            self.probabilities_history.append(class_probability)
            
            # Determine confidence based on recent probability stability
            confidence = self._calculate_confidence(class_probability)
            
            # Update state and apply state machine logic
            class_name = self.class_map[class_idx]
            smoothed_state = self._update_state(class_name, class_probability, confidence)
            
            # Add to performance history if confident
            if confidence > 0.7:
                self.performance_history.append({
                    'predicted': class_name,
                    'probability': class_probability,
                    'confidence': confidence
                })
                
                # Potentially update adaptive threshold
                if self.use_adaptive_threshold and len(self.performance_history) >= 20:
                    self._update_adaptive_threshold()
            
            return {
                'class': smoothed_state,
                'raw_class': class_name,
                'probability': class_probability,
                'probabilities': dict(zip(self.classes, probabilities)),
                'confidence': confidence,
                'threshold': self.adaptive_threshold,
                'valid': True
            }
            
        except Exception as e:
            logging.exception("Error during classification:")
            return {
                'class': 'idle',
                'probability': 0.5,
                'confidence': 0.0,
                'valid': False,
                'message': str(e)
            }
    
    def _calculate_confidence(self, probability: float) -> float:
        """
        Calculate confidence score based on probability and its stability.
        
        Args:
            probability: Current classification probability
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence on distance from decision boundary
        base_confidence = 2 * abs(probability - 0.5)
        
        # If we have history, incorporate stability
        if len(self.probabilities_history) > 1:
            # Standard deviation of recent probabilities (lower means more stable)
            stability = 1.0 - min(1.0, 2 * np.std(list(self.probabilities_history)))
            
            # Combine base confidence with stability
            combined_confidence = 0.7 * base_confidence + 0.3 * stability
        else:
            combined_confidence = base_confidence
        
        return combined_confidence
    
    def _update_state(self, class_name: str, probability: float, confidence: float) -> str:
        """
        Update the current state based on classification and confidence.
        Implements a simple state machine for smoothing.
        
        Args:
            class_name: Classified class name
            probability: Classification probability
            confidence: Classification confidence
            
        Returns:
            Smoothed state
        """
        # Add current raw classification to history
        self.state_history.append(class_name)
        
        # If confidence is too low, maintain current state or return to idle
        if confidence < self.min_confidence:
            # If we've been in a state for a while, maintain it despite low confidence
            if self.current_state != 'idle' and len(self.state_history) >= 3:
                recent_states = list(self.state_history)[-3:]
                # If most recent states agree with current state, maintain it
                if recent_states.count(self.current_state) >= 2:
                    return self.current_state
            
            # Default to idle on low confidence
            return 'idle'
        
        # If probability doesn't meet threshold, return to idle
        effective_threshold = self.adaptive_threshold if self.use_adaptive_threshold else self.threshold
        if probability < effective_threshold:
            return 'idle'
        
        # Majority voting for smoothing
        if len(self.state_history) >= 3:
            recent_states = list(self.state_history)[-3:]
            left_count = recent_states.count('left')
            right_count = recent_states.count('right')
            
            # Strong majority for a class
            if left_count >= 2 and left_count > right_count:
                next_state = 'left'
            elif right_count >= 2 and right_count > left_count:
                next_state = 'right'
            else:
                # No strong majority, use current classification if confident
                next_state = class_name if confidence > 0.75 else self.current_state
        else:
            # Not enough history for voting, use current classification
            next_state = class_name
        
        # Update current state
        self.current_state = next_state
        return next_state
    
    def _update_adaptive_threshold(self) -> None:
        """Update the adaptive threshold based on performance history."""
        # Calculate success rate (how often classifications are confident)
        confident_count = sum(1 for p in self.performance_history if p['confidence'] > 0.7)
        success_rate = confident_count / len(self.performance_history)
        
        # Target success rate is around 0.8
        if success_rate < 0.7:
            # Too many uncertain classifications, lower threshold
            self.adaptive_threshold = max(0.55, self.adaptive_threshold - 0.01)
        elif success_rate > 0.9:
            # Too many "easy" classifications, increase threshold
            self.adaptive_threshold = min(0.8, self.adaptive_threshold + 0.01)
            
        logging.debug(f"Adaptive threshold updated to {self.adaptive_threshold:.2f} (success rate: {success_rate:.2f})")
    
    def save_model(self, filename: str) -> bool:
        """
        Save the trained classifier model to disk.
        
        Args:
            filename: Name of the file to save the model
            
        Returns:
            Success indicator
        """
        if self.classifier is None:
            logging.error("Cannot save model: No trained classifier")
            return False
            
        try:
            model_path = os.path.join(self.model_dir, filename)
            with open(model_path, 'wb') as f:
                model_data = {
                    'classifier': self.classifier,
                    'classes': self.classes,
                    'class_map': self.class_map,
                    'threshold': self.threshold,
                    'adaptive_threshold': self.adaptive_threshold
                }
                pickle.dump(model_data, f)
                
            logging.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logging.exception("Error saving model:")
            return False
    
    def load_model(self, filename: str) -> bool:
        """
        Load a trained classifier model from disk.
        
        Args:
            filename: Name of the file to load the model from
            
        Returns:
            Success indicator
        """
        try:
            model_path = os.path.join(self.model_dir, filename)
            if not os.path.exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                return False
                
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.classifier = model_data['classifier']
            self.classes = model_data.get('classes', ['left', 'right'])
            self.class_map = model_data.get('class_map', {0: 'left', 1: 'right'})
            self.threshold = model_data.get('threshold', 0.65)
            self.adaptive_threshold = model_data.get('adaptive_threshold', self.threshold)
            
            # Reset state
            self.state_history.clear()
            self.probabilities_history.clear()
            self.current_state = 'idle'
            
            logging.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logging.exception("Error loading model:")
            return False
    
    def reset(self) -> None:
        """Reset classifier state (but keeps trained model)."""
        self.state_history.clear()
        self.probabilities_history.clear()
        self.performance_history.clear()
        self.current_state = 'idle'
        self.adaptive_threshold = self.threshold
        logging.info("Classifier state reset")
