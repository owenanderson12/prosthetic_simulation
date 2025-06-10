import numpy as np
import logging
import pickle
import os
import sys
from typing import Dict, List, Tuple, Optional, Union
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from collections import deque

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Classifier:
    """
    Classifier module for EEG-based motor imagery classification.
    
    Implements:
    - Support for both LDA and sklearn pipeline models (e.g., Random Forest)
    - Continuous probability estimation
    - Decision smoothing via state machine
    - Adaptive thresholding
    - Confidence estimation
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize the classifier.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.config = config_dict
        
        # Classification parameters
        self.threshold = config_dict.get('CLASSIFIER_THRESHOLD', 0.65)
        self.min_confidence = config_dict.get('MIN_CONFIDENCE', 0.55)
        
        # Classifier
        self.classifier = None
        self.model_type = None  # 'lda' or 'pipeline'
        self.classes = ['left', 'right']  # Class labels
        self.class_map = {0: 'left', 1: 'right'}  # Map indices to class labels
        
        # State tracking
        self.current_state = 'idle'
        self.state_history = deque(maxlen=10)  # Last 10 states for smoothing
        self.probabilities_history = deque(maxlen=10)  # Last 10 probability estimates
        
        # Adaptive threshold variables
        self.performance_history = deque(maxlen=100)
        self.adaptive_threshold = self.threshold
        self.use_adaptive_threshold = config_dict.get('ADAPTIVE_THRESHOLD', True)
        
        # Features that were most recently used for classification
        self.last_features = None
        
        # Path for model saving/loading
        self.model_dir = config_dict.get('MODEL_DIR', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Logging initialized state
        logging.info("Classifier initialized with threshold: %.2f", self.threshold)
    
    def train(self, 
              features_left: List[np.ndarray], 
              features_right: List[np.ndarray], 
              shrinkage: Optional[Union[str, float]] = 'auto',
              covariance_estimator: Optional[object] = None) -> float:
        """
        Train the LDA classifier on left vs. right motor imagery feature vectors.
        
        Args:
            features_left: List of feature vectors for left hand imagery
            features_right: List of feature vectors for right hand imagery
            shrinkage: Shrinkage parameter for LDA ('auto' or float value)
            covariance_estimator: Custom covariance estimator (added in sklearn 0.24)
            
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
            
            # Create and train LDA classifier with updated parameters
            self.classifier = LinearDiscriminantAnalysis(
                solver='lsqr',
                shrinkage=shrinkage,
                covariance_estimator=covariance_estimator,
                store_covariance=True,
                tol=0.0001
            )
            self.model_type = 'lda'
            
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
            
            # Prepare feature vector based on what's available
            feature_vector = self._prepare_feature_vector(features)
            
            if feature_vector is None:
                return {
                    'class': 'idle',
                    'probability': 0.5,
                    'confidence': 0.0,
                    'valid': False,
                    'message': 'No valid features found'
                }
            
            # Reshape for sklearn
            feature_vector = feature_vector.reshape(1, -1)
            
            # Get classification and probability based on model type
            if self.model_type == 'pipeline':
                # For pipeline models (e.g., Random Forest)
                class_idx = self.classifier.predict(feature_vector)[0]
                probabilities = self.classifier.predict_proba(feature_vector)[0]
            else:
                # For LDA models (backward compatibility)
                class_idx = self.classifier.predict(feature_vector)[0]
                probabilities = self.classifier.predict_proba(feature_vector)[0]
            
            # Get probability for the predicted class
            class_probability = probabilities[int(class_idx)]
            
            # Store probability in history
            self.probabilities_history.append(class_probability)
            
            # Determine confidence based on recent probability stability
            confidence = self._calculate_confidence(class_probability)
            
            # Update state and apply state machine logic
            class_name = self.class_map[int(class_idx)]
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
    
    def _prepare_feature_vector(self, features: Dict) -> Optional[np.ndarray]:
        """
        Prepare the feature vector for classification based on available features.
        Matches exactly how features were extracted during training.
        
        Args:
            features: Dictionary of features from signal processor
            
        Returns:
            Prepared feature vector or None if no valid features
        """
        # Get CSP features (required)
        csp_features = features.get('csp_features')
        if csp_features is None:
            logging.warning("No CSP features available")
            return None
        
        # Get band power features
        mu_erd = features.get('erd_mu', [])
        beta_erd = features.get('erd_beta', [])
        
        # For pipeline models, try to use both CSP and band power features
        if self.model_type == 'pipeline':
            if len(mu_erd) > 0 and len(beta_erd) > 0:
                # Combine features in the same order as training
                combined_vector = np.hstack([csp_features, mu_erd, beta_erd])
                return combined_vector
            else:
                # Fallback to CSP-only features if band power features are missing
                logging.warning("Missing band power features, using CSP features only")
                return csp_features
        else:
            # For LDA models, use only CSP features
            return csp_features
    
    def _get_expected_feature_count(self) -> int:
        """
        Try to determine the expected number of features from the model.
        
        Returns:
            Expected number of features, or 0 if cannot be determined
        """
        try:
            if self.model_type == 'pipeline' and hasattr(self.classifier, 'named_steps'):
                scaler = self.classifier.named_steps.get('scaler')
                if scaler and hasattr(scaler, 'n_features_in_'):
                    return scaler.n_features_in_
                elif scaler and hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                    return len(scaler.scale_)
            
            # For other models, try to infer from the classifier
            if hasattr(self.classifier, 'n_features_in_'):
                return self.classifier.n_features_in_
            elif hasattr(self.classifier, 'coef_') and self.classifier.coef_ is not None:
                return self.classifier.coef_.shape[1]
                
        except Exception as e:
            logging.warning(f"Could not determine expected feature count: {e}")
        
        return 0
    
    def _adjust_feature_length(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """
        Adjust the length of a feature vector to match the target length.
        
        Args:
            features: Input feature vector
            target_length: Desired length
            
        Returns:
            Adjusted feature vector
        """
        features = np.asarray(features)
        current_length = len(features)
        
        if current_length == target_length:
            return features
        elif current_length > target_length:
            # Truncate to target length (keep first features)
            return features[:target_length]
        else:
            # Pad with zeros or repeat last value
            padding = np.zeros(target_length - current_length)
            if current_length > 0:
                # Repeat the last value instead of zeros for better continuity
                padding.fill(features[-1])
            return np.hstack([features, padding])
    
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
        effective_threshold = self.adaptive_threshold if self.use_adaptive_threshold and self.adaptive_threshold is not None else self.threshold
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
            
        logging.info(f"Adaptive threshold updated to {self.adaptive_threshold:.2f} (success rate: {success_rate:.2f})")
    
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
                    'model_type': self.model_type,
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
    
    def is_aggregate_model(self, model_data: Dict) -> bool:
        """
        Check if a loaded model is an aggregate model from train_aggregate_models.py.
        
        Args:
            model_data: Dictionary containing model data
            
        Returns:
            True if the model is an aggregate model, False otherwise
        """
        # Aggregate models have these specific keys
        required_keys = {'classifier', 'features', 'labels', 'classes', 'class_map'}
        return all(key in model_data for key in required_keys)

    def load_model(self, filename: str) -> bool:
        """
        Load a trained classifier model from file.
        
        Args:
            filename: Path to the model file
            
        Returns:
            Success indicator
        """
        try:
            # Try different possible paths for the model file
            possible_paths = [
                os.path.join(self.model_dir, filename),  # Direct in model dir
                os.path.join('models', filename),        # In models directory
                os.path.join('calibration', 'random_forest', filename),  # In calibration/random_forest
                filename  # Absolute path
            ]
            
            filepath = None
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath is None:
                logging.error(f"Model file not found in any of these locations: {possible_paths}")
                return False
            
            logging.info(f"Loading model from: {filepath}")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if this is an aggregate model
            if 'features' in model_data and 'labels' in model_data:
                logging.info("Loading aggregate model with CSP filters")
                
                # Extract CSP filters from the model data
                if 'csp_filters' in model_data:
                    self.csp_filters = model_data['csp_filters']
                    self.csp_patterns = model_data.get('csp_patterns')
                    self.csp_mean = model_data.get('csp_mean')
                    self.csp_std = model_data.get('csp_std')
                    logging.info(f"Loaded CSP filters: shape {self.csp_filters.shape}")
                else:
                    logging.error("CSP filters not found in aggregate model")
                    return False
            
            # Update classifier with loaded model
            self.classifier = model_data['classifier']
            
            # Determine model type if not explicitly set
            if 'model_type' in model_data:
                self.model_type = model_data['model_type']
            else:
                # Auto-detect model type based on classifier type
                from sklearn.pipeline import Pipeline
                if isinstance(self.classifier, Pipeline):
                    self.model_type = 'pipeline'
                    logging.info("Auto-detected model type as 'pipeline'")
                else:
                    self.model_type = 'lda'
                    logging.info("Auto-detected model type as 'lda'")
            
            self.classes = model_data.get('classes', [0., 1.])
            self.class_map = model_data.get('class_map', {0: 'left', 1: 'right'})
            self.threshold = model_data.get('threshold', self.config.get('CLASSIFIER_THRESHOLD', 0.65))
            # Initialize adaptive_threshold properly, fallback to threshold if not in model
            self.adaptive_threshold = model_data.get('adaptive_threshold', self.threshold)
            
            logging.info(f"Loaded {self.model_type} model successfully")
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
