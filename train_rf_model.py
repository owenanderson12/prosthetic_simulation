#!/usr/bin/env python3
"""
train_rf_model.py

Train a Random Forest model using all available session data and save it
in the calibration format for the BCI system.
"""

import os
import sys
import logging
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from dependencies.signal_processor import SignalProcessor

# Configuration
MODEL_DIR = 'models'
CALIBRATION_DIR = 'calibration/random_forest'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CALIBRATION_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_best_aggregate_model():
    """Load the best pre-trained Random Forest model."""
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def create_minimal_calibration_data():
    """Create minimal calibration data structure."""
    # Create dummy calibration data structure
    # This is needed for the calibration loading system to work
    return {
        'baseline_data': np.random.randn(1000, 8),  # Dummy baseline
        'left_data': np.array([np.random.randn(500, 8) for _ in range(5)], dtype=object),
        'right_data': np.array([np.random.randn(500, 8) for _ in range(5)], dtype=object),
        'left_features': np.random.randn(100, 4),  # Dummy features
        'right_features': np.random.randn(100, 4)
    }

def save_rf_calibration_model(model_data, calibration_data):
    """Save Random Forest model in calibration format."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save calibration data (.npz format)
    calibration_filename = f"rf_calibration_{timestamp_str}.npz"
    calibration_path = os.path.join(CALIBRATION_DIR, calibration_filename)
    np.savez(calibration_path, **calibration_data)
    
    # Save Random Forest model (.pkl format)
    model_filename = f"rf_model_{timestamp_str}.pkl"
    model_path = os.path.join(CALIBRATION_DIR, model_filename)
    
    # Prepare model data in the format expected by the classifier
    rf_model_data = {
        'classifier': model_data['classifier'],
        'model_type': 'pipeline',
        'classes': model_data['classes'],
        'class_map': model_data['class_map'],
        'threshold': model_data['threshold'],
        'adaptive_threshold': model_data.get('adaptive_threshold')
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model_data, f)
    
    logging.info(f"Random Forest calibration saved to {calibration_path}")
    logging.info(f"Random Forest model saved to {model_path}")
    
    return calibration_filename, model_filename

def train_optimized_rf_model():
    """Train an optimized Random Forest model using the aggregate training approach."""
    
    # Load the best pre-trained model to get features and labels
    try:
        best_model_data = load_best_aggregate_model()
        X = best_model_data['features']
        y = best_model_data['labels']
        
        logging.info(f"Loaded training data: {X.shape[0]} samples, {X.shape[1]} features")
        logging.info(f"Class distribution: {np.bincount(y.astype(int))}")
        
    except FileNotFoundError:
        logging.error("Best model not found. Please run train_aggregate_models.py first.")
        return None
    
    # Create optimized Random Forest pipeline for the actual data size
    n_samples = X.shape[0]
    
    # Adjust parameters based on dataset size
    if n_samples < 200:
        # Small dataset
        n_estimators = 50
        max_depth = 5
        min_samples_split = 10
        min_samples_leaf = 5
    elif n_samples < 1000:
        # Medium dataset  
        n_estimators = 100
        max_depth = 8
        min_samples_split = 5
        min_samples_leaf = 2
    else:
        # Large dataset
        n_estimators = 200
        max_depth = 15
        min_samples_split = 2
        min_samples_leaf = 1
    
    logging.info(f"Using Random Forest parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Cross-validation
    cv_folds = min(5, n_samples // 20)  # Adjust CV folds based on sample size
    if cv_folds < 2:
        cv_folds = 2
        
    cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy')
    accuracy = cv_scores.mean()
    std = cv_scores.std()
    
    logging.info(f"Cross-validation accuracy: {accuracy:.3f} ± {std:.3f}")
    
    # Train on all data
    pipeline.fit(X, y)
    
    # Prepare model data
    model_data = {
        'classifier': pipeline,
        'features': X,
        'labels': y,
        'classes': [0., 1.],
        'class_map': {0: 'left', 1: 'right'},
        'threshold': config.CLASSIFIER_THRESHOLD,
        'adaptive_threshold': None,
        'cv_accuracy': accuracy,
        'cv_std': std
    }
    
    return model_data

def main():
    """Main function to train and save Random Forest model."""
    print("=" * 60)
    print("OPTIMIZED RANDOM FOREST MODEL TRAINING")
    print("=" * 60)
    
    # Train the model
    model_data = train_optimized_rf_model()
    if model_data is None:
        return 1
    
    # Create minimal calibration data structure
    calibration_data = create_minimal_calibration_data()
    
    # Save in calibration format
    cal_file, model_file = save_rf_calibration_model(model_data, calibration_data)
    
    print(f"✅ Random Forest model trained successfully!")
    print(f"📊 Cross-validation accuracy: {model_data['cv_accuracy']:.1%} ± {model_data['cv_std']:.1%}")
    print(f"💾 Saved to: calibration/random_forest/{cal_file}")
    print(f"🤖 Model file: calibration/random_forest/{model_file}")
    print("")
    print("🚀 Now run: python run_rf_system.py")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 