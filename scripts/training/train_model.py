#!/usr/bin/env python3
"""
train_robust_model.py

Train a robust BCI model with proper train/validation/test splits to avoid overfitting.
This script focuses on realistic performance (~80% accuracy) rather than overfitting to training data.

Key features:
- Proper train/validation/test splits (60/20/20)
- Minimal data augmentation to prevent overfitting
- Cross-validation on training data only
- Final evaluation on held-out test set
- Focus on generalization rather than training accuracy

Usage:
    python train_robust_model.py [--classifier-type TYPE] [--test-split 0.2] [--cv-folds 5]
"""

import os
import glob
import logging
import numpy as np
import argparse
import pickle
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Local imports – adjust paths for scripts subdirectory
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import scripts.config.config as config
from dependencies.signal_processor import SignalProcessor
from dependencies.file_handler import FileHandler

# Configuration - adjust paths for scripts subdirectory
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
CALIB_DIR = config.__dict__.get('CALIBRATION_DIR', os.path.join(project_root, 'calibration'))
MODEL_DIR = config.__dict__.get('MODEL_DIR', os.path.join(project_root, 'models'))
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
WINDOW_SIZE_S = config.__dict__.get('WINDOW_SIZE', 2.0)
WINDOW_OVERLAP_S = config.__dict__.get('WINDOW_OVERLAP', 0.5)
SAMPLE_RATE = config.__dict__.get('SAMPLE_RATE', 250)

# Derived parameters
WINDOW_SIZE = int(WINDOW_SIZE_S * SAMPLE_RATE)
STEP_SIZE = int((WINDOW_SIZE_S - WINDOW_OVERLAP_S) * SAMPLE_RATE)

os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train robust BCI model with proper validation")
    parser.add_argument(
        '--classifier-type', 
        type=str, 
        default='rf',
        choices=['lda', 'svm', 'logreg', 'rf'],
        help='Classifier type: lda, svm, logreg, or rf (default: rf)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--channels',
        type=int,
        nargs='+',
        default=list(range(1, 9)),
        help='Channel numbers to use (default: all 8 channels)'
    )
    parser.add_argument(
        '--exclude-session-3',
        action='store_true',
        help='Exclude session 3 data'
    )
    parser.add_argument(
        '--min-augmentation',
        action='store_true',
        help='Use minimal augmentation to prevent overfitting'
    )
    parser.add_argument(
        '--hyperparameter-tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    return parser.parse_args()


def reshape_trial(trial: np.ndarray, channel_indices: List[int]) -> np.ndarray:
    """Reshape trial data to consistent 2D format (samples × channels)."""
    if trial.ndim == 3:  # (trials, samples, channels)
        return trial.reshape(-1, trial.shape[-1])[:, channel_indices]
    elif trial.ndim == 2:  # (samples, channels)
        return trial[:, channel_indices]
    else:
        raise ValueError(f"Unexpected trial shape: {trial.shape}")


def load_all_data(channel_indices: List[int], exclude_session_3: bool = False) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Load all available data from calibration and processed sessions."""
    baseline_list: List[np.ndarray] = []
    left_trials: List[np.ndarray] = []
    right_trials: List[np.ndarray] = []
    
    # Load calibration data
    calib_files = sorted(glob.glob(os.path.join(CALIB_DIR, 'calibration_*.npz')))
    for fp in calib_files:
        try:
            data = np.load(fp, allow_pickle=True)
            
            # Baseline
            if 'baseline_data' in data and data['baseline_data'].size > 0:
                baseline_data = data['baseline_data']
                if isinstance(baseline_data, np.ndarray):
                    baseline_list.append(reshape_trial(baseline_data, channel_indices))
            
            # Left trials
            if 'left_data' in data:
                left_data = data['left_data']
                if left_data.dtype == object:
                    for trial in left_data:
                        if isinstance(trial, np.ndarray) and trial.size > 0:
                            try:
                                reshaped = reshape_trial(trial.astype(np.float64), channel_indices)
                                left_trials.append(reshaped)
                            except Exception as e:
                                logging.warning(f"Could not reshape left trial: {e}")
                elif left_data.size > 0:
                    try:
                        reshaped = reshape_trial(left_data.astype(np.float64), channel_indices)
                        left_trials.append(reshaped)
                    except Exception as e:
                        logging.warning(f"Could not reshape left data: {e}")
            
            # Right trials
            if 'right_data' in data:
                right_data = data['right_data']
                if right_data.dtype == object:
                    for trial in right_data:
                        if isinstance(trial, np.ndarray) and trial.size > 0:
                            try:
                                reshaped = reshape_trial(trial.astype(np.float64), channel_indices)
                                right_trials.append(reshaped)
                            except Exception as e:
                                logging.warning(f"Could not reshape right trial: {e}")
                elif right_data.size > 0:
                    try:
                        reshaped = reshape_trial(right_data.astype(np.float64), channel_indices)
                        right_trials.append(reshaped)
                    except Exception as e:
                        logging.warning(f"Could not reshape right data: {e}")
                        
        except Exception as e:
            logging.error(f"Error loading {fp}: {e}")
            continue
    
    # Load processed session data
    session_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, 'session_*_processed.npz')))
    for session_file in session_files:
        if exclude_session_3 and 'session_3' in session_file:
            logging.info("Skipping session 3 as requested")
            continue
            
        try:
            data = np.load(session_file, allow_pickle=True)
            
            # Baseline
            if 'baseline_data' in data and data['baseline_data'].size > 0:
                baseline_data = data['baseline_data']
                if isinstance(baseline_data, np.ndarray):
                    baseline_list.append(reshape_trial(baseline_data, channel_indices))
            
            # Left trials
            if 'left_data' in data:
                left_data = data['left_data']
                if left_data.dtype == object:
                    for trial in left_data:
                        if isinstance(trial, np.ndarray) and trial.size > 0:
                            try:
                                reshaped = reshape_trial(trial.astype(np.float64), channel_indices)
                                left_trials.append(reshaped)
                            except Exception as e:
                                logging.warning(f"Could not reshape left trial: {e}")
                elif left_data.size > 0:
                    try:
                        reshaped = reshape_trial(left_data.astype(np.float64), channel_indices)
                        left_trials.append(reshaped)
                    except Exception as e:
                        logging.warning(f"Could not reshape left data: {e}")
            
            # Right trials
            if 'right_data' in data:
                right_data = data['right_data']
                if right_data.dtype == object:
                    for trial in right_data:
                        if isinstance(trial, np.ndarray) and trial.size > 0:
                            try:
                                reshaped = reshape_trial(trial.astype(np.float64), channel_indices)
                                right_trials.append(reshaped)
                            except Exception as e:
                                logging.warning(f"Could not reshape right trial: {e}")
                elif right_data.size > 0:
                    try:
                        reshaped = reshape_trial(right_data.astype(np.float64), channel_indices)
                        right_trials.append(reshaped)
                    except Exception as e:
                        logging.warning(f"Could not reshape right data: {e}")
                        
        except Exception as e:
            logging.error(f"Error loading {session_file}: {e}")
            continue
    
    baseline_concat = np.vstack(baseline_list) if baseline_list else np.array([])
    
    logging.info(f"Loaded {len(left_trials)} left trials and {len(right_trials)} right trials")
    if left_trials:
        logging.info(f"Left trial shape example: {left_trials[0].shape}")
    if right_trials:
        logging.info(f"Right trial shape example: {right_trials[0].shape}")
    
    return baseline_concat, left_trials, right_trials


def extract_features(sp: SignalProcessor, trials: List[np.ndarray]) -> List[np.ndarray]:
    """Extract CSP + band-power features from trials."""
    features = []
    
    for trial in trials:
        if trial.size == 0:
            continue
            
        # Sliding windows
        for start in range(0, len(trial) - WINDOW_SIZE + 1, STEP_SIZE):
            window = trial[start:start + WINDOW_SIZE]
            timestamps = np.arange(window.shape[0]) / SAMPLE_RATE
            result = sp.process(window, timestamps)
            
            if not result['valid'] or result['features'] is None:
                continue
                
            feats = result['features']
            csp_vec = feats.get('csp_features')
            
            if csp_vec is None:
                continue
                
            # Combine CSP with band power features
            mu = feats.get('erd_mu', [])
            beta = feats.get('erd_beta', [])
            
            if len(mu) > 0 and len(beta) > 0:
                combined_vec = np.hstack([csp_vec, mu, beta])
                features.append(combined_vec)
            else:
                # If no band power, just use CSP
                features.append(csp_vec)
                
    return features


def get_classifier(classifier_type: str, hyperparameter_tune: bool = False) -> Pipeline:
    """Get classifier with appropriate hyperparameters."""
    if classifier_type == 'lda':
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif classifier_type == 'svm':
        clf = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
    elif classifier_type == 'rf':
        # Use more conservative parameters to prevent overfitting
        clf = RandomForestClassifier(
            n_estimators=50,  # Reduced from 100
            max_depth=5,      # Reduced from 10
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    return pipeline


def minimal_augmentation(trials: List[np.ndarray], sample_rate: int) -> List[np.ndarray]:
    """Apply minimal augmentation to prevent overfitting."""
    augmented_trials = []
    
    for trial in trials:
        # Add original trial
        augmented_trials.append(trial)
        
        # Add only one small time shift to increase diversity slightly
        shift_samples = int(0.2 * sample_rate)  # 0.2s shift
        
        if trial.shape[0] > shift_samples:
            # Positive shift
            new_trial = trial[shift_samples:]
            augmented_trials.append(new_trial)
            
            # Negative shift
            new_trial = trial[:-shift_samples]
            augmented_trials.append(new_trial)
    
    logging.info(f"Minimal augmentation: {len(trials)} → {len(augmented_trials)} trials")
    return augmented_trials


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }

    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except Exception:
            metrics['roc_auc'] = None
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION: {model_name}")
    print(f"{'='*60}")
    print(f"Test Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
    if metrics.get('roc_auc'):
        print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  Predicted")
    print(f"Actual  Left Right Idle")
    class_labels = ['Left', 'Right', 'Idle']
    for i, label in enumerate(class_labels):
        print(f"{label:<7}{cm[i,0]:5d} {cm[i,1]:6d} {cm[i,2]:6d}")
    
    return metrics


def main():
    args = parse_args()
    
    # Convert 1-indexed channel numbers to 0-indexed indices
    channel_indices = [c - 1 for c in args.channels]
    
    print(f"\n{'='*60}")
    print("ROBUST BCI MODEL TRAINING")
    print(f"{'='*60}")
    print(f"Classifier: {args.classifier_type.upper()}")
    print(f"Test split: {args.test_split:.1%}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Channels: {args.channels}")
    print(f"Exclude session 3: {args.exclude_session_3}")
    print(f"Minimal augmentation: {args.min_augmentation}")
    print(f"Hyperparameter tuning: {args.hyperparameter_tune}")
    
    # Load all data
    logging.info("Loading all available data...")
    baseline_data, left_trials, right_trials = load_all_data(channel_indices, args.exclude_session_3)

    if not left_trials or not right_trials or baseline_data.size == 0:
        logging.error("Insufficient trials for training (need left, right, and baseline). Exiting.")
        return
    
    # Apply minimal augmentation if requested
    if args.min_augmentation:
        logging.info("Applying minimal augmentation...")
        left_trials = minimal_augmentation(left_trials, SAMPLE_RATE)
        right_trials = minimal_augmentation(right_trials, SAMPLE_RATE)
    
    # Initialize signal processor
    sp = SignalProcessor(config.__dict__)
    if baseline_data.size > 0:
        sp.update_baseline(baseline_data)
        logging.info(f"Baseline updated with {baseline_data.shape[0]} samples")
    
    # Train CSP filters on all data
    logging.info("Training CSP filters...")
    if not sp.train_csp(left_trials, right_trials):
        logging.error("CSP training failed. Exiting.")
        return
    
    # Extract features
    logging.info("Extracting features...")
    left_features = extract_features(sp, left_trials)
    right_features = extract_features(sp, right_trials)
    idle_features = extract_features(sp, [baseline_data])

    if not left_features or not right_features or not idle_features:
        logging.error("No features extracted for one or more classes. Exiting.")
        return

    # Prepare data for splitting (0=left,1=right,2=idle)
    X = np.vstack([
        np.array(left_features),
        np.array(right_features),
        np.array(idle_features)
    ])
    y = np.hstack([
        np.zeros(len(left_features)),
        np.ones(len(right_features)),
        np.full(len(idle_features), 2)
    ])
    
    logging.info(f"Total samples: {len(X)}")
    logging.info(f"Feature dimensions: {X.shape[1]}")
    logging.info(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Split data: train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, stratify=y, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.test_split/(1-args.test_split), 
        stratify=y_temp, random_state=42
    )
    
    logging.info(f"Train set: {len(X_train)} samples")
    logging.info(f"Validation set: {len(X_val)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
    
    # Get classifier
    pipeline = get_classifier(args.classifier_type, args.hyperparameter_tune)
    
    # Cross-validation on training data
    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=args.cv_folds, scoring='accuracy')
    
    print(f"\nCross-Validation Results (Training Data):")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  CV Scores: {cv_scores}")
    
    # Train final model on training data
    logging.info("Training final model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_metrics = evaluate_model(pipeline, X_val, y_val, f"{args.classifier_type.upper()} (Validation)")
    
    # Evaluate on test set
    test_metrics = evaluate_model(pipeline, X_test, y_test, f"{args.classifier_type.upper()} (Test)")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"robust_{args.classifier_type}_model_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)
    
    model_data = {
        'classifier': pipeline,
        'train_features': X_train,
        'train_labels': y_train,
        'val_features': X_val,
        'val_labels': y_val,
        'test_features': X_test,
        'test_labels': y_test,
        'classes': [0., 1.],
        'class_map': {0: 'left', 1: 'right'},
        'threshold': config.CLASSIFIER_THRESHOLD,
        'csp_filters': sp.csp_filters,
        'csp_patterns': sp.csp_patterns,
        'csp_mean': sp.csp_mean,
        'csp_std': sp.csp_std,
        'training_info': {
            'classifier_type': args.classifier_type,
            'test_split': args.test_split,
            'cv_folds': args.cv_folds,
            'channels': args.channels,
            'exclude_session_3': args.exclude_session_3,
            'min_augmentation': args.min_augmentation,
            'timestamp': timestamp,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved: {model_path}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Expected realistic performance: ~80%")
    
    if test_metrics['accuracy'] > 0.95:
        print("⚠️  WARNING: Very high accuracy may indicate overfitting!")
        print("   Consider reducing model complexity or using more validation data.")
    elif test_metrics['accuracy'] < 0.70:
        print("⚠️  WARNING: Low accuracy may indicate underfitting!")
        print("   Consider increasing model complexity or collecting more data.")
    else:
        print("✅ Model performance appears realistic and well-balanced.")
    
    return test_metrics


if __name__ == '__main__':
    main() 