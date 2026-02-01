#!/usr/bin/env python3
"""
train_robust_rf.py

Specialized training script for robust Random Forest models optimized for live BCI systems.
This script tests multiple RF configurations to find the optimal balance between robustness
and performance for real-time prosthetic control.

Key features:
- Multiple RF robustness levels (conservative, balanced, performance)
- Corrected methodology with proper train/test splits (no CSP data leakage)
- Stability analysis across CV folds
- Live system deployment considerations
- Feature importance analysis for interpretability

Usage:
    python train_robust_rf.py [--robustness-level LEVEL] [--analyze-stability]
"""

import os
import glob
import logging
import numpy as np
import argparse
import pickle
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Local imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import config
from dependencies.signal_processor import SignalProcessor
from dependencies.file_handler import FileHandler

# Configuration
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
    parser = argparse.ArgumentParser(description="Train robust Random Forest for BCI live system")
    parser.add_argument(
        '--robustness-level',
        type=str,
        default='balanced',
        choices=['conservative', 'balanced', 'performance'],
        help='RF robustness level: conservative (live system), balanced, performance (default: balanced)'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--analyze-stability',
        action='store_true',
        help='Perform detailed stability analysis across CV folds'
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
        help='Exclude session 3 data (recommended if contains artifacts)'
    )
    return parser.parse_args()


def get_rf_configuration(robustness_level: str) -> Dict[str, Any]:
    """Get Random Forest configuration based on robustness level."""
    configurations = {
        'conservative': {
            'n_estimators': 30,        # Faster inference for real-time
            'max_depth': 4,            # Prevent overfitting
            'min_samples_split': 8,    # Conservative splitting
            'min_samples_leaf': 4,     # Large leaf constraint
            'max_features': 'sqrt',    # Feature subsampling
            'bootstrap': True,         # Bootstrap sampling
            'oob_score': True,         # Out-of-bag error estimation
            'n_jobs': -1,             # Parallel processing
            'random_state': 42,
            'description': 'Most robust for live deployment, faster inference'
        },
        'balanced': {
            'n_estimators': 50,        # Good balance
            'max_depth': 6,            # Moderate depth
            'min_samples_split': 5,    # Balanced splitting
            'min_samples_leaf': 2,     # Standard leaf constraint
            'max_features': 'sqrt',    # Feature subsampling
            'bootstrap': True,         # Bootstrap sampling
            'oob_score': True,         # Out-of-bag error estimation
            'n_jobs': -1,             # Parallel processing
            'random_state': 42,
            'description': 'Balanced robustness and performance'
        },
        'performance': {
            'n_estimators': 100,       # More trees for performance
            'max_depth': 8,            # Deeper trees
            'min_samples_split': 2,    # Aggressive splitting
            'min_samples_leaf': 1,     # Minimal leaf constraint
            'max_features': 'sqrt',    # Feature subsampling
            'bootstrap': True,         # Bootstrap sampling
            'oob_score': True,         # Out-of-bag error estimation
            'n_jobs': -1,             # Parallel processing
            'random_state': 42,
            'description': 'Maximum performance, may overfit'
        }
    }
    return configurations[robustness_level]


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


def stability_analysis(model, X_train, y_train, cv_folds: int = 5) -> Dict[str, Any]:
    """Perform detailed stability analysis of RF model."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_feature_importances = []
    oob_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Clone and fit model
        fold_model = model['classifier']
        fold_model.fit(X_fold_train, y_fold_train)

        # Evaluate
        val_pred = fold_model.predict(X_fold_val)
        fold_acc = accuracy_score(y_fold_val, val_pred)
        fold_accuracies.append(fold_acc)

        # Feature importance (from RF within pipeline)
        if hasattr(fold_model, 'feature_importances_'):
            fold_feature_importances.append(fold_model.feature_importances_)
        elif hasattr(fold_model.named_steps['classifier'], 'feature_importances_'):
            fold_feature_importances.append(fold_model.named_steps['classifier'].feature_importances_)

        # OOB score
        rf_clf = fold_model.named_steps['classifier'] if hasattr(fold_model, 'named_steps') else fold_model
        if hasattr(rf_clf, 'oob_score_'):
            oob_scores.append(rf_clf.oob_score_)

    # Calculate stability metrics
    accuracy_stability = {
        'mean': np.mean(fold_accuracies),
        'std': np.std(fold_accuracies),
        'min': np.min(fold_accuracies),
        'max': np.max(fold_accuracies),
        'range': np.max(fold_accuracies) - np.min(fold_accuracies),
        'cv': np.std(fold_accuracies) / np.mean(fold_accuracies)  # Coefficient of variation
    }

    feature_importance_stability = {}
    if fold_feature_importances:
        feature_importances = np.array(fold_feature_importances)
        feature_importance_stability = {
            'mean_importance': np.mean(feature_importances, axis=0),
            'std_importance': np.std(feature_importances, axis=0),
            'stability_score': np.mean(np.std(feature_importances, axis=0))  # Lower = more stable
        }

    oob_stability = {}
    if oob_scores:
        oob_stability = {
            'mean': np.mean(oob_scores),
            'std': np.std(oob_scores),
            'min': np.min(oob_scores),
            'max': np.max(oob_scores)
        }

    return {
        'accuracy_stability': accuracy_stability,
        'feature_importance_stability': feature_importance_stability,
        'oob_stability': oob_stability,
        'fold_accuracies': fold_accuracies
    }


def evaluate_model(model, X_test, y_test, model_name: str) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary')
    }

    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
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
    print(f"Actual  Left Right")
    print(f"Left    {cm[0,0]:4d} {cm[0,1]:5d}")
    print(f"Right   {cm[1,0]:4d} {cm[1,1]:5d}")

    return metrics


def main():
    args = parse_args()

    # Convert 1-indexed channel numbers to 0-indexed indices
    channel_indices = [c - 1 for c in args.channels]

    # Get RF configuration
    rf_config = get_rf_configuration(args.robustness_level)

    print(f"\n{'='*60}")
    print("ROBUST RANDOM FOREST TRAINING FOR BCI LIVE SYSTEM")
    print(f"{'='*60}")
    print(f"Robustness Level: {args.robustness_level.upper()}")
    print(f"Description: {rf_config['description']}")
    print(f"RF Configuration:")
    for key, value in rf_config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    print(f"Test split: {args.test_split:.1%}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Channels: {args.channels}")
    print(f"Exclude session 3: {args.exclude_session_3}")
    print(f"Stability analysis: {args.analyze_stability}")

    # Load all data
    logging.info("Loading all available data...")
    baseline_data, left_trials, right_trials = load_all_data(channel_indices, args.exclude_session_3)

    if not left_trials or not right_trials:
        logging.error("No trials found. Exiting.")
        return

    # Initialize signal processor
    sp = SignalProcessor(config.__dict__)
    if baseline_data.size > 0:
        sp.update_baseline(baseline_data)
        logging.info(f"Baseline updated with {baseline_data.shape[0]} samples")

    # **CORRECTED METHODOLOGY**: Split trials BEFORE training CSP
    logging.info("Splitting trials into train/test to prevent data leakage...")
    train_left, test_left = train_test_split(
        left_trials, test_size=args.test_split, random_state=42
    )
    train_right, test_right = train_test_split(
        right_trials, test_size=args.test_split, random_state=42
    )

    logging.info(f"Training trials: {len(train_left)} left, {len(train_right)} right")
    logging.info(f"Test trials: {len(test_left)} left, {len(test_right)} right")

    # Train CSP filters ONLY on training data
    logging.info("Training CSP filters on training data only...")
    if not sp.train_csp(train_left, train_right):
        logging.error("CSP training failed. Exiting.")
        return

    # Extract features using fitted CSP
    logging.info("Extracting features using fitted CSP...")
    train_left_features = extract_features(sp, train_left)
    train_right_features = extract_features(sp, train_right)
    test_left_features = extract_features(sp, test_left)
    test_right_features = extract_features(sp, test_right)

    if not train_left_features or not train_right_features:
        logging.error("No training features extracted. Exiting.")
        return

    if not test_left_features or not test_right_features:
        logging.error("No test features extracted. Exiting.")
        return

    # Prepare training and test data
    X_train_temp = np.vstack([np.array(train_left_features), np.array(train_right_features)])
    y_train_temp = np.hstack([np.zeros(len(train_left_features)), np.ones(len(train_right_features))])

    X_test = np.vstack([np.array(test_left_features), np.array(test_right_features)])
    y_test = np.hstack([np.zeros(len(test_left_features)), np.ones(len(test_right_features))])

    logging.info(f"Training samples: {len(X_train_temp)}")
    logging.info(f"Test samples: {len(X_test)}")
    logging.info(f"Feature dimensions: {X_train_temp.shape[1]}")
    logging.info(f"Training class distribution: {np.bincount(y_train_temp.astype(int))}")
    logging.info(f"Test class distribution: {np.bincount(y_test.astype(int))}")

    # Split training data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, test_size=args.test_split/(1-args.test_split),
        stratify=y_train_temp, random_state=42
    )

    logging.info(f"Final train set: {len(X_train)} samples")
    logging.info(f"Final validation set: {len(X_val)} samples")
    logging.info(f"Final test set: {len(X_test)} samples")

    # Create RF pipeline
    rf_params = {k: v for k, v in rf_config.items() if k != 'description'}
    clf = RandomForestClassifier(**rf_params)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

    # Cross-validation on training data
    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=args.cv_folds, scoring='accuracy')

    print(f"\nCross-Validation Results (Training Data):")
    print(f"  Mean CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  CV Scores: {cv_scores}")

    # Train final model on training data
    logging.info("Training final Random Forest model...")
    pipeline.fit(X_train, y_train)

    # Get OOB score if available
    rf_classifier = pipeline.named_steps['classifier']
    if hasattr(rf_classifier, 'oob_score_'):
        print(f"  Out-of-Bag Score: {rf_classifier.oob_score_:.3f}")

    # Feature importance analysis
    feature_importances = rf_classifier.feature_importances_
    print(f"\nTop 5 Most Important Features:")
    top_features = np.argsort(feature_importances)[::-1][:5]
    for i, feat_idx in enumerate(top_features):
        print(f"  {i+1}. Feature {feat_idx}: {feature_importances[feat_idx]:.3f}")

    # Stability analysis
    stability_results = None
    if args.analyze_stability:
        print(f"\n{'='*60}")
        print("STABILITY ANALYSIS")
        print(f"{'='*60}")
        logging.info("Performing stability analysis...")
        stability_results = stability_analysis(pipeline, X_train, y_train, args.cv_folds)

        acc_stability = stability_results['accuracy_stability']
        print(f"Accuracy Stability:")
        print(f"  Mean: {acc_stability['mean']:.3f}")
        print(f"  Std: {acc_stability['std']:.3f}")
        print(f"  Range: {acc_stability['range']:.3f}")
        print(f"  CV: {acc_stability['cv']:.3f}")

        if stability_results['oob_stability']:
            oob_stability = stability_results['oob_stability']
            print(f"OOB Stability:")
            print(f"  Mean: {oob_stability['mean']:.3f}")
            print(f"  Std: {oob_stability['std']:.3f}")

        if stability_results['feature_importance_stability']:
            feat_stability = stability_results['feature_importance_stability']
            print(f"Feature Importance Stability Score: {feat_stability['stability_score']:.3f}")

    # Evaluate on validation set
    val_metrics = evaluate_model(pipeline, X_val, y_val, f"Robust RF ({args.robustness_level.title()}) - Validation")

    # Evaluate on test set (final uncontaminated evaluation)
    test_metrics = evaluate_model(pipeline, X_test, y_test, f"Robust RF ({args.robustness_level.title()}) - Test")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"robust_rf_{args.robustness_level}_{timestamp}.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)

    model_data = {
        'classifier': pipeline,
        'rf_config': rf_config,
        'robustness_level': args.robustness_level,
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
        'feature_importances': feature_importances,
        'training_info': {
            'robustness_level': args.robustness_level,
            'test_split': args.test_split,
            'cv_folds': args.cv_folds,
            'channels': args.channels,
            'exclude_session_3': args.exclude_session_3,
            'timestamp': timestamp,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'oob_score': getattr(rf_classifier, 'oob_score_', None),
            'stability_results': stability_results
        }
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    # Live system deployment notes
    print(f"\n{'='*60}")
    print("LIVE SYSTEM DEPLOYMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Model saved: {model_path}")
    print(f"Test Accuracy (uncontaminated): {test_metrics['accuracy']:.3f}")
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    if hasattr(rf_classifier, 'oob_score_'):
        print(f"OOB Score: {rf_classifier.oob_score_:.3f}")

    print(f"\nLive System Recommendations:")
    if args.robustness_level == 'conservative':
        print("✅ This model is optimized for live deployment")
        print("  - Fast inference (~30 trees)")
        print("  - Conservative parameters prevent overfitting")
        print("  - Suitable for real-time BCI control")
    elif args.robustness_level == 'balanced':
        print("⚖️  Balanced model - good for most applications")
        print("  - Good performance/robustness tradeoff")
        print("  - Moderate inference time")
    else:
        print("⚠️  Performance model - may be less robust")
        print("  - Higher risk of overfitting")
        print("  - Slower inference time")

    # Performance recommendations
    if test_metrics['accuracy'] > 0.70:
        print("✅ Good test accuracy for BCI deployment")
    elif test_metrics['accuracy'] > 0.60:
        print("⚖️  Acceptable test accuracy - monitor in live system")
    else:
        print("⚠️  Low test accuracy - consider more data or feature engineering")

    if stability_results and stability_results['accuracy_stability']['cv'] < 0.1:
        print("✅ Model shows good stability across CV folds")
    elif stability_results and stability_results['accuracy_stability']['cv'] < 0.2:
        print("⚖️  Model stability is acceptable")
    else:
        print("⚠️  Model shows high variance - may not be reliable")

    return test_metrics


if __name__ == '__main__':
    main()