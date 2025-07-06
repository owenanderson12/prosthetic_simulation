#!/usr/bin/env python3
"""
train_aggregate_models.py

Aggregate calibration and raw session data to train two LDA classifiers
compatible with the current BCI pipeline, and save the resulting models to the `models/`
folder.

1. CSP-only model        → `aggregate_csp_model.pkl`
2. CSP + band-power model → `aggregate_csp_bp_model.pkl`

The script automatically discovers:
- All `calibration_*.npz` files in the calibration directory
- All raw session data in data/raw/session_*
- Optionally excludes session 3 which may contain artifacts

Usage:
    python train_aggregate_models.py [--exclude-session-3] [--calibration-only] [--raw-only]
"""
import glob
import os
import logging
import numpy as np
import argparse
import pickle
from typing import List, Dict, Tuple, Optional, Any

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add XGBoost import
try:
    from xgboost import XGBClassifier
    _has_xgboost = True
except ImportError:
    _has_xgboost = False
    XGBClassifier = None

# Local imports – adjust paths for scripts subdirectory
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import scripts.config.config as config
from dependencies.signal_processor import SignalProcessor
from dependencies.file_handler import FileHandler

# -----------------------------------------------------------------------------
# Configuration - adjust paths for scripts subdirectory
# -----------------------------------------------------------------------------
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
    parser = argparse.ArgumentParser(description="Train aggregate BCI models from calibration and processed data.")
    parser.add_argument('--exclude-session-3', action='store_true',
                      help='Exclude session 3 data which may contain artifacts')
    parser.add_argument('--calibration-only', action='store_true',
                      help='Only use calibration data, ignore processed sessions')
    parser.add_argument('--processed-only', action='store_true',
                      help='Only use processed session data, ignore calibration files')
    parser.add_argument(
        '--classifier-type', 
        type=str, 
        default='lda',
        choices=['lda', 'svm', 'logreg', 'rf', 'xgb'],
        help='The type of classifier to train: lda, svm, logreg, or rf.'
    )
    parser.add_argument(
        '--channels',
        type=int,
        nargs='+',
        default=list(range(1, 9)),
        help='A list of channel numbers (1-indexed) to use for training. Default is all 8 channels.'
    )
    return parser.parse_args()


def reshape_trial(trial: np.ndarray, channel_indices: List[int]) -> np.ndarray:
    """Reshape trial data to consistent 2D format (samples × channels)."""
    if trial.ndim == 3:  # (trials, samples, channels)
        # Reshape to 2D by concatenating trials
        return trial.reshape(-1, trial.shape[-1])[:, channel_indices]
    elif trial.ndim == 2:  # (samples, channels)
        return trial[:, channel_indices]
    else:
        raise ValueError(f"Unexpected trial shape: {trial.shape}")


def load_processed_session_data(exclude_session_3: bool = False, channel_indices: List[int] = None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Load data from processed session files."""
    baseline_list: List[np.ndarray] = []
    left_trials: List[np.ndarray] = []
    right_trials: List[np.ndarray] = []
    
    # Get all processed session files
    session_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, 'session_*_processed.npz')))
    if not session_files:
        logging.warning("No processed session files found in %s", PROCESSED_DATA_DIR)
        return np.array([]), [], []
        
    for session_file in session_files:
        # Skip session 3 if requested
        if exclude_session_3 and 'session_3' in session_file:
            logging.info("Skipping session 3 as requested")
            continue
            
        logging.info("Loading processed data from %s", session_file)
        
        try:
            data = np.load(session_file, allow_pickle=True)
            
            # Extract data
            if 'baseline_data' in data and data['baseline_data'].size > 0:
                baseline_data = data['baseline_data']
                if isinstance(baseline_data, np.ndarray):
                    baseline_list.append(reshape_trial(baseline_data, channel_indices))
            
            # Handle left trials
            if 'left_data' in data:
                left_data = data['left_data']
                if left_data.dtype == object:
                    # Convert each trial to numpy array
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
            
            # Handle right trials
            if 'right_data' in data:
                right_data = data['right_data']
                if right_data.dtype == object:
                    # Convert each trial to numpy array
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
            logging.error("Error loading %s: %s", session_file, str(e))
            continue
    
    baseline_concat = np.vstack(baseline_list) if baseline_list else np.array([])
    
    # Log trial information
    logging.info(f"Loaded {len(left_trials)} left trials and {len(right_trials)} right trials")
    if left_trials:
        logging.info(f"Left trial shape example: {left_trials[0].shape}")
    if right_trials:
        logging.info(f"Right trial shape example: {right_trials[0].shape}")
    
    return baseline_concat, left_trials, right_trials


def load_all_calibration_files(channel_indices: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Load and combine all calibration datasets found in CALIB_DIR."""
    baseline_list: List[np.ndarray] = []
    left_trials: List[np.ndarray] = []
    right_trials: List[np.ndarray] = []

    file_paths = sorted(glob.glob(os.path.join(CALIB_DIR, 'calibration_*.npz')))
    if not file_paths:
        logging.warning("No calibration files found in %s", CALIB_DIR)
        return np.array([]), [], []

    logging.info("Found %d calibration files", len(file_paths))

    for fp in file_paths:
        data = np.load(fp, allow_pickle=True)
        # Baseline
        if 'baseline_data' in data and data['baseline_data'].size > 0:
            baseline_data = data['baseline_data']
            if isinstance(baseline_data, np.ndarray):
                baseline_list.append(reshape_trial(baseline_data, channel_indices))
        
        # Handle left trials
        if 'left_data' in data:
            left_data = data['left_data']
            if left_data.dtype == object:
                # Convert each trial to numpy array
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
        
        # Handle right trials
        if 'right_data' in data:
            right_data = data['right_data']
            if right_data.dtype == object:
                # Convert each trial to numpy array
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

    baseline_concat = np.vstack(baseline_list) if baseline_list else np.array([])
    
    # Log trial information
    logging.info(f"Loaded {len(left_trials)} left trials and {len(right_trials)} right trials")
    if left_trials:
        logging.info(f"Left trial shape example: {left_trials[0].shape}")
    if right_trials:
        logging.info(f"Right trial shape example: {right_trials[0].shape}")
    
    return baseline_concat, left_trials, right_trials


def extract_features(sp: SignalProcessor, trials: List[np.ndarray]) -> tuple:
    """Slide windows through trials and extract CSP + band-power features."""
    csp_only_feats: List[np.ndarray] = []
    combined_feats: List[np.ndarray] = []

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
                # Skip windows without CSP features (should be rare after training)
                continue
            csp_only_feats.append(csp_vec)
            mu = feats.get('erd_mu', [])
            beta = feats.get('erd_beta', [])
            if len(mu) > 0 and len(beta) > 0:
                combined_vec = np.hstack([csp_vec, mu, beta])
                combined_feats.append(combined_vec)
            else:
                # If no band power features, just use CSP features
                combined_feats.append(csp_vec)
                
    return np.array(csp_only_feats), np.array(combined_feats)


def get_classifier(classifier_type: str) -> Pipeline:
    """Returns a scikit-learn pipeline with a scaler and the specified classifier."""
    if classifier_type == 'lda':
        # LDA with shrinkage is robust to high-dimensional data
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif classifier_type == 'svm':
        # SVC with RBF kernel, probability estimates for soft decisions
        clf = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    elif classifier_type == 'logreg':
        # Logistic Regression, robust solver for smaller datasets
        clf = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
    elif classifier_type == 'rf':
        # Random Forest, good for non-linear interactions
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    elif classifier_type == 'xgb':
        if not _has_xgboost:
            raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'.")
        # You can tune these parameters as needed
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])


def train_and_save(features_left, features_right, model_path, classifier_type: str = 'lda', sp: SignalProcessor = None):
    X = np.vstack([np.array(features_left), np.array(features_right)])
    y = np.hstack([
        np.zeros(len(features_left)),  # 0 = left
        np.ones(len(features_right))   # 1 = right
    ])

    logging.info(f"[DEBUG] Training model: {model_path}")
    logging.info(f"[DEBUG] Classifier type: {classifier_type}")
    logging.info(f"[DEBUG] Feature shape: {X.shape}")
    if X.shape[0] > 0 and X.shape[1] > 0:
        logging.info(f"[DEBUG] First feature vector: {X[0, :5]} ... (showing first 5 values)")
    else:
        logging.warning("[DEBUG] Feature matrix is empty!")

    if X.shape[0] < 10:
        logging.error("Not enough samples to train a model. Skipping %s.", model_path)
        return 0.0

    pipeline = get_classifier(classifier_type)
    logging.info(f"[DEBUG] Pipeline classifier class: {type(pipeline.named_steps['classifier'])}")

    # Cross-validation
    try:
        # Use 5 folds, but only if there are enough samples in each class for the split
        n_splits = 5
        min_samples_per_class = np.min(np.bincount(y.astype(int)))
        if min_samples_per_class < n_splits:
            n_splits = min_samples_per_class
        if n_splits < 2:
            logging.warning("Not enough samples for reliable cross-validation. Skipping.")
            acc = 0.0
        else:
            acc = np.mean(cross_val_score(pipeline, X, y, cv=n_splits, scoring='accuracy'))
    except Exception as e:
        logging.error(f"Cross-validation failed: {e}")
        acc = 0.0

    # Train on all data
    pipeline.fit(X, y)

    # Save model, features, and labels in one file
    model_data = {
        'classifier': pipeline,
        'features': X,
        'labels': y,
        'classes': [0., 1.],
        'class_map': {0: 'left', 1: 'right'},
        'threshold': config.CLASSIFIER_THRESHOLD,
        'adaptive_threshold': None  # This was part of the old structure
    }
    # Add CSP filters if available
    if sp is not None and sp.csp_filters is not None:
        model_data['csp_filters'] = sp.csp_filters
        model_data['csp_patterns'] = sp.csp_patterns
        model_data['csp_mean'] = sp.csp_mean
        model_data['csp_std'] = sp.csp_std
        logging.info("Including CSP filters in saved model")

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    logging.info("Saved model %s (CV accuracy %.3f)", model_path, acc)

    # Debug: Load the model back and print the classifier type
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        clf_type = type(loaded_model['classifier'].named_steps['classifier'])
        logging.info(f"[DEBUG] Loaded classifier type from saved model: {clf_type}")
    except Exception as e:
        logging.error(f"[DEBUG] Could not load and check saved model: {e}")

    return acc


def combine_datasets(calib_data: Tuple, raw_data: Tuple) -> Tuple:
    """Combine calibration and raw session data."""
    calib_baseline, calib_left, calib_right = calib_data
    raw_baseline, raw_left, raw_right = raw_data
    
    # Combine baselines
    if calib_baseline.size > 0 and raw_baseline.size > 0:
        baseline = np.vstack([calib_baseline, raw_baseline])
    else:
        baseline = calib_baseline if calib_baseline.size > 0 else raw_baseline
        
    # Combine trials
    left_trials = calib_left + raw_left
    right_trials = calib_right + raw_right
    
    return baseline, left_trials, right_trials


def augment_trials_with_time_shifts(
    trials: List[np.ndarray],
    shifts_s: List[float],
    sample_rate: int
) -> List[np.ndarray]:
    """Augment trials by creating time-shifted versions."""
    augmented_trials = []
    logging.info(f"Original number of trials to augment: {len(trials)}")
    
    for trial in trials:
        # Add original trial
        augmented_trials.append(trial)
        
        trial_len_samples = trial.shape[0]
        
        for shift in shifts_s:
            shift_samples = int(shift * sample_rate)
            
            if shift_samples == 0:
                continue

            # Create shifted trial
            new_trial = None
            if shift_samples > 0:
                # Positive shift: crop from the start
                if trial_len_samples > shift_samples:
                    new_trial = trial[shift_samples:]
            else:  # shift_samples < 0
                # Negative shift: crop from the end
                if trial_len_samples > abs(shift_samples):
                    new_trial = trial[:shift_samples]
            
            if new_trial is not None and new_trial.shape[0] > 0:
                augmented_trials.append(new_trial)
                    
    logging.info(f"Number of trials after augmentation: {len(augmented_trials)}")
    return augmented_trials


def main():
    args = parse_args()
    
    # Convert 1-indexed channel numbers to 0-indexed indices
    channel_indices = [c - 1 for c in args.channels]

    # Load data based on arguments
    calib_data = (np.array([]), [], [])
    processed_data = (np.array([]), [], [])
    
    if not args.processed_only:
        calib_data = load_all_calibration_files(channel_indices)
        logging.info("Loaded %d left and %d right trials from calibration", 
                    len(calib_data[1]), len(calib_data[2]))
    
    if not args.calibration_only:
        processed_data = load_processed_session_data(args.exclude_session_3, channel_indices)
        logging.info("Loaded %d left and %d right trials from processed sessions%s", 
                    len(processed_data[1]), len(processed_data[2]),
                    " (excluding session 3)" if args.exclude_session_3 else "")
    
    # Combine datasets
    baseline_data, left_trials, right_trials = combine_datasets(calib_data, processed_data)
    
    if not left_trials or not right_trials:
        logging.error("No trials found in the specified data sources. Exiting.")
        return

    # Data Augmentation
    augmentation_shifts_s = [-0.4, -0.2, 0.2, 0.4]  # seconds
    logging.info(f"Augmenting data with time shifts: {augmentation_shifts_s}s")
    
    left_trials = augment_trials_with_time_shifts(left_trials, augmentation_shifts_s, SAMPLE_RATE)
    right_trials = augment_trials_with_time_shifts(right_trials, augmentation_shifts_s, SAMPLE_RATE)

    # Initialize signal processor and set baseline
    sp = SignalProcessor(config.__dict__)
    if baseline_data.size > 0:
        sp.update_baseline(baseline_data)
        logging.info("Baseline updated with %d samples", baseline_data.shape[0])

    # Train global CSP filters
    logging.info("Training global CSP filters ...")
    if not sp.train_csp(left_trials, right_trials):
        logging.error("CSP training failed. Exiting.")
        return

    # Extract features
    logging.info("Extracting features from trials ...")
    left_csp, left_comb = extract_features(sp, left_trials)
    right_csp, right_comb = extract_features(sp, right_trials)

    if len(left_csp) == 0 or len(right_csp) == 0:
        logging.error("No CSP features extracted. Exiting.")
        return

    # Add source info to model names
    name_suffix = ""
    if args.exclude_session_3:
        name_suffix += "_no_s3"
    if args.calibration_only:
        name_suffix += "_calib_only"
    if args.processed_only:
        name_suffix += "_processed_only"
    
    # Add channel info to model name
    if len(args.channels) < 8:
        name_suffix += f"_ch{'_'.join(map(str, args.channels))}"

    # Add classifier type to model name
    name_suffix += f"_{args.classifier_type}"

    # Train CSP-only model
    model_name = f'aggregate_csp_model{name_suffix}.pkl'
    csp_acc = train_and_save(left_csp, right_csp, os.path.join(MODEL_DIR, model_name), args.classifier_type, sp)

    # Train CSP + band-power model
    bp_acc = 0.0
    if len(left_comb) > 0 and left_comb.shape[1] > left_csp.shape[1]:
        model_name = f'aggregate_csp_bp_model{name_suffix}.pkl'
        bp_acc = train_and_save(left_comb, right_comb, os.path.join(MODEL_DIR, model_name), args.classifier_type, sp)
    else:
        logging.warning("Combined feature length equals CSP length; skipping CSP+BP model training.")

    # Write results to a temp file for experiment runner
    with open("temp_results.txt", "w") as f:
        f.write(f"{csp_acc},{bp_acc}")

    logging.info("Model training complete.")


if __name__ == '__main__':
    main() 