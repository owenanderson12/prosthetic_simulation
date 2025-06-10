#!/usr/bin/env python3
"""
analyze_and_optimize_models.py

Comprehensive analysis and optimization of BCI models:
1. Visualize CSP spatial patterns
2. Optimize feature selection
3. Implement robust validation
4. Analyze band power features
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import logging
from typing import Dict, Tuple, List
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
from dependencies.signal_processor import SignalProcessor
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path: str) -> Dict:
    """Load a trained model and its data."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def load_csp_from_calibration(calib_path: str):
    data = np.load(calib_path, allow_pickle=True)
    csp_filters = data['csp_filters'] if 'csp_filters' in data else None
    csp_patterns = data['csp_patterns'] if 'csp_patterns' in data else None
    return csp_filters, csp_patterns

def visualize_csp_patterns(csp_patterns: np.ndarray, channel_names: List[str], title: str, save_path: str = None):
    """Visualize CSP spatial patterns using topomaps."""
    n_components = csp_patterns.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Create standard montage for 8 channels
    montage = make_standard_montage('standard_1020')
    pos = montage.get_positions()['ch_pos']
    
    for i in range(min(n_components, 4)):  # Show first 4 components
        pattern = csp_patterns[i]
        im, _ = plot_topomap(pattern, pos, axes=axes[i], show=False)
        axes[i].set_title(f'CSP Pattern {i+1}')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def optimize_features(X: np.ndarray, 
                     y: np.ndarray,
                     feature_names: List[str],
                     importance_threshold: float = 0.1) -> Tuple[np.ndarray, List[str]]:
    """Select features based on importance threshold."""
    # Calculate feature importance
    importance = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    importance = importance / np.sum(importance)
    
    # Select features above threshold
    selected_idx = importance > importance_threshold
    selected_features = X[:, selected_idx]
    selected_names = [name for i, name in enumerate(feature_names) if selected_idx[i]]
    
    logging.info(f"Selected {sum(selected_idx)} features out of {len(feature_names)}")
    for name, imp in zip(selected_names, importance[selected_idx]):
        logging.info(f"{name}: {imp:.3f}")
    
    return selected_features, selected_names

def robust_validation(X: np.ndarray,
                     y: np.ndarray,
                     groups: np.ndarray,
                     model,
                     n_splits: int = 5) -> Dict:
    """Perform leave-one-session-out cross-validation."""
    logo = LeaveOneGroupOut()
    scores = []
    conf_matrices = []
    sample_counts = []
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        score = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        
        scores.append(score)
        conf_matrices.append(conf_mat)
        sample_counts.append((len(y_train), len(y_test)))
        logging.info(f"Fold {fold+1}: Accuracy={score:.3f}, Confusion Matrix=\n{conf_mat}\nTrain/Test samples: {len(y_train)}/{len(y_test)}")
    
    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'conf_matrices': conf_matrices,
        'sample_counts': sample_counts
    }

def analyze_band_power(X: np.ndarray,
                      y: np.ndarray,
                      feature_names: List[str]) -> Dict:
    """Analyze band power features."""
    # Separate CSP and band power features
    csp_idx = [i for i, name in enumerate(feature_names) if name.startswith('CSP')]
    mu_idx = [i for i, name in enumerate(feature_names) if name == 'Mu Power']
    beta_idx = [i for i, name in enumerate(feature_names) if name == 'Beta Power']
    
    # Calculate mean power for each class
    left_mask = y == 0
    right_mask = y == 1
    
    results = {
        'mu_power': {
            'left': np.mean(X[left_mask][:, mu_idx], axis=0),
            'right': np.mean(X[right_mask][:, mu_idx], axis=0)
        },
        'beta_power': {
            'left': np.mean(X[left_mask][:, beta_idx], axis=0),
            'right': np.mean(X[right_mask][:, beta_idx], axis=0)
        }
    }
    
    return results

def plot_band_power_analysis(results: Dict, save_path: str = None):
    """Plot band power analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mu power
    channels = range(8)
    ax1.plot(channels, results['mu_power']['left'], 'b-', label='Left')
    ax1.plot(channels, results['mu_power']['right'], 'r-', label='Right')
    ax1.set_title('Mu Band Power')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Power')
    ax1.legend()
    
    # Beta power
    ax2.plot(channels, results['beta_power']['left'], 'b-', label='Left')
    ax2.plot(channels, results['beta_power']['right'], 'r-', label='Right')
    ax2.set_title('Beta Band Power')
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Power')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("Band power plots saved and displayed.")
    print("Summary:")
    print("Mu Power (Left):", results['mu_power']['left'])
    print("Mu Power (Right):", results['mu_power']['right'])
    print("Beta Power (Left):", results['beta_power']['left'])
    print("Beta Power (Right):", results['beta_power']['right'])

def main():
    # Create output directory
    os.makedirs('analysis', exist_ok=True)
    
    # Load models
    csp_model_path = 'models/aggregate_csp_model_no_s3.pkl'
    csp_bp_model_path = 'models/aggregate_csp_bp_model_no_s3.pkl'
    calib_path = 'calibration/calibration_20250601_163037.npz'
    
    csp_data = load_model(csp_model_path)
    csp_bp_data = load_model(csp_bp_model_path)
    
    # Load CSP patterns from calibration file
    csp_filters, csp_patterns = load_csp_from_calibration(calib_path)
    
    # 1. Visualize CSP patterns
    logging.info("Visualizing CSP patterns from calibration file...")
    channel_names = [f'CH{i+1}' for i in range(8)]
    if csp_patterns is not None:
        visualize_csp_patterns(csp_patterns, channel_names, 'CSP Spatial Patterns (from calibration)', 'analysis/csp_patterns.png')
    else:
        logging.warning("CSP patterns not found in calibration file")
    
    # 2. Optimize features
    logging.info("\nOptimizing features...")
    X_csp = csp_data['features']
    y_csp = csp_data['labels']
    feature_names_csp = [f'CSP{i+1}' for i in range(4)]
    
    X_csp_bp = csp_bp_data['features']
    y_csp_bp = csp_bp_data['labels']
    feature_names_csp_bp = (
        [f'CSP{i+1}' for i in range(4)] +
        ['Mu Power'] * 8 +
        ['Beta Power'] * 8
    )
    
    # Optimize CSP features
    selected_csp, selected_names_csp = optimize_features(
        X_csp, y_csp, feature_names_csp, importance_threshold=0.1
    )
    
    # Optimize CSP+BP features
    selected_csp_bp, selected_names_csp_bp = optimize_features(
        X_csp_bp, y_csp_bp, feature_names_csp_bp, importance_threshold=0.1
    )
    
    # 3. Robust validation
    logging.info("\nPerforming robust validation...")
    # Create session groups (assuming sessions are ordered in the data)
    n_samples_per_session = len(y_csp) // 4  # Assuming 4 sessions
    groups = np.repeat(range(4), n_samples_per_session)
    
    csp_validation = robust_validation(
        X_csp, y_csp, groups, csp_data['classifier']
    )
    csp_bp_validation = robust_validation(
        X_csp_bp, y_csp_bp, groups, csp_bp_data['classifier']
    )
    
    logging.info(f"CSP-only model: {csp_validation['mean_score']:.3f} ± {csp_validation['std_score']:.3f}")
    logging.info(f"CSP+BP model: {csp_bp_validation['mean_score']:.3f} ± {csp_bp_validation['std_score']:.3f}")
    
    # 4. Analyze band power
    logging.info("\nAnalyzing band power features...")
    band_power_results = analyze_band_power(X_csp_bp, y_csp_bp, feature_names_csp_bp)
    plot_band_power_analysis(band_power_results, 'analysis/band_power_analysis.png')

if __name__ == '__main__':
    main() 