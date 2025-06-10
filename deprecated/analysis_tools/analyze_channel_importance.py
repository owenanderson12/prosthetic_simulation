 #!/usr/bin/env python3
"""
analyze_channel_importance.py

Analyze the importance of each EEG channel to classification for both CSP-only and CSP+BP models.
- For CSP: Use the absolute values of the spatial patterns (CSP patterns) to estimate channel contributions.
- For band power: Use the absolute values of the model coefficients for each channel's feature.
- Visualize as bar plots and print ranked lists.
"""
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import logging
from typing import List

# Paths
CSP_MODEL_PATH = 'models/aggregate_csp_model_no_s3.pkl'
CSP_BP_MODEL_PATH = 'models/aggregate_csp_bp_model_no_s3.pkl'
CALIB_PATH = 'calibration/calibration_20250601_163037.npz'
CHANNEL_NAMES = [f'CH{i+1}' for i in range(8)]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path: str):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_csp_patterns(calib_path: str):
    data = np.load(calib_path, allow_pickle=True)
    if 'csp_patterns' in data:
        return data['csp_patterns']
    else:
        logging.warning('No CSP patterns found in calibration file.')
        return None

def analyze_csp_channel_importance(csp_patterns: np.ndarray, n_components: int = 4) -> np.ndarray:
    # Importance: mean absolute value of each channel across the first n_components patterns
    abs_patterns = np.abs(csp_patterns[:n_components, :])
    channel_importance = abs_patterns.mean(axis=0)
    return channel_importance

def analyze_bp_channel_importance(model_data: dict, n_csp: int = 4, n_channels: int = 8) -> np.ndarray:
    # Coefficients: [CSP1, CSP2, CSP3, CSP4, Mu1..8, Beta1..8]
    clf = model_data['classifier']
    if hasattr(clf, 'coef_'):
        coefs = np.abs(clf.coef_[0])
        mu_coefs = coefs[n_csp:n_csp+n_channels]
        beta_coefs = coefs[n_csp+n_channels:n_csp+2*n_channels]
        # Combine mu and beta for each channel (sum or mean)
        channel_importance = mu_coefs + beta_coefs
        return channel_importance
    else:
        logging.warning('Classifier has no coef_ attribute.')
        return np.zeros(n_channels)

def plot_channel_importance(importances: np.ndarray, channel_names: List[str], title: str, save_path: str = None):
    plt.figure(figsize=(10, 5))
    plt.bar(channel_names, importances)
    plt.title(title)
    plt.ylabel('Importance')
    plt.xlabel('Channel')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_ranked_channels(importances: np.ndarray, channel_names: List[str], model_name: str):
    ranked = sorted(zip(channel_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n{model_name} Channel Importance Ranking:")
    for i, (ch, imp) in enumerate(ranked, 1):
        print(f"{i}. {ch}: {imp:.3f}")

def main():
    os.makedirs('analysis', exist_ok=True)
    # CSP-only model
    csp_model = load_model(CSP_MODEL_PATH)
    csp_patterns = load_csp_patterns(CALIB_PATH)
    if csp_patterns is not None:
        csp_importance = analyze_csp_channel_importance(csp_patterns, n_components=4)
        plot_channel_importance(csp_importance, CHANNEL_NAMES, 'CSP Channel Importance', 'analysis/csp_channel_importance.png')
        print_ranked_channels(csp_importance, CHANNEL_NAMES, 'CSP-only Model')
    else:
        print('CSP patterns not found; skipping CSP channel analysis.')
    # CSP+BP model
    csp_bp_model = load_model(CSP_BP_MODEL_PATH)
    bp_importance = analyze_bp_channel_importance(csp_bp_model, n_csp=4, n_channels=8)
    plot_channel_importance(bp_importance, CHANNEL_NAMES, 'Band Power Channel Importance', 'analysis/bp_channel_importance.png')
    print_ranked_channels(bp_importance, CHANNEL_NAMES, 'CSP+BP Model (Band Power)')

if __name__ == '__main__':
    main()
