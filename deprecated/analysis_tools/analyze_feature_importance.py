#!/usr/bin/env python3
"""
analyze_feature_importance.py

Analyzes and visualizes feature importance from the trained BCI models.
This script:
1. Loads both CSP-only and CSP+BP models
2. Analyzes feature importance using model coefficients
3. Visualizes the results using bar plots
4. Provides statistical analysis of feature contributions
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import logging
from typing import Dict, Tuple, List
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path: str) -> Tuple[object, np.ndarray, np.ndarray]:
    """Load a trained model and its feature data."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Debug: Print model data structure
    logging.info(f"Model data keys: {model_data.keys() if isinstance(model_data, dict) else 'Not a dict'}")
    
    # Extract model and feature data
    model = model_data['classifier']  # Changed from 'model' to 'classifier'
    X = model_data['features']  # Changed from 'X' to 'features'
    y = model_data['labels']    # Changed from 'y' to 'labels'
    
    return model, X, y

def analyze_csp_model(model_path: str) -> Dict:
    """Analyze feature importance for CSP-only model."""
    model, X, y = load_model(model_path)
    
    # Get model coefficients
    coef = model.coef_[0]
    
    # Calculate feature importance scores
    importance = np.abs(coef)
    importance = importance / np.sum(importance)  # Normalize to sum to 1
    
    # Get feature names (CSP components)
    feature_names = [f'CSP{i+1}' for i in range(len(coef))]
    
    return {
        'importance': importance,
        'feature_names': feature_names,
        'coefficients': coef,
        'raw_data': (X, y)
    }

def analyze_csp_bp_model(model_path: str) -> Dict:
    """Analyze feature importance for CSP+BP model."""
    model, X, y = load_model(model_path)
    
    # Get model coefficients
    coef = model.coef_[0]
    
    # Calculate feature importance scores
    importance = np.abs(coef)
    importance = importance / np.sum(importance)  # Normalize to sum to 1
    
    # Get feature names
    n_csp = 4  # Number of CSP components
    feature_names = (
        [f'CSP{i+1}' for i in range(n_csp)] +
        ['Mu Power'] * 8 +  # 8 channels for mu power
        ['Beta Power'] * 8  # 8 channels for beta power
    )
    
    return {
        'importance': importance,
        'feature_names': feature_names,
        'coefficients': coef,
        'raw_data': (X, y)
    }

def plot_feature_importance(importance: np.ndarray, 
                          feature_names: List[str],
                          title: str,
                          save_path: str = None):
    """Plot feature importance as a bar chart."""
    plt.figure(figsize=(12, 6))
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Create bar plot
    bars = plt.barh(range(len(sorted_importance)), sorted_importance)
    
    # Customize plot
    plt.yticks(range(len(sorted_importance)), sorted_names)
    plt.xlabel('Normalized Importance')
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_correlations(X: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Calculate and visualize feature correlations."""
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f',
                square=True)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def main():
    # Model paths
    csp_model_path = 'models/aggregate_csp_model_no_s3.pkl'
    csp_bp_model_path = 'models/aggregate_csp_bp_model_no_s3.pkl'
    
    # Create output directory for plots
    os.makedirs('analysis', exist_ok=True)
    
    # Analyze CSP-only model
    logging.info("Analyzing CSP-only model...")
    csp_results = analyze_csp_model(csp_model_path)
    plot_feature_importance(
        csp_results['importance'],
        csp_results['feature_names'],
        'CSP Component Importance',
        'analysis/csp_importance.png'
    )
    
    # Analyze CSP+BP model
    logging.info("Analyzing CSP+BP model...")
    csp_bp_results = analyze_csp_bp_model(csp_bp_model_path)
    plot_feature_importance(
        csp_bp_results['importance'],
        csp_bp_results['feature_names'],
        'CSP+BP Feature Importance',
        'analysis/csp_bp_importance.png'
    )
    
    # Analyze feature correlations
    logging.info("Analyzing feature correlations...")
    X_csp, _ = csp_results['raw_data']
    X_csp_bp, _ = csp_bp_results['raw_data']
    
    analyze_feature_correlations(X_csp, csp_results['feature_names'])
    analyze_feature_correlations(X_csp_bp, csp_bp_results['feature_names'])
    
    # Print summary statistics
    logging.info("\nFeature Importance Summary:")
    logging.info("\nCSP-only model:")
    for name, imp in zip(csp_results['feature_names'], csp_results['importance']):
        logging.info(f"{name}: {imp:.3f}")
    
    logging.info("\nCSP+BP model:")
    for name, imp in zip(csp_bp_results['feature_names'], csp_bp_results['importance']):
        logging.info(f"{name}: {imp:.3f}")

if __name__ == '__main__':
    main() 