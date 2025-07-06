#!/usr/bin/env python3
"""
evaluate_models.py

Evaluate all available models and generate a performance summary.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path: str) -> Dict:
    """Load a model and return its data."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        return None

def evaluate_model(model_data: Dict) -> Dict:
    """Evaluate a model using cross-validation and in-sample accuracy."""
    try:
        # Extract data
        X = model_data['features']
        y = model_data['labels']
        classifier = model_data['classifier']
        # Cross-validated accuracy
        cv_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
        cv_accuracy = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        # In-sample (training) accuracy
        y_pred = classifier.predict(X)
        train_accuracy = accuracy_score(y, y_pred)
        metrics = {
            'cv_accuracy': cv_accuracy,
            'cv_std': cv_std,
            'train_accuracy': train_accuracy,
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return None

def parse_model_name(filename: str) -> Dict:
    """Parse model name to extract configuration details."""
    name = os.path.basename(filename)
    parts = name.replace('.pkl', '').split('_')
    
    config = {
        'feature_type': 'csp_only' if 'csp_model' in name else 'csp_bp',
        'classifier': parts[-1] if parts[-1] in ['rf', 'svm', 'lda', 'logreg'] else 'lda',
        'exclude_s3': 'no_s3' in name,
        'channels': 'all'
    }
    
    # Extract channel information
    if 'ch' in name:
        ch_idx = name.find('ch')
        ch_end = name.find('_', ch_idx)
        if ch_end == -1:
            ch_end = name.find('.pkl')
        channels = name[ch_idx:ch_end].replace('ch', '').split('_')
        config['channels'] = f"ch{'_'.join(channels)}"
    
    return config

def main():
    # Get all model files
    model_files = glob.glob('models/aggregate_*.pkl')
    if not model_files:
        logging.error("No model files found")
        return
    
    # Evaluate each model
    results = []
    for model_file in model_files:
        logging.info(f"Evaluating {model_file}")
        
        # Load and evaluate model
        model_data = load_model(model_file)
        if model_data is None:
            continue
            
        metrics = evaluate_model(model_data)
        if metrics is None:
            continue
            
        # Get model configuration
        config = parse_model_name(model_file)
        
        # Combine results
        result = {
            'model_file': os.path.basename(model_file),
            **config,
            **metrics
        }
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by cross-validated accuracy
    df = df.sort_values('cv_accuracy', ascending=False)
    
    # Save results
    df.to_csv('model_evaluation_results.csv', index=False)
    
    # Print summary
    print("\nModel Performance Summary:")
    print("=" * 80)
    print(df.to_string())
    
    # Print best model for each category
    print("\nBest Models by Category:")
    print("=" * 80)
    
    # Best overall
    best_overall = df.iloc[0]
    print(f"\nBest Overall Model:")
    print(f"Model: {best_overall['model_file']}")
    print(f"CV Accuracy: {best_overall['cv_accuracy']:.3f} ± {best_overall['cv_std']:.3f}")
    print(f"Train Accuracy: {best_overall['train_accuracy']:.3f}")
    print(f"F1 Score: {best_overall['f1']:.3f}")
    
    # Best by feature type
    for feature_type in df['feature_type'].unique():
        best_feature = df[df['feature_type'] == feature_type].iloc[0]
        print(f"\nBest {feature_type} Model:")
        print(f"Model: {best_feature['model_file']}")
        print(f"CV Accuracy: {best_feature['cv_accuracy']:.3f} ± {best_feature['cv_std']:.3f}")
        print(f"Train Accuracy: {best_feature['train_accuracy']:.3f}")
        print(f"F1 Score: {best_feature['f1']:.3f}")
    
    # Best by classifier
    for classifier in df['classifier'].unique():
        best_clf = df[df['classifier'] == classifier].iloc[0]
        print(f"\nBest {classifier} Model:")
        print(f"Model: {best_clf['model_file']}")
        print(f"CV Accuracy: {best_clf['cv_accuracy']:.3f} ± {best_clf['cv_std']:.3f}")
        print(f"Train Accuracy: {best_clf['train_accuracy']:.3f}")
        print(f"F1 Score: {best_clf['f1']:.3f}")
    
    # Best by channel configuration
    for channels in df['channels'].unique():
        best_ch = df[df['channels'] == channels].iloc[0]
        print(f"\nBest {channels} Model:")
        print(f"Model: {best_ch['model_file']}")
        print(f"CV Accuracy: {best_ch['cv_accuracy']:.3f} ± {best_ch['cv_std']:.3f}")
        print(f"Train Accuracy: {best_ch['train_accuracy']:.3f}")
        print(f"F1 Score: {best_ch['f1']:.3f}")

if __name__ == '__main__':
    main() 