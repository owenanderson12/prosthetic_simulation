#!/usr/bin/env python3
"""
evaluate_all_models.py

Comprehensive evaluation script for all trained BCI models.
Analyzes models from both models/ and calibration/ directories and provides
detailed performance metrics including accuracy, F1-score, precision, recall,
confusion matrix, and model information.

Outputs:
- Console summary with key metrics
- Detailed JSON report
- CSV file with all metrics
- Confusion matrix plots
- Model comparison visualization

Usage:
    python evaluate_all_models.py [--output-dir OUTPUT_DIR] [--save-plots] [--verbose]
"""

import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import argparse
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Local imports
import config
from dependencies.signal_processor import SignalProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class for BCI models."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {}
        self.model_info = {}
        
        # Initialize signal processor for feature extraction
        self.sp = SignalProcessor(config.__dict__)
        
    def find_all_models(self) -> List[str]:
        """Find all model files in the project."""
        model_files = []
        
        # Search in models directory
        models_dir = Path("models")
        if models_dir.exists():
            model_files.extend(glob.glob(str(models_dir / "*.pkl")))
        
        # Search in calibration directory
        calibration_dir = Path("calibration")
        if calibration_dir.exists():
            model_files.extend(glob.glob(str(calibration_dir / "*.pkl")))
            # Also check random_forest subdirectory
            rf_dir = calibration_dir / "random_forest"
            if rf_dir.exists():
                model_files.extend(glob.glob(str(rf_dir / "*.pkl")))
        
        logger.info(f"Found {len(model_files)} model files")
        return model_files
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a model file and extract information."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract model information
            model_info = {
                'path': model_path,
                'filename': os.path.basename(model_path),
                'size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'load_time': datetime.now().isoformat(),
                'model_type': 'unknown'
            }
            
            # Determine model type from filename
            filename = model_info['filename'].lower()
            if 'rf' in filename or 'random' in filename:
                model_info['model_type'] = 'Random Forest'
            elif 'svm' in filename:
                model_info['model_type'] = 'SVM'
            elif 'lda' in filename:
                model_info['model_type'] = 'LDA'
            elif 'logreg' in filename:
                model_info['model_type'] = 'Logistic Regression'
            
            # Extract features information - handle both standard and robust model formats
            if 'features' in model_data:
                # Standard model format
                model_info['n_features'] = model_data['features'].shape[1]
                model_info['n_samples'] = model_data['features'].shape[0]
                model_info['feature_type'] = 'CSP + Band Power' if model_info['n_features'] > 4 else 'CSP Only'
                model_info['model_format'] = 'standard'
            elif 'train_features' in model_data and 'test_features' in model_data:
                # Robust model format with train/val/test splits
                model_info['n_features'] = model_data['train_features'].shape[1]
                model_info['n_samples'] = (model_data['train_features'].shape[0] + 
                                         model_data['val_features'].shape[0] + 
                                         model_data['test_features'].shape[0])
                model_info['feature_type'] = 'CSP + Band Power' if model_info['n_features'] > 4 else 'CSP Only'
                model_info['model_format'] = 'robust'
                model_info['train_samples'] = model_data['train_features'].shape[0]
                model_info['val_samples'] = model_data['val_features'].shape[0]
                model_info['test_samples'] = model_data['test_features'].shape[0]
            else:
                model_info['model_format'] = 'unknown'
            
            # Extract classifier information
            if 'classifier' in model_data:
                classifier = model_data['classifier']
                if hasattr(classifier, 'named_steps'):
                    clf_name = type(classifier.named_steps['classifier']).__name__
                    model_info['classifier_name'] = clf_name
                    
                    # Extract classifier parameters
                    clf = classifier.named_steps['classifier']
                    if hasattr(clf, 'n_estimators'):
                        model_info['n_estimators'] = clf.n_estimators
                    if hasattr(clf, 'max_depth'):
                        model_info['max_depth'] = clf.max_depth
                    if hasattr(clf, 'C'):
                        model_info['C'] = clf.C
                    if hasattr(clf, 'kernel'):
                        model_info['kernel'] = clf.kernel
            
            return model_data, model_info
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            return None, None
    
    def evaluate_model(self, model_data: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model and return performance metrics."""
        try:
            # Extract features and labels based on model format
            if model_info.get('model_format') == 'robust':
                # For robust models, use test set for evaluation
                X = model_data['test_features']
                y = model_data['test_labels']
                logger.info(f"Evaluating robust model on test set: {X.shape[0]} samples")
            else:
                # Standard model format
                X = model_data['features']
                y = model_data['labels']
            
            classifier = model_data['classifier']
            
            # Basic metrics
            y_pred = classifier.predict(X)
            y_pred_proba = classifier.predict_proba(X)[:, 1] if hasattr(classifier, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='binary'),
                'recall': recall_score(y, y_pred, average='binary'),
                'f1_score': f1_score(y, y_pred, average='binary'),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
            
            # ROC AUC if probabilities available
            if y_pred_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                    fpr, tpr, _ = roc_curve(y, y_pred_proba)
                    metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                except:
                    metrics['roc_auc'] = None
                    metrics['roc_curve'] = None
            
            # Cross-validation score - for robust models, use training data
            try:
                if model_info.get('model_format') == 'robust':
                    # Use training data for cross-validation
                    X_cv = model_data['train_features']
                    y_cv = model_data['train_labels']
                else:
                    X_cv = X
                    y_cv = y
                
                cv_scores = cross_val_score(classifier, X_cv, y_cv, cv=5, scoring='accuracy')
                metrics['cv_accuracy_mean'] = cv_scores.mean()
                metrics['cv_accuracy_std'] = cv_scores.std()
            except:
                metrics['cv_accuracy_mean'] = None
                metrics['cv_accuracy_std'] = None
            
            # Class distribution
            metrics['class_distribution'] = {
                'left': int(np.sum(y == 0)),
                'right': int(np.sum(y == 1)),
                'total': len(y)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate a summary table of all model results."""
        summary_data = []
        
        for model_name, result in self.results.items():
            if result['metrics'] is not None:
                row = {
                    'Model': model_name,
                    'Type': result['info']['model_type'],
                    'Features': result['info'].get('feature_type', 'Unknown'),
                    'Size (MB)': round(result['info']['size_mb'], 2),
                    'Accuracy': round(result['metrics']['accuracy'], 3),
                    'F1-Score': round(result['metrics']['f1_score'], 3),
                    'Precision': round(result['metrics']['precision'], 3),
                    'Recall': round(result['metrics']['recall'], 3),
                    'CV Accuracy': round(result['metrics'].get('cv_accuracy_mean', 0), 3) if result['metrics'].get('cv_accuracy_mean') else 'N/A',
                    'ROC AUC': round(result['metrics'].get('roc_auc', 0), 3) if result['metrics'].get('roc_auc') else 'N/A'
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def plot_confusion_matrices(self, save_plots: bool = True):
        """Plot confusion matrices for all models."""
        n_models = len(self.results)
        if n_models == 0:
            return
        
        # Calculate grid dimensions
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(self.results.items()):
            if result['metrics'] is None:
                continue
                
            cm = np.array(result['metrics']['confusion_matrix'])
            
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'])
            ax.set_title(f'{model_name}\nAccuracy: {result["metrics"]["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrices plot to {self.output_dir / 'confusion_matrices.png'}")
        
        plt.show()
    
    def plot_performance_comparison(self, save_plots: bool = True):
        """Plot performance comparison across all models."""
        if not self.results:
            return
        
        # Prepare data
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = []
            labels = []
            
            for model_name, result in self.results.items():
                if result['metrics'] is not None:
                    values.append(result['metrics'][metric])
                    labels.append(model_name)
            
            if values:
                ax = axes[i]
                bars = ax.bar(range(len(values)), values, alpha=0.7)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance comparison plot to {self.output_dir / 'performance_comparison.png'}")
        
        plt.show()
    
    def save_detailed_report(self):
        """Save detailed evaluation report as JSON."""
        report = {
            'evaluation_time': datetime.now().isoformat(),
            'total_models': len(self.results),
            'models': {}
        }
        
        for model_name, result in self.results.items():
            report['models'][model_name] = {
                'info': result['info'],
                'metrics': result['metrics']
            }
        
        report_path = self.output_dir / 'detailed_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved detailed report to {report_path}")
    
    def save_summary_csv(self):
        """Save summary metrics as CSV."""
        summary_df = self.generate_summary_table()
        csv_path = self.output_dir / 'model_evaluation_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary CSV to {csv_path}")
        return summary_df
    
    def print_summary(self):
        """Print a formatted summary to console."""
        print("\n" + "="*80)
        print("BCI MODEL EVALUATION SUMMARY")
        print("="*80)
        
        summary_df = self.generate_summary_table()
        
        if summary_df.empty:
            print("No models were successfully evaluated.")
            return
        
        # Print summary table
        print("\nModel Performance Summary:")
        print("-" * 80)
        print(summary_df.to_string(index=False))
        
        # Print best performing models
        print("\n" + "="*80)
        print("BEST PERFORMING MODELS")
        print("="*80)
        
        # Best accuracy
        best_acc = summary_df.loc[summary_df['Accuracy'].idxmax()]
        print(f"\n🏆 Best Accuracy: {best_acc['Model']} ({best_acc['Accuracy']:.3f})")
        
        # Best F1-score
        best_f1 = summary_df.loc[summary_df['F1-Score'].idxmax()]
        print(f"🎯 Best F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.3f})")
        
        # Best ROC AUC
        roc_models = summary_df[summary_df['ROC AUC'] != 'N/A']
        if not roc_models.empty:
            best_roc = roc_models.loc[roc_models['ROC AUC'].idxmax()]
            print(f"📈 Best ROC AUC: {best_roc['Model']} ({best_roc['ROC AUC']:.3f})")
        
        # Model statistics
        print(f"\n📊 Model Statistics:")
        print(f"   Total Models Evaluated: {len(summary_df)}")
        print(f"   Average Accuracy: {summary_df['Accuracy'].mean():.3f} ± {summary_df['Accuracy'].std():.3f}")
        print(f"   Average F1-Score: {summary_df['F1-Score'].mean():.3f} ± {summary_df['F1-Score'].std():.3f}")
        
        # Feature analysis
        feature_counts = summary_df['Features'].value_counts()
        print(f"\n🔧 Feature Analysis:")
        for feature_type, count in feature_counts.items():
            print(f"   {feature_type}: {count} models")
        
        print("\n" + "="*80)
    
    def evaluate_all_models(self, save_plots: bool = True, verbose: bool = False):
        """Evaluate all models and generate comprehensive report."""
        logger.info("Starting comprehensive model evaluation...")
        
        # Find all models
        model_files = self.find_all_models()
        
        if not model_files:
            logger.warning("No model files found!")
            return
        
        # Evaluate each model
        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            logger.info(f"Evaluating model: {model_name}")
            
            # Load model
            model_data, model_info = self.load_model(model_path)
            
            if model_data is None:
                logger.warning(f"Failed to load model: {model_path}")
                continue
            
            # Evaluate model
            metrics = self.evaluate_model(model_data, model_info)
            
            # Store results
            self.results[model_name] = {
                'info': model_info,
                'metrics': metrics
            }
            
            if verbose and metrics:
                print(f"\n{model_name}:")
                print(f"  Accuracy: {metrics['accuracy']:.3f}")
                print(f"  F1-Score: {metrics['f1_score']:.3f}")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
        
        # Generate reports and visualizations
        logger.info("Generating reports and visualizations...")
        
        # Print summary
        self.print_summary()
        
        # Save reports
        self.save_detailed_report()
        summary_df = self.save_summary_csv()
        
        # Generate plots
        if save_plots:
            self.plot_confusion_matrices(save_plots=True)
            self.plot_performance_comparison(save_plots=True)
        
        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")
        return summary_df


def main():
    """Main function to run model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate all trained BCI models")
    parser.add_argument('--output-dir', default='evaluation_results',
                      help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--save-plots', action='store_true', default=True,
                      help='Save visualization plots (default: True)')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed information for each model')
    
    args = parser.parse_args()
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    summary_df = evaluator.evaluate_all_models(
        save_plots=args.save_plots,
        verbose=args.verbose
    )
    
    return summary_df


if __name__ == '__main__':
    main() 