# BCI System Scripts Guide

This guide provides comprehensive documentation for all scripts in the BCI-Controlled Prosthetic System, organized by functionality and purpose.

## Overview

The BCI system includes a complete suite of scripts for training, evaluation, analysis, preprocessing, and testing. All scripts are organized into logical categories within the `scripts/` directory for easy navigation and maintenance.

## Script Categories

### Training Scripts (`scripts/training/`)
- **Purpose**: Model training with various algorithms and validation strategies
- **Output**: Trained models, validation results, performance metrics
- **Usage**: Development, research, model optimization

### Evaluation Scripts (`scripts/evaluation/`)
- **Purpose**: Model performance assessment and comparison
- **Output**: Evaluation reports, performance metrics, visualizations
- **Usage**: Model selection, performance analysis, research validation

### Analysis Scripts (`scripts/analysis/`)
- **Purpose**: Comprehensive EEG analysis and visualization
- **Output**: Research-grade plots, statistical analysis, publication-ready figures
- **Usage**: Research, data exploration, method validation

### Preprocessing Scripts (`scripts/preprocessing/`)
- **Purpose**: Raw EEG data processing and preparation
- **Output**: Cleaned, processed data ready for analysis/training
- **Usage**: Data pipeline, quality control, format standardization

### Testing Scripts (`scripts/testing/`)
- **Purpose**: System validation and integration testing
- **Output**: Test results, connectivity status, validation reports
- **Usage**: System validation, debugging, integration testing

---

## Training Scripts

### 1. `train_aggregate_models.py`
**Comprehensive multi-algorithm training with data aggregation**

#### Purpose
Trains multiple machine learning algorithms on aggregated data from multiple EEG sessions, providing comprehensive model comparison and selection.

#### Key Features
- **Multiple Algorithms**: LDA, SVM, Random Forest, XGBoost support
- **Data Aggregation**: Combines multiple recording sessions
- **Cross-Validation**: Statistical validation with k-fold CV
- **Feature Engineering**: CSP + band power feature extraction
- **Performance Comparison**: Side-by-side algorithm comparison
- **Model Persistence**: Complete model saving with metadata

#### Usage
```bash
# Basic usage - train all algorithms
python scripts/training/train_aggregate_models.py

# Specify classifier type
python scripts/training/train_aggregate_models.py --classifier-type rf
python scripts/training/train_aggregate_models.py --classifier-type svm
python scripts/training/train_aggregate_models.py --classifier-type lda
python scripts/training/train_aggregate_models.py --classifier-type xgb

# Channel selection
python scripts/training/train_aggregate_models.py --channels 2 3 6 7
python scripts/training/train_aggregate_models.py --channels 3 6  # Minimal set

# Data filtering
python scripts/training/train_aggregate_models.py --exclude-session-3
python scripts/training/train_aggregate_models.py --exclude-session-3 --classifier-type rf

# Enable verbose output
python scripts/training/train_aggregate_models.py --verbose
```

#### Arguments
- `--classifier-type`: Algorithm to use (lda, svm, rf, xgb)
- `--channels`: EEG channels to include (space-separated)
- `--exclude-session-3`: Exclude session 3 data (quality issues)
- `--verbose`: Enable detailed output logging

#### Output
- **Models**: Saved to `models/aggregate_*_model.pkl`
- **Results**: Performance metrics and validation results
- **Plots**: CSP patterns, confusion matrices (if applicable)
- **Logs**: Training progress and performance metrics

#### Example Output
```
Training Random Forest model...
Data shape: (1200, 12) - 12 features, 1200 samples
Cross-validation scores: [0.78, 0.82, 0.79, 0.83, 0.80]
Mean CV accuracy: 0.804 ± 0.019
Test accuracy: 0.815
Model saved: models/aggregate_rf_model.pkl
```

### 2. `train_model.py`
**Robust training with proper validation splits and hyperparameter tuning**

#### Purpose
Implements robust model training with proper train/validation/test splits, hyperparameter optimization, and comprehensive validation to prevent overfitting.

#### Key Features
- **Proper Data Splits**: 60/20/20 train/validation/test splits
- **Hyperparameter Tuning**: Grid search and random search support
- **Overfitting Prevention**: Validation-based early stopping
- **Statistical Validation**: Comprehensive performance analysis
- **Model Interpretability**: Feature importance analysis
- **Production Ready**: Models ready for deployment

#### Usage
```bash
# Basic training
python scripts/training/train_model.py

# Specify algorithm
python scripts/training/train_model.py --classifier-type rf
python scripts/training/train_model.py --classifier-type svm

# Enable hyperparameter tuning
python scripts/training/train_model.py --hyperparameter-tune
python scripts/training/train_model.py --classifier-type rf --hyperparameter-tune

# Custom validation
python scripts/training/train_model.py --cv-folds 10
python scripts/training/train_model.py --test-size 0.3

# Verbose output
python scripts/training/train_model.py --verbose
```

#### Arguments
- `--classifier-type`: ML algorithm (lda, svm, rf, xgb)
- `--hyperparameter-tune`: Enable hyperparameter optimization
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--test-size`: Test set proportion (default: 0.2)
- `--verbose`: Detailed output and progress logging

#### Output
- **Models**: Optimized models saved to `models/`
- **Performance**: Detailed validation results
- **Plots**: Learning curves, feature importance
- **Reports**: Statistical analysis and model comparison

#### Example Output
```
Training with proper validation splits...
Train set: 720 samples, Validation set: 240 samples, Test set: 240 samples
Hyperparameter tuning enabled...
Best parameters: {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5}
Validation accuracy: 0.825
Test accuracy: 0.808
Model saved: models/robust_rf_model.pkl
```

### 3. `train_rf_model.py`
**Specialized Random Forest training optimized for BCI applications**

#### Purpose
Focuses specifically on Random Forest optimization for motor imagery classification, with BCI-specific tuning and calibration format compatibility.

#### Key Features
- **RF-Specific Optimization**: Tailored hyperparameters for BCI data
- **Calibration Compatibility**: Works with existing calibration files
- **Feature Importance**: Detailed analysis of feature contributions
- **Robust Training**: Handles class imbalance and small datasets
- **Production Deployment**: Ready for real-time BCI operation

#### Usage
```bash
# Basic RF training
python scripts/training/train_rf_model.py

# Use specific calibration file
python scripts/training/train_rf_model.py --calibration-file calibration/best_model.npz

# Custom RF parameters
python scripts/training/train_rf_model.py --n-estimators 200 --max-depth 15

# Enable feature analysis
python scripts/training/train_rf_model.py --analyze-features
```

#### Arguments
- `--calibration-file`: Specific calibration file to use
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum tree depth (default: 10)
- `--analyze-features`: Enable feature importance analysis

#### Output
- **Models**: RF models saved to `models/` and `calibration/`
- **Analysis**: Feature importance plots and analysis
- **Performance**: Validation results and accuracy metrics
- **Calibration**: Compatible with existing calibration system

---

## Evaluation Scripts

### 1. `evaluate_all_models.py`
**Comprehensive model evaluation and comparison framework**

#### Purpose
Provides comprehensive evaluation of all trained models with detailed performance metrics, statistical analysis, and visualization.

#### Key Features
- **Model Discovery**: Automatically finds all trained models
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Statistical Analysis**: Confidence intervals, significance testing
- **Visualization**: Confusion matrices, ROC curves, performance plots
- **Comparison**: Side-by-side model comparison
- **Reporting**: JSON, CSV, and visual outputs

#### Usage
```bash
# Basic evaluation
python scripts/evaluation/evaluate_all_models.py

# Save plots and detailed analysis
python scripts/evaluation/evaluate_all_models.py --save-plots

# Verbose output with detailed metrics
python scripts/evaluation/evaluate_all_models.py --verbose

# Custom output directory
python scripts/evaluation/evaluate_all_models.py --output-dir custom_results/

# Include cross-validation
python scripts/evaluation/evaluate_all_models.py --cross-validate
```

#### Arguments
- `--save-plots`: Generate and save visualization plots
- `--verbose`: Detailed output and progress information
- `--output-dir`: Custom output directory for results
- `--cross-validate`: Include cross-validation analysis

#### Output
- **Reports**: `evaluation_results/detailed_evaluation_report.json`
- **Summary**: `evaluation_results/model_evaluation_summary.csv`
- **Plots**: Confusion matrices, ROC curves, performance comparisons
- **Statistics**: Confidence intervals, significance tests

#### Example Output
```
Evaluating 8 models found in models/ directory...

Model Performance Summary:
==========================
aggregate_rf_model.pkl:           Accuracy: 0.847 ± 0.023
robust_rf_model_20250621.pkl:     Accuracy: 0.832 ± 0.019
aggregate_csp_model_rf.pkl:       Accuracy: 0.825 ± 0.021
...

Best Model: aggregate_rf_model.pkl (Accuracy: 0.847)
Results saved to: evaluation_results/
```

### 2. `evaluate_models.py`
**Quick model performance assessment and ranking**

#### Purpose
Provides rapid model evaluation and ranking for quick model selection and performance comparison.

#### Key Features
- **Quick Assessment**: Fast performance evaluation
- **Model Ranking**: Automatic ranking by performance
- **CSV Export**: Results exported to CSV for analysis
- **Console Summary**: Immediate performance feedback
- **Lightweight**: Minimal computational overhead

#### Usage
```bash
# Quick evaluation
python scripts/evaluation/evaluate_models.py

# Custom model directory
python scripts/evaluation/evaluate_models.py --model-dir custom_models/

# Specify test data
python scripts/evaluation/evaluate_models.py --test-data data/processed/test_set.npz
```

#### Arguments
- `--model-dir`: Directory containing models to evaluate
- `--test-data`: Specific test dataset to use

#### Output
- **CSV**: `model_evaluation_results.csv`
- **Console**: Performance summary and rankings
- **Quick Results**: Immediate feedback on model performance

#### Example Output
```
Model Evaluation Results:
========================
1. aggregate_rf_model.pkl          - Accuracy: 0.847
2. robust_rf_model_20250621.pkl    - Accuracy: 0.832
3. aggregate_csp_model_rf.pkl      - Accuracy: 0.825
...

Results saved to: model_evaluation_results.csv
```

---

## Analysis Scripts

### 1. `analyze_mi_eeg.py`
**Comprehensive motor imagery EEG analysis suite**

#### Purpose
Provides comprehensive offline analysis of motor imagery EEG data with ERD/ERS visualization, statistical analysis, and research-grade plotting.

#### Key Features
- **ERD/ERS Analysis**: Event-related desynchronization/synchronization
- **Time-Frequency Analysis**: Spectrograms and wavelet analysis
- **Statistical Analysis**: Significance testing and confidence intervals
- **Visualization**: Publication-ready plots and figures
- **Band Power Analysis**: Detailed frequency band analysis
- **Spatial Analysis**: Channel-wise analysis and topoplots

#### Usage
```bash
# Basic analysis
python scripts/analysis/analyze_mi_eeg.py

# Specify data session
python scripts/analysis/analyze_mi_eeg.py --session 1
python scripts/analysis/analyze_mi_eeg.py --session 2

# Custom analysis parameters
python scripts/analysis/analyze_mi_eeg.py --freq-range 8 30
python scripts/analysis/analyze_mi_eeg.py --baseline-duration 2.0

# Generate all plots
python scripts/analysis/analyze_mi_eeg.py --generate-all-plots

# Export results
python scripts/analysis/analyze_mi_eeg.py --export-results
```

#### Arguments
- `--session`: Specific session to analyze (1, 2, 4)
- `--freq-range`: Frequency range for analysis (default: 8-30 Hz)
- `--baseline-duration`: Baseline period duration (default: 2.0s)
- `--generate-all-plots`: Generate all visualization plots
- `--export-results`: Export analysis results to files

#### Output
- **Plots**: ERD/ERS maps, spectrograms, time-course plots
- **Statistics**: Statistical significance maps, confidence intervals
- **Results**: Analysis results saved to `analysis_results/`
- **Figures**: Publication-ready figures for research

#### Generated Plots
- **ERD/ERS Time-Frequency Maps**: Showing frequency-specific activity changes
- **Channel-wise ERD/ERS**: Individual channel analysis
- **Topographic Maps**: Spatial distribution of activity
- **Band Power Boxplots**: Statistical comparison of frequency bands
- **Spectrograms**: Time-frequency representations

### Analysis Modules (`modules/`)

The analysis modules provide specialized functions for EEG analysis:

#### Signal Processing Modules
- `bandpass_filter.py`: Advanced filtering with quality control
- `extract_epochs.py`: Robust epoch extraction with validation
- `calculate_bandpower.py`: Frequency band power calculation
- `apply_grand_average_reference.py`: Reference electrode methods

#### Visualization Modules
- `plot_erd_time_course.py`: ERD/ERS time-course visualization
- `plot_wavelet_spectrogram.py`: Wavelet-based spectrograms
- `plot_bandpower_boxplots.py`: Statistical band power plots
- `plot_average_band_powers.py`: Average band power visualization

#### Example Module Usage
```python
# Using individual modules
from modules.plot_erd_time_course import plot_erd_time_course
from modules.calculate_bandpower import calculate_bandpower

# Generate ERD time-course plot
plot_erd_time_course(
    data=eeg_data,
    channels=['C3', 'C4'],
    save_path='analysis_results/erd_timecourse.png'
)

# Calculate band powers
mu_power = calculate_bandpower(eeg_data, fs=250, band=(8, 13))
beta_power = calculate_bandpower(eeg_data, fs=250, band=(13, 30))
```

---

## Preprocessing Scripts

### 1. `preprocess_raw_data.py`
**Complete EEG preprocessing pipeline**

#### Purpose
Processes raw EEG data from CSV files into analysis-ready format with quality control, artifact rejection, and standardization.

#### Key Features
- **Format Conversion**: CSV to NPZ format conversion
- **Quality Control**: Automatic artifact detection and rejection
- **Standardization**: Consistent data format across sessions
- **Validation**: Data integrity checking and validation
- **Batch Processing**: Process multiple sessions efficiently
- **Metadata Preservation**: Maintain session information

#### Usage
```bash
# Process all sessions
python scripts/preprocessing/preprocess_raw_data.py

# Process specific session
python scripts/preprocessing/preprocess_raw_data.py --session 1
python scripts/preprocessing/preprocess_raw_data.py --session 2

# Custom processing parameters
python scripts/preprocessing/preprocess_raw_data.py --filter-range 1 45
python scripts/preprocessing/preprocess_raw_data.py --artifact-threshold 100

# Verbose output
python scripts/preprocessing/preprocess_raw_data.py --verbose
```

#### Arguments
- `--session`: Specific session to process (1, 2, 4)
- `--filter-range`: Frequency filter range (default: 1-45 Hz)
- `--artifact-threshold`: Artifact rejection threshold (μV)
- `--verbose`: Detailed processing information

#### Output
- **Processed Data**: Saved to `data/processed/session_X_processed.npz`
- **Quality Reports**: Data quality assessment reports
- **Logs**: Processing logs and validation results
- **Metadata**: Session information and processing parameters

#### Processing Steps
1. **Data Loading**: Load raw CSV files with validation
2. **Quality Assessment**: Initial data quality evaluation
3. **Filtering**: Bandpass filtering (1-45 Hz) and notch filtering
4. **Artifact Rejection**: Automatic artifact detection and removal
5. **Standardization**: Format standardization and validation
6. **Export**: Save processed data with metadata

#### Example Output
```
Processing Session 1...
Loaded 3 CSV files (total: 45,000 samples)
Applied bandpass filter (1-45 Hz)
Rejected 127 artifacts (2.3% of data)
Exported to: data/processed/session_1_processed.npz
Quality score: 8.7/10
```

---

## Testing Scripts

### 1. `test_lsl_stream.py`
**LSL connectivity and stream validation**

#### Purpose
Tests Lab Streaming Layer (LSL) connectivity, validates stream properties, and ensures proper communication for real-time BCI operation.

#### Key Features
- **Stream Discovery**: Find and list available LSL streams
- **Connection Testing**: Validate stream connectivity
- **Data Validation**: Check stream data format and quality
- **Performance Testing**: Assess stream reliability and latency
- **Debug Information**: Detailed diagnostic information

#### Usage
```bash
# Basic LSL testing
python scripts/testing/test_lsl_stream.py

# Test specific stream
python scripts/testing/test_lsl_stream.py --stream-name "OpenBCI_EEG"

# Extended testing with performance metrics
python scripts/testing/test_lsl_stream.py --extended-test

# Continuous monitoring
python scripts/testing/test_lsl_stream.py --monitor --duration 60
```

#### Arguments
- `--stream-name`: Specific stream to test
- `--extended-test`: Comprehensive testing with performance metrics
- `--monitor`: Continuous monitoring mode
- `--duration`: Test duration in seconds

#### Output
- **Stream List**: Available LSL streams
- **Connection Status**: Stream connectivity validation
- **Performance Metrics**: Latency, reliability, data quality
- **Debug Information**: Detailed diagnostic output

#### Example Output
```
LSL Stream Testing Results:
===========================
Found 2 LSL streams:
1. OpenBCI_EEG (8 channels, 250.0 Hz)
2. ProstheticControl (4 channels, 60.0 Hz)

Testing OpenBCI_EEG stream...
✓ Connection successful
✓ Data format valid (8 channels, float32)
✓ Sample rate: 250.0 Hz
✓ Latency: 15.3 ms ± 2.1 ms
✓ Data quality: Good (0.2% packet loss)

Stream ready for BCI operation.
```

### 2. `test_unity_commands.py`
**Unity integration testing and validation**

#### Purpose
Tests Unity integration by sending test commands via LSL and validating the complete BCI-to-Unity communication pipeline.

#### Key Features
- **Command Testing**: Send test commands to Unity
- **Integration Validation**: Verify complete BCI pipeline
- **Performance Testing**: Assess command latency and reliability
- **Debug Mode**: Detailed command logging and analysis
- **Automated Testing**: Scripted test sequences

#### Usage
```bash
# Basic Unity testing
python scripts/testing/test_unity_commands.py

# Send specific test commands
python scripts/testing/test_unity_commands.py --command left
python scripts/testing/test_unity_commands.py --command right
python scripts/testing/test_unity_commands.py --command idle

# Automated test sequence
python scripts/testing/test_unity_commands.py --automated-test

# Performance testing
python scripts/testing/test_unity_commands.py --performance-test
```

#### Arguments
- `--command`: Specific command to test (left, right, idle)
- `--automated-test`: Run automated test sequence
- `--performance-test`: Comprehensive performance testing
- `--debug`: Enable debug output

#### Output
- **Command Status**: Command transmission validation
- **Unity Response**: Unity integration verification
- **Performance Metrics**: Command latency and reliability
- **Integration Report**: Complete pipeline validation

#### Example Output
```
Unity Integration Testing:
==========================
✓ LSL stream created: ProstheticControl
✓ Unity connection established
✓ Test command sent: LEFT (confidence: 0.85)
✓ Unity response: Hand animation triggered
✓ Command latency: 23.7 ms
✓ Integration successful

All tests passed. Unity integration ready.
```

---

## Usage Workflows

### Complete Development Workflow

```bash
# 1. Data Preprocessing
python scripts/preprocessing/preprocess_raw_data.py

# 2. Model Training
python scripts/training/train_aggregate_models.py --classifier-type rf

# 3. Model Evaluation
python scripts/evaluation/evaluate_all_models.py --save-plots

# 4. System Testing
python scripts/testing/test_lsl_stream.py
python scripts/testing/test_unity_commands.py

# 5. Research Analysis
python scripts/analysis/analyze_mi_eeg.py --generate-all-plots
```

### Quick Model Development

```bash
# Fast model training and evaluation
python scripts/training/train_model.py --classifier-type rf --hyperparameter-tune
python scripts/evaluation/evaluate_models.py
```

### Research Analysis Workflow

```bash
# Comprehensive analysis for research
python scripts/preprocessing/preprocess_raw_data.py --session 1 --verbose
python scripts/analysis/analyze_mi_eeg.py --session 1 --generate-all-plots
python scripts/training/train_aggregate_models.py --verbose
python scripts/evaluation/evaluate_all_models.py --save-plots --verbose
```

### Production Deployment

```bash
# Test system components
python scripts/testing/test_lsl_stream.py --extended-test
python scripts/testing/test_unity_commands.py --performance-test

# Deploy best model
python scripts/evaluation/evaluate_all_models.py
# Use results to select best model for main.py
```

---

## Script Dependencies

### Common Dependencies
- **NumPy**: Numerical computing and array operations
- **SciPy**: Scientific computing and signal processing
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Matplotlib**: Plotting and visualization
- **Pandas**: Data manipulation and analysis
- **PyLSL**: Lab Streaming Layer communication

### Optional Dependencies
- **XGBoost**: Gradient boosting classifier
- **Seaborn**: Statistical visualization
- **Joblib**: Parallel computing and model persistence
- **Plotly**: Interactive visualizations

### Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install xgboost seaborn plotly
```

---

## Error Handling and Debugging

### Common Issues and Solutions

#### Import Errors
```bash
# Fix path issues
export PYTHONPATH="${PYTHONPATH}:/path/to/prosthetic"

# Or use absolute imports
python -c "import sys; sys.path.insert(0, '/path/to/prosthetic'); exec(open('scripts/training/train_model.py').read())"
```

#### Data Issues
```bash
# Check data availability
ls data/processed/
ls calibration/

# Verify data format
python -c "import numpy as np; data = np.load('data/processed/session_1_processed.npz'); print(data.files)"
```

#### Model Issues
```bash
# Check model files
ls models/

# Validate model
python -c "import joblib; model = joblib.load('models/best_model.pkl'); print(type(model))"
```

### Debug Mode
Most scripts support verbose output for debugging:
```bash
# Enable verbose output
python scripts/training/train_model.py --verbose
python scripts/evaluation/evaluate_all_models.py --verbose
python scripts/preprocessing/preprocess_raw_data.py --verbose
```

---

## Performance Optimization

### Computational Efficiency
- **Parallel Processing**: Many scripts use multiprocessing for speed
- **Memory Management**: Efficient memory usage for large datasets
- **Caching**: Intermediate results cached when possible
- **Optimization**: NumPy vectorization and efficient algorithms

### Resource Requirements
- **RAM**: 4-8GB recommended for training scripts
- **CPU**: Multi-core processors benefit parallel processing
- **Storage**: ~1GB for models and results
- **GPU**: Not required but can accelerate some operations

### Performance Tips
```bash
# Use parallel processing
export OMP_NUM_THREADS=4

# Optimize memory usage
python scripts/training/train_model.py --batch-size 32

# Cache intermediate results
python scripts/preprocessing/preprocess_raw_data.py --cache-results
```

---

## Contributing to Scripts

### Adding New Scripts

1. **Choose Category**: Place in appropriate `scripts/` subdirectory
2. **Follow Conventions**: Use consistent naming and structure
3. **Add Documentation**: Include docstrings and comments
4. **Test Thoroughly**: Ensure robust error handling
5. **Update Guide**: Add documentation to this guide

### Script Template
```python
#!/usr/bin/env python3
"""
Script Name: descriptive_name.py
Purpose: Brief description of script purpose
Author: Your Name
Date: Creation date
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from dependencies import module_name
from modules import analysis_module

def main():
    """Main function with argument parsing and execution"""
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--arg1', type=str, help='Argument description')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Script logic here
    
if __name__ == '__main__':
    main()
```

This comprehensive guide provides complete documentation for all scripts in the BCI system, enabling efficient development, research, and deployment workflows. 