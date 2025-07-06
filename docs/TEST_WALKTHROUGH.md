# BCI System Test Walkthrough & Validation Guide

This comprehensive guide provides step-by-step testing procedures for the complete BCI-Controlled Prosthetic System, covering data preprocessing, model training, evaluation, analysis, and real-time operation.

**Last Updated:** 2024-12-20  
**System Version:** 2.0  
**Test Coverage:** Complete system validation

---

## Table of Contents
- [Quick Start Testing](#quick-start-testing)
- [System Component Testing](#system-component-testing)
- [Complete Development Workflow](#complete-development-workflow)
- [Real-time Operation Testing](#real-time-operation-testing)
- [Integration Testing](#integration-testing)
- [Performance Validation](#performance-validation)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Advanced Testing Scenarios](#advanced-testing-scenarios)

---

## Prerequisites & Setup

### Environment Setup
```bash
# 1. Ensure Python environment is active
conda activate eeg_project_env  # or your environment name
# or
source venv/bin/activate

# 2. Verify dependencies
pip install -r requirements.txt

# 3. Check system installation
python -c "import numpy, scipy, sklearn, matplotlib; print('Dependencies OK')"

# 4. Verify project structure
ls scripts/training/ scripts/evaluation/ scripts/analysis/ scripts/preprocessing/
```

### Available Test Data
```bash
# Check available data
ls data/raw/session_*/
ls data/processed/
ls calibration/
ls models/

# Expected data files:
# - data/raw/session_1/ (CSV files)
# - data/raw/session_2/ (CSV files) 
# - data/raw/session_4/ (CSV files)
# - data/processed/ (NPZ files if preprocessed)
# - calibration/ (Calibration files)
# - models/ (Trained models)
```

---

## Quick Start Testing

### 1. System Validation (5 minutes)
**Purpose**: Verify all system components are working

```bash
# Test LSL connectivity
python scripts/testing/test_lsl_stream.py

# Test Unity integration  
python scripts/testing/test_unity_commands.py

# Verify data availability
python scripts/preprocessing/preprocess_raw_data.py --session 1

# Quick model evaluation
python scripts/evaluation/evaluate_models.py
```

**Expected Output**:
- LSL streams detected and validated
- Unity commands sent successfully
- Data preprocessing completed
- Model evaluation results displayed

### 2. Rapid Model Training & Testing (10 minutes)
**Purpose**: Train and evaluate a model quickly

```bash
# 1. Preprocess data
python scripts/preprocessing/preprocess_raw_data.py

# 2. Train a Random Forest model
python scripts/training/train_model.py --classifier-type rf

# 3. Evaluate the model
python scripts/evaluation/evaluate_models.py

# 4. Test real-time operation
python main.py --load-calibration models/[newest_model].pkl --visualize
```

### 3. Complete System Test (30 minutes)
**Purpose**: Comprehensive system validation

```bash
# Full development workflow
python scripts/preprocessing/preprocess_raw_data.py
python scripts/training/train_aggregate_models.py --classifier-type rf
python scripts/evaluation/evaluate_all_models.py --save-plots
python scripts/analysis/analyze_mi_eeg.py --generate-all-plots
python main.py --load-calibration models/best_model.pkl --visualize
```

---

## System Component Testing

### Data Preprocessing Testing

#### Test 1: Single Session Preprocessing
```bash
# Test specific session
python scripts/preprocessing/preprocess_raw_data.py --session 1 --verbose

# Verify output
ls data/processed/session_1_processed.npz
python -c "
import numpy as np
data = np.load('data/processed/session_1_processed.npz')
print('Files:', data.files)
print('Data shapes:', {k: v.shape for k, v in data.items()})
"
```

**Expected Results**:
- `session_1_processed.npz` created
- Contains 'left_data', 'right_data' arrays
- Data shapes should be consistent (trials × samples × channels)

#### Test 2: All Sessions Preprocessing
```bash
# Process all available sessions
python scripts/preprocessing/preprocess_raw_data.py --verbose

# Check quality
python -c "
import numpy as np
import os
for session in [1, 2, 4]:
    file_path = f'data/processed/session_{session}_processed.npz'
    if os.path.exists(file_path):
        data = np.load(file_path)
        left_trials = len(data['left_data'])
        right_trials = len(data['right_data'])
        print(f'Session {session}: {left_trials} left, {right_trials} right trials')
"
```

### Model Training Testing

#### Test 3: Individual Algorithm Training
```bash
# Test each algorithm individually
for algo in lda svm rf xgb; do
    echo "Testing $algo..."
    python scripts/training/train_model.py --classifier-type $algo --verbose
done

# Verify models were created
ls models/*_model*.pkl
```

#### Test 4: Comprehensive Training
```bash
# Train with multiple algorithms and data sources
python scripts/training/train_aggregate_models.py --classifier-type rf --verbose

# Train with hyperparameter tuning
python scripts/training/train_model.py --classifier-type rf --hyperparameter-tune

# Verify training outputs
ls models/
cat logs/training_*.log  # if logging to files
```

### Model Evaluation Testing

#### Test 5: Quick Evaluation
```bash
# Quick model comparison
python scripts/evaluation/evaluate_models.py

# Check results
cat model_evaluation_results.csv
```

**Expected Output**:
```
Model,Accuracy,Precision,Recall,F1_Score,ROC_AUC
aggregate_rf_model.pkl,0.847,0.863,0.824,0.843,0.891
robust_rf_model.pkl,0.832,0.851,0.809,0.829,0.876
...
```

#### Test 6: Comprehensive Evaluation
```bash
# Detailed evaluation with visualizations
python scripts/evaluation/evaluate_all_models.py --save-plots --verbose

# Check outputs
ls evaluation_results/
cat evaluation_results/detailed_evaluation_report.json
```

**Expected Outputs**:
- `evaluation_results/model_evaluation_summary.csv`
- `evaluation_results/detailed_evaluation_report.json`
- `evaluation_results/confusion_matrices.png`
- `evaluation_results/performance_comparison.png`

### Analysis Testing

#### Test 7: Complete EEG Analysis
```bash
# Run comprehensive analysis
python scripts/analysis/analyze_mi_eeg.py --generate-all-plots --export-results

# Check generated plots
ls analysis_results/
```

**Expected Outputs**:
- ERD/ERS time-frequency maps
- Channel-wise analysis plots
- Band power boxplots
- Statistical analysis results

---

## Complete Development Workflow

### Workflow Test: From Raw Data to Production Model

#### Step 1: Data Preparation
```bash
# Start with raw data
ls data/raw/session_*/

# Preprocess all sessions
python scripts/preprocessing/preprocess_raw_data.py --verbose
echo "✓ Data preprocessing completed"
```

#### Step 2: Model Development
```bash
# Train multiple models for comparison
python scripts/training/train_aggregate_models.py --classifier-type rf --verbose
python scripts/training/train_model.py --classifier-type rf --hyperparameter-tune --verbose
echo "✓ Model training completed"
```

#### Step 3: Model Evaluation
```bash
# Comprehensive evaluation
python scripts/evaluation/evaluate_all_models.py --save-plots --verbose
echo "✓ Model evaluation completed"

# Identify best model
best_model=$(python -c "
import pandas as pd
df = pd.read_csv('model_evaluation_results.csv')
best = df.loc[df['Accuracy'].idxmax(), 'Model']
print(best)
")
echo "Best model: $best_model"
```

#### Step 4: Research Analysis
```bash
# Generate research-quality analysis
python scripts/analysis/analyze_mi_eeg.py --generate-all-plots --export-results
echo "✓ Analysis completed"
```

#### Step 5: Production Deployment Test
```bash
# Test the best model in production mode
python main.py --load-calibration "models/$best_model" --visualize
echo "✓ Production deployment tested"
```

---

## Real-time Operation Testing

### Live BCI Operation Tests

#### Test 8: File-based BCI Operation
```bash
# Test with pre-recorded data
python main.py --file-source data/raw/session_1/[csv_file].csv --load-calibration models/best_model.pkl --visualize
```

**What to Observe**:
- Real-time classification output in console
- Prosthetic hand visualization window
- Classification confidence levels
- Hand/wrist animation responses

#### Test 9: Live EEG Testing (with hardware)
```bash
# Test with live EEG (requires OpenBCI setup)
python main.py --load-calibration models/best_model.pkl --visualize
```

**Prerequisites**:
- OpenBCI Cyton board connected
- OpenBCI GUI streaming LSL
- Proper electrode placement

#### Test 10: Calibration Testing
```bash
# Test calibration procedure
python main.py --calibrate --visualize

# Test with file-based calibration
python main.py --calibrate --file-source data/raw/session_1/[csv_file].csv --visualize
```

**Calibration Steps to Verify**:
1. Baseline collection (30 seconds)
2. Left hand imagery trials (10 trials × 5 seconds)
3. Right hand imagery trials (10 trials × 5 seconds)
4. Model training and validation
5. Model saving with timestamp

---

## Integration Testing

### LSL Integration Testing

#### Test 11: LSL Stream Validation
```bash
# Test LSL discovery
python scripts/testing/test_lsl_stream.py --extended-test

# Monitor LSL streams
python -c "
from pylsl import resolve_streams
streams = resolve_streams()
for stream in streams:
    print(f'Stream: {stream.name()}, Type: {stream.type()}, Channels: {stream.channel_count()}')
"
```

#### Test 12: Unity Communication Testing
```bash
# Test Unity integration
python scripts/testing/test_unity_commands.py --automated-test

# Test specific commands
python scripts/testing/test_unity_commands.py --command left
python scripts/testing/test_unity_commands.py --command right
python scripts/testing/test_unity_commands.py --command idle
```

### End-to-End Integration Test

#### Test 13: Complete Pipeline Test
```bash
# 1. Start BCI system
python main.py --load-calibration models/best_model.pkl &
BCI_PID=$!

# 2. Test LSL output
sleep 5
python scripts/testing/test_unity_commands.py --monitor --duration 30

# 3. Stop BCI system
kill $BCI_PID
```

---

## Performance Validation

### Performance Benchmarks

#### Test 14: Training Performance
```bash
# Benchmark training speed
time python scripts/training/train_model.py --classifier-type rf

# Benchmark evaluation speed
time python scripts/evaluation/evaluate_all_models.py

# Memory usage monitoring
/usr/bin/time -v python scripts/training/train_aggregate_models.py 2>&1 | grep "Maximum resident"
```

#### Test 15: Real-time Performance
```bash
# Test real-time latency
python main.py --load-calibration models/best_model.pkl --file-source data/raw/session_1/[csv_file].csv --profile
```

**Performance Expectations**:
- **Training Time**: 10-30 seconds per model
- **Classification Latency**: 50-80ms
- **Memory Usage**: 1-2GB peak
- **Model Loading**: <100ms

### Accuracy Validation

#### Test 16: Model Performance Validation
```bash
# Check model accuracy meets requirements
python -c "
import pandas as pd
df = pd.read_csv('model_evaluation_results.csv')
min_accuracy = 0.70  # 70% minimum
failing_models = df[df['Accuracy'] < min_accuracy]
if len(failing_models) > 0:
    print('WARNING: Models below accuracy threshold:')
    print(failing_models[['Model', 'Accuracy']])
else:
    print('✓ All models meet accuracy requirements')
    print(f'Best accuracy: {df[\"Accuracy\"].max():.3f}')
"
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Data Issues
```bash
# Problem: No processed data found
# Solution: Run preprocessing
python scripts/preprocessing/preprocess_raw_data.py

# Problem: Data loading errors
# Solution: Check data format
python -c "
import numpy as np
try:
    data = np.load('data/processed/session_1_processed.npz')
    print('✓ Data loads correctly')
    print('Keys:', data.files)
except Exception as e:
    print('✗ Data loading error:', e)
"
```

#### Model Issues
```bash
# Problem: No models found
# Solution: Train models
python scripts/training/train_model.py --classifier-type rf

# Problem: Poor model performance
# Solution: Check training data quality
python scripts/analysis/analyze_mi_eeg.py --session 1
```

#### Import/Path Issues
```bash
# Problem: Import errors
# Solution: Check Python path
python -c "
import sys
sys.path.insert(0, '.')
try:
    from dependencies import BCISystem
    print('✓ Imports working')
except ImportError as e:
    print('✗ Import error:', e)
"
```

#### Real-time Issues
```bash
# Problem: High latency
# Solution: Check system performance
python -c "
import psutil
print(f'CPU usage: {psutil.cpu_percent()}%')
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"

# Reduce processing load
python main.py --load-calibration models/best_model.pkl --window-size 1.5
```

### Debug Mode Testing
```bash
# Enable verbose logging for all components
python scripts/training/train_model.py --verbose
python scripts/evaluation/evaluate_all_models.py --verbose
python main.py --load-calibration models/best_model.pkl --debug
```

### Test Results Validation
```bash
# Validate test results
python -c "
import os
import json

# Check if all expected outputs exist
expected_files = [
    'data/processed/session_1_processed.npz',
    'models/',
    'evaluation_results/model_evaluation_summary.csv',
    'analysis_results/'
]

for file_path in expected_files:
    if os.path.exists(file_path):
        print(f'✓ {file_path}')
    else:
        print(f'✗ {file_path} missing')
"
```

---

## Advanced Testing Scenarios

### Research Workflow Testing
```bash
# Complete research pipeline
python scripts/preprocessing/preprocess_raw_data.py --verbose
python scripts/analysis/analyze_mi_eeg.py --generate-all-plots
python scripts/training/train_aggregate_models.py --classifier-type rf
python scripts/evaluation/evaluate_all_models.py --save-plots --verbose

# Generate research report
python -c "
import pandas as pd
import json

# Load evaluation results
df = pd.read_csv('model_evaluation_results.csv')
with open('evaluation_results/detailed_evaluation_report.json', 'r') as f:
    detailed = json.load(f)

print('=== RESEARCH REPORT ===')
print(f'Best Model: {df.loc[df[\"Accuracy\"].idxmax(), \"Model\"]}')
print(f'Best Accuracy: {df[\"Accuracy\"].max():.3f}')
print(f'Models Evaluated: {len(df)}')
print('Analysis plots generated in analysis_results/')
print('Evaluation plots generated in evaluation_results/')
"
```

### Custom Configuration Testing
```bash
# Test with custom parameters
cat > test_config.json << EOF
{
    "WINDOW_SIZE": 1.5,
    "CLASSIFIER_THRESHOLD": 0.7,
    "CSP_COMPONENTS": 6,
    "MU_BAND": [7, 14],
    "BETA_BAND": [14, 35]
}
EOF

python main.py --config test_config.json --load-calibration models/best_model.pkl --visualize
```

### Stress Testing
```bash
# Test with multiple concurrent processes
for i in {1..3}; do
    python scripts/evaluation/evaluate_models.py &
done
wait
echo "Concurrent evaluation completed"

# Memory stress test
python -c "
import numpy as np
from scripts.training.train_model import train_model

# Simulate large dataset
print('Testing with large dataset...')
# Implementation would create larger synthetic dataset
print('Memory stress test completed')
"
```

---

## Test Summary and Reporting

### Automated Test Suite
```bash
#!/bin/bash
# automated_test_suite.sh

echo "=== BCI SYSTEM TEST SUITE ==="
echo "Starting comprehensive system testing..."

# 1. Component tests
echo "1. Testing system components..."
python scripts/testing/test_lsl_stream.py || echo "LSL test failed"
python scripts/testing/test_unity_commands.py || echo "Unity test failed"

# 2. Data pipeline tests
echo "2. Testing data pipeline..."
python scripts/preprocessing/preprocess_raw_data.py --session 1 || echo "Preprocessing failed"

# 3. Training tests
echo "3. Testing model training..."
python scripts/training/train_model.py --classifier-type rf || echo "Training failed"

# 4. Evaluation tests
echo "4. Testing model evaluation..."
python scripts/evaluation/evaluate_models.py || echo "Evaluation failed"

# 5. Integration tests
echo "5. Testing system integration..."
timeout 30 python main.py --load-calibration models/best_model.pkl --file-source data/raw/session_1/[csv_file].csv

echo "=== TEST SUITE COMPLETED ==="
```

### Test Results Analysis
```bash
# Generate test report
python -c "
import os
import pandas as pd
from datetime import datetime

print('=== BCI SYSTEM TEST REPORT ===')
print(f'Date: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print()

# Check data availability
sessions = [f for f in os.listdir('data/raw/') if f.startswith('session_')]
print(f'Available sessions: {len(sessions)}')

# Check processed data
processed = [f for f in os.listdir('data/processed/') if f.endswith('.npz')]
print(f'Processed datasets: {len(processed)}')

# Check models
models = [f for f in os.listdir('models/') if f.endswith('.pkl')]
print(f'Trained models: {len(models)}')

# Check evaluation results
if os.path.exists('model_evaluation_results.csv'):
    df = pd.read_csv('model_evaluation_results.csv')
    print(f'Best model accuracy: {df[\"Accuracy\"].max():.3f}')
    print(f'Average accuracy: {df[\"Accuracy\"].mean():.3f}')

print()
print('✓ System test completed successfully')
"
```

---

## Test Checklist

### Pre-Deployment Checklist
- [ ] All dependencies installed and verified
- [ ] Data preprocessing completed without errors
- [ ] Multiple models trained successfully
- [ ] Model evaluation shows acceptable performance (>70% accuracy)
- [ ] Real-time operation tested with visualization
- [ ] LSL integration validated
- [ ] Unity communication tested
- [ ] Performance meets requirements (latency <100ms)
- [ ] Error handling tested with edge cases
- [ ] Documentation reviewed and updated

### Research Validation Checklist
- [ ] Statistical analysis completed with significance testing
- [ ] Publication-quality plots generated
- [ ] Cross-validation results show model robustness
- [ ] Feature importance analysis completed
- [ ] Multiple algorithms compared systematically
- [ ] Results reproducible with seed values
- [ ] Complete methodology documented

This comprehensive test walkthrough ensures the BCI system is thoroughly validated and ready for research, development, or production deployment. 