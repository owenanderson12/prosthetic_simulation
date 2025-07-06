# EEG-Controlled Prosthetic Hand System

A comprehensive, real-time Brain-Computer Interface (BCI) system that enables EEG-based control of prosthetic devices using motor imagery signals. The system processes neural signals to classify left vs. right hand motor intentions and translates them into control commands for prosthetic simulations.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧠 Overview

This system implements a complete BCI pipeline from raw EEG acquisition to prosthetic control with comprehensive training, evaluation, and analysis capabilities:

- **Real-time Processing**: Continuous 2-second windowed analysis with 75% overlap
- **Advanced Classification**: Multiple ML algorithms (LDA, SVM, Random Forest, XGBoost) with CSP spatial filtering
- **Comprehensive Training**: Robust model training with proper train/validation/test splits
- **Evaluation Suite**: Detailed performance analysis and model comparison tools
- **Data Processing**: Complete preprocessing pipeline from raw EEG to processed datasets
- **Unity Integration**: Full prosthetic simulation control via LSL
- **Analysis Tools**: Offline ERD/ERS analysis and comprehensive visualization

### Key Performance Metrics
- **Latency**: 50-80ms processing delay
- **Accuracy**: 75-85% classification performance on test data
- **Reliability**: Robust artifact rejection and connection handling
- **Scalability**: Supports multiple models and evaluation frameworks

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd prosthetic

# Create virtual environment (Python 3.8+ required)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Hardware Setup

**For OpenBCI Cyton Board:**
1. Install [OpenBCI GUI](https://openbci.com/downloads)
2. Connect Cyton board and start streaming
3. Enable LSL output in OpenBCI GUI
4. Set LSL stream name to "OpenBCI_EEG"

**Electrode Placement (10-20 System):**
- CH1: Reference electrode
- CH2: C3 (Left motor cortex)
- CH3: CP1 (Left centro-parietal) 
- CH4: Cz (Central reference)
- CH5: CP2 (Right centro-parietal)
- CH6: C4 (Right motor cortex)
- CH7: Additional channel
- CH8: Ground

### 3. Data Preprocessing

```bash
# Preprocess raw EEG data files
python scripts/preprocessing/preprocess_raw_data.py

# Or process specific session
python scripts/preprocessing/preprocess_raw_data.py --session 1
```

### 4. Model Training

```bash
# Train comprehensive models with multiple algorithms
python scripts/training/train_aggregate_models.py

# Train robust models with proper validation
python scripts/training/train_model.py --classifier-type rf

# Train Random Forest specialized model
python scripts/training/train_rf_model.py
```

### 5. Model Evaluation

```bash
# Evaluate all trained models
python scripts/evaluation/evaluate_all_models.py

# Quick model comparison
python scripts/evaluation/evaluate_models.py
```

### 6. BCI System Operation

```bash
# Start with calibration
python main.py --calibrate

# Use trained model
python main.py --load-calibration models/best_model.pkl

# With visualization
python main.py --load-calibration models/best_model.pkl --visualize
```

## 📋 System Architecture

### Core Components

#### Main System (`main.py`)
- **Entry Point**: Command-line interface with multiple modes
- **System Management**: Lifecycle management and configuration loading
- **Integration**: Coordinates all system components

#### BCI System (`dependencies/BCISystem.py`)
- **System Orchestration**: Manages all BCI components
- **Data Flow**: Coordinates signal processing and classification
- **State Management**: Handles calibration and processing modes
- **Dynamic Configuration**: Runtime parameter adjustment

#### Signal Processing (`dependencies/signal_processor.py`)
- **Spatial Filtering**: Common Spatial Pattern (CSP) implementation
- **Temporal Filtering**: Butterworth bandpass filtering
- **Feature Extraction**: ERD/ERS quantification and band power analysis
- **Artifact Rejection**: Adaptive thresholding and quality assessment

#### Classification (`dependencies/classifier.py`)
- **Multiple Algorithms**: LDA, SVM, Random Forest, XGBoost support
- **Confidence Estimation**: Probabilistic classification with thresholds
- **Adaptive Performance**: Dynamic threshold adjustment
- **Model Persistence**: Save/load trained models

#### Data Sources (`dependencies/data_source.py`, `dependencies/eeg_acquisition.py`)
- **Live EEG**: Real-time LSL stream processing
- **File Replay**: CSV file-based data simulation
- **Artificial Data**: Synthetic data generation for testing

#### Calibration System (`dependencies/calibration.py`)
- **Guided Procedure**: Step-by-step calibration process
- **Quality Monitoring**: Real-time signal quality assessment
- **Model Training**: Integrated CSP + classifier training
- **Validation**: Performance assessment and model saving

### Scripts Organization

#### Training Scripts (`scripts/training/`)
- **`train_aggregate_models.py`**: Comprehensive model training with multiple algorithms
- **`train_model.py`**: Robust model training with proper validation splits
- **`train_rf_model.py`**: Specialized Random Forest model training

#### Evaluation Scripts (`scripts/evaluation/`)
- **`evaluate_all_models.py`**: Comprehensive model evaluation and comparison
- **`evaluate_models.py`**: Quick model performance assessment

#### Analysis Scripts (`scripts/analysis/`)
- **`analyze_mi_eeg.py`**: Complete offline EEG analysis with ERD/ERS visualization

#### Preprocessing Scripts (`scripts/preprocessing/`)
- **`preprocess_raw_data.py`**: Raw EEG data preprocessing pipeline

#### Testing Scripts (`scripts/testing/`)
- **`test_lsl_stream.py`**: LSL stream connectivity testing
- **`test_unity_commands.py`**: Unity integration testing

### Analysis Modules (`modules/`)
- **Signal Processing**: Bandpass filtering, epoch extraction, artifact rejection
- **Visualization**: ERD/ERS plotting, spectrogram generation, band power analysis
- **Feature Extraction**: CSP implementation, band power calculation
- **Quality Assessment**: Signal quality monitoring and validation

## 🔧 Usage Examples

### Training Workflow

```bash
# 1. Preprocess raw data
python scripts/preprocessing/preprocess_raw_data.py

# 2. Train multiple models
python scripts/training/train_aggregate_models.py --classifier-type rf

# 3. Train robust validation model
python scripts/training/train_model.py --classifier-type rf --hyperparameter-tune

# 4. Evaluate all models
python scripts/evaluation/evaluate_all_models.py --save-plots
```

### BCI Operation Modes

```bash
# Calibration mode
python main.py --calibrate --visualize

# Live operation with best model
python main.py --load-calibration models/best_model.pkl

# File-based testing
python main.py --file-source data/raw/session_1/test_data.csv --load-calibration models/test_model.pkl

# Unity integration
python main.py --load-calibration models/unity_model.pkl --no-wait-unity
```

### Analysis and Visualization

```bash
# Complete EEG analysis
python scripts/analysis/analyze_mi_eeg.py

# Custom analysis with specific modules
python -c "
from modules.plot_wavelet_spectrogram import plot_wavelet_spectrogram
from modules.plot_erd_time_course import plot_erd_time_course
# Your analysis code here
"
```

## ⚙️ Configuration

### System Configuration (`config.py`)

```python
# Signal Processing Parameters
SAMPLE_RATE = 250              # Hz - EEG sampling rate
WINDOW_SIZE = 2.0              # seconds - Classification window
WINDOW_OVERLAP = 0.5           # seconds - Window overlap
MU_BAND = (8, 13)              # Hz - Mu rhythm frequency band
BETA_BAND = (13, 30)           # Hz - Beta rhythm frequency band

# Classification Parameters
CLASSIFIER_THRESHOLD = 0.65    # Confidence threshold for actions
MIN_CONFIDENCE = 0.55          # Minimum confidence for output
ADAPTIVE_THRESHOLD = True      # Enable adaptive thresholding
CSP_COMPONENTS = 4             # Number of CSP spatial filters

# Training Parameters
CALIBRATION_TRIALS = 10        # Trials per class during calibration
TRIAL_DURATION = 5.0           # seconds - Length of each trial
BASELINE_DURATION = 30.0       # seconds - Baseline collection period
```

### Model Training Configuration

```bash
# Different classifier types
python scripts/training/train_aggregate_models.py --classifier-type lda
python scripts/training/train_aggregate_models.py --classifier-type svm
python scripts/training/train_aggregate_models.py --classifier-type rf
python scripts/training/train_aggregate_models.py --classifier-type xgb

# Channel selection
python scripts/training/train_aggregate_models.py --channels 2 3 6 7

# Data exclusion
python scripts/training/train_aggregate_models.py --exclude-session-3
```

## 📊 Model Evaluation

### Comprehensive Evaluation

The system provides detailed model evaluation with:

- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-Validation**: K-fold validation with statistical analysis
- **Confusion Matrices**: Detailed classification analysis
- **Model Comparison**: Side-by-side performance comparison
- **Visualization**: Performance plots and confusion matrix heatmaps

```bash
# Generate comprehensive evaluation report
python scripts/evaluation/evaluate_all_models.py --save-plots --verbose

# Results saved to:
# - evaluation_results/detailed_evaluation_report.json
# - evaluation_results/model_evaluation_summary.csv
# - evaluation_results/confusion_matrices.png
# - evaluation_results/performance_comparison.png
```

### Performance Analysis

```bash
# Quick model comparison
python scripts/evaluation/evaluate_models.py

# Generates:
# - model_evaluation_results.csv
# - Console summary with best models by category
```

## 🏗️ Project Structure

```
prosthetic/
├── main.py                    # Main BCI system entry point
├── config.py                  # System configuration
├── requirements.txt           # Python dependencies
│
├── scripts/                   # Organized scripts
│   ├── training/              # Model training scripts
│   │   ├── train_aggregate_models.py
│   │   ├── train_model.py
│   │   └── train_rf_model.py
│   ├── evaluation/            # Model evaluation scripts
│   │   ├── evaluate_all_models.py
│   │   └── evaluate_models.py
│   ├── analysis/              # Data analysis scripts
│   │   └── analyze_mi_eeg.py
│   ├── preprocessing/         # Data preprocessing scripts
│   │   └── preprocess_raw_data.py
│   ├── testing/               # System testing scripts
│   │   ├── test_lsl_stream.py
│   │   └── test_unity_commands.py
│   └── config/                # Configuration files
│       └── config.py
│
├── dependencies/              # Core BCI system modules
│   ├── BCISystem.py          # Main system orchestrator
│   ├── signal_processor.py   # Signal processing pipeline
│   ├── classifier.py         # Classification engine
│   ├── calibration.py        # Calibration system
│   ├── data_source.py        # Data source abstraction
│   ├── eeg_acquisition.py    # Live EEG acquisition
│   ├── file_handler.py       # File-based data handling
│   ├── simulation_interface.py # Unity integration
│   └── visualization.py      # Real-time visualization
│
├── modules/                   # Analysis and utility modules
│   ├── bandpass_filter.py    # Signal filtering
│   ├── extract_epochs.py     # Epoch extraction
│   ├── calculate_*.py        # Feature calculation
│   ├── plot_*.py            # Visualization functions
│   ├── NeurofeedbackProcessor.py
│   └── LSLDataCollector.py
│
├── unity_integration/         # Unity integration components
│   ├── BCIProstheticController.cs
│   └── UnityReadySignal.cs
│
├── data/                      # Data storage
│   ├── raw/                  # Raw EEG recordings
│   └── processed/            # Processed datasets
│
├── models/                    # Trained models
├── calibration/              # Calibration data
├── evaluation_results/       # Evaluation outputs
├── analysis_results/         # Analysis outputs
└── docs/                     # Documentation
    ├── README.md
    ├── SYSTEM_COMPLETE.md
    ├── ARCHITECTURE.md
    └── API_REFERENCE.md
```

## 🔌 Unity Integration

### LSL Communication

The system creates an LSL output stream "ProstheticControl" for Unity:

```csharp
// Unity C# Script Example
using LSL;

public class BCIProstheticController : MonoBehaviour 
{
    private StreamInlet inlet;
    
    void Start() 
    {
        // Find and connect to BCI stream
        StreamInfo[] streams = LSL.resolve_stream("name", "ProstheticControl");
        if (streams.Length > 0) 
        {
            inlet = new StreamInlet(streams[0]);
        }
    }
    
    void Update() 
    {
        float[] sample = new float[4];
        if (inlet != null && inlet.pull_sample(sample, 0.0) != 0) 
        {
            ProcessBCICommand(sample);
        }
    }
    
    void ProcessBCICommand(float[] data)
    {
        // data[0] = hand_state (0.0-1.0)
        // data[1] = wrist_state (0.0-1.0)
        // data[2] = command_type (0=idle, 1=left, 2=right)
        // data[3] = confidence (0.0-1.0)
        
        if (data[3] > 0.65) // Confidence threshold
        {
            if (data[2] == 1) // Left hand command
            {
                // Control hand open/close
                ControlHandOpenClose(data[0]);
            }
            else if (data[2] == 2) // Right hand command
            {
                // Control wrist rotation
                ControlWristRotation(data[1]);
            }
        }
    }
}
```

### Command Mapping

- **Left Hand Imagery** → Hand open/close control
- **Right Hand Imagery** → Wrist rotation control
- **Idle State** → Maintain current position
- **Confidence < Threshold** → No action

## 🐛 Troubleshooting

### Common Issues

**Model Training Failures**
```bash
# Check data availability
ls data/processed/
ls calibration/

# Verify preprocessing
python scripts/preprocessing/preprocess_raw_data.py --session 1

# Check model training with verbose output
python scripts/training/train_model.py --verbose
```

**Poor Classification Performance**
```bash
# Evaluate all models to find best performer
python scripts/evaluation/evaluate_all_models.py

# Try different algorithms
python scripts/training/train_aggregate_models.py --classifier-type rf
python scripts/training/train_aggregate_models.py --classifier-type svm

# Check data quality
python scripts/analysis/analyze_mi_eeg.py
```

**LSL Connection Issues**
```bash
# Test LSL connectivity
python scripts/testing/test_lsl_stream.py

# Check available streams
python -c "from pylsl import resolve_streams; print([s.name() for s in resolve_streams()])"
```

**Unity Integration Problems**
```bash
# Test Unity commands
python scripts/testing/test_unity_commands.py

# Check LSL output
python main.py --load-calibration models/best_model.pkl --no-wait-unity
```

## 📈 Performance Optimization

### Model Selection

```bash
# Compare all trained models
python scripts/evaluation/evaluate_all_models.py

# Best practices:
# 1. Use Random Forest for robust performance
# 2. Ensure proper train/validation/test splits
# 3. Apply cross-validation for model selection
# 4. Monitor overfitting with separate test sets
```

### Real-time Performance

```python
# In config.py - optimize for speed
WINDOW_SIZE = 1.5              # Smaller window for faster processing
WINDOW_OVERLAP = 0.25          # Less overlap for higher update rate
CSP_COMPONENTS = 3             # Fewer components for speed
```

### Data Quality

```bash
# Preprocess with quality checks
python scripts/preprocessing/preprocess_raw_data.py

# Analyze signal quality
python scripts/analysis/analyze_mi_eeg.py

# Check for artifacts and noise
# Verify electrode impedance
# Ensure proper grounding
```

## 🔬 Research and Development

### Data Collection

```bash
# Collect new calibration data
python main.py --calibrate

# Process research sessions
python scripts/preprocessing/preprocess_raw_data.py --session [N]

# Analyze motor imagery patterns
python scripts/analysis/analyze_mi_eeg.py
```

### Experimental Analysis

```bash
# Comprehensive EEG analysis
python scripts/analysis/analyze_mi_eeg.py

# Custom analysis with modules
python -c "
from modules.plot_wavelet_spectrogram import plot_wavelet_spectrogram
from modules.plot_erd_time_course import plot_erd_time_course
from modules.calculate_bandpower import bandpower_boxplot
# Your custom analysis here
"
```

### Model Development

```bash
# Experiment with different algorithms
python scripts/training/train_aggregate_models.py --classifier-type xgb
python scripts/training/train_model.py --classifier-type svm --hyperparameter-tune

# Channel selection experiments
python scripts/training/train_aggregate_models.py --channels 2 3 6 7
python scripts/training/train_aggregate_models.py --channels 3 6  # Minimal set
```

## 📚 Documentation

- **`docs/README.md`**: This comprehensive guide
- **`docs/SYSTEM_COMPLETE.md`**: System implementation details
- **`docs/ARCHITECTURE.md`**: Technical architecture overview
- **`docs/API_REFERENCE.md`**: Module and function reference
- **`docs/TEST_WALKTHROUGH.md`**: Testing procedures and validation

## 🤝 Contributing

### Development Setup

```bash
# Fork and clone repository
git clone <your-fork-url>
cd prosthetic

# Install development dependencies
pip install -r requirements.txt

# Run tests
python scripts/testing/test_lsl_stream.py
python scripts/testing/test_unity_commands.py
```

### Extension Points

1. **New Algorithms**: Add to `dependencies/classifier.py`
2. **Feature Extraction**: Extend `dependencies/signal_processor.py`
3. **Analysis Tools**: Add to `modules/`
4. **Training Methods**: Create new scripts in `scripts/training/`
5. **Evaluation Metrics**: Enhance `scripts/evaluation/`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: Comprehensive docs in the `docs/` directory
- **Examples**: Working examples in `scripts/` and `modules/`
- **Community**: Join discussions in GitHub Discussions

---

**System Requirements:**
- Python 3.8+
- OpenBCI Cyton board or compatible LSL EEG source
- 8GB+ RAM recommended for model training
- Multi-core CPU for optimal real-time performance

**Tested Platforms:**
- macOS 10.15+ (Apple Silicon and Intel)
- Ubuntu 18.04+ (x86_64)
- Windows 10+ (x86_64)

**Key Features:**
- ✅ Real-time EEG processing (50-80ms latency)
- ✅ Multiple ML algorithms (LDA, SVM, RF, XGBoost)
- ✅ Comprehensive evaluation suite
- ✅ Unity prosthetic integration
- ✅ Robust data preprocessing
- ✅ Advanced analysis tools
- ✅ Modular architecture 