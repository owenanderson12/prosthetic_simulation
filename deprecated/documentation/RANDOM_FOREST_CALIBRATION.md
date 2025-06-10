# Random Forest Calibration System

This guide explains how to use the new Random Forest calibration system for the EEG-controlled prosthetic hand.

## Overview

The Random Forest calibration system provides improved accuracy over the default LDA classifier by:
- Using Random Forest with 100 trees and max depth of 10
- Combining CSP features with band power features
- Applying proper feature scaling
- Storing calibrations in a dedicated `calibration/random_forest/` folder

## Quick Start

### 1. Run Calibration

```bash
python run_rf_calibration.py
```

This will:
- Use session 2 data as the source
- Run the calibration procedure
- Train a Random Forest model
- Save results to `calibration/random_forest/`

**Follow the on-screen instructions during calibration:**
- Rest during baseline collection
- Imagine left hand movement when prompted
- Imagine right hand movement when prompted
- Complete the specified number of trials

### 2. Run the System with Random Forest

```bash
python run_rf_system.py
```

This will:
- Automatically find the latest Random Forest calibration
- Load the trained model
- Start the BCI system with session 2 data
- Begin real-time classification

## File Structure

```
calibration/random_forest/
├── rf_calibration_YYYYMMDD_HHMMSS.npz  # Calibration data
└── rf_model_YYYYMMDD_HHMMSS.pkl        # Trained Random Forest model
```

## Expected Performance

Based on previous testing:
- **Random Forest accuracy**: ~84% (vs ~60% for LDA)
- **Feature importance**: CSP1, CSP3 are most important
- **Training time**: ~30 seconds for calibration data
- **Classification speed**: Real-time capable

## Manual Usage

You can also use the standard main.py with Random Forest calibrations:

```bash
# Run calibration manually
python main.py --file-source data/raw/session_2/MI_EEG_20250325_195036.csv --calibrate --no-wait-unity

# Load specific Random Forest calibration
python main.py --file-source data/raw/session_2/MI_EEG_20250325_195036.csv --load-calibration random_forest/rf_calibration_20250607_123456.npz
```

## Troubleshooting

### No calibration files found
- Run `python run_rf_calibration.py` first
- Check that `calibration/random_forest/` directory exists

### Low accuracy during calibration
- Ensure good signal quality during data collection
- Try increasing the number of calibration trials in config
- Check that CSP filters are being trained properly

### Model loading errors
- Verify the calibration completed successfully
- Check that both .npz and .pkl files were created
- Ensure session 2 data files exist in `data/raw/session_2/`

## Configuration

Key parameters in `config.py`:
- `WINDOW_SIZE`: Time window for feature extraction (default: 2.0s)
- `WINDOW_OVERLAP`: Overlap between windows (default: 0.5s)
- `CLASSIFIER_THRESHOLD`: Confidence threshold for classification (default: 0.65)

## Comparison with LDA

| Feature | LDA | Random Forest |
|---------|-----|---------------|
| Accuracy | ~60% | ~84% |
| Feature Types | CSP only or Band Power only | CSP + Band Power combined |
| Overfitting Risk | Low | Medium (mitigated by max_depth=10) |
| Training Time | Fast | Moderate |
| Interpretability | High | Lower |
| Robustness | Good | Excellent |

The Random Forest approach provides significantly better performance for this motor imagery classification task. 