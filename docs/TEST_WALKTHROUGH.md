# BCI System Test Walkthrough with Pre-Saved Files

This guide walks you through testing the EEG-controlled prosthetic hand system using pre-recorded data files.

## Prerequisites

1. **Environment Setup** (if not already done):
   ```bash
   # Fix numpy issues if present
   conda deactivate
   conda activate eeg_project_env
   conda install numpy --force-reinstall
   ```

2. **Available Data Files**:
   - `data/raw/session_1/MI_EEG_20250322_183114.csv` (14MB)
   - `data/raw/session_1/MI_EEG_20250322_173724.csv` (15MB)
   - `data/raw/session_2/` (multiple files)
   - `data/raw/session_3/` (multiple files)

## Test Methods

### Method 1: Quick Automated Test

Run the automated test script that handles everything:

```bash
python test_file_simulation.py
```

This will:
1. Load the first available data file
2. Initialize the BCI system with visualization
3. Run calibration on the file data
4. Save the calibration model
5. Start real-time processing
6. Display the prosthetic hand visualization
7. Run for 30 seconds (or until Ctrl+C)

### Method 2: Manual Step-by-Step

#### Step 1: Run Calibration on File Data

```bash
python main.py --calibrate --file-source data/raw/session_1/MI_EEG_20250322_183114.csv --visualize
```

This will:
- Load the CSV file
- Show visualization window
- Run through calibration stages:
  - Baseline collection (30 seconds)
  - Left hand imagery trials (10 trials × 5 seconds)
  - Right hand imagery trials (10 trials × 5 seconds)
  - Train classifier

#### Step 2: Use Saved Calibration

After calibration is complete and saved, you can reuse it:

```bash
python main.py --load-calibration calibration/demo_calibration.pkl --file-source data/raw/session_1/MI_EEG_20250322_183114.csv --visualize
```

### Method 3: Quick Command Check

To see what commands you should run based on available files:

```bash
python quick_test_simulation.py
```

## Understanding the Visualization

When the system runs with `--visualize`, you'll see:

### Hand State Display
- **Closed fist** (0.0) ← → **Open hand** (1.0)
- Controlled by **left motor imagery** classification
- Fingers animate between closed and open positions

### Wrist Rotation Display
- **Left rotation** (0.0) ← → **Right rotation** (1.0)
- Controlled by **right motor imagery** classification
- Hand rotates around wrist joint

### Classification Info
- **Current state**: Shows "left", "right", or "idle"
- **Confidence bar**: Shows classification confidence (0-100%)
- Commands only execute when confidence > threshold (65% default)

## System Components in Action

### 1. **Data Source (File Handler)**
- Loads CSV file with EEG data and timestamps
- Simulates real-time streaming at original sample rate (250Hz)
- Provides same interface as live EEG acquisition

### 2. **Signal Processing Pipeline**
- Bandpass filtering (1-45Hz)
- Artifact rejection
- Common Average Reference (CAR)
- Band power extraction (mu: 8-13Hz, beta: 13-30Hz)
- CSP feature extraction (after calibration)

### 3. **Classification**
- Linear Discriminant Analysis (LDA)
- State machine for decision smoothing
- Adaptive confidence thresholding
- Outputs: left/right/idle with confidence scores

### 4. **Simulation Interface**
- Creates LSL output stream "ProstheticControl"
- Sends commands: [hand_state, wrist_state, command_type, confidence]
- Smooths transitions between states

### 5. **Visualization**
- Tkinter-based GUI
- Real-time animation of hand and wrist
- Classification status display
- Updates at 20Hz for smooth animation

## Expected Output

### Console Output
```
BCI system started
Visualization started
Calibration started
Calibration: Starting baseline collection...
Calibration: Baseline complete
Calibration: Starting left imagery trials...
[Progress updates...]
Calibration: Training classifier...
Calibration: Complete! Accuracy: 0.82
BCI processing started
HAND COMMAND: OPEN/CLOSE (confidence: 0.73)
HAND COMMAND: ROTATE (confidence: 0.68)
[Continues with classified commands...]
```

### Visual Output
- Window titled "Prosthetic Hand Visualization"
- Animated hand opening/closing
- Animated wrist rotation
- Real-time confidence meter
- State indicator (left/right/idle)

## Troubleshooting

### No Hand Movement
- Check confidence threshold in visualization
- Ensure calibration completed successfully
- Verify data file contains motor imagery markers

### Visualization Window Not Appearing
- Ensure `--visualize` flag is used
- Check for GUI library issues (tkinter)
- Try running without conda environment

### Poor Classification
- Data file may not have clear motor imagery patterns
- Try different data files
- Adjust classification threshold in config.py

## Advanced Testing

### Custom Configuration
Create `custom_config.json`:
```json
{
    "CLASSIFIER_THRESHOLD": 0.6,
    "WINDOW_SIZE": 1.5,
    "CSP_COMPONENTS": 6
}
```

Run with:
```bash
python main.py --config custom_config.json --file-source data/raw/session_1/MI_EEG_20250322_183114.csv --visualize
```

### Different Data Files
Test with various sessions:
```bash
# Session 2 data
python main.py --calibrate --file-source data/raw/session_2/MI_EEG_20250325_195036.csv --visualize

# Session 3 data  
python main.py --calibrate --file-source data/raw/session_3/[filename].csv --visualize
```

## Next Steps

1. **Unity Integration**: Connect Unity prosthetic hand model to LSL stream
2. **Live EEG Testing**: Replace file source with OpenBCI hardware
3. **Performance Analysis**: Run `analyze_mi_eeg.py` on collected data
4. **Custom Models**: Experiment with different ML algorithms

## Summary

The file-based testing allows you to:
- Validate the entire BCI pipeline without hardware
- Test different processing parameters
- Debug classification issues
- Demonstrate the system to others
- Develop Unity integration offline

The visualization provides immediate feedback on system performance and helps understand how motor imagery translates to prosthetic control. 