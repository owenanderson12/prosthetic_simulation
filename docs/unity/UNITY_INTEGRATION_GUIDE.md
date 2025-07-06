# Unity Integration Guide for BCI Prosthetic System

## Overview

This guide explains how to integrate the Unity prosthetic hand simulation with the BCI system. The system uses Lab Streaming Layer (LSL) to send real-time motor imagery classification results from Python to Unity.

## System Architecture

```
EEG Data Source → Python BCI System → LSL Stream → Unity Simulation
```

### Data Flow:
1. **EEG Acquisition**: Live EEG or CSV file data
2. **Signal Processing**: Filtering, CSP feature extraction
3. **Classification**: LDA classifier (left/right motor imagery)
4. **LSL Streaming**: Sends commands via "ProstheticControl" stream
5. **Unity Reception**: Receives commands and animates prosthetic hand

## NEW: Unity Connection Synchronization

The BCI system now waits for Unity to connect before starting processing. This ensures commands are not lost when Unity starts after the BCI system.

### How it Works:
1. BCI system creates the "ProstheticControl" LSL stream
2. System waits up to 30 seconds for Unity to connect
3. Unity can optionally send a "UnityReady" signal stream
4. Once connected, BCI processing begins automatically

### Bypassing Unity Wait:
If you want to test without Unity, use the `--no-wait-unity` flag:
```bash
python main.py --load-calibration calibration.npz --file-source data.csv --no-wait-unity
```

## LSL Stream Format

The `ProstheticControl` stream sends 4-channel data at irregular intervals:

| Channel | Name | Range | Description |
|---------|------|-------|-------------|
| 0 | hand_state | 0.0-1.0 | Hand position (0=closed, 1=open) |
| 1 | wrist_state | 0.0-1.0 | Wrist rotation (0=left, 1=right) |
| 2 | command_type | 0,1,2 | Command type (0=idle, 1=hand, 2=wrist) |
| 3 | confidence | 0.0-1.0 | Classification confidence |

## Setup Instructions

### 1. Python BCI System Setup

```bash
# Activate environment
conda activate eeg_project_env

# Verify all dependencies
pip install -r requirements.txt
```

### 2. Unity Project Setup

#### For Unity Source Project:
1. Copy `unity_integration/BCIProstheticController.cs` to your Unity project's Scripts folder
2. (Optional) Copy `unity_integration/UnityReadySignal.cs` for automatic BCI synchronization
3. Ensure LSL4Unity is installed in your Unity project
4. Attach the BCIProstheticController script to your prosthetic hand GameObject
5. (Optional) Attach UnityReadySignal to any GameObject for auto-sync
6. Configure the script references in the Inspector:
   - Hand Animator: Reference to the hand's Animator component
   - Wrist Transform: Reference to the wrist joint Transform
   - Adjust smoothing speeds as needed

#### For Compiled iOS Build:
The provided Xcode project already includes LSL support. To rebuild with modifications:
1. Open the Unity project source (if available)
2. Add the BCIProstheticController script
3. Build for iOS platform

### 3. Running the Complete System

#### NEW Synchronized Workflow (Recommended):

**Step 1: Start Python BCI System**
```bash
# For file-based testing
python main.py --load-calibration calibration_20250601_162544.npz --file-source data/raw/session_2/MI_EEG_20250325_195036.csv

# For live EEG
python main.py --load-calibration calibration_20250601_162544.npz
```

The system will now display:
```
========================================
Waiting for Unity application to connect...
Please start your Unity prosthetic hand simulation.
========================================
```

**Step 2: Start Unity Simulation**
- Run the Unity project in the Unity Editor
- Or build and run on your target platform

**Step 3: Automatic Connection**
- BCI system detects Unity and shows: "✓ Unity connected to ProstheticControl stream!"
- Processing begins automatically
- Motor imagery commands flow to Unity in real-time

#### Legacy Workflow (Start Unity First):
If you prefer the old workflow or need to bypass the wait:

```bash
# Skip waiting for Unity
python main.py --load-calibration calibration.npz --file-source data.csv --no-wait-unity
```

#### Configuration Options:
In `config.py`, you can adjust:
```python
WAIT_FOR_UNITY = True  # Set to False to disable waiting
UNITY_CONNECTION_TIMEOUT = 30.0  # Seconds to wait for Unity
```

## Testing the Integration

### Quick Test Script
Create a test script to send manual commands:

```python
# test_lsl_output.py
from pylsl import StreamInfo, StreamOutlet
import time
import numpy as np

# Create outlet
info = StreamInfo('ProstheticControl', 'Control', 4, 0, 'float32', 'TestBCI')
outlet = StreamOutlet(info)

print("Sending test commands...")
while True:
    # Test hand open/close
    for i in range(50):
        hand_state = i / 50.0
        outlet.push_sample([hand_state, 0.5, 1, 0.8])
        time.sleep(0.02)
    
    # Test wrist rotation
    for i in range(50):
        wrist_state = i / 50.0
        outlet.push_sample([0.5, wrist_state, 2, 0.8])
        time.sleep(0.02)
```

### Monitor LSL Streams
Use LSL tools to monitor active streams:

```bash
# Install pylsl tools
pip install pylsl

# Python script to view streams
python -c "from pylsl import resolve_streams; print([s.name() for s in resolve_streams()])"
```

## Troubleshooting

### No LSL Connection
1. Check firewall settings - LSL uses multicast
2. Ensure both applications are on the same network
3. Try specifying IP in Unity LSL resolver settings

### Poor Classification Performance
1. Re-calibrate with better electrode contact
2. Increase calibration trials in config.py
3. Ensure proper motor imagery technique
4. Check signal quality during calibration

### Laggy Animation
1. Reduce `command_smoothing` in config.py
2. Increase smoothing speeds in Unity script
3. Check system CPU usage

### Unity-Specific Issues
1. **Missing LSL4Unity**: Install from https://github.com/labstreaminglayer/LSL4Unity
2. **Script not working**: Ensure script is attached to GameObject
3. **No animation**: Check Animator parameter names match script

## Performance Optimization

### Python Side:
```python
# In config.py, adjust:
WINDOW_SIZE = 1.5  # Smaller window = faster response
CLASSIFIER_THRESHOLD = 0.7  # Higher = fewer false positives
SIMULATION_UPDATE_RATE = 30  # Lower = less CPU usage
```

### Unity Side:
```csharp
// In BCIProstheticController:
handSmoothSpeed = 5.0f;  // Higher = snappier response
wristSmoothSpeed = 5.0f;  // Adjust to preference
```

## Example Session

1. **Terminal 1 - Start BCI System:**
```bash
cd /path/to/prosthetic
conda activate eeg_project_env
python main.py --load-calibration calibration_20250601_135846.npz --file-source data/raw/session_2/MI_EEG_20250325_195036.csv
```

2. **Unity - Start Simulation:**
   - Open Unity project or run built application
   - Verify "BCI Prosthetic Control" GUI shows "Connected"

3. **Expected Output:**
   - Python: "HAND COMMAND: OPEN/CLOSE (confidence: 0.75)"
   - Unity: Hand animation responds to commands
   - Smooth transitions between states

## Advanced Features

### Custom Command Mapping
Modify `simulation_interface.py` to add new commands:

```python
# Add new command types
elif class_name == 'custom':
    cmd_type = 3  # New command type
    # Add custom logic
```

### Multi-class Extension
The system can be extended for more than 2 classes:
1. Collect additional calibration data
2. Modify classifier for multi-class
3. Update Unity script to handle new commands

## Support

For issues or questions:
1. Check `bci_system.log` for Python errors
2. Check Unity Console for script errors
3. Verify LSL stream connectivity
4. Ensure calibration quality is sufficient 