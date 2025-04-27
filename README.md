# EEG-Controlled Prosthetic Hand System

A real-time system that enables EEG-based control of a simulated prosthetic hand using motor imagery signals.

## Overview

This system processes EEG signals from motor imagery (imagining hand movements) to control a prosthetic hand simulation. It classifies left vs. right hand imagery to trigger corresponding movements in a Unity-based hand simulation.

## Features

- Real-time EEG signal acquisition via LSL
- Advanced signal processing with spatial filtering and artifact rejection
- CSP feature extraction and LDA classification
- Guided calibration procedure for personalized models
- Adaptive classification thresholding for improved performance
- Ability to save/load calibration profiles

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd prosthetic
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Calibration Mode

To start the system in calibration mode:

```bash
python main.py --calibrate
```

Follow the on-screen instructions to complete the calibration procedure:
1. Baseline collection (relax and stay still)
2. Left/right motor imagery trials (imagine opening/closing your left or right hand)
3. Classifier training

### Using a Pre-trained Model

To use a previously saved calibration:

```bash
python main.py --load-calibration <calibration_filename>
```

### Configuration

You can modify the default configuration by editing `config.py` or by providing a custom JSON configuration file:

```bash
python main.py --config custom_config.json
```

## Connecting to EEG Hardware

The system expects an LSL stream named "OpenBCI_EEG" by default. You can change this in the configuration file.

For OpenBCI hardware:
1. Use the OpenBCI GUI to stream data via LSL
2. Set the LSL stream name to match the configured `EEG_STREAM_NAME`

## Unity Simulation Integration

To connect the processed EEG signals to a Unity hand simulation:
1. Make sure the Unity project includes LSL integration
2. Set up an LSL inlet in Unity matching the `UNITY_OUTPUT_STREAM` configuration
3. Map the classification outputs to appropriate hand movements:
   - "left" classification → hand opening/closing
   - "right" classification → wrist rotation

## Project Structure

- `main.py`: Main entry point and system orchestration
- `config.py`: Configuration parameters
- `dependencies/eeg_acquisition.py`: EEG data acquisition module
- `dependencies/signal_processor.py`: Signal processing and feature extraction
- `dependencies/classifier.py`: LDA classification with state machine
- `dependencies/calibration.py`: Guided calibration procedure

## License

[Include license information here] 