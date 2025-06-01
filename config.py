"""
Configuration parameters for the EEG-controlled prosthetic hand system.

This file contains all configuration parameters used across the system modules.
"""

import os

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CALIBRATION_DIR = os.path.join(BASE_DIR, "calibration")

# Create necessary directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, CALIBRATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- LSL Stream Names ---
EEG_STREAM_NAME = "OpenBCI_EEG"
MARKER_STREAM_NAME = "MI_MarkerStream"
UNITY_OUTPUT_STREAM = "ProstheticControl"

# --- EEG Acquisition ---
SAMPLE_RATE = 250  # Hz
BUFFER_SIZE = 2500  # 10 seconds at 250Hz
CONNECTION_TIMEOUT = 10  # seconds
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]
MI_CHANNEL_INDICES = [2, 3, 5, 6]  # C3, CP1, C4, CP2 (adjust based on montage)
QUALITY_THRESHOLD = 0.6  # Signal quality threshold

# --- Signal Processing ---
WINDOW_SIZE = 2.0  # seconds
WINDOW_OVERLAP = 0.5  # seconds (75% overlap)
MU_BAND = (8, 13)  # Hz
BETA_BAND = (13, 30)  # Hz
FILTER_ORDER = 4
FILTER_BAND = (1, 45)  # Hz
ARTIFACT_AMPLITUDE_THRESHOLD = 5000  # μV (increased to handle DC offset)
ARTIFACT_VARIANCE_THRESHOLD = 1000000  # μV² (increased for raw data)

# --- Classification ---
CLASSIFIER_THRESHOLD = 0.60  # confidence level required for actuation
MIN_CONFIDENCE = 0.50
ADAPTIVE_THRESHOLD = True
CSP_COMPONENTS = 4  # number of CSP components to use

# --- Calibration ---
CALIBRATION_TRIALS = 5  # number of trials per class
TRIAL_DURATION = 5.0  # seconds
REST_DURATION = 3.0  # seconds
BASELINE_DURATION = 15.0  # seconds
CUE_DURATION = 2.0  # seconds

# --- Simulation Interface ---
HAND_OPEN_CLOSE_SPEED = 0.1  # normalized speed for open/close
WRIST_ROTATION_SPEED = 0.1  # normalized speed for rotation
COMMAND_SMOOTHING = 0.7  # smoothing factor for commands (0-1)
SIMULATION_UPDATE_RATE = 60  # Hz, target update rate for simulation
WAIT_FOR_UNITY = True  # Whether to wait for Unity to connect before starting
UNITY_CONNECTION_TIMEOUT = 30.0  # seconds to wait for Unity connection

# --- System ---
LOG_LEVEL = "DEBUG"
LOG_FILE = os.path.join(BASE_DIR, "bci_system.log")
DEBUG_MODE = False  # Enable additional debug information

# --- Feature Dictionary for Configuration UI ---
CONFIG_FEATURES = {
    "basic": {
        "title": "Basic Settings",
        "settings": [
            {"name": "WINDOW_SIZE", "type": "float", "min": 0.5, "max": 5.0, "step": 0.1, "unit": "seconds"},
            {"name": "WINDOW_OVERLAP", "type": "float", "min": 0.0, "max": 0.95, "step": 0.05, "unit": "seconds"},
            {"name": "CLASSIFIER_THRESHOLD", "type": "float", "min": 0.5, "max": 0.95, "step": 0.05}
        ]
    },
    "advanced": {
        "title": "Advanced Settings",
        "settings": [
            {"name": "MU_BAND", "type": "range", "min": 1, "max": 40, "unit": "Hz"},
            {"name": "BETA_BAND", "type": "range", "min": 1, "max": 40, "unit": "Hz"},
            {"name": "FILTER_BAND", "type": "range", "min": 0.1, "max": 100, "unit": "Hz"},
            {"name": "ADAPTIVE_THRESHOLD", "type": "bool"},
            {"name": "CSP_COMPONENTS", "type": "int", "min": 2, "max": 8, "step": 2}
        ]
    },
    "calibration": {
        "title": "Calibration Settings",
        "settings": [
            {"name": "CALIBRATION_TRIALS", "type": "int", "min": 5, "max": 30, "step": 5},
            {"name": "TRIAL_DURATION", "type": "float", "min": 3.0, "max": 10.0, "step": 0.5, "unit": "seconds"},
            {"name": "REST_DURATION", "type": "float", "min": 1.0, "max": 5.0, "step": 0.5, "unit": "seconds"},
            {"name": "BASELINE_DURATION", "type": "float", "min": 10.0, "max": 60.0, "step": 5.0, "unit": "seconds"}
        ]
    },
    "simulation": {
        "title": "Simulation Settings",
        "settings": [
            {"name": "HAND_OPEN_CLOSE_SPEED", "type": "float", "min": 0.05, "max": 0.5, "step": 0.05},
            {"name": "WRIST_ROTATION_SPEED", "type": "float", "min": 0.05, "max": 0.5, "step": 0.05},
            {"name": "COMMAND_SMOOTHING", "type": "float", "min": 0.0, "max": 0.95, "step": 0.05}
        ]
    }
} 