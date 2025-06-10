#!/usr/bin/env python3
"""
run_rf_calibration.py

Run Random Forest calibration using session 2 data as the source.
This will create a properly trained Random Forest model with the actual session data.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import main
import argparse

def run_calibration():
    """Run Random Forest calibration with session 2 data."""
    
    # Set up arguments to use session 2 data and calibration mode
    sys.argv = [
        'run_rf_calibration.py',
        '--file-source', 'data/raw/session_2/MI_EEG_20250325_195036.csv',
        '--calibrate',
        '--no-wait-unity'  # Skip Unity connection for calibration
    ]
    
    print("=" * 60)
    print("RANDOM FOREST CALIBRATION")
    print("=" * 60)
    print("Using session 2 data as source for calibration")
    print("Will save Random Forest model to calibration/random_forest/")
    print("Follow the calibration instructions when they appear")
    print("=" * 60)
    
    # Run the main system
    return main.main()

if __name__ == "__main__":
    sys.exit(run_calibration()) 