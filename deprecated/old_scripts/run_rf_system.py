#!/usr/bin/env python3
"""
run_rf_system.py

Run the BCI system with the latest Random Forest calibration model.
"""

import sys
import os
import glob

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import main

def find_latest_rf_calibration():
    """Find the most recent Random Forest calibration file."""
    rf_dir = "calibration/random_forest"
    if not os.path.exists(rf_dir):
        return None
        
    calibration_files = glob.glob(os.path.join(rf_dir, "rf_calibration_*.npz"))
    if not calibration_files:
        return None
        
    # Sort by filename (contains timestamp) to get latest
    latest_file = sorted(calibration_files)[-1]
    return os.path.basename(latest_file)

def run_rf_system():
    """Run the BCI system with Random Forest model."""
    
    # Find latest Random Forest calibration
    latest_calibration = find_latest_rf_calibration()
    
    if not latest_calibration:
        print("Error: No Random Forest calibration files found!")
        print("Please run 'python run_rf_calibration.py' first to create a calibration.")
        return 1
    
    # Set up arguments to use session 2 data and load the Random Forest calibration
    sys.argv = [
        'run_rf_system.py',
        '--file-source', 'data/raw/session_2/MI_EEG_20250325_195036.csv',
        '--load-calibration', f'random_forest/{latest_calibration}',
    ]
    
    print("=" * 60)
    print("RANDOM FOREST BCI SYSTEM")
    print("=" * 60)
    print(f"Loading Random Forest calibration: {latest_calibration}")
    print("Using session 2 data as source")
    print("System will start processing automatically")
    print("=" * 60)
    
    # Run the main system
    return main.main()

if __name__ == "__main__":
    sys.exit(run_rf_system()) 