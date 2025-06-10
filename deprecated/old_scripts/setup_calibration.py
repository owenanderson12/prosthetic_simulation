#!/usr/bin/env python3
"""
setup_calibration.py

Set up the calibration files for the random forest model.
"""

import os
import shutil
import numpy as np
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_calibration():
    """Set up calibration files for the random forest model."""
    # Create directories if they don't exist
    os.makedirs('calibration', exist_ok=True)
    
    # Source and destination paths
    source_model = 'models/best_model.pkl'
    dest_model = 'calibration/best_model.pkl'
    
    # Copy the model file
    if os.path.exists(source_model):
        shutil.copy2(source_model, dest_model)
        logging.info(f"Copied model from {source_model} to {dest_model}")
    else:
        logging.error(f"Source model not found: {source_model}")
        return False
    
    # Create minimal calibration data
    calibration_data = {
        'baseline_data': np.array([]),
        'left_data': np.array([]),
        'right_data': np.array([]),
        'left_features': np.array([]),
        'right_features': np.array([])
    }
    
    # Save calibration data
    calibration_file = 'calibration/best_model.npz'
    np.savez(calibration_file, **calibration_data)
    logging.info(f"Created calibration file: {calibration_file}")
    
    return True

if __name__ == '__main__':
    if setup_calibration():
        print("\n✅ Calibration files set up successfully!")
        print("\nTo run the BCI system with the best model:")
        print("python main.py --file-source data/raw/session_2/MI_EEG_20250325_195036.csv --load-calibration best_model.npz")
    else:
        print("\n❌ Failed to set up calibration files") 