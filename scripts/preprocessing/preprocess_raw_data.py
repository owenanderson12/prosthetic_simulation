#!/usr/bin/env python3
"""
preprocess_raw_data.py

Preprocesses raw EEG data files to create properly epoched and labeled datasets
compatible with both the BCI system and analysis pipeline.

Steps:
1. Load and clean raw CSV files
2. Apply preprocessing (filtering, artifact rejection)
3. Extract and label trials
4. Save processed data in a consistent format

Usage:
    python preprocess_raw_data.py [--session SESSION_NUM] [--output OUTPUT_DIR]
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import necessary modules
import scripts.config.config as config
from dependencies.signal_processor import SignalProcessor
from scripts.preprocessing.bandpass_filter import bandpass_filter
from scripts.preprocessing.extract_epochs import extract_epochs
from scripts.preprocessing.apply_grand_average_reference import apply_grand_average_reference

# Configuration - adjust paths for scripts subdirectory
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
SAMPLE_RATE = 250  # Hz

# Event markers
MARKER_RIGHT = "1.0"  # Right hand imagery start
MARKER_LEFT = "2.0"   # Left hand imagery start
MARKER_STOP = "3.0"   # End of imagery period

# Epoching parameters (in seconds)
EPOCH_START = -2.0  # Start 2 seconds before the marker
EPOCH_END = 3.0     # End 3 seconds after the marker

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess raw EEG data files")
    parser.add_argument('--session', type=int, help='Process specific session number')
    parser.add_argument('--output', type=str, default=PROCESSED_DATA_DIR,
                      help='Output directory for processed files')
    return parser.parse_args()

def load_and_clean_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, str]]]:
    """
    Load and clean a raw CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (eeg_data, timestamps, events)
    """
    logging.info(f"Loading {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert all channels to numeric, coerce errors to NaN
    channels = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]
    for ch in channels:
        df[ch] = pd.to_numeric(df[ch], errors='coerce')
    df["lsl_timestamp"] = pd.to_numeric(df["lsl_timestamp"], errors='coerce')
    
    # Drop rows with NaN values
    df.dropna(subset=["lsl_timestamp"] + channels, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Extract timestamps relative to start
    timestamps = df["lsl_timestamp"].values - df["lsl_timestamp"].values[0]
    
    # Extract EEG data
    eeg_data = df[channels].values
    
    # Extract events
    events = []
    for i, marker_val in enumerate(df["marker"]):
        if str(marker_val) == MARKER_RIGHT:
            events.append((timestamps[i], "Right"))
        elif str(marker_val) == MARKER_LEFT:
            events.append((timestamps[i], "Left"))
    
    logging.info(f"Loaded {len(df)} samples and {len(events)} events")
    return eeg_data, timestamps, events

def preprocess_session(session_dir: str, output_dir: str) -> None:
    """
    Preprocess all files in a session directory.
    
    Args:
        session_dir: Path to session directory
        output_dir: Output directory for processed files
    """
    # Create SignalProcessor instance
    sp = SignalProcessor(config.__dict__)
    
    # Get all CSV files in the session directory
    csv_files = glob.glob(os.path.join(session_dir, "*.csv"))
    if not csv_files:
        logging.warning(f"No CSV files found in {session_dir}")
        return
    
    # Process each file
    all_baseline = []
    all_left_trials = []
    all_right_trials = []
    
    for csv_file in csv_files:
        # Load and clean data
        eeg_data, timestamps, events = load_and_clean_csv(csv_file)
        
        # Apply grand average reference
        referenced_eeg = apply_grand_average_reference(eeg_data)
        
        # Apply bandpass filter
        filtered_eeg = bandpass_filter(referenced_eeg, SAMPLE_RATE, 
                                     sp.filter_band[0], sp.filter_band[1], 
                                     order=sp.filter_order)
        
        # Reject artifacts
        clean_data, artifact_mask = sp._reject_artifacts(filtered_eeg)
        
        # Extract epochs
        epochs_dict = extract_epochs(clean_data, timestamps, events,
                                   fs=SAMPLE_RATE,
                                   t_start=EPOCH_START,
                                   t_end=EPOCH_END)
        
        # Collect baseline periods (pre-trial)
        baseline_start_idx = int(abs(EPOCH_START) * SAMPLE_RATE)
        for condition in ['Left', 'Right']:
            if condition in epochs_dict:
                for trial in epochs_dict[condition]:
                    baseline = trial[:baseline_start_idx]
                    if not np.any(artifact_mask[baseline_start_idx:baseline_start_idx+len(baseline)]):
                        all_baseline.append(baseline)
        
        # Collect trials
        if 'Left' in epochs_dict:
            all_left_trials.extend(epochs_dict['Left'])
        if 'Right' in epochs_dict:
            all_right_trials.extend(epochs_dict['Right'])
    
    # Convert lists to arrays
    baseline_data = np.vstack(all_baseline) if all_baseline else np.array([])
    
    # Create output directory if needed
    session_name = os.path.basename(session_dir)
    output_path = os.path.join(output_dir, f"{session_name}_processed.npz")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    np.savez(output_path,
             baseline_data=baseline_data,
             left_data=all_left_trials,
             right_data=all_right_trials,
             sample_rate=SAMPLE_RATE,
             epoch_start=EPOCH_START,
             epoch_end=EPOCH_END)
    
    logging.info(f"Saved processed data to {output_path}")
    logging.info(f"  Baseline periods: {len(all_baseline)}")
    logging.info(f"  Left trials: {len(all_left_trials)}")
    logging.info(f"  Right trials: {len(all_right_trials)}")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get session directories
    if args.session:
        session_dirs = [os.path.join(RAW_DATA_DIR, f"session_{args.session}")]
        if not os.path.exists(session_dirs[0]):
            logging.error(f"Session directory not found: {session_dirs[0]}")
            return
    else:
        session_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "session_*")))
    
    # Process each session
    for session_dir in session_dirs:
        logging.info(f"\nProcessing {session_dir}")
        try:
            preprocess_session(session_dir, args.output)
        except Exception as e:
            logging.exception(f"Error processing {session_dir}:")
            continue

if __name__ == "__main__":
    main() 