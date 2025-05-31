#!/usr/bin/env python3
"""
EEG-Controlled Prosthetic Hand System

This main script initializes and coordinates all modules for the EEG-based
prosthetic hand control system. It manages the signal acquisition, processing,
classification, and simulation interface components.

Usage:
    python main.py [--calibrate] [--load-calibration FILE] [--config CONFIG_FILE] [--file-source FILE_PATH] [--visualize]
"""

import os
import sys
import time
import argparse
import logging
import threading
import json
from typing import Dict, Any

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system configuration
import config

# Add dependencies directory to the path
DEPENDENCIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dependencies')
sys.path.append(DEPENDENCIES_DIR)

# Import module components
from dependencies.eeg_acquisition import EEGAcquisition
from dependencies.signal_processor import SignalProcessor
from dependencies.classifier import Classifier
from dependencies.calibration import CalibrationSystem, CalibrationStage
from dependencies.BCISystem import BCISystem
# The simulation interface will be imported when you implement it
# from simulation_interface import SimulationInterface
# Same for visualization
# from visualization import Visualization

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EEG-Controlled Prosthetic Hand System")
    
    parser.add_argument('--calibrate', action='store_true', 
                      help='Start the system in calibration mode')
    parser.add_argument('--load-calibration', type=str, metavar='FILE',
                      help='Load a saved calibration file')
    parser.add_argument('--config', type=str, metavar='CONFIG_FILE',
                      help='Path to an alternative configuration file')
    parser.add_argument('--file-source', type=str, metavar='FILE_PATH',
                      help='Use a pre-recorded EEG file as data source instead of live acquisition')
    parser.add_argument('--visualize', action='store_true',
                      help='Enable visualization of the prosthetic hand movements')
    
    return parser.parse_args()

def load_config(config_file=None):
    """
    Load configuration from a file or use the default.
    
    Args:
        config_file: Path to a custom configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                
            # Merge with default config
            for key, value in custom_config.items():
                setattr(config, key, value)
                
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
    
    # Convert config module to dictionary
    config_dict = {key: getattr(config, key) for key in dir(config) 
                  if not key.startswith('__') and not callable(getattr(config, key))}
    
    return config_dict

def main():
    """Main entry point for the BCI system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config_dict = load_config(args.config)
    
    try:
        # Determine data source type
        source_type = "file" if args.file_source else "live"
        source_path = args.file_source if args.file_source else None
        
        # Create BCI system
        bci_system = BCISystem(config_dict, source_type=source_type, source_path=source_path)
        
        # Start the system
        if not bci_system.start():
            print("Failed to start BCI system. Check logs for details.")
            return 1
        
        # If visualization is not needed and not explicitly requested, disable it
        if not args.visualize and bci_system.visualization:
            bci_system.visualization.stop()
            bci_system.visualization = None
            print("Visualization disabled. Use --visualize to enable it.")
        
        # Load calibration if specified
        if args.load_calibration:
            if bci_system.load_calibration(args.load_calibration):
                print(f"Calibration loaded from {args.load_calibration}")
            else:
                print(f"Failed to load calibration from {args.load_calibration}")
                return 1
        
        # Start calibration if specified
        if args.calibrate:
            print("Starting calibration mode...")
            bci_system.start_calibration(lambda state: print(f"Calibration: {state['message']}"))
            
            # Wait for calibration to complete (this would be handled by UI in a real app)
            try:
                while bci_system.calibration_mode:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nCalibration interrupted.")
                bci_system.stop()
                return 0
        
        # If calibration loaded or completed, start processing
        if args.load_calibration or args.calibrate:
            print("Starting BCI processing...")
            bci_system.start_processing()
        
        # Keep running until interrupted
        print("BCI system running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            bci_system.stop()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
