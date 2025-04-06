# This file will be used to create the scripts to train Logan's response
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor Imagery EEG Data Collection Script

This script implements a motor imagery experiment using PsychoPy and LSL integration.
It presents instructions for left/right hand motor imagery and records EEG data
with precise timing markers.

Requirements:
  pip install pylsl psychopy pandas numpy scipy socket
"""

import threading
import time
import random
import csv
import os
import sys
from collections import deque
from datetime import datetime
import numpy as np
import logging
from scipy import signal
import socket
import json
import mmap
import struct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Configuration Parameters ---
EEG_STREAM_NAME = "OpenBCI_EEG"
MARKER_STREAM_NAME = "MI_MarkerStream"

# UDP Configuration
UDP_IP = "127.0.0.1"  # localhost
UDP_PORT = 5005
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

SAMPLE_RATE = 250  # Hz
EEG_CHANNELS = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

# Mu band parameters
MU_BAND_LOW = 8  # Hz
MU_BAND_HIGH = 13  # Hz

# Beta band parameters
BETA_BAND_LOW = 13  # Hz
BETA_BAND_HIGH = 30  # Hz

WINDOW_SIZE = 250  # samples (1 second at 250 Hz)
METER_UPDATE_RATE = 0.1  # seconds

# Simulation parameters
SIMULATION_MODE = False  # Set to True to enable keyboard control
POWER_STEP = 0.1  # How much to change power with each keypress
MIN_POWER = 0.0
MAX_POWER = 1.0

# Motor Imagery Experiment Parameters
NUM_TRIALS = 40  # Configurable number of trials
INSTRUCTION_DURATION = 2.0  # seconds to show left/right instruction
IMAGERY_DURATION = 4.0  # seconds for motor imagery
INTER_TRIAL_INTERVAL = 2.0  # seconds between trials

# Marker values
MARKER_RIGHT = "1"  # right hand imagery
MARKER_LEFT = "2"   # left hand imagery
MARKER_STOP = "3"   # end of imagery period

# File/directories
RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Data collection parameters
MERGE_THRESHOLD = 0.002  # seconds threshold for aligning EEG and marker timestamps
POLL_SLEEP = 0.001      # sleep time between polls in collector loop

# At the top with other constants
SHARED_MEMORY_NAME = "MuBandPower"
SHARED_MEMORY_SIZE = 12  # 4 bytes for mu power + 4 bytes for beta power + 4 bytes for hand type

def setup_shared_memory():
    # Create or open shared memory
    shm = mmap.mmap(-1, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME)
    return shm

def send_shared_memory_data(mu_power, beta_power, hand_type, shm):
    try:
        # Pack the data into binary format
        data = struct.pack('fff', float(mu_power), float(beta_power), float(hand_type == "right"))
        # Write to shared memory
        shm.seek(0)
        shm.write(data)
        shm.flush()
    except Exception as e:
        logging.error(f"Error writing to shared memory: {e}")

###############################################################################
#                         DATA COLLECTOR
###############################################################################
class LSLDataCollector(threading.Thread):
    """
    Background thread that collects and synchronizes EEG and marker data.
    """
    def __init__(self, stop_event, mu_callback=None, beta_callback=None):
        super().__init__()
        self.stop_event = stop_event
        self.eeg_buffer = deque()
        self.marker_buffer = deque()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = os.path.join(RAW_DATA_DIR, f"MI_EEG_{timestamp_str}.csv")
        self.eeg_inlet = None
        self.marker_inlet = None
        self.clock_offset = 0.0
        self.mu_callback = mu_callback
        self.beta_callback = beta_callback
        self.eeg_window = deque(maxlen=WINDOW_SIZE)
        self.last_mu_update = 0
        self.last_beta_update = 0
        
        # Initialize Mu band filter
        nyq = SAMPLE_RATE / 2
        mu_low = MU_BAND_LOW / nyq
        mu_high = MU_BAND_HIGH / nyq
        self.mu_b, self.mu_a = signal.butter(4, [mu_low, mu_high], btype='band')
        
        # Initialize Beta band filter
        beta_low = BETA_BAND_LOW / nyq
        beta_high = BETA_BAND_HIGH / nyq
        self.beta_b, self.beta_a = signal.butter(4, [beta_low, beta_high], btype='band')

    def process_mu_band(self, eeg_data):
        """Process EEG data to extract Mu band power."""
        if len(self.eeg_window) < WINDOW_SIZE:
            return None
            
        # Convert deque to numpy array
        eeg_window = np.array(list(self.eeg_window))
        
        # Apply bandpass filter
        filtered_data = signal.filtfilt(self.mu_b, self.mu_a, eeg_window, axis=0)
        
        # Calculate power in Mu band
        mu_power = np.mean(np.abs(filtered_data) ** 2, axis=0)
        
        # Normalize power (0-1 range)
        mu_power = (mu_power - np.min(mu_power)) / (np.max(mu_power) - np.min(mu_power))
        
        # Average across channels
        avg_mu_power = np.mean(mu_power)
        
        return avg_mu_power

    def process_beta_band(self, eeg_data):
        """Process EEG data to extract Beta band power."""
        if len(self.eeg_window) < WINDOW_SIZE:
            return None
            
        # Convert deque to numpy array
        eeg_window = np.array(list(self.eeg_window))
        
        # Apply bandpass filter
        filtered_data = signal.filtfilt(self.beta_b, self.beta_a, eeg_window, axis=0)
        
        # Calculate power in Beta band
        beta_power = np.mean(np.abs(filtered_data) ** 2, axis=0)
        
        # Normalize power (0-1 range)
        beta_power = (beta_power - np.min(beta_power)) / (np.max(beta_power) - np.min(beta_power))
        
        # Average across channels
        avg_beta_power = np.mean(beta_power)
        
        return avg_beta_power

    def resolve_streams(self):
        try:
            if SIMULATION_MODE:
                logging.info("Running in simulation mode - no LSL streams needed")
                return
                
            logging.info("Resolving EEG LSL stream...")
            from pylsl import resolve_byprop, StreamInlet
            eeg_streams = resolve_byprop("name", EEG_STREAM_NAME, timeout=10)
            if not eeg_streams:
                logging.error(f"No EEG stream found with name '{EEG_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            self.clock_offset = self.eeg_inlet.time_correction()
            logging.info(f"Computed clock offset: {self.clock_offset:.6f} seconds")

            logging.info("Resolving Marker LSL stream...")
            marker_streams = resolve_byprop("name", MARKER_STREAM_NAME, timeout=10)
            if not marker_streams:
                logging.error(f"No Marker stream found with name '{MARKER_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.marker_inlet = StreamInlet(marker_streams[0], max_buflen=360)

            logging.info("LSL streams resolved. Starting data collection...")
        except Exception as e:
            logging.exception("Exception during stream resolution:")
            sys.exit(1)

    def flush_remaining(self, writer):
        """Flush any remaining buffered data to the CSV."""
        while self.eeg_buffer or self.marker_buffer:
            if self.eeg_buffer and self.marker_buffer:
                ts_eeg, eeg_data = self.eeg_buffer[0]
                ts_marker, marker = self.marker_buffer[0]
                if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                    row = [ts_eeg] + eeg_data + [marker]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
                    self.marker_buffer.popleft()
                elif ts_marker < ts_eeg:
                    row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                    writer.writerow(row)
                    self.marker_buffer.popleft()
                else:
                    row = [ts_eeg] + eeg_data + [""]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
            elif self.eeg_buffer:
                ts_eeg, eeg_data = self.eeg_buffer.popleft()
                row = [ts_eeg] + eeg_data + [""]
                writer.writerow(row)
            elif self.marker_buffer:
                ts_marker, marker = self.marker_buffer.popleft()
                row = [ts_marker] + ([""] * len(EEG_CHANNELS)) + [marker]
                writer.writerow(row)

    def run(self):
        from pylsl import resolve_byprop, StreamInlet
        self.resolve_streams()
        try:
            with open(self.output_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = ["lsl_timestamp"] + EEG_CHANNELS + ["marker"]
                writer.writerow(header)
                while not self.stop_event.is_set():
                    if not SIMULATION_MODE:
                        try:
                            sample_eeg, ts_eeg = self.eeg_inlet.pull_sample(timeout=0.0)
                            if sample_eeg is not None and ts_eeg is not None:
                                self.eeg_buffer.append((ts_eeg, sample_eeg))
                                self.eeg_window.append(sample_eeg)
                                
                                # Process Mu band and update meter if callback exists
                                current_time = time.time()
                                if (self.mu_callback and 
                                    current_time - self.last_mu_update >= METER_UPDATE_RATE):
                                    mu_power = self.process_mu_band(sample_eeg)
                                    if mu_power is not None:
                                        self.mu_callback(mu_power)
                                        self.last_mu_update = current_time
                        except Exception as e:
                            logging.exception("Error pulling EEG sample:")
                        try:
                            sample_marker, ts_marker = self.marker_inlet.pull_sample(timeout=0.0)
                            if sample_marker is not None and ts_marker is not None:
                                adjusted_ts_marker = ts_marker - self.clock_offset
                                marker_val = sample_marker[0]
                                self.marker_buffer.append((adjusted_ts_marker, marker_val))
                        except Exception as e:
                            logging.exception("Error pulling Marker sample:")

                        # Merge data from both buffers
                        while self.eeg_buffer and self.marker_buffer:
                            ts_eeg, eeg_data = self.eeg_buffer[0]
                            ts_marker, marker = self.marker_buffer[0]
                            if abs(ts_marker - ts_eeg) < MERGE_THRESHOLD:
                                row = [ts_eeg] + eeg_data + [marker]
                                writer.writerow(row)
                                self.eeg_buffer.popleft()
                                self.marker_buffer.popleft()
                            elif ts_marker < ts_eeg:
                                row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                                writer.writerow(row)
                                self.marker_buffer.popleft()
                            else:
                                row = [ts_eeg] + eeg_data + [""]
                                writer.writerow(row)
                                self.eeg_buffer.popleft()
                    time.sleep(POLL_SLEEP)
                logging.info("Stop event set. Flushing remaining data...")
                self.flush_remaining(writer)
        except Exception as e:
            logging.exception("Exception in data collector run loop:")
        finally:
            logging.info(f"Data collection stopped. Data saved to {self.output_csv}")

###############################################################################
#                        EXPERIMENT LOGIC
###############################################################################
def run_motor_imagery_experiment():
    from pylsl import StreamInfo, StreamOutlet
    global mu_power, beta_power  # Add beta_power to global variables
    
    # Create LSL Marker stream
    try:
        marker_info = StreamInfo(MARKER_STREAM_NAME, "Markers", 1, 0, "string", "marker_id")
        marker_outlet = StreamOutlet(marker_info)
        logging.info("Marker stream created.")
    except Exception as e:
        logging.exception("Failed to create LSL Marker stream:")
        sys.exit(1)

    # PsychoPy setup
    try:
        from psychopy import visual, core, event
        win = visual.Window(
            size=(1024, 768),
            color="black",
            units="norm",
            fullscr=False
        )
        logging.info("PsychoPy window created.")
    except Exception as e:
        logging.exception("Error setting up PsychoPy window:")
        sys.exit(1)

    # Create visual stimuli
    instruction_text = visual.TextStim(win, text="", height=0.15, color="white")
    ready_text = visual.TextStim(win, text="Get Ready", height=0.15, color="white")
    
    # Create Mu band feedback meter (right side)
    meter_width = 0.2
    meter_height = 0.6
    mu_meter_pos = (0.7, 0)  # Position on right side of screen
    mu_meter_bg = visual.Rect(win, width=meter_width, height=meter_height, 
                          pos=mu_meter_pos, fillColor='gray', lineColor='white')
    mu_meter_fill = visual.Rect(win, width=meter_width, height=0, 
                           pos=mu_meter_pos, fillColor='green')
    mu_meter_text = visual.TextStim(win, text="Mu Band Power", height=0.05,
                               pos=(mu_meter_pos[0], mu_meter_pos[1] + meter_height/2 + 0.1))
    
    # Create Beta band feedback meter (left side)
    beta_meter_pos = (-0.7, 0)  # Position on left side of screen
    beta_meter_bg = visual.Rect(win, width=meter_width, height=meter_height, 
                          pos=beta_meter_pos, fillColor='gray', lineColor='white')
    beta_meter_fill = visual.Rect(win, width=meter_width, height=0, 
                           pos=beta_meter_pos, fillColor='blue')
    beta_meter_text = visual.TextStim(win, text="Beta Band Power", height=0.05,
                               pos=(beta_meter_pos[0], beta_meter_pos[1] + meter_height/2 + 0.1))
    
    # Create simulation instructions
    sim_text = visual.TextStim(win, text="", height=0.05, color="white",
                             pos=(0, -0.8))
    if SIMULATION_MODE:
        sim_text.text = "Simulation Mode: Use UP/DOWN arrows to control Mu power, LEFT/RIGHT for Beta power"

    # Generate randomized trial list
    trials = ["right", "left"] * (NUM_TRIALS // 2)
    random.shuffle(trials)
    
    # Show initial instructions
    instruction_text.text = "Motor Imagery Experiment\n\nImagine moving your hand when instructed\n\nPress SPACE to begin"
    instruction_text.draw()
    sim_text.draw()
    win.flip()
    event.waitKeys(keyList=["space"])
    
    # Main experiment loop
    for trial_num, hand in enumerate(trials, 1):
        # Display get ready message
        ready_text.draw()
        sim_text.draw()
        win.flip()
        core.wait(1.0)
        
        # Show instruction (right or left hand)
        instruction_text.text = f"{hand.upper()} HAND"
        instruction_text.draw()
        sim_text.draw()
        win.flip()
        core.wait(INSTRUCTION_DURATION)
        
        # Show START cue and send marker
        instruction_text.text = "START"
        instruction_text.draw()
        sim_text.draw()
        # Schedule marker to be sent on next flip
        marker_val = MARKER_RIGHT if hand == "right" else MARKER_LEFT
        win.callOnFlip(lambda m=marker_val: marker_outlet.push_sample([m]))
        win.flip()
        core.wait(1.0)  # Show START for 1 second
        
        # Show HOLD during imagery period with Mu band feedback
        instruction_text.text = "HOLD"
        start_time = time.time()
        logging.info("Starting HOLD period")
        
        while time.time() - start_time < IMAGERY_DURATION:
            # Handle keyboard input in simulation mode
            if SIMULATION_MODE:
                keys = event.getKeys(keyList=['up', 'down', 'left', 'right'])
                if keys:
                    if keys[0] == 'up':
                        mu_power = min(MAX_POWER, mu_power + POWER_STEP)
                        logging.debug(f"Mu power increased to: {mu_power}")
                    elif keys[0] == 'down':
                        mu_power = max(MIN_POWER, mu_power - POWER_STEP)
                        logging.debug(f"Mu power decreased to: {mu_power}")
                    elif keys[0] == 'right':
                        beta_power = min(MAX_POWER, beta_power + POWER_STEP)
                        logging.debug(f"Beta power increased to: {beta_power}")
                    elif keys[0] == 'left':
                        beta_power = max(MIN_POWER, beta_power - POWER_STEP)
                        logging.debug(f"Beta power decreased to: {beta_power}")
            
            # Update meter heights based on current power
            mu_meter_fill.height = mu_power * meter_height
            beta_meter_fill.height = beta_power * meter_height
            
            # Send data to Unity
            shm = setup_shared_memory()
            send_shared_memory_data(mu_power, beta_power, hand, shm)
            
            # Clear the window
            win.clearBuffer()
            
            # Draw everything
            instruction_text.draw()
            mu_meter_bg.draw()
            mu_meter_fill.draw()
            mu_meter_text.draw()
            beta_meter_bg.draw()
            beta_meter_fill.draw()
            beta_meter_text.draw()
            sim_text.draw()
            
            # Update the display
            win.flip()
            core.wait(0.01)  # Small delay to prevent CPU overload
        
        # Show STOP and send stop marker
        instruction_text.text = "STOP"
        instruction_text.draw()
        sim_text.draw()
        win.callOnFlip(lambda: marker_outlet.push_sample([MARKER_STOP]))
        win.flip()
        core.wait(1.0)
        
        # Inter-trial interval
        win.flip()  # clear screen
        core.wait(INTER_TRIAL_INTERVAL)
        
        logging.info(f"Completed trial {trial_num}/{len(trials)}: {hand} hand")
        
        # Check for quit
        if event.getKeys(keyList=["escape"]):
            break

    # Cleanup
    win.close()
    logging.info("Motor imagery experiment finished.")

###############################################################################
#                                   MAIN
###############################################################################
def main():
    logging.info("Starting Motor Imagery experiment script")
    
    # Create shared variables for Mu and Beta band power
    global mu_power, beta_power
    mu_power = 0.0
    beta_power = 0.0
    
    def update_mu_meter(power):
        global mu_power
        mu_power = power
    
    def update_beta_meter(power):
        global beta_power
        beta_power = power
    
    # Start data collector thread with Mu and Beta band callbacks
    stop_event = threading.Event()
    collector = LSLDataCollector(stop_event, mu_callback=update_mu_meter, beta_callback=update_beta_meter)
    collector.start()

    # Run the experiment
    run_motor_imagery_experiment()

    # Stop the collector thread gracefully
    logging.info("Experiment complete. Stopping data collector thread...")
    stop_event.set()
    collector.join()

    logging.info("All processes complete. Exiting script.")

if __name__ == "__main__":
    main() 
