#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_MI_EEG.py

Analyzes motor imagery EEG data collected via the motor_imagery_eeg.py script.
Steps:
  1. Load CSV data.
  2. Preprocess (bandpass filter) the data.
  3. Identify markers and epoch data based on motor imagery start events ("1"=Right, "2"=Left).
  4. Plot ERPs for the channels of interest.
  5. Perform a wavelet transform on the averaged signals and plot the spectrogram.
  6. Generate band power boxplots for comparing conditions.
  7. Plot time course of average mu and beta band powers for each condition.

Requirements:
    pip install numpy pandas scipy matplotlib pywt

Usage:
    python analyze_MI_EEG.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import modular functions
from modules.bandpass_filter import bandpass_filter
from modules.extract_epochs import extract_epochs
from modules.plot_erp import plot_erp
from modules.plot_bandpower_boxplots import plot_bandpower_boxplots
from modules.plot_wavelet_spectrogram import plot_wavelet_spectrogram
from modules.compute_bipolar_derivation import compute_bipolar_derivation
from modules.apply_grand_average_reference import apply_grand_average_reference
from modules.plot_average_band_powers import plot_average_band_powers

########################################################################
# Configuration
########################################################################

DATA_FILE = "/Users/owenanderson/Documents/NeurEx/Projects/Prosthetics/neurofeedback/MI_Neurofeedback/data/raw/MI_EEG_20250423_191831.csv" 
SAMPLE_RATE = 250  # Hz, matches the original MI script's sampling rate
FILTER_BAND = (1.0, 40.0)  # Bandpass filter range (in Hz)

# Channels of interest
CHANNELS_OF_INTEREST = {
    "LeftHemisphere": ["CH2", "CH3", "CH3-CH2"],  # Added bipolar derivation
    "RightHemisphere": ["CH6", "CH7", "CH6-CH7"]  # Added bipolar derivation
}

# Event markers from the original data-collection script
MARKER_RIGHT = "1.0"  # Right hand imagery start
MARKER_LEFT = "2.0"   # Left hand imagery start
MARKER_STOP = "3.0"   # End of imagery period

# Epoching parameters (in seconds)
# Include 1 second before the start marker for baseline
EPOCH_START = -2.0  # Start 1 second before the marker
EPOCH_END = 3.0    # End 4 seconds after the marker

# Define baseline period for normalization
BASELINE_START = -2.0  # Start of baseline period
BASELINE_END = 0.0    # End of baseline period (stimulus onset)

def main():
    # 1. LOAD CSV DATA
    df = pd.read_csv(DATA_FILE)

    # The script writes lines even if marker is empty. Let's keep only rows with valid CH1 data, ignoring partial lines.
    # We assume all channels are numeric. Force convert to numeric, coerce missing -> NaN, then drop them.
    for ch in ["CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8"]:
        df[ch] = pd.to_numeric(df[ch], errors='coerce')
    df["lsl_timestamp"] = pd.to_numeric(df["lsl_timestamp"], errors='coerce')

    # Drop rows missing a timestamp or channel data
    df.dropna(subset=["lsl_timestamp","CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert time reference from raw LSL timestamps (seconds)
    timestamps = df["lsl_timestamp"].values - df["lsl_timestamp"].values[0]

    # 2. PREPROCESS
    # First apply grand average reference
    raw_eeg = df[["CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8"]].values
    referenced_eeg = apply_grand_average_reference(raw_eeg)
    
    # Then apply bandpass filter
    filtered_eeg = bandpass_filter(referenced_eeg, SAMPLE_RATE, FILTER_BAND[0], FILTER_BAND[1], order=4)

    # 3. IDENTIFY MARKERS AND EXTRACT EVENTS
    events = []
    for i, marker_val in enumerate(df["marker"]):
        if str(marker_val) == MARKER_RIGHT:
            events.append((timestamps[i], "Right"))
        elif str(marker_val) == MARKER_LEFT:
            events.append((timestamps[i], "Left"))

    # 4. EPOCH THE DATA
    epochs_dict = extract_epochs(filtered_eeg, timestamps, events,
                                 fs=SAMPLE_RATE,
                                 t_start=EPOCH_START,
                                 t_end=EPOCH_END)

    # 4b. COMPUTE BIPOLAR DERIVATIONS
    epochs_dict = compute_bipolar_derivation(epochs_dict)

    # 5. PLOT ERPs FOR CHANNELS OF INTEREST
    # Skip this for ERD analysis (uncomment if needed)
    # for condition_label in ["Right", "Left"]:
    #    if condition_label not in epochs_dict:
    #        print(f"No epochs found for condition {condition_label}, skipping.")
    #        continue
    #
    #    condition_epochs = epochs_dict[condition_label]
    #
    #    for region_name, ch_names in CHANNELS_OF_INTEREST.items():
    #        for ch_name in ch_names:
    #            if '-' in ch_name:  # This is a bipolar derivation
    #                if ch_name == "CH3-CH2":
    #                    ch_idx = 8
    #                else:  # CH6-CH7
    #                    ch_idx = 9
    #            else:
    #                ch_idx = int(ch_name.replace("CH", "")) - 1
    #
    #            plot_erp(
    #                epochs_dict=condition_epochs,
    #                fs=SAMPLE_RATE,
    #                channel_indices=[ch_idx],
    #                channel_labels=[ch_name],
    #                condition_name=f"{condition_label} ({region_name})",
    #                epoch_start=EPOCH_START,
    #                epoch_end=EPOCH_END
    #            )

    # 6. WAVELET TRANSFORM & SPECTROGRAM OPTIMIZED FOR ERD/ERS VISUALIZATION
    print("\n== Generating ERD/ERS visualizations ==")
    for condition_label in ["Right", "Left"]:
        if condition_label not in epochs_dict:
            continue
        cond_epochs = epochs_dict[condition_label]
        if cond_epochs.shape[0] == 0:
            continue

        # Average across all trials
        avg_cond = np.mean(cond_epochs, axis=0)  # shape (epoch_samples, n_channels+2)

        for region_name, ch_names in CHANNELS_OF_INTEREST.items():
            for ch_name in ch_names:
                if '-' in ch_name:  # Bipolar derivation
                    if ch_name == "CH3-CH2":
                        ch_idx = 8
                    else:  # CH6-CH7
                        ch_idx = 9
                else:
                    ch_idx = int(ch_name.replace("CH", "")) - 1
                
                signal_1d = avg_cond[:, ch_idx]
                plot_wavelet_spectrogram(signal_1d, SAMPLE_RATE, condition_label, ch_name, 
                                         EPOCH_START, EPOCH_END, BASELINE_START, BASELINE_END)

    # 7. Generate band power boxplots
    print("\n== Generating Band Power Boxplots ==")
    plot_bandpower_boxplots(epochs_dict, SAMPLE_RATE, EPOCH_START)
    
    # 8. Plot average band powers over time for each condition
    print("\n== Generating Average Band Power Time Courses ==")
    plot_average_band_powers(epochs_dict, SAMPLE_RATE, EPOCH_START, BASELINE_END)


if __name__ == "__main__":
    main()
