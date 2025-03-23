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

Requirements:
    pip install numpy pandas scipy matplotlib pywt

Usage:
    python analyze_MI_EEG.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt

########################################################################
# Configuration
########################################################################

DATA_FILE = "/Users/owenanderson/Documents/NeurEx/Projects/Prosthetics/prosthetic/data/raw/MI_EEG_20250322_173724.csv"  # <-- Replace with your actual CSV filename if different
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
EPOCH_START = -1.0  # Start 1 second before the marker
EPOCH_END = 4.0    # End 4 seconds after the marker

# Define baseline period for normalization
BASELINE_START = -1.0  # Start of baseline period
BASELINE_END = 0.0    # End of baseline period (stimulus onset)

########################################################################
# Helper Functions
########################################################################

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter to the data.
    data  : 1D array of samples or 2D array [n_samples, n_channels]
    fs    : Sample rate
    lowcut, highcut : Filter passband edges in Hz
    order : Filter order
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def extract_epochs(signal_data, timestamps, events, fs, t_start, t_end):
    """
    Extract epochs from continuous signal_data given event timestamps.
    Includes baseline period before the event.
    
    Parameters:
    -----------
    signal_data : array (n_samples, n_channels)
        The continuous EEG data
    timestamps : array (n_samples,)
        Timestamps for each sample
    events : list of tuples (timestamp, label)
        Event markers with their timestamps
    fs : float
        Sampling rate in Hz
    t_start : float
        Start time relative to event (negative for baseline)
    t_end : float
        End time relative to event
    """
    epochs_dict = {}
    epoch_len = int((t_end - t_start) * fs)
    samples_in_baseline = int(abs(t_start) * fs)  # number of samples in baseline period

    print(f"\nEpoch extraction parameters:")
    print(f"Epoch window: {t_start}s to {t_end}s")
    print(f"Baseline period: {t_start}s to 0s")
    print(f"Samples per epoch: {epoch_len}")
    print(f"Baseline samples: {samples_in_baseline}")

    for (evt_time, evt_label) in events:
        # Convert event time to sample index
        evt_sample = np.argmin(np.abs(timestamps - evt_time))
        
        # Calculate start and end samples
        start_sample = evt_sample + int(t_start * fs)  # Will be negative offset for baseline
        end_sample = start_sample + epoch_len
        
        # Verify we have enough data
        if start_sample < 0:
            print(f"Warning: Skipping epoch at {evt_time:.2f}s - insufficient baseline data")
            continue
        if end_sample >= len(signal_data):
            print(f"Warning: Skipping epoch at {evt_time:.2f}s - reaches beyond end of recording")
            continue
            
        # Extract epoch including baseline
        epoch_data = signal_data[start_sample:end_sample, :]
        
        # Verify baseline period
        baseline_end_idx = samples_in_baseline
        baseline = epoch_data[:baseline_end_idx, :]
        if len(baseline) != samples_in_baseline:
            print(f"Warning: Incorrect baseline length at {evt_time:.2f}s")
            continue

        # Store in dictionary by condition
        if evt_label not in epochs_dict:
            epochs_dict[evt_label] = []
        epochs_dict[evt_label].append(epoch_data)

    # Convert list of epochs to numpy arrays
    for label in epochs_dict:
        epochs_dict[label] = np.array(epochs_dict[label])
        print(f"\nCondition: {label}")
        print(f"Number of epochs: {len(epochs_dict[label])}")
        print(f"Epoch shape: {epochs_dict[label].shape}")
        
        # Verify first epoch timing
        n_baseline_samples = int(abs(t_start) * fs)
        print(f"Baseline samples in each epoch: {n_baseline_samples}")

    return epochs_dict


def plot_erp(epochs_dict, fs, channel_indices, channel_labels, condition_name):
    """
    Given epoch data, compute and plot the averaged ERP for each channel.
    epochs_dict  : 3D array (n_epochs, n_timepoints, n_channels)
    fs           : Sample rate
    channel_indices : which columns in epochs to plot
    channel_labels  : channel names
    condition_name  : e.g. "Right hand" or "Left hand"
    """
    t_points = epochs_dict.shape[1]
    time_axis = np.linspace(EPOCH_START, EPOCH_END, t_points, endpoint=False)

    # Average across all epochs
    erp = np.mean(epochs_dict[:, :, channel_indices], axis=0)  # shape (n_timepoints, n_selected_channels)

    # Plot each channel
    for i, ch_idx in enumerate(channel_indices):
        plt.figure(figsize=(12, 6))
        plt.title(f"Motor Imagery ERP\n{condition_name} - {channel_labels[i]} ({len(epochs_dict)} trials averaged)")
        plt.plot(time_axis, erp[:, i], 'b-', linewidth=2, label='ERP')
        
        # Add marker lines
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Imagery Start')
        plt.axvline(x=4.0, color='g', linestyle='--', alpha=0.5, label='Imagery Stop')
        
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (µV) [filtered]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add text box with channel info
        hemisphere = "Left" if channel_labels[i] in ["CH2", "CH3"] else "Right"
        plt.text(0.02, 0.98, f"Channel: {channel_labels[i]}\nHemisphere: {hemisphere}", 
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.show()


def plot_wavelet_spectrogram(signal, fs, condition_name, channel_name):
    """
    Computes a continuous wavelet transform of the given 1D signal
    and plots a time-frequency spectrogram with normalization against the entire signal.

    signal       : 1D array
    fs           : Sampling rate
    condition_name : "Left" or "Right"
    channel_name : e.g. "CH2"
    """
    # Define scales focusing on our frequency bands of interest (roughly 4-40 Hz)
    freq_min, freq_max = 4, 40  # Hz
    scales = np.logspace(np.log10(fs/freq_max), np.log10(fs/freq_min), num=50)
    waveletname = 'morl'

    # Print debug info about signal
    print(f"\nWavelet analysis for {condition_name} - {channel_name}")
    print(f"Signal length: {len(signal)} samples")
    print(f"Time window: {EPOCH_START}s to {EPOCH_END}s")

    # Continuous wavelet transform
    coefficients, freqs = pywt.cwt(signal, scales, waveletname, 1.0/fs)
    power = (np.abs(coefficients)) ** 2

    # Create time axis
    t_points = len(signal)
    time_axis = np.linspace(EPOCH_START, EPOCH_END, t_points, endpoint=False)
    
    # Use entire signal for baseline
    baseline_power = np.mean(power, axis=1, keepdims=True)
    print(f"Global baseline power shape: {baseline_power.shape}")
    
    # Convert to percent change from baseline
    # (power - baseline_mean) / baseline_mean * 100
    power_norm = (power - baseline_power) / baseline_power * 100

    # Plot spectrogram
    plt.figure(figsize=(12, 8))
    condition_title = "Left Hand Movement" if condition_name == "Left" else "Right Hand Movement"
    plt.title(f"Motor Imagery Time-Frequency Analysis\n{condition_title} - {channel_name}\n(normalized to entire epoch average)")
    
    # Plot the spectrogram with jet colormap
    vmax = np.percentile(np.abs(power_norm), 95)  # Use 95th percentile for better scaling
    im = plt.imshow(power_norm, 
                    extent=[time_axis[0], time_axis[-1], freqs[-1], freqs[0]],  # Flip freq axis to show low freqs at bottom
                    aspect='auto', 
                    cmap='jet',  # Changed to jet colormap for better similarity to reference
                    vmin=-vmax, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Power Change from Global Average (%)')
    
    # Add marker lines
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7, label='Imagery Start')
    plt.axvline(x=4.0, color='k', linestyle='--', alpha=0.7, label='Imagery Stop')
    
    # Add frequency band markers
    plt.axhline(y=8, color='black', linestyle=':', alpha=0.5, label='µ band start')
    plt.axhline(y=13, color='black', linestyle=':', alpha=0.5, label='µ band end/β band start')
    plt.axhline(y=30, color='black', linestyle=':', alpha=0.5, label='β band end')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    # Add text box with channel info
    hemisphere = "Left" if channel_name in ["CH2", "CH3"] else "Right"
    plt.text(0.02, 0.98, f"Channel: {channel_name}\nHemisphere: {hemisphere}\nNormalized to entire epoch", 
            transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

########################################################################
# Main Analysis
########################################################################

def compute_bipolar_derivation(epochs_dict):
    """
    Compute bipolar derivations for specific channel pairs.
    Adds new virtual channels to the epochs data.
    """
    for condition in epochs_dict:
        # Get the original epoch data
        epochs = epochs_dict[condition]  # shape: (n_epochs, n_timepoints, n_channels)
        
        # Create new array with space for derivations
        n_epochs, n_timepoints, n_channels = epochs.shape
        new_epochs = np.zeros((n_epochs, n_timepoints, n_channels + 2))  # +2 for the two derivations
        
        # Copy original data
        new_epochs[:, :, :n_channels] = epochs
        
        # Compute CH3-CH2 (indices 2-1)
        new_epochs[:, :, n_channels] = epochs[:, :, 2] - epochs[:, :, 1]
        
        # Compute CH6-CH7 (indices 5-6)
        new_epochs[:, :, n_channels + 1] = epochs[:, :, 5] - epochs[:, :, 6]
        
        # Update the epochs dictionary
        epochs_dict[condition] = new_epochs
        
    return epochs_dict

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

    # 2. PREPROCESS (Bandpass filter)
    # shape (n_samples, n_channels)
    raw_eeg = df[["CH1","CH2","CH3","CH4","CH5","CH6","CH7","CH8"]].values
    filtered_eeg = bandpass_filter(raw_eeg, SAMPLE_RATE, FILTER_BAND[0], FILTER_BAND[1], order=4)

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
    for condition_label in ["Right", "Left"]:
        if condition_label not in epochs_dict:
            print(f"No epochs found for condition {condition_label}, skipping.")
            continue

        condition_epochs = epochs_dict[condition_label]

        for region_name, ch_names in CHANNELS_OF_INTEREST.items():
            for ch_name in ch_names:
                if '-' in ch_name:  # This is a bipolar derivation
                    # For bipolar channels, use the last two columns we added
                    if ch_name == "CH3-CH2":
                        ch_idx = 8  # First added derivation
                    else:  # CH6-CH7
                        ch_idx = 9  # Second added derivation
                else:
                    # For regular channels, use original indexing
                    ch_idx = int(ch_name.replace("CH", "")) - 1

                # Plot individual channel or derivation
                plot_erp(
                    epochs_dict=condition_epochs,
                    fs=SAMPLE_RATE,
                    channel_indices=[ch_idx],  # Pass single channel index
                    channel_labels=[ch_name],  # Pass single channel name
                    condition_name=f"{condition_label} ({region_name})"
                )

    # 6. WAVELET TRANSFORM & SPECTROGRAM
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
                plot_wavelet_spectrogram(signal_1d, SAMPLE_RATE, condition_label, ch_name)


if __name__ == "__main__":
    main()
