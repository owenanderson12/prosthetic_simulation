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
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

########################################################################
# Configuration
########################################################################

DATA_FILE = "/Users/owenanderson/Documents/NeurEx/Projects/Prosthetics/prosthetic/data/raw/MI_EEG_20250325_195036.csv" 
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

def bandpower_boxplot(signal, fs, condition_name, channel_name):
    """
    Computes the bandpower of the given 1D signal in the mu and beta frequency bands,
    normalized relative to the baseline period.
    
    Parameters:
    -----------
    signal : array-like
        The EEG signal data
    fs : float
        Sampling rate in Hz
    condition_name : str
        Name of the condition (e.g., "Right" or "Left")
    channel_name : str
        Name of the channel (e.g., "CH2")
    """
    # Define frequency bands
    mu_band = (8, 13)  # Hz
    beta_band = (13, 30)  # Hz
    
    # Extract baseline period (first second of data)
    baseline_samples = int(abs(EPOCH_START) * fs)
    baseline_signal = signal[:baseline_samples]
    task_signal = signal[baseline_samples:]
    
    # Compute power spectral density using Welch's method for both periods
    from scipy.signal import welch
    
    # Use the same nperseg for both periods to ensure consistent frequency arrays
    nperseg = min(len(baseline_signal), len(task_signal))
    f, Pxx_baseline = welch(baseline_signal, fs=fs, nperseg=nperseg)
    f, Pxx_task = welch(task_signal, fs=fs, nperseg=nperseg)
    
    # Calculate power in each band for both periods
    mu_mask = (f >= mu_band[0]) & (f <= mu_band[1])
    beta_mask = (f >= beta_band[0]) & (f <= beta_band[1])
    
    # Calculate baseline power
    mu_power_baseline = np.mean(Pxx_baseline[mu_mask])
    beta_power_baseline = np.mean(Pxx_baseline[beta_mask])
    
    # Calculate task power
    mu_power_task = np.mean(Pxx_task[mu_mask])
    beta_power_task = np.mean(Pxx_task[beta_mask])
    
    # Normalize to percent change from baseline
    mu_power_norm = ((mu_power_task - mu_power_baseline) / mu_power_baseline) * 100
    beta_power_norm = ((beta_power_task - beta_power_baseline) / beta_power_baseline) * 100
    
    return mu_power_norm, beta_power_norm

def plot_bandpower_boxplots(epochs_dict):
    """
    Creates boxplots showing the normalized power in mu and beta frequency bands for each condition.
    Uses CH3 for left side and CH6 for right side analysis.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary containing epochs for each condition
    """
    # Initialize lists to store power values
    mu_powers = []
    beta_powers = []
    labels = []
    
    # Process each condition
    for condition in ["Right", "Left"]:
        if condition not in epochs_dict:
            continue
            
        epochs = epochs_dict[condition]
        
        # Process each epoch
        for epoch in epochs:
            # Get contralateral and ipsilateral channels
            if condition == "Right":
                contra_ch = 2  # CH3 (Left hemisphere)
                ipsi_ch = 5    # CH6 (Right hemisphere)
            else:
                contra_ch = 5  # CH6 (Right hemisphere)
                ipsi_ch = 2    # CH3 (Left hemisphere)
            
            # Calculate power for contralateral hemisphere
            mu_power_contra, beta_power_contra = bandpower_boxplot(
                epoch[:, contra_ch], SAMPLE_RATE, condition, f"Contra_{condition}"
            )
            mu_powers.append(mu_power_contra)
            beta_powers.append(beta_power_contra)
            labels.append(f"{condition} Contra")
            
            # Calculate power for ipsilateral hemisphere
            mu_power_ipsi, beta_power_ipsi = bandpower_boxplot(
                epoch[:, ipsi_ch], SAMPLE_RATE, condition, f"Ipsi_{condition}"
            )
            mu_powers.append(mu_power_ipsi)
            beta_powers.append(beta_power_ipsi)
            labels.append(f"{condition} Ipsi")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot mu band power
    data_mu = [mu_powers[i::4] for i in range(4)]  # Group by condition and hemisphere
    ax1.boxplot(data_mu, labels=["Right Contra (CH6)", "Right Ipsi (CH3)", "Left Contra (CH3)", "Left Ipsi (CH6)"])
    ax1.set_title("μ Band Power (8-13 Hz)")
    ax1.set_ylabel("Power Change from Baseline (%)")
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Plot beta band power
    data_beta = [beta_powers[i::4] for i in range(4)]  # Group by condition and hemisphere
    ax2.boxplot(data_beta, labels=["Right Contra (CH6)", "Right Ipsi (CH3)", "Left Contra (CH3)", "Left Ipsi (CH6)"])
    ax2.set_title("β Band Power (13-30 Hz)")
    ax2.set_ylabel("Power Change from Baseline (%)")
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def plot_wavelet_spectrogram(signal, fs, condition_name, channel_name):
    """
    Computes a continuous wavelet transform of the given 1D signal
    and plots a time-frequency spectrogram optimized for ERD/ERS visualization.

    signal       : 1D array
    fs           : Sampling rate
    condition_name : "Left" or "Right"
    channel_name : e.g. "CH2"
    """
    # Define scales focusing on motor-related frequency bands (mu and beta rhythms)
    freq_min, freq_max = 4, 30  # Hz - focus on mu/alpha (8-13Hz) and beta (13-30Hz)
    
    # Create logarithmically spaced scales to get better frequency resolution
    # Use more scales for smoother frequency resolution
    n_scales = 80  # Increased for better resolution
    scales = np.logspace(np.log10(fs/freq_max), np.log10(fs/freq_min), num=n_scales)
    waveletname = 'morl'

    # Apply light smoothing to reduce noise in the signal
    signal_smooth = np.convolve(signal, np.ones(5)/5, mode='same')

    # Continuous wavelet transform
    coefficients, freqs = pywt.cwt(signal_smooth, scales, waveletname, 1.0/fs)
    power = (np.abs(coefficients)) ** 2

    # Apply light smoothing in time dimension to reduce striping
    power = gaussian_filter(power, sigma=(0.5, 2))

    # Create time axis
    t_points = len(signal)
    time_axis = np.linspace(EPOCH_START, EPOCH_END, t_points, endpoint=False)
    
    # Use baseline period for normalization (pre-stimulus, -1.0 to 0s)
    baseline_mask = (time_axis >= BASELINE_START) & (time_axis < BASELINE_END)
    baseline_power = np.mean(power[:, baseline_mask], axis=1, keepdims=True)
    
    # Convert to percent change from baseline - better for ERD/ERS visualization
    # Negative values = ERD (desynchronization), Positive values = ERS (synchronization)
    power_norm = (power - baseline_power) / baseline_power * 100

    # Plot spectrogram with adjusted size to match reference images
    plt.figure(figsize=(10, 6))
    condition_title = "Left Hand Movement" if condition_name == "Left" else "Right Hand Movement"
    plt.title(f"{condition_title} - {channel_name}\nTime-Frequency Map (% change from baseline)")
    
    # Check the actual range of values for proper scaling
    actual_min = np.min(power_norm)
    actual_max = np.max(power_norm)
    # Optimize color scaling for ERD/ERS visualization
    # Set fixed scale but ensure it covers the actual data range
    vmin = max(actual_min, -50)  # Set fixed scale for ERD (blue)
    vmax = min(actual_max, 100)  # Set fixed scale for ERS (red)
    
    print(f"Power range for {channel_name}: {actual_min:.1f} to {actual_max:.1f}")
    print(f"Using color scale from {vmin:.1f} to {vmax:.1f}")
    
    # Need to flip the power_norm array to match the orientation in the reference image
    # We want low frequencies at the top, so we flip the array vertically
    power_norm_flipped = np.flipud(power_norm)
    
    # Plot with explicit extent to match the reference image
    # Use the exact time and frequency limits to ensure proper scaling
    im = plt.imshow(power_norm_flipped, 
                    extent=[time_axis[0], time_axis[-1], freq_min, freq_max],
                    aspect='auto', 
                    cmap='jet',
                    vmin=vmin, vmax=vmax)
    
    # Add colorbar on the right side (like in reference image)
    cbar = plt.colorbar(im, label='Power Change from Baseline (%)')
    
    # Add marker lines
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    plt.axvline(x=4.0, color='k', linestyle='--', alpha=0.7)
    
    # Add frequency band markers with dotted lines (like in reference image)
    plt.axhline(y=8, color='k', linestyle=':', alpha=0.5)
    plt.axhline(y=13, color='k', linestyle=':', alpha=0.5)
    plt.axhline(y=30, color='k', linestyle=':', alpha=0.5)
    
    # Add text labels for frequency bands to the right of the plot
    plt.text(time_axis[-1] + 0.2, 10.5, 'μ (8-13 Hz)', verticalalignment='center')
    plt.text(time_axis[-1] + 0.2, 21.5, 'β (13-30 Hz)', verticalalignment='center')
    
    # Set y-axis ticks to invert the direction (low frequencies at top)
    plt.gca().invert_yaxis()
    
    # Set exact axis limits to match reference
    plt.xlim(time_axis[0], time_axis[-1])
    
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    
    # Add text box with hemisphere info in top left (like reference image)
    hemisphere = "Left" if channel_name in ["CH2", "CH3", "CH3-CH2"] else "Right"
    plt.text(0.02, 0.92, f"Channel: {channel_name}\nHemisphere: {hemisphere}", 
            transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Also plot the ERD/ERS time course for mu and beta bands
    plot_erd_time_course(power_norm, freqs, time_axis, condition_name, channel_name)


def plot_erd_time_course(power_norm, freqs, time_axis, condition_name, channel_name):
    """
    Plot the ERD/ERS time course for mu and beta frequency bands.
    This shows the time course of power changes which is clearer for ERD analysis.
    
    power_norm: Normalized power (% change from baseline)
    freqs: Frequency array
    time_axis: Time axis
    condition_name: "Left" or "Right"
    channel_name: e.g. "CH2"
    """
    # Define frequency bands
    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    
    # Average power change across frequencies in each band
    mu_power = np.mean(power_norm[mu_mask, :], axis=0)
    beta_power = np.mean(power_norm[beta_mask, :], axis=0)
    
    # Apply smoothing using scipy's savgol_filter which preserves array length
    
    # Use a window size of about 200ms
    window_size = int(0.2 * SAMPLE_RATE)
    # Make sure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    # Make sure window size is smaller than the signal length
    window_size = min(window_size, len(mu_power) - 1)
    
    # Apply the filter with quadratic polynomial
    mu_power_smooth = savgol_filter(mu_power, window_size, 2)
    beta_power_smooth = savgol_filter(beta_power, window_size, 2)
    
    # Plot ERD/ERS time course
    plt.figure(figsize=(12, 5))  # Adjusted to match reference image
    condition_title = "Left Hand Movement" if condition_name == "Left" else "Right Hand Movement"
    plt.title(f"{condition_title} - {channel_name}\nEvent-Related Desynchronization/Synchronization")
    
    # Ensure arrays have same length (they should already, but just to be safe)
    assert len(time_axis) == len(mu_power_smooth), f"Time axis length {len(time_axis)} != mu power length {len(mu_power_smooth)}"
    assert len(time_axis) == len(beta_power_smooth), f"Time axis length {len(time_axis)} != beta power length {len(beta_power_smooth)}"
    
    # Plot lines for mu and beta bands
    plt.plot(time_axis, mu_power_smooth, 'b-', linewidth=2, label='μ (8-13 Hz)')
    plt.plot(time_axis, beta_power_smooth, 'r-', linewidth=2, label='β (13-30 Hz)')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Highlight the ERD area
    plt.fill_between(time_axis, 0, mu_power_smooth, where=(mu_power_smooth < 0), color='b', alpha=0.3, label='μ ERD')
    plt.fill_between(time_axis, 0, beta_power_smooth, where=(beta_power_smooth < 0), color='r', alpha=0.3, label='β ERD')
    
    # Add marker lines
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7, label='Imagery Start')
    plt.axvline(x=4.0, color='k', linestyle='--', alpha=0.7, label='Imagery Stop')
    
    # Auto-scale y-axis based on data
    # First get the min and max values
    all_data = np.concatenate([mu_power_smooth, beta_power_smooth])
    data_min = np.min(all_data)
    data_max = np.max(all_data)
    
    # Add a margin of 10% of the range
    margin = 0.1 * (data_max - data_min)
    
    # Ensure zero is included in the range
    if data_min > 0:
        data_min = -margin  # Include some negative space if all data is positive
    if data_max < 0:
        data_max = margin   # Include some positive space if all data is negative
    
    # Set the limits with margins
    y_min = data_min - margin
    y_max = data_max + margin
    
    # Set limits, ensuring the range is reasonable (not too compressed or expanded)
    # If the range includes both ERD and ERS, make sure we can see them well
    # Override with fixed limits if auto-scaling produces too small or too large a range
    plt.ylim(y_min, y_max)
    
    print(f"Y-axis range for {channel_name} time course: {y_min:.1f} to {y_max:.1f}")
    
    plt.xlim(time_axis[0], time_axis[-1])
    
    plt.xlabel("Time (s)")
    plt.ylabel("Power Change from Baseline (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text indicating the meaning of negative values
    plt.text(0.05, 0.05, "Negative values = ERD (Motor Imagery)", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()


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

def apply_grand_average_reference(data):
    """
    Apply grand average reference to the EEG data.
    This involves subtracting the mean of all channels from each channel.
    
    Parameters:
    -----------
    data : array (n_samples, n_channels)
        The EEG data to be referenced
        
    Returns:
    --------
    referenced_data : array (n_samples, n_channels)
        The grand average referenced EEG data
    """
    # Calculate grand average across all channels
    grand_avg = np.mean(data, axis=1, keepdims=True)
    
    # Subtract grand average from each channel
    referenced_data = data - grand_avg
    
    return referenced_data

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
    #                condition_name=f"{condition_label} ({region_name})"
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
                #plot_wavelet_spectrogram(signal_1d, SAMPLE_RATE, condition_label, ch_name)

    # Add bandpower boxplots after the wavelet analysis
    print("\n== Generating Band Power Boxplots ==")
    plot_bandpower_boxplots(epochs_dict)


if __name__ == "__main__":
    main()
