"""
Average Band Power Visualization Module

This module provides functions for visualizing average mu and beta band powers across conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def calculate_band_power(signal, fs, freq_band):
    """
    Calculate power in a specific frequency band using Welch's method.
    
    Parameters:
    -----------
    signal : array-like
        The EEG signal data
    fs : float
        Sampling rate in Hz
    freq_band : tuple
        Frequency band (low, high) in Hz
        
    Returns:
    --------
    band_power : float
        Power in the specified frequency band
    """
    # Calculate power spectral density using Welch's method
    f, Pxx = welch(signal, fs=fs)
    
    # Get the indices corresponding to the frequency band
    idx_band = np.logical_and(f >= freq_band[0], f <= freq_band[1])
    
    # Calculate mean power in the band
    band_power = np.mean(Pxx[idx_band])
    
    return band_power

def plot_average_band_powers(epochs_dict, fs, epoch_start, baseline_end=0.0):
    """
    Plot average mu and beta band powers for each condition separately.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary containing epochs for each condition
    fs : float
        Sampling rate in Hz
    epoch_start : float
        Start time of epoch relative to event (negative for baseline)
    baseline_end : float, optional
        End time of baseline period (default=0.0, stimulus onset)
    """
    # Define frequency bands
    mu_band = (8, 13)    # Hz
    beta_band = (13, 30) # Hz
    
    # Create a figure with 2 rows (mu and beta) and 2 columns (conditions)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Process each condition
    for col, condition in enumerate(["Right", "Left"]):
        if condition not in epochs_dict:
            continue
            
        # Get epochs for this condition
        epochs = epochs_dict[condition]
        
        # Convert baseline_end to sample index
        baseline_end_idx = int((baseline_end - epoch_start) * fs)
        
        # Calculate channel indices for plotting
        ch_indices = {}
        for region_name, ch_names in [("Left Hemisphere", ["CH3",]), 
                                       ("Right Hemisphere", ["CH6"])]:
            for ch_name in ch_names:
                if ch_name == "CH3-CH2":
                    ch_indices[ch_name] = 8
                elif ch_name == "CH6-CH7":
                    ch_indices[ch_name] = 9
                else:
                    ch_indices[ch_name] = int(ch_name.replace("CH", "")) - 1
        
        # Average across all epochs
        avg_epoch = np.mean(epochs, axis=0)  # shape: (timepoints, channels)
        
        # Create time array for x-axis (convert from samples to seconds)
        time = np.linspace(epoch_start, epoch_start + avg_epoch.shape[0]/fs, avg_epoch.shape[0])
        
        # Calculate and plot mu band power for each channel
        plot_title = f"{condition} Hand Imagery - μ Band Power (8-13 Hz)"
        ax = axes[0, col]
        for ch_name, ch_idx in ch_indices.items():
            # Get the signal for this channel
            signal = avg_epoch[:, ch_idx]
            
            # Calculate power across time using a sliding window
            window_size = int(0.25 * fs)  # 500ms window
            step_size = int(0.05 * fs)    # 100ms step
            times = []
            powers = []
            
            # Reference power (from baseline period)
            baseline_signal = signal[:baseline_end_idx]
            ref_power = calculate_band_power(baseline_signal, fs, mu_band)
            
            # Calculate power at each time step
            for i in range(0, len(signal) - window_size, step_size):
                segment = signal[i:i+window_size]
                # Center time of the window
                t = time[i + window_size//2]
                times.append(t)
                # Calculate power and normalize to baseline
                power = calculate_band_power(segment, fs, mu_band)
                rel_power = ((power - ref_power) / ref_power) * 100
                powers.append(rel_power)
            
            # Plot the power over time
            ax.plot(times, powers, label=ch_name)
        
        # Add vertical lines at t=0 (stimulus onset)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(plot_title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power Change from Baseline (%)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate and plot beta band power for each channel
        plot_title = f"{condition} Hand Imagery - β Band Power (13-30 Hz)"
        ax = axes[1, col]
        for ch_name, ch_idx in ch_indices.items():
            # Get the signal for this channel
            signal = avg_epoch[:, ch_idx]
            
            # Calculate power across time using a sliding window
            window_size = int(0.25 * fs)  # 500ms window
            step_size = int(0.05 * fs)    # 100ms step
            times = []
            powers = []
            
            # Reference power (from baseline period)
            baseline_signal = signal[:baseline_end_idx]
            ref_power = calculate_band_power(baseline_signal, fs, beta_band)
            
            # Calculate power at each time step
            for i in range(0, len(signal) - window_size, step_size):
                segment = signal[i:i+window_size]
                # Center time of the window
                t = time[i + window_size//2]
                times.append(t)
                # Calculate power and normalize to baseline
                power = calculate_band_power(segment, fs, beta_band)
                rel_power = ((power - ref_power) / ref_power) * 100
                powers.append(rel_power)
            
            # Plot the power over time
            ax.plot(times, powers, label=ch_name)
        
        # Add vertical lines at t=0 (stimulus onset)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(plot_title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power Change from Baseline (%)')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show() 