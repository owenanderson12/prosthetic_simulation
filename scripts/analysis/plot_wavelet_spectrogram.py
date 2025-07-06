"""
Wavelet Spectrogram Module

This module provides functions for computing and visualizing time-frequency representations of EEG signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import gaussian_filter
from scripts.analysis.plot_erd_time_course import plot_erd_time_course

def plot_wavelet_spectrogram(signal, fs, condition_name, channel_name, epoch_start, epoch_end, baseline_start, baseline_end):
    """
    Computes a continuous wavelet transform of the given 1D signal
    and plots a time-frequency spectrogram optimized for ERD/ERS visualization.

    Parameters:
    -----------
    signal : array-like
        1D EEG signal array
    fs : float
        Sampling rate in Hz
    condition_name : str
        "Left" or "Right"
    channel_name : str
        e.g. "CH2"
    epoch_start : float
        Start time of epoch relative to event (negative for baseline)
    epoch_end : float
        End time of epoch relative to event
    baseline_start : float
        Start time of baseline period
    baseline_end : float
        End time of baseline period
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
    time_axis = np.linspace(epoch_start, epoch_end, t_points, endpoint=False)
    
    # Use baseline period for normalization (pre-stimulus)
    baseline_mask = (time_axis >= baseline_start) & (time_axis < baseline_end)
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