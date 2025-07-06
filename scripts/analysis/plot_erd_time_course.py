"""
ERD/ERS Time Course Module

This module provides functions for plotting the time course of event-related desynchronization/synchronization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def plot_erd_time_course(power_norm, freqs, time_axis, condition_name, channel_name):
    """
    Plot the ERD/ERS time course for mu and beta frequency bands.
    This shows the time course of power changes which is clearer for ERD analysis.
    
    Parameters:
    -----------
    power_norm : array
        Normalized power (% change from baseline)
    freqs : array
        Frequency array
    time_axis : array
        Time axis
    condition_name : str
        "Left" or "Right"
    channel_name : str
        e.g. "CH2"
    """
    # Define frequency bands
    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    
    # Average power change across frequencies in each band
    mu_power = np.mean(power_norm[mu_mask, :], axis=0)
    beta_power = np.mean(power_norm[beta_mask, :], axis=0)
    
    # Apply smoothing using scipy's savgol_filter which preserves array length
    # Use a window size of about 200ms
    window_size = int(0.2 * len(time_axis) / (time_axis[-1] - time_axis[0]))
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