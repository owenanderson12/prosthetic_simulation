"""
ERP Plotting Module

This module provides functions for plotting Event-Related Potentials from EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_erp(epochs_dict, fs, channel_indices, channel_labels, condition_name, epoch_start, epoch_end):
    """
    Given epoch data, compute and plot the averaged ERP for each channel.
    
    Parameters:
    -----------
    epochs_dict  : 3D array (n_epochs, n_timepoints, n_channels)
        Epoch data for a specific condition
    fs           : float
        Sample rate in Hz
    channel_indices : list
        Which columns in epochs to plot
    channel_labels  : list
        Channel names
    condition_name  : str
        e.g. "Right hand" or "Left hand"
    epoch_start : float
        Start time of epoch relative to event (negative for baseline)
    epoch_end : float
        End time of epoch relative to event
    """
    t_points = epochs_dict.shape[1]
    time_axis = np.linspace(epoch_start, epoch_end, t_points, endpoint=False)

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