"""
Bandpass Filtering Module

This module provides functions for filtering EEG signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Apply a zero-phase Butterworth bandpass filter to the data.
    
    Parameters:
    -----------
    data  : 1D array of samples or 2D array [n_samples, n_channels]
        The EEG data to filter
    fs    : float
        Sample rate in Hz
    lowcut, highcut : float
        Filter passband edges in Hz
    order : int, optional
        Filter order (default=4)
        
    Returns:
    --------
    filtered_data : array, same shape as data
        The filtered EEG data
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data 