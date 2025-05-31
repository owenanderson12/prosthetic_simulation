"""
Band Power Analysis Module

This module provides functions for calculating band power in EEG signals.
"""

import numpy as np
from scipy.signal import welch

def calculate_bandpower(signal, fs, epoch_start):
    """
    Computes the bandpower of the given 1D signal in the mu and beta frequency bands,
    normalized relative to the baseline period.
    
    Parameters:
    -----------
    signal : array-like
        The EEG signal data
    fs : float
        Sampling rate in Hz
    epoch_start : float
        Start time of epoch relative to event (negative for baseline)
        
    Returns:
    --------
    mu_power_norm : float
        Normalized power in the mu band (8-13 Hz)
    beta_power_norm : float
        Normalized power in the beta band (13-30 Hz)
    """
    # Define frequency bands
    mu_band = (8, 13)  # Hz
    beta_band = (13, 30)  # Hz
    
    # Extract baseline period (first second of data)
    baseline_samples = int(abs(epoch_start) * fs)
    baseline_signal = signal[:baseline_samples]
    task_signal = signal[baseline_samples:]
    
    # Compute power spectral density using Welch's method for both periods
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

def bandpower_boxplot(signal, fs, condition, position, epoch_start):
    """
    Compute the band power for boxplot visualization of mu and beta bands.
    This function calculates power in the mu and beta frequency bands for a given signal,
    normalized to the baseline period.
    
    Parameters:
    -----------
    signal : array-like
        The EEG signal data (1D array)
    fs : float
        Sampling rate in Hz
    condition : str
        Condition label ('Right' or 'Left')
    position : str
        Position label ('Contra_Right', 'Ipsi_Right', 'Contra_Left', 'Ipsi_Left')
    epoch_start : float
        Start time of epoch relative to event (negative for baseline)
        
    Returns:
    --------
    mu_power_norm : float
        Normalized power in the mu band (8-13 Hz) as percentage change from baseline
    beta_power_norm : float
        Normalized power in the beta band (13-30 Hz) as percentage change from baseline
    """
    # Define frequency bands
    mu_band = (8, 13)  # Hz
    beta_band = (13, 30)  # Hz
    
    # Extract baseline period and task period
    baseline_samples = int(abs(epoch_start) * fs)
    baseline_signal = signal[:baseline_samples]
    task_signal = signal[baseline_samples:]
    
    # Compute power spectral density using Welch's method for both periods
    # Use the same nperseg for both to ensure consistent frequency arrays
    nperseg = min(len(baseline_signal), len(task_signal))
    nperseg = min(nperseg, fs*2)  # Limit to 2-second windows maximum
    
    f_baseline, Pxx_baseline = welch(baseline_signal, fs=fs, nperseg=nperseg)
    f_task, Pxx_task = welch(task_signal, fs=fs, nperseg=nperseg)
    
    # Ensure frequency arrays are identical
    if not np.array_equal(f_baseline, f_task):
        # Use the shorter one if they differ
        min_len = min(len(f_baseline), len(f_task))
        f = f_baseline[:min_len]
        Pxx_baseline = Pxx_baseline[:min_len]
        Pxx_task = Pxx_task[:min_len]
    else:
        f = f_baseline
    
    # Calculate power in each band for both periods
    mu_mask = (f >= mu_band[0]) & (f <= mu_band[1])
    beta_mask = (f >= beta_band[0]) & (f <= beta_band[1])
    
    # Calculate mean power in each band for baseline
    mu_power_baseline = np.mean(Pxx_baseline[mu_mask])
    beta_power_baseline = np.mean(Pxx_baseline[beta_mask])
    
    # Calculate mean power in each band for task
    mu_power_task = np.mean(Pxx_task[mu_mask])
    beta_power_task = np.mean(Pxx_task[beta_mask])
    
    # Normalize to percent change from baseline
    mu_power_norm = ((mu_power_task - mu_power_baseline) / mu_power_baseline) * 100
    beta_power_norm = ((beta_power_task - beta_power_baseline) / beta_power_baseline) * 100
    
    return mu_power_norm, beta_power_norm 