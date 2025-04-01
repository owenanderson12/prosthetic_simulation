"""
Epoch Extraction Module

This module provides functions for extracting epochs from continuous EEG data.
"""

import numpy as np

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
        
    Returns:
    --------
    epochs_dict : dict
        Dictionary with condition labels as keys and arrays of epochs as values
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