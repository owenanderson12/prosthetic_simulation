"""
Bipolar Derivation Module

This module provides functions for computing bipolar derivations of EEG signals.
"""

import numpy as np

def compute_bipolar_derivation(epochs_dict):
    """
    Compute bipolar derivations for specific channel pairs.
    Adds new virtual channels to the epochs data.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary containing epochs for each condition
        
    Returns:
    --------
    epochs_dict : dict
        Dictionary with bipolar derivations added as new channels
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