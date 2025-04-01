"""
EEG Referencing Module

This module provides functions for re-referencing EEG signals.
"""

import numpy as np

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