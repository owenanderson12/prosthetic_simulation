"""
Band Power Visualization Module

This module provides functions for visualizing band power in EEG signals.
"""

import matplotlib.pyplot as plt
import numpy as np
from modules.calculate_bandpower import bandpower_boxplot

def plot_bandpower_boxplots(epochs_dict, sample_rate, epoch_start):
    """
    Creates boxplots showing the normalized power in mu and beta frequency bands for each condition.
    Uses CH3 for left side and CH6 for right side analysis.
    
    Parameters:
    -----------
    epochs_dict : dict
        Dictionary containing epochs for each condition
    sample_rate : float
        Sampling rate in Hz
    epoch_start : float
        Start time of epoch relative to event (negative for baseline)
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
                epoch[:, contra_ch], sample_rate, condition, f"Contra_{condition}", epoch_start
            )
            mu_powers.append(mu_power_contra)
            beta_powers.append(beta_power_contra)
            labels.append(f"{condition} Contra")
            
            # Calculate power for ipsilateral hemisphere
            mu_power_ipsi, beta_power_ipsi = bandpower_boxplot(
                epoch[:, ipsi_ch], sample_rate, condition, f"Ipsi_{condition}", epoch_start
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