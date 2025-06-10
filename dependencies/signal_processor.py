import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import signal, linalg
import mne
import os
import sys
import scipy.linalg
from scipy.signal import welch

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class SignalProcessor:
    """
    Signal processor module for EEG data processing in motor imagery BCI.
    
    This module handles:
    - Spatial filtering
    - Temporal filtering
    - Artifact rejection
    - Feature extraction
    - ERD/ERS quantification
    - Data normalization
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize the signal processor.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.config = config_dict
        
        # Parameters
        self.sample_rate = config_dict.get('SAMPLE_RATE', 250)
        self.window_size = int(config_dict.get('WINDOW_SIZE', 2.0) * self.sample_rate)
        self.window_overlap = int(config_dict.get('WINDOW_OVERLAP', 0.5) * self.sample_rate)
        self.channels = config_dict.get('EEG_CHANNELS', ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
        
        # Frequency bands
        self.mu_band = config_dict.get('MU_BAND', (8, 13))
        self.beta_band = config_dict.get('BETA_BAND', (13, 30))
        
        # Motor imagery relevant channels (C3, CP1, C4, CP2)
        self.mi_channels = config_dict.get('MI_CHANNEL_INDICES', [2, 3, 5, 6])
        
        # Filter design
        self.filter_order = config_dict.get('FILTER_ORDER', 4)
        self.filter_band = config_dict.get('FILTER_BAND', (1, 45))
        self._design_filters()
        
        # Baseline statistics
        self.baseline_mu_power = {ch: None for ch in range(len(self.channels))}
        self.baseline_beta_power = {ch: None for ch in range(len(self.channels))}
        
        # CSP matrices (will be set during calibration)
        self.csp_filters = None
        self.csp_patterns = None
        self.csp_mean = None
        self.csp_std = None
        
        # Artifact rejection thresholds
        self.artifact_amplitude_threshold = config_dict.get('ARTIFACT_AMPLITUDE_THRESHOLD', 100)  # μV
        self.artifact_variance_threshold = config_dict.get('ARTIFACT_VARIANCE_THRESHOLD', 30)    # μV²
        
        # Flag for initialized state
        self.is_initialized = False
        
    def _design_filters(self) -> None:
        """Design bandpass and notch filters based on configuration."""
        # Bandpass filter for general preprocessing
        nyquist = self.sample_rate / 2
        low, high = self.filter_band
        low_normalized = low / nyquist
        high_normalized = high / nyquist
        self.bandpass_b, self.bandpass_a = signal.butter(
            self.filter_order, 
            [low_normalized, high_normalized], 
            btype='bandpass'
        )
        
        # Notch filter for power line noise (50/60 Hz)
        line_freq = 60  
        notch_low = (line_freq - 2) / nyquist
        notch_high = (line_freq + 2) / nyquist
        self.notch_b, self.notch_a = signal.butter(
            self.filter_order, 
            [notch_low, notch_high], 
            btype='bandstop'
        )
        
        # Filters for specific frequency bands
        mu_low, mu_high = self.mu_band
        mu_low_normalized = mu_low / nyquist
        mu_high_normalized = mu_high / nyquist
        self.mu_b, self.mu_a = signal.butter(
            self.filter_order, 
            [mu_low_normalized, mu_high_normalized], 
            btype='bandpass'
        )
        
        beta_low, beta_high = self.beta_band
        beta_low_normalized = beta_low / nyquist
        beta_high_normalized = beta_high / nyquist
        self.beta_b, self.beta_a = signal.butter(
            self.filter_order, 
            [beta_low_normalized, beta_high_normalized], 
            btype='bandpass'
        )
    
    def process(self, data: np.ndarray, timestamps: np.ndarray) -> Dict:
        """
        Process EEG data and extract features.
        
        Args:
            data: EEG data array (samples × channels)
            timestamps: Array of timestamps for each sample
            
        Returns:
            Dictionary with processing results and features
        """
        try:
            # Validate input
            if data is None or data.size == 0:
                return {'valid': False, 'error': 'No data provided'}
                
            if data.shape[1] != len(self.channels):
                return {'valid': False, 'error': f'Expected {len(self.channels)} channels, got {data.shape[1]}'}
                
            # Apply same preprocessing pipeline as training
            # 1. Grand average reference (Common Average Reference)
            referenced_data = self._apply_spatial_filtering(data)
            
            # 2. Apply bandpass and notch filters 
            filtered_data = self._apply_filters(referenced_data)
            
            # 3. Artifact rejection
            clean_data, artifact_mask = self._reject_artifacts(filtered_data)
            
            # Apply CSP filters if available
            csp_features = None
            if self.csp_filters is not None:
                try:
                    # Apply CSP filters to preprocessed data
                    csp_data = np.dot(clean_data, self.csp_filters.T)
                    
                    # Extract variance features
                    csp_features = np.var(csp_data, axis=0)
                    
                    # Normalize if mean and std are available
                    if self.csp_mean is not None and self.csp_std is not None:
                        csp_features = (csp_features - self.csp_mean) / self.csp_std
                        
                except Exception as e:
                    logging.error(f"Error extracting CSP features: {e}")
                    csp_features = None
            
            # Extract band power features from preprocessed data
            mu_erd = []
            beta_erd = []
            
            for ch in range(len(self.channels)):
                # Get channel data
                ch_data = clean_data[:, ch]
                
                # Extract band power
                mu_power = self._extract_band_power(ch_data, self.mu_band)
                beta_power = self._extract_band_power(ch_data, self.beta_band)
                
                # Calculate ERD/ERS
                if self.baseline_mu_power[ch] is not None and self.baseline_beta_power[ch] is not None:
                    mu_erd.append((mu_power - self.baseline_mu_power[ch]) / self.baseline_mu_power[ch])
                    beta_erd.append((beta_power - self.baseline_beta_power[ch]) / self.baseline_beta_power[ch])
                else:
                    mu_erd.append(mu_power)
                    beta_erd.append(beta_power)
            
            # Combine features
            features = {}
            if csp_features is not None:
                features['csp_features'] = csp_features
            features['erd_mu'] = np.array(mu_erd)
            features['erd_beta'] = np.array(beta_erd)
            
            return {
                'valid': True,
                'features': features,
                'timestamps': timestamps
            }
            
        except Exception as e:
            logging.exception("Error processing data:")
            return {'valid': False, 'error': str(e)}
    
    def _reject_artifacts(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify and mark artifacts in EEG data.
        
        Args:
            eeg_data: Raw EEG data (samples x channels)
            
        Returns:
            Tuple of (cleaned_data, artifact_mask)
        """
        num_samples = eeg_data.shape[0]
        artifact_mask = np.zeros(num_samples, dtype=bool)
        
        # Check amplitude artifacts
        amplitude_mask = np.any(np.abs(eeg_data) > self.artifact_amplitude_threshold, axis=1)
        artifact_mask |= amplitude_mask
        
        # Check variance artifacts (in sliding windows)
        window_length = min(int(0.2 * self.sample_rate), num_samples)  # 200ms windows
        for i in range(0, num_samples - window_length, window_length // 2):
            window = eeg_data[i:i+window_length, :]
            var = np.var(window, axis=0)
            if np.any(var > self.artifact_variance_threshold):
                artifact_mask[i:i+window_length] = True
        
        # Replace artifacts with interpolated values or previous clean values
        cleaned_data = eeg_data.copy()
        for ch in range(eeg_data.shape[1]):
            ch_data = cleaned_data[:, ch]
            ch_artifacts = artifact_mask.copy()
            
            # If all samples are artifacts, use mean
            if np.all(ch_artifacts):
                cleaned_data[:, ch] = np.mean(eeg_data[:, ch])
                continue
                
            # Interpolate artifacts
            clean_indices = np.where(~ch_artifacts)[0]
            artifact_indices = np.where(ch_artifacts)[0]
            
            if len(artifact_indices) > 0 and len(clean_indices) > 0:
                # Use nearest-value interpolation for simplicity
                for idx in artifact_indices:
                    # Find nearest clean sample
                    nearest_idx = clean_indices[np.argmin(np.abs(clean_indices - idx))]
                    cleaned_data[idx, ch] = cleaned_data[nearest_idx, ch]
        
        return cleaned_data, artifact_mask
    
    def _apply_filters(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass and notch filters to EEG data.
        
        Args:
            eeg_data: EEG data to filter
            
        Returns:
            Filtered EEG data
        """
        filtered_data = np.zeros_like(eeg_data)
        
        for ch in range(eeg_data.shape[1]):
            # Apply bandpass filter
            filtered = signal.filtfilt(self.bandpass_b, self.bandpass_a, eeg_data[:, ch])
            
            # Apply notch filter
            filtered = signal.filtfilt(self.notch_b, self.notch_a, filtered)
            
            filtered_data[:, ch] = filtered
            
        return filtered_data
    
    def _apply_spatial_filtering(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply spatial filtering to EEG data (CAR or bipolar derivations).
        
        Args:
            eeg_data: Filtered EEG data
            
        Returns:
            Spatially filtered EEG data
        """
        # Common Average Reference (CAR)
        car_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
        
        # TODO: Implement bipolar derivations if needed
        # For motor imagery, often C3-reference and C4-reference are used
        
        return car_data
    
    def _calculate_band_powers(self, eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate band powers for mu and beta frequency bands.
        
        Args:
            eeg_data: Filtered EEG data
            
        Returns:
            Dictionary with mu and beta band powers
        """
        powers = {}
        
        # Calculate mu band power
        mu_filtered = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            mu_filtered[:, ch] = signal.filtfilt(self.mu_b, self.mu_a, eeg_data[:, ch])
        
        # Power is calculated as mean squared amplitude
        mu_power = np.mean(mu_filtered ** 2, axis=0)
        powers['mu'] = mu_power
        
        # Calculate beta band power
        beta_filtered = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            beta_filtered[:, ch] = signal.filtfilt(self.beta_b, self.beta_a, eeg_data[:, ch])
        
        beta_power = np.mean(beta_filtered ** 2, axis=0)
        powers['beta'] = beta_power
        
        return powers
    
    def _calculate_erd_ers(self, band_powers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate event-related desynchronization/synchronization relative to baseline.
        
        Args:
            band_powers: Dictionary with band powers
            
        Returns:
            Dictionary with ERD/ERS values (negative = ERD, positive = ERS)
        """
        erd_ers = {}
        
        # Skip if baseline not set
        if all(v is None for v in self.baseline_mu_power.values()):
            # Return zeros if baseline not available
            erd_ers['mu'] = np.zeros_like(band_powers['mu'])
            erd_ers['beta'] = np.zeros_like(band_powers['beta'])
            return erd_ers
        
        # Calculate mu ERD/ERS: (power - baseline) / baseline * 100
        mu_erd = np.zeros_like(band_powers['mu'])
        for ch in range(len(band_powers['mu'])):
            if self.baseline_mu_power[ch] is not None and self.baseline_mu_power[ch] > 0:
                mu_erd[ch] = (band_powers['mu'][ch] - self.baseline_mu_power[ch]) / self.baseline_mu_power[ch] * 100
        
        erd_ers['mu'] = mu_erd
        
        # Calculate beta ERD/ERS
        beta_erd = np.zeros_like(band_powers['beta'])
        for ch in range(len(band_powers['beta'])):
            if self.baseline_beta_power[ch] is not None and self.baseline_beta_power[ch] > 0:
                beta_erd[ch] = (band_powers['beta'][ch] - self.baseline_beta_power[ch]) / self.baseline_beta_power[ch] * 100
        
        erd_ers['beta'] = beta_erd
        
        return erd_ers
    
    def update_baseline(self, data: np.ndarray) -> None:
        """Update baseline power values with new data.
        
        Args:
            data: EEG data array of shape (n_samples, n_channels)
        """
        if data is None or len(data) == 0:
            return

        try:
            # Apply same preprocessing pipeline as main process method
            # 1. Grand average reference
            referenced_data = self._apply_spatial_filtering(data)
            
            # 2. Apply bandpass and notch filters 
            filtered_data = self._apply_filters(referenced_data)
            
            # 3. Artifact rejection
            clean_data, artifact_mask = self._reject_artifacts(filtered_data)
            
            # Extract band power features for each channel from preprocessed data
            for ch_idx in range(clean_data.shape[1]):
                ch_data = clean_data[:, ch_idx]
                
                # Extract power in mu and beta bands
                mu_power = self._extract_band_power(ch_data, self.mu_band)
                beta_power = self._extract_band_power(ch_data, self.beta_band)
                
                # Update baseline values using exponential moving average
                if self.baseline_mu_power[ch_idx] is None:
                    self.baseline_mu_power[ch_idx] = mu_power
                    self.baseline_beta_power[ch_idx] = beta_power
                else:
                    self.baseline_mu_power[ch_idx] = 0.95 * self.baseline_mu_power[ch_idx] + 0.05 * mu_power
                    self.baseline_beta_power[ch_idx] = 0.95 * self.baseline_beta_power[ch_idx] + 0.05 * beta_power
            
            logging.debug("Baseline power values updated")
            
        except Exception as e:
            logging.error(f"Error updating baseline: {str(e)}")
    
    def train_csp(self, left_trials: List[np.ndarray], right_trials: List[np.ndarray]) -> bool:
        """
        Train CSP filters using left and right motor imagery trials.
        
        Args:
            left_trials: List of left hand motor imagery trials
            right_trials: List of right hand motor imagery trials
            
        Returns:
            Success indicator
        """
        try:
            if not left_trials or not right_trials:
                logging.error("No trials provided for CSP training")
                return False
                
            # Convert trials to numpy arrays if needed
            left_data = np.vstack([trial for trial in left_trials if isinstance(trial, np.ndarray)])
            right_data = np.vstack([trial for trial in right_trials if isinstance(trial, np.ndarray)])
            
            if left_data.size == 0 or right_data.size == 0:
                logging.error("No valid trial data for CSP training")
                return False
                
            # Ensure data has correct shape
            if left_data.shape[1] != len(self.channels) or right_data.shape[1] != len(self.channels):
                logging.error(f"Trial data has incorrect number of channels. Expected {len(self.channels)}, got {left_data.shape[1]} and {right_data.shape[1]}")
                return False
                
            # Calculate covariance matrices
            cov_left = np.cov(left_data.T)
            cov_right = np.cov(right_data.T)
            
            # Solve generalized eigenvalue problem
            eigenvalues, eigenvectors = scipy.linalg.eigh(cov_left, cov_left + cov_right)
            
            # Sort eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Select filters (use all channels)
            self.csp_filters = eigenvectors.T
            self.csp_patterns = np.linalg.pinv(self.csp_filters)
            
            # Calculate mean and std for normalization
            left_features = np.var(np.dot(left_data, self.csp_filters.T), axis=0)
            right_features = np.var(np.dot(right_data, self.csp_filters.T), axis=0)
            all_features = np.vstack([left_features, right_features])
            
            self.csp_mean = np.mean(all_features, axis=0)
            self.csp_std = np.std(all_features, axis=0)
            
            logging.info(f"Trained CSP filters: shape {self.csp_filters.shape}")
            return True
            
        except Exception as e:
            logging.exception("Error training CSP filters:")
            return False
    
    def _extract_csp_features(self, eeg_data: np.ndarray, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Extract CSP features from EEG data.
        
        Args:
            eeg_data: Filtered EEG data
            normalize: Whether to normalize features
            
        Returns:
            CSP features
        """
        if self.csp_filters is None:
            return None
        
        try:
            # Apply CSP filters
            csp_data = np.dot(eeg_data, self.csp_filters)
            
            # Compute log-variance features
            var = np.var(csp_data, axis=0)
            log_var = np.log(var + 1e-10)  # Add small constant to avoid log(0)
            
            # Normalize if requested
            if normalize and self.csp_mean is not None and self.csp_std is not None:
                # Apply z-score normalization
                features = (log_var - self.csp_mean) / (self.csp_std + 1e-10)
            else:
                features = log_var
            
            return features
            
        except Exception as e:
            logging.exception("Error extracting CSP features:")
            return None
    
    def reset(self) -> None:
        """Reset processor state (baselines, etc.)."""
        self.baseline_mu_power = {ch: None for ch in range(len(self.channels))}
        self.baseline_beta_power = {ch: None for ch in range(len(self.channels))}
        self.csp_filters = None
        self.csp_patterns = None
        self.csp_mean = None
        self.csp_std = None
        logging.info("Signal processor reset")

    def reset_baseline(self) -> None:
        """Reset only the baseline power values (keeps CSP filters)."""
        self.baseline_mu_power = {ch: None for ch in range(len(self.channels))}
        self.baseline_beta_power = {ch: None for ch in range(len(self.channels))}
        logging.info("Baseline power values reset")

    def _extract_band_power(self, data: np.ndarray, band: Tuple[float, float]) -> float:
        """Extract power in a specific frequency band using Welch's method.
        
        Args:
            data: 1D array of EEG data
            band: Tuple of (low_freq, high_freq) defining the frequency band
            
        Returns:
            float: Power in the specified frequency band
        """
        # Compute power spectral density using Welch's method
        freqs, psd = welch(data, fs=self.sample_rate, nperseg=min(256, len(data)))
        
        # Find indices corresponding to the frequency band
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        
        # Compute mean power in the band
        band_power = np.mean(psd[idx_band])
        
        return band_power
