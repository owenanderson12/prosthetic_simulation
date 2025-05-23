import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import signal, linalg
import mne

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
    
    def __init__(self, config: Dict):
        """
        Initialize the signal processor.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Parameters
        self.sample_rate = config.get('SAMPLE_RATE', 250)
        self.window_size = int(config.get('WINDOW_SIZE', 2.0) * self.sample_rate)
        self.window_overlap = int(config.get('WINDOW_OVERLAP', 0.5) * self.sample_rate)
        self.channels = config.get('EEG_CHANNELS', ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
        
        # Frequency bands
        self.mu_band = config.get('MU_BAND', (8, 13))
        self.beta_band = config.get('BETA_BAND', (13, 30))
        
        # Motor imagery relevant channels (C3, CP1, C4, CP2)
        self.mi_channels = config.get('MI_CHANNEL_INDICES', [2, 3, 5, 6])
        
        # Filter design
        self.filter_order = config.get('FILTER_ORDER', 4)
        self.filter_band = config.get('FILTER_BAND', (1, 45))
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
        self.artifact_amplitude_threshold = config.get('ARTIFACT_AMPLITUDE_THRESHOLD', 100)  # μV
        self.artifact_variance_threshold = config.get('ARTIFACT_VARIANCE_THRESHOLD', 30)    # μV²
        
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
        line_freq = 50  # Adjust for your region (50 Hz in Europe, 60 Hz in US)
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
    
    def process(self, eeg_data: np.ndarray, timestamps: np.ndarray) -> Dict:
        """
        Process a chunk of EEG data to extract features for motor imagery classification.
        
        Args:
            eeg_data: Raw EEG data (samples x channels)
            timestamps: Timestamps for the samples
            
        Returns:
            Dictionary containing processed features and metadata
        """
        # Check if data is valid
        if eeg_data.size == 0:
            return {
                'valid': False,
                'message': 'Empty data',
                'features': None
            }
        
        # Preprocessing
        try:
            # Artifact rejection
            clean_data, artifact_mask = self._reject_artifacts(eeg_data)
            
            if np.sum(~artifact_mask) < 0.7 * len(artifact_mask):
                return {
                    'valid': False,
                    'message': 'Too many artifacts',
                    'features': None
                }
            
            # Apply filters
            filtered_data = self._apply_filters(clean_data)
            
            # Apply spatial filtering
            spatial_filtered = self._apply_spatial_filtering(filtered_data)
            
            # Extract features
            band_powers = self._calculate_band_powers(spatial_filtered)
            erd_ers = self._calculate_erd_ers(band_powers)
            
            # Extract CSP features if available
            csp_features = self._extract_csp_features(filtered_data) if self.csp_filters is not None else None
            
            # Construct feature vector
            features = {
                'mu_power': band_powers['mu'],
                'beta_power': band_powers['beta'],
                'erd_mu': erd_ers['mu'],
                'erd_beta': erd_ers['beta'],
                'csp_features': csp_features
            }
            
            return {
                'valid': True,
                'features': features,
                'artifact_ratio': np.mean(artifact_mask),
                'timestamp': timestamps[-1]
            }
            
        except Exception as e:
            logging.exception("Error in signal processing:")
            return {
                'valid': False,
                'message': str(e),
                'features': None
            }
    
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
    
    def update_baseline(self, eeg_data: np.ndarray) -> None:
        """
        Update the baseline power values using resting state data.
        
        Args:
            eeg_data: Filtered EEG data during resting state
        """
        try:
            # Apply filters
            filtered_data = self._apply_filters(eeg_data)
            
            # Apply spatial filtering
            spatial_filtered = self._apply_spatial_filtering(filtered_data)
            
            # Calculate band powers
            band_powers = self._calculate_band_powers(spatial_filtered)
            
            # Update baseline values
            for ch in range(len(band_powers['mu'])):
                # Initialize or update with moving average
                if self.baseline_mu_power[ch] is None:
                    self.baseline_mu_power[ch] = band_powers['mu'][ch]
                    self.baseline_beta_power[ch] = band_powers['beta'][ch]
                else:
                    # Exponential moving average with 0.3 weight for new data
                    self.baseline_mu_power[ch] = 0.7 * self.baseline_mu_power[ch] + 0.3 * band_powers['mu'][ch]
                    self.baseline_beta_power[ch] = 0.7 * self.baseline_beta_power[ch] + 0.3 * band_powers['beta'][ch]
            
            logging.info("Baseline power values updated")
            
        except Exception as e:
            logging.exception("Error updating baseline:")
    
    def train_csp(self, left_trials: List[np.ndarray], right_trials: List[np.ndarray], n_components: int = 4) -> None:
        def _clean_trials(trials, hand_label):
            cleaned = []
            for idx, trial in enumerate(trials):
                original_trial = trial.copy()
                if trial.size == 0:
                    logging.warning(f"{hand_label} trial {idx} is empty. Skipping.")
                    continue
                # Handle NaN/Inf by interpolation or replacement if possible
                nan_mask = np.isnan(trial)
                inf_mask = np.isinf(trial)
                n_bad = np.sum(nan_mask | inf_mask)
                total = trial.size
                if n_bad > 0:
                    # Try to interpolate only if <20% of data is bad
                    if n_bad / total < 0.2:
                        trial = trial.astype(float)  # ensure float for NaN assignment
                        trial[nan_mask] = np.nan
                        trial[inf_mask] = np.nan
                        # Interpolate NaNs along each channel (axis 0)
                        for ch in range(trial.shape[1]):
                            channel = trial[:, ch]
                            if np.any(np.isnan(channel)):
                                not_nan = ~np.isnan(channel)
                                if np.sum(not_nan) > 1:
                                    channel[np.isnan(channel)] = np.interp(
                                        np.flatnonzero(np.isnan(channel)),
                                        np.flatnonzero(not_nan),
                                        channel[not_nan]
                                    )
                                else:
                                    # If all values are NaN, skip trial
                                    logging.warning(f"{hand_label} trial {idx} channel {ch} all NaN after interpolation. Skipping trial.")
                                    break
                        # After interpolation, check again for NaN
                        if np.isnan(trial).any() or np.isinf(trial).any():
                            logging.warning(f"{hand_label} trial {idx} still contains NaN/Inf after interpolation. Skipping.")
                            continue
                        logging.info(f"{hand_label} trial {idx}: {n_bad}/{total} ({n_bad/total:.1%}) NaN/Inf values interpolated.")
                    else:
                        logging.warning(f"{hand_label} trial {idx} contains {n_bad}/{total} ({n_bad/total:.1%}) NaN/Inf values. Too many to fix, skipping.")
                        continue
                # Handle near-zero variance: allow if not all values are constant
                flat_fraction = np.mean(np.ptp(trial, axis=0) == 0)
                if flat_fraction > 0.5:
                    logging.warning(f"{hand_label} trial {idx} has >50% zero-variance channels. Skipping.")
                    continue
                elif flat_fraction > 0:
                    logging.info(f"{hand_label} trial {idx} has {flat_fraction:.1%} zero-variance channels, but keeping trial.")
                cleaned.append(trial)
            return cleaned

        left_trials_clean = _clean_trials(left_trials, "Left")
        right_trials_clean = _clean_trials(right_trials, "Right")

        if len(left_trials_clean) < 5 or len(right_trials_clean) < 5:
            logging.error(f"Not enough valid trials after cleaning (left: {len(left_trials_clean)}, right: {len(right_trials_clean)}). Aborting CSP training.")
            return

        left_trials = left_trials_clean
        right_trials = right_trials_clean

        """
        Train Common Spatial Pattern filters for left vs right hand motor imagery.
        
        Args:
            left_trials: List of EEG data arrays for left hand imagery
            right_trials: List of EEG data arrays for right hand imagery
            n_components: Number of CSP components to keep
        """
        try:
            # Ensure we have data for both classes
            if not left_trials or not right_trials:
                logging.error("Not enough trials for CSP training")
                return
            
            # Preprocess each trial
            processed_left = []
            for trial in left_trials:
                # Apply filters
                filtered = self._apply_filters(trial)
                processed_left.append(filtered)
            
            processed_right = []
            for trial in right_trials:
                # Apply filters
                filtered = self._apply_filters(trial)
                processed_right.append(filtered)
            
            # Compute covariance matrices
            cov_left = self._compute_covariance_matrices(processed_left)
            cov_right = self._compute_covariance_matrices(processed_right)
            
            # Train CSP
            self.csp_filters, self.csp_patterns = self._train_csp_from_covariance(cov_left, cov_right, n_components)
            
            # Get normalization parameters from training data
            all_features = []
            for trial in processed_left + processed_right:
                features = self._extract_csp_features(trial, normalize=True)
                if features is not None:
                    all_features.append(features)
            
            all_features = np.vstack(all_features)
            self.csp_mean = np.mean(all_features, axis=0)
            self.csp_std = np.std(all_features, axis=0)
            
            logging.info(f"CSP filters trained with {len(left_trials)} left and {len(right_trials)} right trials")
            
        except Exception as e:
            logging.exception("Error during CSP training:")
    
    def _compute_covariance_matrices(self, trials: List[np.ndarray]) -> np.ndarray:
        """
        Compute covariance matrices for a list of trials.
        
        Args:
            trials: List of EEG data arrays
            
        Returns:
            Average covariance matrix
        """
        n_channels = trials[0].shape[1]
        covs = np.zeros((len(trials), n_channels, n_channels))
        
        for i, trial in enumerate(trials):
            # Normalize
            trial = trial - np.mean(trial, axis=0)
            # Compute covariance
            cov = np.dot(trial.T, trial) / (trial.shape[0] - 1)
            covs[i] = cov
        
        # Average covariance
        avg_cov = np.mean(covs, axis=0)
        # Make it symmetric
        avg_cov = (avg_cov + avg_cov.T) / 2
        
        return avg_cov
    
    def _train_csp_from_covariance(self, cov_a: np.ndarray, cov_b: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        # Validate covariance matrices
        if np.any(np.isnan(cov_a)) or np.any(np.isnan(cov_b)) or np.any(np.isinf(cov_a)) or np.any(np.isinf(cov_b)):
            logging.error("Covariance matrices contain NaN or Inf. Aborting CSP training.")
            raise ValueError("Covariance matrices contain NaN or Inf.")
        if np.all(cov_a == cov_a.flat[0]) or np.all(cov_b == cov_b.flat[0]):
            logging.error("Covariance matrix has zero variance. Aborting CSP training.")
            raise ValueError("Covariance matrix has zero variance.")

        """
        Train CSP filters from covariance matrices.
        
        Args:
            cov_a: Covariance matrix for class A
            cov_b: Covariance matrix for class B
            n_components: Number of components to keep
            
        Returns:
            Tuple of (filters, patterns)
        """
        # Solve generalized eigenvalue problem
        evals, evecs = linalg.eigh(cov_a, cov_a + cov_b)
        
        # Sort eigenvectors by eigenvalues in descending order
        indices = np.argsort(evals)[::-1]
        evecs = evecs[:, indices]
        
        # Select most discriminative components
        n_keep = min(n_components, len(evals))
        
        # Keep first and last n_components/2 eigenvectors
        if n_keep < 2:
            filters = evecs[:, :n_keep]
        else:
            half = n_keep // 2
            filters = np.hstack((evecs[:, :half], evecs[:, -half:]))
        
        # Compute patterns (inverse of filters)
        patterns = linalg.pinv(filters).T
        
        return filters, patterns
    
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
