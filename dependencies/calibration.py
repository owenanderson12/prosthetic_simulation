import numpy as np
import logging
import time
import threading
import os
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
from enum import Enum
from pylsl import StreamInfo, StreamOutlet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pickle

# Add parent directory to path to import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class CalibrationStage(Enum):
    """Enumeration for calibration stages."""
    REST = 0
    LEFT = 1
    RIGHT = 2
    COMPLETE = 3
    IDLE = 4

class CalibrationSystem:
    """
    Calibration system for motor imagery BCI.
    
    Handles:
    - Guided calibration procedure
    - Collection of baseline/resting state
    - Collection of left and right motor imagery data
    - Quick classifier training
    - Parameter optimization
    - Saving/loading calibration profiles
    """
    
    def __init__(self, config: Dict, eeg_acquisition=None, signal_processor=None, classifier=None):
        """
        Initialize calibration system.
        
        Args:
            config: Dictionary with configuration parameters
            eeg_acquisition: EEG acquisition module instance (optional)
            signal_processor: Signal processor module instance (optional)
            classifier: Classifier module instance (optional)
        """
        self.config = config
        self.eeg_acquisition = eeg_acquisition
        self.signal_processor = signal_processor
        self.classifier = classifier
        
        # Calibration parameters
        self.num_trials = config.get('CALIBRATION_TRIALS', 10)
        self.trial_duration = config.get('TRIAL_DURATION', 5.0)  # seconds
        self.rest_duration = config.get('REST_DURATION', 3.0)    # seconds
        self.baseline_duration = config.get('BASELINE_DURATION', 30.0)  # seconds
        self.cue_duration = config.get('CUE_DURATION', 2.0)      # seconds
        
        # Storage for collected data
        self.baseline_data = []
        self.left_data = []
        self.right_data = []
        self.left_features = []
        self.right_features = []
        
        # Calibration state
        self.current_stage = CalibrationStage.IDLE
        self.trial_count = {
            CalibrationStage.LEFT: 0,
            CalibrationStage.RIGHT: 0
        }
        self.start_time = 0.0
        self.stage_start_time = 0.0
        self.timestamp_offset = 0.0
        
        # Calibration threading
        self._calibration_thread = None
        self._stop_event = threading.Event()
        
        # Markers outlet for triggering events
        self.marker_outlet = None
        self._setup_marker_stream()
        
        # Callback for UI updates
        self.update_callback = None
        
        # Path for saving calibration data
        self.calibration_dir = config.get('CALIBRATION_DIR', 'calibration')
        os.makedirs(self.calibration_dir, exist_ok=True)
        
        logging.info("Calibration system initialized")
    
    def _setup_marker_stream(self) -> None:
        """Set up LSL marker stream for sending calibration event markers."""
        try:
            marker_stream_name = self.config.get('MARKER_STREAM_NAME', 'MI_MarkerStream')
            # Create LSL outlet for markers
            info = StreamInfo(
                name=marker_stream_name,
                type='Markers',
                channel_count=1,
                nominal_srate=0,  # Irregular sampling rate
                channel_format='string',
                source_id='mi_calibration'
            )
            self.marker_outlet = StreamOutlet(info)
            logging.info(f"Created LSL marker stream: {marker_stream_name}")
        except Exception as e:
            logging.exception("Failed to create marker stream:")
            self.marker_outlet = None
    
    def start_calibration(self, update_callback: Optional[Callable] = None) -> bool:
        """
        Start the calibration procedure.
        
        Args:
            update_callback: Function to call for UI updates
            
        Returns:
            Success indicator
        """
        if self._calibration_thread is not None and self._calibration_thread.is_alive():
            logging.warning("Calibration already running")
            return False
            
        if self.eeg_acquisition is None or self.signal_processor is None or self.classifier is None:
            logging.error("Cannot start calibration: Missing required modules")
            return False
            
        # Reset state
        self._stop_event.clear()
        self.current_stage = CalibrationStage.IDLE
        self.trial_count = {
            CalibrationStage.LEFT: 0,
            CalibrationStage.RIGHT: 0
        }
        self.update_callback = update_callback
        
        # Clear previous data
        self.baseline_data = []
        self.left_data = []
        self.right_data = []
        self.left_features = []
        self.right_features = []
        
        # Start calibration in a separate thread
        self._calibration_thread = threading.Thread(target=self._run_calibration)
        self._calibration_thread.daemon = True
        self._calibration_thread.start()
        
        logging.info("Calibration started")
        return True
    
    def stop_calibration(self) -> None:
        """Stop the ongoing calibration procedure."""
        if self._calibration_thread is not None and self._calibration_thread.is_alive():
            self._stop_event.set()
            self._calibration_thread.join(timeout=2.0)
            logging.info("Calibration stopped")
    
    def _run_calibration(self) -> None:
        """Main calibration procedure running in a separate thread."""
        try:
            logging.info("Starting calibration thread. Calibration parameters: num_trials=%d, trial_duration=%.2f, rest_duration=%.2f", self.num_trials, self.trial_duration, self.rest_duration)
            
            # 1. Make sure we have a good connection
            if not self.eeg_acquisition.is_signal_good():
                logging.error("Signal quality too poor for calibration")
                self._update_ui("Signal quality too poor. Check electrodes and try again.")
                return
                
            self.start_time = time.time()
            logging.info(f"Calibration global start_time: {self.start_time}")
            self.timestamp_offset = 0.0
            if hasattr(self.eeg_acquisition, 'inlet') and self.eeg_acquisition.inlet:
                self.timestamp_offset = self.eeg_acquisition.clock_offset
            logging.info(f"EEG clock offset: {self.timestamp_offset}")
                
            # 2. Collect baseline data (resting state)
            self._update_ui("Preparing for baseline collection. Please relax.")
            time.sleep(3.0)  # Give time to read instructions
            
            logging.info("Starting baseline collection.")
            self._collect_baseline()
            logging.info("Baseline collection complete.")
            if self._stop_event.is_set():
                logging.info("Calibration aborted after baseline collection.")
                return
                
            # 3. Train the signal processor with baseline data
            self._update_ui("Processing baseline data...")
            self._process_baseline()
            logging.info("Baseline processed. Entering calibration trials.")
        
            # 4. Run calibration trials alternating between left and right
            trials_complete = 0
            while (trials_complete < 2 * self.num_trials) and not self._stop_event.is_set():
                # Alternate between left and right
                if trials_complete % 2 == 0:
                    stage = CalibrationStage.LEFT
                    cue = "LEFT HAND"
                else:
                    stage = CalibrationStage.RIGHT
                    cue = "RIGHT HAND"
                    
                # Rest between trials
                self.current_stage = CalibrationStage.REST
                self._update_ui(f"Rest. Next: {cue}")
                self._wait(self.rest_duration)
                if self._stop_event.is_set():
                    break
                    
                # Run trial
                self.current_stage = stage
                self.stage_start_time = time.time()
                self.trial_count[stage] += 1
                logging.info(f"Starting trial: stage={stage.name}, trial_number={self.trial_count[stage]}, stage_start_time={self.stage_start_time}")
            
                # Send marker
                if self.marker_outlet:
                    marker = "2" if stage == CalibrationStage.LEFT else "1"  # 2=left, 1=right
                    self.marker_outlet.push_sample([marker])
                    logging.info(f"Sent marker {marker} for stage {stage.name}")
            
                # Show cue
                self._update_ui(f"Imagine {cue} movement")
            
                # Collect data for this trial
                self._collect_trial_data(stage)
                logging.info(f"Completed trial: stage={stage.name}, trial_number={self.trial_count[stage]}")
                if self._stop_event.is_set():
                    logging.info("Calibration aborted during trials loop.")
                    break
                    
                # End of trial marker
                if self.marker_outlet:
                    self.marker_outlet.push_sample(["3"])  # 3=end of trial
                    
                trials_complete += 1
                
                # Give feedback on progress
                progress = int((trials_complete / (2 * self.num_trials)) * 100)
                self._update_ui(f"Progress: {progress}%")
                logging.info(f"Trials complete: {trials_complete}/{2 * self.num_trials}, Progress: {progress}%")
        
            # 5. Process collected data and train classifier
            if not self._stop_event.is_set():
                self.current_stage = CalibrationStage.COMPLETE
                self._update_ui("Processing calibration data...")
            
                success = self._train_classifier()
            
                if success:
                    self._update_ui("Calibration complete! Classifier trained successfully.")
                    # Save calibration data
                    self._save_calibration_data()
                else:
                    self._update_ui("Calibration complete, but classifier training failed. Try again.")
                    
            logging.info("Calibration procedure finished")
        except Exception as e:
            logging.exception("Error during calibration (exception details logged):")
            self._update_ui(f"Error during calibration: {str(e)}")
    
    def _wait(self, duration: float) -> None:
        """
        Wait for specified duration with abort possibility.
        
        Args:
            duration: Time to wait in seconds
        """
        end_time = time.time() + duration
        while time.time() < end_time:
            if self._stop_event.is_set():
                break
            time.sleep(0.1)
    
    def _update_ui(self, message: str) -> None:
        """
        Update UI with calibration status.
        
        Args:
            message: Message to display
        """
        if self.update_callback:
            try:
                self.update_callback({
                    'stage': self.current_stage,
                    'message': message,
                    'trial_count': dict(self.trial_count),
                    'time_elapsed': time.time() - self.start_time if self.start_time > 0 else 0
                })
            except Exception as e:
                logging.exception("Error in UI callback:")
    
    def _collect_baseline(self) -> None:
        """Collect baseline (resting state) data."""
        self.current_stage = CalibrationStage.REST
        self._update_ui("Collecting baseline. Please relax and stay still...")
        self.stage_start_time = time.time()
        
        # Send marker
        if self.marker_outlet:
            self.marker_outlet.push_sample(["0"])  # 0=baseline/rest
            
        # Collect data for baseline_duration seconds
        end_time = time.time() + self.baseline_duration
        baseline_chunks = []
        
        while time.time() < end_time and not self._stop_event.is_set():
            # Get latest EEG data
            data, timestamps = self.eeg_acquisition.get_chunk(window_size=int(0.5 * self.eeg_acquisition.sample_rate))
            
            if data.size > 0:
                baseline_chunks.append(data)
                
            # Brief sleep to prevent CPU overload
            time.sleep(0.1)
            
            # Update progress
            progress = int(((time.time() - self.stage_start_time) / self.baseline_duration) * 100)
            self._update_ui(f"Collecting baseline: {progress}%")
            
        # Combine all data
        if baseline_chunks:
            self.baseline_data = np.vstack(baseline_chunks)
            logging.info(f"Collected baseline data: {self.baseline_data.shape[0]} samples")
        else:
            logging.warning("No baseline data collected")
            
        # End of baseline marker
        if self.marker_outlet:
            self.marker_outlet.push_sample(["3"])  # 3=end of trial/period
    
    def _process_baseline(self) -> None:
        """Process the collected baseline data to update signal processor."""
        if self.baseline_data.size == 0:
            logging.warning("No baseline data to process")
            return
            
        # Update signal processor with baseline data
        self.signal_processor.update_baseline(self.baseline_data)
        logging.info("Baseline processed and applied to signal processor")
    
    def _collect_trial_data(self, stage: CalibrationStage) -> None:
        """
        Collect data for a single calibration trial.
        
        Args:
            stage: The current calibration stage (LEFT or RIGHT)
        """
        start_time = time.time()
        end_time = start_time + self.trial_duration
        logging.info(f"[TrialData] Trial start: stage={stage.name}, start_time={start_time}, expected_end_time={end_time}, duration={self.trial_duration}")
        trial_chunks = []
        
        while time.time() < end_time and not self._stop_event.is_set():
            # Get latest EEG data
            try:
                data, timestamps = self.eeg_acquisition.get_chunk(window_size=int(0.25 * self.eeg_acquisition.sample_rate))
            except Exception as e:
                logging.exception(f"[TrialData] Exception during EEG data acquisition: {e}")
                break
        
            if data.size > 0:
                trial_chunks.append(data)
                logging.debug(f"[TrialData] Got data chunk: shape={data.shape}, stage={stage.name}")
                
                # Process this chunk for features
                try:
                    result = self.signal_processor.process(data, timestamps)
                except Exception as e:
                    logging.exception(f"[TrialData] Exception during signal processing: {e}")
                    continue
                if result['valid'] and result['features'] is not None:
                    # Store features for classifier training
                    feature_vector = result['features'].get('csp_features')
                    if feature_vector is None:
                        # Fallback to basic features if CSP not available
                        mu_erd = result['features'].get('erd_mu', [0])
                        beta_erd = result['features'].get('erd_beta', [0])
                        feature_vector = np.hstack([mu_erd, beta_erd])
                    
                    if stage == CalibrationStage.LEFT:
                        self.left_features.append(feature_vector)
                    else:
                        self.right_features.append(feature_vector)
            
            # Brief sleep to prevent CPU overload
            time.sleep(0.1)
            
            # Update progress
            elapsed = time.time() - start_time
            progress = int((elapsed / self.trial_duration) * 100)
            logging.debug(f"[TrialData] Progress update: stage={stage.name}, elapsed={elapsed:.2f}, progress={progress}%, current_time={time.time()}, trial_count={self.trial_count[stage]}")
            
            if stage == CalibrationStage.LEFT:
                self._update_ui(f"Left hand trial {self.trial_count[stage]}/{self.num_trials}: {progress}%")
            else:
                self._update_ui(f"Right hand trial {self.trial_count[stage]}/{self.num_trials}: {progress}%")
        
        # Store the collected data
        if trial_chunks:
            trial_data = np.vstack(trial_chunks)
            logging.info(f"[TrialData] Trial end: stage={stage.name}, trial_number={self.trial_count[stage]}, actual_samples={trial_data.shape[0]}, expected_duration={self.trial_duration}, actual_duration={time.time() - start_time:.2f}")
            if stage == CalibrationStage.LEFT:
                self.left_data.append(trial_data)
                logging.info(f"Collected left trial {self.trial_count[stage]}: {trial_data.shape[0]} samples")
            else:
                self.right_data.append(trial_data)
                logging.info(f"Collected right trial {self.trial_count[stage]}: {trial_data.shape[0]} samples")
        else:
            logging.warning(f"[TrialData] No data collected for trial: stage={stage.name}, trial_number={self.trial_count[stage]}")
    
    def _train_classifier(self) -> bool:
        """
        Train the Random Forest classifier using collected calibration data.
        
        Returns:
            Success indicator
        """
        try:
            # Get window parameters
            window_size = int(self.config.get('WINDOW_SIZE', 2.0) * self.eeg_acquisition.sample_rate)
            window_overlap = int(self.config.get('WINDOW_OVERLAP', 0.5) * self.eeg_acquisition.sample_rate)
            step_size = window_size - window_overlap
            
            # 1. First train CSP filters if we have enough data
            if (len(self.left_data) >= 3 and len(self.right_data) >= 3):
                self._update_ui("Training CSP filters...")
                self.signal_processor.train_csp(self.left_data, self.right_data)
                
                # Clear previous features
                self.left_features = []
                self.right_features = []
                
                # Process left trials with proper windowing
                self._update_ui("Extracting features from left trials...")
                for trial_idx, trial in enumerate(self.left_data):
                    # Segment trial into overlapping windows
                    for start in range(0, len(trial) - window_size + 1, step_size):
                        end = start + window_size
                        window = trial[start:end]
                        
                        # Create timestamps for window
                        timestamps = np.arange(window.shape[0]) / self.eeg_acquisition.sample_rate
                        
                        # Process window
                        result = self.signal_processor.process(window, timestamps)
                        if result['valid'] and result['features'] is not None:
                            # Prefer CSP features if available
                            csp_features = result['features'].get('csp_features')
                            if csp_features is not None:
                                # Combine CSP + band power features for Random Forest
                                mu_erd = result['features'].get('erd_mu', [])
                                beta_erd = result['features'].get('erd_beta', [])
                                if len(mu_erd) > 0 and len(beta_erd) > 0:
                                    feature_vector = np.hstack([csp_features, mu_erd, beta_erd])
                                else:
                                    feature_vector = csp_features
                                self.left_features.append(feature_vector)
                            else:
                                # Fallback to band power features
                                mu_erd = result['features'].get('erd_mu', [])
                                beta_erd = result['features'].get('erd_beta', [])
                                if len(mu_erd) > 0 and len(beta_erd) > 0:
                                    feature_vector = np.hstack([mu_erd, beta_erd])
                                    self.left_features.append(feature_vector)
                
                # Process right trials with proper windowing
                self._update_ui("Extracting features from right trials...")
                for trial_idx, trial in enumerate(self.right_data):
                    # Segment trial into overlapping windows
                    for start in range(0, len(trial) - window_size + 1, step_size):
                        end = start + window_size
                        window = trial[start:end]
                        
                        # Create timestamps for window
                        timestamps = np.arange(window.shape[0]) / self.eeg_acquisition.sample_rate
                        
                        # Process window
                        result = self.signal_processor.process(window, timestamps)
                        if result['valid'] and result['features'] is not None:
                            # Prefer CSP features if available
                            csp_features = result['features'].get('csp_features')
                            if csp_features is not None:
                                # Combine CSP + band power features for Random Forest
                                mu_erd = result['features'].get('erd_mu', [])
                                beta_erd = result['features'].get('erd_beta', [])
                                if len(mu_erd) > 0 and len(beta_erd) > 0:
                                    feature_vector = np.hstack([csp_features, mu_erd, beta_erd])
                                else:
                                    feature_vector = csp_features
                                self.right_features.append(feature_vector)
                            else:
                                # Fallback to band power features
                                mu_erd = result['features'].get('erd_mu', [])
                                beta_erd = result['features'].get('erd_beta', [])
                                if len(mu_erd) > 0 and len(beta_erd) > 0:
                                    feature_vector = np.hstack([mu_erd, beta_erd])
                                    self.right_features.append(feature_vector)
                                    
                logging.info(f"Extracted {len(self.left_features)} left features and {len(self.right_features)} right features")
                
            # 2. If we still don't have enough features, try without CSP
            if len(self.left_features) < 5 or len(self.right_features) < 5:
                logging.warning("Insufficient CSP features, extracting band power features...")
                self._update_ui("Extracting band power features...")
                
                # Clear and re-extract without CSP
                self.left_features = []
                self.right_features = []
                
                # Reset CSP filters to ensure we don't use them
                self.signal_processor.csp_filters = None
                
                # Extract features from collected trial chunks
                for trial in self.left_data:
                    for start in range(0, len(trial) - window_size + 1, step_size):
                        end = start + window_size
                        window = trial[start:end]
                        timestamps = np.arange(window.shape[0]) / self.eeg_acquisition.sample_rate
                        result = self.signal_processor.process(window, timestamps)
                        if result['valid'] and result['features'] is not None:
                            mu_erd = result['features'].get('erd_mu', [])
                            beta_erd = result['features'].get('erd_beta', [])
                            if len(mu_erd) > 0 and len(beta_erd) > 0:
                                feature_vector = np.hstack([mu_erd, beta_erd])
                                self.left_features.append(feature_vector)
                
                for trial in self.right_data:
                    for start in range(0, len(trial) - window_size + 1, step_size):
                        end = start + window_size
                        window = trial[start:end]
                        timestamps = np.arange(window.shape[0]) / self.eeg_acquisition.sample_rate
                        result = self.signal_processor.process(window, timestamps)
                        if result['valid'] and result['features'] is not None:
                            mu_erd = result['features'].get('erd_mu', [])
                            beta_erd = result['features'].get('erd_beta', [])
                            if len(mu_erd) > 0 and len(beta_erd) > 0:
                                feature_vector = np.hstack([mu_erd, beta_erd])
                                self.right_features.append(feature_vector)
            
            # 3. Train Random Forest classifier
            if not self.left_features or not self.right_features:
                logging.error("Not enough feature data to train classifier")
                return False
                
            self._update_ui(f"Training Random Forest classifier with {len(self.left_features)} left and {len(self.right_features)} right features...")
            
            # Prepare training data
            X = np.vstack([np.array(self.left_features), np.array(self.right_features)])
            y = np.hstack([
                np.zeros(len(self.left_features)),  # 0 = left
                np.ones(len(self.right_features))   # 1 = right
            ])
            
            # Create Random Forest pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            accuracy = cv_scores.mean()
            
            # Train on all data
            pipeline.fit(X, y)
            
            # Update classifier with trained pipeline
            self.classifier.classifier = pipeline
            self.classifier.model_type = 'pipeline'
            self.classifier.classes = [0., 1.]
            self.classifier.class_map = {0: 'left', 1: 'right'}
            
            logging.info(f"Random Forest classifier trained with accuracy: {accuracy:.3f}")
            self._update_ui(f"Random Forest classifier trained with cross-validation accuracy: {accuracy:.1%}")
            
            return accuracy > 0.5  # Lower threshold for success
            
        except Exception as e:
            logging.exception("Error during Random Forest classifier training:")
            return False
    
    def _save_calibration_data(self) -> None:
        """Save calibration data for later use in Random Forest subfolder."""
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use random_forest subfolder
            rf_calibration_dir = os.path.join(self.calibration_dir, 'random_forest')
            os.makedirs(rf_calibration_dir, exist_ok=True)
            
            filename = f"rf_calibration_{timestamp_str}.npz"
            filepath = os.path.join(rf_calibration_dir, filename)
            
            # Prepare data to save
            save_data = {
                'baseline_data': self.baseline_data if self.baseline_data.size > 0 else np.array([]),
                'left_data': np.array(self.left_data, dtype=object) if self.left_data else np.array([]),
                'right_data': np.array(self.right_data, dtype=object) if self.right_data else np.array([]),
                'left_features': np.array(self.left_features) if self.left_features else np.array([]),
                'right_features': np.array(self.right_features) if self.right_features else np.array([])
            }
            
            # Add CSP filters if available
            if self.signal_processor.csp_filters is not None:
                save_data['csp_filters'] = self.signal_processor.csp_filters
                save_data['csp_patterns'] = self.signal_processor.csp_patterns
                save_data['csp_mean'] = self.signal_processor.csp_mean
                save_data['csp_std'] = self.signal_processor.csp_std
                logging.info("Saving CSP filters with calibration data")
            
            # Save calibration data
            np.savez(filepath, **save_data)
            
            # Save trained Random Forest classifier
            model_filename = f"rf_model_{timestamp_str}.pkl"
            model_filepath = os.path.join(rf_calibration_dir, model_filename)
            
            model_data = {
                'classifier': self.classifier.classifier,
                'model_type': self.classifier.model_type,
                'classes': self.classifier.classes,
                'class_map': self.classifier.class_map,
                'threshold': self.classifier.threshold,
                'adaptive_threshold': self.classifier.adaptive_threshold
            }
            
            with open(model_filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"Random Forest calibration data saved to {filepath}")
            logging.info(f"Random Forest model saved to {model_filepath}")
            self._update_ui(f"Random Forest calibration data saved to random_forest/ folder")
            
        except Exception as e:
            logging.exception("Error saving Random Forest calibration data:")
    
    def load_calibration(self, filename: str) -> bool:
        """
        Load a saved calibration.
        
        Args:
            filename: Name of the calibration file to load
            
        Returns:
            Success indicator
        """
        try:
            if not filename.endswith('.npz'):
                filename += '.npz'
            
            # Try different possible paths for the calibration file
            possible_paths = [
                os.path.join(self.calibration_dir, filename),  # Direct in calibration dir
                os.path.join(self.calibration_dir, 'random_forest', filename),  # In random_forest subdir
                os.path.join(self.calibration_dir, 'random_forest', f"rf_{filename}"),  # With rf_ prefix
                os.path.join(self.calibration_dir, 'random_forest', f"rf_calibration_{filename}")  # Full rf_calibration_ prefix
            ]
            
            filepath = None
            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath is None:
                logging.error(f"Calibration file not found in any of these locations: {possible_paths}")
                return False
            
            logging.info(f"Found calibration file at: {filepath}")
            
            # Load calibration data
            data = np.load(filepath, allow_pickle=True)
            
            # Extract data
            self.baseline_data = data['baseline_data']
            self.left_data = data['left_data']
            self.right_data = data['right_data']
            self.left_features = data['left_features']
            self.right_features = data['right_features']
            
            # Apply baseline to signal processor
            if self.baseline_data.size > 0 and self.signal_processor is not None:
                self.signal_processor.update_baseline(self.baseline_data)
                logging.info("Baseline power values updated")
                
            # Load CSP filters if available, otherwise retrain them
            if 'csp_filters' in data and self.signal_processor is not None:
                self.signal_processor.csp_filters = data['csp_filters']
                self.signal_processor.csp_patterns = data.get('csp_patterns')
                self.signal_processor.csp_mean = data.get('csp_mean')
                self.signal_processor.csp_std = data.get('csp_std')
                logging.info(f"Loaded CSP filters: shape {self.signal_processor.csp_filters.shape}")
            elif self.signal_processor is not None and len(self.left_data) > 0 and len(self.right_data) > 0:
                logging.info("CSP filters not found in calibration file. Retraining from saved data...")
                # Convert object arrays to lists of numpy arrays
                left_trials = [trial for trial in self.left_data]
                right_trials = [trial for trial in self.right_data]
                # Train CSP filters
                self.signal_processor.train_csp(left_trials, right_trials)
                if self.signal_processor.csp_filters is not None:
                    logging.info(f"Successfully retrained CSP filters: shape {self.signal_processor.csp_filters.shape}")
                else:
                    logging.error("Failed to train CSP filters from saved data")
                    return False
            else:
                logging.error("No data available to train CSP filters")
                return False
                
            # Try to load associated model
            model_filename = None
            model_filepath = None
            
            # Check if this is a Random Forest calibration (in random_forest subfolder)
            if 'random_forest' in filepath:
                # Random Forest model - extract just the base filename
                base_filename = os.path.basename(filepath)
                model_filename = base_filename.replace('rf_calibration_', 'rf_model_').replace('.npz', '.pkl')
                # Get the directory of the calibration file (already includes random_forest)
                calibration_dir = os.path.dirname(filepath)
                model_filepath = os.path.join(calibration_dir, model_filename)
            else:
                # Legacy LDA model
                model_filename = filename.replace('calibration_', 'model_').replace('.npz', '.pkl')
                model_filepath = os.path.join(self.config.get('MODEL_DIR', 'models'), model_filename)
            
            if os.path.exists(model_filepath):
                if self.classifier is not None:
                    # Load the model data directly
                    try:
                        with open(model_filepath, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        # Update classifier with loaded model
                        self.classifier.classifier = model_data['classifier']
                        self.classifier.model_type = model_data.get('model_type', 'lda')
                        self.classifier.classes = model_data.get('classes', [0., 1.])
                        self.classifier.class_map = model_data.get('class_map', {0: 'left', 1: 'right'})
                        self.classifier.threshold = model_data.get('threshold', self.config.get('CLASSIFIER_THRESHOLD', 0.65))
                        self.classifier.adaptive_threshold = model_data.get('adaptive_threshold')
                        
                        logging.info(f"Loaded {model_data.get('model_type', 'LDA')} model from {model_filepath}")
                        
                        # Log model performance if available
                        if 'cv_accuracy' in model_data:
                            logging.info(f"Model CV accuracy: {model_data['cv_accuracy']:.1%}")
                            
                    except Exception as e:
                        logging.error(f"Failed to load model from {model_filepath}: {e}")
                        return False
                else:
                    logging.error("No classifier available to load model into")
                    return False
            else:
                logging.warning(f"No associated model file found: {model_filepath}")
                # For Random Forest calibrations, this is critical
                if 'random_forest' in filepath:
                    logging.error("Random Forest calibration requires associated .pkl model file")
                    return False
            
            logging.info(f"Calibration data loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.exception("Error loading calibration data:")
            return False
    
    def get_calibration_progress(self) -> Dict:
        """
        Get current calibration progress.
        
        Returns:
            Dictionary with progress information
        """
        return {
            'stage': self.current_stage,
            'left_trials': self.trial_count[CalibrationStage.LEFT],
            'right_trials': self.trial_count[CalibrationStage.RIGHT],
            'baseline_samples': len(self.baseline_data),
            'left_features': len(self.left_features),
            'right_features': len(self.right_features),
            'time_elapsed': time.time() - self.start_time if self.start_time > 0 else 0
        }
