import threading
import csv
import logging
import os
import time
import sys
from datetime import datetime
from collections import deque
import numpy as np
from typing import Tuple, List, Optional, Dict

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class EEGAcquisition:
    """
    EEG data acquisition module that connects to LSL EEG stream and provides real-time access
    to EEG data with quality monitoring and buffer management.
    """
    def __init__(self, config_dict: Dict):
        """
        Initialize the EEG acquisition module.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.config = config_dict
        self.sample_rate = config_dict.get('SAMPLE_RATE', 250)
        self.eeg_stream_name = config_dict.get('EEG_STREAM_NAME', 'OpenBCI_EEG')
        self.buffer_size = config_dict.get('BUFFER_SIZE', 2500)  # 10 seconds at 250Hz
        self.connection_timeout = config_dict.get('CONNECTION_TIMEOUT', 10)
        self.channels = config_dict.get('EEG_CHANNELS', ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
        
        # Channels of interest for motor imagery (C3, CP1, C4, CP2)
        self.mi_channel_indices = config_dict.get('MI_CHANNEL_INDICES', [2, 3, 5, 6])  # Adjust based on montage
        
        # Data buffer (numpy for efficient processing)
        self.buffer = np.zeros((self.buffer_size, len(self.channels)))
        self.timestamps = np.zeros(self.buffer_size)
        self.buffer_head = 0
        self.buffer_filled = False
        
        # Signal quality metrics
        self.signal_quality = {ch: 1.0 for ch in self.channels}  # 1.0 = perfect, 0.0 = unusable
        self.quality_threshold = config_dict.get('QUALITY_THRESHOLD', 0.6)
        
        # Connection status
        self.connected = False
        self.inlet = None
        self.clock_offset = 0.0
        
        # Counter for samples since last reset
        self.samples_count = 0
        
        # Thread for background update
        self._update_thread = None
        self._stop_event = threading.Event()
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure data directory exists before creating files
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        self.output_csv = os.path.join(config.RAW_DATA_DIR, f"MI_EEG_{timestamp_str}.csv")
        self.eeg_inlet = None
        self.marker_inlet = None
        self.clock_offset = 0.0
        
        # Create a separate buffer for high-priority neurofeedback processing
        self.neurofeedback_queue = deque(maxlen=5)  # Small buffer to prevent overflow

    def connect(self) -> bool:
        """Connect to the LSL EEG stream."""
        try:
            from pylsl import resolve_byprop, StreamInlet
            logging.info(f"Resolving EEG LSL stream with name '{self.eeg_stream_name}'...")
            eeg_streams = resolve_byprop("name", self.eeg_stream_name, timeout=self.connection_timeout)
            
            if not eeg_streams:
                logging.error(f"No EEG stream found with name '{self.eeg_stream_name}'.")
                return False
                
            self.inlet = StreamInlet(eeg_streams[0], max_buflen=self.buffer_size)
            self.clock_offset = self.inlet.time_correction()
            logging.info(f"Connected to '{self.eeg_stream_name}' with clock offset: {self.clock_offset:.6f} seconds")
            self.connected = True
            return True
            
        except Exception as e:
            logging.exception("Exception during EEG stream connection:")
            self.connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the LSL stream."""
        if self.inlet:
            del self.inlet
            self.inlet = None
        self.connected = False
        logging.info("Disconnected from EEG stream")
    
    def start_background_update(self) -> None:
        """Start background thread for continuous data update."""
        if self._update_thread is not None and self._update_thread.is_alive():
            logging.warning("Background update already running")
            return
            
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._background_update)
        self._update_thread.daemon = True
        self._update_thread.start()
        logging.info("Started background EEG update thread")
    
    def stop_background_update(self) -> None:
        """Stop the background update thread."""
        if self._update_thread is not None:
            self._stop_event.set()
            self._update_thread.join(timeout=1.0)
            self._update_thread = None
            logging.info("Stopped background EEG update thread")
    
    def _background_update(self) -> None:
        """Background thread function to continuously update the buffer."""
        while not self._stop_event.is_set():
            self.update_buffer()
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
    
    def update_buffer(self) -> int:
        """
        Update the buffer with new samples from the LSL stream.
        
        Returns:
            int: Number of new samples added to the buffer
        """
        if not self.connected or not self.inlet:
            return 0
            
        try:
            # Try to get chunk of samples
            samples, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=32)
            
            if not samples:
                return 0
                
            # Add samples to buffer
            num_samples = len(samples)
            for i, (sample, ts) in enumerate(zip(samples, timestamps)):
                # Adjust timestamp with clock offset
                adj_ts = ts + self.clock_offset
                
                # Store in circular buffer
                idx = (self.buffer_head + i) % self.buffer_size
                self.buffer[idx] = sample
                self.timestamps[idx] = adj_ts
            
            # Update buffer head
            self.buffer_head = (self.buffer_head + num_samples) % self.buffer_size
            self.samples_count += num_samples
            
            # Mark buffer as filled once we've collected enough samples
            if self.samples_count >= self.buffer_size:
                self.buffer_filled = True
            
            # Update signal quality (basic implementation - could be more sophisticated)
            self._update_signal_quality(samples)
            
            return num_samples
            
        except Exception as e:
            logging.exception("Error updating EEG buffer:")
            return 0
    
    def _update_signal_quality(self, samples: List[List[float]]) -> None:
        """
        Update signal quality metrics based on recent samples.
        
        This is a basic implementation checking for flatlines and extreme values.
        More sophisticated metrics can be implemented.
        
        Args:
            samples: List of EEG samples
        """
        if not samples:
            return
            
        # Convert to numpy array for easier processing
        samples_array = np.array(samples)
        
        for i, ch in enumerate(self.channels):
            # Check for flatlines
            if len(samples) > 1:
                diff = np.diff(samples_array[:, i])
                flatline_ratio = np.sum(np.abs(diff) < 0.01) / len(diff)
                
                # Check for extreme values
                extreme_vals = np.sum(np.abs(samples_array[:, i]) > 100) / len(samples)
                
                # Update quality (simple heuristic)
                quality = 1.0 - (flatline_ratio * 0.5 + extreme_vals * 0.5)
                
                # Smooth the quality metric
                self.signal_quality[ch] = 0.9 * self.signal_quality[ch] + 0.1 * quality
    
    def get_chunk(self, window_size: int = 500, channels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a chunk of the most recent EEG data.
        
        Args:
            window_size: Number of samples to return
            channels: Indices of channels to return (default: all)
            
        Returns:
            Tuple of (data_array, timestamps_array)
        """
        if not self.connected:
            return np.array([]), np.array([])
            
        # Ensure window size doesn't exceed buffer
        window_size = min(window_size, self.buffer_size)
        
        # Calculate indices for the requested window
        if self.buffer_filled:
            # If buffer is full, start from current head minus window size
            start_idx = (self.buffer_head - window_size) % self.buffer_size
            indices = [(start_idx + i) % self.buffer_size for i in range(window_size)]
        else:
            # If buffer not full yet, return what we have from the beginning
            indices = list(range(min(self.samples_count, window_size)))
        
        # Extract data and timestamps
        data = self.buffer[indices]
        timestamps = self.timestamps[indices]
        
        # Filter channels if requested
        if channels is not None:
            data = data[:, channels]
            
        return data, timestamps
    
    def get_motor_imagery_data(self, window_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get motor imagery channels data.
        
        Args:
            window_size: Number of samples to return
            
        Returns:
            Tuple of (data_array, timestamps_array) for MI channels
        """
        return self.get_chunk(window_size, self.mi_channel_indices)
    
    def check_signal_quality(self) -> Dict[str, float]:
        """
        Check the signal quality of all channels.
        
        Returns:
            Dictionary of channel names to quality values (0.0-1.0)
        """
        return self.signal_quality
    
    def is_signal_good(self) -> bool:
        """
        Check if overall signal quality is acceptable.
        
        Returns:
            Boolean indicating if signal quality is above threshold
        """
        return all(q >= self.quality_threshold for q in self.signal_quality.values())
    
    def reset_buffer(self) -> None:
        """Reset the EEG data buffer."""
        self.buffer = np.zeros((self.buffer_size, len(self.channels)))
        self.timestamps = np.zeros(self.buffer_size)
        self.buffer_head = 0
        self.buffer_filled = False
        self.samples_count = 0
        logging.info("EEG buffer reset")

    def resolve_streams(self):
        try:
            logging.info("Resolving EEG LSL stream...")
            from pylsl import resolve_byprop, StreamInlet
            eeg_streams = resolve_byprop("name", config.EEG_STREAM_NAME, timeout=10)
            if not eeg_streams:
                logging.error(f"No EEG stream found with name '{config.EEG_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.eeg_inlet = StreamInlet(eeg_streams[0], max_buflen=360)
            self.clock_offset = self.eeg_inlet.time_correction()
            logging.info(f"Computed clock offset: {self.clock_offset:.6f} seconds")

            logging.info("Resolving Marker LSL stream...")
            marker_streams = resolve_byprop("name", config.MARKER_STREAM_NAME, timeout=10)
            if not marker_streams:
                logging.error(f"No Marker stream found with name '{config.MARKER_STREAM_NAME}'. Exiting.")
                sys.exit(1)
            self.marker_inlet = StreamInlet(marker_streams[0], max_buflen=360)

            logging.info("LSL streams resolved. Starting data collection...")
        except Exception as e:
            logging.exception("Exception during stream resolution:")
            sys.exit(1)

    def flush_remaining(self, writer):
        """Flush any remaining buffered data to the CSV."""
        while self.eeg_buffer or self.marker_buffer:
            if self.eeg_buffer and self.marker_buffer:
                ts_eeg, eeg_data = self.eeg_buffer[0]
                ts_marker, marker = self.marker_buffer[0]
                if abs(ts_marker - ts_eeg) < config.MERGE_THRESHOLD:
                    row = [ts_eeg] + eeg_data + [marker]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
                    self.marker_buffer.popleft()
                elif ts_marker < ts_eeg:
                    row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                    writer.writerow(row)
                    self.marker_buffer.popleft()
                else:
                    row = [ts_eeg] + eeg_data + [""]
                    writer.writerow(row)
                    self.eeg_buffer.popleft()
            elif self.eeg_buffer:
                ts_eeg, eeg_data = self.eeg_buffer.popleft()
                row = [ts_eeg] + eeg_data + [""]
                writer.writerow(row)
            elif self.marker_buffer:
                ts_marker, marker = self.marker_buffer.popleft()
                row = [ts_marker] + ([""] * len(config.EEG_CHANNELS)) + [marker]
                writer.writerow(row)

    def run(self):
        from pylsl import resolve_byprop, StreamInlet
        self.resolve_streams()
        try:
            with open(self.output_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = ["lsl_timestamp"] + config.EEG_CHANNELS + ["marker"]
                writer.writerow(header)
                while not self._stop_event.is_set():
                    try:
                        # Use chunk processing for efficiency when available
                        samples, timestamps = self.eeg_inlet.pull_chunk(timeout=0.0, max_samples=32)
                        if samples and timestamps:
                            for sample, ts in zip(samples, timestamps):
                                # Process for neurofeedback immediately
                                self.neurofeedback_processor.process_sample(sample)
                                
                                # Buffer for data saving
                                self.eeg_buffer.append((ts, sample))
                    except Exception as e:
                        # If chunk not available, fallback to single sample
                        try:
                            sample_eeg, ts_eeg = self.eeg_inlet.pull_sample(timeout=0.0)
                            if sample_eeg is not None and ts_eeg is not None:
                                self.eeg_buffer.append((ts_eeg, sample_eeg))
                                # Process sample for neurofeedback
                                self.neurofeedback_processor.process_sample(sample_eeg)
                        except Exception as inner_e:
                            logging.exception("Error pulling EEG sample:")
                    
                    try:
                        sample_marker, ts_marker = self.marker_inlet.pull_sample(timeout=0.0)
                        if sample_marker is not None and ts_marker is not None:
                            adjusted_ts_marker = ts_marker - self.clock_offset
                            marker_val = sample_marker[0]
                            self.marker_buffer.append((adjusted_ts_marker, marker_val))
                    except Exception as e:
                        pass

                    # Merge data from both buffers (file saving is lower priority than processing)
                    while self.eeg_buffer and self.marker_buffer:
                        ts_eeg, eeg_data = self.eeg_buffer[0]
                        ts_marker, marker = self.marker_buffer[0]
                        if abs(ts_marker - ts_eeg) < config.MERGE_THRESHOLD:
                            row = [ts_eeg] + eeg_data + [marker]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                            self.marker_buffer.popleft()
                        elif ts_marker < ts_eeg:
                            row = [ts_marker] + ([""] * len(eeg_data)) + [marker]
                            writer.writerow(row)
                            self.marker_buffer.popleft()
                        else:
                            row = [ts_eeg] + eeg_data + [""]
                            writer.writerow(row)
                            self.eeg_buffer.popleft()
                    
                    # Flush buffer periodically to avoid excessive memory use
                    if len(self.eeg_buffer) > 1000:
                        while len(self.eeg_buffer) > 500:
                            ts_eeg, eeg_data = self.eeg_buffer.popleft()
                            row = [ts_eeg] + eeg_data + [""]
                            writer.writerow(row)
                            
                    time.sleep(config.POLL_SLEEP)
                logging.info("Stop event set. Flushing remaining data...")
                self.flush_remaining(writer)
        except Exception as e:
            logging.exception("Exception in data collector run loop:")
        finally:
            logging.info(f"Data collection stopped. Data saved to {self.output_csv}")