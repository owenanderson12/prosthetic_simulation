import numpy as np
import pandas as pd
import logging
import os
import time
import threading
import csv
import sys
from typing import Dict, List, Tuple, Optional, Union
from collections import deque

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class FileHandler:
    """
    File handler for loading and streaming pre-recorded EEG data.
    
    This module simulates real-time EEG data acquisition by:
    - Loading data from various file formats
    - Streaming data at the original recording rate
    - Supporting the same interface as the EEGAcquisition class
    """
    
    def __init__(self, config_dict: Dict, file_path: str):
        """
        Initialize the file handler.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            file_path: Path to the source file
        """
        self.config = config_dict
        self.file_path = file_path
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Data buffer (similar to EEGAcquisition)
        self.sample_rate = config_dict.get('SAMPLE_RATE', 250)
        self.buffer_size = config_dict.get('BUFFER_SIZE', 2500)  # 10 seconds at 250Hz
        self.channels = config_dict.get('EEG_CHANNELS', ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
        self.mi_channel_indices = config_dict.get('MI_CHANNEL_INDICES', [2, 3, 5, 6])
        
        self.buffer = np.zeros((self.buffer_size, len(self.channels)))
        self.timestamps = np.zeros(self.buffer_size)
        self.buffer_head = 0
        self.buffer_filled = False
        
        # Signal quality (always good for file data)
        self.signal_quality = {ch: 1.0 for ch in self.channels}
        
        # Connection status
        self.connected = False
        self.samples_count = 0
        
        # Raw data loaded from file
        self.raw_data = None
        self.markers = None
        self.current_position = 0
        
        # Thread for background update
        self._update_thread = None
        self._stop_event = threading.Event()
        
        # File parsing
        self.file_extension = os.path.splitext(file_path)[1].lower()
        
        logging.info(f"File handler initialized with file: {file_path}")
    
    def connect(self) -> bool:
        """
        Load the data file and prepare for streaming.
        
        Returns:
            Success indicator
        """
        try:
            # Load data based on file extension
            if self.file_extension == '.csv':
                self._load_csv()
            elif self.file_extension == '.npy':
                self._load_npy()
            elif self.file_extension in ['.xdf', '.bdf', '.edf']:
                self._load_mne_format()
            else:
                logging.error(f"Unsupported file format: {self.file_extension}")
                return False
            
            # If data loaded successfully
            if self.raw_data is not None:
                self.connected = True
                logging.info(f"Loaded file with {len(self.raw_data)} samples")
                return True
            else:
                logging.error("Failed to load data from file")
                return False
            
        except Exception as e:
            logging.exception(f"Error loading file {self.file_path}:")
            return False
    
    def _load_csv(self) -> None:
        """Load data from a CSV file (Neurofeedback_BCI.py format)."""
        try:
            # Read the CSV file
            df = pd.read_csv(self.file_path)
            
            # Check expected format
            if 'lsl_timestamp' not in df.columns:
                raise ValueError("CSV file does not contain 'lsl_timestamp' column")
            
            # Extract timestamps
            self.timestamps_full = df['lsl_timestamp'].values
            
            # OpenBCI conversion constants
            # Cyton board: 24-bit ADC, gain = 24, Vref = 4.5V
            ADC_24BIT = 2**23 - 1  # 8388607
            GAIN = 24.0
            VREF = 4.5  # Volts
            
            # Conversion factor to microvolts
            # (Vref / Gain / ADC_max) * 1e6 to convert to microvolts
            scale_factor = (VREF / GAIN / ADC_24BIT) * 1e6
            
            # Extract EEG data
            channel_data = []
            for ch in self.channels:
                if ch in df.columns:
                    raw_values = df[ch].values
                    
                    # Convert from raw ADC to microvolts
                    # Handle NaN values
                    mask = ~np.isnan(raw_values)
                    converted_values = np.zeros_like(raw_values)
                    converted_values[mask] = raw_values[mask] * scale_factor
                    converted_values[~mask] = 0.0  # Replace NaN with 0
                    
                    channel_data.append(converted_values)
                    logging.debug(f"Channel {ch}: Raw range [{np.nanmin(raw_values):.0f}, {np.nanmax(raw_values):.0f}] -> "
                                f"μV range [{np.min(converted_values[mask]):.2f}, {np.max(converted_values[mask]):.2f}]")
                else:
                    # If channel not found, use zeros
                    logging.warning(f"Channel {ch} not found in CSV, using zeros")
                    channel_data.append(np.zeros(len(df)))
            
            # Convert to numpy array with shape (samples, channels)
            self.raw_data = np.column_stack(channel_data)
            
            # Extract markers if available
            if 'marker' in df.columns:
                self.markers = df['marker'].values
            
            logging.info(f"Loaded CSV file with {len(df)} samples and {len(channel_data)} channels (converted to μV)")
            
        except Exception as e:
            logging.exception("Error loading CSV file:")
            raise
    
    def _load_npy(self) -> None:
        """Load data from a NumPy .npy file."""
        try:
            # Load the data
            raw_data = np.load(self.file_path, allow_pickle=True)
            
            # Check if it's a dictionary with data and timestamps
            if isinstance(raw_data, np.ndarray) and hasattr(raw_data, 'dtype') and raw_data.dtype.names is not None:
                # Structured array with named fields
                if 'data' in raw_data.dtype.names and 'timestamps' in raw_data.dtype.names:
                    self.raw_data = raw_data['data']
                    self.timestamps_full = raw_data['timestamps']
                    if 'markers' in raw_data.dtype.names:
                        self.markers = raw_data['markers']
                    logging.info(f"Loaded structured NumPy array with {len(self.raw_data)} samples")
                    return
            
            # If raw_data is a dictionary
            if isinstance(raw_data, dict):
                if 'data' in raw_data and 'timestamps' in raw_data:
                    self.raw_data = raw_data['data']
                    self.timestamps_full = raw_data['timestamps']
                    if 'markers' in raw_data:
                        self.markers = raw_data['markers']
                    logging.info(f"Loaded NumPy dict with {len(self.raw_data)} samples")
                    return
            
            # If raw_data is just a plain array, assume it's the EEG data without timestamps
            if isinstance(raw_data, np.ndarray):
                if len(raw_data.shape) == 2:
                    self.raw_data = raw_data
                    # Generate artificial timestamps at the specified sample rate
                    self.timestamps_full = np.arange(len(raw_data)) / self.sample_rate
                    logging.info(f"Loaded raw NumPy array with {len(self.raw_data)} samples")
                    return
            
            raise ValueError("Unsupported NumPy file format")
            
        except Exception as e:
            logging.exception("Error loading NumPy file:")
            raise
    
    def _load_mne_format(self) -> None:
        """Load data from MNE-supported formats (EDF, BDF, XDF)."""
        try:
            import mne
            
            # Load raw data
            if self.file_extension == '.xdf':
                streams = mne.io.read_raw_xdf(self.file_path, verbose=False)
                if isinstance(streams, list):
                    # Find EEG stream
                    for stream in streams:
                        if 'eeg' in stream.info['type'].lower():
                            raw = stream
                            break
                    else:
                        raw = streams[0]  # Default to first stream if no EEG found
                else:
                    raw = streams
            elif self.file_extension in ['.edf', '.bdf']:
                raw = mne.io.read_raw_edf(self.file_path, verbose=False) if self.file_extension == '.edf' else mne.io.read_raw_bdf(self.file_path, verbose=False)
            else:
                raise ValueError(f"Unsupported file extension: {self.file_extension}")
                
            # Extract data
            data, times = raw[:, :]
            
            # Transpose to get (samples, channels)
            self.raw_data = data.T
            self.timestamps_full = times
            
            # Try to extract events/markers
            try:
                events = mne.find_events(raw)
                self.markers = np.zeros(len(times))
                for event in events:
                    idx = np.argmin(np.abs(times - event[0] / raw.info['sfreq']))
                    self.markers[idx] = event[2]
            except:
                logging.warning("Could not extract events from the data")
                
            logging.info(f"Loaded {self.file_extension} file with {len(self.raw_data)} samples and {self.raw_data.shape[1]} channels")
            
        except ImportError:
            logging.error("Could not import MNE. Please install with: pip install mne")
            raise
        except Exception as e:
            logging.exception(f"Error loading {self.file_extension} file:")
            raise
    
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        self.stop_background_update()
        self.connected = False
        self.raw_data = None
        self.markers = None
        self.current_position = 0
        logging.info("Disconnected from file data source")
    
    def start_background_update(self) -> None:
        """Start background thread for continuous data update simulation."""
        if self._update_thread is not None and self._update_thread.is_alive():
            logging.warning("Background update already running")
            return
            
        if not self.connected:
            logging.error("Cannot start background update: Not connected to data source")
            return
            
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._background_update)
        self._update_thread.daemon = True
        self._update_thread.start()
        logging.info("Started background file streaming thread")
    
    def stop_background_update(self) -> None:
        """Stop the background update thread."""
        if self._update_thread is not None:
            self._stop_event.set()
            self._update_thread.join(timeout=1.0)
            self._update_thread = None
            logging.info("Stopped background file streaming thread")
    
    def _background_update(self) -> None:
        """
        Background thread function to simulate real-time data streaming.
        This streams the pre-recorded data at the original recording rate.
        """
        time_per_sample = 1.0 / self.sample_rate
        
        while not self._stop_event.is_set() and self.current_position < len(self.raw_data):
            # Calculate how many samples to add based on time elapsed
            samples_to_add = 1  # Default to 1 sample per update
            
            # Add samples to buffer
            for _ in range(samples_to_add):
                if self.current_position >= len(self.raw_data):
                    break
                    
                # Add sample to circular buffer
                idx = self.buffer_head % self.buffer_size
                self.buffer[idx] = self.raw_data[self.current_position]
                self.timestamps[idx] = self.timestamps_full[self.current_position]
                
                # Update pointers
                self.buffer_head = (self.buffer_head + 1) % self.buffer_size
                self.current_position += 1
                self.samples_count += 1
                
                # Mark buffer as filled once we've collected enough samples
                if self.samples_count >= self.buffer_size:
                    self.buffer_filled = True
            
            # Sleep to simulate real-time acquisition
            time.sleep(time_per_sample * samples_to_add)
    
    def update_buffer(self) -> int:
        """
        Update the buffer with new samples from the file.
        
        Returns:
            int: Number of new samples added to the buffer
        """
        if not self.connected or self.current_position >= len(self.raw_data):
            return 0
            
        # Add one sample per call to simulate real-time behavior
        samples_to_add = min(1, len(self.raw_data) - self.current_position)
        
        for _ in range(samples_to_add):
            # Add sample to circular buffer
            idx = self.buffer_head % self.buffer_size
            self.buffer[idx] = self.raw_data[self.current_position]
            self.timestamps[idx] = self.timestamps_full[self.current_position]
            
            # Update pointers
            self.buffer_head = (self.buffer_head + 1) % self.buffer_size
            self.current_position += 1
            self.samples_count += 1
            
            # Mark buffer as filled once we've collected enough samples
            if self.samples_count >= self.buffer_size:
                self.buffer_filled = True
        
        return samples_to_add
    
    def get_chunk(self, window_size: int = 500, channels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a chunk of EEG data from the buffer.
        
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
        For file data, always return perfect quality.
        
        Returns:
            Dictionary of channel names to quality values (0.0-1.0)
        """
        return self.signal_quality
    
    def is_signal_good(self) -> bool:
        """
        Check if overall signal quality is acceptable.
        For file data, always return True.
        
        Returns:
            Boolean indicating if signal quality is above threshold
        """
        return True
    
    def reset_buffer(self) -> None:
        """Reset the data buffer and position in the file."""
        self.buffer = np.zeros((self.buffer_size, len(self.channels)))
        self.timestamps = np.zeros(self.buffer_size)
        self.buffer_head = 0
        self.buffer_filled = False
        self.samples_count = 0
        self.current_position = 0
        logging.info("File handler buffer reset")
    
    def seek(self, position: int) -> bool:
        """
        Seek to a specific position in the file.
        
        Args:
            position: Sample index to seek to
            
        Returns:
            Success indicator
        """
        if not self.connected or position >= len(self.raw_data):
            return False
            
        self.current_position = position
        self.reset_buffer()
        logging.info(f"Seek to position {position}")
        return True

    