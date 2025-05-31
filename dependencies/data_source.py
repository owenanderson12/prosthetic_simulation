import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Union, Literal
import numpy as np
from pylsl import StreamInlet, resolve_byprop

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import necessary components
from dependencies.eeg_acquisition import EEGAcquisition
from dependencies.file_handler import FileHandler


class ArtificialEEGSource:
    """Connects to an artificial EEG LSL stream and provides data chunks"""

    def __init__(self, config_dict: Dict, stream_name="ArtificialEEG"):
        self.config = config_dict
        self.sample_rate = config_dict.get('SAMPLE_RATE', 250)
        print(f"Looking for artificial LSL stream: {stream_name}")
        streams = resolve_byprop('name', stream_name, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream found with name '{stream_name}'.")
        self.inlet = StreamInlet(streams[0])
        self.connected = True
    
    def connect(self) -> bool:
        """Already connected in __init__"""
        return self.connected
    
    def disconnect(self) -> None:
        """Disconnect from the artificial stream"""
        self.connected = False
    
    def get_chunk(self, window_size=500, channels=None):
        """
        Retrieves a chunk of EEG data and corresponding timestamps.
        - window_size: Number of samples to retrieve
        - channels: list of channel indices to retrieve, or None for all
        returns: (data, timestamps) as numpy arrays
        """
        buffer = []
        ts_buffer = []
        while len(buffer) < window_size:
            sample, timestamp = self.inlet.pull_sample(timeout=2.0)
            if sample is not None:
                buffer.append(sample)
                ts_buffer.append(timestamp)
        data = np.array(buffer)
        timestamps = np.array(ts_buffer)
        if channels is not None:
            data = data[:, channels]
        return data, timestamps
    
    def get_motor_imagery_data(self, window_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Get motor imagery channels data."""
        mi_channels = self.config.get('MI_CHANNEL_INDICES', [2, 3, 5, 6])
        return self.get_chunk(window_size, mi_channels)
    
    def check_signal_quality(self) -> Dict[str, float]:
        """For artificial data, always return perfect quality"""
        channels = self.config.get('EEG_CHANNELS', ['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
        return {ch: 1.0 for ch in channels}
    
    def is_signal_good(self) -> bool:
        """For artificial data, signal is always good"""
        return True
    
    def reset_buffer(self) -> None:
        """No buffer to reset for artificial source"""
        pass


class DataSource:
    """
    Abstract factory for EEG data sources.
    
    This class provides a unified interface for different data sources:
    - Live EEG acquisition via LSL
    - Pre-recorded file playback
    - Artificial data stream for testing
    """
    
    def __init__(self, 
                config_dict: Dict,
                source_type: Literal["live", "file", "artificial"] = "live", 
                source_path: Optional[str] = None,
                stream_name: str = "ArtificialEEG"):
        """
        Initialize the data source.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            source_type: Type of data source ("live", "file", or "artificial")
            source_path: Path to the source file (required for "file" source type)
            stream_name: LSL stream name for artificial source
        """
        self.config = config_dict
        self.source_type = source_type
        self.source_path = source_path
        self.stream_name = stream_name
        self.data_acquisition = None
        
        # Check that source_path is provided for file source type
        if source_type == "file" and not source_path:
            raise ValueError("source_path must be provided for file source type")
            
        # Initialize the appropriate data source
        self._initialize_source()
        
        logging.info(f"Data source initialized with type: {source_type}")
    
    def _initialize_source(self) -> None:
        """Initialize the appropriate data source based on source_type."""
        try:
            if self.source_type == "live":
                self.data_acquisition = EEGAcquisition(self.config)
            elif self.source_type == "file":
                self.data_acquisition = FileHandler(self.config, self.source_path)
            elif self.source_type == "artificial":
                self.data_acquisition = ArtificialEEGSource(self.config, self.stream_name)
            else:
                raise ValueError(f"Invalid source type: {self.source_type}")
        except Exception as e:
            logging.exception("Error initializing data source:")
            raise
    
    def connect(self) -> bool:
        """
        Connect to the data source.
        
        Returns:
            Success indicator
        """
        if not self.data_acquisition:
            logging.error("Data acquisition not initialized")
            return False
            
        return self.data_acquisition.connect()
    
    def disconnect(self) -> None:
        """Disconnect from the data source."""
        if self.data_acquisition:
            self.data_acquisition.disconnect()
    
    def start_background_update(self) -> None:
        """Start background thread for continuous data update."""
        if self.data_acquisition and hasattr(self.data_acquisition, 'start_background_update'):
            self.data_acquisition.start_background_update()
    
    def stop_background_update(self) -> None:
        """Stop the background update thread."""
        if self.data_acquisition and hasattr(self.data_acquisition, 'stop_background_update'):
            self.data_acquisition.stop_background_update()
    
    def get_chunk(self, window_size: int = 500, channels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a chunk of EEG data from the source.
        
        Args:
            window_size: Number of samples to return
            channels: Indices of channels to return (default: all)
            
        Returns:
            Tuple of (data_array, timestamps_array)
        """
        if not self.data_acquisition:
            return np.array([]), np.array([])
            
        return self.data_acquisition.get_chunk(window_size, channels)
    
    def get_motor_imagery_data(self, window_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get motor imagery channels data.
        
        Args:
            window_size: Number of samples to return
            
        Returns:
            Tuple of (data_array, timestamps_array) for MI channels
        """
        if not self.data_acquisition:
            return np.array([]), np.array([])
            
        return self.data_acquisition.get_motor_imagery_data(window_size)
    
    def check_signal_quality(self) -> Dict[str, float]:
        """
        Check the signal quality of all channels.
        
        Returns:
            Dictionary of channel names to quality values (0.0-1.0)
        """
        if not self.data_acquisition:
            return {}
            
        return self.data_acquisition.check_signal_quality()
    
    def is_signal_good(self) -> bool:
        """
        Check if overall signal quality is acceptable.
        
        Returns:
            Boolean indicating if signal quality is above threshold
        """
        if not self.data_acquisition:
            return False
            
        return self.data_acquisition.is_signal_good()
    
    def reset_buffer(self) -> None:
        """Reset the data buffer."""
        if self.data_acquisition:
            self.data_acquisition.reset_buffer()
    
    @property
    def sample_rate(self):
        """Get the sample rate from the data source"""
        if hasattr(self.data_acquisition, 'sample_rate'):
            return self.data_acquisition.sample_rate
        return self.config.get('SAMPLE_RATE', 250)  # Default fallback
    
    @property
    def clock_offset(self):
        """Get the clock offset for LSL synchronization"""
        if hasattr(self.data_acquisition, 'clock_offset'):
            return self.data_acquisition.clock_offset
        return 0.0  # Default for non-LSL sources
