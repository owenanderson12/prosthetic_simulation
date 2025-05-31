import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Union, Literal
import numpy as np

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import necessary components
from dependencies.eeg_acquisition import EEGAcquisition
from dependencies.file_handler import FileHandler

class DataSource:
    """
    Abstract factory for EEG data sources.
    
    This class provides a unified interface for different data sources,
    whether live EEG acquisition or pre-recorded file playback.
    """
    
    def __init__(self, 
                config_dict: Dict,
                source_type: Literal["live", "file"] = "live", 
                source_path: Optional[str] = None):
        """
        Initialize the data source.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            source_type: Type of data source ("live" or "file")
            source_path: Path to the source file (required for "file" source type)
        """
        self.config = config_dict
        self.source_type = source_type
        self.source_path = source_path
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
        if self.data_acquisition:
            self.data_acquisition.start_background_update()
    
    def stop_background_update(self) -> None:
        """Stop the background update thread."""
        if self.data_acquisition:
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