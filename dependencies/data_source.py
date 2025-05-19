from typing import Literal, Optional, List, Tuple
from pylsl import StreamInlet, resolve_byprop
import numpy as np
from dependencies.eeg_acquisition import EEGAcquisition
import config


class ArtificialEEGSource:
    """connects to an artifical EEG LSL stream and provides data chunks"""

    def __init__(self, stream_name="ArtificialEEG"):
        print(f"looking for artifical LSL stream: {stream_name}")
        streams = resolve_byprop('name', stream_name, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream found with name '{stream_name}'.")
        self.inlet = StreamInlet(streams[0])
    
    def get_chunk(self, window_size=500, channels=None):
        """
        Retrieves a chunk of EEG data and corresponding timestamps.
        - window_size: Number of samples to retrieve
        - channels: list of channel indicies to retrieve, or None for all
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

class DataSource:
    def __init__(self, source_type: Literal["live","artificial","file"], source_path: Optional[str] = None, stream_name: str = "ArtificialEEG"):
        """
        DataSource provides a unified interface for different EEG data sources.
        - source_type: 'live', 'artificial', or 'file'
        - source_path: path to file (if using file source)
        - stream_name: LSL stream name for artificial source
        """
        self.source_type = source_type
        self.source_path = source_path
        self.stream_name = stream_name
        self.connected = False

        # Initialize the appropriate source and connect if needed
        if self.source_type == "artificial":
            # Artificial EEG via LSL
            self.source = ArtificialEEGSource(stream_name=self.stream_name)
            self.connected = True  # ArtificialEEGSource connects on init
        elif self.source_type == "live":
            # Live EEG via EEGAcquisition
            self.source = EEGAcquisition(config.__dict__)
            self.connected = False
        elif self.source_type == "file":
            # TODO: Implement or import file-based EEG data source
            self.source = None  # Placeholder
        else:
            raise ValueError("Invalid data source type. Must be 'live', 'artificial', or 'file'.")

    def is_signal_good(self):
        """
        Return signal quality status. For artificial data, always True.
        """
        if self.source_type == "artificial":
            return True
        if hasattr(self.source, "is_signal_good"):
            return self.source.is_signal_good()
        return True

    @property
    def sample_rate(self):
        if hasattr(self.source, "sample_rate"):
            return self.source.sample_rate
        return 250  # Default fallback

    def __getattr__(self, name):
        # Special-case artificial source: provide sensible defaults for expected attributes/methods
        if self.source_type == "artificial":
            if name == "clock_offset":
                return 0.0
            if name == "stop_background_update":
                def no_op(*args, **kwargs):
                    pass
                return no_op
            if name == "disconnect":
                def no_op(*args, **kwargs):
                    pass
                return no_op
        # Delegate to the underlying source if possible
        if self.source is not None and hasattr(self.source, name):
            return getattr(self.source, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")



    def connect(self):
        """
        Explicitly connect/start the EEG data source if needed (for live data).
        """
        if self.source_type == "live" and not self.connected:
            if hasattr(self.source, "connect"):
                success = self.source.connect()
                if success and hasattr(self.source, "start_background_update"):
                    self.source.start_background_update()
                self.connected = success
                return success
            else:
                raise RuntimeError("Live EEG source does not support connect().")
        # For artificial/simulated, connection happens at init
        return True


    def get_chunk(self, window_size: int = 500, channels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a chunk of EEG data and timestamps from the selected source.
        - window_size: number of samples to retrieve
        - channels: list of channel indices to select (or None for all)
        Returns: (data, timestamps) as numpy arrays
        """
        if self.source_type == "artificial":
            # Delegate to the ArtificialEEGSource instance
            return self.source.get_chunk(window_size=window_size, channels=channels)
        elif self.source_type == "live":

            # Ensure connection for live EEG
            if not self.connected:
                self.connect()
            return self.source.get_chunk(window_size=window_size, channels=channels)
        elif self.source_type == "file":

            # TODO: Delegate to the file-based EEG data source instance
            raise NotImplementedError("File source not implemented yet.")
        else:
            raise ValueError("Invalid data source type. Must be 'live', 'artificial', or 'file'.")
