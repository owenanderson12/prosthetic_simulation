from typing import Literal, Optional, List, Tuple
import numpy as np

class DataSource:
    def __init__(self, source_type: Literal["live","file"], source_path: Optional[str] = None):
        self.source_type = source_type
        self.source_path = source_path

    def get_chunk(self, window_size: int = 500, channels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.source_type == "live":
            # Implement logic to get data from live source
            pass
        elif self.source_type == "file":
            # Implement logic to get data from file
            pass
        else:
            raise ValueError("Invalid data source type. Must be 'live' or 'file'.")