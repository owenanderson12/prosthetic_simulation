import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.bandpass_filter import bandpass_filter

import numpy as np
from pylsl import StreamInlet, resolve_byprop

# Parameters for bandpass filter
FS = 250  # Sampling frequency in Hz (adjust as needed)
LOWCUT = 1.0  # Low cutoff frequency in Hz
HIGHCUT = 40.0  # High cutoff frequency in Hz
ORDER = 4  # Filter order
BUFFER_SIZE = 250  # Number of samples to buffer before filtering (e.g., 1 second)

print("Looking for EEG stream...")
streams = resolve_byprop("type", "EEG", timeout=5)
if not streams:
    raise RuntimeError("No EEG stream found.")

inlet = StreamInlet(streams[0])

buffer = []

print("Collecting EEG samples and applying bandpass filter...")
while True:
    sample, timestamp = inlet.pull_sample()
    buffer.append(sample)
    if len(buffer) >= BUFFER_SIZE:
        data = np.array(buffer)
        filtered = bandpass_filter(data, FS, LOWCUT, HIGHCUT, order=ORDER)
        # Print the first 10 rows of filtered data
        print(f"Filtered data (first 10 rows):\n{filtered[:10]}")
        buffer = []  # Clear buffer for next batch
