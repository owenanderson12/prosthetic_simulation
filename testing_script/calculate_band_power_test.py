import numpy as np
from MI_NFB_calculate_band_power import calculate_band_power
from pylsl import StreamInlet, resolve_byprop

fs = 250  # Sampling frequency in Hz (should match artificial_data_stream.py)
duration_sec = 10  # length of sample in seconds
n_samples = fs * duration_sec

# Connect to the artificial EEG LSL stream
print("Looking for ArtificialEEG stream...")
streams = resolve_byprop('name', 'ArtificialEEG', timeout=5)
if not streams:
    raise RuntimeError("No ArtificialEEG stream found.")

inlet = StreamInlet(streams[0])

# Collect a buffer of samples
print(f"Collecting {n_samples} samples from ArtificialEEG stream...")
buffer = []
while len(buffer) < n_samples:
    sample, timestamp = inlet.pull_sample()
    buffer.append(sample)

# Convert to numpy array and select a single channel (e.g., channel 0)
data = np.array(buffer)
signal = data[:, 3]  # change index as needed

# Calculate band power in the mu (8-13 Hz) and beta (13-30 Hz) bands
mu_power, freqs, psd = calculate_band_power(signal, fs, (8, 13))
beta_power, _, _ = calculate_band_power(signal, fs, (13, 30))

print(f"Mu band power: {mu_power}")
print(f"Beta band power: {beta_power}")

# Plot the PSD
import matplotlib.pyplot as plt
plt.plot(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density of Simulated EEG')
plt.show()