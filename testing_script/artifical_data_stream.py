from pylsl import StreamInfo, StreamOutlet
import pandas as pd
import time

# Load your EEG data from CSV
df = pd.read_csv("/home/aaronschott/VS/projects/OLEMAA/prosthetic/Data/raw/session_3/MI_EEG_20250406_163935.csv")

# Extract channel data columns (columns 2–9)
channel_columns = df.columns[1:9]

# Define LSL stream parameters
n_channels = len(channel_columns)
sampling_rate = 250  # Adjust this to match your data's sampling rate
stream_name = "ArtificialEEG"
stream_type = "EEG"
channel_format = 'float32'

# Create StreamInfo and StreamOutlet for EEG
info = StreamInfo(stream_name, stream_type, n_channels, sampling_rate, channel_format, 'artificial12345')
outlet = StreamOutlet(info)

# (Optional) Create StreamInfo and StreamOutlet for Markers
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'marker12345')
marker_outlet = StreamOutlet(marker_info)

print("Streaming EEG and markers via LSL...")

# Stream data
timestamps = df.iloc[:, 0].values
data = df[channel_columns].values.astype('float32')
markers = df.iloc[:, 9].values

start_time = time.time()

for idx in range(len(df)):
    sample = data[idx]
    outlet.push_sample(sample)

    # Push marker if available (non-empty cell)
    marker = str(markers[idx]).strip()
    if marker and marker.lower() != 'nan':
        marker_outlet.push_sample([marker])

    # Maintain correct sampling rate using timestamps
    if idx < len(df) - 1:
        interval = timestamps[idx + 1] - timestamps[idx]
        elapsed = time.time() - start_time
        expected_elapsed = timestamps[idx + 1] - timestamps[0]
        sleep_time = expected_elapsed - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

print("Data streaming complete.")
