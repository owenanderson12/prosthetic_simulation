from pylsl import StreamInlet, resolve_streams, resolve_byprop

# Resolve EEG stream
print("Looking for EEG stream...")
streams = resolve_byprop("type", "EEG", timeout=5)

# Create inlet to receive data
inlet = StreamInlet(streams[0])

while True:
    sample, timestamp = inlet.pull_sample()
    print(timestamp, sample)
