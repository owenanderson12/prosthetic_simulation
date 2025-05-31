from dependencies.data_source import DataSource

# Create a DataSource instance for the artificial stream
ds = DataSource(source_type="artificial", stream_name="ArtificialEEG")

# Try to get a chunk of data (e.g., 500 samples)
data, timestamps = ds.get_chunk(window_size=500)

print("Data shape:", data.shape)
print("Timestamps shape:", timestamps.shape)
print("First 5 timestamps:", timestamps[:5])
print("First 5 samples:\n", data[:5])