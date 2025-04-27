#!/opt/anaconda3/envs/eeg_project_env/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

from modules.bandpass_filter import bandpass_filter
from modules.apply_grand_average_reference import apply_grand_average_reference
from modules.calculate_bandpower import calculate_bandpower

########################################################################
# Configuration
########################################################################

DATA_FILE = "/Users/owenanderson/Documents/NeurEx/Projects/Prosthetics/neurofeedback/MI_Neurofeedback/data/raw/MI_EEG_20250423_191831.csv" 
SAMPLE_RATE = 250  # Hz, matches the original MI script's sampling rate
FILTER_BAND = (1.0, 40.0)  # Bandpass filter range (in Hz)

# Channel name mapping
CHANNEL_MAPPING = {
    "CH1": "O1",
    "CH2": "FC1",
    "CH3": "C3",
    "CH4": "Fp1",
    "CH5": "Fp2",
    "CH6": "C4",
    "CH7": "FC2",
    "CH8": "O2"
}

# Channels of interest (with updated names)
CHANNELS_OF_INTEREST = {
    "LeftHemisphere": ["CP1", "C3", "C3-CP1"],  # Updated from CH2, CH3
    "RightHemisphere": ["C4", "CP2", "C4-CP2"]  # Updated from CH6, CH7
}

# Event markers from the original data-collection script
MARKER_RIGHT = "1.0"  # Right hand imagery start
MARKER_LEFT = "2.0"   # Left hand imagery start
MARKER_STOP = "3.0"   # End of imagery period

# Event IDs for MNE
EVENT_IDS = {
    "Right": 1,
    "Left": 2
}

# Epoching parameters (in seconds)
# Include 1 second before the start marker for baseline
EPOCH_START = -2.0  # Start 2 seconds before the marker
EPOCH_END = 3.0     # End 3 seconds after the marker

# Define baseline period for normalization
BASELINE_START = -2.0  # Start of baseline period
BASELINE_END = 0.0     # End of baseline period (stimulus onset)

#def main():
# 1. LOAD CSV DATA
df = pd.read_csv(DATA_FILE)
print('csv read')

# The script writes lines even if marker is empty. Let's keep only rows with valid data, ignoring partial lines.
# We assume all channels are numeric. Force convert to numeric, coerce missing -> NaN, then drop them.
original_channels = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]
for ch in original_channels:
    df[ch] = pd.to_numeric(df[ch], errors='coerce')
df["lsl_timestamp"] = pd.to_numeric(df["lsl_timestamp"], errors='coerce')

# Drop rows missing a timestamp or channel data
df.dropna(subset=["lsl_timestamp"] + original_channels, inplace=True)
df.reset_index(drop=True, inplace=True)

# Rename channels immediately after reading the CSV
for old_name, new_name in CHANNEL_MAPPING.items():
    df.rename(columns={old_name: new_name}, inplace=True)

# Convert time reference from raw LSL timestamps (seconds)
timestamps = df["lsl_timestamp"].values - df["lsl_timestamp"].values[0]

# 2. PREPROCESS
# First apply grand average reference
new_channel_names = list(CHANNEL_MAPPING.values())
raw_eeg = df[new_channel_names].values
referenced_eeg = apply_grand_average_reference(raw_eeg)

# Then apply bandpass filter
filtered_eeg = bandpass_filter(referenced_eeg, SAMPLE_RATE, FILTER_BAND[0], FILTER_BAND[1], order=4)

# 3. IDENTIFY MARKERS AND EXTRACT EVENTS FOR MNE
mne_events = []
event_descriptions = []

for i, marker_val in enumerate(df["marker"]):
    if str(marker_val) == MARKER_RIGHT:
        mne_events.append([i, 0, EVENT_IDS["Right"]])
        event_descriptions.append((timestamps[i], "Right"))
    elif str(marker_val) == MARKER_LEFT:
        mne_events.append([i, 0, EVENT_IDS["Left"]])
        event_descriptions.append((timestamps[i], "Left"))

# Convert to numpy array for MNE
mne_events = np.array(mne_events, dtype=int)

# 4. CREATE MNE OBJECTS
# Create info object
info = mne.create_info(
    ch_names=new_channel_names,
    sfreq=SAMPLE_RATE,
    ch_types=['eeg'] * len(new_channel_names)
)

# Create raw object from filtered data
raw = mne.io.RawArray(filtered_eeg.T, info)  # Note: MNE expects channels × time

# Set montage (using standard 10-20 names)
try:
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    print("Successfully set the montage.")
except ValueError as e:
    print(f"Montage error: {e}")
    print("Continuing without channel positions...")

# Create epochs
epochs = mne.Epochs(
    raw=raw,
    events=mne_events,
    event_id=EVENT_IDS,
    tmin=EPOCH_START,
    tmax=EPOCH_END,
    baseline=(BASELINE_START, BASELINE_END),
    preload=True
)

print(f"Created MNE Epochs object: {len(epochs)} epochs total")
print(f"Epochs per condition: {epochs.event_id}")

# Extract epochs data for visualization
epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
print(f"Epochs data shape: {epochs_data.shape}")

# 5. Compute features using MNE epochs
X = []
y = []
key = ['mu_power_left', 'beta_power_left', 'mu_power_right', 'beta_power_right']

# Get indices for channels of interest
c3_idx = new_channel_names.index("C3")  # Left motor area
c4_idx = new_channel_names.index("C4")  # Right motor area

# Extract and process each condition separately
for condition, event_id in EVENT_IDS.items():
    condition_epochs = epochs[condition].get_data()
    print(f"\nCondition: {condition}")
    print(f"Number of epochs: {len(condition_epochs)}")
    print(f"Shape of epochs: {condition_epochs.shape}")
    
    for epoch_idx, epoch in enumerate(condition_epochs):
        # Note: MNE epochs data has shape (epochs, channels, times)
        mu_power_left, beta_power_left = calculate_bandpower(
            epoch[c3_idx, :], SAMPLE_RATE, EPOCH_START)
        mu_power_right, beta_power_right = calculate_bandpower(
            epoch[c4_idx, :], SAMPLE_RATE, EPOCH_START)
        
        # Here you could store features for machine learning
        X.append([mu_power_left, beta_power_left, mu_power_right, beta_power_right])
        y.append(event_id)

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

epochs_train = epochs.copy().crop(tmin=0.0, tmax=3.0)
labels = epochs.events[:, -1] 

# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)

# Check class balance and print epoch counts
print(f"\nClass balance:")
print(f"Right class: {np.sum(labels == EVENT_IDS['Right'])} epochs")
print(f"Left class: {np.sum(labels == EVENT_IDS['Left'])} epochs")

# Apply additional bandpass filtering for CSP
# CSP works best on specific frequency bands like mu (8-12 Hz) and beta (13-30 Hz)
# We'll create filtered copies of our epochs for CSP
print("\nApplying motor-specific frequency filtering for CSP...")
epochs_motor = epochs_train.copy().filter(8.0, 30.0)  # Filter to motor-relevant frequencies
epochs_data_motor = epochs_motor.get_data(copy=False)

cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_motor)

# Assemble a classifier with proper regularization
lda = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')  # Use shrinkage for stability
csp = CSP(n_components=4, reg=0.1, log=True, norm_trace=False)  # Add regularization (reg=0.1)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_data_motor, labels, cv=cv, n_jobs=None)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print(f"Classification accuracy: {np.mean(scores):.3f} / Chance level: {class_balance:.3f}")

# plot CSP patterns estimated on full data for visualization
try:
    print("\nGenerating CSP patterns visualization...")
    csp.fit_transform(epochs_data_motor, labels)
    
    # Force plot to display
    fig = csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
    plt.show()  # Explicitly call plt.show() to ensure the plot is displayed
    
    # Check if figure was created successfully
    if fig is not None:
        print("CSP patterns figure generated successfully.")
        # You can also save the figure to a file
        #fig_path = "csp_patterns.png"
        #fig.savefig(fig_path)
        #print(f"CSP patterns saved to: {fig_path}")
    else:
        print("Warning: Figure object is None.")
except Exception as e:
    print(f"Error in CSP visualization: {e}")
    print("Trying with higher regularization...")
    try:
        # Try with higher regularization as a fallback
        csp = CSP(n_components=4, reg=0.5, log=True, norm_trace=False)
        csp.fit_transform(epochs_data_motor, labels)
        fig = csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
        plt.show()  # Explicitly call plt.show()
        
        # Save the figure
        fig_path = "csp_patterns_high_reg.png"
        fig.savefig(fig_path)
        print(f"CSP patterns (high reg) saved to: {fig_path}")
    except Exception as e2:
        print(f"Second attempt also failed: {e2}")
        print("Debugging montage and channel info...")
        print(f"Channel types: {[ch_type for ch_type in epochs.get_channel_types()]}")
        print(f"Montage: {epochs.info['dev_head_t']}")
        
        # Try simplest approach - plot with default params
        try:
            csp.plot_patterns(epochs.info)
            plt.show()
        except Exception as e3:
            print(f"Simple plot attempt failed: {e3}")
            
# Add a call to plt.pause to keep plots open if using interactive mode
plt.pause(0.1)