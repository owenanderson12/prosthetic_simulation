#!/usr/bin/env python3
# Real-time EEG Motor Imagery BCI Script
# Follows the detailed pipeline techniques from the attached "Pipeline Draft" document

# Required libraries
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from scipy.signal import butter, filtfilt, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
import time

# ------------------ Configuration Parameters ------------------
SAMPLE_RATE = 250  # Sampling rate (matches OpenBCI Cyton)
CHANNELS = [2, 3, 4]  # EEG channels: C3, Cz, C4
WINDOW_SIZE = int(1.5 * SAMPLE_RATE)  # 1.5 second epochs
STEP_SIZE = int(0.1 * SAMPLE_RATE)    # Update every 0.1 second (sliding window)

# Frequency bands
MU_BAND = (8, 12)
BETA_BAND = (13, 30)

# Classifier and CSP setup
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
classifier = LDA()

# ------------------ Signal Processing Functions ------------------
def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    """ Bandpass filter for EEG data """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, freq, fs=250, Q=30):
    """ Notch filter to remove 60 Hz power-line interference """
    nyq = 0.5 * fs
    low = (freq - freq/Q) / nyq
    high = (freq + freq/Q) / nyq
    b, a = butter(2, [low, high], btype='bandstop')
    return filtfilt(b, a, signal)

def preprocess(signal):
    """ Apply preprocessing (bandpass and notch filtering) """
    filtered_signal = bandpass_filter(signal, SAMPLE_RATE, 0.5, 30)
    filtered_signal = notch_filter(filtered_signal, 60, SAMPLE_RATE)
    return filtered_signal

def extract_features(epoch):
    """ Extract features using CSP and PSD (Welch) methods """
    filtered_epoch = preprocess(epoch)

    # CSP Features
    csp_features = csp.transform(filtered_epoch[np.newaxis, :, :])[0]

    # PSD Features in Mu and Beta bands
    psd_features = []
    for band in [MU_BAND, BETA_BAND]:
        freqs, psd = welch(filtered_epoch, fs=SAMPLE_RATE, nperseg=WINDOW_SIZE)
        band_power = np.mean(psd[(freqs >= band[0]) & (freqs <= band[1])], axis=-1)
        psd_features.extend(band_power)

    # Combine CSP and PSD features
    return np.hstack([csp_features, psd_features])

# ------------------ Real-time BCI Loop ------------------
def run_bci():
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyUSB0'  # Modify for your system
    board = BoardShim(BoardShim.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream()

    print("Starting real-time BCI loop...")

    try:
        while True:
            # Get current data from OpenBCI
            data = board.get_current_board_data(WINDOW_SIZE)
            eeg_data = data[CHANNELS, -WINDOW_SIZE:]

            # Feature extraction
            features = extract_features(eeg_data)

            # Predict motor imagery using trained LDA classifier
            prediction = classifier.predict(features.reshape(1, -1))[0]

            # Real-time prosthetic hand control logic
            if prediction == 1:
                print("Grip Imagined - Prosthetic Hand Closing")
                # Insert Arduino command for closing hand
            else:
                print("Open Imagined - Prosthetic Hand Opening")
                # Insert Arduino command for opening hand

            time.sleep(STEP_SIZE / SAMPLE_RATE)

    except KeyboardInterrupt:
        print("Real-time BCI stopped.")
    finally:
        board.stop_stream()
        board.release_session()

# ------------------ Modifications for Testing ------------------
# For initial testing and debugging:
# 1. Use prerecorded EEG data instead of real-time streaming.
# 2. Validate preprocessing separately by visualizing filtered signals.
# 3. Perform offline CSP and LDA training using recorded calibration datasets.
# 4. Use mock predictions (fixed values) to verify prosthetic control logic without real EEG input.
# Example:
# prediction = 1  # Manually set prediction for testing

# ------------------ Main Entry Point ------------------
if __name__ == '__main__':
    # Load pretrained CSP and classifier here (e.g., using joblib.load)
    run_bci()
