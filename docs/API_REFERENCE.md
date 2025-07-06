# BCI System API Reference

This document provides detailed API documentation for the BCI-Controlled Prosthetic System, including all major modules, classes, and functions.

**Version:** 2.0  
**Last Updated:** 2024-12-20  
**Coverage:** Core system components and modules

---

## Table of Contents
- [Core System Components](#core-system-components)
- [Signal Processing Modules](#signal-processing-modules)
- [Analysis Modules](#analysis-modules)
- [Data Handling](#data-handling)
- [Configuration Management](#configuration-management)
- [Utility Functions](#utility-functions)
- [Integration Interfaces](#integration-interfaces)

---

## Core System Components

### BCISystem (`dependencies/BCISystem.py`)

Main system orchestrator that coordinates all BCI components.

#### Class: `BCISystem`

```python
class BCISystem:
    """Main BCI system orchestrator for coordinating all components"""
    
    def __init__(self, config=None):
        """
        Initialize BCI system with configuration
        
        Args:
            config: Configuration object or dict with system parameters
        """
        
    def start_calibration(self, visualize=False) -> bool:
        """
        Start guided calibration procedure
        
        Args:
            visualize (bool): Enable real-time visualization
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        
    def start_processing(self, model_path=None, visualize=False) -> bool:
        """
        Start real-time BCI processing
        
        Args:
            model_path (str): Path to trained model file
            visualize (bool): Enable real-time visualization
            
        Returns:
            bool: True if processing started successfully
        """
        
    def stop(self):
        """Stop all system components and cleanup resources"""
        
    def get_status(self) -> dict:
        """
        Get current system status
        
        Returns:
            dict: System status information
        """
```

#### Key Methods:

```python
def set_data_source(self, source_type: str, **kwargs):
    """
    Configure data source for the system
    
    Args:
        source_type (str): 'live', 'file', or 'artificial'
        **kwargs: Source-specific parameters
    """

def load_calibration(self, calibration_path: str) -> bool:
    """
    Load existing calibration model
    
    Args:
        calibration_path (str): Path to calibration file
        
    Returns:
        bool: True if loaded successfully
    """

def get_classification_result(self) -> dict:
    """
    Get latest classification result
    
    Returns:
        dict: {
            'class': str,           # 'left', 'right', 'idle'
            'confidence': float,    # 0.0-1.0
            'probability': float,   # Raw classifier probability
            'timestamp': float      # Unix timestamp
        }
    """
```

### Signal Processor (`dependencies/signal_processor.py`)

Advanced signal processing pipeline for EEG data.

#### Class: `SignalProcessor`

```python
class SignalProcessor:
    """Advanced signal processing for motor imagery BCI"""
    
    def __init__(self, sample_rate=250, channels=8):
        """
        Initialize signal processor
        
        Args:
            sample_rate (int): EEG sampling rate in Hz
            channels (int): Number of EEG channels
        """
        
    def process(self, eeg_data: np.ndarray, timestamps: np.ndarray) -> dict:
        """
        Process EEG window and extract features
        
        Args:
            eeg_data (np.ndarray): EEG data window [samples × channels]
            timestamps (np.ndarray): Corresponding timestamps
            
        Returns:
            dict: {
                'valid': bool,              # Processing successful
                'features': dict,           # Extracted features
                'quality': float,           # Signal quality score
                'artifacts_rejected': int   # Number of artifacts removed
            }
        """
        
    def train_csp(self, left_trials: List[np.ndarray], 
                  right_trials: List[np.ndarray]) -> bool:
        """
        Train Common Spatial Pattern filters
        
        Args:
            left_trials: List of left motor imagery trials
            right_trials: List of right motor imagery trials
            
        Returns:
            bool: True if training successful
        """
```

#### Signal Processing Methods:

```python
def apply_spatial_filter(self, data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference and CSP filters
    
    Args:
        data (np.ndarray): EEG data [samples × channels]
        
    Returns:
        np.ndarray: Spatially filtered data
    """

def apply_temporal_filter(self, data: np.ndarray) -> np.ndarray:
    """
    Apply bandpass and notch filtering
    
    Args:
        data (np.ndarray): Raw EEG data
        
    Returns:
        np.ndarray: Temporally filtered data
    """

def extract_band_power(self, data: np.ndarray) -> dict:
    """
    Extract band power features (mu, beta)
    
    Args:
        data (np.ndarray): Filtered EEG data
        
    Returns:
        dict: {
            'mu_power': np.ndarray,    # Mu band power per channel
            'beta_power': np.ndarray,  # Beta band power per channel
            'erd_mu': float,           # ERD in mu band
            'erd_beta': float          # ERD in beta band
        }
    """

def detect_artifacts(self, data: np.ndarray) -> dict:
    """
    Detect and characterize artifacts in EEG data
    
    Args:
        data (np.ndarray): EEG data window
        
    Returns:
        dict: {
            'has_artifacts': bool,     # Artifacts detected
            'artifact_channels': list, # Channels with artifacts
            'artifact_types': list,    # Types of artifacts found
            'quality_score': float     # Overall quality (0-1)
        }
    """
```

### Classifier (`dependencies/classifier.py`)

Multi-algorithm classification engine with confidence estimation.

#### Class: `Classifier`

```python
class Classifier:
    """Multi-algorithm classification with adaptive performance"""
    
    def __init__(self, classifier_type='lda', **kwargs):
        """
        Initialize classifier
        
        Args:
            classifier_type (str): 'lda', 'svm', 'rf', 'xgb'
            **kwargs: Algorithm-specific parameters
        """
        
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train classifier on feature data
        
        Args:
            X (np.ndarray): Feature matrix [samples × features]
            y (np.ndarray): Labels [samples]
            
        Returns:
            dict: Training results and performance metrics
        """
        
    def predict(self, X: np.ndarray) -> dict:
        """
        Classify feature vector(s)
        
        Args:
            X (np.ndarray): Feature vector(s) [samples × features]
            
        Returns:
            dict: {
                'class': str or np.ndarray,        # Predicted class(es)
                'probability': float or np.ndarray, # Class probabilities
                'confidence': float or np.ndarray   # Confidence scores
            }
        """
```

#### Classification Methods:

```python
def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                   cv_folds=5) -> dict:
    """
    Perform cross-validation
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        cv_folds (int): Number of CV folds
        
    Returns:
        dict: Cross-validation results with statistics
    """

def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                           param_grid: dict = None) -> dict:
    """
    Optimize hyperparameters using grid search
    
    Args:
        X (np.ndarray): Training features
        y (np.ndarray): Training labels
        param_grid (dict): Parameter grid for optimization
        
    Returns:
        dict: Best parameters and optimization results
    """

def get_feature_importance(self) -> np.ndarray:
    """
    Get feature importance scores (if supported by algorithm)
    
    Returns:
        np.ndarray: Feature importance scores
    """

def save_model(self, filepath: str, metadata: dict = None):
    """
    Save trained model with metadata
    
    Args:
        filepath (str): Path to save model
        metadata (dict): Additional metadata to save
    """

def load_model(self, filepath: str) -> dict:
    """
    Load trained model from file
    
    Args:
        filepath (str): Path to model file
        
    Returns:
        dict: Model metadata and loading status
    """
```

---

## Signal Processing Modules

### Bandpass Filter (`modules/bandpass_filter.py`)

```python
def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                          fs: int, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter
    
    Args:
        data (np.ndarray): Input signal [samples × channels]
        lowcut (float): Low cutoff frequency (Hz)
        highcut (float): High cutoff frequency (Hz)
        fs (int): Sampling frequency (Hz)
        order (int): Filter order
        
    Returns:
        np.ndarray: Filtered signal
    """

def notch_filter(data: np.ndarray, notch_freq: float, fs: int,
                quality: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove line noise
    
    Args:
        data (np.ndarray): Input signal
        notch_freq (float): Notch frequency (Hz)
        fs (int): Sampling frequency (Hz)
        quality (float): Quality factor
        
    Returns:
        np.ndarray: Filtered signal
    """
```

### Epoch Extraction (`modules/extract_epochs.py`)

```python
def extract_epochs(data: np.ndarray, events: np.ndarray, 
                   tmin: float, tmax: float, baseline: tuple = None) -> dict:
    """
    Extract epochs from continuous EEG data
    
    Args:
        data (np.ndarray): Continuous EEG data [samples × channels]
        events (np.ndarray): Event markers [n_events × 3]
        tmin (float): Start time relative to event (seconds)
        tmax (float): End time relative to event (seconds)  
        baseline (tuple): Baseline correction period (tmin, tmax)
        
    Returns:
        dict: {
            'epochs': np.ndarray,      # Extracted epochs [trials × samples × channels]
            'events': np.ndarray,      # Corresponding events
            'times': np.ndarray,       # Time vector
            'baseline_corrected': bool # Whether baseline correction applied
        }
    """

def create_sliding_windows(data: np.ndarray, window_size: int,
                          overlap: int = 0) -> np.ndarray:
    """
    Create sliding windows from continuous data
    
    Args:
        data (np.ndarray): Continuous data [samples × channels]
        window_size (int): Window size in samples
        overlap (int): Overlap between windows in samples
        
    Returns:
        np.ndarray: Windowed data [windows × window_size × channels]
    """
```

### Band Power Calculation (`modules/calculate_bandpower.py`)

```python
def bandpower(data: np.ndarray, fs: int, band: tuple,
              method: str = 'welch', window: str = 'hann') -> np.ndarray:
    """
    Calculate band power using spectral methods
    
    Args:
        data (np.ndarray): EEG data [samples × channels] or [samples]
        fs (int): Sampling frequency
        band (tuple): Frequency band (low, high) in Hz
        method (str): Method ('welch', 'periodogram', 'multitaper')
        window (str): Window function for spectral estimation
        
    Returns:
        np.ndarray: Band power values [channels] or scalar
    """

def relative_bandpower(data: np.ndarray, fs: int, band: tuple,
                      total_band: tuple = (0.5, 40)) -> np.ndarray:
    """
    Calculate relative band power (band power / total power)
    
    Args:
        data (np.ndarray): EEG data
        fs (int): Sampling frequency  
        band (tuple): Band of interest (low, high) Hz
        total_band (tuple): Total frequency range for normalization
        
    Returns:
        np.ndarray: Relative band power values
    """

def calculate_erd_ers(baseline_power: np.ndarray, 
                     task_power: np.ndarray) -> np.ndarray:
    """
    Calculate Event-Related Desynchronization/Synchronization
    
    Args:
        baseline_power (np.ndarray): Baseline period power
        task_power (np.ndarray): Task period power
        
    Returns:
        np.ndarray: ERD/ERS values (negative = ERD, positive = ERS)
    """
```

---

## Analysis Modules

### ERD/ERS Analysis (`modules/plot_erd_time_course.py`)

```python
def plot_erd_time_course(epochs: np.ndarray, fs: int, freqs: np.ndarray,
                        baseline: tuple, channels: list = None,
                        save_path: str = None) -> matplotlib.figure.Figure:
    """
    Plot ERD/ERS time-course analysis
    
    Args:
        epochs (np.ndarray): Epoched data [trials × samples × channels]
        fs (int): Sampling frequency
        freqs (np.ndarray): Frequencies of interest
        baseline (tuple): Baseline period (tmin, tmax) in seconds
        channels (list): Channel names or indices to plot
        save_path (str): Path to save figure
        
    Returns:
        matplotlib.figure.Figure: Generated figure
    """

def compute_time_frequency(epochs: np.ndarray, fs: int, 
                          freqs: np.ndarray, n_cycles: float = 7.0) -> dict:
    """
    Compute time-frequency representation using wavelets
    
    Args:
        epochs (np.ndarray): Epoched data
        fs (int): Sampling frequency
        freqs (np.ndarray): Frequencies for analysis
        n_cycles (float): Number of cycles for wavelet
        
    Returns:
        dict: {
            'power': np.ndarray,     # Time-frequency power
            'times': np.ndarray,     # Time vector
            'freqs': np.ndarray      # Frequency vector
        }
    """
```

### Wavelet Spectrogram (`modules/plot_wavelet_spectrogram.py`)

```python
def plot_wavelet_spectrogram(data: np.ndarray, fs: int, channel: int = 0,
                            freqs: np.ndarray = None, figsize: tuple = (12, 6),
                            save_path: str = None) -> matplotlib.figure.Figure:
    """
    Generate wavelet-based spectrogram
    
    Args:
        data (np.ndarray): EEG data [samples × channels]
        fs (int): Sampling frequency
        channel (int): Channel index to plot
        freqs (np.ndarray): Frequency range for analysis
        figsize (tuple): Figure size (width, height)
        save_path (str): Path to save figure
        
    Returns:
        matplotlib.figure.Figure: Generated spectrogram
    """

def continuous_wavelet_transform(signal: np.ndarray, fs: int,
                                freqs: np.ndarray) -> np.ndarray:
    """
    Compute continuous wavelet transform
    
    Args:
        signal (np.ndarray): Input signal [samples]
        fs (int): Sampling frequency
        freqs (np.ndarray): Frequencies for analysis
        
    Returns:
        np.ndarray: Wavelet coefficients [freqs × samples]
    """
```

### Statistical Analysis (`modules/plot_bandpower_boxplots.py`)

```python
def plot_bandpower_boxplots(left_powers: np.ndarray, right_powers: np.ndarray,
                           band_names: list, channels: list = None,
                           save_path: str = None) -> matplotlib.figure.Figure:
    """
    Create boxplots comparing band powers between conditions
    
    Args:
        left_powers (np.ndarray): Band powers for left condition
        right_powers (np.ndarray): Band powers for right condition
        band_names (list): Names of frequency bands
        channels (list): Channel names
        save_path (str): Path to save figure
        
    Returns:
        matplotlib.figure.Figure: Generated boxplot figure
    """

def statistical_comparison(group1: np.ndarray, group2: np.ndarray,
                          test: str = 'ttest') -> dict:
    """
    Perform statistical comparison between groups
    
    Args:
        group1 (np.ndarray): First group data
        group2 (np.ndarray): Second group data
        test (str): Statistical test ('ttest', 'wilcoxon', 'permutation')
        
    Returns:
        dict: {
            'statistic': float,    # Test statistic
            'p_value': float,      # P-value
            'effect_size': float,  # Effect size (Cohen's d)
            'significant': bool    # Significance at p < 0.05
        }
    """
```

---

## Data Handling

### File Handler (`dependencies/file_handler.py`)

```python
class FileHandler:
    """Handle various EEG file formats and data sources"""
    
    def load_csv_data(self, filepath: str) -> dict:
        """
        Load EEG data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            dict: {
                'data': np.ndarray,        # EEG data [samples × channels]
                'timestamps': np.ndarray,   # Timestamps
                'sample_rate': float,      # Sampling rate
                'channels': list           # Channel names
            }
        """
        
    def save_processed_data(self, data: dict, filepath: str):
        """
        Save processed data to NPZ format
        
        Args:
            data (dict): Processed data dictionary
            filepath (str): Output file path
        """
        
    def load_processed_data(self, filepath: str) -> dict:
        """
        Load processed data from NPZ file
        
        Args:
            filepath (str): Path to NPZ file
            
        Returns:
            dict: Loaded data dictionary
        """
```

### Data Source (`dependencies/data_source.py`)

```python
class DataSource:
    """Abstract base class for EEG data sources"""
    
    def connect(self) -> bool:
        """
        Connect to data source
        
        Returns:
            bool: True if connection successful
        """
        
    def disconnect(self):
        """Disconnect from data source"""
        
    def get_chunk(self, n_samples: int = None) -> tuple:
        """
        Get chunk of EEG data
        
        Args:
            n_samples (int): Number of samples to retrieve
            
        Returns:
            tuple: (data, timestamps) arrays
        """
        
    def is_connected(self) -> bool:
        """
        Check connection status
        
        Returns:
            bool: True if connected
        """

class LiveEEGSource(DataSource):
    """Live EEG data from LSL stream"""
    
    def __init__(self, stream_name: str = "OpenBCI_EEG"):
        """
        Initialize live EEG source
        
        Args:
            stream_name (str): Name of LSL stream to connect to
        """

class FileEEGSource(DataSource):
    """EEG data from file with real-time simulation"""
    
    def __init__(self, filepath: str, loop: bool = True):
        """
        Initialize file EEG source
        
        Args:
            filepath (str): Path to EEG data file
            loop (bool): Loop file when end is reached
        """
```

---

## Configuration Management

### Configuration (`config.py`)

```python
# Signal Processing Configuration
SAMPLE_RATE = 250                    # Hz - EEG sampling rate
WINDOW_SIZE = 2.0                   # seconds - Classification window
WINDOW_OVERLAP = 0.5                # seconds - Window overlap
MU_BAND = (8, 13)                   # Hz - Mu rhythm frequency band
BETA_BAND = (13, 30)                # Hz - Beta rhythm frequency band

# Classification Configuration  
CLASSIFIER_THRESHOLD = 0.65         # Confidence threshold for actions
MIN_CONFIDENCE = 0.55               # Minimum confidence for output
ADAPTIVE_THRESHOLD = True           # Enable adaptive thresholding
CSP_COMPONENTS = 4                  # Number of CSP spatial filters

# Hardware Configuration
LSL_STREAM_NAME = "OpenBCI_EEG"     # LSL input stream name
OUTPUT_STREAM_NAME = "ProstheticControl"  # LSL output stream name
CHANNELS = 8                        # Number of EEG channels

def load_config(config_file: str = None) -> dict:
    """
    Load configuration from file or use defaults
    
    Args:
        config_file (str): Path to JSON configuration file
        
    Returns:
        dict: Configuration parameters
    """

def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        bool: True if configuration is valid
    """

def update_config(updates: dict):
    """
    Update global configuration parameters
    
    Args:
        updates (dict): Parameters to update
    """
```

---

## Utility Functions

### Model Utilities

```python
def save_model_with_metadata(model: object, filepath: str, 
                            metadata: dict = None):
    """
    Save model with comprehensive metadata
    
    Args:
        model: Trained model object
        filepath (str): Path to save model
        metadata (dict): Additional metadata
    """

def load_model_with_validation(filepath: str) -> dict:
    """
    Load model and validate compatibility
    
    Args:
        filepath (str): Path to model file
        
    Returns:
        dict: {
            'model': object,           # Loaded model
            'metadata': dict,          # Model metadata
            'compatible': bool,        # Version compatibility
            'warnings': list           # Compatibility warnings
        }
    """

def compare_models(model_paths: list) -> pandas.DataFrame:
    """
    Compare multiple models and their performance
    
    Args:
        model_paths (list): List of model file paths
        
    Returns:
        pandas.DataFrame: Comparison results
    """
```

### Data Utilities

```python
def validate_eeg_data(data: np.ndarray, fs: int) -> dict:
    """
    Validate EEG data quality and format
    
    Args:
        data (np.ndarray): EEG data to validate
        fs (int): Sampling frequency
        
    Returns:
        dict: {
            'valid': bool,             # Data is valid
            'issues': list,            # List of issues found
            'recommendations': list,   # Recommended fixes
            'quality_score': float     # Overall quality (0-1)
        }
    """

def preprocess_eeg_batch(data_files: list, output_dir: str,
                        filter_params: dict = None) -> dict:
    """
    Batch preprocess multiple EEG files
    
    Args:
        data_files (list): List of file paths to process
        output_dir (str): Output directory for processed files
        filter_params (dict): Preprocessing parameters
        
    Returns:
        dict: Processing results and statistics
    """

def merge_sessions(session_files: list) -> dict:
    """
    Merge multiple session files into combined dataset
    
    Args:
        session_files (list): List of session file paths
        
    Returns:
        dict: Merged dataset with metadata
    """
```

### Performance Utilities

```python
def profile_processing_speed(data: np.ndarray, processor: object,
                           n_iterations: int = 100) -> dict:
    """
    Profile signal processing speed
    
    Args:
        data (np.ndarray): Test data
        processor: Signal processor object
        n_iterations (int): Number of test iterations
        
    Returns:
        dict: {
            'mean_time': float,        # Mean processing time (ms)
            'std_time': float,         # Standard deviation (ms)
            'samples_per_second': float, # Processing throughput
            'real_time_factor': float  # Real-time capability factor
        }
    """

def memory_usage_monitor(func: callable, *args, **kwargs) -> dict:
    """
    Monitor memory usage during function execution
    
    Args:
        func: Function to monitor
        *args, **kwargs: Function arguments
        
    Returns:
        dict: {
            'peak_memory_mb': float,   # Peak memory usage
            'memory_delta_mb': float,  # Memory change
            'execution_time': float,   # Execution time
            'result': object          # Function result
        }
    """
```

---

## Integration Interfaces

### LSL Interface (`dependencies/simulation_interface.py`)

```python
class LSLInterface:
    """Interface for Lab Streaming Layer communication"""
    
    def __init__(self, stream_name: str, stream_type: str = "Control"):
        """
        Initialize LSL interface
        
        Args:
            stream_name (str): Name of LSL stream
            stream_type (str): Type of LSL stream
        """
        
    def create_outlet(self, channel_count: int, sample_rate: float,
                     channel_format: str = 'float32') -> bool:
        """
        Create LSL outlet stream
        
        Args:
            channel_count (int): Number of data channels
            sample_rate (float): Sampling rate
            channel_format (str): Data format
            
        Returns:
            bool: True if outlet created successfully
        """
        
    def send_sample(self, data: list, timestamp: float = None):
        """
        Send data sample through LSL stream
        
        Args:
            data (list): Data values to send
            timestamp (float): Optional timestamp
        """
        
    def close_outlet(self):
        """Close LSL outlet stream"""

def find_lsl_streams(stream_name: str = None, stream_type: str = None,
                    timeout: float = 1.0) -> list:
    """
    Find available LSL streams
    
    Args:
        stream_name (str): Filter by stream name
        stream_type (str): Filter by stream type
        timeout (float): Search timeout in seconds
        
    Returns:
        list: List of available streams
    """
```

### Unity Interface

```python
class UnityCommandInterface:
    """Interface for sending commands to Unity simulation"""
    
    def __init__(self):
        """Initialize Unity command interface"""
        
    def send_hand_command(self, hand_state: float, confidence: float):
        """
        Send hand control command
        
        Args:
            hand_state (float): Hand state (0.0=closed, 1.0=open)
            confidence (float): Classification confidence
        """
        
    def send_wrist_command(self, wrist_state: float, confidence: float):
        """
        Send wrist control command
        
        Args:
            wrist_state (float): Wrist rotation (0.0-1.0)
            confidence (float): Classification confidence
        """
        
    def send_idle_command(self):
        """Send idle/no-action command"""
        
    def get_feedback(self) -> dict:
        """
        Get feedback from Unity simulation
        
        Returns:
            dict: Unity feedback data
        """
```

---

## Error Handling

### Exception Classes

```python
class BCIError(Exception):
    """Base exception for BCI system errors"""
    pass

class DataSourceError(BCIError):
    """Errors related to data source connection/operation"""
    pass

class ProcessingError(BCIError):
    """Errors in signal processing pipeline"""
    pass

class ClassificationError(BCIError):
    """Errors in classification/prediction"""
    pass

class ConfigurationError(BCIError):
    """Configuration validation errors"""
    pass

class ModelError(BCIError):
    """Model loading/saving errors"""
    pass
```

### Error Handling Utilities

```python
def handle_graceful_shutdown(system: BCISystem, error: Exception):
    """
    Handle graceful system shutdown on critical errors
    
    Args:
        system: BCI system instance
        error: Exception that triggered shutdown
    """

def log_error(error: Exception, context: dict = None):
    """
    Log error with context information
    
    Args:
        error: Exception to log
        context: Additional context information
    """

def validate_system_health() -> dict:
    """
    Validate overall system health
    
    Returns:
        dict: Health status and recommendations
    """
```

---

## Usage Examples

### Basic System Usage

```python
# Initialize and start BCI system
from dependencies.BCISystem import BCISystem

# Create system instance
bci = BCISystem()

# Set data source
bci.set_data_source('file', filepath='data/raw/session_1/file.csv')

# Start calibration
success = bci.start_calibration(visualize=True)

if success:
    # Start real-time processing
    bci.start_processing(visualize=True)
    
    # Get classification results
    while True:
        result = bci.get_classification_result()
        if result:
            print(f"Class: {result['class']}, Confidence: {result['confidence']}")
```

### Custom Signal Processing

```python
# Create custom signal processor
from dependencies.signal_processor import SignalProcessor

processor = SignalProcessor(sample_rate=250, channels=8)

# Process EEG data
eeg_window = np.random.randn(500, 8)  # 2 seconds at 250 Hz
timestamps = np.arange(500) / 250.0

result = processor.process(eeg_window, timestamps)

if result['valid']:
    features = result['features']
    print(f"CSP features: {features['csp_features']}")
    print(f"Band powers: {features['band_powers']}")
```

### Model Training and Evaluation

```python
# Train and evaluate model
from dependencies.classifier import Classifier
import numpy as np

# Create classifier
clf = Classifier(classifier_type='rf')

# Train on feature data
X_train = np.random.randn(1000, 12)  # 1000 samples, 12 features
y_train = np.random.randint(0, 2, 1000)  # Binary labels

results = clf.train(X_train, y_train)
print(f"Training accuracy: {results['accuracy']}")

# Predict new data
X_test = np.random.randn(100, 12)
predictions = clf.predict(X_test)
print(f"Predictions: {predictions['class']}")
print(f"Confidences: {predictions['confidence']}")
```

This API reference provides comprehensive documentation for all major components of the BCI system, enabling developers to extend, integrate, and customize the system for their specific needs. 