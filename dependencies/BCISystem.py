import os
import sys
import time
import argparse
import logging
import threading
import json
from typing import Dict, Any

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import system configuration
import config

# Add dependencies directory to the path
DEPENDENCIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dependencies')
sys.path.append(DEPENDENCIES_DIR)

# Import module components
from dependencies.data_source import DataSource
from dependencies.signal_processor import SignalProcessor
from dependencies.classifier import Classifier
from dependencies.calibration import CalibrationSystem, CalibrationStage

class BCISystem:
    """
    Main BCI system class that orchestrates all components.
    """
    
    def __init__(self, config_dict: Dict[str, Any], source_type: str = "live"):
        """
        Initialize the BCI system.
        
        Args:
            config_dict: Configuration dictionary
            source_type: EEG data source type ('live' or 'artificial')
        """
        self.config = config_dict
        self.source_type = source_type
        self._setup_logging()
        
        # System state
        self.running = False
        self.calibration_mode = False
        self.processing_enabled = False
        self._stop_event = threading.Event()
        self._processing_thread = None
        
        # Initialize components
        logging.info("Initializing BCI system components...")
        self.eeg_acquisition = None
        self.signal_processor = None
        self.classifier = None
        self.calibration = None
        self.simulation = None
        self.visualization = None
        
        # Initialize all modules
        self._initialize_modules()
        
        logging.info("BCI system initialized")
    
    def _setup_logging(self) -> None:
        """Configure the logging system."""
        log_level = getattr(logging, self.config.get('LOG_LEVEL', 'INFO'))
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(self.config.get('LOG_FILE', 'bci_system.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _initialize_modules(self) -> None:
        """Initialize all system modules."""
        try:
            # EEG Acquisition (via unified DataSource interface)
            self.eeg_acquisition = DataSource(source_type=self.source_type)
            
            # Signal Processor
            self.signal_processor = SignalProcessor(self.config)
            
            # Classifier
            self.classifier = Classifier(self.config)
            
            # Calibration System
            self.calibration = CalibrationSystem(
                self.config,
                self.eeg_acquisition,
                self.signal_processor,
                self.classifier
            )
            
            # Simulation Interface (to be implemented)
            # self.simulation = SimulationInterface(self.config)
            
            # Visualization (to be implemented)
            # self.visualization = Visualization(self.config)
            
        except Exception as e:
            logging.exception("Error initializing modules:")
            sys.exit(1)
    
    def start(self) -> bool:
        """
        Start the BCI system.
        
        Returns:
            Success indicator
        """
        if self.running:
            logging.warning("System already running")
            return False
        
        try:
            # For DataSource, no explicit connect/start needed; handled internally if needed
            # (If your DataSource or EEGAcquisition requires explicit connection logic, consider adding it to DataSource)
            
            # Start simulation interface if available
            # if self.simulation:
            #     self.simulation.connect()
            
            # Reset stop event
            self._stop_event.clear()
            
            # Set system as running
            self.running = True
            
            logging.info("BCI system started")
            return True
            
        except Exception as e:
            logging.exception("Error starting BCI system:")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop the BCI system and all components."""
        if not self.running:
            return
            
        logging.info("Stopping BCI system...")
        
        # Signal processing thread to stop
        self._stop_event.set()
        
        # Stop processing if running
        self.stop_processing()
        
        # Stop calibration if running
        if self.calibration_mode:
            self.calibration.stop_calibration()
            self.calibration_mode = False
        
        # Stop EEG acquisition
        if self.eeg_acquisition:
            self.eeg_acquisition.stop_background_update()
            self.eeg_acquisition.disconnect()
        
        # Stop simulation if running
        # if self.simulation:
        #     self.simulation.disconnect()
        
        self.running = False
        logging.info("BCI system stopped")
    
    def start_calibration(self, callback=None) -> bool:
        """
        Start the calibration procedure.
        
        Args:
            callback: Function to call for UI updates
            
        Returns:
            Success indicator
        """
        if self.calibration_mode:
            logging.warning("Calibration already running")
            return False
        
        if not self.running:
            if not self.start():
                logging.error("Could not start the BCI system for calibration")
                return False
        
        # Give the EEG acquisition time to collect some initial data
        time.sleep(2.0)
        
        if self.calibration.start_calibration(callback):
            self.calibration_mode = True
            logging.info("Calibration started")
            return True
        else:
            logging.error("Failed to start calibration")
            return False
    
    def load_calibration(self, filename: str) -> bool:
        """
        Load a saved calibration.
        
        Args:
            filename: Name of the calibration file to load
            
        Returns:
            Success indicator
        """
        return self.calibration.load_calibration(filename)
    
    def start_processing(self) -> bool:
        """
        Start the main BCI processing loop.
        
        Returns:
            Success indicator
        """
        if self.processing_enabled:
            logging.warning("Processing already running")
            return False
            
        if not self.running:
            if not self.start():
                logging.error("Could not start the BCI system for processing")
                return False
        
        # Check if we have a trained classifier
        if self.classifier.classifier is None:
            logging.error("Cannot start processing: No trained classifier")
            return False
            
        # Start processing in a separate thread
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        self.processing_enabled = True
        logging.info("BCI processing started")
        return True
    
    def stop_processing(self) -> None:
        """Stop the main BCI processing loop."""
        if not self.processing_enabled:
            return
            
        self.processing_enabled = False
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
            
        logging.info("BCI processing stopped")
    
    def _processing_loop(self) -> None:
        """Main BCI processing loop running in a separate thread."""
        logging.info("Processing loop started")
        
        try:
            while not self._stop_event.is_set() and self.processing_enabled:
                # Get the latest EEG data
                window_size = int(self.config.get('WINDOW_SIZE', 2.0) * self.config.get('SAMPLE_RATE', 250))
                data, timestamps = self.eeg_acquisition.get_chunk(window_size=window_size)
                
                if data.size == 0:
                    time.sleep(0.1)  # No data available, wait briefly
                    continue
                
                # Process the data
                processing_result = self.signal_processor.process(data, timestamps)
                
                # Skip if processing failed
                if not processing_result['valid']:
                    logging.debug(f"Invalid processing result: {processing_result.get('message', 'Unknown error')}")
                    time.sleep(0.05)
                    continue
                
                # Classify the processed data
                classification_result = self.classifier.classify(processing_result['features'])
                
                # Skip if classification failed
                if not classification_result['valid']:
                    logging.debug(f"Invalid classification result: {classification_result.get('message', 'Unknown error')}")
                    time.sleep(0.05)
                    continue
                
                # Use the classification result to control the simulation
                self._handle_classification(classification_result)
                
                # Brief sleep to prevent CPU overload
                time.sleep(0.01)
                
        except Exception as e:
            logging.exception("Error in processing loop:")
            self.processing_enabled = False
    
    def _handle_classification(self, classification_result: Dict) -> None:
        """
        Handle the classification result and send commands to the simulation.
        
        Args:
            classification_result: Classification result from the classifier
        """
        # Extract the classified state and confidence
        state = classification_result['class']
        confidence = classification_result['confidence']
        
        # Log the classification at debug level
        logging.debug(f"Classification: {state}, Confidence: {confidence:.2f}")
        
        # When simulation interface is implemented, send the command there
        # if self.simulation:
        #     self.simulation.send_command(state, confidence)
        
        # For now, just print the command to console if above threshold
        if state != 'idle' and confidence > self.config.get('CLASSIFIER_THRESHOLD', 0.65):
            command = "OPEN/CLOSE" if state == 'left' else "ROTATE"
            print(f"HAND COMMAND: {command} (confidence: {confidence:.2f})")