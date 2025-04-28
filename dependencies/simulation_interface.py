from pylsl import StreamInfo, StreamOutlet
import numpy as np
import logging

class SimulationInterface:
    def __init__(self, config=None):
        """Initialize the simulation interface with configuration."""
        self.config = config or {}
        self.outlet = None
        self.stream_created = False
        
    def createStream(self):
        """Create an LSL stream for sending hand gesture predictions."""
        try:
            # --- Configuration ---
            stream_name = self.config.get('LSL_STREAM_NAME', 'HandGesturePredictions')
            stream_type = self.config.get('LSL_STREAM_TYPE', 'Markers')
            channel_count = self.config.get('LSL_CHANNEL_COUNT', 2)  # State and confidence
            sampling_rate = self.config.get('LSL_SAMPLING_RATE', 0)  # Irregular rate for event markers
            channel_format = self.config.get('LSL_CHANNEL_FORMAT', 'float32')  # For probabilities
            source_id = self.config.get('LSL_SOURCE_ID', 'MyMLClassifier')

            # --- Stream Info and Outlet ---
            info = StreamInfo(stream_name, stream_type, channel_count, sampling_rate, channel_format, source_id)
            self.outlet = StreamOutlet(info)
            self.stream_created = True

            logging.info(f"LSL Outlet '{stream_name}' started.")
            return True
        except Exception as e:
            logging.error(f"Failed to create LSL stream: {e}")
            return False

    def send_command(self, state, confidence):
        """Send a command to the LSL stream.
        
        Args:
            state: The classified state (e.g., 'left', 'right', 'idle')
            confidence: The confidence value of the classification
        """
        if not self.stream_created or self.outlet is None:
            logging.warning("LSL stream not created. Call createStream() first.")
            return False
            
        try:
            # Convert state to a numeric value for easier processing in Unity
            # You can adjust this mapping based on your Unity script's expectations
            state_value = state  # Default/idle
            #if state == 'left':
            #    state_value = 1
            #elif state == 'right':
            #    state_value = 2
                
            # Send the data to the LSL stream
            # Format: [state_value, confidence]
            self.outlet.push_sample([float(state_value), float(confidence)])
            
            logging.debug(f"Sent command to LSL: state={state_value}, confidence={confidence:.2f}")
            return True
        except Exception as e:
            logging.error(f"Failed to send command to LSL stream: {e}")
            return False
            
    def disconnect(self):
        """Clean up the LSL stream resources."""
        # No explicit disconnect needed for pylsl, but we can clean up references
        self.outlet = None
        self.stream_created = False
        logging.info("LSL stream disconnected.")
