import logging
import time
import threading
import os
import sys
import numpy as np
from typing import Dict, Optional, Union, Callable
from pylsl import StreamInfo, StreamOutlet

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class SimulationInterface:
    """
    Interface to the prosthetic hand simulation/visualization.
    
    This module handles:
    - Sending control commands to the prosthetic simulation
    - Streaming data via LSL for real-time control
    - Optional visualization of hand movements
    - Support for both direct LSL streaming and file-based input for testing
    """
    
    def __init__(self, config_dict: Dict):
        """
        Initialize the simulation interface.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.config = config_dict
        self.outlet = None
        self.stream_created = False
        
        # Simulation parameters
        self.hand_open_close_speed = config_dict.get('HAND_OPEN_CLOSE_SPEED', 0.1)
        self.wrist_rotation_speed = config_dict.get('WRIST_ROTATION_SPEED', 0.1)
        self.command_smoothing = config_dict.get('COMMAND_SMOOTHING', 0.7)
        
        # State tracking
        self.current_hand_state = 0.5  # 0 = closed, 1 = open, 0.5 = neutral
        self.current_wrist_state = 0.5  # 0 = left, 1 = right, 0.5 = neutral
        
        # Visualization callback (optional, to be set by external visualizer)
        self.visualization_callback = None
        
        # Update thread
        self._update_thread = None
        self._stop_event = threading.Event()
        self._commands_queue = []
        
        logging.info("Simulation interface initialized")
    
    def create_stream(self) -> bool:
        """
        Create an LSL stream for sending hand control commands.
        
        Returns:
            Success indicator
        """
        try:
            # Stream configuration
            stream_name = self.config.get('UNITY_OUTPUT_STREAM', 'ProstheticControl')
            stream_type = 'Control'
            channel_count = 4  # [hand_state, wrist_state, command_type, confidence]
            sampling_rate = 0  # Irregular rate
            channel_format = 'float32'
            source_id = 'NeuroBCIProsthetic'
            
            # Create stream info
            info = StreamInfo(
                name=stream_name,
                type=stream_type,
                channel_count=channel_count,
                nominal_srate=sampling_rate,
                channel_format=channel_format,
                source_id=source_id
            )
            
            # Add metadata
            info.desc().append_child_value("manufacturer", "NeurEx")
            channels = info.desc().append_child("channels")
            channels.append_child("channel").append_child_value("label", "hand_state")
            channels.append_child("channel").append_child_value("label", "wrist_state")
            channels.append_child("channel").append_child_value("label", "command_type")
            channels.append_child("channel").append_child_value("label", "confidence")
            
            # Create outlet
            self.outlet = StreamOutlet(info)
            self.stream_created = True
            
            logging.info(f"LSL stream '{stream_name}' created")
            return True
            
        except Exception as e:
            logging.exception("Error creating LSL stream:")
            return False
    
    def start_update_thread(self) -> bool:
        """
        Start the simulation update thread.
        
        Returns:
            Success indicator
        """
        if self._update_thread is not None and self._update_thread.is_alive():
            logging.warning("Update thread already running")
            return False
            
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        logging.info("Simulation update thread started")
        return True
    
    def stop_update_thread(self) -> None:
        """Stop the simulation update thread."""
        if self._update_thread is not None:
            self._stop_event.set()
            self._update_thread.join(timeout=1.0)
            self._update_thread = None
            logging.info("Simulation update thread stopped")
    
    def _update_loop(self) -> None:
        """Update thread main loop for smooth command execution."""
        update_interval = 1.0 / self.config.get('SIMULATION_UPDATE_RATE', 60)
        
        while not self._stop_event.is_set():
            # Process any queued commands
            if self._commands_queue:
                cmd = self._commands_queue.pop(0)
                self._process_command(cmd)
            
            # Update any active visualizations
            if self.visualization_callback:
                try:
                    self.visualization_callback({
                        'hand_state': self.current_hand_state,
                        'wrist_state': self.current_wrist_state
                    })
                except Exception as e:
                    logging.error(f"Error in visualization callback: {e}")
            
            # Sleep for the desired update interval
            time.sleep(update_interval)
    
    def _process_command(self, command: Dict) -> None:
        """
        Process a command by updating the hand/wrist state.
        
        Args:
            command: Command dictionary containing class, confidence, etc.
        """
        class_name = command.get('class', 'idle')
        confidence = command.get('confidence', 0.0)
        
        # Skip processing if confidence is too low
        if confidence < self.config.get('MIN_CONFIDENCE', 0.55):
            return
            
        # Determine command type
        cmd_type = 0  # 0 = idle
        
        if class_name == 'left':
            # Left class = hand open/close
            cmd_type = 1
            # Update hand state
            target = 1.0 if self.current_hand_state < 0.5 else 0.0
            # Apply smoothing
            self.current_hand_state = (1 - self.command_smoothing) * target + self.command_smoothing * self.current_hand_state
            
        elif class_name == 'right':
            # Right class = wrist rotation
            cmd_type = 2
            # Update wrist state
            target = 1.0 if self.current_wrist_state < 0.5 else 0.0
            # Apply smoothing
            self.current_wrist_state = (1 - self.command_smoothing) * target + self.command_smoothing * self.current_wrist_state
        
        # Send to LSL stream if available
        if self.stream_created and self.outlet:
            self.outlet.push_sample([
                self.current_hand_state,
                self.current_wrist_state,
                float(cmd_type),
                float(confidence)
            ])
    
    def send_command(self, classification_result: Dict) -> bool:
        """
        Send a command based on a classification result.
        
        Args:
            classification_result: Result from the classifier including class and confidence
            
        Returns:
            Success indicator
        """
        try:
            # Queue the command for processing
            self._commands_queue.append(classification_result)
            
            # Direct processing if update thread not running
            if self._update_thread is None or not self._update_thread.is_alive():
                self._process_command(classification_result)
            
            return True
            
        except Exception as e:
            logging.exception("Error sending command:")
            return False
    
    def set_visualization_callback(self, callback: Callable) -> None:
        """
        Set a callback function for visualization updates.
        
        Args:
            callback: Function to call with state updates
        """
        self.visualization_callback = callback
        logging.info("Visualization callback set")
    
    def reset_state(self) -> None:
        """Reset the simulation state to neutral positions."""
        self.current_hand_state = 0.5
        self.current_wrist_state = 0.5
        
        # Send neutral state to outlet if available
        if self.stream_created and self.outlet:
            self.outlet.push_sample([0.5, 0.5, 0.0, 0.0])
        
        logging.info("Simulation state reset")
    
    def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        self.stop_update_thread()
        self.outlet = None
        self.stream_created = False
        logging.info("Simulation interface disconnected")
