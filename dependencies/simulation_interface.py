import logging
import time
import threading
import os
import sys
import numpy as np
from typing import Dict, Optional, Union, Callable
from pylsl import StreamInfo, StreamOutlet, resolve_stream

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
        self.unity_connected = False
        
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
    
    def wait_for_unity_connection(self, timeout: float = 30.0, check_ready_stream: bool = True) -> bool:
        """
        Wait for Unity to connect to the LSL stream.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_ready_stream: Whether to check for a Unity ready signal stream
            
        Returns:
            True if Unity connected, False if timeout
        """
        if not self.stream_created:
            logging.error("LSL stream not created. Call create_stream() first.")
            return False
            
        logging.info("Waiting for Unity to connect...")
        print("\n========================================")
        print("Waiting for Unity application to connect...")
        print("Please start your Unity prosthetic hand simulation.")
        print("========================================\n")
        
        start_time = time.time()
        
        # First, optionally check for Unity ready stream
        if check_ready_stream:
            logging.info("Checking for Unity ready signal...")
            ready_check_interval = 1.0
            ready_timeout = min(10.0, timeout / 2)  # Check for ready signal for half the timeout
            
            while time.time() - start_time < ready_timeout:
                try:
                    # Look for a Unity ready signal stream
                    unity_ready_streams = resolve_stream('name', 'UnityReady', timeout=0.5)
                    if unity_ready_streams:
                        logging.info("Unity ready signal detected!")
                        self.unity_connected = True
                        print("✓ Unity application detected and ready!")
                        return True
                except Exception as e:
                    logging.debug(f"Error checking for Unity ready stream: {e}")
                
                # Show progress
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                print(f"\rWaiting for Unity... {remaining:.1f}s remaining", end="", flush=True)
                time.sleep(ready_check_interval)
        
        # Check if outlet has consumers (Unity connected as inlet)
        logging.info("Checking for outlet consumers...")
        consumer_check_interval = 0.5
        
        while time.time() - start_time < timeout:
            try:
                # Check if outlet has consumers
                if self.outlet and self.outlet.have_consumers():
                    logging.info("Unity consumer detected on LSL stream!")
                    self.unity_connected = True
                    print("\n✓ Unity connected to ProstheticControl stream!")
                    
                    # Send initial handshake/test signal
                    self._send_handshake()
                    return True
                    
            except Exception as e:
                logging.debug(f"Error checking outlet consumers: {e}")
            
            # Show progress
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            print(f"\rWaiting for Unity connection... {remaining:.1f}s remaining", end="", flush=True)
            
            time.sleep(consumer_check_interval)
        
        # Timeout reached
        print("\n✗ Timeout waiting for Unity connection.")
        logging.warning(f"Unity did not connect within {timeout} seconds")
        
        # Ask user if they want to continue anyway
        print("\nWould you like to:")
        print("1. Continue without Unity (commands will be sent but not received)")
        print("2. Wait longer")
        print("3. Exit")
        
        try:
            choice = input("\nEnter choice (1/2/3): ").strip()
            if choice == '1':
                logging.info("Continuing without Unity connection")
                self.unity_connected = False
                return True
            elif choice == '2':
                # Recursive call with same timeout
                return self.wait_for_unity_connection(timeout, check_ready_stream)
            else:
                return False
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return False
    
    def _send_handshake(self) -> None:
        """Send a handshake signal to Unity to confirm connection."""
        try:
            # Send a special handshake command
            # command_type = -1 indicates handshake
            self.outlet.push_sample([0.5, 0.5, -1.0, 1.0])
            logging.info("Handshake signal sent to Unity")
        except Exception as e:
            logging.error(f"Error sending handshake: {e}")
    
    def is_unity_connected(self) -> bool:
        """
        Check if Unity is currently connected.
        
        Returns:
            True if Unity is connected, False otherwise
        """
        if not self.stream_created or not self.outlet:
            return False
            
        try:
            # Check if outlet has consumers
            has_consumers = self.outlet.have_consumers()
            
            # Update connection status
            if has_consumers and not self.unity_connected:
                logging.info("Unity connection detected")
                self.unity_connected = True
            elif not has_consumers and self.unity_connected:
                logging.warning("Unity connection lost")
                self.unity_connected = False
                
            return has_consumers
            
        except Exception as e:
            logging.error(f"Error checking Unity connection: {e}")
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
        connection_check_interval = 2.0  # Check connection every 2 seconds
        last_connection_check = time.time()
        
        while not self._stop_event.is_set():
            # Periodically check Unity connection
            if time.time() - last_connection_check > connection_check_interval:
                self.is_unity_connected()
                last_connection_check = time.time()
            
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
            
            # Log warning if Unity not connected
            if not self.unity_connected and cmd_type != 0:
                logging.debug("Command sent but Unity may not be connected")
    
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
        self.unity_connected = False
        logging.info("Simulation interface disconnected")
