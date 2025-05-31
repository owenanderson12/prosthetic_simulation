import logging
import threading
import time
import os
import sys
import numpy as np
from typing import Dict, Optional, Callable
import tkinter as tk
from tkinter import ttk

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Visualization:
    """
    Visualization module for the prosthetic hand movements.
    
    This module provides a simple graphical interface to visualize:
    - Hand open/close state
    - Wrist rotation
    - Classification confidence
    """
    
    def __init__(self, config_dict: Dict, simulation_interface=None):
        """
        Initialize the visualization module.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            simulation_interface: Optional simulation interface to connect to
        """
        self.config = config_dict
        self.simulation_interface = simulation_interface
        
        # State variables
        self.hand_state = 0.5  # 0 = closed, 1 = open, 0.5 = neutral
        self.wrist_state = 0.5  # 0 = left, 1 = right, 0.5 = neutral
        self.confidence = 0.0
        self.current_class = 'idle'
        
        # GUI variables
        self.root = None
        self.hand_canvas = None
        self.wrist_canvas = None
        self.class_label = None
        self.confidence_progressbar = None
        
        # Thread for non-blocking UI updates
        self._gui_thread = None
        self._stop_event = threading.Event()
        
        logging.info("Visualization module initialized")
    
    def start(self) -> bool:
        """
        Start the visualization in a separate thread.
        
        Returns:
            Success indicator
        """
        if self._gui_thread is not None and self._gui_thread.is_alive():
            logging.warning("Visualization already running")
            return False
            
        # Connect to simulation interface if provided
        if self.simulation_interface:
            self.simulation_interface.set_visualization_callback(self.update_state)
        
        # Start GUI thread
        self._stop_event.clear()
        self._gui_thread = threading.Thread(target=self._gui_loop)
        self._gui_thread.daemon = True
        self._gui_thread.start()
        
        logging.info("Visualization started")
        return True
    
    def stop(self) -> None:
        """Stop the visualization thread."""
        if self._gui_thread is not None:
            self._stop_event.set()
            self._gui_thread.join(timeout=1.0)
            self._gui_thread = None
            logging.info("Visualization stopped")
    
    def _gui_loop(self) -> None:
        """Main GUI thread function."""
        try:
            # Create main window
            self.root = tk.Tk()
            self.root.title("Prosthetic Hand Visualization")
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.root.geometry("600x400")
            
            # Configure styles
            style = ttk.Style()
            style.configure("TFrame", background="#f0f0f0")
            style.configure("TLabel", background="#f0f0f0", font=("Arial", 12))
            style.configure("TButton", font=("Arial", 10))
            
            # Create main frame
            main_frame = ttk.Frame(self.root, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create hand visualization
            hand_frame = ttk.LabelFrame(main_frame, text="Hand State", padding=10)
            hand_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            
            self.hand_canvas = tk.Canvas(hand_frame, width=250, height=100, bg="white")
            self.hand_canvas.pack(fill=tk.BOTH, expand=True)
            
            # Create wrist visualization
            wrist_frame = ttk.LabelFrame(main_frame, text="Wrist Rotation", padding=10)
            wrist_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
            
            self.wrist_canvas = tk.Canvas(wrist_frame, width=250, height=100, bg="white")
            self.wrist_canvas.pack(fill=tk.BOTH, expand=True)
            
            # Create classification info
            info_frame = ttk.LabelFrame(main_frame, text="Classification", padding=10)
            info_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
            
            ttk.Label(info_frame, text="Current state:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.class_label = ttk.Label(info_frame, text="idle")
            self.class_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            ttk.Label(info_frame, text="Confidence:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.confidence_progressbar = ttk.Progressbar(info_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
            self.confidence_progressbar.grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
            # Configure grid weights
            main_frame.columnconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(0, weight=2)
            main_frame.rowconfigure(1, weight=1)
            
            # Initial draw
            self._draw_hand_state()
            self._draw_wrist_state()
            
            # Start update loop
            self._update_gui()
            
            # Start Tkinter main loop
            self.root.mainloop()
            
        except Exception as e:
            logging.exception("Error in visualization GUI:")
    
    def _update_gui(self) -> None:
        """Update GUI components periodically."""
        if self.root and not self._stop_event.is_set():
            self._draw_hand_state()
            self._draw_wrist_state()
            self.class_label.config(text=self.current_class)
            self.confidence_progressbar['value'] = self.confidence * 100
            
            # Schedule next update
            self.root.after(50, self._update_gui)
    
    def _draw_hand_state(self) -> None:
        """Draw the hand open/close visualization."""
        if not self.hand_canvas:
            return
            
        # Clear canvas
        self.hand_canvas.delete("all")
        
        # Get canvas dimensions
        width = self.hand_canvas.winfo_width()
        height = self.hand_canvas.winfo_height()
        
        # Draw hand representation
        # Center rectangle represents palm
        palm_width = 60
        palm_height = 80
        palm_x = (width - palm_width) / 2
        palm_y = (height - palm_height) / 2
        
        # Draw palm
        self.hand_canvas.create_rectangle(
            palm_x, palm_y, palm_x + palm_width, palm_y + palm_height,
            fill="#FFD700", outline="black", width=2
        )
        
        # Draw fingers - angle depends on hand_state
        # 0 = closed, 1 = open
        finger_length = 40
        finger_width = 12
        finger_spacing = 15
        
        # Calculate finger openness (angle)
        # At hand_state=0, fingers are fully closed (angle=0)
        # At hand_state=1, fingers are fully open (angle=90)
        angle = self.hand_state * 70  # 0 to 70 degrees
        
        # Convert to radians
        angle_rad = np.radians(angle)
        
        # Draw 4 fingers
        for i in range(4):
            # Base position
            base_x = palm_x + (i * (finger_width + finger_spacing)) + 5
            base_y = palm_y
            
            # Calculate end position based on angle
            end_x = base_x + np.sin(angle_rad) * finger_length
            end_y = base_y - np.cos(angle_rad) * finger_length
            
            # Draw finger
            self.hand_canvas.create_rectangle(
                base_x, base_y, base_x + finger_width, base_y - 5,
                fill="#FFD700", outline="black", width=1
            )
            
            self.hand_canvas.create_line(
                base_x + finger_width/2, base_y - 2,
                end_x + finger_width/2, end_y,
                fill="black", width=finger_width, capstyle=tk.ROUND
            )
        
        # Draw thumb (opposite side)
        thumb_base_x = palm_x + palm_width
        thumb_base_y = palm_y + palm_height/2
        
        # Thumb angle is opposite to fingers
        thumb_angle_rad = np.radians(self.hand_state * -70)
        thumb_end_x = thumb_base_x + np.cos(thumb_angle_rad) * finger_length
        thumb_end_y = thumb_base_y + np.sin(thumb_angle_rad) * finger_length
        
        self.hand_canvas.create_rectangle(
            thumb_base_x - 5, thumb_base_y - finger_width/2,
            thumb_base_x, thumb_base_y + finger_width/2,
            fill="#FFD700", outline="black", width=1
        )
        
        self.hand_canvas.create_line(
            thumb_base_x - 2, thumb_base_y,
            thumb_end_x, thumb_end_y,
            fill="black", width=finger_width, capstyle=tk.ROUND
        )
        
        # Draw state label
        state_text = f"Hand: {self.hand_state:.2f}"
        self.hand_canvas.create_text(
            width/2, height-10,
            text=state_text, font=("Arial", 10)
        )
    
    def _draw_wrist_state(self) -> None:
        """Draw the wrist rotation visualization."""
        if not self.wrist_canvas:
            return
            
        # Clear canvas
        self.wrist_canvas.delete("all")
        
        # Get canvas dimensions
        width = self.wrist_canvas.winfo_width()
        height = self.wrist_canvas.winfo_height()
        
        # Draw wrist representation
        # Central circle for wrist joint
        center_x = width / 2
        center_y = height / 2
        wrist_radius = 20
        
        # Draw wrist circle
        self.wrist_canvas.create_oval(
            center_x - wrist_radius, center_y - wrist_radius,
            center_x + wrist_radius, center_y + wrist_radius,
            fill="#A0A0A0", outline="black", width=2
        )
        
        # Draw arm (fixed)
        arm_width = 40
        arm_height = 15
        self.wrist_canvas.create_rectangle(
            0, center_y - arm_height/2,
            center_x - wrist_radius, center_y + arm_height/2,
            fill="#A0A0A0", outline="black", width=1
        )
        
        # Calculate hand rotation angle
        # At wrist_state=0, hand is rotated left (-60 degrees)
        # At wrist_state=1, hand is rotated right (60 degrees)
        # At wrist_state=0.5, hand is neutral (0 degrees)
        rotation_angle = (self.wrist_state - 0.5) * 120  # -60 to 60 degrees
        
        # Convert to radians
        rotation_rad = np.radians(rotation_angle)
        
        # Draw hand part (rotates)
        hand_length = 80
        hand_width = 30
        
        # Calculate rotated hand rectangle coordinates
        # Starting point is the wrist center
        hand_end_x = center_x + np.cos(rotation_rad) * hand_length
        hand_end_y = center_y + np.sin(rotation_rad) * hand_length
        
        # Calculate perpendicular points for hand width
        perp_rad = rotation_rad + np.pi/2
        half_width = hand_width / 2
        perp_x = np.cos(perp_rad) * half_width
        perp_y = np.sin(perp_rad) * half_width
        
        # Define hand polygon
        hand_points = [
            center_x + perp_x, center_y + perp_y,  # Top left
            hand_end_x + perp_x, hand_end_y + perp_y,  # Top right
            hand_end_x - perp_x, hand_end_y - perp_y,  # Bottom right
            center_x - perp_x, center_y - perp_y  # Bottom left
        ]
        
        # Draw hand
        self.wrist_canvas.create_polygon(
            hand_points, fill="#FFD700", outline="black", width=1
        )
        
        # Draw indicator line inside wrist
        line_length = wrist_radius * 0.8
        line_end_x = center_x + np.cos(rotation_rad) * line_length
        line_end_y = center_y + np.sin(rotation_rad) * line_length
        
        self.wrist_canvas.create_line(
            center_x, center_y, line_end_x, line_end_y,
            fill="black", width=3, arrow=tk.LAST
        )
        
        # Draw state label
        state_text = f"Wrist: {self.wrist_state:.2f}"
        self.wrist_canvas.create_text(
            width/2, height-10,
            text=state_text, font=("Arial", 10)
        )
    
    def update_state(self, state_dict: Dict) -> None:
        """
        Update the visualization state.
        
        Args:
            state_dict: Dictionary containing state values
        """
        if 'hand_state' in state_dict:
            self.hand_state = state_dict['hand_state']
        
        if 'wrist_state' in state_dict:
            self.wrist_state = state_dict['wrist_state']
        
        if 'confidence' in state_dict:
            self.confidence = state_dict['confidence']
        
        if 'class' in state_dict:
            self.current_class = state_dict['class']
    
    def _on_close(self) -> None:
        """Handle window close event."""
        self._stop_event.set()
        if self.root:
            self.root.destroy()
            self.root = None
