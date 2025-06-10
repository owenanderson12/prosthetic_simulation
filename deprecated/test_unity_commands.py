#!/usr/bin/env python3
"""
Test script to send manual hand commands to Unity via LSL.
This bypasses the BCI system to test if Unity receives LSL data properly.
"""

from pylsl import StreamInfo, StreamOutlet
import time

def test_unity_commands():
    print("Creating LSL stream 'ProstheticControl'...")
    
    # Create the same stream as the BCI system
    info = StreamInfo('ProstheticControl', 'Control', 4, 30, 'float32', 'TestBCI')
    outlet = StreamOutlet(info)
    
    print("Stream created. Starting Unity and watch for hand movements...")
    print("Sending test commands every 2 seconds:")
    print("Format: [hand_state, wrist_state, command_type, confidence]")
    print("Command types: 0=idle, 1=hand, 2=wrist")
    print("-" * 60)
    
    try:
        cycle = 0
        while True:
            cycle += 1
            
            if cycle % 3 == 1:
                # Hand open command
                command = [1.0, 0.5, 1, 0.8]  # Hand open, neutral wrist, hand command, high confidence
                print(f"Cycle {cycle}: HAND OPEN   - {command}")
                
            elif cycle % 3 == 2:
                # Hand close command  
                command = [0.0, 0.5, 1, 0.8]  # Hand closed, neutral wrist, hand command, high confidence
                print(f"Cycle {cycle}: HAND CLOSE  - {command}")
                
            else:
                # Idle command
                command = [0.5, 0.5, 0, 0.5]  # Neutral position, idle command, medium confidence
                print(f"Cycle {cycle}: IDLE        - {command}")
            
            outlet.push_sample(command)
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nTest stopped.")
    
    print("Cleaning up...")

if __name__ == "__main__":
    test_unity_commands() 