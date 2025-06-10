#!/usr/bin/env python3
"""
Simple LSL stream test to verify what data is being sent to Unity.
"""

import time
import numpy as np
from pylsl import StreamInlet, resolve_stream

def test_lsl_receiver():
    print("Looking for LSL stream 'ProstheticControl'...")
    
    # Resolve the stream
    streams = resolve_stream('name', 'ProstheticControl')
    
    if not streams:
        print("No 'ProstheticControl' stream found!")
        print("Make sure the BCI system is running.")
        return
    
    print(f"Found stream: {streams[0].name()}")
    
    # Create inlet
    inlet = StreamInlet(streams[0])
    
    print("Listening for data (press Ctrl+C to stop)...")
    print("Format: [hand_state, wrist_state, command_type, confidence]")
    print("Command types: 0=idle, 1=hand, 2=wrist")
    print("-" * 60)
    
    try:
        while True:
            # Pull sample
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            
            if sample:
                hand_state = sample[0]
                wrist_state = sample[1] 
                command_type = int(sample[2])
                confidence = sample[3]
                
                cmd_name = ["IDLE", "HAND", "WRIST"][command_type] if command_type <= 2 else "UNKNOWN"
                
                print(f"Time: {timestamp:.3f} | Hand: {hand_state:.2f} | Wrist: {wrist_state:.2f} | "
                      f"Cmd: {cmd_name} ({command_type}) | Conf: {confidence:.2f}")
                      
            else:
                print(".", end="", flush=True)
                
    except KeyboardInterrupt:
        print("\nStopped.")
    
    inlet.close_stream()

if __name__ == "__main__":
    test_lsl_receiver() 