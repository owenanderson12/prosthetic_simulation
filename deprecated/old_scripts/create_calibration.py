#!/usr/bin/env python3
import numpy as np
import os

# Create minimal calibration data that works with the system
save_data = {
    'baseline_data': np.random.randn(1000, 8),  # Dummy baseline data
    'left_data': np.array([np.random.randn(500, 8) for _ in range(3)], dtype=object),
    'right_data': np.array([np.random.randn(500, 8) for _ in range(3)], dtype=object),
    'left_features': np.random.randn(10, 4),  # Dummy features
    'right_features': np.random.randn(10, 4)
}

# Save to calibration directory
os.makedirs('calibration', exist_ok=True)
np.savez('calibration/best_model.npz', **save_data)
print('Created minimal calibration file: calibration/best_model.npz') 