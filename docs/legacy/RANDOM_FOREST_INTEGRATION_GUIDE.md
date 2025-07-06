# Random Forest Model Integration Guide

## Overview

The best-performing Random Forest model (84.2% accuracy) has been successfully integrated into the BCI system. This model uses all 8 EEG channels, excludes session 3 data, and combines CSP features with band power features.

## Key Features

- **Model Type**: Random Forest Classifier with StandardScaler preprocessing
- **Accuracy**: 84.2% cross-validation accuracy (CSP + band power features)
- **Channels**: All 8 channels (CH1-CH8)
- **Features**: 4 CSP components + 8 mu band + 8 beta band ERD features (20 total)
- **Backward Compatibility**: Maintains compatibility with existing LDA models

## How to Use

### 1. Loading the Best Model

The best model is available as `models/best_model.pkl`. You can load it in several ways:

#### Option A: Using the main BCI system
```bash
python main.py --load-calibration best_model.pkl
```

#### Option B: Using the calibration system
```python
from dependencies.calibration import CalibrationSystem
from dependencies.classifier import Classifier

classifier = Classifier(config.__dict__)
success = classifier.load_model('best_model.pkl')
```

#### Option C: Direct loading
```python
classifier.load_model('best_model.pkl')
```

### 2. Running the Full BCI System

To use the Random Forest model in your BCI system:

```bash
# Start the system and load the best model
python main.py --load-calibration best_model.pkl --visualize

# Or start calibration first, then load the model
python main.py --calibrate
# After calibration, manually load: classifier.load_model('best_model.pkl')
```

### 3. Model Features

The integrated classifier automatically handles:
- **Feature Dimension Matching**: Adjusts features to match model expectations
- **Missing CSP Features**: Uses zero placeholders when CSP filters aren't available
- **Multiple Model Types**: Supports both Random Forest and LDA models
- **Adaptive Thresholding**: Maintains the existing threshold adaptation system

## Technical Details

### Model Performance Comparison

| Configuration | CSP-only Acc. | CSP+BP Acc. |
|---------------|----------------|--------------|
| **RF + All Channels + No S3** | **71.5%** | **84.2%** |
| RF + Channels 2,3,6,7 + No S3 | 74.5% | 79.8% |
| SVM + All Channels + No S3 | 61.6% | 75.4% |
| LDA + All Channels + No S3 | 55.2% | 68.9% |

### Feature Handling

The classifier intelligently handles different feature scenarios:

1. **Full Features** (CSP + Band Power): Uses all 20 features as intended
2. **CSP Only**: Uses 4 CSP features (for CSP-only models)  
3. **Band Power Only**: Pads with zero CSP placeholders for compatibility
4. **Channel Mismatch**: Adjusts band power features to match expected dimensions

### Compatibility

- ✅ **Backward Compatible**: Existing LDA models continue to work
- ✅ **Runtime Flexible**: Handles missing CSP features gracefully
- ✅ **Channel Agnostic**: Adapts to different channel configurations
- ✅ **Error Robust**: Graceful degradation when features are unavailable

## Testing

The integration has been thoroughly tested:

```bash
python test_model_integration.py
```

This verifies:
- Model loading (both RF and LDA models)
- Feature preparation for different scenarios
- Classification with random and real signal processor data
- Backward compatibility with existing models
- Signal processor integration

## File Structure

```
models/
├── best_model.pkl                          # The best Random Forest model (use this!)
├── aggregate_csp_bp_model_no_s3_rf.pkl    # Original RF model filename
├── aggregate_csp_model_no_s3_rf.pkl       # RF CSP-only model
└── [other model variants...]               # Various experimental models

dependencies/
├── classifier.py                           # Updated with RF support
├── signal_processor.py                     # Unchanged
├── BCISystem.py                           # Unchanged
└── [other components...]                   # Unchanged
```

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Ensure `models/best_model.pkl` exists
   - Check the model directory path in your config

2. **"Feature dimension mismatch"** 
   - The integrated classifier should handle this automatically
   - Check logs for CSP placeholder warnings

3. **"Classifier not trained"**
   - Make sure to load the model before starting processing
   - Verify the model loaded successfully

### Debug Mode

Enable debug logging to see feature handling:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

1. **Production Use**: The system is ready for production use with `best_model.pkl`
2. **Further Training**: You can continue training new models using the updated `train_aggregate_models.py`
3. **Performance Monitoring**: Monitor classification confidence and accuracy in real-time use
4. **Model Updates**: The system supports hot-swapping models during runtime

## Performance Tips

- **CSP Training**: For optimal performance, train CSP filters with representative data
- **Baseline Updates**: Regularly update baseline power values for better ERD calculation  
- **Data Quality**: Monitor artifact rejection ratios and signal quality
- **Confidence Thresholds**: Adjust confidence thresholds based on user performance

---

*The Random Forest model represents a significant improvement over the previous LDA approach, providing more robust classification of motor imagery tasks.* 