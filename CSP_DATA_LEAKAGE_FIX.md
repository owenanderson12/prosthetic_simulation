# CSP Data Leakage Fix - Corrected BCI Model Training

## Problem Identified

The original `scripts/training/train_model.py` contained a critical **data leakage issue** that severely inflated test accuracies.

### The Issue
```python
# WRONG: Original problematic code
sp.train_csp(left_trials, right_trials)  # Uses ALL trials including test data
left_features = extract_features(sp, left_trials)
right_features = extract_features(sp, right_trials)
# ... then split features into train/test
```

**Problem**: CSP spatial filters were trained on ALL data before train/test split, meaning:
1. CSP covariance matrices included test data
2. Spatial filters "saw" the test data during training
3. Test set evaluation used contaminated features
4. This inflated test accuracy by ~34%

## Solution Implemented

### Fixed Methodology
```python
# CORRECT: Fixed approach
# 1. Split trials FIRST
train_left, test_left = train_test_split(left_trials, test_size=0.2, random_state=42)
train_right, test_right = train_test_split(right_trials, test_size=0.2, random_state=42)

# 2. Train CSP on training trials ONLY
sp.train_csp(train_left, train_right)

# 3. Extract features using fitted CSP
train_features = extract_features(sp, np.vstack([train_left, train_right]))
test_features = extract_features(sp, np.vstack([test_left, test_right]))
```

### Key Changes Made
1. **Split trials before CSP training** - prevents test data leakage
2. **Train CSP only on training data** - ensures spatial filters are unbiased
3. **Apply fitted CSP to extract features** - maintains proper train/test separation
4. **Verified no other preprocessing leakage** - StandardScaler properly isolated

## Results Comparison

### Before Fix (Data Leakage Present)
- **Best model accuracy: 99.7%** ⚠️
- aggregate_csp_bp_model_rf: **99.3%** ⚠️
- Training accuracy: **98.8%** (huge gap from CV: 72.2%) ⚠️

### After Fix (Corrected Methodology) ✅
- **Cross-validation accuracy: 73.2% ± 5.0%**
- **Validation accuracy: 81.6%**
- **Test accuracy: 65.5%** (uncontaminated)

### Impact Analysis
- **Data leakage inflated accuracy by ~34%** (99.7% → 65.5%)
- The old CV accuracy (~72%) was actually realistic
- Corrected 65.5% test accuracy is realistic for 8-channel EEG BCI systems
- Eliminated the suspicious gap between CV and training accuracy

## Validation of Fix

### Evidence the Fix is Correct
1. **Realistic performance**: 65.5% aligns with published BCI literature for 8-channel systems
2. **Consistent metrics**: CV (73.2%) and validation (81.6%) are in reasonable range
3. **Proper methodology**: Trials split before any feature extraction
4. **No remaining leakage**: StandardScaler isolated to training data only

### Technical Details
- **CSP normalization**: Mean/std computed from training data only, applied to all data
- **Feature extraction**: Sliding window approach maintained
- **Cross-validation**: Performed on training data only (proper practice)
- **Test set**: Completely held out until final evaluation

## Methodology Explanation

### Why This Fix Matters
CSP (Common Spatial Patterns) learns optimal spatial filters by maximizing variance differences between classes. If trained on test data:
1. **Spatial filters are optimized for test patterns** → inflated test performance
2. **Covariance estimation biased** → unrealistic feature separability
3. **Model evaluation invalidated** → cannot trust performance metrics

### Correct BCI Training Pipeline
1. **Data collection**: Raw EEG trials per condition
2. **Trial-level split**: Separate trials before any processing
3. **CSP training**: Learn spatial filters on training trials only
4. **Feature extraction**: Apply fitted CSP to extract features
5. **Classifier training**: Train on extracted training features
6. **Evaluation**: Test on completely unseen trial features

### Expected Performance Ranges
- **Cross-validation**: 70-80% (realistic for motor imagery BCI)
- **Test accuracy**: 60-75% (accounts for generalization gap)
- **Clinical deployment**: Often 5-10% lower than lab conditions

## Files Modified

- `scripts/training/train_model.py`: Fixed CSP training methodology (lines 421-475)
- `CSP_DATA_LEAKAGE_FIX.md`: This documentation file

## Usage

Run corrected training:
```bash
python scripts/training/train_model.py
```

Expected realistic output:
- Cross-validation: ~73% accuracy
- Test accuracy: ~66% (uncontaminated)

## References & Best Practices

### BCI Training Guidelines
1. **Always split trials before preprocessing** when using subject-specific methods
2. **CSP must be trained on training data only** - never include test trials
3. **Validate with proper cross-validation** - temporal splits for real-world deployment
4. **Expect 60-80% accuracy** for motor imagery with <10 channels

### Common Data Leakage Sources in BCI
- ❌ CSP trained on all data (this fix)
- ❌ ICA artifact removal using all data
- ❌ Feature normalization across all data
- ❌ Temporal filtering with zero-phase (future information)

### Verification Checklist
- ✅ CSP training uses only training trials
- ✅ Feature normalization computed from training data only
- ✅ Cross-validation performed on training data
- ✅ Test accuracy is realistic (not >90% unless very clean data)
- ✅ No suspicious CV vs training accuracy gaps

---

**Summary**: This fix reduced reported accuracy from an impossible 99.7% to a realistic 65.5%, providing honest performance estimates for BCI system deployment.