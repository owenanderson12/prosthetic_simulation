# BCI System Architecture Documentation

## Overview

This document provides a comprehensive technical overview of the BCI-Controlled Prosthetic System architecture. The system is designed as a modular, extensible platform for motor imagery-based brain-computer interfaces with comprehensive training, evaluation, and real-time operation capabilities.

## System Architecture Layers

### 1. Application Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   Training      │   Evaluation    │   Analysis      │   Main    │
│   Scripts       │   Scripts       │   Scripts       │   System  │
│                 │                 │                 │           │
│ • Multi-algo    │ • Comprehensive │ • ERD/ERS       │ • Real-   │
│   training      │   evaluation    │   analysis      │   time    │
│ • Validation    │ • Model comp.   │ • Visualization │   BCI     │
│ • Persistence   │ • Reporting     │ • Research      │ • Unity   │
│                 │                 │   tools         │   control │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### 2. Core BCI Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                       Core BCI Layer                            │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│  BCISystem      │ Signal          │ Classifier      │ Data      │
│  Orchestrator   │ Processor       │ Engine          │ Sources   │
│                 │                 │                 │           │
│ • Component     │ • CSP filtering │ • Multi-algo    │ • Live    │
│   coordination  │ • Artifact      │   support       │   EEG     │
│ • State mgmt    │   rejection     │ • Confidence    │ • File    │
│ • Threading     │ • Feature       │   estimation    │   replay  │
│ • Error         │   extraction    │ • Adaptive      │ • Test    │
│   handling      │ • ERD/ERS       │   thresholding  │   data    │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### 3. Processing & Analysis Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                Processing & Analysis Layer                       │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│   Calibration   │  Preprocessing  │  Visualization  │ Analysis  │
│   System        │  Pipeline       │  Engine         │ Modules   │
│                 │                 │                 │           │
│ • Guided        │ • Quality       │ • Real-time     │ • Band    │
│   procedure     │   assessment    │   feedback      │   power   │
│ • Quality       │ • Artifact      │ • Classification│ • ERD/ERS │
│   monitoring    │   removal       │   confidence    │ • Spectra │
│ • Model         │ • Epoching      │ • System status │ • Stats   │
│   training      │ • Normalization │ • PyQt5 GUI     │ • Plots   │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

### 4. Communication & Integration Layer

```
┌─────────────────────────────────────────────────────────────────┐
│             Communication & Integration Layer                    │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│ LSL Integration │ Unity Interface │ File Management │ Logging   │
│                 │                 │                 │           │
│ • Stream        │ • Command       │ • Model I/O     │ • System  │
│   creation      │   translation  │ • Data storage  │   events  │
│ • Real-time     │ • Smoothing     │ • Config mgmt   │ • Debug   │
│   streaming     │ • Feedback      │ • Result export │   info    │
│ • Connection    │ • C# scripts    │ • Format        │ • Error   │
│   management    │ • Cross-        │   conversion    │   traces  │
│                 │   platform      │                 │           │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
```

## Core Component Architecture

### BCISystem Orchestrator (`dependencies/BCISystem.py`)

```python
class BCISystem:
    """Main system orchestrator coordinating all BCI components"""
    
    # Core Components
    ├── DataSource          # EEG data acquisition
    ├── SignalProcessor     # Feature extraction pipeline
    ├── Classifier          # ML classification engine
    ├── CalibrationSystem   # Guided calibration
    ├── SimulationInterface # Unity communication
    └── Visualization       # Real-time GUI
    
    # System States
    ├── Calibration Mode    # Data collection and training
    ├── Processing Mode     # Real-time classification
    └── Idle Mode          # System standby
    
    # Threading Model
    ├── Main Thread        # System coordination
    ├── Data Thread        # Background data acquisition
    ├── Processing Thread  # Real-time classification
    └── UI Thread         # Visualization updates
```

**Key Responsibilities:**
- Component lifecycle management
- Inter-component communication
- State management and transitions
- Error handling and recovery
- Performance monitoring

### Signal Processing Pipeline (`dependencies/signal_processor.py`)

```python
class SignalProcessor:
    """Advanced signal processing for motor imagery BCI"""
    
    # Processing Stages
    ├── Spatial Filtering
    │   ├── Common Average Reference (CAR)
    │   └── Common Spatial Patterns (CSP)
    │
    ├── Temporal Filtering
    │   ├── Butterworth Bandpass (1-45 Hz)
    │   └── Notch Filtering (60 Hz)
    │
    ├── Artifact Rejection
    │   ├── Amplitude Thresholding
    │   ├── Variance Analysis
    │   └── Quality Assessment
    │
    ├── Feature Extraction
    │   ├── Band Power Calculation
    │   ├── ERD/ERS Quantification
    │   ├── CSP Feature Vectors
    │   └── Baseline Normalization
    │
    └── Data Validation
        ├── Feature Quality Checks
        ├── Temporal Consistency
        └── Statistical Validation
```

**Processing Flow:**
1. **Raw EEG Input** → Multi-channel time series
2. **Spatial Filtering** → CAR + CSP transformation
3. **Temporal Filtering** → Bandpass + notch filtering
4. **Artifact Rejection** → Quality-based sample rejection
5. **Feature Extraction** → CSP features + band power
6. **Validation** → Feature quality assessment
7. **Output** → Validated feature vectors for classification

### Classification Engine (`dependencies/classifier.py`)

```python
class Classifier:
    """Multi-algorithm classification with adaptive performance"""
    
    # Supported Algorithms
    ├── Linear Discriminant Analysis (LDA)
    │   ├── Shrinkage regularization
    │   ├── Fast real-time inference
    │   └── Good baseline performance
    │
    ├── Support Vector Machine (SVM)
    │   ├── RBF kernel support
    │   ├── Probability estimation
    │   └── Non-linear classification
    │
    ├── Random Forest (RF)
    │   ├── Ensemble learning
    │   ├── Feature importance
    │   ├── Robust performance
    │   └── Overfitting resistance
    │
    └── XGBoost
        ├── Gradient boosting
        ├── Advanced regularization
        ├── High performance
        └── Hyperparameter optimization
    
    # Performance Features
    ├── Confidence Estimation
    ├── Adaptive Thresholding
    ├── Cross-Validation
    ├── Model Persistence
    └── Performance Monitoring
```

### Data Source Architecture (`dependencies/data_source.py`)

```python
class DataSource:
    """Unified interface for multiple EEG data sources"""
    
    # Source Types
    ├── LiveEEG (LSL)
    │   ├── Real-time streaming
    │   ├── OpenBCI integration
    │   ├── Connection monitoring
    │   └── Quality assessment
    │
    ├── FileReplay (CSV)
    │   ├── Timestamp synchronization
    │   ├── Real-time simulation
    │   ├── Format validation
    │   └── Progress tracking
    │
    └── ArtificialData
        ├── Synthetic signal generation
        ├── Configurable parameters
        ├── Testing and validation
        └── Development support
    
    # Common Interface
    ├── connect() / disconnect()
    ├── get_chunk() / get_sample()
    ├── quality_monitoring()
    └── error_handling()
```

## Training Pipeline Architecture

### Training Scripts Organization

```
scripts/training/
├── train_aggregate_models.py    # Multi-algorithm comprehensive training
│   ├── Data aggregation from multiple sessions
│   ├── Algorithm comparison (LDA, SVM, RF, XGBoost)
│   ├── Feature engineering (CSP + band power)
│   ├── Cross-validation with statistical analysis
│   └── Model persistence with metadata
│
├── train_model.py              # Robust training with validation splits
│   ├── Proper train/validation/test splits
│   ├── Overfitting prevention strategies
│   ├── Hyperparameter optimization
│   ├── Performance visualization
│   └── Statistical significance testing
│
└── train_rf_model.py           # Specialized Random Forest training
    ├── RF-specific optimization
    ├── Calibration format compatibility
    ├── Performance tuning
    └── Production deployment ready
```

### Training Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline Flow                      │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Raw Data  │───▶│Preprocessing│───▶│   Feature   │         │
│  │ Collection  │    │ & Quality   │    │ Extraction  │         │
│  └─────────────┘    │ Control     │    └─────────────┘         │
│                     └─────────────┘           │                │
│                                               ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Model     │◀───│  Training   │◀───│    Data     │         │
│  │ Validation  │    │ Multiple    │    │  Splitting  │         │
│  │ & Testing   │    │ Algorithms  │    │ (60/20/20)  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                                      │                │
│         ▼                                      ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Performance │    │   Model     │    │Cross-Validation      │         │
│  │ Reporting   │    │Persistence  │    │& Statistics │         │
│  │ & Analysis  │    │& Metadata   │    │  Analysis   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Evaluation Framework Architecture

### Evaluation Components

```python
# Comprehensive Model Evaluator
class ModelEvaluator:
    """Complete model evaluation and comparison framework"""
    
    # Evaluation Metrics
    ├── Classification Metrics
    │   ├── Accuracy, Precision, Recall, F1-Score
    │   ├── ROC-AUC and PR-AUC
    │   ├── Confusion Matrix Analysis
    │   └── Class-specific Performance
    │
    ├── Statistical Analysis
    │   ├── Cross-Validation Statistics
    │   ├── Confidence Intervals
    │   ├── Significance Testing
    │   └── Performance Distributions
    │
    ├── Model Comparison
    │   ├── Side-by-side Performance
    │   ├── Algorithm Ranking
    │   ├── Feature Importance
    │   └── Computational Efficiency
    │
    └── Visualization & Reporting
        ├── Confusion Matrix Heatmaps
        ├── ROC Curve Comparisons
        ├── Performance Bar Charts
        ├── Statistical Summary Tables
        ├── JSON Detailed Reports
        └── CSV Export for Analysis
```

### Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   Evaluation Pipeline                           │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Model     │───▶│   Load &    │───▶│ Performance │         │
│  │ Discovery   │    │ Validate    │    │ Assessment  │         │
│  └─────────────┘    │  Models     │    └─────────────┘         │
│                     └─────────────┘           │                │
│                                               ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Reporting  │◀───│ Statistical │◀───│   Metric    │         │
│  │ Generation  │    │  Analysis   │    │Calculation  │         │
│  │ (JSON/CSV)  │    │ & Testing   │    │ & Cross-Val │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                                      │                │
│         ▼                                      ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │Visualization│    │   Model     │    │ Performance │         │
│  │ Generation  │    │ Comparison  │    │  Ranking    │         │
│  │ & Plotting  │    │ & Ranking   │    │ & Selection │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## Real-time Operation Architecture

### Real-time Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                Real-time BCI Processing                         │
│                                                                 │
│ EEG Stream ──┐                                                 │
│              │    ┌─────────────┐    ┌─────────────┐           │
│              ├───▶│   Signal    │───▶│  Feature    │           │
│              │    │ Processing  │    │ Extraction  │           │
│ File Source ─┤    │  (50-80ms)  │    │  (CSP+BP)   │           │
│              │    └─────────────┘    └─────────────┘           │
│ Test Data ───┘           │                    │                │
│                          ▼                    ▼                │
│              ┌─────────────┐    ┌─────────────┐                │
│              │  Artifact   │    │Classification│                │
│              │ Rejection & │    │& Confidence │                │
│              │  Quality    │    │ Estimation  │                │
│              │ Assessment  │    └─────────────┘                │
│              └─────────────┘           │                        │
│                                       ▼                        │
│              ┌─────────────┐    ┌─────────────┐                │
│              │   Unity     │◀───│  Command    │                │
│              │ Integration │    │Generation & │                │
│              │   (LSL)     │    │  Smoothing  │                │
│              └─────────────┘    └─────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### Threading Architecture

```python
# Main System Threading Model
class BCISystem:
    
    # Thread Organization
    ├── Main Thread
    │   ├── System coordination
    │   ├── Component management
    │   ├── State transitions
    │   └── Error handling
    │
    ├── Data Acquisition Thread
    │   ├── Background EEG streaming
    │   ├── Buffer management
    │   ├── Quality monitoring
    │   └── Connection handling
    │
    ├── Processing Thread
    │   ├── Real-time signal processing
    │   ├── Feature extraction
    │   ├── Classification
    │   └── Command generation
    │
    ├── LSL Output Thread
    │   ├── Unity communication
    │   ├── Command streaming
    │   ├── Connection management
    │   └── Error recovery
    │
    └── Visualization Thread (Optional)
        ├── GUI updates
        ├── Plot generation
        ├── Performance monitoring
        └── User interaction
```

## Data Flow Architecture

### Training Data Flow

```
Raw EEG Data (CSV)
        │
        ▼
┌─────────────────┐
│  Preprocessing  │ ── Quality Control
│   • Load CSV    │ ── Format Validation  
│   • Clean Data  │ ── Artifact Detection
│   • Normalize   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    Epoching     │ ── Trial Segmentation
│  • Extract      │ ── Label Assignment
│    Trials       │ ── Baseline Correction
│  • Label Data   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Feature         │ ── CSP Training
│ Extraction      │ ── Band Power Calc
│  • CSP Filters  │ ── ERD/ERS Analysis
│  • Band Power   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Model Training  │ ── Algorithm Selection
│  • Train/Val/   │ ── Cross-Validation
│    Test Split   │ ── Hyperparameter Opt
│  • Multiple     │ ── Performance Analysis
│    Algorithms   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Model Storage   │ ── Serialization
│  • Save Model   │ ── Metadata Storage
│  • Save Results │ ── Version Control
│  • Export Data  │
└─────────────────┘
```

### Real-time Data Flow

```
Live EEG Stream (LSL)
        │
        ▼
┌─────────────────┐
│ Data Buffering  │ ── Circular Buffer
│  • LSL Inlet    │ ── Timestamp Sync
│  • Buffer Mgmt  │ ── Quality Check
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Window          │ ── Sliding Window
│ Extraction      │ ── Overlap Management
│  • 2s Windows   │ ── Data Validation
│  • 75% Overlap  │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Signal          │ ── CSP Application
│ Processing      │ ── Filtering
│  • Apply CSP    │ ── Artifact Rejection
│  • Extract      │ ── Quality Assessment
│    Features     │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Classification  │ ── Model Inference
│  • ML Predict   │ ── Confidence Calc
│  • Confidence   │ ── Threshold Check
│  • Thresholding │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Command         │ ── Unity Protocol
│ Generation      │ ── LSL Streaming
│  • LSL Stream   │ ── Error Handling
│  • Unity Data   │
└─────────────────┘
```

## Integration Architecture

### Unity Integration

```csharp
// Unity Integration Architecture
namespace BCIIntegration {
    
    // LSL Communication Layer
    class LSLManager {
        ├── Stream Discovery
        ├── Connection Management
        ├── Data Reception
        ├── Error Handling
        └── Reconnection Logic
    }
    
    // Command Processing
    class CommandProcessor {
        ├── Data Parsing
        ├── Command Validation
        ├── Confidence Filtering
        ├── Smoothing Algorithms
        └── State Management
    }
    
    // Prosthetic Control
    class ProstheticController {
        ├── Hand Animation Control
        ├── Wrist Rotation Control
        ├── Visual Feedback
        ├── State Visualization
        └── Performance Monitoring
    }
    
    // System Integration
    class BCISystem {
        ├── Component Coordination
        ├── Performance Monitoring
        ├── Debug Information
        ├── Configuration Management
        └── User Interface
    }
}
```

### LSL Protocol

```python
# LSL Stream Format
stream_info = {
    'name': 'ProstheticControl',
    'type': 'Control',
    'channel_count': 4,
    'sampling_rate': 60.0,
    'channel_format': 'float32',
    'source_id': 'BCI_Prosthetic_v1.0'
}

# Data Channels
data_packet = [
    hand_state,      # [0] Hand open/close state (0.0-1.0)
    wrist_state,     # [1] Wrist rotation state (0.0-1.0)  
    command_type,    # [2] Command type (0=idle, 1=left, 2=right)
    confidence       # [3] Classification confidence (0.0-1.0)
]
```

## Performance Characteristics

### Computational Complexity

| Component | Time Complexity | Space Complexity | Real-time Constraint |
|-----------|----------------|------------------|---------------------|
| CSP Filtering | O(n×c²) | O(c²) | < 20ms |
| Band Power | O(n×log(n)) | O(n) | < 10ms |
| LDA Classification | O(c) | O(c) | < 5ms |
| RF Classification | O(t×log(s)) | O(t×s) | < 15ms |
| SVM Classification | O(s×c) | O(s×c) | < 10ms |
| Total Pipeline | O(n×c²) | O(n+c²) | < 80ms |

Where: n=samples, c=channels, t=trees, s=support vectors

### Memory Architecture

```python
# Memory Management Strategy
class MemoryManager:
    
    # Circular Buffers
    ├── EEG Buffer (10s @ 250Hz)     # 20,000 samples × 8 channels
    ├── Feature Buffer (100 windows) # Rolling feature history
    ├── Classification Buffer (50)   # Recent classifications
    └── Performance Buffer (1000)    # Performance metrics
    
    # Pre-allocated Arrays
    ├── Window Array (500 × 8)       # Current processing window
    ├── Feature Vector (12-20)       # CSP + band power features
    ├── CSP Filters (8 × 4)         # Spatial filter matrix
    └── Baseline Stats (8 × 2)       # Channel baseline statistics
    
    # Memory Optimization
    ├── In-place Operations          # Minimize allocations
    ├── Vectorized Computations      # NumPy optimization
    ├── Buffer Reuse                 # Circular buffer management
    └── Garbage Collection Control   # Minimize GC impact
```

## Configuration Architecture

### Hierarchical Configuration

```python
# Configuration Hierarchy
configuration_system = {
    
    # Base Configuration (config.py)
    'base_config': {
        'signal_processing': {...},
        'classification': {...},
        'hardware': {...},
        'performance': {...}
    },
    
    # User Configuration (JSON override)
    'user_config': {
        'custom_parameters': {...},
        'experimental_settings': {...},
        'performance_tuning': {...}
    },
    
    # Runtime Configuration
    'runtime_config': {
        'adaptive_parameters': {...},
        'session_specific': {...},
        'performance_adjustments': {...}
    },
    
    # Model Configuration (embedded in models)
    'model_config': {
        'training_parameters': {...},
        'preprocessing_settings': {...},
        'validation_results': {...}
    }
}
```

### Configuration Management

```python
class ConfigurationManager:
    """Centralized configuration management"""
    
    # Configuration Sources
    ├── Static Configuration (config.py)
    ├── User Overrides (JSON files)
    ├── Model Metadata (embedded)
    ├── Runtime Adaptations (dynamic)
    └── Performance Optimizations (automatic)
    
    # Configuration Categories
    ├── Signal Processing Parameters
    ├── Machine Learning Settings
    ├── Real-time Performance Tuning
    ├── Hardware Interface Configuration
    ├── Unity Integration Settings
    ├── Logging and Debugging Options
    └── Experimental Feature Flags
    
    # Configuration Validation
    ├── Parameter Range Checking
    ├── Compatibility Verification
    ├── Performance Impact Assessment
    └── Error Prevention
```

## Error Handling Architecture

### Multi-level Error Handling

```python
# Error Handling Strategy
error_handling_system = {
    
    # Component Level
    'component_errors': {
        'detection': 'Local error detection and logging',
        'recovery': 'Component-specific recovery strategies',
        'escalation': 'Error escalation to system level'
    },
    
    # System Level  
    'system_errors': {
        'coordination': 'Cross-component error coordination',
        'recovery': 'System-wide recovery procedures',
        'fallback': 'Graceful degradation strategies'
    },
    
    # User Level
    'user_errors': {
        'notification': 'User-friendly error messages',
        'guidance': 'Recovery action suggestions',
        'support': 'Debug information collection'
    },
    
    # Critical Errors
    'critical_errors': {
        'safety': 'Immediate system shutdown',
        'preservation': 'Data and state preservation',
        'reporting': 'Detailed error reporting'
    }
}
```

## Security Considerations

### Security Architecture

```python
# Security Framework
security_measures = {
    
    # Data Security
    'data_protection': {
        'encryption': 'EEG data encryption at rest',
        'anonymization': 'Personal data anonymization',
        'access_control': 'Role-based access control'
    },
    
    # Communication Security
    'communication': {
        'lsl_security': 'LSL stream authentication',
        'unity_interface': 'Secure Unity communication',
        'network_isolation': 'Network security measures'
    },
    
    # System Security
    'system_protection': {
        'input_validation': 'Comprehensive input validation',
        'resource_limits': 'Resource usage limits',
        'privilege_separation': 'Minimum privilege principle'
    },
    
    # Compliance
    'regulatory': {
        'medical_compliance': 'Medical device regulations',
        'privacy_protection': 'GDPR/HIPAA compliance',
        'audit_trail': 'Complete audit logging'
    }
}
```

## Extensibility Architecture

### Plugin Architecture

```python
# Extensible Component Framework
class ExtensibilityFramework:
    """Framework for system extensions and customizations"""
    
    # Extension Points
    ├── Signal Processing Extensions
    │   ├── Custom Filter Implementations
    │   ├── Novel Feature Extraction Methods
    │   ├── Advanced Artifact Rejection
    │   └── Experimental Processing Techniques
    │
    ├── Classification Extensions
    │   ├── New ML Algorithm Integration
    │   ├── Deep Learning Models
    │   ├── Ensemble Methods
    │   └── Custom Decision Logic
    │
    ├── Data Source Extensions
    │   ├── New Hardware Support
    │   ├── Alternative Data Formats
    │   ├── Streaming Protocols
    │   └── Simulation Environments
    │
    ├── Analysis Extensions
    │   ├── Custom Visualization Tools
    │   ├── Advanced Statistical Analysis
    │   ├── Research-specific Metrics
    │   └── Publication-ready Outputs
    │
    └── Integration Extensions
        ├── Alternative Unity Interfaces
        ├── Other Simulation Platforms
        ├── Hardware Control Systems
        └── External System Integration
```

This architecture provides a robust, scalable, and maintainable foundation for BCI research and development while ensuring production-ready reliability and performance. 