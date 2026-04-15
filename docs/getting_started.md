# Getting Started

This guide helps you set up and use the ML pipeline for electromagnetic multipole analysis.

## Installation

### 1. Basic Dependencies
```bash
# Navigate to project directory
cd /path/to/diplom

# Install core dependencies
pip install -r requirements.txt
```

### 2. ML Dependencies (Optional)
For neural network training:
```bash
# Install PyTorch and related packages
pip install -r requirements-ml.txt
```

### 3. Development Dependencies (Optional)
For development and testing:
```bash
pip install -r requirements-dev.txt
```

### 4. Verify Installation
```bash
# Run basic tests to verify setup
pytest tests/unit/test_config.py -v
```

## Quick Start Examples

### Example 1: Field Decomposition
Decompose electromagnetic field data into multipole coefficients:

```python
from src.pipeline.decomposition import decompose_field

# Decompose a field file
result = decompose_field(
    field_path="data/Fields.txt",
    maxorder=15
)

print(f"Decomposition completed in {result['elapsed_time']:.2f}s")
print(f"Number of modes: {result['n_modes']}")
print(f"Output file: {result['output_path']}")
```

### Example 2: Train a Baseline Model
Train a Ridge regression model for power-to-coefficients prediction:

```python
from src.models.registry import create_baseline
from src.core.data_generator import DataGenerator

# Create model and data
model = create_baseline(maxorder=5, baseline_type="ridge")
generator = DataGenerator.for_ml_training()

# Generate synthetic training data
dataset = generator.generate_batch(maxorder=5, n_samples=1000)

# Prepare features (simplified)
X = dataset['power'].reshape(1000, -1)  # Flatten power patterns
y = pack_coefficients(dataset['coefficients_e'], dataset['coefficients_m'])

# Split data
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]  
X_val, y_val = X[train_size:], y[train_size:]

# Train model
result = model.fit(X_train, y_train, X_val, y_val)
print(f"Training MSE: {result['train_mse']:.6f}")
```

### Example 3: Model Inference
Use a trained model to make predictions:

```python
from src.api.inference import InferenceEngine
from src.models.registry import create_mlp

# Load or create model
model = create_mlp(maxorder=5)
# model.load("path/to/trained/model")  # Load pre-trained model

# Create inference engine  
engine = InferenceEngine(model)

# Make predictions from field data
import numpy as np
n_phi, n_theta = 360, 179
E_theta = np.random.random((n_phi, n_theta)) + 1j * np.random.random((n_phi, n_theta))
E_phi = np.random.random((n_phi, n_theta)) + 1j * np.random.random((n_phi, n_theta))

result = engine.predict_from_fields(E_theta, E_phi)
print(f"Predicted coefficients: E={result.coefficients_e.shape}, M={result.coefficients_m.shape}")
```

## Core Concepts

### Configuration System
The system uses hierarchical configuration with environment variable support:

```python
from src.core.config import Config

# Default configuration
config = Config()

# Environment-driven configuration (reads env vars)
config = Config.from_env()

# Specialized configurations
base_config, mlp_config = Config.for_mlp(
    hidden_size=1024,
    learning_rate=0.001
)
```

### Library Manager
Efficient loading of multipole library files:

```python
from src.core.library_manager import get_library_manager

# Get global manager instance
manager = get_library_manager(maxorder=15)

# Load all library modes (fast batch loading)
manager.load()

# Access individual modes
e_amplitude, e_theta = manager.get_e_mode(l=2, m=1)
m_amplitude, m_theta = manager.get_m_mode(l=2, m=1)

print(f"Library loaded: {manager.get_stats()}")
```

### Data Generation
Unified system for generating synthetic electromagnetic field data:

```python
from src.core.data_generator import DataGenerator

# For reproducible pipeline data
pipeline_gen = DataGenerator.for_pipeline()
sample = pipeline_gen.generate_sample(maxorder=5, sample_id=0)

# For diverse ML training data  
ml_gen = DataGenerator.for_ml_training()
batch = ml_gen.generate_batch(maxorder=5, n_samples=100)

print(f"Generated {batch['n_samples']} samples")
print(f"Field shape: {batch['amplitude'].shape}")
print(f"Coefficient shapes: E={batch['coefficients_e'].shape}, M={batch['coefficients_m'].shape}")
```

### Model Registry
Factory system for creating different model types:

```python
from src.models.registry import get_model_registry, create_model

# List available models
registry = get_model_registry()
print(f"Available models: {registry.list_models()}")

# Create models using factory functions
mlp_model = create_model("mlp", maxorder=15, hidden_size=512)
baseline_model = create_model("baseline", maxorder=15, baseline_type="ridge")

print(f"MLP info: {mlp_model.get_model_info()}")
```

## Configuration

### Environment Variables
Configure the system using environment variables:

```bash
# Pipeline configuration
export MAXORDER=10
export LIBRARY_TYPE=fast

# ML configuration  
export N_SAMPLES=50000
export EPOCHS=100
export BATCH_SIZE=512
export LEARNING_RATE=0.001
export DEVICE=cuda

# Experiment tracking
export EXPERIMENT_NAME=my_experiment
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Configuration Files
You can also create configuration programmatically:

```python
from src.core.config import MLPConfig, Config

# Custom MLP configuration
mlp_config = MLPConfig(
    input_dim=256,
    output_dim=60,  # For maxorder=5
    hidden_size=1024,
    n_hidden_layers=4,
    dropout_rate=0.1,
    learning_rate=0.0005,
    epochs=200
)

# Create model with custom config
model = MLPModel(mlp_config)
```

## File Structure

Understanding the project layout:

```
src/                     # Source code
├── core/               # Core components (config, data, library)
├── pipeline/          # Scientific pipeline (decomposition)  
├── models/           # ML models (MLP, baselines)
├── api/             # Inference and serving APIs
└── cli/            # Command-line interfaces

tests/              # Test suite
├── unit/          # Component tests
└── integration/  # End-to-end tests

data/              # Data management
├── raw/          # Input data
├── processed/   # Processed data  
└── ml/         # ML datasets

models/           # Model artifacts
├── artifacts/   # Trained models
└── tracking/   # Experiment logs

Chersie/         # Library data
├── Fields0.5/      # Standard library  
└── FieldsFast0.5/  # Fast library

docs/           # Documentation
```

## Performance Tips

### 1. Use Batch Processing
```python
# Efficient batch inference
from src.api.inference import BatchProcessor

processor = BatchProcessor(engine)
results = processor.process_field_batch(
    large_E_theta_batch, 
    large_E_phi_batch,
    progress_callback=lambda p: print(f"Progress: {p['samples_processed']}")
)
```

### 2. Configure Memory Usage
```python
from src.api.inference import InferenceConfig

# Configure for your system
config = InferenceConfig(
    batch_size=128,        # Smaller for less memory
    memory_limit_mb=4096,  # Set memory limit
    gc_frequency=50        # Garbage collection frequency
)
```

### 3. Use Appropriate Device
```python
# Automatic device selection
config = Config.from_env()  # Will detect best device

# Or explicit device
mlp_config = MLPConfig(device="cuda")  # Force GPU if available
```

## Next Steps

1. **Run Examples**: Try the complete examples in `examples/` directory
2. **Read Architecture**: Understand the system design in `docs/architecture.md`  
3. **Migration Guide**: If upgrading from legacy code, see `docs/migration_guide.md`
4. **API Reference**: Detailed API documentation in source code docstrings
5. **Testing**: Run the full test suite with `pytest tests/`

## Common Issues

### Import Errors
Make sure you're running from the project root directory:
```bash
cd /path/to/diplom  # Project root
python -c "from src.core.config import Config; print('Success!')"
```

### Missing Dependencies
Install the appropriate requirements file:
```bash
pip install -r requirements-ml.txt  # For PyTorch features
```

### Library Not Found
Verify the library directories exist:
```bash
ls Chersie/  # Should show Fields0.5/ and FieldsFast0.5/
```

### Memory Issues
Reduce batch sizes or use smaller models for testing:
```python
config = MLPConfig(hidden_size=128, batch_size=32)  # Smaller configuration
```