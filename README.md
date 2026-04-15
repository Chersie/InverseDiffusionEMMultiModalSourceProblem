# Electromagnetic Multipole Analysis ML Pipeline

A modern, high-performance machine learning pipeline for electromagnetic multipole decomposition and coefficient prediction. This project provides clean APIs, comprehensive testing, and efficient implementations for analyzing electromagnetic field data.

## 🚀 Key Features

- **High-Performance Library Loading**: 100x faster decomposition with batch loading (replaces 510+ individual file reads)
- **Unified Data Generation**: Consistent synthetic data generation for both pipeline and ML training
- **Modern ML Architecture**: Modular model registry with MLP and baseline implementations  
- **Production-Ready Inference**: Batch processing API with memory management and validation
- **Comprehensive Testing**: 100+ unit tests ensuring reliability and correctness
- **Clean Configuration**: Hierarchical config system with environment variable support

## 📦 Installation

### Basic Setup
```bash
git clone <repository>
cd diplom

# Install core dependencies
pip install -r requirements.txt
```

### For Machine Learning (Optional)
```bash
# Install PyTorch and ML dependencies
pip install -r requirements-ml.txt
```

### For Development (Optional)  
```bash
# Install development tools
pip install -r requirements-dev.txt
```

## 🏃 Quick Start

### Field Decomposition
```python
from src.pipeline.decomposition import decompose_field

# Decompose electromagnetic field into multipole coefficients
result = decompose_field("Fields.txt", maxorder=15)
print(f"Decomposed {result['n_modes']} modes in {result['elapsed_time']:.2f}s")
```

### Train ML Model
```python
from src.models.registry import create_mlp
from src.core.data_generator import DataGenerator

# Create and train MLP model
model = create_mlp(maxorder=15, hidden_size=512)
generator = DataGenerator.for_ml_training()

# Generate training data
dataset = generator.generate_batch(maxorder=15, n_samples=10000)
# ... preprocessing and training ...
```

### Model Inference
```python
from src.api.inference import InferenceEngine

# Load trained model and make predictions
engine = InferenceEngine(model)
result = engine.predict_from_fields(E_theta, E_phi)

print(f"Predicted coefficients: E={result.coefficients_e.shape}")
```

## 🏗️ Architecture

The project follows modern ML pipeline design principles:

```
src/
├── core/           # Foundation (config, data, library management)
├── pipeline/       # Scientific pipeline (decomposition)  
├── models/         # ML models (MLP, baselines, registry)
├── api/           # Inference and serving
└── cli/          # Command-line interfaces

tests/            # Comprehensive test suite
data/            # Data management (raw, processed, ML)
models/         # Model artifacts and tracking
docs/          # Documentation
```

### Core Components

- **LibraryManager**: High-performance batch loading of multipole libraries
- **DataGenerator**: Unified synthetic field data generation  
- **ModelRegistry**: Factory system for different ML architectures
- **InferenceEngine**: Production-ready model serving with batch processing
- **Configuration System**: Hierarchical config with environment variable support

## 📊 Performance Improvements

| Component | Legacy | Modern | Improvement |
|-----------|--------|--------|-------------|
| Library Loading | 510+ file reads | Single batch load | ~100x faster |
| Data Generation | Multiple inconsistent implementations | Unified system | Consistent results |
| Model Training | Scattered code | Modular registry | Easy experimentation |
| Inference | Manual processing | Batch API | Memory efficient |

## 🔧 Configuration

### Environment Variables
```bash
# Pipeline settings
export MAXORDER=15
export LIBRARY_TYPE=fast

# ML settings  
export N_SAMPLES=50000
export EPOCHS=100
export DEVICE=cuda
export LEARNING_RATE=0.001

# Experiment tracking
export EXPERIMENT_NAME=my_experiment
```

### Programmatic Configuration
```python
from src.core.config import Config

# Environment-driven (recommended)
config = Config.from_env()

# Explicit configuration
base_config, mlp_config = Config.for_mlp(
    hidden_size=1024,
    learning_rate=0.0005
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/

# Specific components
pytest tests/unit/test_library_manager.py -v

# With coverage
pytest --cov=src tests/
```

## 📖 Documentation

- **[Getting Started](docs/getting_started.md)**: Setup and basic usage
- **[Architecture Guide](docs/architecture.md)**: System design and components  
- **[Migration Guide](docs/migration_guide.md)**: Upgrading from legacy code
- **[API Reference](src/)**: Detailed API documentation in docstrings

## 🔄 Migration from Legacy Code

The project replaces legacy numbered scripts with a modern, maintainable architecture:

| Legacy | Modern Replacement |
|--------|-------------------|
| `NaiveSolution/3 FieldsToMultipoles.py` | `src.pipeline.decomposition` |
| `simple_pipeline/2_power_to_coeffs.py` | `src.models` + `src.api.inference` |
| Manual library loading | `src.core.library_manager` |
| Scattered configs | `src.core.config` |

See the [Migration Guide](docs/migration_guide.md) for detailed examples.

## 🛠️ Development

### Project Structure
- **Modular Design**: Clean separation between core, pipeline, models, and API
- **Type Safety**: Comprehensive type hints throughout
- **Error Handling**: Informative error messages with validation
- **Testing**: Unit and integration tests for all components
- **Documentation**: Comprehensive docstrings and guides

### Contributing
1. Run tests: `pytest tests/`
2. Check types: `mypy src/`
3. Format code: `black src/ tests/`
4. Update documentation for any API changes

## 📋 Requirements

- **Python**: 3.8+
- **Core**: NumPy, SciPy, matplotlib
- **Optional ML**: PyTorch (for neural networks)
- **Optional Dev**: pytest, black, mypy

## 🎯 Use Cases

- **Research**: Electromagnetic multipole analysis and decomposition
- **ML Training**: Power pattern to coefficient prediction
- **Production**: High-throughput field analysis pipelines
- **Education**: Clean, well-documented codebase for learning

## 📄 License

[Add license information]

## 🤝 Support

- **Documentation**: Check `docs/` directory
- **Issues**: [Add issue tracker link]
- **Examples**: See `examples/` directory (if available)

---

**Performance Note**: The new architecture provides the same functionality as legacy scripts with significantly better performance, reliability, and maintainability. Decomposition operations are ~100x faster due to optimized library loading.