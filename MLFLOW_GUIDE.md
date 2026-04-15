# MLFlow Integration Guide

## Overview

This project now includes comprehensive MLFlow integration for experiment tracking, model registry, hyperparameter optimization, model serving, and custom dashboards. The integration provides a complete MLOps platform for electromagnetic multipole machine learning research.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the MLFlow Platform

```bash
# Start the complete platform (recommended)
python scripts/start_mlflow_platform.py

# Or start just the MLFlow server
python scripts/start_mlflow_server.py start --background
```

### 3. Access the Platform

After starting, you'll have access to:

- **MLFlow UI**: http://localhost:5000 - Experiment tracking and model registry
- **Custom Dashboard**: http://localhost:8502 - Advanced analytics and visualizations  
- **Model API**: http://localhost:8001 - Model serving REST API
- **API Documentation**: http://localhost:8001/docs - Interactive API docs

### 4. Run Your First Experiment

```bash
# Simple training with automatic MLFlow tracking
python experiments/train_simple.py

# Full experiment with YAML configuration
python experiments/scripts/train_mlp.py --config experiments/configs/mlp_basic.yaml
```

## 📊 Platform Components

### 1. Experiment Tracking

All training scripts automatically log:
- **Parameters**: Model configuration, hyperparameters, data settings
- **Metrics**: Training/validation losses, performance metrics, system metrics
- **Artifacts**: Model checkpoints, plots, preprocessing pipelines, configs
- **System Info**: GPU usage, memory, training time, git commit

**Example Usage:**
```python
from experiments.utils.mlflow_manager import mlflow_training_session

with mlflow_training_session("my_experiment") as session:
    # Training code here
    session.log_training_metrics(epoch, {"loss": loss, "accuracy": acc})
    session.register_model(model, input_example=X_test[:5])
```

### 2. Model Registry

Automatic model registration with:
- **Version Control**: Semantic versioning with release notes
- **Stage Management**: Development → Staging → Production workflow
- **Performance Tracking**: Metrics stored with each model version
- **Auto-promotion**: Models automatically promoted based on performance

**Example Usage:**
```python
from src.models.mlflow_integration import get_model_registry

registry = get_model_registry()

# Register a model
version = registry.register_model(model, "my_model", input_example=X_test)

# Promote to production
registry.promote_model("my_model", version, "Production")

# Load model from registry
model = registry.load_model("my_model", stage="Production")
```

### 3. Hyperparameter Optimization

Automated hyperparameter tuning with Optuna integration:
- **Intelligent Search**: TPE sampler with early stopping
- **MLFlow Integration**: All trials logged automatically
- **Parallel Execution**: Multiple workers support
- **Pruning**: Early termination of poor trials

**Example Usage:**
```python
from experiments.utils.hyperparameter_tuning import optimize_mlp_hyperparameters

best_params = optimize_mlp_hyperparameters(
    maxorder=5,
    n_samples=2000,
    n_trials=100,
    experiment_name="hyperopt_experiment"
)
```

### 4. Model Serving

Production-ready model serving with:
- **REST API**: FastAPI-based high-performance serving
- **Auto-loading**: Models loaded from registry automatically
- **Batch Processing**: Efficient batch inference support
- **Health Checks**: Built-in monitoring and health endpoints

**Example Usage:**
```bash
# Start model serving (included in platform startup)
python -m src.api.model_serving --model-name my_model --model-stage Production

# Make predictions via API
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [[1.0, 2.0, 3.0, ...]]}'
```

### 5. Custom Dashboards

Advanced visualization and analytics:
- **Interactive Plots**: Plotly-based interactive visualizations
- **Experiment Comparison**: Side-by-side experiment analysis
- **Hyperparameter Analysis**: Parameter importance and correlation
- **Performance Tracking**: Training progress and model comparison
- **Real-time Updates**: Auto-refreshing dashboards

## 🔧 Configuration

### MLFlow Server Configuration

Configure MLFlow server in `mlflow_server.conf`:

```ini
host=127.0.0.1
port=5000
backend_store_uri=sqlite:///mlflow.db
default_artifact_root=mlartifacts
serve_artifacts=true
workers=1
```

### Environment Variables

```bash
# MLFlow Configuration
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="electromagnetic_multipole_analysis"

# Model Serving
export MODEL_NAME="electromagnetic_multipole_model"
export MODEL_STAGE="Production"

# Dashboard
export DASHBOARD_AUTO_REFRESH=true
export DASHBOARD_REFRESH_INTERVAL=30
```

## 📈 Training Scripts Integration

### Enhanced Simple Training

The `train_simple.py` script now includes:
- Automatic experiment tracking
- Model registration
- Plot generation and logging
- Performance metrics tracking

```python
# All MLFlow functionality is automatic
python experiments/train_simple.py
```

### Advanced Training Pipeline

The `train_mlp.py` script supports:
- YAML-based configuration
- Comprehensive logging
- Hyperparameter optimization integration
- Model registry automation

```python
python experiments/scripts/train_mlp.py --config experiments/configs/mlp_basic.yaml
```

## 🔍 Hyperparameter Optimization

### Configuration File

Create `experiments/configs/hyperparameter_tuning.yaml`:

```yaml
optimization:
  n_trials: 100
  objective_metric: "val_loss"
  direction: "minimize"
  
search_space:
  hidden_size:
    type: "categorical"
    choices: [128, 256, 512, 1024]
  
  learning_rate:
    type: "float"
    low: 1e-5
    high: 1e-2
    log: true
```

### Run Optimization

```bash
python experiments/utils/hyperparameter_tuning.py \
    --maxorder 5 \
    --n_trials 100 \
    --experiment_name "hyperopt_mlp"
```

## 🚢 Model Deployment

### Local Deployment

```python
from experiments.utils.deployment import deploy_model

result = deploy_model(
    model_name="my_model",
    target="local",
    environment="production"
)
```

### Docker Deployment

```python
result = deploy_model(
    model_name="my_model",
    target="docker",
    environment="production"
)
```

### Kubernetes Deployment

```python
result = deploy_model(
    model_name="my_model", 
    target="kubernetes",
    environment="production"
)
```

## 📊 Dashboard Usage

### Streamlit Dashboard

The custom dashboard provides:

1. **Metrics Comparison**: Compare metrics across experiments
2. **Hyperparameter Analysis**: Parameter importance and correlation
3. **Training Timeline**: Progress visualization over time
4. **Performance Heatmaps**: Hyperparameter vs. performance visualization
5. **Best Runs**: Top-performing experiments table

### Export Dashboard

Generate standalone HTML dashboard:

```python
from experiments.utils.mlflow_dashboards import export_dashboard_html

export_dashboard_html(output_path="my_dashboard.html")
```

## 🛠 Management Commands

### Platform Management

```bash
# Start complete platform
python scripts/start_mlflow_platform.py

# Check platform status
python scripts/start_mlflow_platform.py --status

# Stop platform
python scripts/start_mlflow_platform.py --stop
```

### MLFlow Server Management

```bash
# Start server
python scripts/start_mlflow_server.py start

# Stop server  
python scripts/start_mlflow_server.py stop

# Server status
python scripts/start_mlflow_server.py status
```

## 🔧 Troubleshooting

### Common Issues

1. **MLFlow server won't start**
   - Check if port 5000 is available
   - Verify database permissions
   - Check logs for detailed error messages

2. **Models not appearing in registry**
   - Ensure MLFlow server is running
   - Check tracking URI configuration
   - Verify model registration code

3. **Dashboard not loading data**
   - Confirm experiment names in configuration
   - Check MLFlow server connectivity
   - Verify experiment data exists

4. **Model serving API errors**
   - Ensure model exists in registry
   - Check model loading permissions
   - Verify API endpoint configuration

### Log Locations

- MLFlow server logs: Check terminal output
- Model serving logs: Check API server output  
- Dashboard logs: Streamlit terminal output
- Training logs: Experiment directories

### Debug Mode

Enable detailed logging:

```bash
python scripts/start_mlflow_platform.py --debug
```

## 📚 Examples

### Complete Training Workflow

```python
# 1. Start platform
# python scripts/start_mlflow_platform.py

# 2. Run training with tracking
from experiments.utils.mlflow_manager import mlflow_training_session

with mlflow_training_session("electromagnetic_analysis") as session:
    # Generate data
    dataset = generator.generate_batch(maxorder=5, n_samples=1000)
    
    # Train model
    model = create_mlp(...)
    result = model.fit(X_train, y_train, X_val, y_val)
    
    # Log metrics
    session.log_training_metrics(epoch, result['metrics'])
    
    # Register model
    session.register_model(model, input_example=X_test[:5])

# 3. View results in MLFlow UI (http://localhost:5000)
# 4. Analyze in custom dashboard (http://localhost:8502)
# 5. Deploy model via API (http://localhost:8001)
```

### Hyperparameter Optimization Workflow

```python
# 1. Configure optimization
config = OptimizationConfig(
    n_trials=50,
    objective_metric="val_loss",
    search_space={
        "hidden_size": {"type": "categorical", "choices": [256, 512, 1024]},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}
    }
)

# 2. Run optimization
optimizer = HyperparameterOptimizer(config)
study = optimizer.optimize(training_data, validation_data)

# 3. Get best parameters
best_params = optimizer.get_best_config()

# 4. Train final model with best parameters
final_model = create_mlp(**best_params)
```

## 🎯 Best Practices

1. **Experiment Organization**
   - Use descriptive experiment names
   - Include relevant tags for filtering
   - Document experiment goals

2. **Model Registry**
   - Always provide model descriptions
   - Use semantic versioning
   - Test models before promoting to production

3. **Hyperparameter Optimization**
   - Start with broad search spaces
   - Use pruning to save computation
   - Validate best parameters on test set

4. **Model Serving**
   - Monitor model performance in production
   - Implement proper error handling
   - Use health checks for reliability

5. **Dashboard Usage**
   - Regular monitoring of experiment progress
   - Compare models before deployment
   - Track performance degradation

## 📖 Additional Resources

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

This MLFlow integration provides a production-ready MLOps platform for electromagnetic multipole machine learning research. All components work together seamlessly to provide experiment tracking, model management, optimization, serving, and monitoring capabilities.