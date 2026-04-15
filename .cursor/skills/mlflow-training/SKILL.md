---
name: mlflow-training
description: Guide on how to train models using MLFlow, generate plots, and evaluate on the test dataset. Use when the user asks to train a model, track experiments, or evaluate on the E_in_plane dataset.
---

# MLFlow Training, Plotting, and Evaluation

## Quick Start

This project uses a comprehensive MLFlow integration for tracking experiments, managing models, and generating visualizations.

Before running any experiments, ensure the MLFlow platform is running:
```bash
python scripts/start_mlflow_platform.py
```
This starts the tracking server (port 5000), model API (port 8001), and custom dashboard (port 8502).

## Training with MLFlow

There are two main ways to train models with MLFlow tracking:

### 1. Simple Training (Config-free)

Use `experiments/train_simple.py` for quick, code-based configuration testing. This script automatically handles data generation, preprocessing, training, evaluation, plotting, and model registration.

```bash
python experiments/train_simple.py
```

### 2. Comprehensive Training (YAML Config)

Use `experiments/scripts/train_mlp.py` for robust, configuration-driven training. This is the preferred method for serious experiments.

```bash
python experiments/scripts/train_mlp.py --config experiments/configs/mlp_basic.yaml
```

**Key MLFlow Manager Methods:**
When writing custom training scripts, use the `MLFlowExperimentManager`:

```python
from experiments.utils.mlflow_manager import create_experiment_manager

# Initialize manager
manager = create_experiment_manager("my_experiment_name", config)

# Start run
if manager.start_experiment(run_name="my_run", tags={"type": "test"}):
    # ... training code ...
    
    # Log metrics
    manager.log_model_performance({"test_mse": 0.123, "test_r2": 0.85})
    
    # Register model
    manager.register_model(model, "my_model_name", input_example=X_test[:5])
    
    # Log plots
    manager.log_plots(Path("path/to/plots"))
    
    # Finish
    manager.finish_experiment()
```

## Generating Plots

The project includes a robust plotting utility module: `experiments/utils/plotting.py`.

**Available Plotting Functions:**
*   `plot_training_curves(history, save_path)`: Plots train and validation loss over epochs.
*   `plot_prediction_scatter(y_true, y_pred, title, save_path)`: Creates a scatter plot of predicted vs. true values.
*   `plot_coefficient_comparison(...)`: Compares specific multipole coefficients.
*   `create_experiment_summary_plot(...)`: Generates a comprehensive multi-panel summary figure.

**Example Usage in a Script:**
```python
from experiments.utils.plotting import plot_prediction_scatter
from pathlib import Path

plot_dir = Path("results/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

plot_prediction_scatter(
    y_test, predictions,
    title="Test Predictions",
    save_path=plot_dir / "scatter.png"
)

# If using MLFlowManager, log the directory:
manager.log_plots(plot_dir)
```

## Evaluating on the Test Dataset (`E_in_plane`)

A dedicated test dataset (`E_in_plane` and `Multipoles_in_plane`) is available for final model evaluation.

### Using the Evaluation Script

Use the `experiments/scripts/evaluate_model.py` script to load a registered model and score it against the test dataset.

```bash
# Evaluate the latest Production version of a model
python experiments/scripts/evaluate_model.py --model-name my_registered_model_name

# Evaluate a specific version
python experiments/scripts/evaluate_model.py --model-name my_registered_model_name --model-version 2

# Evaluate with a limit (for quick testing)
python experiments/scripts/evaluate_model.py --model-name my_registered_model_name --limit 100
```

### How it Works Internally

1.  **`TestDatasetLoader`**: Located in `src/core/dataset_loader.py`, this class parses the raw `.txt` files from the `E_in_plane` (features) and `Multipoles_in_plane` (targets) directories.
2.  **Preprocessing**: The evaluation script attempts to load the most recent `PreprocessingPipeline` from `experiments/results/` to transform the raw test data into the format expected by the model.
3.  **Inference & Logging**: The model predicts on the transformed test data. Metrics (MSE, MAE, R²) and a scatter plot are generated and logged to a new MLFlow experiment named `test_dataset_evaluation`.