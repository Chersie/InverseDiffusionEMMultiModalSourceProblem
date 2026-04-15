# Quick Start Guide

Get up and running with MLP model training in minutes!

## 🚀 Immediate Start

### 1. Test the Setup
```bash
# Navigate to project root
cd /path/to/diplom

# Run the quick example
python experiments/example_train.py
```

This will train a small MLP model in ~30 seconds to verify everything works.

### 2. Basic MLP Training
```bash
cd experiments
python scripts/train_mlp.py --config configs/mlp_basic.yaml
```

### 3. Baseline Comparison
```bash
cd experiments  
python scripts/train_baseline.py --config configs/baseline_comparison.yaml
```

## 🎯 Common Training Commands

### Quick Testing (Small Models)
```bash
# Small MLP for quick testing
python scripts/train_mlp.py --config configs/mlp_basic.yaml --maxorder 3

# Compare with baseline
python scripts/train_baseline.py --config configs/baseline_comparison.yaml --maxorder 3
```

### Production Training (Large Models)
```bash
# Large MLP for best performance
python scripts/train_mlp.py --config configs/mlp_large.yaml

# With custom parameters
python scripts/train_mlp.py --config configs/mlp_large.yaml --maxorder 10 --device cuda
```

### Experiment Sweeps
```bash
# Run predefined experiment types
python scripts/run_experiment.py mlp-basic
python scripts/run_experiment.py baseline
python scripts/run_experiment.py hyperparameter-sweep
python scripts/run_experiment.py maxorder-comparison
```

## 📊 View Results

### MLflow Tracking UI
```bash
# Start MLflow UI (from project root)
mlflow ui --host 0.0.0.0 --port 5000

# Open in browser: http://localhost:5000
```

### Experiment Analysis
```python
from experiments.scripts.experiment_utils import compare_experiments, plot_training_comparison

# Compare different experiments
df = compare_experiments(['mlp_basic', 'baseline_comparison'])
print(df[['experiment_name', 'test_mse', 'test_r2', 'training_time']])

# Plot comparison
plot_training_comparison(['mlp_basic', 'baseline_comparison'])
```

## 📁 Results Structure

After training, find your results in:
```
experiments/results/
├── mlp_basic_20240414_123456/
│   ├── config.yaml          # Experiment configuration
│   ├── model/               # Trained model files
│   ├── preprocessing/       # Preprocessing pipeline
│   ├── checkpoints/        # Model checkpoints
│   └── logs/              # Training logs
```

## ⚙️ Configuration

### Modify Existing Configs
Edit files in `experiments/configs/`:
- `mlp_basic.yaml` - Quick testing
- `mlp_large.yaml` - Production training  
- `baseline_comparison.yaml` - Baseline models

### Key Parameters to Adjust

**Model Architecture:**
```yaml
model:
  maxorder: 15        # Multipole order (3-15)
  hidden_size: 512    # Network size (128-2048)
  n_hidden_layers: 2  # Depth (1-6)
  dropout_rate: 0.1   # Regularization (0.0-0.5)
```

**Training:**
```yaml
training:
  n_samples: 10000    # Dataset size (1000-100000)
  epochs: 50          # Training epochs (10-500)
  batch_size: 128     # Batch size (32-512)
  learning_rate: 0.001 # Learning rate (0.0001-0.01)
```

**Performance:**
```yaml
device: "cuda"        # "cpu", "cuda", "mps", "auto"
preprocessing:
  pca_components: 256 # Feature reduction (64-1024)
```

## 🔧 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're in project root
cd /path/to/diplom
python -c "from src.core.config import Config; print('✅ OK')"
```

**Missing Dependencies:**
```bash
pip install -r requirements-ml.txt  # For neural networks
pip install -r requirements.txt     # For baselines
```

**GPU Issues:**
```bash
# Check PyTorch GPU support
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Force CPU if needed
python scripts/train_mlp.py --config configs/mlp_basic.yaml --device cpu
```

**Memory Issues:**
```yaml
# Reduce batch size and model size in config
training:
  batch_size: 32      # Smaller batches
model:
  hidden_size: 128    # Smaller model
```

## 📈 Performance Tips

### For Speed
- Use `mlp_basic.yaml` config
- Set `maxorder: 3-5`
- Use `device: "cuda"` if available
- Increase `batch_size` if memory allows

### For Accuracy
- Use `mlp_large.yaml` config  
- Set `maxorder: 15`
- Increase `n_samples: 50000+`
- Use `hidden_size: 1024+`

### For Comparison
- Always train baseline first: `baseline_comparison.yaml`
- Use same `maxorder` and `n_samples` for fair comparison
- Check MLflow for side-by-side metrics

---

🎉 **You're ready to start training!** Begin with the quick example, then move to production configs as needed.