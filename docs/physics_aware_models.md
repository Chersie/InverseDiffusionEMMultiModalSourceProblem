# Physics-Aware Neural Networks for Electromagnetic Multipole Prediction

## Overview

Physics-aware neural networks incorporate domain knowledge about electromagnetic field theory, spherical harmonics, and energy conservation directly into the network architecture or training process. This goes beyond the standard approach of simply using a "frozen physics decoder" at the output.

## Available Model Types

### 1. **Multipole-Aware Network** (`multipole_aware`)

**Physics Insight**: Different multipole orders (l) represent different spatial scales and should be processed separately, respecting the spherical harmonic structure.

**Architecture**:
```
Input → Shared Feature Extractor → Separate Heads per l-order → Combined Coefficients
```

- **Shared features**: Extract common patterns across all orders
- **Order-specific heads**: Dedicated sub-networks for l=1, l=2, l=3, etc.
- **Output structure**: Respects the natural (l,m) mode organization

**Use Case**: Best when different multipole orders exhibit distinct patterns or when you want to analyze contributions per spatial scale.

```bash
python -m src.cli.run_train_physics_aware \
  --model-type multipole_aware \
  --n-samples 10000 --maxorder 5 \
  --hidden-size 512 --epochs 100
```

### 2. **Energy-Conserving Network** (`energy_conserving`)

**Physics Insight**: Total radiated power must be conserved - the network should predict coefficients that maintain energy balance with the input measurements.

**Architecture**:
```
Input → Backbone Network → Raw Coefficients
  ↓
Input → Energy Head → Energy Scaling → Final Coefficients
```

- **Energy constraint**: Coefficients are scaled to match input power
- **Power prediction**: Separate head predicts energy normalization
- **Conservation**: Built into the forward pass, not just the loss

**Use Case**: When energy conservation is critical and power measurements are highly accurate.

```bash
python -m src.cli.run_train_physics_aware \
  --model-type energy_conserving \
  --energy-conservation-weight 2.0 \
  --n-samples 10000 --maxorder 5
```

### 3. **Hybrid Physics-ML Model** (`hybrid`)

**Physics Insight**: Start with known physics (analytical solutions) and learn only the corrections/higher-order effects that simple models miss.

**Architecture**:
```
Input → Analytical Baseline (simple physics)
  ↓
Input + Baseline → Correction Network → Learned Residuals
  ↓
Adaptive Mixing → Final Prediction
```

- **Physics baseline**: Simple analytical model (e.g., dipole approximation)
- **Learned corrections**: Complex network learns residuals
- **Adaptive mixing**: Network learns how to combine physics + ML

**Use Case**: When you have good analytical approximations but need to capture complex effects or measurement errors.

```bash
python -m src.cli.run_train_physics_aware \
  --model-type hybrid \
  --n-samples 10000 --maxorder 5 \
  --epochs 150
```

### 4. **Symmetry-Aware Network** (`symmetry_aware`)

**Physics Insight**: Electromagnetic fields must respect reciprocity and other fundamental symmetries of Maxwell's equations.

**Architecture**:
```
Input → Standard Network → Raw Output → Symmetry Constraints → Final Output
```

- **Reciprocity enforcement**: Ensures electromagnetic reciprocity theorem
- **Symmetry operations**: Applied as architectural constraints
- **Physical consistency**: Output always respects known symmetries

**Use Case**: When physical consistency is more important than raw accuracy metrics.

```bash
python -m src.cli.run_train_physics_aware \
  --model-type symmetry_aware \
  --symmetry-regularization 0.5 \
  --n-samples 10000 --maxorder 5
```

## Physics-Aware Loss Functions

All models use enhanced loss functions that incorporate physics constraints:

### Standard Physics Loss
```
L = L_shape + λ_amp * L_amplitude
```

### Additional Physics Constraints
```
L_total = L_physics + λ_energy * L_energy_conservation + λ_symmetry * L_symmetry
```

Where:
- **L_energy_conservation**: Penalty when total power doesn't match input
- **L_symmetry**: Penalty for violating electromagnetic symmetries
- **Weights**: Configurable via `--energy-conservation-weight`, `--symmetry-regularization`

## Comparison with Baseline Models

| Model | Physics Knowledge | Architecture | Best Use Case |
|-------|------------------|--------------|---------------|
| **Ridge** | None (linear) | PCA → Linear | Simple baselines |
| **Standard MLP** | Frozen decoder only | PCA → MLP → Decoder | General nonlinear fitting |
| **Multipole-Aware** | Mode structure | Structured heads per l | Spatial scale analysis |
| **Energy-Conserving** | Power conservation | Energy-constrained architecture | High-accuracy power measurements |
| **Hybrid** | Analytical baseline | Physics + ML combination | Known approximations exist |
| **Symmetry-Aware** | EM symmetries | Symmetry-constrained output | Physical consistency priority |

## Training and Evaluation

All physics-aware models automatically include:

### Enhanced Evaluation Metrics
- **21 comprehensive metrics**: Standard + polarization-aware
- **Coefficient validation**: Detailed True vs Predicted analysis
- **Physics constraints**: Energy conservation, symmetry checks
- **Visualization**: Difference maps show where physics constraints help

### MLflow Integration
```bash
# View all model comparisons
python -m mlflow ui

# All experiments appear in the same dashboard for easy comparison
```

### Artifacts Generated
- **`models/artifacts/physics_aware_[type]_*/`**
  - Model weights (`model.pt`)
  - Coefficient analysis (CSV tables, heatmaps)
  - Validation images (with difference maps)
  - Physics constraint metrics

## Performance Characteristics

### Computational Cost
- **Multipole-Aware**: ~1.2x standard MLP (separate heads)
- **Energy-Conserving**: ~1.1x standard MLP (energy head overhead)
- **Hybrid**: ~1.5x standard MLP (baseline + correction networks)
- **Symmetry-Aware**: ~1.05x standard MLP (symmetry operations)

### Memory Usage
- **Similar to standard MLP** for most model types
- **Hybrid model**: Higher due to dual networks
- **All models**: Same dataset requirements

### Training Stability
- **Physics constraints generally improve stability**
- **Energy conservation**: Can help with gradient scaling
- **Symmetry constraints**: Reduce parameter space, faster convergence
- **Hybrid models**: May need longer training due to complexity

## Recommended Usage

### Start with Multipole-Aware
```bash
python -m src.cli.run_train_physics_aware \
  --model-type multipole_aware \
  --n-samples 1000 --maxorder 5 \
  --epochs 50
```

### For High-Precision Applications
```bash
python -m src.cli.run_train_physics_aware \
  --model-type energy_conserving \
  --energy-conservation-weight 2.0 \
  --n-samples 5000 --maxorder 7
```

### For Research/Analysis
```bash  
python -m src.cli.run_train_physics_aware \
  --model-type hybrid \
  --n-samples 10000 --maxorder 10 \
  --epochs 200
```

## Extension Points

The physics-aware framework is designed for easy extension:

### Adding New Physics Constraints
1. Modify the forward pass in the relevant model class
2. Add constraint terms to the loss function  
3. Update the config with new hyperparameters

### New Model Types
1. Create new class inheriting from `nn.Module`
2. Add to `create_physics_aware_model()` factory
3. Update CLI arguments

### Custom Physics Knowledge
The hybrid model framework makes it easy to incorporate:
- **Analytical solutions** (dipole, monopole, etc.)
- **Measurement calibration** models
- **Domain-specific constraints** from antenna theory

All physics-aware models maintain full compatibility with the existing evaluation framework, providing seamless comparison with baseline approaches! 🚀