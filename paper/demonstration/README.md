# Demonstration Plots Generator

This directory contains scripts for generating demonstration plots for trained models on validation datasets.

## generate_plots.py

A comprehensive script that loads a trained model from a checkpoint and generates evaluation plots on a specified dataset.

### Usage

```bash
uv run python paper/demonstration/generate_plots.py \
    --model-checkpoint path/to/checkpoint.pt \
    --dataset path/to/dataset \
    --output-dir path/to/output
```

### Required Arguments

- `--model-checkpoint`: Path to the model checkpoint file (.pt)
- `--dataset`: Path to the dataset (directory with real antenna data or specific split)

### Optional Arguments

- `--output-dir`: Directory where plots will be saved (default: `paper/demonstration/figures`)
- `--config`: Optional path to the experiment config JSON file
- `--feature-extractor`: Optional path to the feature extractor pickle file
- `--plot-types`: Comma-separated list of plot types to generate (default: `all`)
  - Available: `r2_distribution`, `bin_accuracy`, `field_comparison`, `coef_scatter`, `loss_curves`
- `--split`: Dataset split to use for evaluation (default: `val`)
  - Choices: `train`, `val`, `test`, `holdout`
- `--n-samples`: Maximum number of samples to use from dataset
- `--n-field-samples`: Number of samples to visualize in field comparison (default: 3)
- `--output-prefix`: Prefix for output filenames

### Generated Plots

The script generates the following plots in the specified output directory:

1. **R² Distribution** (`r2_distribution.pdf/png`)
   - Distribution of R² scores across samples
   - Shows median R² value

2. **Bin Accuracy Distribution** (`bin_accuracy_distribution.pdf/png`)
   - Distribution of bin accuracy scores across samples
   - Shows median bin accuracy value

3. **Field Comparison** (`field_comparison_sample_*.pdf/png`)
   - Side-by-side comparison of predicted vs true power patterns
   - One figure per selected sample
   - Shows P_pred, P_true, and signed residual

4. **Coefficient Scatter** (`coef_scatter.pdf/png`)
   - Scatter plots of predicted vs target coefficients
   - One subplot per coefficient block (Re a^E, Im a^E, Re a^M, Im a^M)

5. **Loss Curves** (`loss_curves.pdf/png`)
   - Training/validation loss curves over time
   - Only generated if training history is available in checkpoint

### Example Usage

Generate all plots for a model on validation data:

```bash
uv run python paper/demonstration/generate_plots.py \
    --model-checkpoint experiments/baseline/figures_real_augmented_best/checkpoints/best.pt \
    --dataset data/raw/real_antenna \
    --split val \
    --output-dir paper/demonstration/figures
```

Generate only specific plot types:

```bash
uv run python paper/demonstration/generate_plots.py \
    --model-checkpoint experiments/baseline/figures_real_augmented_best/checkpoints/best.pt \
    --dataset data/raw/real_antenna \
    --plot-types r2_distribution,bin_accuracy \
    --output-dir paper/demonstration/figures
```

Generate plots for holdout data with custom prefix:

```bash
uv run python paper/demonstration/generate_plots.py \
    --model-checkpoint experiments/baseline/figures_real_augmented_best/checkpoints/best.pt \
    --dataset data/raw/real_antenna \
    --split holdout \
    --output-prefix holdout_ \
    --output-dir paper/demonstration/figures
```

### Integration with Multirun

This script is designed to work with the multirun workflow for generating demonstration plots for the best candidates:

```bash
# After running multirun, identify the best candidates
# Then generate plots for each best candidate
for candidate in $(find multirun -name "best.pt"); do
    uv run python paper/demonstration/generate_plots.py \
        --model-checkpoint "$candidate" \
        --dataset data/raw/real_antenna \
        --split holdout \
        --output-dir "paper/demonstration/figures/$(basename $(dirname $candidate))"
done
```

### Notes

- The script automatically detects and loads the feature extractor from the checkpoint directory
- If a config file is provided, it will be used to extract model parameters (l_max, scale_factor, model_name)
- The script uses the existing plotting functionality from `src/mpinv/analysis/plots/`
- All plots are generated in both PDF (vector, for publications) and PNG (raster, for presentations) formats
- The script follows the project's architectural invariants and coding standards