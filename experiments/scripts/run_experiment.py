#!/usr/bin/env python3
"""
Experiment Runner

Convenience script to run different types of experiments with simple commands.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_mlp_basic():
    """Run basic MLP experiment."""
    cmd = [
        sys.executable, "scripts/train_mlp.py",
        "--config", "configs/mlp_basic.yaml"
    ]
    subprocess.run(cmd, cwd="experiments")


def run_mlp_large():
    """Run large MLP experiment."""
    cmd = [
        sys.executable, "scripts/train_mlp.py", 
        "--config", "configs/mlp_large.yaml"
    ]
    subprocess.run(cmd, cwd="experiments")


def run_baseline():
    """Run baseline comparison."""
    cmd = [
        sys.executable, "scripts/train_baseline.py",
        "--config", "configs/baseline_comparison.yaml"
    ]
    subprocess.run(cmd, cwd="experiments")


def run_hyperparameter_sweep():
    """Run hyperparameter sweep for MLP."""
    # Different hidden sizes
    hidden_sizes = [128, 256, 512, 1024]
    
    for hidden_size in hidden_sizes:
        print(f"\n{'='*60}")
        print(f"Running MLP with hidden_size={hidden_size}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, "scripts/train_mlp.py",
            "--config", "configs/mlp_basic.yaml",
            "--hidden_size", str(hidden_size)
        ]
        subprocess.run(cmd, cwd="experiments")


def run_maxorder_comparison():
    """Compare different maxorder values."""
    maxorders = [3, 5, 10, 15]
    
    for maxorder in maxorders:
        print(f"\n{'='*60}")
        print(f"Running with maxorder={maxorder}")
        print(f"{'='*60}")
        
        # Run MLP
        cmd = [
            sys.executable, "scripts/train_mlp.py",
            "--config", "configs/mlp_basic.yaml", 
            "--maxorder", str(maxorder)
        ]
        subprocess.run(cmd, cwd="experiments")
        
        # Run baseline
        cmd = [
            sys.executable, "scripts/train_baseline.py",
            "--config", "configs/baseline_comparison.yaml",
            "--maxorder", str(maxorder)
        ]
        subprocess.run(cmd, cwd="experiments")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run ML experiments")
    parser.add_argument("experiment", choices=[
        "mlp-basic", "mlp-large", "baseline", 
        "hyperparameter-sweep", "maxorder-comparison"
    ], help="Type of experiment to run")
    
    args = parser.parse_args()
    
    # Change to experiments directory
    experiments_dir = Path(__file__).parent.parent
    print(f"Running experiments from: {experiments_dir}")
    
    if args.experiment == "mlp-basic":
        run_mlp_basic()
    elif args.experiment == "mlp-large":
        run_mlp_large()
    elif args.experiment == "baseline":
        run_baseline()
    elif args.experiment == "hyperparameter-sweep":
        run_hyperparameter_sweep()
    elif args.experiment == "maxorder-comparison":
        run_maxorder_comparison()


if __name__ == "__main__":
    main()