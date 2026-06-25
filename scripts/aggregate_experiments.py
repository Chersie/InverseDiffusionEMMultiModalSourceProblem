#!/usr/bin/env python3
"""Aggregate experiment results from multirun directories into tables for Chapter 6."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_metrics(path: Path) -> dict[str, Any] | None:
    """Load metrics.json file."""
    try:
        with path.open("r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def extract_val_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract validation metrics from metrics dict."""
    val_metrics = {}
    for key, value in metrics.items():
        if key.startswith("report/val/"):
            val_metrics[key.replace("report/val/", "")] = value
    return val_metrics


def format_float(value: float, decimals: int = 4) -> str:
    """Format float value with specified decimals."""
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def aggregate_step1() -> None:
    """Aggregate Step 1 results (loss functions)."""
    print("\n" + "=" * 80)
    print("ТАБЛИЦА 6.1. Результаты исследования функций потерь (Step 1)")
    print("=" * 80)
    
    # Step 1 has 4 sub-experiments: pure, mixed, rank, combined
    step1_dirs = {
        "pure": "multirun/final_step1_pure",
        "mixed": "multirun/final_step1_mixed", 
        "rank": "multirun/final_step1_rank",
        "combined": "multirun/final_step1_combined",
    }
    
    # Collect all results
    all_results = []
    
    for exp_name, exp_dir in step1_dirs.items():
        exp_path = Path(exp_dir)
        if not exp_path.exists():
            continue
            
        for run_dir in sorted(exp_path.iterdir()):
            if not run_dir.is_dir():
                continue
                
            metrics_path = run_dir / "metrics.json"
            metrics = load_metrics(metrics_path)
            if metrics is None:
                continue
                
            # Extract key metrics
            val_metrics = extract_val_metrics(metrics)
            
            # Get hyperparameters from the path or metrics
            # For now, use the directory structure
            result = {
                "experiment": exp_name,
                "run_id": run_dir.name,
                "field_nrmse_w": val_metrics.get("field_nrmse_w", np.nan),
                "spearman_rho_P": val_metrics.get("spearman_rho_P", np.nan),
                "bin_accuracy_P": val_metrics.get("bin_accuracy_P", np.nan),
                "coef_mse": val_metrics.get("coef_mse", np.nan),
                "coef_mse_amb_aware": val_metrics.get("coef_mse_amb_aware", np.nan),
            }
            
            # Calculate composite score
            field_nrmse = result["field_nrmse_w"]
            spearman = result["spearman_rho_P"]
            if not np.isnan(field_nrmse) and not np.isnan(spearman):
                result["composite"] = field_nrmse - 0.5 * spearman
            else:
                result["composite"] = np.nan
                
            all_results.append(result)
    
    # Sort by composite score
    all_results.sort(key=lambda x: x["composite"] if not np.isnan(x["composite"]) else float("inf"))
    
    # Print table
    print(f"\n{'Эксперимент':<15} {'field_nrmse_w':<15} {'spearman_rho_P':<15} {'bin_accuracy_P':<15} {'coef_mse':<15} {'Composite':<15}")
    print("-" * 90)
    
    for i, result in enumerate(all_results, 1):
        print(f"{result['experiment']:<15} "
              f"{format_float(result['field_nrmse_w']):<15} "
              f"{format_float(result['spearman_rho_P']):<15} "
              f"{format_float(result['bin_accuracy_P']):<15} "
              f"{format_float(result['coef_mse']):<15} "
              f"{format_float(result['composite']):<15}")
    
    print(f"\nВсего результатов: {len(all_results)}")
    
    # Find best
    if all_results:
        best = min(all_results, key=lambda x: x["composite"] if not np.isnan(x["composite"]) else float("inf"))
        print(f"\nЛучшая конфигурация: {best['experiment']}")
        print(f"  Composite: {format_float(best['composite'])}")
        print(f"  field_nrmse_w: {format_float(best['field_nrmse_w'])}")
        print(f"  spearman_rho_P: {format_float(best['spearman_rho_P'])}")


def aggregate_step2() -> None:
    """Aggregate Step 2 results (features)."""
    print("\n" + "=" * 80)
    print("ТАБЛИЦА 6.2. Результаты исследования признаковых представлений (Step 2)")
    print("=" * 80)
    
    # Step 2 features: raw_flat, raw_plus_sh, power_pca, power_pca_small, pca_cv, cv_only, subsample_stride4
    step2_dir = Path("multirun/final_step2_features")
    
    if not step2_dir.exists():
        print(f"Директория {step2_dir} не найдена")
        return
    
    # Feature names correspond to run IDs 0-6
    feature_names = [
        "raw_flat",
        "raw_plus_sh", 
        "power_pca",
        "power_pca_small",
        "pca_cv",
        "cv_only",
        "subsample_stride4",
    ]
    
    all_results = []
    
    for run_dir in sorted(step2_dir.iterdir()):
        if not run_dir.is_dir():
            continue
            
        metrics_path = run_dir / "metrics.json"
        metrics = load_metrics(metrics_path)
        if metrics is None:
            continue
            
        run_id = int(run_dir.name)
        if run_id >= len(feature_names):
            continue
            
        feature_name = feature_names[run_id]
        
        val_metrics = extract_val_metrics(metrics)
        
        result = {
            "feature": feature_name,
            "run_id": run_id,
            "field_nrmse_w": val_metrics.get("field_nrmse_w", np.nan),
            "spearman_rho_P": val_metrics.get("spearman_rho_P", np.nan),
            "bin_accuracy_P": val_metrics.get("bin_accuracy_P", np.nan),
            "coef_mse": val_metrics.get("coef_mse", np.nan),
            "coef_mse_amb_aware": val_metrics.get("coef_mse_amb_aware", np.nan),
        }
        
        # Calculate composite score
        field_nrmse = result["field_nrmse_w"]
        spearman = result["spearman_rho_P"]
        if not np.isnan(field_nrmse) and not np.isnan(spearman):
            result["composite"] = field_nrmse - 0.5 * spearman
        else:
            result["composite"] = np.nan
            
        all_results.append(result)
    
    # Sort by composite score
    all_results.sort(key=lambda x: x["composite"] if not np.isnan(x["composite"]) else float("inf"))
    
    # Print table
    print(f"\n{'Признаки':<20} {'field_nrmse_w':<15} {'spearman_rho_P':<15} {'bin_accuracy_P':<15} {'coef_mse':<15} {'Composite':<15}")
    print("-" * 95)
    
    for i, result in enumerate(all_results, 1):
        # Highlight best values
        field_nrmse_str = format_float(result["field_nrmse_w"])
        spearman_str = format_float(result["spearman_rho_P"])
        bin_acc_str = format_float(result["bin_accuracy_P"])
        coef_mse_str = format_float(result["coef_mse"])
        composite_str = format_float(result["composite"])
        
        if i == 1:  # Best composite
            field_nrmse_str = f"**{field_nrmse_str}**"
            spearman_str = f"**{spearman_str}**"
            bin_acc_str = f"**{bin_acc_str}**"
            composite_str = f"**{composite_str}**"
        
        print(f"{result['feature']:<20} "
              f"{field_nrmse_str:<15} "
              f"{spearman_str:<15} "
              f"{bin_acc_str:<15} "
              f"{coef_mse_str:<15} "
              f"{composite_str:<15}")
    
    print(f"\nВсего результатов: {len(all_results)}")
    
    # Find best
    if all_results:
        best = min(all_results, key=lambda x: x["composite"] if not np.isnan(x["composite"]) else float("inf"))
        print(f"\nЛучшая конфигурация: {best['feature']}")
        print(f"  Composite: {format_float(best['composite'])}")
        print(f"  field_nrmse_w: {format_float(best['field_nrmse_w'])}")
        print(f"  spearman_rho_P: {format_float(best['spearman_rho_P'])}")


def aggregate_step3() -> None:
    """Aggregate Step 3 results (scheduling)."""
    print("\n" + "=" * 80)
    print("ТАБЛИЦА 6.3. Результаты исследования расписания обучения (Step 3)")
    print("=" * 80)
    
    # Step 3: 3 backbone policies × 2 truncation modes = 6 cells
    step3_dir = Path("multirun/final_step3_scheduling")
    
    if not step3_dir.exists():
        print(f"Директория {step3_dir} не найдена")
        return
    
    # Configuration names correspond to run IDs 0-5
    # Order: freeze_after_stage1 × {true, false}, trainable_always × {true, false}, all_trainable_active_boost × {true, false}
    config_names = [
        ("freeze_after_stage1", True),
        ("freeze_after_stage1", False),
        ("trainable_always", True),
        ("trainable_always", False),
        ("all_trainable_active_boost", True),
        ("all_trainable_active_boost", False),
    ]
    
    all_results = []
    
    for run_dir in sorted(step3_dir.iterdir()):
        if not run_dir.is_dir():
            continue
            
        metrics_path = run_dir / "metrics.json"
        metrics = load_metrics(metrics_path)
        if metrics is None:
            continue
            
        run_id = int(run_dir.name)
        if run_id >= len(config_names):
            continue
            
        policy, truncate = config_names[run_id]
        
        val_metrics = extract_val_metrics(metrics)
        
        result = {
            "policy": policy,
            "truncate_target": truncate,
            "run_id": run_id,
            "field_nrmse_w": val_metrics.get("field_nrmse_w", np.nan),
            "spearman_rho_P": val_metrics.get("spearman_rho_P", np.nan),
            "bin_accuracy_P": val_metrics.get("bin_accuracy_P", np.nan),
            "coef_mse": val_metrics.get("coef_mse", np.nan),
            "coef_mse_amb_aware": val_metrics.get("coef_mse_amb_aware", np.nan),
        }
        
        # Calculate composite score
        field_nrmse = result["field_nrmse_w"]
        spearman = result["spearman_rho_P"]
        if not np.isnan(field_nrmse) and not np.isnan(spearman):
            result["composite"] = field_nrmse - 0.5 * spearman
        else:
            result["composite"] = np.nan
            
        all_results.append(result)
    
    # Sort by composite score
    all_results.sort(key=lambda x: x["composite"] if not np.isnan(x["composite"]) else float("inf"))
    
    # Print table
    print(f"\n{'Политика':<30} {'Truncate':<10} {'field_nrmse_w':<15} {'spearman_rho_P':<15} {'bin_accuracy_P':<15} {'Composite':<15}")
    print("-" * 100)
    
    for i, result in enumerate(all_results, 1):
        # Highlight best values
        field_nrmse_str = format_float(result["field_nrmse_w"])
        spearman_str = format_float(result["spearman_rho_P"])
        bin_acc_str = format_float(result["bin_accuracy_P"])
        composite_str = format_float(result["composite"])
        
        if i == 1:  # Best composite
            field_nrmse_str = f"**{field_nrmse_str}**"
            spearman_str = f"**{spearman_str}**"
            bin_acc_str = f"**{bin_acc_str}**"
            composite_str = f"**{composite_str}**"
        
        truncate_str = "Да" if result["truncate_target"] else "Нет"
        
        print(f"{result['policy']:<30} {truncate_str:<10} "
              f"{field_nrmse_str:<15} "
              f"{spearman_str:<15} "
              f"{bin_acc_str:<15} "
              f"{composite_str:<15}")
    
    print(f"\nВсего результатов: {len(all_results)}")
    
    # Find best
    if all_results:
        best = min(all_results, key=lambda x: x["composite"] if not np.isnan(x["composite"]) else float("inf"))
        print(f"\nЛучшая конфигурация: {best['policy']}, truncate={best['truncate_target']}")
        print(f"  Composite: {format_float(best['composite'])}")
        print(f"  field_nrmse_w: {format_float(best['field_nrmse_w'])}")
        print(f"  spearman_rho_P: {format_float(best['spearman_rho_P'])}")


def main() -> None:
    """Main function."""
    print("Агрегация результатов экспериментов для Главы 6")
    print("=" * 80)
    
    aggregate_step1()
    aggregate_step2()
    aggregate_step3()
    
    print("\n" + "=" * 80)
    print("Агрегация завершена")
    print("=" * 80)


if __name__ == "__main__":
    main()
