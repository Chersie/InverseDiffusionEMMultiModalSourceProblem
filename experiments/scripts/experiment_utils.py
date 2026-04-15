"""
Experiment Utilities

Common utilities for running experiments, comparing models, and analyzing results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def compare_experiments(experiment_names: List[str], metrics: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple experiments using MLflow.
    
    Args:
        experiment_names: List of experiment names to compare
        metrics: List of metrics to compare (default: common metrics)
        
    Returns:
        DataFrame with experiment comparison
    """
    if metrics is None:
        metrics = ['test_mse', 'test_r2', 'training_time', 'final_train_loss', 'final_val_loss']
    
    client = MlflowClient()
    comparison_data = []
    
    for exp_name in experiment_names:
        try:
            experiment = client.get_experiment_by_name(exp_name)
            if experiment is None:
                logger.warning(f"Experiment '{exp_name}' not found")
                continue
            
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            
            for run in runs:
                run_data = {
                    'experiment_name': exp_name,
                    'run_id': run.info.run_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                }
                
                # Add parameters
                for key, value in run.data.params.items():
                    run_data[f'param_{key}'] = value
                
                # Add metrics
                for metric in metrics:
                    if metric in run.data.metrics:
                        run_data[metric] = run.data.metrics[metric]
                    else:
                        run_data[metric] = None
                
                comparison_data.append(run_data)
        
        except Exception as e:
            logger.error(f"Error processing experiment '{exp_name}': {e}")
    
    return pd.DataFrame(comparison_data)


def plot_training_comparison(experiment_names: List[str], save_path: Optional[Path] = None):
    """Plot comparison of training metrics across experiments."""
    df = compare_experiments(experiment_names)
    
    if df.empty:
        logger.warning("No data found for comparison")
        return
    
    # Filter successful runs
    df_success = df[df['status'] == 'FINISHED'].copy()
    
    if df_success.empty:
        logger.warning("No successful runs found")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Test MSE comparison
    if 'test_mse' in df_success.columns:
        axes[0, 0].boxplot([df_success[df_success['experiment_name'] == name]['test_mse'].dropna() 
                           for name in experiment_names], 
                          labels=experiment_names)
        axes[0, 0].set_title('Test MSE Comparison')
        axes[0, 0].set_ylabel('Test MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Test R² comparison
    if 'test_r2' in df_success.columns:
        axes[0, 1].boxplot([df_success[df_success['experiment_name'] == name]['test_r2'].dropna()
                           for name in experiment_names],
                          labels=experiment_names)
        axes[0, 1].set_title('Test R² Comparison')
        axes[0, 1].set_ylabel('Test R²')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    if 'training_time' in df_success.columns:
        axes[1, 0].boxplot([df_success[df_success['experiment_name'] == name]['training_time'].dropna()
                           for name in experiment_names],
                          labels=experiment_names)
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Training Time (s)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Parameter comparison (example: hidden_size for MLP)
    if 'param_hidden_size' in df_success.columns:
        for i, name in enumerate(experiment_names):
            exp_data = df_success[df_success['experiment_name'] == name]
            if not exp_data.empty and 'test_mse' in exp_data.columns:
                axes[1, 1].scatter(exp_data['param_hidden_size'].astype(float), 
                                 exp_data['test_mse'], 
                                 label=name, alpha=0.7)
        axes[1, 1].set_xlabel('Hidden Size')
        axes[1, 1].set_ylabel('Test MSE')
        axes[1, 1].set_title('Hidden Size vs Performance')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    return fig


def get_best_runs(experiment_name: str, metric: str = 'test_mse', n_best: int = 5) -> pd.DataFrame:
    """Get best runs from an experiment based on a metric."""
    df = compare_experiments([experiment_name])
    
    if df.empty or metric not in df.columns:
        logger.warning(f"No data found for experiment '{experiment_name}' or metric '{metric}'")
        return pd.DataFrame()
    
    # Filter successful runs and sort by metric
    df_success = df[df['status'] == 'FINISHED'].copy()
    df_success = df_success.dropna(subset=[metric])
    
    # Sort by metric (ascending for loss metrics, descending for accuracy metrics)
    ascending = 'mse' in metric.lower() or 'loss' in metric.lower()
    df_best = df_success.nsmallest(n_best, metric) if ascending else df_success.nlargest(n_best, metric)
    
    return df_best


def create_experiment_report(experiment_name: str, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Create a comprehensive experiment report."""
    df = compare_experiments([experiment_name])
    
    if df.empty:
        logger.warning(f"No data found for experiment '{experiment_name}'")
        return {}
    
    # Filter successful runs
    df_success = df[df['status'] == 'FINISHED'].copy()
    
    if df_success.empty:
        logger.warning("No successful runs found")
        return {}
    
    report = {
        'experiment_name': experiment_name,
        'total_runs': len(df),
        'successful_runs': len(df_success),
        'failed_runs': len(df) - len(df_success),
    }
    
    # Metrics summary
    metrics_cols = [col for col in df_success.columns if not col.startswith('param_') and 
                   col not in ['experiment_name', 'run_id', 'status', 'start_time']]
    
    report['metrics_summary'] = {}
    for metric in metrics_cols:
        if metric in df_success.columns:
            values = df_success[metric].dropna()
            if not values.empty:
                report['metrics_summary'][metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'median': float(values.median())
                }
    
    # Best runs
    if 'test_mse' in df_success.columns:
        best_runs = get_best_runs(experiment_name, 'test_mse', n_best=3)
        report['best_runs'] = best_runs[['run_id', 'test_mse', 'test_r2', 'training_time']].to_dict('records')
    
    # Parameter analysis
    param_cols = [col for col in df_success.columns if col.startswith('param_')]
    report['parameter_analysis'] = {}
    
    for param_col in param_cols:
        param_name = param_col.replace('param_', '')
        values = df_success[param_col].dropna()
        if not values.empty:
            try:
                # Try to convert to numeric for analysis
                numeric_values = pd.to_numeric(values, errors='coerce').dropna()
                if not numeric_values.empty:
                    report['parameter_analysis'][param_name] = {
                        'unique_values': len(values.unique()),
                        'most_common': values.value_counts().index[0],
                        'numeric_mean': float(numeric_values.mean()),
                        'numeric_std': float(numeric_values.std())
                    }
            except:
                report['parameter_analysis'][param_name] = {
                    'unique_values': len(values.unique()),
                    'most_common': values.value_counts().index[0]
                }
    
    # Save report
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Experiment report saved to {output_path}")
    
    return report


def analyze_hyperparameter_impact(experiment_name: str, param_name: str, metric: str = 'test_mse'):
    """Analyze the impact of a hyperparameter on model performance."""
    df = compare_experiments([experiment_name])
    
    if df.empty:
        logger.warning(f"No data found for experiment '{experiment_name}'")
        return
    
    param_col = f'param_{param_name}'
    if param_col not in df.columns or metric not in df.columns:
        logger.warning(f"Parameter '{param_name}' or metric '{metric}' not found")
        return
    
    # Filter successful runs
    df_success = df[df['status'] == 'FINISHED'].copy()
    df_analysis = df_success[[param_col, metric]].dropna()
    
    if df_analysis.empty:
        logger.warning("No data available for analysis")
        return
    
    # Convert parameter to numeric if possible
    try:
        df_analysis[param_col] = pd.to_numeric(df_analysis[param_col])
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df_analysis[param_col], df_analysis[metric], alpha=0.7)
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.title(f'Impact of {param_name} on {metric}')
        
        # Add trend line
        z = np.polyfit(df_analysis[param_col], df_analysis[metric], 1)
        p = np.poly1d(z)
        plt.plot(df_analysis[param_col], p(df_analysis[param_col]), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = df_analysis[param_col].corr(df_analysis[metric])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'correlation': correlation,
            'trend_line_coefficients': z.tolist(),
            'n_points': len(df_analysis)
        }
        
    except ValueError:
        # Categorical parameter
        grouped = df_analysis.groupby(param_col)[metric].agg(['mean', 'std', 'count'])
        print(f"\nImpact of {param_name} on {metric}:")
        print(grouped)
        
        # Box plot
        df_analysis.boxplot(column=metric, by=param_col, figsize=(10, 6))
        plt.title(f'Impact of {param_name} on {metric}')
        plt.suptitle('')
        plt.show()
        
        return grouped.to_dict()


def load_experiment_config(experiment_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration from directory."""
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cleanup_failed_experiments(experiment_name: str, dry_run: bool = True):
    """Clean up failed experiment runs."""
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        failed_runs = [run for run in runs if run.info.status == 'FAILED']
        
        logger.info(f"Found {len(failed_runs)} failed runs")
        
        if not dry_run:
            for run in failed_runs:
                client.delete_run(run.info.run_id)
                logger.info(f"Deleted failed run: {run.info.run_id}")
        else:
            logger.info("Dry run - no runs deleted")
            for run in failed_runs:
                logger.info(f"Would delete: {run.info.run_id}")
                
    except Exception as e:
        logger.error(f"Error cleaning up experiments: {e}")