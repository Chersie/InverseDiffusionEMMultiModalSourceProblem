"""
Custom MLFlow Dashboards and Visualization Interfaces

Enhanced visualization and analysis tools for electromagnetic multipole
ML experiments with domain-specific dashboards and interactive plots.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None
    MLFLOW_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    import streamlit.components.v1 as components
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.mlflow_config import get_mlflow_manager
from src.models.mlflow_integration import get_model_registry

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for custom dashboards."""
    
    # MLFlow connection
    tracking_uri: Optional[str] = None
    experiment_names: List[str] = None
    
    # Dashboard settings
    title: str = "Electromagnetic Multipole ML Dashboard"
    theme: str = "plotly_white"  # plotly theme
    auto_refresh: bool = True
    refresh_interval: int = 30  # seconds
    
    # Visualization settings
    max_runs_displayed: int = 100
    default_metrics: List[str] = None
    
    # Analysis settings
    enable_statistical_analysis: bool = True
    enable_model_comparison: bool = True
    enable_hyperparameter_analysis: bool = True
    
    def __post_init__(self):
        if self.experiment_names is None:
            self.experiment_names = ["electromagnetic_multipole_analysis"]
        if self.default_metrics is None:
            self.default_metrics = ["test_mse", "test_r2", "training_time"]


class MLFlowAnalyzer:
    """Analyzes MLFlow experiments and generates insights."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.mlflow_manager = get_mlflow_manager()
        self.client = self.mlflow_manager.client
        
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLFlow not available for dashboard analysis")
    
    def get_experiments_data(self) -> pd.DataFrame:
        """Get experiment data as DataFrame."""
        
        experiments_data = []
        
        for exp_name in self.config.experiment_names:
            try:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if experiment is None:
                    logger.warning(f"Experiment not found: {exp_name}")
                    continue
                
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=self.config.max_runs_displayed,
                    order_by=["start_time DESC"]
                )
                
                if not runs.empty:
                    runs['experiment_name'] = exp_name
                    experiments_data.append(runs)
                    
            except Exception as e:
                logger.error(f"Failed to get data for experiment {exp_name}: {e}")
        
        if experiments_data:
            return pd.concat(experiments_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def analyze_hyperparameter_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze hyperparameter importance using correlation with target metrics."""
        
        importance_scores = {}
        
        # Find hyperparameter columns (usually start with 'params.')
        param_cols = [col for col in df.columns if col.startswith('params.')]
        
        if not param_cols:
            return importance_scores
        
        # Use primary metric for analysis
        target_metric = f"metrics.{self.config.default_metrics[0]}"
        if target_metric not in df.columns:
            return importance_scores
        
        for param_col in param_cols:
            try:
                # Convert parameter values to numeric if possible
                param_values = pd.to_numeric(df[param_col], errors='coerce')
                metric_values = pd.to_numeric(df[target_metric], errors='coerce')
                
                # Calculate correlation (use absolute value for importance)
                correlation = param_values.corr(metric_values)
                if not np.isnan(correlation):
                    param_name = param_col.replace('params.', '')
                    importance_scores[param_name] = abs(correlation)
                    
            except Exception as e:
                logger.debug(f"Could not analyze parameter {param_col}: {e}")
        
        return importance_scores
    
    def get_best_runs(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get best performing runs based on primary metric."""
        
        if df.empty:
            return df
        
        primary_metric = f"metrics.{self.config.default_metrics[0]}"
        if primary_metric in df.columns:
            # Assume lower is better for MSE-like metrics
            if 'mse' in primary_metric.lower() or 'loss' in primary_metric.lower():
                return df.nsmallest(n, primary_metric)
            else:
                return df.nlargest(n, primary_metric)
        
        return df.head(n)
    
    def analyze_training_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze training trends over time."""
        
        trends = {}
        
        if df.empty or 'start_time' not in df.columns:
            return trends
        
        # Convert start_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Group by day and calculate average metrics
        daily_metrics = df.groupby(df['start_time'].dt.date).agg({
            f"metrics.{metric}": 'mean' for metric in self.config.default_metrics 
            if f"metrics.{metric}" in df.columns
        }).reset_index()
        
        trends['daily_averages'] = daily_metrics.to_dict('records')
        trends['total_runs'] = len(df)
        trends['date_range'] = {
            'start': df['start_time'].min().isoformat(),
            'end': df['start_time'].max().isoformat()
        }
        
        return trends


class PlotlyVisualizer:
    """Creates interactive Plotly visualizations for MLFlow data."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available. Install with: pip install plotly")
    
    def create_metrics_comparison_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive metrics comparison plot."""
        
        if df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        fig = make_subplots(
            rows=len(self.config.default_metrics), 
            cols=1,
            subplot_titles=[f"{metric.replace('_', ' ').title()}" for metric in self.config.default_metrics],
            vertical_spacing=0.08
        )
        
        for i, metric in enumerate(self.config.default_metrics, 1):
            metric_col = f"metrics.{metric}"
            if metric_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[metric_col],
                        mode='lines+markers',
                        name=metric,
                        hovertemplate=f"{metric}: %{{y:.6f}}<br>Run: %{{x}}<extra></extra>"
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title="Metrics Comparison Across Runs",
            height=300 * len(self.config.default_metrics),
            showlegend=False,
            template=self.config.theme
        )
        
        return fig
    
    def create_hyperparameter_analysis_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create hyperparameter importance and correlation plot."""
        
        analyzer = MLFlowAnalyzer(self.config)
        importance_scores = analyzer.analyze_hyperparameter_importance(df)
        
        if not importance_scores:
            return go.Figure().add_annotation(text="No hyperparameter data available", x=0.5, y=0.5)
        
        # Sort by importance
        sorted_params = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        params, scores = zip(*sorted_params) if sorted_params else ([], [])
        
        fig = go.Figure(data=go.Bar(
            x=list(params),
            y=list(scores),
            text=[f"{score:.3f}" for score in scores],
            textposition='auto',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title="Hyperparameter Importance (Correlation with Primary Metric)",
            xaxis_title="Hyperparameters",
            yaxis_title="Importance Score (|Correlation|)",
            template=self.config.theme
        )
        
        return fig
    
    def create_training_timeline_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create training timeline visualization."""
        
        if df.empty or 'start_time' not in df.columns:
            return go.Figure().add_annotation(text="No timeline data available", x=0.5, y=0.5)
        
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # Create timeline plot
        fig = go.Figure()
        
        primary_metric = f"metrics.{self.config.default_metrics[0]}"
        if primary_metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df['start_time'],
                y=df[primary_metric],
                mode='lines+markers',
                marker=dict(
                    size=8,
                    color=df[primary_metric],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=self.config.default_metrics[0])
                ),
                hovertemplate=(
                    f"Time: %{{x}}<br>"
                    f"{self.config.default_metrics[0]}: %{{y:.6f}}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="Training Progress Over Time",
            xaxis_title="Training Time",
            yaxis_title=self.config.default_metrics[0],
            template=self.config.theme
        )
        
        return fig
    
    def create_model_performance_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create performance heatmap for hyperparameter combinations."""
        
        if df.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Find categorical hyperparameters for heatmap
        param_cols = [col for col in df.columns if col.startswith('params.')]
        
        if len(param_cols) < 2:
            return go.Figure().add_annotation(text="Need at least 2 hyperparameters", x=0.5, y=0.5)
        
        # Use first two categorical parameters
        x_param = param_cols[0]
        y_param = param_cols[1]
        
        # Create pivot table for heatmap
        primary_metric = f"metrics.{self.config.default_metrics[0]}"
        if primary_metric not in df.columns:
            return go.Figure().add_annotation(text="Primary metric not found", x=0.5, y=0.5)
        
        pivot_data = df.pivot_table(
            index=y_param,
            columns=x_param,
            values=primary_metric,
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            text=np.round(pivot_data.values, 4),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title=self.config.default_metrics[0])
        ))
        
        fig.update_layout(
            title=f"Performance Heatmap: {x_param.replace('params.', '')} vs {y_param.replace('params.', '')}",
            xaxis_title=x_param.replace('params.', ''),
            yaxis_title=y_param.replace('params.', ''),
            template=self.config.theme
        )
        
        return fig
    
    def create_experiment_summary_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive experiment summary dashboard."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Metrics Trend",
                "Best Runs Comparison", 
                "Hyperparameter Importance",
                "Training Timeline"
            ),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "bar"}, {"secondary_y": False}]]
        )
        
        if not df.empty:
            # Top left: Metrics trend
            primary_metric = f"metrics.{self.config.default_metrics[0]}"
            if primary_metric in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index[:20], y=df[primary_metric][:20], mode='lines+markers'),
                    row=1, col=1
                )
            
            # Top right: Best runs
            best_runs = df.head(10)
            if not best_runs.empty and primary_metric in best_runs.columns:
                fig.add_trace(
                    go.Bar(x=best_runs.index, y=best_runs[primary_metric]),
                    row=1, col=2
                )
            
            # Bottom left: Hyperparameter importance
            analyzer = MLFlowAnalyzer(self.config)
            importance = analyzer.analyze_hyperparameter_importance(df)
            if importance:
                params, scores = zip(*list(importance.items())[:10])
                fig.add_trace(
                    go.Bar(x=list(params), y=list(scores)),
                    row=2, col=1
                )
            
            # Bottom right: Timeline
            if 'start_time' in df.columns:
                df['start_time'] = pd.to_datetime(df['start_time'])
                if primary_metric in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['start_time'], y=df[primary_metric], mode='markers'),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title="Experiment Summary Dashboard",
            height=800,
            showlegend=False,
            template=self.config.theme
        )
        
        return fig


class StreamlitDashboard:
    """Streamlit-based interactive dashboard for MLFlow experiments."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.analyzer = MLFlowAnalyzer(config)
        self.visualizer = PlotlyVisualizer(config)
        
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not available. Install with: pip install streamlit")
    
    def create_dashboard(self):
        """Create interactive Streamlit dashboard."""
        
        st.set_page_config(
            page_title=self.config.title,
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title(self.config.title)
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Dashboard Controls")
            
            # Experiment selection
            selected_experiments = st.multiselect(
                "Select Experiments",
                self.config.experiment_names,
                default=self.config.experiment_names
            )
            
            # Metric selection
            selected_metrics = st.multiselect(
                "Select Metrics",
                self.config.default_metrics,
                default=self.config.default_metrics
            )
            
            # Number of runs to display
            max_runs = st.slider(
                "Max Runs to Display",
                min_value=10,
                max_value=500,
                value=self.config.max_runs_displayed,
                step=10
            )
            
            # Auto refresh
            auto_refresh = st.checkbox("Auto Refresh", value=self.config.auto_refresh)
            
            if st.button("Refresh Data"):
                st.rerun()
        
        # Update config based on selections
        self.config.experiment_names = selected_experiments
        self.config.default_metrics = selected_metrics
        self.config.max_runs_displayed = max_runs
        
        # Load data
        with st.spinner("Loading experiment data..."):
            df = self.analyzer.get_experiments_data()
        
        if df.empty:
            st.error("No experiment data found. Please check your MLFlow server and experiment names.")
            return
        
        # Main dashboard content
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", len(df))
        
        with col2:
            if f"metrics.{selected_metrics[0]}" in df.columns:
                best_score = df[f"metrics.{selected_metrics[0]}"].min()  # Assuming lower is better
                st.metric(f"Best {selected_metrics[0]}", f"{best_score:.6f}")
        
        with col3:
            recent_runs = df[df['start_time'] > (datetime.now() - timedelta(days=7))]
            st.metric("Runs (Last 7 Days)", len(recent_runs))
        
        with col4:
            avg_training_time = df.get('metrics.training_time', pd.Series()).mean()
            if not pd.isna(avg_training_time):
                st.metric("Avg Training Time", f"{avg_training_time:.1f}s")
        
        st.markdown("---")
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Metrics Comparison",
            "🔍 Hyperparameter Analysis", 
            "📈 Training Timeline",
            "🌡️ Performance Heatmap",
            "📋 Best Runs"
        ])
        
        with tab1:
            st.subheader("Metrics Comparison Across Runs")
            metrics_plot = self.visualizer.create_metrics_comparison_plot(df)
            st.plotly_chart(metrics_plot, use_container_width=True)
        
        with tab2:
            st.subheader("Hyperparameter Importance Analysis")
            hyperparam_plot = self.visualizer.create_hyperparameter_analysis_plot(df)
            st.plotly_chart(hyperparam_plot, use_container_width=True)
            
            # Show importance table
            importance = self.analyzer.analyze_hyperparameter_importance(df)
            if importance:
                importance_df = pd.DataFrame(
                    list(importance.items()), 
                    columns=['Parameter', 'Importance Score']
                ).sort_values('Importance Score', ascending=False)
                st.dataframe(importance_df, use_container_width=True)
        
        with tab3:
            st.subheader("Training Progress Timeline")
            timeline_plot = self.visualizer.create_training_timeline_plot(df)
            st.plotly_chart(timeline_plot, use_container_width=True)
        
        with tab4:
            st.subheader("Hyperparameter Performance Heatmap")
            heatmap_plot = self.visualizer.create_model_performance_heatmap(df)
            st.plotly_chart(heatmap_plot, use_container_width=True)
        
        with tab5:
            st.subheader("Best Performing Runs")
            best_runs = self.analyzer.get_best_runs(df, n=20)
            
            # Display columns of interest
            display_cols = ['run_id', 'experiment_name'] + [
                col for col in best_runs.columns 
                if col.startswith('metrics.') or col.startswith('params.')
            ]
            display_cols = [col for col in display_cols if col in best_runs.columns]
            
            st.dataframe(best_runs[display_cols], use_container_width=True)
        
        # Auto refresh
        if auto_refresh:
            import time
            time.sleep(self.config.refresh_interval)
            st.rerun()


def create_dashboard(config: Optional[DashboardConfig] = None) -> StreamlitDashboard:
    """Create and return a dashboard instance."""
    config = config or DashboardConfig()
    return StreamlitDashboard(config)


def export_dashboard_html(config: Optional[DashboardConfig] = None, 
                         output_path: Path = Path("mlflow_dashboard.html")):
    """Export dashboard as standalone HTML file."""
    
    config = config or DashboardConfig()
    analyzer = MLFlowAnalyzer(config)
    visualizer = PlotlyVisualizer(config)
    
    # Get data
    df = analyzer.get_experiments_data()
    
    if df.empty:
        logger.warning("No data available for dashboard export")
        return
    
    # Create visualizations
    plots = {
        'metrics_comparison': visualizer.create_metrics_comparison_plot(df),
        'hyperparameter_analysis': visualizer.create_hyperparameter_analysis_plot(df),
        'timeline': visualizer.create_training_timeline_plot(df),
        'heatmap': visualizer.create_model_performance_heatmap(df),
        'summary': visualizer.create_experiment_summary_dashboard(df)
    }
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{config.title}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .plot-container {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>{config.title}</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Experiment Summary</h2>
        <div id="summary" class="plot-container"></div>
        
        <h2>Metrics Comparison</h2>
        <div id="metrics" class="plot-container"></div>
        
        <h2>Hyperparameter Analysis</h2>
        <div id="hyperparams" class="plot-container"></div>
        
        <h2>Training Timeline</h2>
        <div id="timeline" class="plot-container"></div>
        
        <h2>Performance Heatmap</h2>
        <div id="heatmap" class="plot-container"></div>
        
        <script>
    """
    
    for plot_id, plot in plots.items():
        plot_json = plot.to_json()
        if plot_id == 'summary':
            html_content += f"Plotly.newPlot('summary', {plot_json});\n"
        elif plot_id == 'metrics_comparison':
            html_content += f"Plotly.newPlot('metrics', {plot_json});\n"
        elif plot_id == 'hyperparameter_analysis':
            html_content += f"Plotly.newPlot('hyperparams', {plot_json});\n"
        elif plot_id == 'timeline':
            html_content += f"Plotly.newPlot('timeline', {plot_json});\n"
        elif plot_id == 'heatmap':
            html_content += f"Plotly.newPlot('heatmap', {plot_json});\n"
    
    html_content += """
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Dashboard exported to: {output_path}")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MLFlow Dashboard")
    parser.add_argument("--mode", choices=["streamlit", "export"], default="streamlit",
                       help="Dashboard mode")
    parser.add_argument("--experiments", nargs="+", 
                       default=["electromagnetic_multipole_analysis"],
                       help="Experiment names to include")
    parser.add_argument("--output", type=Path, default="mlflow_dashboard.html",
                       help="Output file for export mode")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = DashboardConfig(experiment_names=args.experiments)
    
    if args.mode == "streamlit":
        dashboard = create_dashboard(config)
        dashboard.create_dashboard()
    else:
        export_dashboard_html(config, args.output)