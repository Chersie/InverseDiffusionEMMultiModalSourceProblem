#!/usr/bin/env python3
"""
MLFlow Platform Startup Script

Comprehensive script to start the complete MLFlow platform with:
- MLFlow tracking server
- Model registry
- Custom dashboards
- Model serving capabilities
- Monitoring and management utilities
"""

import sys
import os
import time
import subprocess
import threading
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.start_mlflow_server import MLFlowServerManager
from src.core.mlflow_config import MLFlowConfig, get_mlflow_manager
from src.api.model_serving import ModelServingConfig, create_model_server
from experiments.utils.mlflow_dashboards import DashboardConfig, create_dashboard

logger = logging.getLogger(__name__)


class MLFlowPlatformManager:
    """Manages the complete MLFlow platform stack."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file
        self.project_root = Path(__file__).parent.parent
        
        # Component managers
        self.server_manager: Optional[MLFlowServerManager] = None
        self.model_server = None
        self.dashboard_process = None
        
        # Process tracking
        self.processes = {}
        self.running = False
        
        # Load configuration
        self.load_configuration()
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_configuration(self):
        """Load platform configuration."""
        
        # MLFlow server configuration
        self.mlflow_config = MLFlowConfig.from_env()
        
        # Model serving configuration
        self.serving_config = ModelServingConfig(
            host="0.0.0.0",
            port=8001,  # Different from MLFlow UI port
            default_model_name="electromagnetic_multipole_model",
            auto_reload_models=True
        )
        
        # Dashboard configuration
        self.dashboard_config = DashboardConfig(
            experiment_names=["electromagnetic_multipole_analysis"],
            auto_refresh=True,
            refresh_interval=30
        )
        
        logger.info("Platform configuration loaded")
    
    def start_mlflow_server(self) -> bool:
        """Start MLFlow tracking server."""
        
        logger.info("Starting MLFlow tracking server...")
        
        self.server_manager = MLFlowServerManager()
        
        def run_server():
            try:
                self.server_manager.start_server(background=False)
            except Exception as e:
                logger.error(f"MLFlow server failed: {e}")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        for _ in range(30):
            if self.server_manager.is_server_running():
                logger.info("✅ MLFlow tracking server started successfully")
                return True
            time.sleep(1)
        
        logger.error("❌ MLFlow server failed to start")
        return False
    
    def start_model_serving(self) -> bool:
        """Start model serving API."""
        
        logger.info("Starting model serving API...")
        
        try:
            self.model_server = create_model_server(self.serving_config)
            
            def run_model_server():
                try:
                    self.model_server.run()
                except Exception as e:
                    logger.error(f"Model server failed: {e}")
            
            server_thread = threading.Thread(target=run_model_server, daemon=True)
            server_thread.start()
            
            # Give it a moment to start
            time.sleep(3)
            logger.info(f"✅ Model serving API started on port {self.serving_config.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start model serving: {e}")
            return False
    
    def start_dashboard(self) -> bool:
        """Start custom dashboard."""
        
        logger.info("Starting custom dashboard...")
        
        try:
            # Start Streamlit dashboard in subprocess
            dashboard_script = self.project_root / "experiments" / "utils" / "mlflow_dashboards.py"
            
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_script),
                "--server.port", "8502",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            
            # Give it time to start
            time.sleep(5)
            
            if self.dashboard_process.poll() is None:
                logger.info("✅ Custom dashboard started on port 8502")
                return True
            else:
                logger.error("❌ Dashboard process exited early")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        
        dependencies = {
            "mlflow": "MLFlow tracking and registry",
            "fastapi": "Model serving API",
            "streamlit": "Custom dashboards",
            "plotly": "Interactive visualizations",
            "optuna": "Hyperparameter optimization"
        }
        
        missing = []
        
        for package, description in dependencies.items():
            try:
                __import__(package)
            except ImportError:
                missing.append(f"{package} ({description})")
        
        if missing:
            logger.error("Missing required dependencies:")
            for dep in missing:
                logger.error(f"  - {dep}")
            logger.error("Please install missing dependencies and try again")
            return False
        
        logger.info("✅ All dependencies available")
        return True
    
    def print_platform_info(self):
        """Print information about the running platform."""
        
        print("\n" + "="*60)
        print("🚀 MLFLOW PLATFORM SUCCESSFULLY STARTED")
        print("="*60)
        
        # MLFlow UI
        mlflow_url = f"http://{self.mlflow_config.server.host}:{self.mlflow_config.server.port}"
        print(f"📊 MLFlow UI:        {mlflow_url}")
        
        # Model serving API
        serving_url = f"http://{self.serving_config.host}:{self.serving_config.port}"
        print(f"🤖 Model API:        {serving_url}")
        print(f"📋 API Docs:         {serving_url}/docs")
        
        # Custom dashboard
        dashboard_url = "http://localhost:8502"
        print(f"📈 Dashboard:        {dashboard_url}")
        
        print("\n🔧 Platform Components:")
        print("  ✅ MLFlow Tracking Server")
        print("  ✅ MLFlow Model Registry") 
        print("  ✅ Model Serving API")
        print("  ✅ Custom Visualization Dashboard")
        print("  ✅ Experiment Tracking")
        print("  ✅ Hyperparameter Optimization")
        print("  ✅ Model Deployment System")
        
        print("\n💡 Quick Start:")
        print("  1. Visit MLFlow UI to view experiments")
        print("  2. Use the Dashboard for advanced analytics")
        print("  3. Test the Model API at /docs endpoint")
        print("  4. Run training scripts - they'll auto-log to MLFlow")
        
        print("\n⚠️  To stop the platform:")
        print("  Press Ctrl+C or run: python scripts/start_mlflow_platform.py --stop")
        print("="*60)
    
    def start_platform(self) -> bool:
        """Start the complete MLFlow platform."""
        
        logger.info("Starting MLFlow Platform...")
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        self.running = True
        
        # Start components in sequence
        components = [
            ("MLFlow Server", self.start_mlflow_server),
            ("Model Serving", self.start_model_serving),
            ("Dashboard", self.start_dashboard)
        ]
        
        for component_name, start_func in components:
            if not self.running:  # Check if we've been asked to stop
                break
                
            success = start_func()
            if not success:
                logger.error(f"Failed to start {component_name}")
                self.stop_platform()
                return False
            
            # Brief pause between components
            time.sleep(2)
        
        if self.running:
            self.print_platform_info()
            return True
        else:
            return False
    
    def stop_platform(self):
        """Stop all platform components."""
        
        logger.info("Stopping MLFlow Platform...")
        self.running = False
        
        # Stop dashboard
        if self.dashboard_process and self.dashboard_process.poll() is None:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                logger.info("✅ Dashboard stopped")
            except:
                self.dashboard_process.kill()
        
        # Stop MLFlow server
        if self.server_manager:
            self.server_manager.stop_server()
        
        # Model server stops automatically (daemon thread)
        
        logger.info("✅ Platform stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_platform()
        sys.exit(0)
    
    def run(self):
        """Run the platform and wait for shutdown."""
        
        if self.start_platform():
            try:
                # Keep main thread alive
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")
            finally:
                self.stop_platform()
        else:
            logger.error("Failed to start platform")
            sys.exit(1)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="MLFlow Platform Manager")
    parser.add_argument("--config", type=Path, help="Configuration file")
    parser.add_argument("--stop", action="store_true", help="Stop running platform")
    parser.add_argument("--status", action="store_true", help="Show platform status")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create platform manager
    platform = MLFlowPlatformManager(args.config)
    
    if args.stop:
        # Stop any running platform
        platform.stop_platform()
        return
    
    if args.status:
        # Show status
        mlflow_manager = get_mlflow_manager()
        info = mlflow_manager.get_server_info()
        
        print("\n📊 MLFlow Platform Status")
        print("="*40)
        print(f"MLFlow Available: {info['available']}")
        print(f"Server Running: {info['server_running']}")
        print(f"Tracking URI: {info.get('tracking_uri', 'N/A')}")
        print(f"Experiments: {info.get('experiment_count', 0)}")
        print(f"Runs: {info.get('run_count', 0)}")
        
        if 'error' in info:
            print(f"Error: {info['error']}")
        
        return
    
    # Start platform
    logger.info("🚀 Starting MLFlow Platform...")
    platform.run()


if __name__ == "__main__":
    main()