#!/usr/bin/env python3
"""
MLFlow Server Management Script

This script sets up and manages the MLFlow tracking server with PostgreSQL backend
and artifact storage for the electromagnetic multipole ML pipeline.
"""

import sys
import os
import subprocess
import time
import psutil
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.dependencies import print_environment_info

class MLFlowServerManager:
    """Manages MLFlow server lifecycle and configuration."""
    
    def __init__(self, config_path: Path = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "mlflow_server.conf"
        
        # Default configuration
        self.config = {
            'host': '127.0.0.1',
            'port': 5000,
            'backend_store_uri': 'sqlite:///mlflow.db',  # Will upgrade to PostgreSQL
            'default_artifact_root': str(self.project_root / 'mlartifacts'),
            'serve_artifacts': True,
            'workers': 1
        }
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file if it exists."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            # Convert string values to appropriate types
                            if value.lower() in ('true', 'false'):
                                value = value.lower() == 'true'
                            elif value.isdigit():
                                value = int(value)
                            self.config[key] = value
                print(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                print(f"Warning: Failed to load config: {e}")
                print("Using default configuration")
    
    def save_config(self):
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            f.write("# MLFlow Server Configuration\n")
            f.write("# Generated automatically - modify as needed\n\n")
            for key, value in self.config.items():
                f.write(f"{key}={value}\n")
        print(f"Configuration saved to {self.config_path}")
    
    def is_server_running(self) -> bool:
        """Check if MLFlow server is already running on the configured port."""
        try:
            for conn in psutil.net_connections():
                if (conn.laddr.port == self.config['port'] and 
                    conn.laddr.ip == self.config['host'] and
                    conn.status == 'LISTEN'):
                    return True
        except (psutil.AccessDenied, OSError):
            pass
        return False
    
    def setup_database(self):
        """Set up database backend (SQLite for now, PostgreSQL option available)."""
        db_uri = self.config['backend_store_uri']
        
        if db_uri.startswith('sqlite:'):
            # Ensure SQLite database directory exists
            db_path = db_uri.replace('sqlite:///', '')
            if not db_path.startswith('/'):
                db_path = self.project_root / db_path
            db_path = Path(db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Using SQLite database: {db_path}")
            
        elif db_uri.startswith('postgresql:'):
            print("PostgreSQL backend detected - ensure database is running")
            # Note: PostgreSQL setup would go here for production
            
        else:
            print(f"Using database backend: {db_uri}")
    
    def setup_artifact_storage(self):
        """Set up artifact storage directory."""
        artifact_root = Path(self.config['default_artifact_root'])
        artifact_root.mkdir(parents=True, exist_ok=True)
        print(f"Artifact storage: {artifact_root}")
    
    def start_server(self, background: bool = False):
        """Start the MLFlow tracking server."""
        if self.is_server_running():
            print(f"MLFlow server already running on {self.config['host']}:{self.config['port']}")
            return
        
        # Setup database and storage
        self.setup_database()
        self.setup_artifact_storage()
        
        # Build MLFlow server command
        cmd = [
            sys.executable, "-m", "mlflow", "server",
            "--host", self.config['host'],
            "--port", str(self.config['port']),
            "--backend-store-uri", self.config['backend_store_uri'],
            "--default-artifact-root", self.config['default_artifact_root']
        ]
        
        if self.config['serve_artifacts']:
            cmd.append("--serve-artifacts")
        
        if self.config.get('workers', 1) > 1:
            cmd.extend(["--workers", str(self.config['workers'])])
        
        print("Starting MLFlow server...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Web UI will be available at: http://{self.config['host']}:{self.config['port']}")
        
        if background:
            # Start in background
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            print(f"Server started in background (PID: {process.pid})")
            
            # Wait a moment and check if server started successfully
            time.sleep(3)
            if self.is_server_running():
                print("✅ MLFlow server started successfully")
            else:
                print("❌ Server may have failed to start")
                stdout, stderr = process.communicate(timeout=1)
                if stderr:
                    print(f"Error: {stderr.decode()}")
            
        else:
            # Start in foreground
            try:
                subprocess.run(cmd, cwd=self.project_root, check=True)
            except KeyboardInterrupt:
                print("\n🛑 Server stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"❌ Server failed to start: {e}")
    
    def stop_server(self):
        """Stop the MLFlow server if running."""
        if not self.is_server_running():
            print("MLFlow server is not running")
            return
        
        # Find and kill MLFlow processes
        killed = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if ('mlflow' in proc.info['name'] or 
                    any('mlflow' in cmd for cmd in proc.info['cmdline'] or [])):
                    print(f"Stopping MLFlow process (PID: {proc.info['pid']})")
                    proc.terminate()
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if killed:
            time.sleep(2)
            if not self.is_server_running():
                print("✅ MLFlow server stopped successfully")
            else:
                print("⚠️ Server may still be running")
        else:
            print("No MLFlow processes found")
    
    def status(self):
        """Show server status and configuration."""
        print("🔧 MLFlow Server Status")
        print("=" * 50)
        
        # Server status
        if self.is_server_running():
            print("✅ Server: RUNNING")
            print(f"   URL: http://{self.config['host']}:{self.config['port']}")
        else:
            print("❌ Server: STOPPED")
        
        # Configuration
        print("\n📋 Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        
        # Database status
        print(f"\n💾 Database: {self.config['backend_store_uri']}")
        if self.config['backend_store_uri'].startswith('sqlite:'):
            db_path = self.config['backend_store_uri'].replace('sqlite:///', '')
            if not db_path.startswith('/'):
                db_path = self.project_root / db_path
            if Path(db_path).exists():
                size = Path(db_path).stat().st_size
                print(f"   Size: {size:,} bytes")
            else:
                print("   Status: Not created yet")
        
        # Artifact storage
        artifact_path = Path(self.config['default_artifact_root'])
        print(f"\n📁 Artifacts: {artifact_path}")
        if artifact_path.exists():
            try:
                files = len(list(artifact_path.rglob('*')))
                print(f"   Files: {files}")
            except:
                print("   Status: Unable to count files")
        else:
            print("   Status: Directory not created yet")


def main():
    parser = argparse.ArgumentParser(description="Manage MLFlow tracking server")
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'], 
                       help="Action to perform")
    parser.add_argument('--background', '-b', action='store_true',
                       help="Start server in background")
    parser.add_argument('--config', '-c', type=Path,
                       help="Configuration file path")
    parser.add_argument('--port', '-p', type=int, default=5000,
                       help="Server port (default: 5000)")
    parser.add_argument('--host', default='127.0.0.1',
                       help="Server host (default: 127.0.0.1)")
    
    args = parser.parse_args()
    
    # Create server manager
    manager = MLFlowServerManager(args.config)
    
    # Override config with command line args
    if args.port != 5000:
        manager.config['port'] = args.port
    if args.host != '127.0.0.1':
        manager.config['host'] = args.host
    
    # Print environment info
    print("🔧 MLFlow Server Manager")
    print("=" * 50)
    print_environment_info()
    print()
    
    # Execute action
    if args.action == 'start':
        manager.start_server(background=args.background)
        if args.background:
            print(f"\n💡 To view experiments, open: http://{manager.config['host']}:{manager.config['port']}")
            print("💡 To stop server: python scripts/start_mlflow_server.py stop")
    
    elif args.action == 'stop':
        manager.stop_server()
    
    elif args.action == 'restart':
        manager.stop_server()
        time.sleep(2)
        manager.start_server(background=args.background)
    
    elif args.action == 'status':
        manager.status()
    
    # Save config for future use
    manager.save_config()


if __name__ == "__main__":
    main()