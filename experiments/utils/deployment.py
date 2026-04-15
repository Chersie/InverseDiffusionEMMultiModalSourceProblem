"""
Deployment Utilities for MLFlow Model Serving

Utilities for deploying models to various environments with automated
configuration and monitoring setup.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import yaml
import subprocess
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.model_serving import ModelServingConfig, DeploymentManager
from src.core.mlflow_config import get_mlflow_manager

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    
    # Model information
    model_name: str
    model_version: Optional[str] = None
    model_stage: str = "Production"
    
    # Deployment target
    target: str = "local"  # "local", "docker", "kubernetes", "cloud"
    
    # Server configuration
    server_config: ModelServingConfig = None
    
    # Environment configuration
    environment: str = "production"  # "development", "staging", "production"
    
    # Monitoring
    enable_monitoring: bool = True
    enable_health_checks: bool = True
    
    # Scaling
    min_replicas: int = 1
    max_replicas: int = 3
    auto_scaling: bool = False
    
    # Resources
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    
    def __post_init__(self):
        if self.server_config is None:
            self.server_config = ModelServingConfig()


class DockerDeployment:
    """Handles Docker-based model deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_dockerfile(self, output_path: Path) -> Path:
        """Generate Dockerfile for model serving."""
        
        dockerfile_content = f"""
# MLFlow Model Serving Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY experiments/utils/ ./experiments/utils/

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI={get_mlflow_manager().config.get_tracking_uri()}
ENV MODEL_NAME={self.config.model_name}
ENV MODEL_STAGE={self.config.model_stage}

# Expose port
EXPOSE {self.config.server_config.port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{self.config.server_config.port}/health || exit 1

# Run the model server
CMD ["python", "-m", "src.api.model_serving", \\
     "--model-name", "{self.config.model_name}", \\
     "--model-stage", "{self.config.model_stage}", \\
     "--host", "0.0.0.0", \\
     "--port", "{self.config.server_config.port}"]
"""
        
        dockerfile_path = output_path / "Dockerfile"
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content.strip())
        
        logger.info(f"Generated Dockerfile: {dockerfile_path}")
        return dockerfile_path
    
    def generate_docker_compose(self, output_path: Path) -> Path:
        """Generate docker-compose.yml for model serving."""
        
        compose_config = {
            'version': '3.8',
            'services': {
                'model-server': {
                    'build': '.',
                    'ports': [f"{self.config.server_config.port}:{self.config.server_config.port}"],
                    'environment': {
                        'MLFLOW_TRACKING_URI': get_mlflow_manager().config.get_tracking_uri(),
                        'MODEL_NAME': self.config.model_name,
                        'MODEL_STAGE': self.config.model_stage,
                        'ENVIRONMENT': self.config.environment
                    },
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': f"curl -f http://localhost:{self.config.server_config.port}/health || exit 1",
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    }
                }
            }
        }
        
        if self.config.enable_monitoring:
            # Add monitoring services
            compose_config['services']['prometheus'] = {
                'image': 'prom/prometheus:latest',
                'ports': ['9090:9090'],
                'volumes': ['./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml'],
                'restart': 'unless-stopped'
            }
            
            compose_config['services']['grafana'] = {
                'image': 'grafana/grafana:latest',
                'ports': ['3000:3000'],
                'environment': {
                    'GF_SECURITY_ADMIN_PASSWORD': 'admin'
                },
                'volumes': [
                    'grafana-storage:/var/lib/grafana',
                    './monitoring/grafana:/etc/grafana/provisioning'
                ],
                'restart': 'unless-stopped'
            }
            
            compose_config['volumes'] = {
                'grafana-storage': {}
            }
        
        compose_path = output_path / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated docker-compose.yml: {compose_path}")
        return compose_path
    
    def build_image(self, context_path: Path, tag: Optional[str] = None) -> bool:
        """Build Docker image for model serving."""
        
        tag = tag or f"{self.config.model_name}-server:latest"
        
        try:
            cmd = ["docker", "build", "-t", tag, str(context_path)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info(f"Built Docker image: {tag}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build Docker image: {e.stderr}")
            return False
    
    def deploy(self, output_dir: Path) -> Dict[str, Any]:
        """Deploy model using Docker."""
        
        # Generate deployment files
        dockerfile = self.generate_dockerfile(output_dir)
        compose_file = self.generate_docker_compose(output_dir)
        
        # Build image
        image_built = self.build_image(output_dir)
        
        if image_built:
            # Start services
            try:
                cmd = ["docker-compose", "-f", str(compose_file), "up", "-d"]
                subprocess.run(cmd, check=True, cwd=output_dir)
                
                return {
                    "status": "deployed",
                    "target": "docker",
                    "files": [str(dockerfile), str(compose_file)],
                    "endpoint": f"http://localhost:{self.config.server_config.port}",
                    "deployment_time": datetime.now().isoformat()
                }
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start Docker services: {e}")
                return {"status": "failed", "error": str(e)}
        else:
            return {"status": "failed", "error": "Docker image build failed"}


class KubernetesDeployment:
    """Handles Kubernetes-based model deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_deployment_yaml(self, output_path: Path) -> Path:
        """Generate Kubernetes deployment YAML."""
        
        deployment_config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f"{self.config.model_name}-server",
                'labels': {
                    'app': f"{self.config.model_name}-server",
                    'version': self.config.model_version or 'latest',
                    'environment': self.config.environment
                }
            },
            'spec': {
                'replicas': self.config.min_replicas,
                'selector': {
                    'matchLabels': {
                        'app': f"{self.config.model_name}-server"
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': f"{self.config.model_name}-server"
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'model-server',
                            'image': f"{self.config.model_name}-server:latest",
                            'ports': [{'containerPort': self.config.server_config.port}],
                            'env': [
                                {'name': 'MODEL_NAME', 'value': self.config.model_name},
                                {'name': 'MODEL_STAGE', 'value': self.config.model_stage},
                                {'name': 'ENVIRONMENT', 'value': self.config.environment}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': self.config.cpu_request,
                                    'memory': self.config.memory_request
                                },
                                'limits': {
                                    'cpu': self.config.cpu_limit,
                                    'memory': self.config.memory_limit
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': self.config.server_config.port
                                },
                                'initialDelaySeconds': 60,
                                'periodSeconds': 30
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health', 
                                    'port': self.config.server_config.port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            }
                        }]
                    }
                }
            }
        }
        
        deployment_path = output_path / "deployment.yaml"
        deployment_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(deployment_path, 'w') as f:
            yaml.dump(deployment_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated Kubernetes deployment: {deployment_path}")
        return deployment_path
    
    def generate_service_yaml(self, output_path: Path) -> Path:
        """Generate Kubernetes service YAML."""
        
        service_config = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.config.model_name}-service",
                'labels': {
                    'app': f"{self.config.model_name}-server"
                }
            },
            'spec': {
                'selector': {
                    'app': f"{self.config.model_name}-server"
                },
                'ports': [{
                    'port': 80,
                    'targetPort': self.config.server_config.port,
                    'protocol': 'TCP'
                }],
                'type': 'LoadBalancer'
            }
        }
        
        service_path = output_path / "service.yaml"
        with open(service_path, 'w') as f:
            yaml.dump(service_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated Kubernetes service: {service_path}")
        return service_path
    
    def generate_hpa_yaml(self, output_path: Path) -> Path:
        """Generate Horizontal Pod Autoscaler YAML."""
        
        if not self.config.auto_scaling:
            return None
        
        hpa_config = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.config.model_name}-hpa"
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': f"{self.config.model_name}-server"
                },
                'minReplicas': self.config.min_replicas,
                'maxReplicas': self.config.max_replicas,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    }
                ]
            }
        }
        
        hpa_path = output_path / "hpa.yaml"
        with open(hpa_path, 'w') as f:
            yaml.dump(hpa_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Generated HPA: {hpa_path}")
        return hpa_path
    
    def deploy(self, output_dir: Path) -> Dict[str, Any]:
        """Deploy model to Kubernetes."""
        
        # Generate manifests
        deployment_yaml = self.generate_deployment_yaml(output_dir)
        service_yaml = self.generate_service_yaml(output_dir)
        hpa_yaml = self.generate_hpa_yaml(output_dir)
        
        files_generated = [str(deployment_yaml), str(service_yaml)]
        if hpa_yaml:
            files_generated.append(str(hpa_yaml))
        
        try:
            # Apply manifests
            for manifest in files_generated:
                cmd = ["kubectl", "apply", "-f", manifest]
                subprocess.run(cmd, check=True)
            
            return {
                "status": "deployed",
                "target": "kubernetes",
                "files": files_generated,
                "deployment_time": datetime.now().isoformat()
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy to Kubernetes: {e}")
            return {"status": "failed", "error": str(e)}


class DeploymentOrchestrator:
    """Orchestrates model deployment across different targets."""
    
    def __init__(self):
        self.deployment_manager = DeploymentManager()
    
    def create_deployment_config(self,
                               model_name: str,
                               target: str = "local",
                               environment: str = "production",
                               **kwargs) -> DeploymentConfig:
        """Create deployment configuration with defaults."""
        
        return DeploymentConfig(
            model_name=model_name,
            target=target,
            environment=environment,
            **kwargs
        )
    
    def deploy(self, config: DeploymentConfig, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Deploy model according to configuration."""
        
        output_dir = output_dir or Path(f"deployments/{config.model_name}_{config.environment}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save deployment configuration
        config_path = output_dir / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        # Deploy according to target
        if config.target == "local":
            return self.deployment_manager.deploy_model(
                model_name=config.model_name,
                version=config.model_version,
                stage=config.model_stage,
                deployment_target="local",
                config=config.server_config
            )
        
        elif config.target == "docker":
            docker_deployment = DockerDeployment(config)
            return docker_deployment.deploy(output_dir)
        
        elif config.target == "kubernetes":
            k8s_deployment = KubernetesDeployment(config)
            return k8s_deployment.deploy(output_dir)
        
        else:
            raise ValueError(f"Unsupported deployment target: {config.target}")
    
    def promote_and_deploy(self, 
                          model_name: str,
                          version: str,
                          target_stage: str = "Production",
                          deployment_config: Optional[DeploymentConfig] = None) -> Dict[str, Any]:
        """Promote model and deploy automatically."""
        
        # Use deployment manager for promotion and deployment
        return self.deployment_manager.promote_and_deploy(
            model_name=model_name,
            version=version,
            target_stage=target_stage,
            deployment_target=deployment_config.target if deployment_config else "local",
            config=deployment_config.server_config if deployment_config else None
        )


# Convenience functions
def deploy_model(model_name: str,
                target: str = "local",
                environment: str = "production",
                **kwargs) -> Dict[str, Any]:
    """Deploy a model with simple configuration."""
    
    orchestrator = DeploymentOrchestrator()
    config = orchestrator.create_deployment_config(
        model_name=model_name,
        target=target,
        environment=environment,
        **kwargs
    )
    
    return orchestrator.deploy(config)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy MLFlow model")
    parser.add_argument("model_name", help="Name of the model to deploy")
    parser.add_argument("--target", choices=["local", "docker", "kubernetes"], 
                       default="local", help="Deployment target")
    parser.add_argument("--environment", choices=["development", "staging", "production"],
                       default="production", help="Deployment environment")
    parser.add_argument("--model-version", help="Specific model version to deploy")
    parser.add_argument("--model-stage", default="Production", help="Model stage to deploy")
    parser.add_argument("--output-dir", type=Path, help="Output directory for deployment files")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    result = deploy_model(
        model_name=args.model_name,
        target=args.target,
        environment=args.environment,
        model_version=args.model_version,
        model_stage=args.model_stage
    )
    
    print(json.dumps(result, indent=2))