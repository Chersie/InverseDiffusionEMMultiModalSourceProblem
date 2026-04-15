"""
Dependency Management and Optional Import Handling.

This module provides utilities for managing optional dependencies and 
graceful degradation when certain packages are not available.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Optional Import Management
# =============================================================================

class OptionalDependency:
    """
    Context manager for optional dependencies with graceful error handling.
    
    Usage:
        with OptionalDependency("torch", "PyTorch") as torch:
            if torch is not None:
                # Use torch normally
                model = torch.nn.Linear(10, 1)
            else:
                # Fallback behavior
                raise RuntimeError("PyTorch required for neural network training")
    """
    
    def __init__(
        self, 
        module_name: str, 
        display_name: Optional[str] = None,
        install_command: Optional[str] = None,
        min_version: Optional[str] = None
    ):
        self.module_name = module_name
        self.display_name = display_name or module_name
        self.install_command = install_command or f"pip install {module_name}"
        self.min_version = min_version
        self._module = None
        self._attempted_import = False
    
    def __enter__(self) -> Optional[Any]:
        """Attempt to import the module."""
        if not self._attempted_import:
            try:
                self._module = importlib.import_module(self.module_name)
                
                # Check version if specified
                if self.min_version and hasattr(self._module, '__version__'):
                    from packaging import version
                    if version.parse(self._module.__version__) < version.parse(self.min_version):
                        warnings.warn(
                            f"{self.display_name} version {self._module.__version__} "
                            f"is below minimum required version {self.min_version}"
                        )
                
            except ImportError:
                self._module = None
            
            self._attempted_import = True
        
        return self._module
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass
    
    def require(self) -> Any:
        """
        Require the dependency to be available, raising informative error if not.
        
        Returns:
            The imported module
            
        Raises:
            ImportError: If the dependency is not available
        """
        with self as module:
            if module is None:
                raise ImportError(
                    f"{self.display_name} is required for this functionality.\n"
                    f"Install it with: {self.install_command}\n"
                    f"Or install ML dependencies: pip install -r requirements-ml.txt"
                )
            return module


# =============================================================================
# Common Dependencies
# =============================================================================

# PyTorch dependencies
TORCH = OptionalDependency(
    "torch", 
    "PyTorch", 
    "pip install -r requirements-ml.txt",
    min_version="2.2.0"
)

TORCHVISION = OptionalDependency(
    "torchvision",
    "TorchVision", 
    "pip install -r requirements-ml.txt"
)

# Other optional dependencies
PANDAS = OptionalDependency(
    "pandas",
    "Pandas",
    "pip install pandas"
)

PYVISTA = OptionalDependency(
    "pyvista",
    "PyVista", 
    "pip install pyvista"
)

MLFLOW = OptionalDependency(
    "mlflow",
    "MLflow",
    "pip install mlflow"
)


# =============================================================================
# Decorators for Optional Dependencies 
# =============================================================================

def requires_torch(func: F) -> F:
    """Decorator to require PyTorch for a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        TORCH.require()
        return func(*args, **kwargs)
    return wrapper


def requires_ml_dependencies(func: F) -> F:
    """Decorator to require ML dependencies (torch) for a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        TORCH.require()
        return func(*args, **kwargs)
    return wrapper


def optional_torch(fallback_return: Any = None):
    """
    Decorator for functions that can optionally use PyTorch.
    
    Args:
        fallback_return: Value to return if PyTorch is not available
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with TORCH as torch:
                if torch is not None:
                    return func(*args, **kwargs)
                else:
                    warnings.warn(
                        f"PyTorch not available, {func.__name__} returning fallback value"
                    )
                    return fallback_return
        return wrapper
    return decorator


# =============================================================================
# Environment Detection
# =============================================================================

def get_available_ml_backends() -> list[str]:
    """Get list of available ML backends."""
    backends = []
    
    with TORCH as torch:
        if torch is not None:
            backends.append("pytorch")
            
            # Check for device availability
            if torch.cuda.is_available():
                backends.append("cuda")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                backends.append("mps")
    
    return backends


def get_compute_device() -> str:
    """
    Get the best available compute device.
    
    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    with TORCH as torch:
        if torch is not None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
    
    return "cpu"


def check_ml_environment() -> dict[str, Any]:
    """
    Check the ML environment and return status information.
    
    Returns:
        Dictionary with environment information
    """
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "available_backends": get_available_ml_backends(),
        "compute_device": get_compute_device(),
        "dependencies": {}
    }
    
    # Check each dependency
    dependencies = [
        ("torch", TORCH),
        ("pandas", PANDAS), 
        ("mlflow", MLFLOW),
        ("pyvista", PYVISTA)
    ]
    
    for name, dep in dependencies:
        with dep as module:
            if module is not None:
                version = getattr(module, '__version__', 'unknown')
                info["dependencies"][name] = {
                    "available": True,
                    "version": version
                }
            else:
                info["dependencies"][name] = {
                    "available": False,
                    "install_command": dep.install_command
                }
    
    return info


def print_environment_info() -> None:
    """Print environment information in a readable format."""
    info = check_ml_environment()
    
    print("=== ML Pipeline Environment ===")
    print(f"Python: {info['python_version']}")
    print(f"Compute Device: {info['compute_device']}")
    print(f"Available Backends: {', '.join(info['available_backends']) or 'None'}")
    
    print("\n=== Dependencies ===")
    for name, dep_info in info["dependencies"].items():
        if dep_info["available"]:
            print(f"✓ {name}: {dep_info['version']}")
        else:
            print(f"✗ {name}: Not installed ({dep_info['install_command']})")
    
    print()


# =============================================================================
# Validation Functions
# =============================================================================

def validate_ml_environment(require_torch: bool = True) -> None:
    """
    Validate that the ML environment is properly set up.
    
    Args:
        require_torch: Whether to require PyTorch to be available
        
    Raises:
        ImportError: If required dependencies are missing
    """
    info = check_ml_environment()
    missing = []
    
    if require_torch and not info["dependencies"]["torch"]["available"]:
        missing.append("PyTorch (install with: pip install -r requirements-ml.txt)")
    
    if not info["dependencies"]["pandas"]["available"]:
        missing.append("Pandas (install with: pip install pandas)")
    
    if missing:
        raise ImportError(
            "Missing required dependencies:\n" + 
            "\n".join(f"- {dep}" for dep in missing)
        )


if __name__ == "__main__":
    # Print environment info when run as script
    print_environment_info()