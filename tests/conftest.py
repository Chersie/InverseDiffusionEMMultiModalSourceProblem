"""
Pytest configuration and shared fixtures.

This module provides common test fixtures and configuration for the entire
test suite, ensuring consistent test environments and data.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Tuple

import numpy as np
import pytest

from src.core.config import Config, MLConfig, PipelineConfig
from src.core.data_generator import DataGenerator, LatinSquareConfig, GridConfig
from src.core.library_manager import LibraryManager, reset_library_manager
from src.models.registry import get_model_registry, reset_model_registry


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture(scope="session")
def test_config() -> Config:
    """Create test configuration with safe defaults."""
    return Config()


@pytest.fixture(scope="session")
def test_ml_config() -> MLConfig:
    """Create test ML configuration."""
    return MLConfig(
        n_samples=100,  # Small for fast tests
        epochs=2,       # Minimal training
        batch_size=16,  # Small batches
        seed=42
    )


@pytest.fixture(scope="session")
def test_pipeline_config() -> PipelineConfig:
    """Create test pipeline configuration."""
    return PipelineConfig(
        default_maxorder=3,  # Small for fast tests
        library_type="fast"
    )


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def small_maxorder() -> int:
    """Small maxorder for fast testing."""
    return 3


@pytest.fixture(scope="session")
def test_grid_shape() -> Tuple[int, int]:
    """Small grid shape for testing."""
    return (8, 6)  # 8 phi x 6 theta (much smaller than 360x179)


@pytest.fixture(scope="session")
def synthetic_coefficients(small_maxorder: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic multipole coefficients."""
    # Create Latin square generator
    latin_config = LatinSquareConfig(mode="fixed", scale=0.1)
    generator = DataGenerator.for_pipeline()
    generator.latin_generator.config = latin_config
    
    # Generate coefficient arrays
    a_e, a_m = generator.latin_generator.generate_coefficients_array(small_maxorder, sample_id=0)
    
    return a_e, a_m


@pytest.fixture(scope="session") 
def synthetic_field_data(test_grid_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Generate synthetic electromagnetic field data."""
    n_phi, n_theta = test_grid_shape
    
    # Create synthetic field amplitudes
    amplitude = np.random.random((n_phi, n_theta, 2)) + 1j * np.random.random((n_phi, n_theta, 2))
    amplitude = amplitude.astype(np.complex64)
    
    # Create coordinate grids
    phi = np.arange(n_phi) * (360.0 / n_phi)
    theta = (np.arange(n_theta) + 1) * (180.0 / (n_theta + 1))
    
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
    
    # Compute power
    power = np.sum(np.abs(amplitude) ** 2, axis=-1)
    
    return {
        "amplitude": amplitude,
        "power": power,
        "theta": theta_grid,
        "phi": phi_grid,
        "n_phi": n_phi,
        "n_theta": n_theta
    }


@pytest.fixture(scope="session")
def ml_dataset_small(small_maxorder: int) -> Dict[str, np.ndarray]:
    """Generate small ML dataset for testing."""
    n_samples = 50
    n_modes = small_maxorder * (small_maxorder + 2)
    n_features = 64  # Small PCA size
    
    # Generate synthetic features (PCA-compressed field data)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate synthetic targets (packed coefficients)
    a_e = (np.random.randn(n_samples, n_modes) + 1j * np.random.randn(n_samples, n_modes)).astype(np.complex64)
    a_m = (np.random.randn(n_samples, n_modes) + 1j * np.random.randn(n_samples, n_modes)).astype(np.complex64)
    
    # Pack coefficients
    y = np.concatenate([a_e.real, a_e.imag, a_m.real, a_m.imag], axis=1).astype(np.float32)
    
    # Create splits
    n_train = int(0.7 * n_samples)
    n_val = int(0.2 * n_samples)
    
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_samples)
    
    return {
        "X": X,
        "y": y,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "n_modes": n_modes
    }


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_model_dir(temp_dir: Path) -> Path:
    """Temporary directory for model artifacts."""
    model_dir = temp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_data_dir(temp_dir: Path) -> Path:
    """Temporary directory for data files."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# =============================================================================
# Mock Library Fixtures
# =============================================================================

@pytest.fixture
def mock_library_data(small_maxorder: int, test_grid_shape: Tuple[int, int]) -> Dict[str, Any]:
    """Generate mock library mode data."""
    n_phi, n_theta = test_grid_shape
    
    library_modes = {}
    
    # Generate modes for each (l, m) pair
    for l in range(1, small_maxorder + 1):
        for m in range(-l, l + 1):
            # Create synthetic mode data
            amplitude = np.random.random((n_phi, n_theta, 2)) + 1j * np.random.random((n_phi, n_theta, 2))
            theta_grid = np.ones((n_phi, n_theta)) * np.arange(n_theta)[None, :] * (180.0 / (n_theta + 1))
            
            library_modes[('E', l, m)] = (amplitude.astype(np.complex64), theta_grid.astype(np.float32))
            library_modes[('M', l, m)] = (amplitude.astype(np.complex64), theta_grid.astype(np.float32))
    
    return {
        "modes": library_modes,
        "maxorder": small_maxorder,
        "grid_shape": test_grid_shape
    }


@pytest.fixture
def mock_library_files(temp_data_dir: Path, mock_library_data: Dict[str, Any]) -> Path:
    """Create mock library files for testing."""
    library_dir = temp_data_dir / "test_library"
    library_dir.mkdir(parents=True, exist_ok=True)
    
    modes = mock_library_data["modes"]
    maxorder = mock_library_data["maxorder"]
    n_phi, n_theta = mock_library_data["grid_shape"]
    
    # Create library files
    for (mode_type, l, m), (amplitude, theta) in modes.items():
        filename = f"{mode_type}_l{l}_m{m}.txt"
        filepath = library_dir / filename
        
        with open(filepath, 'w') as f:
            # Write header lines (simplified)
            for i in range(43):
                f.write(f"# Header line {i}\n")
            
            # Write data lines: theta phi power |E_theta| phase |E_phi| phase
            for j in range(n_phi):
                for i in range(n_theta):
                    theta_val = theta[j, i]
                    phi_val = j * (360.0 / n_phi)
                    
                    E_theta = amplitude[j, i, 0]
                    E_phi = amplitude[j, i, 1]
                    
                    power = abs(E_theta)**2 + abs(E_phi)**2
                    
                    f.write(f"{theta_val} {phi_val} {power} "
                           f"{abs(E_theta)} {np.angle(E_theta)} "
                           f"{abs(E_phi)} {np.angle(E_phi)}\n")
    
    return library_dir


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state between tests."""
    # Reset registries to ensure clean state
    reset_library_manager()
    reset_model_registry()
    yield
    # Cleanup after test
    reset_library_manager() 
    reset_model_registry()


# =============================================================================
# Skip Markers
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require GPU (skip if no GPU available)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "requires_sklearn: marks tests that require scikit-learn"
    )
    config.addinivalue_line(
        "markers",
        "requires_torch: marks tests that require PyTorch"
    )


# =============================================================================
# Test Utilities
# =============================================================================

def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = ""
) -> None:
    """Assert that two arrays are close with informative error messages."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)
    except AssertionError as e:
        # Add shape information to error message
        shape_msg = f"Shapes: actual={actual.shape}, expected={expected.shape}"
        if actual.shape == expected.shape:
            diff_stats = f"Max abs diff: {np.max(np.abs(actual - expected)):.2e}"
            shape_msg += f", {diff_stats}"
        raise AssertionError(f"{msg}\n{shape_msg}") from e


def assert_complex_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = ""
) -> None:
    """Assert that two complex arrays are close."""
    assert_arrays_close(actual.real, expected.real, rtol, atol, f"{msg} (real part)")
    assert_arrays_close(actual.imag, expected.imag, rtol, atol, f"{msg} (imag part)")


# Add utility functions to pytest namespace
pytest.assert_arrays_close = assert_arrays_close
pytest.assert_complex_arrays_close = assert_complex_arrays_close