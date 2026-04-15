"""
Unified Data Generation System for Multipole Analysis.

This module consolidates the two different Latin-square synthetic data generators
into a single, consistent system that supports both fixed (pipeline) and 
randomized (ML training) coefficient generation modes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.core.config import Config, DataConfig, PipelineConfig
from src.core.dependencies import PANDAS, optional_torch

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

CoefficientsDict = Dict[int, Dict[int, complex]]  # {l: {m: coeff}}
CoefficientsArray = np.ndarray  # (n_modes,) complex array
ModeList = list[tuple[int, int]]  # [(l, m), ...]
FieldAmplitude = np.ndarray  # (n_phi, n_theta, 2) complex array


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass(frozen=True)
class LatinSquareConfig:
    """Configuration for Latin square coefficient generation."""
    
    # Generation mode
    mode: str = "fixed"  # "fixed" or "random"
    
    # Scaling parameters
    scale: float = 1.0
    
    # Fixed mode parameters (for reproducible pipeline generation)
    seed_e_re: int = 0
    seed_e_im: int = 1
    seed_m_re: int = 2
    seed_m_im: int = 3
    
    # Random mode parameters (for ML training diversity)
    base_seed: int = 42
    sample_id_multiplier: int = 104729
    enable_permutations: bool = True
    
    def __post_init__(self):
        if self.mode not in ("fixed", "random"):
            raise ValueError(f"Invalid mode: {self.mode}. Use 'fixed' or 'random'")


@dataclass(frozen=True) 
class GridConfig:
    """Configuration for angular grid generation."""
    
    # Grid dimensions
    n_phi: int = 360
    n_theta: int = 179  # 180 - 1 (excluding poles)
    
    # Angular step
    angle_step_deg: float = 1.0
    
    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        return self.n_phi * self.n_theta
    
    def validate(self) -> None:
        """Validate grid configuration."""
        if self.n_phi != int(360 / self.angle_step_deg):
            raise ValueError(
                f"n_phi ({self.n_phi}) inconsistent with angle_step_deg ({self.angle_step_deg})"
            )
        if self.n_theta != int(180 / self.angle_step_deg) - 1:
            raise ValueError(
                f"n_theta ({self.n_theta}) inconsistent with angle_step_deg ({self.angle_step_deg})"
            )


# =============================================================================
# Core Latin Square Generator
# =============================================================================

class LatinSquareGenerator:
    """
    Unified Latin square coefficient generator.
    
    Supports both fixed (reproducible) and random (diverse) generation modes
    while maintaining consistent algorithms and interfaces.
    """
    
    def __init__(self, config: Optional[LatinSquareConfig] = None):
        """
        Initialize the generator.
        
        Args:
            config: Latin square configuration (uses default if None)
        """
        self.config = config or LatinSquareConfig()
        logger.debug(f"Initialized LatinSquareGenerator (mode={self.config.mode})")
    
    def generate_coefficients_dict(
        self, 
        maxorder: int, 
        sample_id: int = 0
    ) -> Tuple[CoefficientsDict, CoefficientsDict]:
        """
        Generate coefficients in dictionary format {l: {m: coeff}}.
        
        Args:
            maxorder: Maximum multipole order
            sample_id: Sample identifier (used for random mode)
            
        Returns:
            Tuple of (E_coeffs, M_coeffs) dictionaries
        """
        if self.config.mode == "fixed":
            return self._generate_fixed_dict(maxorder)
        else:
            return self._generate_random_dict(maxorder, sample_id)
    
    def generate_coefficients_array(
        self, 
        maxorder: int, 
        sample_id: int = 0
    ) -> Tuple[CoefficientsArray, CoefficientsArray]:
        """
        Generate coefficients in array format (n_modes,).
        
        Args:
            maxorder: Maximum multipole order
            sample_id: Sample identifier (used for random mode)
            
        Returns:
            Tuple of (E_coeffs, M_coeffs) arrays
        """
        if self.config.mode == "fixed":
            return self._generate_fixed_array(maxorder, sample_id)
        else:
            return self._generate_random_array(maxorder, sample_id)
    
    def _generate_fixed_dict(self, maxorder: int) -> Tuple[CoefficientsDict, CoefficientsDict]:
        """Generate fixed coefficients in dictionary format."""
        n = 2 * maxorder + 1
        
        # Create Latin squares with fixed seeds
        def latin_square(seed: int) -> np.ndarray:
            ls = np.array([
                (np.arange(n) + i + seed) % n for i in range(n)
            ], dtype=np.float64)
            return self.config.scale * (2.0 * (ls + 1) / (n + 1) - 1.0)
        
        ls_e_re = latin_square(self.config.seed_e_re)
        ls_e_im = latin_square(self.config.seed_e_im)
        ls_m_re = latin_square(self.config.seed_m_re)
        ls_m_im = latin_square(self.config.seed_m_im)
        
        # Fill coefficient dictionaries
        a_e: CoefficientsDict = {}
        a_m: CoefficientsDict = {}
        
        for l in range(1, maxorder + 1):
            a_e[l] = {}
            a_m[l] = {}
            for m in range(-l, l + 1):
                row = l - 1
                col = m + maxorder
                a_e[l][m] = ls_e_re[row, col] + 1j * ls_e_im[row, col]
                a_m[l][m] = ls_m_re[row, col] + 1j * ls_m_im[row, col]
        
        return a_e, a_m
    
    def _generate_random_dict(
        self, 
        maxorder: int, 
        sample_id: int
    ) -> Tuple[CoefficientsDict, CoefficientsDict]:
        """Generate random coefficients in dictionary format."""
        # Generate arrays first, then convert to dict format
        a_e_arr, a_m_arr = self._generate_random_array(maxorder, sample_id)
        
        # Convert to dictionary format
        mode_pairs = get_mode_list(maxorder)
        a_e: CoefficientsDict = {}
        a_m: CoefficientsDict = {}
        
        for k, (l, m) in enumerate(mode_pairs):
            if l not in a_e:
                a_e[l] = {}
                a_m[l] = {}
            a_e[l][m] = a_e_arr[k]
            a_m[l][m] = a_m_arr[k]
        
        return a_e, a_m
    
    def _generate_fixed_array(
        self, 
        maxorder: int, 
        sample_id: int
    ) -> Tuple[CoefficientsArray, CoefficientsArray]:
        """Generate fixed coefficients in array format."""
        # Generate dict format first, then convert to arrays
        a_e_dict, a_m_dict = self._generate_fixed_dict(maxorder)
        
        # Convert to array format
        mode_pairs = get_mode_list(maxorder)
        n_modes = len(mode_pairs)
        
        a_e = np.zeros(n_modes, dtype=np.complex64)
        a_m = np.zeros(n_modes, dtype=np.complex64)
        
        for k, (l, m) in enumerate(mode_pairs):
            a_e[k] = a_e_dict[l][m]
            a_m[k] = a_m_dict[l][m]
        
        return a_e, a_m
    
    def _generate_random_array(
        self, 
        maxorder: int, 
        sample_id: int
    ) -> Tuple[CoefficientsArray, CoefficientsArray]:
        """Generate random coefficients in array format."""
        n = 2 * maxorder + 1
        mode_pairs = get_mode_list(maxorder)
        n_modes = len(mode_pairs)
        
        # Create deterministic but diverse RNG for this sample
        rng = np.random.default_rng(
            self.config.base_seed + sample_id * self.config.sample_id_multiplier
        )
        
        # Generate random seeds and permutations
        seed_e_re = int(rng.integers(0, n))
        seed_e_im = int(rng.integers(0, n))
        seed_m_re = int(rng.integers(0, n))
        seed_m_im = int(rng.integers(0, n))
        
        if self.config.enable_permutations:
            row_perm_e = rng.permutation(n)
            col_perm_e = rng.permutation(n)
            row_perm_m = rng.permutation(n)
            col_perm_m = rng.permutation(n)
        else:
            # Use identity permutations
            row_perm_e = np.arange(n)
            col_perm_e = np.arange(n)
            row_perm_m = np.arange(n)
            col_perm_m = np.arange(n)
        
        def latin_value(row: int, col: int, shift: int) -> float:
            """Generate Latin square value."""
            v = (col + row + shift) % n
            return self.config.scale * (2.0 * (v + 1) / (n + 1) - 1.0)
        
        # Generate coefficient arrays
        a_e = np.zeros(n_modes, dtype=np.complex64)
        a_m = np.zeros(n_modes, dtype=np.complex64)
        
        for k, (l, m) in enumerate(mode_pairs):
            row = l - 1
            col = m + maxorder
            
            # Apply permutations
            row_e = int(row_perm_e[row])
            col_e = int(col_perm_e[col])
            row_m = int(row_perm_m[row])
            col_m = int(col_perm_m[col])
            
            # Generate complex coefficients
            a_e[k] = (
                latin_value(row_e, col_e, seed_e_re) + 
                1j * latin_value(row_e, col_e, seed_e_im)
            )
            a_m[k] = (
                latin_value(row_m, col_m, seed_m_re) + 
                1j * latin_value(row_m, col_m, seed_m_im)
            )
        
        return a_e, a_m
    
    def generate_batch_arrays(
        self, 
        maxorder: int, 
        n_samples: int,
        start_sample_id: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of coefficient arrays.
        
        Args:
            maxorder: Maximum multipole order
            n_samples: Number of samples to generate
            start_sample_id: Starting sample ID
            
        Returns:
            Tuple of (E_coeffs, M_coeffs) with shape (n_samples, n_modes)
        """
        mode_pairs = get_mode_list(maxorder)
        n_modes = len(mode_pairs)
        
        a_e_batch = np.zeros((n_samples, n_modes), dtype=np.complex64)
        a_m_batch = np.zeros((n_samples, n_modes), dtype=np.complex64)
        
        for i in range(n_samples):
            sample_id = start_sample_id + i
            a_e, a_m = self.generate_coefficients_array(maxorder, sample_id)
            a_e_batch[i] = a_e
            a_m_batch[i] = a_m
        
        return a_e_batch, a_m_batch


# =============================================================================
# Field Computation
# =============================================================================

class FieldGenerator:
    """
    Unified field generation from multipole coefficients.
    
    Handles computation of electromagnetic fields from coefficients using
    the multipole expansion, supporting both pipeline and ML training workflows.
    """
    
    def __init__(
        self, 
        grid_config: Optional[GridConfig] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the field generator.
        
        Args:
            grid_config: Grid configuration (creates default if None)
            config: Main configuration (creates default if None)
        """
        self.grid_config = grid_config or GridConfig()
        self.config = config or Config()
        
        # Validate grid configuration
        self.grid_config.validate()
        
        # Cache for multipole field computation module
        self._mpfield_module: Optional[Any] = None
        
        logger.debug(f"Initialized FieldGenerator (grid={self.grid_config.n_phi}x{self.grid_config.n_theta})")
    
    def _get_mpfield_module(self) -> Any:
        """Get or load the multipole field computation module."""
        if self._mpfield_module is None:
            # Import the mpfield module from new location
            import importlib.util
            
            # Try new location first
            module_path = self.config.paths.project_root / "src" / "core" / "mpfield.py"
            if not module_path.exists():
                # Fallback to legacy location if needed
                module_path = self.config.paths.chersie_dir / "MPField_Spherical_Fast.py"
                if not module_path.exists():
                    raise FileNotFoundError(
                        f"MPField module not found in either location:\n"
                        f"  - {self.config.paths.project_root / 'src' / 'core' / 'mpfield.py'}\n"
                        f"  - {self.config.paths.chersie_dir / 'MPField_Spherical_Fast.py'}\n"
                        f"Please ensure the MPField module is available."
                    )
            
            spec = importlib.util.spec_from_file_location("mpfield_fast", module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load module: {module_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._mpfield_module = module
            
            logger.debug(f"Loaded MPField module from {module_path}")
        
        return self._mpfield_module
    
    def build_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the angular grid for field computation, aligned with torch-harmonics.
        
        This uses the SAME grid sampling as torch-harmonics to ensure that:
        1. P-fields computed from data generation exactly match
        2. P-fields computed from physics loss reconstruction  
        3. No interpolation artifacts or systematic errors in physics loss
        
        Returns:
            Tuple of (theta, phi) arrays with shape (n_phi, n_theta)
        """
        # CRITICAL: Use grid identical to torch-harmonics for proper physics loss alignment
        # This prevents the fundamental grid mismatch that was causing incorrect training
        
        # Theta: equiangular from 0 to pi (includes poles)
        theta_rad = np.linspace(0, np.pi, self.grid_config.n_theta)
        
        # Phi: equiangular from 0 to 2*pi, but exclude last point due to periodicity
        # This matches torch-harmonics convention
        phi_rad = np.linspace(0, 2*np.pi, self.grid_config.n_phi, endpoint=False)
        
        # Create 2D grids - note the phi/theta order matches our array convention
        # Shape will be (n_phi, n_theta) to match existing code expectations
        phi_2d = np.broadcast_to(phi_rad[:, np.newaxis], (self.grid_config.n_phi, self.grid_config.n_theta))
        theta_2d = np.broadcast_to(theta_rad[np.newaxis, :], (self.grid_config.n_phi, self.grid_config.n_theta))
        
        return theta_2d, phi_2d
    
    def validate_grid_alignment(self, physics_theta: np.ndarray, physics_phi: np.ndarray) -> bool:
        """
        Validate that data generation grid aligns with physics computation grid.
        
        Args:
            physics_theta: Theta grid from physics layers  
            physics_phi: Phi grid from physics layers
            
        Returns:
            True if grids are aligned within tolerance
        """
        theta_data, phi_data = self.build_grid()
        
        # Convert to same format for comparison
        theta_data_1d = theta_data[0, :]  # Extract 1D theta
        phi_data_1d = phi_data[:, 0]      # Extract 1D phi
        
        # Check shapes
        if theta_data_1d.shape != physics_theta.shape:
            logger.warning(f"Theta shape mismatch: data={theta_data_1d.shape}, physics={physics_theta.shape}")
            return False
            
        if phi_data_1d.shape != physics_phi.shape:
            logger.warning(f"Phi shape mismatch: data={phi_data_1d.shape}, physics={physics_phi.shape}")
            return False
        
        # Check values with small tolerance
        theta_close = np.allclose(theta_data_1d, physics_theta, rtol=1e-6, atol=1e-8)
        phi_close = np.allclose(phi_data_1d, physics_phi, rtol=1e-6, atol=1e-8)
        
        if not theta_close:
            logger.warning("Theta grids not aligned!")
            logger.warning(f"Data theta range: [{theta_data_1d[0]:.6f}, {theta_data_1d[-1]:.6f}]")
            logger.warning(f"Physics theta range: [{physics_theta[0]:.6f}, {physics_theta[-1]:.6f}]")
            
        if not phi_close:
            logger.warning("Phi grids not aligned!")
            logger.warning(f"Data phi range: [{phi_data_1d[0]:.6f}, {phi_data_1d[-1]:.6f}]")
            logger.warning(f"Physics phi range: [{physics_phi[0]:.6f}, {physics_phi[-1]:.6f}]")
        
        is_aligned = theta_close and phi_close
        if is_aligned:
            logger.info("✅ Data generation and physics grids are properly aligned")
        else:
            logger.error("❌ Grid alignment validation failed!")
            
        return is_aligned
    
    def compute_field_from_dict(
        self,
        a_e: CoefficientsDict,
        a_m: CoefficientsDict,
        maxorder: int
    ) -> FieldAmplitude:
        """
        Compute electromagnetic field from coefficient dictionaries.
        
        Args:
            a_e: Electric multipole coefficients {l: {m: coeff}}
            a_m: Magnetic multipole coefficients {l: {m: coeff}}
            maxorder: Maximum multipole order
            
        Returns:
            Field amplitude array with shape (n_phi, n_theta, 2)
        """
        mpfield_module = self._get_mpfield_module()
        theta, phi = self.build_grid()
        
        # Initialize amplitude array
        amplitude = np.zeros((theta.shape[0], theta.shape[1], 2), dtype=complex)
        
        # Sum over all multipole modes
        for l in range(1, maxorder + 1):
            for m in range(-l, l + 1):
                if l in a_e and m in a_e[l]:
                    # Compute E and M field contributions for this mode
                    amp_e = mpfield_module.field_for_multipole(l, m, theta, phi, electric=True)
                    amp_m = mpfield_module.field_for_multipole(l, m, theta, phi, electric=False)
                    
                    # Add weighted contributions
                    amplitude += a_e[l][m] * amp_e + a_m[l][m] * amp_m
        
        return amplitude
    
    def compute_field_from_array(
        self,
        a_e: CoefficientsArray,
        a_m: CoefficientsArray,
        maxorder: int
    ) -> FieldAmplitude:
        """
        Compute electromagnetic field from coefficient arrays.
        
        Args:
            a_e: Electric coefficients array (n_modes,)
            a_m: Magnetic coefficients array (n_modes,)
            maxorder: Maximum multipole order
            
        Returns:
            Field amplitude array with shape (n_phi, n_theta, 2)
        """
        # Convert arrays to dictionary format
        mode_pairs = get_mode_list(maxorder)
        
        a_e_dict: CoefficientsDict = {}
        a_m_dict: CoefficientsDict = {}
        
        for k, (l, m) in enumerate(mode_pairs):
            if l not in a_e_dict:
                a_e_dict[l] = {}
                a_m_dict[l] = {}
            a_e_dict[l][m] = a_e[k]
            a_m_dict[l][m] = a_m[k]
        
        return self.compute_field_from_dict(a_e_dict, a_m_dict, maxorder)
    
    def compute_power(self, amplitude: FieldAmplitude) -> np.ndarray:
        """
        Compute power from field amplitude.
        
        Args:
            amplitude: Field amplitude (n_phi, n_theta, 2)
            
        Returns:
            Power array (n_phi, n_theta)
        """
        return np.sum(np.abs(amplitude) ** 2, axis=-1)


# =============================================================================
# Unified Data Generation Interface
# =============================================================================

class DataGenerator:
    """
    Unified data generation system combining coefficient generation and field computation.
    
    Provides a single interface for generating synthetic electromagnetic data
    for both pipeline operations and ML training.
    """
    
    def __init__(
        self,
        latin_config: Optional[LatinSquareConfig] = None,
        grid_config: Optional[GridConfig] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the data generator.
        
        Args:
            latin_config: Latin square configuration
            grid_config: Grid configuration  
            config: Main configuration
        """
        self.config = config or Config()
        self.latin_generator = LatinSquareGenerator(latin_config)
        self.field_generator = FieldGenerator(grid_config, config)
        
        logger.info(
            f"Initialized DataGenerator "
            f"(mode={self.latin_generator.config.mode}, "
            f"grid={self.field_generator.grid_config.n_phi}x{self.field_generator.grid_config.n_theta})"
        )
    
    @classmethod
    def for_pipeline(cls, config: Optional[Config] = None) -> DataGenerator:
        """Create data generator optimized for pipeline operations (fixed mode)."""
        latin_config = LatinSquareConfig(mode="fixed")
        return cls(latin_config=latin_config, config=config)
    
    @classmethod
    def for_ml_training(cls, config: Optional[Config] = None) -> DataGenerator:
        """Create data generator optimized for ML training (random mode)."""
        latin_config = LatinSquareConfig(mode="random")
        return cls(latin_config=latin_config, config=config)
    
    def generate_sample(
        self, 
        maxorder: int, 
        sample_id: int = 0,
        return_format: str = "dict"
    ) -> Dict[str, Any]:
        """
        Generate a complete synthetic sample.
        
        Args:
            maxorder: Maximum multipole order
            sample_id: Sample identifier
            return_format: "dict" or "array" for coefficient format
            
        Returns:
            Dictionary containing coefficients, fields, and power
        """
        # Generate coefficients
        if return_format == "dict":
            a_e, a_m = self.latin_generator.generate_coefficients_dict(maxorder, sample_id)
            amplitude = self.field_generator.compute_field_from_dict(a_e, a_m, maxorder)
        elif return_format == "array":
            a_e, a_m = self.latin_generator.generate_coefficients_array(maxorder, sample_id)
            amplitude = self.field_generator.compute_field_from_array(a_e, a_m, maxorder)
        else:
            raise ValueError(f"Invalid return_format: {return_format}. Use 'dict' or 'array'")
        
        # Compute power
        power = self.field_generator.compute_power(amplitude)
        
        # Build grid coordinates
        theta, phi = self.field_generator.build_grid()
        
        return {
            "coefficients_e": a_e,
            "coefficients_m": a_m,
            "amplitude": amplitude,
            "power": power,
            "theta": theta,
            "phi": phi,
            "maxorder": maxorder,
            "sample_id": sample_id
        }
    
    def generate_batch(
        self, 
        maxorder: int, 
        n_samples: int,
        start_sample_id: int = 0
    ) -> Dict[str, Any]:
        """
        Generate a batch of samples for ML training.
        
        Args:
            maxorder: Maximum multipole order
            n_samples: Number of samples
            start_sample_id: Starting sample ID
            
        Returns:
            Dictionary with batched data
        """
        # Generate coefficient batch
        a_e_batch, a_m_batch = self.latin_generator.generate_batch_arrays(
            maxorder, n_samples, start_sample_id
        )
        
        # Compute fields for each sample
        n_phi, n_theta = self.field_generator.grid_config.n_phi, self.field_generator.grid_config.n_theta
        amplitude_batch = np.zeros((n_samples, n_phi, n_theta, 2), dtype=complex)
        power_batch = np.zeros((n_samples, n_phi, n_theta), dtype=float)
        
        # Vectorized field computation using fast batch approach
        import time
        logger.info(f"Computing fields for {n_samples} samples (maxorder={maxorder}) using vectorized approach...")
        start_time = time.time()
        
        # Pre-compute grid once for all samples
        theta, phi = self.field_generator.build_grid()
        mpfield_module = self.field_generator._get_mpfield_module()
        
        # Pre-compute all mode field patterns (vectorized)
        mode_pairs = get_mode_list(maxorder)
        mode_fields_e = {}  # Cache for electric field patterns
        mode_fields_m = {}  # Cache for magnetic field patterns
        
        logger.info(f"  Pre-computing {len(mode_pairs)} mode field patterns...")
        for l in range(1, maxorder + 1):
            for m in range(-l, l + 1):
                # Compute field pattern for this mode once (vectorized over entire grid)
                mode_fields_e[(l, m)] = mpfield_module.field_for_multipole(l, m, theta, phi, electric=True)
                mode_fields_m[(l, m)] = mpfield_module.field_for_multipole(l, m, theta, phi, electric=False)
        
        precompute_time = time.time() - start_time
        logger.info(f"  Mode patterns computed in {precompute_time:.2f}s")
        
        # Fast vectorized computation for all samples
        for i in range(n_samples):
            if i % 25 == 0 or i == n_samples - 1:
                elapsed = time.time() - start_time
                logger.info(f"  Processing sample {i+1}/{n_samples} (elapsed: {elapsed:.1f}s)")
            
            # Fast vectorized field computation using pre-computed patterns
            amplitude = np.zeros((theta.shape[0], theta.shape[1], 2), dtype=complex)
            
            # Convert arrays to dict for easy access
            mode_pairs_list = get_mode_list(maxorder)
            for k, (l, m) in enumerate(mode_pairs_list):
                coeff_e = a_e_batch[i, k]
                coeff_m = a_m_batch[i, k]
                
                # Skip zero coefficients for efficiency
                if abs(coeff_e) < 1e-12 and abs(coeff_m) < 1e-12:
                    continue
                
                # Vectorized addition using pre-computed patterns
                amplitude += coeff_e * mode_fields_e[(l, m)] + coeff_m * mode_fields_m[(l, m)]
            
            power = self.field_generator.compute_power(amplitude)
            amplitude_batch[i] = amplitude
            power_batch[i] = power
        
        total_time = time.time() - start_time
        logger.info(f"Vectorized field computation completed in {total_time:.2f}s ({total_time/n_samples:.3f}s per sample)")
        
        # Build grid (same for all samples)
        theta, phi = self.field_generator.build_grid()
        
        return {
            "coefficients_e": a_e_batch,
            "coefficients_m": a_m_batch,
            "amplitude": amplitude_batch,
            "power": power_batch,
            "theta": theta,
            "phi": phi,
            "maxorder": maxorder,
            "n_samples": n_samples,
            "start_sample_id": start_sample_id
        }


# =============================================================================
# Utility Functions
# =============================================================================

def get_mode_list(maxorder: int) -> ModeList:
    """Get list of (l, m) mode pairs for given maxorder."""
    return [(l, m) for l in range(1, maxorder + 1) for m in range(-l, l + 1)]


def get_n_modes(maxorder: int) -> int:
    """Get number of modes for given maxorder."""
    return maxorder * (maxorder + 2)


def pack_coefficients(a_e: np.ndarray, a_m: np.ndarray) -> np.ndarray:
    """
    Pack complex coefficient arrays into real matrix.
    
    Args:
        a_e: Electric coefficients (..., n_modes)
        a_m: Magnetic coefficients (..., n_modes)
        
    Returns:
        Packed real array (..., 4*n_modes) [re(E), im(E), re(M), im(M)]
    """
    return np.concatenate([a_e.real, a_e.imag, a_m.real, a_m.imag], axis=-1).astype(np.float32)


def unpack_coefficients(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unpack real matrix into complex coefficient arrays.
    
    Args:
        y: Packed real array (..., 4*n_modes)
        
    Returns:
        Tuple of (a_e, a_m) complex arrays (..., n_modes)
    """
    n_total = y.shape[-1]
    n_modes = n_total // 4
    
    if n_total != 4 * n_modes:
        raise ValueError(f"Invalid array size: {n_total}. Must be multiple of 4")
    
    a_e = y[..., :n_modes] + 1j * y[..., n_modes:2*n_modes]
    a_m = y[..., 2*n_modes:3*n_modes] + 1j * y[..., 3*n_modes:4*n_modes]
    
    return a_e.astype(np.complex64), a_m.astype(np.complex64)


# =============================================================================
# Export Functions for Backward Compatibility
# =============================================================================

def generate_pipeline_fields(
    output_path: Optional[Path] = None,
    maxorder: Optional[int] = None,
    config: Optional[Config] = None
) -> Path:
    """
    Generate pipeline fields file (backward compatible interface).
    
    Args:
        output_path: Output file path
        maxorder: Maximum multipole order  
        config: Configuration object
        
    Returns:
        Path to generated fields file
    """
    if config is None:
        config = Config()
    
    if maxorder is None:
        maxorder = config.pipeline.default_maxorder
    
    if output_path is None:
        output_path = config.paths.naive_dir / "Fields.txt"
    
    # Create pipeline data generator
    generator = DataGenerator.for_pipeline(config)
    
    # Generate single sample (sample_id=0 for reproducibility)
    sample = generator.generate_sample(maxorder, sample_id=0)
    
    # Write fields file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        amplitude = sample["amplitude"]
        power = sample["power"]
        n_phi, n_theta = amplitude.shape[:2]
        
        for j in range(n_phi):
            for i in range(n_theta):
                # Calculate angles in degrees
                c_theta = (i + 1) * config.data.angle_step_deg  # 1-based indexing
                c_phi = j * config.data.angle_step_deg  # 0-based indexing
                
                # Write: theta phi power |E_theta| phase_theta |E_phi| phase_phi
                f.write(
                    f"{c_theta} {c_phi} {power[j, i]} "
                    f"{np.abs(amplitude[j, i, 0])} {np.angle(amplitude[j, i, 0])} "
                    f"{np.abs(amplitude[j, i, 1])} {np.angle(amplitude[j, i, 1])}\n"
                )
    
    logger.info(f"Generated pipeline fields file: {output_path}")
    return output_path