"""
Optimized Decomposition Engine for Multipole Analysis.

This module provides high-performance field decomposition using the library manager,
replacing per-mode file loading with batch processing and memory caching.
Reduces decomposition time from 510+ file reads to single batch library load.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.config import Config
from src.core.library_manager import LibraryManager, get_library_manager

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

DecompositionResult = Dict[str, Any]
CoefficientData = Tuple[float, float]  # (real, imag)
ModeResults = Dict[Tuple[str, int, int], CoefficientData]  # {(type, l, m): (re, im)}


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class DecompositionConfig:
    """Configuration for field decomposition."""
    
    # Input/output settings
    field_file: str = "Fields.txt"
    result_file_template: str = "Results_{stem}.txt"
    
    # Decomposition parameters
    maxorder: int = 15
    angle_step_deg: float = 1.0
    
    # Processing options
    validate_input: bool = True
    compute_power_metrics: bool = True
    save_intermediate: bool = False
    
    # Performance settings
    use_batching: bool = True
    batch_size: int = 50  # Number of modes to process together
    
    # Output formatting
    precision: int = 8  # Decimal precision for results
    include_headers: bool = False


# =============================================================================
# Field File Parser
# =============================================================================

class FieldFileParser:
    """
    Parser for field data files with validation and preprocessing.
    
    Supports the standard field file format:
    theta phi power |E_theta| phase_theta |E_phi| phase_phi
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the parser.
        
        Args:
            config: Configuration object (creates default if None)
        """
        self.config = config or Config()
        logger.debug("Initialized FieldFileParser")
    
    def parse_field_file(self, field_path: Path) -> Dict[str, np.ndarray]:
        """
        Parse field file and return structured data.
        
        Args:
            field_path: Path to field file
            
        Returns:
            Dictionary with parsed field data
            
        Raises:
            ValueError: If file format is invalid
        """
        logger.info(f"Parsing field file: {field_path}")
        
        if not field_path.exists():
            raise FileNotFoundError(f"Field file not found: {field_path}")
        
        # Calculate expected dimensions
        n_phi = int(360 / self.config.data.angle_step_deg)
        n_theta = int(180 / self.config.data.angle_step_deg) - 1
        expected_rows = n_phi * n_theta
        
        # Read and parse file
        rows = []
        with field_path.open('r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) >= 7:  # theta phi power |E_theta| phase |E_phi| phase
                    try:
                        row = [float(p) for p in parts[:7]]
                        rows.append(row)
                    except ValueError as e:
                        logger.warning(f"Invalid data on line {line_num}: {e}")
                        continue
        
        if len(rows) < expected_rows:
            raise ValueError(
                f"Field file {field_path} has {len(rows)} data rows; "
                f"expected at least {expected_rows}"
            )
        
        # Validate grid ordering
        self._validate_grid_ordering(rows, n_phi, n_theta)
        
        # Convert to structured arrays
        theta_grid = np.zeros((n_phi, n_theta), dtype=float)
        phi_grid = np.zeros((n_phi, n_theta), dtype=float)
        power_grid = np.zeros((n_phi, n_theta), dtype=float)
        amplitude_grid = np.zeros((n_phi, n_theta, 2), dtype=complex)
        
        for j in range(n_phi):
            for i in range(n_theta):
                idx = n_theta * j + i
                row = rows[idx]
                
                theta_grid[j, i] = row[0]
                phi_grid[j, i] = row[1]
                power_grid[j, i] = row[2]
                
                # Convert magnitude and phase to complex amplitude
                amplitude_grid[j, i, 0] = row[3] * np.exp(1j * row[4])  # E_theta
                amplitude_grid[j, i, 1] = row[5] * np.exp(1j * row[6])  # E_phi
        
        logger.info(f"Parsed field file: {len(rows)} rows, grid {n_phi}x{n_theta}")
        
        return {
            "theta": theta_grid,
            "phi": phi_grid,
            "power": power_grid,
            "amplitude": amplitude_grid,
            "n_phi": n_phi,
            "n_theta": n_theta,
            "n_rows": len(rows)
        }
    
    def _validate_grid_ordering(
        self, 
        rows: List[List[float]], 
        n_phi: int, 
        n_theta: int
    ) -> None:
        """Validate expected grid ordering."""
        if not rows:
            return
        
        # Check first point
        theta0, phi0 = rows[0][0], rows[0][1]
        expected_theta0 = self.config.data.angle_step_deg  # 1 degree
        expected_phi0 = 0.0
        
        if not (np.isclose(theta0, expected_theta0, atol=1e-6) and 
                np.isclose(phi0, expected_phi0, atol=1e-6)):
            logger.warning(
                f"Unexpected grid start: (theta,phi)=({theta0},{phi0}), "
                f"expected ({expected_theta0},{expected_phi0})"
            )
        
        # Check grid progression
        if len(rows) >= n_theta:
            theta_next, phi_next = rows[n_theta][0], rows[n_theta][1]
            expected_theta_next = expected_theta0
            expected_phi_next = self.config.data.angle_step_deg
            
            if not (np.isclose(theta_next, expected_theta_next, atol=1e-6) and
                    np.isclose(phi_next, expected_phi_next, atol=1e-6)):
                logger.warning(
                    f"Unexpected grid progression at row {n_theta}: "
                    f"(theta,phi)=({theta_next},{phi_next}), "
                    f"expected ({expected_theta_next},{expected_phi_next})"
                )


# =============================================================================
# Optimized Decomposition Engine
# =============================================================================

class DecompositionEngine:
    """
    High-performance multipole decomposition engine.
    
    Features:
    - Uses LibraryManager for fast mode access
    - Batch processing for improved performance
    - Comprehensive validation and error handling
    - Flexible output formatting
    - Memory-efficient computation
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        decomp_config: Optional[DecompositionConfig] = None,
        library_manager: Optional[LibraryManager] = None
    ):
        """
        Initialize the decomposition engine.
        
        Args:
            config: Main configuration object
            decomp_config: Decomposition-specific configuration
            library_manager: Library manager instance (creates default if None)
        """
        self.config = config or Config()
        self.decomp_config = decomp_config or DecompositionConfig()
        
        # Initialize library manager
        if library_manager is not None:
            self.library_manager = library_manager
        else:
            self.library_manager = get_library_manager(
                maxorder=self.decomp_config.maxorder,
                config=self.config
            )
        
        # Initialize field parser
        self.parser = FieldFileParser(self.config)
        
        logger.info(
            f"Initialized DecompositionEngine "
            f"(maxorder={self.decomp_config.maxorder}, "
            f"batching={'enabled' if self.decomp_config.use_batching else 'disabled'})"
        )
    
    def decompose_field_file(
        self, 
        field_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> DecompositionResult:
        """
        Decompose a field file into multipole coefficients.
        
        Args:
            field_path: Path to field file
            output_path: Output path for results (auto-generated if None)
            
        Returns:
            Dictionary with decomposition results
        """
        field_path = Path(field_path)
        
        if output_path is None:
            stem = field_path.stem
            output_path = field_path.parent / self.decomp_config.result_file_template.format(stem=stem)
        else:
            output_path = Path(output_path)
        
        logger.info(f"Starting decomposition: {field_path} -> {output_path}")
        start_time = time.time()
        
        # Parse field file
        field_data = self.parser.parse_field_file(field_path)
        
        # Ensure library is loaded
        self.library_manager.preload_for_decomposition(self.decomp_config.maxorder)
        
        # Perform decomposition
        coefficients = self._decompose_field_data(field_data)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_results(coefficients, output_path)
        
        # Compute metrics if requested
        metrics = {}
        if self.decomp_config.compute_power_metrics:
            metrics = self._compute_decomposition_metrics(field_data, coefficients)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Decomposition completed in {elapsed_time:.2f}s")
        
        return {
            "field_path": str(field_path),
            "output_path": str(output_path),
            "coefficients": coefficients,
            "metrics": metrics,
            "elapsed_time": elapsed_time,
            "field_data_shape": (field_data["n_phi"], field_data["n_theta"]),
            "n_modes": len([k for k in coefficients if k[0] in ("E", "M")])
        }
    
    def decompose_multiple_files(
        self,
        field_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[DecompositionResult]:
        """
        Decompose multiple field files efficiently.
        
        Args:
            field_paths: List of field file paths
            output_dir: Output directory (uses input file directories if None)
            
        Returns:
            List of decomposition results
        """
        field_paths = [Path(p) for p in field_paths]
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting batch decomposition of {len(field_paths)} files")
        
        # Preload library once for all decompositions
        self.library_manager.preload_for_decomposition(self.decomp_config.maxorder)
        
        results = []
        total_time = 0.0
        
        for i, field_path in enumerate(field_paths):
            logger.info(f"Processing file {i+1}/{len(field_paths)}: {field_path.name}")
            
            # Generate output path
            if output_dir is not None:
                output_path = output_dir / self.decomp_config.result_file_template.format(
                    stem=field_path.stem
                )
            else:
                output_path = None
            
            try:
                result = self.decompose_field_file(field_path, output_path)
                results.append(result)
                total_time += result["elapsed_time"]
                
            except Exception as e:
                logger.error(f"Failed to decompose {field_path}: {e}")
                results.append({
                    "field_path": str(field_path),
                    "error": str(e),
                    "elapsed_time": 0.0
                })
        
        avg_time = total_time / len(field_paths) if field_paths else 0.0
        logger.info(
            f"Batch decomposition completed: {len(results)} files, "
            f"total {total_time:.2f}s, average {avg_time:.2f}s per file"
        )
        
        return results
    
    def _decompose_field_data(self, field_data: Dict[str, np.ndarray]) -> ModeResults:
        """
        Perform the actual multipole decomposition.
        
        Args:
            field_data: Parsed field data
            
        Returns:
            Dictionary with mode coefficients {(type, l, m): (real, imag)}
        """
        amplitude = field_data["amplitude"]
        n_phi, n_theta = field_data["n_phi"], field_data["n_theta"]
        
        # Compute angular integration weight
        d_omega = (self.config.data.angle_step_deg * np.pi / 180.0) ** 2
        
        coefficients = {}
        
        if self.decomp_config.use_batching:
            coefficients = self._decompose_batched(amplitude, d_omega)
        else:
            coefficients = self._decompose_sequential(amplitude, d_omega)
        
        return coefficients
    
    def _decompose_batched(
        self, 
        amplitude: np.ndarray, 
        d_omega: float
    ) -> ModeResults:
        """Perform batched decomposition for improved performance."""
        coefficients = {}
        
        # Collect all modes to process
        modes_to_process = []
        for l in range(1, self.decomp_config.maxorder + 1):
            for m in range(-l, l + 1):
                modes_to_process.append(('E', l, m))
                modes_to_process.append(('M', l, m))
        
        # Process in batches
        batch_size = self.decomp_config.batch_size
        n_modes = len(modes_to_process)
        
        logger.debug(f"Processing {n_modes} modes in batches of {batch_size}")
        
        for batch_start in range(0, n_modes, batch_size):
            batch_end = min(batch_start + batch_size, n_modes)
            batch_modes = modes_to_process[batch_start:batch_end]
            
            logger.debug(f"Processing batch {batch_start//batch_size + 1}: modes {batch_start}-{batch_end-1}")
            
            # Process this batch
            for mode_type, l, m in batch_modes:
                try:
                    # Get library mode data
                    lib_amp, lib_theta = self.library_manager.get_mode(mode_type, l, m)
                    
                    # Compute inner product
                    integrand = (
                        np.conj(lib_amp[..., 0]) * amplitude[..., 0] +
                        np.conj(lib_amp[..., 1]) * amplitude[..., 1]
                    )
                    
                    # Apply sin(theta) weighting for proper integration
                    theta_rad = np.deg2rad(lib_theta)
                    weighted_integrand = integrand * np.sin(theta_rad)
                    
                    # Integrate over solid angle
                    coefficient = np.sum(weighted_integrand) * d_omega
                    
                    # Store result
                    coefficients[(mode_type, l, m)] = (coefficient.real, coefficient.imag)
                    
                except Exception as e:
                    logger.warning(f"Failed to process mode {mode_type}({l},{m}): {e}")
                    coefficients[(mode_type, l, m)] = (0.0, 0.0)
        
        return coefficients
    
    def _decompose_sequential(
        self, 
        amplitude: np.ndarray, 
        d_omega: float
    ) -> ModeResults:
        """Perform sequential decomposition (non-batched)."""
        coefficients = {}
        
        # Process E modes
        for l in range(1, self.decomp_config.maxorder + 1):
            for m in range(-l, l + 1):
                try:
                    e_amp, e_theta = self.library_manager.get_e_mode(l, m)
                    
                    integrand = (
                        np.conj(e_amp[..., 0]) * amplitude[..., 0] +
                        np.conj(e_amp[..., 1]) * amplitude[..., 1]
                    )
                    
                    theta_rad = np.deg2rad(e_theta)
                    weighted_integrand = integrand * np.sin(theta_rad)
                    coefficient = np.sum(weighted_integrand) * d_omega
                    
                    coefficients[('E', l, m)] = (coefficient.real, coefficient.imag)
                    
                except Exception as e:
                    logger.warning(f"Failed to process E mode ({l},{m}): {e}")
                    coefficients[('E', l, m)] = (0.0, 0.0)
        
        # Process M modes  
        for l in range(1, self.decomp_config.maxorder + 1):
            for m in range(-l, l + 1):
                try:
                    m_amp, m_theta = self.library_manager.get_m_mode(l, m)
                    
                    integrand = (
                        np.conj(m_amp[..., 0]) * amplitude[..., 0] +
                        np.conj(m_amp[..., 1]) * amplitude[..., 1]
                    )
                    
                    theta_rad = np.deg2rad(m_theta)
                    weighted_integrand = integrand * np.sin(theta_rad)
                    coefficient = np.sum(weighted_integrand) * d_omega
                    
                    coefficients[('M', l, m)] = (coefficient.real, coefficient.imag)
                    
                except Exception as e:
                    logger.warning(f"Failed to process M mode ({l},{m}): {e}")
                    coefficients[('M', l, m)] = (0.0, 0.0)
        
        return coefficients
    
    def _save_results(self, coefficients: ModeResults, output_path: Path) -> None:
        """Save decomposition results to file."""
        logger.debug(f"Saving results to {output_path}")
        
        with output_path.open('w') as f:
            # Write header if requested
            if self.decomp_config.include_headers:
                f.write("# Multipole decomposition results\n")
                f.write("# Format: type l m real_coeff imag_coeff\n")
            
            # Write coefficients in order
            for l in range(1, self.decomp_config.maxorder + 1):
                for m in range(-l, l + 1):
                    # Write E mode
                    if ('E', l, m) in coefficients:
                        real_part, imag_part = coefficients[('E', l, m)]
                        f.write(f"E {l} {m} {real_part:.{self.decomp_config.precision}g} "
                               f"{imag_part:.{self.decomp_config.precision}g}\n")
                    
                    # Write M mode
                    if ('M', l, m) in coefficients:
                        real_part, imag_part = coefficients[('M', l, m)]
                        f.write(f"M {l} {m} {real_part:.{self.decomp_config.precision}g} "
                               f"{imag_part:.{self.decomp_config.precision}g}\n")
        
        logger.debug(f"Saved {len(coefficients)} coefficients")
    
    def _compute_decomposition_metrics(
        self, 
        field_data: Dict[str, np.ndarray], 
        coefficients: ModeResults
    ) -> Dict[str, float]:
        """Compute decomposition quality metrics."""
        metrics = {}
        
        try:
            # Total power from field data
            original_power = np.sum(field_data["power"])
            metrics["original_total_power"] = float(original_power)
            
            # Total power from coefficients
            coeff_power = 0.0
            for (mode_type, l, m), (real_part, imag_part) in coefficients.items():
                coeff_power += real_part**2 + imag_part**2
            
            metrics["coefficient_total_power"] = float(coeff_power)
            
            # Power conservation ratio
            if original_power > 0:
                metrics["power_conservation_ratio"] = coeff_power / original_power
            else:
                metrics["power_conservation_ratio"] = 0.0
            
            # Number of significant modes (above threshold)
            threshold = 0.01 * np.sqrt(coeff_power / len(coefficients))
            significant_modes = sum(
                1 for (_, _, _), (re, im) in coefficients.items()
                if np.sqrt(re**2 + im**2) > threshold
            )
            metrics["significant_modes"] = significant_modes
            metrics["total_modes"] = len(coefficients)
            
        except Exception as e:
            logger.warning(f"Failed to compute metrics: {e}")
        
        return metrics


# =============================================================================
# Convenience Functions
# =============================================================================

def decompose_field(
    field_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    maxorder: int = 15,
    config: Optional[Config] = None
) -> DecompositionResult:
    """
    Convenience function for single field decomposition.
    
    Args:
        field_path: Path to field file
        output_path: Output path (auto-generated if None)
        maxorder: Maximum multipole order
        config: Configuration object
        
    Returns:
        Decomposition result dictionary
    """
    decomp_config = DecompositionConfig(maxorder=maxorder)
    engine = DecompositionEngine(config=config, decomp_config=decomp_config)
    
    return engine.decompose_field_file(field_path, output_path)


def decompose_batch(
    field_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    maxorder: int = 15,
    config: Optional[Config] = None
) -> List[DecompositionResult]:
    """
    Convenience function for batch field decomposition.
    
    Args:
        field_paths: List of field file paths
        output_dir: Output directory
        maxorder: Maximum multipole order
        config: Configuration object
        
    Returns:
        List of decomposition results
    """
    decomp_config = DecompositionConfig(maxorder=maxorder)
    engine = DecompositionEngine(config=config, decomp_config=decomp_config)
    
    return engine.decompose_multiple_files(field_paths, output_dir)