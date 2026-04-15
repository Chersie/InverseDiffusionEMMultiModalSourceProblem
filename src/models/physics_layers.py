#!/usr/bin/env python3
"""
Physics-informed layers for differentiable multipole field computation.

This module implements PyTorch layers that compute electromagnetic fields
from multipole coefficients using vector spherical harmonics, enabling
gradients to flow through the physics computation for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import torch_harmonics as harmonics


class DifferentiableMultipoleField(nn.Module):
    """
    Differentiable multipole field computation using torch-harmonics.
    
    Converts packed multipole coefficients to electromagnetic power field P
    using vector spherical harmonic transforms. This enables physics-informed
    loss functions that optimize for field accuracy rather than coefficient accuracy.
    """
    
    def __init__(
        self, 
        maxorder: int, 
        grid_shape: Tuple[int, int] = (360, 179),
        grid_type: str = "equiangular",
        device: Optional[Union[torch.device, str]] = None
    ):
        """
        Initialize differentiable multipole field computation.
        
        Args:
            maxorder: Maximum multipole order (L_max)
            grid_shape: (n_phi, n_theta) grid resolution
            grid_type: Grid type for spherical harmonics ("equiangular" or "legendre-gauss")
            device: Device to place computations on
        """
        super().__init__()
        
        self.maxorder = maxorder
        self.n_phi, self.n_theta = grid_shape
        self.grid_type = grid_type
        # Ensure device is a torch.device object, not a string
        if isinstance(device, str):
            self.device = torch.device(device)
        elif device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Calculate number of modes for each multipole type
        self.n_modes = sum(2 * l + 1 for l in range(1, maxorder + 1))
        
        # Initialize vector spherical harmonic transforms
        # Note: lmax in torch-harmonics is inclusive, and we start from l=1
        self.lmax = maxorder
        
        # Create inverse transform (coefficients -> grid) for field computation
        # Use lmax+1 to include inactive band 0 (required for correct indexing)
        self.vsht_inverse = harmonics.InverseRealVectorSHT(
            self.n_theta, self.n_phi, 
            lmax=self.lmax + 1,  # +1 for inactive band 0
            grid=self.grid_type
        ).to(self.device)
        
        # Get the actual coefficient shape by doing a dummy forward transform
        self._actual_coeff_shape = self._get_actual_coefficient_shape()
        
        # Cache for coordinate grids and computation artifacts
        self._theta_grid = None
        self._phi_grid = None
        self._cached_forward_result = None
        self._last_input_hash = None
        
        # Numerical stability parameters
        self.eps = 1e-8  # Small value to prevent division by zero
        self.max_coeff_value = 1e6  # Clamp coefficients to prevent overflow
    def _get_actual_coefficient_shape(self) -> tuple:
        """Get the actual coefficient shape from torch-harmonics by doing a dummy transform."""
        # Create a dummy field with the target grid shape
        dummy_field = torch.zeros(1, 2, self.n_theta, self.n_phi, device=self.device)
        
        # Create forward transform to get coefficient shape
        # Use same lmax as inverse transform (includes inactive band 0)
        vsht_forward = harmonics.RealVectorSHT(
            self.n_theta, self.n_phi, 
            lmax=self.lmax + 1,  # +1 for inactive band 0 (consistent with inverse)
            grid=self.grid_type
        ).to(self.device)
        
        # Get coefficient shape
        dummy_coeffs = vsht_forward(dummy_field)
        return dummy_coeffs.shape[2:]  # (l_bands, m_coeffs)
    
    def _get_coordinate_grids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get spherical coordinate grids (cached), aligned with data generation."""
        if self._theta_grid is None or self._phi_grid is None:
            # Create coordinate grids exactly matching data generation for proper alignment
            if self.grid_type == "equiangular":
                # Theta: equiangular from 0 to pi (includes poles) - matches data generation
                theta = torch.linspace(0, np.pi, self.n_theta, device=self.device, dtype=torch.float32)
                # Phi: from 0 to 2*pi, exclude endpoint for periodicity.
                # torch.linspace has no endpoint=False; use arange instead.
                phi = torch.arange(self.n_phi, device=self.device, dtype=torch.float32) * (2 * np.pi / self.n_phi)
            else:
                # Legendre-Gauss grid (experimental - still using equiangular for alignment)
                theta = torch.linspace(0, np.pi, self.n_theta, device=self.device, dtype=torch.float32)
                phi = torch.arange(self.n_phi, device=self.device, dtype=torch.float32) * (2 * np.pi / self.n_phi)
            
            # Create meshgrid - order matches data generation expectations
            self._theta_grid, self._phi_grid = torch.meshgrid(theta, phi, indexing='ij')
            
        return self._theta_grid, self._phi_grid
    
    def validate_grid_alignment_with_data(self, data_theta: np.ndarray, data_phi: np.ndarray) -> bool:
        """
        Validate that physics computation grid aligns with data generation grid.
        
        Args:
            data_theta: Theta grid from data generation (2D)
            data_phi: Phi grid from data generation (2D)
            
        Returns:
            True if grids are aligned within tolerance
        """
        theta_grid, phi_grid = self._get_coordinate_grids()
        
        # Convert torch tensors to numpy for comparison
        theta_physics = theta_grid.cpu().numpy()
        phi_physics = phi_grid.cpu().numpy()
        
        # Check shapes
        if theta_physics.shape != data_theta.shape:
            print(f"WARNING: Theta shape mismatch: physics={theta_physics.shape}, data={data_theta.shape}")
            return False
            
        if phi_physics.shape != data_phi.shape:
            print(f"WARNING: Phi shape mismatch: physics={phi_physics.shape}, data={data_phi.shape}")
            return False
        
        # Check values with tolerance
        theta_aligned = np.allclose(theta_physics, data_theta, rtol=1e-6, atol=1e-8)
        phi_aligned = np.allclose(phi_physics, data_phi, rtol=1e-6, atol=1e-8)
        
        if not theta_aligned:
            print("WARNING: Theta grids not aligned!")
            
        if not phi_aligned:
            print("WARNING: Phi grids not aligned!")
        
        is_aligned = theta_aligned and phi_aligned
        if is_aligned:
            print("✅ Physics and data generation grids are properly aligned")
        else:
            print("❌ Grid alignment validation failed!")
            
        return is_aligned
    
    def unpack_coefficients(self, coeffs_packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack coefficients from flat format to separate real/imaginary parts.
        
        Args:
            coeffs_packed: Shape (batch_size, 4*n_modes) packed coefficients
            
        Returns:
            a_e_real, a_e_imag, a_m_real, a_m_imag: Each shape (batch_size, n_modes)
        """
        batch_size = coeffs_packed.shape[0]
        
        # Split into 4 equal parts
        coeffs_split = torch.split(coeffs_packed, self.n_modes, dim=1)
        
        if len(coeffs_split) != 4:
            raise ValueError(f"Expected coeffs_packed to have size 4*{self.n_modes}={4*self.n_modes}, "
                           f"but got {coeffs_packed.shape[1]}")
        
        a_e_real, a_e_imag, a_m_real, a_m_imag = coeffs_split
        
        return a_e_real, a_e_imag, a_m_real, a_m_imag
    
    def coefficients_to_sht_format(
        self, 
        a_e_real: torch.Tensor, 
        a_e_imag: torch.Tensor,
        a_m_real: torch.Tensor, 
        a_m_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert coefficient arrays to format expected by torch-harmonics.
        
        Args:
            a_e_real, a_e_imag: Electric multipole coefficients (batch_size, n_modes)
            a_m_real, a_m_imag: Magnetic multipole coefficients (batch_size, n_modes)
            
        Returns:
            sht_coeffs: Shape (batch_size, 2, l_bands, m_coeffs) for torch-harmonics
        """
        batch_size = a_e_real.shape[0]
        device = a_e_real.device
        
        # Get actual coefficient shape from torch-harmonics
        l_bands, m_coeffs = self._actual_coeff_shape
        
        # Initialize coefficient array for torch-harmonics
        # Shape: (batch, 2, l_bands, m_coeffs)
        # Component 0: poloidal (electric-like)  
        # Component 1: toroidal (magnetic-like)
        sht_coeffs = torch.zeros(
            batch_size, 2, l_bands, m_coeffs,
            dtype=torch.complex64, device=device
        )
        
        # Map our coefficients to torch-harmonics format
        # Our coefficients are ordered by (l, m) with l from 1 to maxorder
        coeff_idx = 0
        
        for l in range(1, self.maxorder + 1):  # Start from l=1 (no monopole)
            # Map l to torch-harmonics l index  
            # CORRECTED: Band 0 is inactive, so map l=1 to band 1 (first active)
            th_l_idx = l  # l=1→band1, l=2→band2, etc. (band 0 stays empty/inactive)
            
            if th_l_idx >= l_bands:
                # Skip if this l is beyond what torch-harmonics supports
                coeff_idx += 2 * l + 1  # Skip all m values for this l
                continue
                
            for m in range(-l, l + 1):
                # CORRECTED m-indexing for real SHT
                # Real SHT stores only non-negative m modes: m_idx = 0,1,2,...
                # Negative m modes are handled via conjugate symmetry
                th_m_idx = abs(m)  # Map m=-1→1, m=0→0, m=1→1, etc.
                
                if 0 <= th_m_idx < m_coeffs:
                    # Handle conjugate symmetry for real SHT
                    if m == 0:
                        # m=0: purely real coefficient
                        e_coeff = torch.complex(a_e_real[:, coeff_idx], torch.zeros_like(a_e_real[:, coeff_idx]))
                        m_coeff = torch.complex(a_m_real[:, coeff_idx], torch.zeros_like(a_m_real[:, coeff_idx]))
                    elif m > 0:
                        # m>0: use coefficient as-is
                        e_coeff = torch.complex(a_e_real[:, coeff_idx], a_e_imag[:, coeff_idx])
                        m_coeff = torch.complex(a_m_real[:, coeff_idx], a_m_imag[:, coeff_idx])
                    else:  # m < 0
                        # m<0: apply conjugate symmetry relation Y_{l,-m} = (-1)^m * conj(Y_{l,m})
                        phase_factor = (-1) ** abs(m)
                        e_coeff = phase_factor * torch.complex(a_e_real[:, coeff_idx], -a_e_imag[:, coeff_idx])
                        m_coeff = phase_factor * torch.complex(a_m_real[:, coeff_idx], -a_m_imag[:, coeff_idx])
                    
                    # Accumulate coefficients (handle case where m and -m map to same th_m_idx)
                    if th_m_idx < m_coeffs:
                        sht_coeffs[:, 0, th_l_idx, th_m_idx] += e_coeff
                        sht_coeffs[:, 1, th_l_idx, th_m_idx] += m_coeff
                
                coeff_idx += 1
        
        return sht_coeffs
    
    def compute_electromagnetic_fields(self, sht_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute electromagnetic fields from spherical harmonic coefficients.
        
        Args:
            sht_coeffs: Shape (batch_size, 2, lmax+1, 2*lmax+1)
            
        Returns:
            E_theta, E_phi: Electric field components, each shape (batch_size, n_theta, n_phi)
        """
        # Use inverse vector SHT to get vector field components
        # Output shape: (batch_size, 2, n_theta, n_phi) where 2 = [theta, phi] components
        vector_field = self.vsht_inverse(sht_coeffs)
        
        # Extract theta and phi components
        E_theta = vector_field[:, 0, :, :]  # (batch_size, n_theta, n_phi)
        E_phi = vector_field[:, 1, :, :]    # (batch_size, n_theta, n_phi)
        
        return E_theta, E_phi
    
    def compute_power_field(self, E_theta: torch.Tensor, E_phi: torch.Tensor) -> torch.Tensor:
        """
        Compute power field P = |E_theta|^2 + |E_phi|^2 with numerical stability.
        
        Args:
            E_theta, E_phi: Electric field components (batch_size, n_theta, n_phi)
            
        Returns:
            P_field: Power field (batch_size, n_theta, n_phi)
        """
        # Numerical stability: handle potentially complex fields
        if E_theta.is_complex():
            E_theta_power = torch.abs(E_theta)**2
        else:
            E_theta_power = E_theta**2
            
        if E_phi.is_complex():
            E_phi_power = torch.abs(E_phi)**2
        else:
            E_phi_power = E_phi**2
        
        # Compute power field
        P_field = E_theta_power + E_phi_power
        
        # Numerical stability: ensure non-negative (should be by construction, but be safe)
        P_field = torch.clamp(P_field, min=0.0)
        
        # Add small epsilon to prevent zero values in loss computation
        P_field = P_field + self.eps
        
        return P_field
    
    def forward(self, coeffs_packed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: coefficients -> electromagnetic fields -> power field.
        
        Args:
            coeffs_packed: Shape (batch_size, 4*n_modes) packed coefficients
            
        Returns:
            P_field: Power field (batch_size, n_theta, n_phi)
        """
        # Numerical stability: check for NaN or Inf in input
        if torch.isnan(coeffs_packed).any() or torch.isinf(coeffs_packed).any():
            raise ValueError("Input coefficients contain NaN or Inf values")
        
        # Numerical stability: clamp coefficients to prevent overflow
        coeffs_packed = torch.clamp(coeffs_packed, -self.max_coeff_value, self.max_coeff_value)
        
        # 1. Unpack coefficients
        a_e_real, a_e_imag, a_m_real, a_m_imag = self.unpack_coefficients(coeffs_packed)
        
        # 2. Convert to spherical harmonic format
        sht_coeffs = self.coefficients_to_sht_format(a_e_real, a_e_imag, a_m_real, a_m_imag)
        
        # 3. Compute electromagnetic fields
        E_theta, E_phi = self.compute_electromagnetic_fields(sht_coeffs)
        
        # 4. Compute power field with numerical stability
        P_field = self.compute_power_field(E_theta, E_phi)
        
        # Numerical stability: check output
        if torch.isnan(P_field).any() or torch.isinf(P_field).any():
            # Replace NaN/Inf with small positive values
            P_field = torch.where(torch.isnan(P_field) | torch.isinf(P_field), 
                                 torch.full_like(P_field, self.eps), P_field)
        
        return P_field
    
    def get_field_components(self, coeffs_packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get field components separately (useful for visualization).
        
        Args:
            coeffs_packed: Shape (batch_size, 4*n_modes) packed coefficients
            
        Returns:
            E_theta, E_phi: Field components (batch_size, n_theta, n_phi)
            P_field: Power field (batch_size, n_theta, n_phi)
        """
        # Forward pass but return intermediate results
        a_e_real, a_e_imag, a_m_real, a_m_imag = self.unpack_coefficients(coeffs_packed)
        sht_coeffs = self.coefficients_to_sht_format(a_e_real, a_e_imag, a_m_real, a_m_imag)
        E_theta, E_phi = self.compute_electromagnetic_fields(sht_coeffs)
        P_field = self.compute_power_field(E_theta, E_phi)
        
        return E_theta, E_phi, P_field
    
    def configure_stability(
        self, 
        eps: float = None, 
        max_coeff_value: float = None
    ) -> None:
        """
        Configure numerical stability parameters.
        
        Args:
            eps: Small value to prevent division by zero and add to power fields
            max_coeff_value: Maximum allowed coefficient value (clamping)
        """
        if eps is not None:
            self.eps = eps
        if max_coeff_value is not None:
            self.max_coeff_value = max_coeff_value


class PhysicsPowerLoss(nn.Module):
    """
    Physics-informed loss function that optimizes for P field accuracy.
    
    Instead of comparing predicted and true coefficients directly,
    this loss computes the electromagnetic fields from predicted coefficients
    and compares the resulting power fields with the true power fields.
    """
    
    def __init__(
        self, 
        maxorder: int, 
        grid_shape: Tuple[int, int] = (360, 179),
        grid_type: str = "equiangular",
        reduction: str = "mean",
        device: Optional[Union[torch.device, str]] = None,
        field_weight: float = 0.0
    ):
        """
        Initialize physics-informed power loss.
        
        Args:
            maxorder: Maximum multipole order
            grid_shape: (n_phi, n_theta) grid resolution for field computation
            grid_type: Grid type for spherical harmonics
            reduction: Loss reduction method ("mean", "sum", "none")
            device: Device for computations
            field_weight: Weight for E-field component losses (0 = power only)
        """
        super().__init__()
        
        self.reduction = reduction
        
        # Create differentiable field generator
        self.field_generator = DifferentiableMultipoleField(
            maxorder=maxorder,
            grid_shape=grid_shape,
            grid_type=grid_type,
            device=device
        )
        
        # Loss computation parameters
        self.loss_eps = 1e-8  # Regularization for loss computation
        self.max_loss_value = 1e6  # Clamp loss to prevent instability
        self.field_weight = field_weight  # Weight for E-field component losses
        
    def forward(self, pred_coeffs: torch.Tensor, true_P: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Args:
            pred_coeffs: Predicted coefficients (batch_size, 4*n_modes)
            true_P: True power field (batch_size, n_theta, n_phi) or (batch_size, n_phi, n_theta)
            
        Returns:
            loss: Scalar loss value
        """
        # Generate predicted P field from coefficients
        pred_P = self.field_generator(pred_coeffs)
        
        # Handle potential spatial-dimension mismatches (fallback only — should not trigger
        # after the grid alignment fix that stores P targets as (N, n_theta, n_phi)).
        if true_P.shape != pred_P.shape:
            pred_n_theta, pred_n_phi = pred_P.shape[1], pred_P.shape[2]
            
            if true_P.shape[-2:] == pred_P.shape[-2:][::-1]:
                true_P = true_P.transpose(-2, -1)
            
            # If still different shapes, resize true_P to match pred_P
            if true_P.shape != pred_P.shape:
                
                # Resize true_P to match predicted P field resolution
                # true_P shape: (batch, true_n_theta, true_n_phi)
                # pred_P shape: (batch, pred_n_theta, pred_n_phi)
                
                # Ensure true_P is float for interpolation
                true_P = true_P.float()
                
                # Add channel dimension for interpolation: (batch, 1, height, width)
                true_P_4d = true_P.unsqueeze(1)
                
                # Bilinear interpolation to resize
                true_P_resized = F.interpolate(
                    true_P_4d, 
                    size=(pred_n_theta, pred_n_phi),
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Remove channel dimension: (batch, height, width)
                true_P = true_P_resized.squeeze(1)
                
                # Ensure same device and dtype as pred_P
                true_P = true_P.to(pred_P.device, pred_P.dtype)
        else:
            # Grid shapes match - this is the expected behavior with proper alignment
            pass  # No action needed when grids are properly aligned
        
        # Note: Field generator already adds eps for numerical stability,
        # so no need to add regularization again here (avoid double regularization)
        pred_P_reg = pred_P
        true_P_reg = true_P
        
        # Apply spherical area weighting (sin(θ)) for physically meaningful loss
        theta_grid, phi_grid = self.field_generator._get_coordinate_grids()
        
        # Compute area weights: sin(θ) for each grid point
        # Shape: (n_theta, n_phi) -> broadcast to (batch_size, n_theta, n_phi)
        sin_theta_weights = torch.sin(theta_grid).to(pred_P.device, pred_P.dtype)
        sin_theta_weights = sin_theta_weights.unsqueeze(0)  # Add batch dimension
        
        # Normalize weights so they don't change the overall loss scale
        sin_theta_weights = sin_theta_weights / sin_theta_weights.mean()
        
        # Compute weighted squared differences
        squared_diff = (pred_P_reg - true_P_reg) ** 2
        weighted_squared_diff = squared_diff * sin_theta_weights
        
        # Apply reduction
        if self.reduction == "mean":
            loss = weighted_squared_diff.mean()
        elif self.reduction == "sum":
            loss = weighted_squared_diff.sum()
        else:  # "none"
            loss = weighted_squared_diff
        
        # Numerical stability: clamp loss to prevent instability
        loss = torch.clamp(loss, max=self.max_loss_value)
        
        # Check for NaN/Inf in loss and provide better handling
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf detected in physics loss!")
            print(f"  pred_P stats: min={pred_P.min():.6f}, max={pred_P.max():.6f}, mean={pred_P.mean():.6f}")
            print(f"  true_P stats: min={true_P.min():.6f}, max={true_P.max():.6f}, mean={true_P.mean():.6f}")
            print("  Using fallback loss that still provides gradients")
            
            # Use a loss that's still connected to the inputs for gradient flow
            # This provides more useful gradient information than a constant
            fallback_loss = torch.mean(torch.abs(pred_P - true_P)) + 1.0
            fallback_loss = torch.clamp(fallback_loss, max=self.max_loss_value)
            return fallback_loss
        
        return loss
    
    def forward_with_components(
        self, 
        pred_coeffs: torch.Tensor, 
        true_P: torch.Tensor,
        true_E_theta: Optional[torch.Tensor] = None,
        true_E_phi: Optional[torch.Tensor] = None,
        field_weight: float = 0.1
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute physics loss with optional field component losses.
        
        Args:
            pred_coeffs: Predicted coefficients
            true_P: True power field
            true_E_theta, true_E_phi: Optional true field components
            field_weight: Weight for field component losses relative to power loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Get predicted field components
        pred_E_theta, pred_E_phi, pred_P = self.field_generator.get_field_components(pred_coeffs)
        
        # Power field loss
        if true_P.shape != pred_P.shape and true_P.shape[-2:] == pred_P.shape[-2:][::-1]:
            true_P = true_P.transpose(-2, -1)
        power_loss = F.mse_loss(pred_P, true_P, reduction=self.reduction)
        
        loss_dict = {"power_loss": power_loss}
        total_loss = power_loss
        
        # Optional field component losses
        if true_E_theta is not None:
            if true_E_theta.shape != pred_E_theta.shape and true_E_theta.shape[-2:] == pred_E_theta.shape[-2:][::-1]:
                true_E_theta = true_E_theta.transpose(-2, -1)
            E_theta_loss = F.mse_loss(pred_E_theta, true_E_theta, reduction=self.reduction)
            loss_dict["E_theta_loss"] = E_theta_loss
            total_loss = total_loss + field_weight * E_theta_loss
        
        if true_E_phi is not None:
            if true_E_phi.shape != pred_E_phi.shape and true_E_phi.shape[-2:] == pred_E_phi.shape[-2:][::-1]:
                true_E_phi = true_E_phi.transpose(-2, -1)
            E_phi_loss = F.mse_loss(pred_E_phi, true_E_phi, reduction=self.reduction)
            loss_dict["E_phi_loss"] = E_phi_loss
            total_loss = total_loss + field_weight * E_phi_loss
        
        return total_loss, loss_dict
    
    def configure_stability(
        self, 
        loss_eps: float = None, 
        max_loss_value: float = None,
        field_generator_eps: float = None,
        field_generator_max_coeff: float = None
    ) -> None:
        """
        Configure numerical stability parameters for loss computation.
        
        Args:
            loss_eps: Regularization epsilon for loss computation
            max_loss_value: Maximum allowed loss value (clamping)
            field_generator_eps: Epsilon for field generator
            field_generator_max_coeff: Max coefficient value for field generator
        """
        if loss_eps is not None:
            self.loss_eps = loss_eps
        if max_loss_value is not None:
            self.max_loss_value = max_loss_value
        if field_generator_eps is not None or field_generator_max_coeff is not None:
            self.field_generator.configure_stability(field_generator_eps, field_generator_max_coeff)
    
    def evaluate_streaming(
        self,
        pred_coeffs_path: str,
        true_P_path: str,
        output_metrics_path: str,
        batch_size: int = 500
    ) -> Dict[str, float]:
        """
        Evaluate physics loss in streaming mode using memory-mapped data.
        
        Args:
            pred_coeffs_path: Path to predicted coefficients (.npy)
            true_P_path: Path to true P field targets (.npy)
            output_metrics_path: Path to save metrics (.json)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of computed metrics
        """
        import numpy as np
        import json
        from pathlib import Path
        
        logger.info(f"Streaming physics loss evaluation...")
        
        # Load memory-mapped arrays
        pred_coeffs_mm = np.load(pred_coeffs_path, mmap_mode='r')
        true_P_mm = np.load(true_P_path, mmap_mode='r')
        n_samples = pred_coeffs_mm.shape[0]
        
        if true_P_mm.shape[0] != n_samples:
            raise ValueError(f"Sample count mismatch: coeffs={n_samples}, P_true={true_P_mm.shape[0]}")
        
        logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
        
        # Accumulate metrics
        total_loss = 0.0
        total_samples = 0
        losses = []
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                current_batch_size = end_idx - start_idx
                
                if batch_idx % 10 == 0:
                    logger.info(f"  Evaluating batch {batch_idx + 1}/{n_batches}")
                
                # Load batch
                pred_coeffs_batch = torch.from_numpy(pred_coeffs_mm[start_idx:end_idx].copy()).float().to(self.device)
                true_P_batch = torch.from_numpy(true_P_mm[start_idx:end_idx].copy()).float().to(self.device)
                
                # Compute loss
                batch_loss = self.forward(pred_coeffs_batch, true_P_batch)
                
                total_loss += batch_loss.item() * current_batch_size
                total_samples += current_batch_size
                losses.append(batch_loss.item())
                
                # Clear GPU memory
                del pred_coeffs_batch, true_P_batch
                
                # Periodic cleanup
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    import gc
                    gc.collect()
        
        # Compute final metrics
        mean_loss = total_loss / total_samples
        loss_std = np.std(losses) if len(losses) > 1 else 0.0
        
        metrics = {
            "mean_physics_loss": float(mean_loss),
            "loss_std": float(loss_std),
            "n_samples": total_samples,
            "n_batches": n_batches,
            "batch_size": batch_size
        }
        
        # Save metrics
        output_path = Path(output_metrics_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ Physics evaluation metrics: loss={mean_loss:.6f} ± {loss_std:.6f}")
        logger.info(f"✅ Metrics saved to: {output_path}")
        
        return metrics


# =============================================================================
# Utility Functions for Streaming Physics Evaluation
# =============================================================================

def evaluate_physics_model_streaming(
    model_coeffs_path: str,
    true_P_path: str,
    output_dir: str,
    maxorder: int = 5,
    batch_size: int = 500,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Comprehensive streaming evaluation of a physics model.
    
    Args:
        model_coeffs_path: Path to model coefficient predictions (.npy)
        true_P_path: Path to true P field targets (.npy)
        output_dir: Directory for output files and metrics
        maxorder: Maximum multipole order
        batch_size: Batch size for processing
        device: Device for field computation
        
    Returns:
        Dictionary with evaluation results and output paths
    """
    from pathlib import Path
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Comprehensive physics model evaluation in streaming mode")
    logger.info(f"  Model coeffs: {model_coeffs_path}")
    logger.info(f"  True P field: {true_P_path}")
    logger.info(f"  Output dir: {output_dir}")
    
    # Create field generator
    field_generator = DifferentiableMultipoleField(
        maxorder=maxorder,
        grid_shape=(360, 179),  # Standard grid
        device=torch.device(device)
    )
    
    # 1. Evaluate P field reconstruction
    P_pred_path = output_dir / "P_predicted.npy"
    field_generator.evaluate_batch_streaming(
        model_coeffs_path, 
        str(P_pred_path),
        batch_size=batch_size
    )
    
    # 2. Evaluate all field components
    components_dir = output_dir / "field_components"
    component_paths = field_generator.evaluate_streaming_with_components(
        model_coeffs_path,
        str(components_dir),
        batch_size=batch_size
    )
    
    # 3. Compute physics loss metrics
    loss_eval = PhysicsPowerLoss(
        maxorder=maxorder,
        grid_shape=(360, 179),
        device=torch.device(device)
    )
    
    metrics_path = output_dir / "physics_metrics.json"
    physics_metrics = loss_eval.evaluate_streaming(
        model_coeffs_path,
        true_P_path,
        str(metrics_path),
        batch_size=batch_size
    )
    
    # Combine results
    results = {
        "physics_metrics": physics_metrics,
        "output_paths": {
            "P_predicted": str(P_pred_path),
            "metrics": str(metrics_path),
            **component_paths
        },
        "processing_info": {
            "maxorder": maxorder,
            "batch_size": batch_size,
            "device": device,
            "grid_shape": (360, 179)
        }
    }
    
    # Save comprehensive results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Comprehensive physics evaluation complete: {results_path}")
    
    return results