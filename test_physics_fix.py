#!/usr/bin/env python3
"""
Test the physics layer fix to verify gradient flow is restored.
"""

import torch
import numpy as np
from src.models.physics_layers import DifferentiableMultipoleField

def test_fixed_physics_layer():
    """Test the fixed physics layer for gradient flow."""
    
    print("🧪 TESTING FIXED PHYSICS LAYER")
    print("=" * 60)
    
    maxorder = 5
    grid_sizes = [(32, 32), (64, 32)]  # Test small working size and previously broken size
    
    for n_phi, n_theta in grid_sizes:
        print(f"\n🔍 Testing grid size: ({n_phi}, {n_theta})")
        
        try:
            # Create fixed physics layer
            field_gen = DifferentiableMultipoleField(
                maxorder=maxorder,
                grid_shape=(n_phi, n_theta),
                device=torch.device('cpu')
            )
            
            # Create test coefficients
            n_modes = sum(2*l + 1 for l in range(1, maxorder + 1))
            n_coeffs = 4 * n_modes
            coeffs = torch.randn(5, n_coeffs, requires_grad=True, dtype=torch.float32)
            
            print(f"  📊 Input shape: {coeffs.shape}")
            
            # Forward pass
            P_field = field_gen(coeffs)
            loss = P_field.sum()
            
            print(f"  📊 P field shape: {P_field.shape}")
            print(f"  📊 P field stats: mean={P_field.mean().item():.6f}, std={P_field.std().item():.6f}")
            print(f"  📊 Loss: {loss.item():.6f}")
            
            # Backward pass 
            loss.backward()
            
            # Check gradients
            if coeffs.grad is not None:
                grad_norm = coeffs.grad.norm().item()
                grad_nonzero = (coeffs.grad.abs() > 1e-10).sum().item()
                grad_total = coeffs.grad.numel()
                
                print(f"  ✅ Gradient norm: {grad_norm:.6f}")
                print(f"  📊 Non-zero grads: {grad_nonzero}/{grad_total} ({100*grad_nonzero/grad_total:.1f}%)")
                
                if grad_norm > 1e-6:
                    print(f"  🎉 GRADIENT FLOW: WORKING!")
                    status = "WORKING"
                else:
                    print(f"  ❌ GRADIENT FLOW: STILL ZERO")
                    status = "ZERO"
            else:
                print(f"  ❌ No gradients computed")
                status = "NO_GRAD"
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")
            status = "ERROR"
    
    print(f"\n" + "=" * 60)

def test_coefficient_range():
    """Test different coefficient values to verify the physics computation."""
    
    print("🧪 TESTING COEFFICIENT SENSITIVITY")
    print("=" * 60)
    
    maxorder = 3  # Smaller for detailed analysis
    grid_shape = (16, 16)  # Small grid
    
    field_gen = DifferentiableMultipoleField(
        maxorder=maxorder,
        grid_shape=grid_shape,
        device=torch.device('cpu')
    )
    
    n_modes = sum(2*l + 1 for l in range(1, maxorder + 1))
    n_coeffs = 4 * n_modes
    
    # Test different coefficient patterns
    test_patterns = {
        "zeros": torch.zeros(1, n_coeffs),
        "ones": torch.ones(1, n_coeffs),
        "random": torch.randn(1, n_coeffs),
        "dipole_only": torch.zeros(1, n_coeffs)
    }
    
    # Set only l=1, m=0 component for dipole test
    test_patterns["dipole_only"][0, 1] = 1.0  # l=1, m=0 electric component
    
    for name, coeffs in test_patterns.items():
        coeffs.requires_grad_(True)
        
        P_field = field_gen(coeffs)
        loss = P_field.sum()
        
        print(f"📊 {name:12s}: P_sum={loss.item():8.4f}, P_mean={P_field.mean().item():8.6f}")
        
        loss.backward()
        grad_norm = coeffs.grad.norm().item() if coeffs.grad is not None else 0.0
        print(f"                   grad_norm={grad_norm:8.6f}")

if __name__ == "__main__":
    test_fixed_physics_layer()
    print("\n")
    test_coefficient_range()