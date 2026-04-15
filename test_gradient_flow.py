#!/usr/bin/env python3
"""
Test gradient flow through DifferentiableMultipoleField at different grid resolutions.
"""

import torch
import numpy as np
from src.models.physics_layers import DifferentiableMultipoleField
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gradient_flow(maxorder=5, grid_shapes=None, n_samples=10):
    """
    Test gradient flow through DifferentiableMultipoleField for various grid sizes.
    
    Args:
        maxorder: Maximum multipole order
        grid_shapes: List of (n_phi, n_theta) tuples to test
        n_samples: Number of coefficient samples to test
    """
    
    if grid_shapes is None:
        # Test a range of grid sizes from small to full resolution
        grid_shapes = [
            (16, 16),    # Very small
            (32, 16),    # Small
            (37, 179),   # Known working size from exploration
            (64, 32),    # Medium
            (128, 64),   # Large
            (180, 90),   # Standard-ish
            (360, 179),  # Full resolution (problematic)
        ]
    
    # Calculate number of coefficients for the given maxorder
    n_modes = sum(2*l + 1 for l in range(1, maxorder + 1))  # l=1 to maxorder
    n_coeffs = 4 * n_modes  # 4 components per mode (a_e_real, a_e_imag, a_m_real, a_m_imag)
    
    print(f"Testing gradient flow with maxorder={maxorder}, n_modes={n_modes}, n_coeffs={n_coeffs}")
    print("=" * 80)
    
    results = []
    
    for n_phi, n_theta in grid_shapes:
        print(f"\n🧪 Testing grid size: ({n_phi}, {n_theta})")
        
        try:
            # Create the field generator
            field_gen = DifferentiableMultipoleField(
                maxorder=maxorder,
                grid_shape=(n_phi, n_theta),
                device=torch.device('cpu')
            )
            
            # Create random coefficient input (requires gradient)
            coeffs_packed = torch.randn(n_samples, n_coeffs, requires_grad=True, dtype=torch.float32)
            
            # Forward pass
            P_field = field_gen(coeffs_packed)
            
            print(f"  ✅ Forward pass successful: {coeffs_packed.shape} -> {P_field.shape}")
            print(f"  📊 P field stats: mean={P_field.mean().item():.6f}, std={P_field.std().item():.6f}")
            print(f"  📈 P field range: [{P_field.min().item():.6f}, {P_field.max().item():.6f}]")
            
            # Create a simple loss (sum of P field)
            loss = P_field.sum()
            print(f"  🎯 Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            if coeffs_packed.grad is not None:
                grad_norm = coeffs_packed.grad.norm().item()
                grad_mean = coeffs_packed.grad.mean().item()
                grad_std = coeffs_packed.grad.std().item()
                grad_max = coeffs_packed.grad.abs().max().item()
                grad_nonzero = (coeffs_packed.grad.abs() > 1e-10).sum().item()
                grad_total = coeffs_packed.grad.numel()
                
                print(f"  ✅ Gradient computed!")
                print(f"     📊 Grad norm: {grad_norm:.8f}")
                print(f"     📊 Grad mean: {grad_mean:.8f}")
                print(f"     📊 Grad std: {grad_std:.8f}")
                print(f"     📊 Grad max: {grad_max:.8f}")
                print(f"     📊 Non-zero grads: {grad_nonzero}/{grad_total} ({100*grad_nonzero/grad_total:.1f}%)")
                
                gradient_working = grad_norm > 1e-8
                if gradient_working:
                    print(f"  🎉 GRADIENT FLOW: WORKING")
                    status = "WORKING"
                else:
                    print(f"  ❌ GRADIENT FLOW: ZERO/MINIMAL")
                    status = "ZERO"
            else:
                print(f"  ❌ No gradient computed!")
                gradient_working = False
                status = "NO_GRAD"
                grad_norm = 0.0
            
            results.append({
                'grid_shape': (n_phi, n_theta),
                'grid_size': n_phi * n_theta,
                'status': status,
                'gradient_working': gradient_working,
                'grad_norm': grad_norm,
                'p_field_mean': P_field.mean().item(),
                'p_field_std': P_field.std().item(),
                'loss': loss.item()
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'grid_shape': (n_phi, n_theta),
                'grid_size': n_phi * n_theta,
                'status': 'ERROR',
                'gradient_working': False,
                'grad_norm': 0.0,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 GRADIENT FLOW SUMMARY")
    print("=" * 80)
    
    working_grids = []
    zero_grads = []
    error_grids = []
    
    for result in results:
        grid_shape = result['grid_shape']
        status = result['status']
        
        if status == 'WORKING':
            working_grids.append(grid_shape)
            print(f"✅ {grid_shape}: WORKING (grad_norm={result['grad_norm']:.2e})")
        elif status == 'ZERO' or status == 'NO_GRAD':
            zero_grads.append(grid_shape)
            print(f"❌ {grid_shape}: NO GRADIENTS")
        else:
            error_grids.append(grid_shape)
            print(f"💥 {grid_shape}: ERROR - {result.get('error', 'Unknown')}")
    
    print(f"\n📊 RESULTS:")
    print(f"  🎉 Working gradient flow: {len(working_grids)} grids")
    print(f"  ❌ Zero/no gradients: {len(zero_grads)} grids")
    print(f"  💥 Errors: {len(error_grids)} grids")
    
    if zero_grads:
        print(f"\n🚨 PROBLEMATIC GRIDS (zero gradients):")
        for grid in zero_grads:
            print(f"  - {grid}")
            
    if working_grids:
        print(f"\n✅ WORKING GRIDS:")
        for grid in working_grids:
            print(f"  - {grid}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    
    # Check if there's a pattern related to grid size
    working_sizes = [result['grid_size'] for result in results if result['status'] == 'WORKING']
    zero_sizes = [result['grid_size'] for result in results if result['status'] in ['ZERO', 'NO_GRAD']]
    
    if working_sizes and zero_sizes:
        max_working = max(working_sizes)
        min_zero = min(zero_sizes)
        print(f"  - Largest working grid size: {max_working}")
        print(f"  - Smallest zero-gradient grid size: {min_zero}")
        
        if max_working < min_zero:
            print(f"  🎯 PATTERN: Gradient flow fails above grid size ~{max_working}")
        else:
            print(f"  🤔 No clear size-based pattern detected")
    
    # Check specific problematic resolution
    full_res_result = next((r for r in results if r['grid_shape'] == (360, 179)), None)
    if full_res_result:
        print(f"\n🎯 FULL RESOLUTION (360, 179) STATUS: {full_res_result['status']}")
        if full_res_result['status'] != 'WORKING':
            print(f"  ⚠️  This confirms the training issue - physics loss cannot train at full resolution!")
    
    return results

def find_maximum_working_grid(maxorder=5, max_attempts=10):
    """Find the maximum grid size that still allows gradient flow."""
    
    print(f"\n🔍 Finding maximum working grid size for maxorder={maxorder}")
    print("-" * 60)
    
    # Start with known working sizes and gradually increase
    test_sizes = [
        (16, 16), (32, 32), (64, 64), (96, 96), (128, 128), 
        (180, 90), (256, 128), (300, 150), (360, 179)
    ]
    
    max_working = None
    
    for n_phi, n_theta in test_sizes:
        print(f"Testing ({n_phi}, {n_theta})...", end=" ")
        
        try:
            field_gen = DifferentiableMultipoleField(
                maxorder=maxorder,
                grid_shape=(n_phi, n_theta),
                device=torch.device('cpu')
            )
            
            n_modes = sum(2*l + 1 for l in range(1, maxorder + 1))
            n_coeffs = 4 * n_modes
            coeffs = torch.randn(5, n_coeffs, requires_grad=True, dtype=torch.float32)
            
            P_field = field_gen(coeffs)
            loss = P_field.sum()
            loss.backward()
            
            if coeffs.grad is not None and coeffs.grad.norm().item() > 1e-8:
                max_working = (n_phi, n_theta)
                print("✅ WORKING")
            else:
                print("❌ ZERO GRAD")
                break
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            break
    
    if max_working:
        print(f"\n🎉 Maximum working grid size: {max_working}")
        print(f"   Grid points: {max_working[0] * max_working[1]}")
    else:
        print(f"\n❌ No working grid sizes found!")
    
    return max_working

if __name__ == "__main__":
    print("🧪 GRADIENT FLOW INVESTIGATION")
    print("=" * 80)
    
    # Test standard maxorder=5 (from physics config)
    results = test_gradient_flow(maxorder=5)
    
    # Find maximum working grid
    max_working = find_maximum_working_grid(maxorder=5)
    
    # Save results for reference
    import json
    with open('gradient_flow_test_results.json', 'w') as f:
        json.dump({
            'maxorder': 5,
            'results': results,
            'max_working_grid': max_working,
            'summary': {
                'working_grids': [r['grid_shape'] for r in results if r['status'] == 'WORKING'],
                'zero_grad_grids': [r['grid_shape'] for r in results if r['status'] in ['ZERO', 'NO_GRAD']],
                'error_grids': [r['grid_shape'] for r in results if r['status'] == 'ERROR']
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to gradient_flow_test_results.json")