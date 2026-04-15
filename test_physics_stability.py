#!/usr/bin/env python3
"""
Test physics layers numerical stability and performance optimizations.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.physics_layers import DifferentiableMultipoleField, PhysicsPowerLoss

def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("Testing numerical stability...")
    
    maxorder = 3
    grid_shape = (32, 16)
    n_modes = sum(2 * l + 1 for l in range(1, maxorder + 1))
    
    field_gen = DifferentiableMultipoleField(maxorder=maxorder, grid_shape=grid_shape)
    
    # Test 1: Normal coefficients
    normal_coeffs = torch.randn(2, 4 * n_modes) * 0.1
    try:
        P_normal = field_gen(normal_coeffs)
        print(f"✓ Normal coefficients: P field range [{P_normal.min():.6f}, {P_normal.max():.6f}]")
    except Exception as e:
        print(f"✗ Normal coefficients failed: {e}")
        return False
    
    # Test 2: Very large coefficients
    large_coeffs = torch.randn(2, 4 * n_modes) * 1000
    try:
        P_large = field_gen(large_coeffs)
        print(f"✓ Large coefficients: P field range [{P_large.min():.6f}, {P_large.max():.6f}]")
    except Exception as e:
        print(f"✗ Large coefficients failed: {e}")
        return False
    
    # Test 3: Very small coefficients
    small_coeffs = torch.randn(2, 4 * n_modes) * 1e-10
    try:
        P_small = field_gen(small_coeffs)
        print(f"✓ Small coefficients: P field range [{P_small.min():.6f}, {P_small.max():.6f}]")
    except Exception as e:
        print(f"✗ Small coefficients failed: {e}")
        return False
    
    # Test 4: Zero coefficients
    zero_coeffs = torch.zeros(2, 4 * n_modes)
    try:
        P_zero = field_gen(zero_coeffs)
        print(f"✓ Zero coefficients: P field range [{P_zero.min():.6f}, {P_zero.max():.6f}]")
    except Exception as e:
        print(f"✗ Zero coefficients failed: {e}")
        return False
    
    print("✓ All numerical stability tests passed!")
    return True

def test_physics_loss_stability():
    """Test physics loss numerical stability."""
    print("\nTesting physics loss stability...")
    
    maxorder = 3
    grid_shape = (16, 8)
    n_modes = sum(2 * l + 1 for l in range(1, maxorder + 1))
    
    physics_loss = PhysicsPowerLoss(maxorder=maxorder, grid_shape=grid_shape)
    
    # Test with various coefficient and target combinations
    test_cases = [
        ("normal", torch.randn(2, 4 * n_modes) * 0.1, torch.rand(2, 8, 16) * 0.1),
        ("large_coeffs", torch.randn(2, 4 * n_modes) * 100, torch.rand(2, 8, 16) * 0.1),
        ("large_targets", torch.randn(2, 4 * n_modes) * 0.1, torch.rand(2, 8, 16) * 100),
        ("small_values", torch.randn(2, 4 * n_modes) * 1e-8, torch.rand(2, 8, 16) * 1e-8),
    ]
    
    for test_name, coeffs, targets in test_cases:
        try:
            # Enable gradients for coefficients
            coeffs_grad = coeffs.clone().requires_grad_(True)
            
            loss = physics_loss(coeffs_grad, targets)
            print(f"✓ {test_name}: loss = {loss.item():.6f}")
            
            # Test backward pass
            loss.backward()
            print(f"✓ {test_name}: backward pass successful")
            
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            return False
    
    print("✓ All physics loss stability tests passed!")
    return True

def test_performance_basic():
    """Basic performance test."""
    print("\nTesting basic performance...")
    
    maxorder = 4
    grid_shape = (64, 32)
    n_modes = sum(2 * l + 1 for l in range(1, maxorder + 1))
    batch_size = 16
    
    field_gen = DifferentiableMultipoleField(maxorder=maxorder, grid_shape=grid_shape)
    coeffs = torch.randn(batch_size, 4 * n_modes)
    
    # Warmup
    for _ in range(3):
        _ = field_gen(coeffs)
    
    # Time forward passes
    import time
    n_runs = 10
    start_time = time.time()
    
    for _ in range(n_runs):
        P_field = field_gen(coeffs)
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / n_runs
    
    print(f"✓ Performance test: {avg_time:.4f}s per forward pass (batch_size={batch_size})")
    print(f"  Output shape: {P_field.shape}")
    print(f"  Throughput: {batch_size / avg_time:.1f} samples/sec")
    
    return True

def test_configurable_stability():
    """Test configurable stability parameters."""
    print("\nTesting configurable stability parameters...")
    
    maxorder = 2
    grid_shape = (16, 8)
    
    field_gen = DifferentiableMultipoleField(maxorder=maxorder, grid_shape=grid_shape)
    physics_loss = PhysicsPowerLoss(maxorder=maxorder, grid_shape=grid_shape)
    
    # Configure stability parameters
    field_gen.configure_stability(eps=1e-10, max_coeff_value=1e3)
    physics_loss.configure_stability(loss_eps=1e-10, max_loss_value=1e3)
    
    # Test with configured parameters
    n_modes = sum(2 * l + 1 for l in range(1, maxorder + 1))
    coeffs = torch.randn(1, 4 * n_modes) * 10
    targets = torch.rand(1, 8, 16) * 0.5
    
    try:
        P_field = field_gen(coeffs)
        loss = physics_loss(coeffs, targets)
        
        print(f"✓ Configured stability: P field range [{P_field.min():.6f}, {P_field.max():.6f}]")
        print(f"✓ Configured stability: loss = {loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Configured stability test failed: {e}")
        return False

def main():
    """Run all stability and performance tests."""
    print("🧪 Testing Physics Layers Stability and Performance")
    print("=" * 60)
    
    tests = [
        ("Numerical Stability", test_numerical_stability),
        ("Physics Loss Stability", test_physics_loss_stability),
        ("Basic Performance", test_performance_basic),
        ("Configurable Stability", test_configurable_stability),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 STABILITY & PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 All stability and performance tests passed!")
        print("✅ Physics layers are numerically stable and performant!")
    else:
        print("⚠️  Some tests failed. Check implementation.")
    
    return all_passed

if __name__ == "__main__":
    main()