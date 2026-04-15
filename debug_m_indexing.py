#!/usr/bin/env python3
"""
Debug the m-indexing issue in coefficient mapping.
"""

import torch
import numpy as np
from src.models.physics_layers import DifferentiableMultipoleField

def debug_coefficient_mapping():
    """Debug which coefficients actually affect the output."""
    
    print("🔍 DEBUGGING COEFFICIENT MAPPING")
    print("=" * 60)
    
    maxorder = 3  # Smaller for detailed analysis
    grid_shape = (16, 16)
    
    field_gen = DifferentiableMultipoleField(
        maxorder=maxorder,
        grid_shape=grid_shape,
        device=torch.device('cpu')
    )
    
    n_modes = sum(2*l + 1 for l in range(1, maxorder + 1))  # l=1,2,3: 3+5+7=15 modes
    n_coeffs = 4 * n_modes  # 60 coefficients total
    
    print(f"📊 maxorder={maxorder}, n_modes={n_modes}, n_coeffs={n_coeffs}")
    print(f"📊 Grid: {grid_shape}")
    
    # Test each coefficient individually
    nonzero_coeffs = []
    
    print(f"\n🧪 Testing individual coefficients (first 20):")
    for i in range(min(20, n_coeffs)):
        coeffs = torch.zeros(1, n_coeffs, dtype=torch.float32)
        coeffs[0, i] = 1.0  # Set only one coefficient
        
        P_field = field_gen(coeffs)
        P_sum = P_field.sum().item()
        P_max = P_field.max().item()
        
        # Determine which (l,m) this coefficient represents
        coeff_type = ["a_e_real", "a_e_imag", "a_m_real", "a_m_imag"][i % 4]
        mode_idx = i // 4
        
        # Convert mode_idx to (l,m)
        mode_count = 0
        l_val, m_val = None, None
        for l in range(1, maxorder + 1):
            for m in range(-l, l + 1):
                if mode_count == mode_idx:
                    l_val, m_val = l, m
                    break
                mode_count += 1
            if l_val is not None:
                break
        
        status = "✅" if P_sum > 1e-6 else "❌"
        print(f"  {status} coeff[{i:2d}]: {coeff_type:8s} l={l_val}, m={m_val:2d} → P_sum={P_sum:8.4f}")
        
        if P_sum > 1e-6:
            nonzero_coeffs.append((i, coeff_type, l_val, m_val, P_sum))
    
    print(f"\n📊 Coefficients that produce non-zero output:")
    for i, coeff_type, l, m, P_sum in nonzero_coeffs:
        print(f"  coeff[{i:2d}]: {coeff_type} (l={l}, m={m}) → P_sum={P_sum:.4f}")
    
    if not nonzero_coeffs:
        print("  ❌ NO coefficients produce non-zero output!")
    
    print(f"\n📊 Summary: {len(nonzero_coeffs)}/{min(20, n_coeffs)} coefficients are effective")

def analyze_sht_coefficient_structure():
    """Analyze the SHT coefficient tensor structure."""
    
    print("\n🔍 ANALYZING SHT COEFFICIENT STRUCTURE")
    print("=" * 60)
    
    maxorder = 3
    grid_shape = (16, 16)
    
    field_gen = DifferentiableMultipoleField(
        maxorder=maxorder,
        grid_shape=grid_shape,
        device=torch.device('cpu')
    )
    
    # Get the actual coefficient shape
    l_bands, m_coeffs = field_gen._actual_coeff_shape
    print(f"📊 SHT coefficient tensor shape: (batch, 2, {l_bands}, {m_coeffs})")
    print(f"📊 lmax used: {maxorder + 1} (maxorder + 1)")
    
    # Create test coefficient tensor and see what produces output
    batch_size = 1
    sht_coeffs = torch.zeros(batch_size, 2, l_bands, m_coeffs, dtype=torch.complex64)
    
    print(f"\n🧪 Testing individual SHT coefficient positions:")
    active_positions = []
    
    for comp in [0, 1]:  # Component 0: E-like, 1: M-like
        for l_idx in range(l_bands):
            for m_idx in range(min(5, m_coeffs)):  # Test first few m values
                sht_coeffs.fill_(0.0)
                sht_coeffs[0, comp, l_idx, m_idx] = 1.0
                
                # Use the SHT inverse directly
                field = field_gen.vsht_inverse(sht_coeffs)
                field_norm = torch.norm(field).item()
                
                status = "✅" if field_norm > 1e-6 else "❌"
                comp_name = "E" if comp == 0 else "M"
                print(f"  {status} SHT[{comp_name}, l={l_idx}, m={m_idx}] → field_norm={field_norm:.6f}")
                
                if field_norm > 1e-6:
                    active_positions.append((comp, l_idx, m_idx, field_norm))
    
    print(f"\n📊 Active SHT positions:")
    for comp, l_idx, m_idx, norm in active_positions:
        comp_name = "E" if comp == 0 else "M"
        print(f"  {comp_name}[l={l_idx}, m={m_idx}] → norm={norm:.6f}")

if __name__ == "__main__":
    debug_coefficient_mapping()
    analyze_sht_coefficient_structure()