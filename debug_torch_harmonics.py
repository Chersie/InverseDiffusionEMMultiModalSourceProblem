#!/usr/bin/env python3
"""
Debug torch-harmonics indexing and conventions to understand correct coefficient mapping.
"""

import torch
import numpy as np
import torch_harmonics as harmonics

def investigate_torch_harmonics_indexing():
    """Investigate torch-harmonics indexing conventions."""
    
    print("🔍 TORCH-HARMONICS INDEXING INVESTIGATION")
    print("=" * 70)
    
    # Test different grid sizes to understand the library
    test_grids = [(32, 16), (64, 32), (16, 16)]
    maxorders = [3, 5]
    
    for nlat, nlon in test_grids:
        for lmax in maxorders:
            print(f"\n🧪 Testing nlat={nlat}, nlon={nlon}, lmax={lmax}")
            
            try:
                # Create inverse transform (coefficients -> fields)
                inverse_transform = harmonics.InverseRealVectorSHT(
                    nlat=nlat, nlon=nlon, lmax=lmax, 
                    grid="equiangular", norm="ortho"
                )
                
                # Get coefficient tensor shape
                # Should be (batch_size, 2, lmax, mmax) for vector fields
                mmax = inverse_transform.mmax
                print(f"  📊 Expected coeff shape: (batch, 2, {lmax}, {mmax})")
                print(f"  📊 mmax = {mmax} (should be ~nlon//2 + 1 = {nlon//2 + 1})")
                
                # Create empty coefficients
                batch_size = 1
                coeffs = torch.zeros(batch_size, 2, lmax, mmax, dtype=torch.complex64)
                
                # Test each band individually to see which are active
                print(f"  🧪 Testing individual bands...")
                for l_idx in range(lmax):
                    # Set a single coefficient to 1.0
                    coeffs.fill_(0.0)
                    if mmax > 0:
                        coeffs[0, 0, l_idx, 0] = 1.0  # Component 0, l=l_idx, m=0
                        
                        # Transform to fields
                        field = inverse_transform(coeffs)
                        field_norm = torch.norm(field).item()
                        
                        print(f"    l_idx={l_idx}: field_norm={field_norm:.6f}")
                        
                        if field_norm > 1e-6:
                            print(f"      ✅ ACTIVE band (produces non-zero field)")
                        else:
                            print(f"      ❌ INACTIVE band (zero field)")
                
                # Test m-indexing
                print(f"  🧪 Testing m-indexing...")
                coeffs.fill_(0.0)
                
                # For l_idx=1 (if it exists and is active), test different m values
                if lmax > 1:
                    l_test = 1
                    print(f"    Testing l_idx={l_test}:")
                    for m_idx in range(min(3, mmax)):  # Test first few m values
                        coeffs.fill_(0.0)
                        coeffs[0, 0, l_test, m_idx] = 1.0
                        
                        field = inverse_transform(coeffs)
                        field_norm = torch.norm(field).item()
                        print(f"      m_idx={m_idx}: field_norm={field_norm:.6f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print(f"\n" + "=" * 70)

def test_round_trip_transform():
    """Test round-trip: field -> forward SHT -> inverse SHT -> field"""
    
    print("🔄 ROUND-TRIP TRANSFORM TEST")
    print("=" * 70)
    
    nlat, nlon, lmax = 32, 32, 5
    
    try:
        # Create forward and inverse transforms
        forward_sht = harmonics.RealVectorSHT(
            nlat=nlat, nlon=nlon, lmax=lmax, 
            grid="equiangular", norm="ortho"
        )
        inverse_sht = harmonics.InverseRealVectorSHT(
            nlat=nlat, nlon=nlon, lmax=lmax,
            grid="equiangular", norm="ortho"
        )
        
        print(f"📊 Grid: {nlat}x{nlon}, lmax={lmax}")
        print(f"📊 Forward SHT mmax: {forward_sht.mmax}")
        print(f"📊 Inverse SHT mmax: {inverse_sht.mmax}")
        
        # Create a simple test field (e.g., a dipole-like pattern)
        theta = torch.linspace(0, np.pi, nlat)
        phi = torch.linspace(0, 2*np.pi, nlon)[:-1]  # Remove last point for periodicity
        if len(phi) < nlon:
            phi = torch.cat([phi, torch.tensor([2*np.pi])])
        
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        
        # Simple dipole-like field: E_theta ~ cos(theta), E_phi = 0
        E_theta = torch.cos(theta_grid).unsqueeze(0)  # Add batch dimension
        E_phi = torch.zeros_like(E_theta)
        
        # Combine into vector field
        field_input = torch.stack([E_theta, E_phi], dim=1)  # (batch, 2, nlat, nlon)
        
        print(f"📊 Input field shape: {field_input.shape}")
        print(f"📊 Input field norm: {torch.norm(field_input).item():.6f}")
        
        # Forward transform
        coeffs = forward_sht(field_input)
        print(f"📊 Coefficients shape: {coeffs.shape}")
        print(f"📊 Coefficients norm: {torch.norm(coeffs).item():.6f}")
        
        # Find non-zero coefficients
        threshold = 1e-6
        nonzero_mask = torch.abs(coeffs) > threshold
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False)
        
        print(f"📊 Non-zero coefficients: {len(nonzero_indices)}")
        for idx in nonzero_indices[:10]:  # Show first 10
            batch, comp, l, m = idx.tolist()
            value = coeffs[batch, comp, l, m]
            print(f"  coeffs[{batch}, {comp}, {l}, {m}] = {value:.6f}")
        
        # Inverse transform
        field_reconstructed = inverse_sht(coeffs)
        print(f"📊 Reconstructed field shape: {field_reconstructed.shape}")
        print(f"📊 Reconstructed field norm: {torch.norm(field_reconstructed).item():.6f}")
        
        # Check reconstruction error
        error = torch.norm(field_reconstructed - field_input).item()
        relative_error = error / torch.norm(field_input).item()
        
        print(f"📊 Reconstruction error: {error:.6f}")
        print(f"📊 Relative error: {relative_error:.6f}")
        
        if relative_error < 1e-3:
            print("✅ Round-trip successful!")
        else:
            print("❌ Round-trip has significant error")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_our_coefficient_mapping():
    """Analyze how our coefficient mapping compares to torch-harmonics expectations."""
    
    print("🔍 OUR COEFFICIENT MAPPING ANALYSIS")
    print("=" * 70)
    
    maxorder = 5
    n_modes = sum(2*l + 1 for l in range(1, maxorder + 1))  # Our convention: l=1 to maxorder
    n_coeffs = 4 * n_modes
    
    print(f"📊 Our convention:")
    print(f"   maxorder = {maxorder}")
    print(f"   l values: 1 to {maxorder}")
    print(f"   n_modes = {n_modes}")
    print(f"   n_coeffs = {n_coeffs} (4 components per mode)")
    
    print(f"\n📊 Torch-harmonics expectation for lmax={maxorder}:")
    print(f"   l indices: 0 to {maxorder-1}")
    print(f"   bands: {maxorder}")
    
    print(f"\n🚨 POTENTIAL MAPPING ISSUES:")
    print(f"   1. Our l=1 maps to th_l_idx=0, but band 0 might be inactive")
    print(f"   2. Our l=2 maps to th_l_idx=1, which might be the first active band")
    print(f"   3. Our l={maxorder} maps to th_l_idx={maxorder-1}, last band")
    
    # Test with actual torch-harmonics shapes
    nlat, nlon = 32, 32
    try:
        inverse_sht = harmonics.InverseRealVectorSHT(
            nlat=nlat, nlon=nlon, lmax=maxorder,
            grid="equiangular", norm="ortho"
        )
        
        mmax = inverse_sht.mmax
        print(f"\n📊 For {nlat}x{nlon} grid with lmax={maxorder}:")
        print(f"   mmax = {mmax}")
        print(f"   Expected coefficient shape: (batch, 2, {maxorder}, {mmax})")
        print(f"   Our m-center approach: m_center = {mmax // 2}")
        
        print(f"\n🔍 M-indexing analysis:")
        print(f"   Physical m values: -l to +l")
        print(f"   For l=1: m ∈ [-1, 0, 1] → th_m_idx ∈ [{mmax//2 - 1}, {mmax//2}, {mmax//2 + 1}]")
        print(f"   For l=2: m ∈ [-2, -1, 0, 1, 2] → th_m_idx ∈ [{mmax//2 - 2}, ..., {mmax//2 + 2}]")
        
        if mmax//2 + maxorder >= mmax:
            print(f"   ❌ PROBLEM: th_m_idx can exceed mmax-1 = {mmax-1}")
        else:
            print(f"   ✅ M-indexing might be OK (within bounds)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    investigate_torch_harmonics_indexing()
    print("\n")
    test_round_trip_transform() 
    print("\n")
    analyze_our_coefficient_mapping()