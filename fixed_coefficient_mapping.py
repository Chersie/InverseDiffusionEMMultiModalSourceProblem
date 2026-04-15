#!/usr/bin/env python3
"""
Fixed coefficient mapping for torch-harmonics based on investigation results.
"""

import torch
import numpy as np
import torch_harmonics as harmonics

def corrected_coefficients_to_sht_format(coeffs_packed, maxorder, grid_shape, device):
    """
    CORRECTED version of coefficients_to_sht_format based on investigation.
    
    Key fixes:
    1. Map our l=1 to th_l_idx=1 (first ACTIVE band), not 0 (inactive)
    2. Use proper m-indexing that matches torch-harmonics expectations
    3. Handle the fact that band 0 is inactive
    """
    
    batch_size = coeffs_packed.shape[0]
    n_phi, n_theta = grid_shape
    
    # Split packed coefficients
    n_modes = coeffs_packed.shape[1] // 4
    a_e_real, a_e_imag, a_m_real, a_m_imag = torch.split(coeffs_packed, n_modes, dim=1)
    
    # Create inverse transform to get proper dimensions
    inverse_sht = harmonics.InverseRealVectorSHT(
        nlat=n_theta, nlon=n_phi, lmax=maxorder+1,  # +1 to account for inactive band 0
        grid="equiangular", norm="ortho"
    )
    
    l_bands = maxorder + 1  # Include band 0 (inactive) + bands 1 to maxorder (active)
    mmax = inverse_sht.mmax
    
    print(f"🔧 CORRECTED mapping: l_bands={l_bands}, mmax={mmax}")
    print(f"🔧 Band 0: INACTIVE (reserved)")  
    print(f"🔧 Bands 1-{maxorder}: ACTIVE (our l=1 to l={maxorder})")
    
    # Initialize coefficient tensor  
    sht_coeffs = torch.zeros(
        batch_size, 2, l_bands, mmax,
        dtype=torch.complex64, device=device
    )
    
    # CORRECTED mapping: our coefficients to torch-harmonics format
    coeff_idx = 0
    
    for l in range(1, maxorder + 1):  # Our l values: 1 to maxorder
        # FIXED: Map to ACTIVE bands, skipping inactive band 0
        th_l_idx = l  # l=1→band1, l=2→band2, ..., l=maxorder→band_maxorder
        
        print(f"🔧 Mapping our l={l} → th_l_idx={th_l_idx}")
        
        if th_l_idx >= l_bands:
            # Skip if beyond supported bands
            coeff_idx += 2 * l + 1
            continue
            
        for m in range(-l, l + 1):
            # IMPROVED m-indexing: use forward SHT convention
            # For real spherical harmonics, only positive m frequencies are stored
            # m=0: stored at index 0
            # m>0: stored at indices 1, 2, 3, ...
            # m<0: handled via conjugate symmetry
            
            if m == 0:
                th_m_idx = 0
            elif m > 0:
                th_m_idx = m
            else:  # m < 0
                # For negative m, use conjugate of positive m coefficient
                th_m_idx = abs(m)
            
            if th_m_idx < mmax:
                # Create complex coefficient
                coeff_real = a_e_real[:, coeff_idx] if m >= 0 else a_e_real[:, coeff_idx]
                coeff_imag = a_e_imag[:, coeff_idx] if m >= 0 else -a_e_imag[:, coeff_idx]
                
                complex_coeff_e = torch.complex(coeff_real, coeff_imag)
                
                coeff_real = a_m_real[:, coeff_idx] if m >= 0 else a_m_real[:, coeff_idx] 
                coeff_imag = a_m_imag[:, coeff_idx] if m >= 0 else -a_m_imag[:, coeff_idx]
                
                complex_coeff_m = torch.complex(coeff_real, coeff_imag)
                
                # Apply conjugate symmetry factor for negative m
                if m < 0:
                    factor = (-1) ** abs(m)
                    complex_coeff_e *= factor
                    complex_coeff_m *= factor
                
                # Store coefficients
                sht_coeffs[:, 0, th_l_idx, th_m_idx] = complex_coeff_e
                sht_coeffs[:, 1, th_l_idx, th_m_idx] = complex_coeff_m
                
                print(f"    m={m:2d} → th_m_idx={th_m_idx}, coeff_idx={coeff_idx}")
            
            coeff_idx += 1
    
    print(f"🔧 Total coefficients mapped: {coeff_idx}")
    print(f"🔧 Non-zero SHT coeffs: {torch.count_nonzero(sht_coeffs).item()}")
    
    return sht_coeffs, inverse_sht

def test_corrected_mapping():
    """Test the corrected coefficient mapping."""
    
    print("🧪 TESTING CORRECTED COEFFICIENT MAPPING")
    print("=" * 70)
    
    maxorder = 5
    grid_shape = (32, 32)  # Known working size
    batch_size = 1
    device = torch.device('cpu')
    
    # Create test coefficients (simple dipole: only l=1, m=0 component)
    n_modes = sum(2*l + 1 for l in range(1, maxorder + 1))
    n_coeffs = 4 * n_modes
    
    coeffs_packed = torch.zeros(batch_size, n_coeffs, dtype=torch.float32)
    
    # Set l=1, m=0 component to 1.0 (should be our first 3 coefficients: m=-1,0,1)
    # For l=1, we have m=-1,0,1, so m=0 is at index 1
    coeffs_packed[0, 1] = 1.0  # a_e_real for l=1, m=0
    
    print(f"📊 Test input: l=1, m=0 component = 1.0")
    print(f"📊 Input shape: {coeffs_packed.shape}")
    
    try:
        # Apply corrected mapping
        sht_coeffs, inverse_sht = corrected_coefficients_to_sht_format(
            coeffs_packed, maxorder, grid_shape, device
        )
        
        # Transform to field
        field = inverse_sht(sht_coeffs)
        field_norm = torch.norm(field).item()
        
        print(f"📊 SHT coeffs shape: {sht_coeffs.shape}")
        print(f"📊 Output field shape: {field.shape}")
        print(f"📊 Field norm: {field_norm:.6f}")
        
        if field_norm > 1e-3:
            print("✅ CORRECTED MAPPING WORKS! Non-zero field produced")
            
            # Analyze which coefficients are non-zero
            nonzero_mask = torch.abs(sht_coeffs) > 1e-6
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False)
            
            print(f"📊 Non-zero SHT coefficients:")
            for idx in nonzero_indices:
                batch, comp, l, m = idx.tolist()
                value = sht_coeffs[batch, comp, l, m]
                comp_name = "E" if comp == 0 else "M"
                print(f"  {comp_name}[l={l}, m={m}] = {value:.6f}")
                
        else:
            print("❌ Still producing zero field - more debugging needed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def create_physics_layer_fix():
    """Create a fixed version of the DifferentiableMultipoleField class."""
    
    fix_code = '''
def corrected_coefficients_to_sht_format(self, coeffs_packed: torch.Tensor) -> torch.Tensor:
    """
    CORRECTED coefficient mapping based on torch-harmonics investigation.
    
    Key fixes:
    1. Skip inactive band 0, map our l=1 to band 1 (first active)
    2. Use proper real SHT m-indexing 
    3. Handle conjugate symmetry correctly
    """
    
    batch_size = coeffs_packed.shape[0]
    
    # Split coefficients
    n_modes = coeffs_packed.shape[1] // 4
    a_e_real, a_e_imag, a_m_real, a_m_imag = torch.split(coeffs_packed, n_modes, dim=1)
    
    # Use maxorder+1 to include inactive band 0
    l_bands = self.maxorder + 1
    mmax = self.n_phi // 2 + 1  # Standard for real FFT
    
    # Initialize with extra band for inactive band 0
    sht_coeffs = torch.zeros(
        batch_size, 2, l_bands, mmax,
        dtype=torch.complex64, device=self.device
    )
    
    coeff_idx = 0
    
    for l in range(1, self.maxorder + 1):
        # CORRECTED: map to active bands, band 0 stays empty/inactive
        th_l_idx = l  # l=1→band1, l=2→band2, etc.
        
        for m in range(-l, l + 1):
            # Proper real SHT m-indexing
            if m >= 0:
                th_m_idx = m
            else:
                # Skip negative m for real SHT (handled via symmetry)
                coeff_idx += 1
                continue
                
            if th_l_idx < l_bands and th_m_idx < mmax:
                # Handle conjugate symmetry for negative m
                if m == 0:
                    # m=0: purely real
                    sht_coeffs[:, 0, th_l_idx, th_m_idx] = torch.complex(
                        a_e_real[:, coeff_idx], torch.zeros_like(a_e_real[:, coeff_idx])
                    )
                    sht_coeffs[:, 1, th_l_idx, th_m_idx] = torch.complex(
                        a_m_real[:, coeff_idx], torch.zeros_like(a_m_real[:, coeff_idx])
                    )
                else:
                    # m>0: complex coefficient
                    sht_coeffs[:, 0, th_l_idx, th_m_idx] = torch.complex(
                        a_e_real[:, coeff_idx], a_e_imag[:, coeff_idx]
                    )
                    sht_coeffs[:, 1, th_l_idx, th_m_idx] = torch.complex(
                        a_m_real[:, coeff_idx], a_m_imag[:, coeff_idx]
                    )
            
            coeff_idx += 1
    
    return sht_coeffs
'''
    
    print("🔧 PHYSICS LAYER FIX")
    print("=" * 70)
    print("The corrected mapping should be applied to DifferentiableMultipoleField")
    print("Key changes needed in src/models/physics_layers.py:")
    print("1. Change th_l_idx = l - 1 to th_l_idx = l")
    print("2. Use lmax = maxorder + 1 (to include inactive band 0)")  
    print("3. Fix m-indexing for real SHT conventions")
    print("4. Handle conjugate symmetry properly")
    
    return fix_code

if __name__ == "__main__":
    test_corrected_mapping()
    print("\n")
    fix_code = create_physics_layer_fix()
    
    # Save the analysis
    with open("physics_layer_fix_analysis.md", "w") as f:
        f.write("""# Physics Layer Fix Analysis

## Problem Identified
Our coefficient mapping in `DifferentiableMultipoleField` has critical errors:

1. **Band 0 Inactive**: torch-harmonics band 0 produces zero fields
2. **Wrong L-mapping**: Our `l=1` maps to inactive band 0 instead of active band 1  
3. **M-indexing Issues**: Centered m-indexing doesn't match real SHT conventions

## Root Cause
```python
th_l_idx = l - 1  # WRONG: maps l=1 to inactive band 0
```

## Solution  
```python
th_l_idx = l      # CORRECT: maps l=1 to active band 1
lmax = maxorder + 1  # Include inactive band 0 in tensor
```

## Impact
This fix should restore gradient flow and enable physics-informed training.
""")
    
    print(f"\n💾 Analysis saved to physics_layer_fix_analysis.md")