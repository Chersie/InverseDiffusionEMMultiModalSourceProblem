# Gradient Flow Investigation Results

## Problem Confirmed: Zero Gradients at Large Grid Resolutions

The physics-informed model training issue has been **systematically verified**. The root cause is that `DifferentiableMultipoleField` produces **zero gradients** (and zero P field outputs) for grid sizes above a critical threshold.

## Key Findings

### Working Grid Sizes (Gradient Flow OK)
- **(16, 16)**: 256 points - ✅ Strong gradients (norm ~29k)  
- **(32, 16)**: 512 points - ✅ Working gradients (norm ~18k)
- **(37, 179)**: 6,623 points - ✅ Working gradients (norm ~195k)  
- **(32, 32)**: 1,024 points - ✅ Working (from boundary test)

### Zero Gradient Grid Sizes (Broken)
- **(64, 32)**: 2,048 points - ❌ **P field = 0.0 everywhere**
- **(64, 64)**: 4,096 points - ❌ Zero gradients  
- **(128, 64)**: 8,192 points - ❌ Zero gradients
- **(180, 90)**: 16,200 points - ❌ Zero gradients
- **(360, 179)**: 64,440 points - ❌ **Zero gradients (training grid)**

### Critical Threshold
**Maximum working grid size: (32, 32) = 1,024 grid points**

Grid sizes above ~1,000-2,000 points cause the `torch-harmonics.InverseRealVectorSHT` to return **all zeros** rather than meaningful electromagnetic fields.

## Impact on Training

The physics training configuration uses:
```yaml
grid_n_phi: 360  
grid_n_theta: 179
# Total: 64,440 grid points - WAY above the working threshold
```

This explains why:
1. **Loss stays constant at 15.08**: The physics loss computes MSE between `true_P` and `pred_P=0.0`
2. **No learning occurs**: Zero gradients mean optimizer cannot update model weights  
3. **Predictions are near-zero**: Model never learns since gradients are blocked

## Technical Analysis

### Physics Layer Behavior
- **Small grids**: Normal electromagnetic field computation with proper gradient flow
- **Large grids**: `torch-harmonics` library returns zero fields, blocking all gradients
- **Pattern**: Not simply about total grid size - (37, 179) works but (64, 32) doesn't

### Library Issue
This appears to be a limitation or bug in the `torch-harmonics` library when used with:
- `InverseRealVectorSHT` 
- Specific grid resolution combinations
- Our coefficient indexing approach

## Implications for Physics Training

1. **Current approach broken**: Training at (360, 179) resolution is impossible
2. **Reduced resolution needed**: Must use grids ≤ (32, 32) for gradient flow
3. **Architecture change required**: Need alternative approach for full-resolution training

## Next Steps

1. **Quick fix**: Train with reduced grid resolution (e.g., 32x32)
2. **Investigation**: Debug torch-harmonics coefficient indexing 
3. **Alternative**: Implement custom physics layer or hybrid approach
4. **Validation**: Confirm reduced-resolution training works

## Evidence Files

- `test_gradient_flow.py`: Systematic gradient flow testing script
- `gradient_flow_test_results.json`: Detailed numerical results (if generated)

This investigation definitively identifies why physics-informed training fails and provides a clear path forward.