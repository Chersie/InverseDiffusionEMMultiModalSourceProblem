# Physics Layer Fix Analysis

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
