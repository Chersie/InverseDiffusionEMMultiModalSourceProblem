#!/usr/bin/env python3
"""
Analyze scaling mismatch between normalized features and raw P field targets.
"""

import numpy as np
import torch
from pathlib import Path
from src.core.data_generator import DataGenerator, LatinSquareConfig, GridConfig
from src.api.preprocessing import PreprocessingPipeline, PreprocessingConfig

def analyze_feature_target_scaling():
    """Analyze the scaling of features vs targets in our pipeline."""
    
    print("🔍 ANALYZING FEATURE-TARGET SCALING")
    print("=" * 70)
    
    # Generate test dataset  
    maxorder = 3
    n_samples = 200
    grid_shape = (360, 179)
    
    print(f"📊 Generating {n_samples} samples for analysis...")
    
    # Generate data  
    latin_config = LatinSquareConfig(mode='random', scale=1.0)
    grid_config = GridConfig(n_phi=grid_shape[0], n_theta=grid_shape[1])
    generator = DataGenerator(latin_config=latin_config, grid_config=grid_config)
    dataset = generator.generate_batch(maxorder=maxorder, n_samples=n_samples)
    
    # Extract fields and coefficients
    E_theta = dataset['amplitude'][..., 0]  # (n_samples, n_phi, n_theta)
    E_phi = dataset['amplitude'][..., 1] 
    
    # Compute P field targets: use |z|^2 so complex amplitudes give real power density
    P_targets = (np.abs(E_theta)**2 + np.abs(E_phi)**2).astype(np.float32)
    
    print(f"📊 Raw data shapes:")
    print(f"   E_theta: {E_theta.shape}")
    print(f"   E_phi: {E_phi.shape}") 
    print(f"   P_targets: {P_targets.shape}")
    
    # Analyze raw data statistics
    print(f"\n📊 Raw data statistics:")
    print(f"   E_theta: mean={E_theta.mean():.6f}, std={E_theta.std():.6f}, range=[{E_theta.min():.6f}, {E_theta.max():.6f}]")
    print(f"   E_phi:   mean={E_phi.mean():.6f}, std={E_phi.std():.6f}, range=[{E_phi.min():.6f}, {E_phi.max():.6f}]")
    print(f"   P_field: mean={P_targets.mean():.6f}, std={P_targets.std():.6f}, range=[{P_targets.min():.6f}, {P_targets.max():.6f}]")
    
    # Set up preprocessing for features
    config = PreprocessingConfig(
        pca_components=64,
        normalize_features=True,
        normalize_targets=False  # Current setting - no target normalization
    )
    
    preprocessing = PreprocessingPipeline(config)
    
    # Fit preprocessing on the field data
    print(f"\n🔧 Applying preprocessing to features...")
    preprocessing.fit(E_theta, E_phi)
    
    # Transform features
    features_processed = preprocessing.transform_features(E_theta, E_phi)
    
    print(f"📊 Processed feature statistics:")
    print(f"   Features: shape={features_processed.shape}")
    print(f"   Features: mean={features_processed.mean():.6f}, std={features_processed.std():.6f}")
    print(f"   Features: range=[{features_processed.min():.6f}, {features_processed.max():.6f}]")
    
    # Analyze scaling mismatch
    feature_scale = features_processed.std()
    target_scale = P_targets.std()
    scaling_ratio = target_scale / feature_scale
    
    print(f"\n⚖️  SCALING ANALYSIS:")
    print(f"   Feature scale (std): {feature_scale:.6f}")
    print(f"   Target scale (std):  {target_scale:.6f}")  
    print(f"   Scaling ratio (target/feature): {scaling_ratio:.2f}")
    
    if scaling_ratio > 10:
        print(f"   🚨 LARGE SCALING MISMATCH! Targets are {scaling_ratio:.0f}x larger than features")
        print(f"   💡 This can cause training difficulties (poor conditioning)")
    elif scaling_ratio > 3:
        print(f"   ⚠️  Moderate scaling mismatch. Targets are {scaling_ratio:.1f}x larger")
    else:
        print(f"   ✅ Reasonable scaling ratio")
    
    return {
        'features_processed': features_processed,
        'P_targets': P_targets,
        'feature_scale': feature_scale,
        'target_scale': target_scale,
        'scaling_ratio': scaling_ratio,
        'preprocessing': preprocessing
    }

def test_target_normalization_options():
    """Test different target normalization strategies."""
    
    print(f"\n🧪 TESTING TARGET NORMALIZATION OPTIONS")
    print("=" * 70)
    
    # Generate test data
    maxorder = 3
    n_samples = 200
    grid_shape = (32, 16)  # Smaller grid for testing
    
    latin_config = LatinSquareConfig(mode='random', scale=1.0)
    grid_config = GridConfig(n_phi=grid_shape[0], n_theta=grid_shape[1])
    generator = DataGenerator(latin_config=latin_config, grid_config=grid_config)
    dataset = generator.generate_batch(maxorder=maxorder, n_samples=n_samples)
    
    E_theta = dataset['amplitude'][..., 0]
    E_phi = dataset['amplitude'][..., 1]
    P_targets = (np.abs(E_theta)**2 + np.abs(E_phi)**2).astype(np.float32)
    
    print(f"📊 Original P targets: mean={P_targets.mean():.4f}, std={P_targets.std():.4f}")
    
    # Test different normalization strategies
    strategies = {
        'log_transform': np.log(P_targets + 1e-8),
        'sqrt_transform': np.sqrt(P_targets),
        'standardization': (P_targets - P_targets.mean()) / P_targets.std(),
        'min_max_scaling': (P_targets - P_targets.min()) / (P_targets.max() - P_targets.min()),
        'robust_scaling': (P_targets - np.median(P_targets)) / np.percentile(P_targets, [25, 75]).ptp()
    }
    
    print(f"\n📊 Target normalization strategies:")
    for name, transformed in strategies.items():
        print(f"   {name:15s}: mean={transformed.mean():.4f}, std={transformed.std():.4f}, range=[{transformed.min():.4f}, {transformed.max():.4f}]")
    
    return strategies

def test_physics_training_with_scaling():
    """Test how scaling affects physics training convergence."""
    
    print(f"\n🧪 TESTING PHYSICS TRAINING WITH DIFFERENT SCALING")
    print("=" * 70)
    
    # Test parameters
    maxorder = 2  # Small for fast testing
    n_samples = 100
    grid_shape = (16, 16)
    n_epochs = 5
    
    latin_config = LatinSquareConfig(mode='random', scale=1.0)
    grid_config = GridConfig(n_phi=grid_shape[0], n_theta=grid_shape[1])
    generator = DataGenerator(latin_config=latin_config, grid_config=grid_config)
    dataset = generator.generate_batch(maxorder=maxorder, n_samples=n_samples)
    
    E_theta = dataset['amplitude'][..., 0]
    E_phi = dataset['amplitude'][..., 1] 
    P_targets = (np.abs(E_theta)**2 + np.abs(E_phi)**2).astype(np.float32)
    
    # Setup preprocessing for features
    config = PreprocessingConfig(pca_components=32, normalize_features=True)
    preprocessing = PreprocessingPipeline(config)
    preprocessing.fit(E_theta, E_phi)
    features = preprocessing.transform_features(E_theta, E_phi)
    
    # Test different target scalings
    target_variants = {
        'raw': P_targets,
        'log': np.log(P_targets + 1e-8),
        'sqrt': np.sqrt(P_targets),
        'standard': (P_targets - P_targets.mean()) / P_targets.std()
    }
    
    # Simple test: see which scaling gives better initial loss values
    results = {}
    
    for variant_name, targets in target_variants.items():
        print(f"\n🧪 Testing '{variant_name}' target scaling:")
        
        # Convert to tensors
        X = torch.from_numpy(features).float()
        y = torch.from_numpy(targets).float()
        
        print(f"   Features: mean={X.mean():.4f}, std={X.std():.4f}")
        print(f"   Targets:  mean={y.mean():.4f}, std={y.std():.4f}")
        
        # Simple loss computation (MSE)
        mean_prediction = y.mean()  # Simplest possible "model"
        mse_loss = torch.mean((y - mean_prediction)**2).item()
        
        print(f"   Baseline MSE loss: {mse_loss:.6f}")
        
        # Estimate gradient magnitude (proxy for training difficulty)
        grad_estimate = torch.std(y - mean_prediction).item()
        print(f"   Gradient estimate: {grad_estimate:.6f}")
        
        results[variant_name] = {
            'mse_loss': mse_loss,
            'grad_estimate': grad_estimate,
            'target_mean': y.mean().item(),
            'target_std': y.std().item()
        }
    
    print(f"\n📊 SUMMARY - Target Scaling Comparison:")
    print(f"{'Variant':12s} {'MSE Loss':>12s} {'Grad Est':>12s} {'Target Std':>12s}")
    print("-" * 50)
    for variant, stats in results.items():
        print(f"{variant:12s} {stats['mse_loss']:>12.6f} {stats['grad_estimate']:>12.6f} {stats['target_std']:>12.4f}")
    
    # Recommendation
    best_variant = min(results.keys(), key=lambda k: results[k]['mse_loss'])
    print(f"\n💡 RECOMMENDATION: '{best_variant}' scaling shows best initial behavior")
    
    return results

def create_scaling_fix():
    """Create configuration recommendations for proper scaling."""
    
    print(f"\n🔧 SCALING FIX RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = """
# Target Scaling Options for Physics Training

## Option 1: Log Transform (for wide dynamic range)
preprocessing:
  normalize_targets: true
  target_transform: "log"  # log(P + eps)
  target_eps: 1e-8

## Option 2: Square Root Transform (physics motivated)  
preprocessing:
  normalize_targets: true
  target_transform: "sqrt"  # sqrt(P)

## Option 3: Standardization (statistics-based)
preprocessing:
  normalize_targets: true 
  target_transform: "standard"  # (P - mean) / std

## Option 4: Adaptive Scaling
preprocessing:
  normalize_targets: true
  target_transform: "adaptive"  # Auto-detect best scaling
"""
    
    print(recommendations)
    
    print(f"\n🎯 IMPLEMENTATION PLAN:")
    print(f"1. Add target_transform parameter to PreprocessingConfig")
    print(f"2. Implement transform/inverse_transform methods")  
    print(f"3. Apply transforms in preprocessing pipeline")
    print(f"4. Handle inverse transforms during evaluation")
    print(f"5. Update physics training configs to use scaling")
    
    return recommendations

if __name__ == "__main__":
    # Run analysis
    scaling_analysis = analyze_feature_target_scaling()
    target_strategies = test_target_normalization_options()
    training_results = test_physics_training_with_scaling()
    recommendations = create_scaling_fix()
    
    # Save results
    print(f"\n💾 Analysis complete - key findings:")
    print(f"   Scaling ratio: {scaling_analysis['scaling_ratio']:.1f}x")
    print(f"   Feature scale: {scaling_analysis['feature_scale']:.4f}")
    print(f"   Target scale: {scaling_analysis['target_scale']:.4f}")
    
    if scaling_analysis['scaling_ratio'] > 5:
        print(f"   🚨 Action needed: Implement target normalization")
    else:
        print(f"   ✅ Scaling is manageable")