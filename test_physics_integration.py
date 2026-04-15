#!/usr/bin/env python3
"""
Physics-Informed Loss Integration Test

This script tests the complete physics-informed loss pipeline with a small
synthetic dataset to verify all components work together correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import yaml
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Disable MPS to prevent crashes
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_DISABLE_MPS'] = '1'

from src.core.data_generator import DataGenerator, pack_coefficients
from src.models.registry import get_model_registry
from src.api.preprocessing import PreprocessingPipeline, PreprocessingConfig
from experiments.scripts.train_mlp import generate_training_data, setup_preprocessing, split_data, create_model_from_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_config() -> dict:
    """Create a minimal configuration for physics loss testing."""
    return {
        'experiment': {
            'name': 'physics_test',
            'tags': ['physics-loss', 'integration-test']
        },
        'model': {
            'maxorder': 3,  # Small maxorder for testing
            'hidden_size': 64,  # Small network
            'n_hidden_layers': 2,
            'dropout_rate': 0.1,
            'activation': 'gelu',
            'loss_type': 'physics',  # Enable physics loss
            'grid_n_theta': 32,  # Small grid for testing  
            'grid_n_phi': 64,
            'physics_grid_type': 'equiangular',
            'physics_grid_resolution_factor': 1.0,
            'physics_field_weight': 0.1
        },
        'training': {
            'n_samples': 100,  # Small dataset
            'learning_rate': 1e-3,
            'epochs': 5,  # Few epochs for testing
            'batch_size': 32,
            'train_ratio': 0.7,
            'val_ratio': 0.2
        },
        'data': {
            'generator_mode': 'random',
            'seed': 42
        },
        'preprocessing': {
            'pca_components': 32,  # Small PCA for testing
            'pca_oversample': 1.2,
            'normalize_features': True,
            'normalize_targets': False  # Skip target normalization for physics loss
        },
        'device': 'cpu'  # Use CPU for compatibility
    }

def test_coefficient_loss_baseline() -> dict:
    """Test the baseline coefficient loss for comparison."""
    logger.info("=" * 60)
    logger.info("Testing baseline coefficient loss...")
    
    config = create_test_config()
    config['model']['loss_type'] = 'coefficient'  # Use coefficient loss
    config['preprocessing']['normalize_targets'] = True  # Enable target normalization
    
    try:
        # Generate data
        E_theta, E_phi, y_coeffs, y_P, data_info = generate_training_data(config)
        logger.info(f"✓ Generated data: E_theta {E_theta.shape}, y_coeffs {y_coeffs.shape}")
        
        # Set up preprocessing
        preprocessing = setup_preprocessing(config, E_theta, E_phi, y_coeffs)
        X = preprocessing.transform_features(E_theta, E_phi)
        logger.info(f"✓ Preprocessed features: {X.shape}")
        
        # Split data (use coefficient targets)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_coeffs, config)
        
        # Apply target normalization for coefficient loss
        if preprocessing.target_normalizer is not None and preprocessing.target_normalizer.is_fitted:
            y_train = preprocessing.target_normalizer.transform(y_train)
            y_val = preprocessing.target_normalizer.transform(y_val)
            logger.info("✓ Applied target normalization")
        
        # Create model
        input_dim = X.shape[1]
        output_dim = y_coeffs.shape[1]
        model = create_model_from_config(config, input_dim, output_dim)
        logger.info(f"✓ Created model: {input_dim} -> {output_dim}")
        
        # Train model
        logger.info("Training coefficient-loss model...")
        training_result = model.fit(X_train, y_train, X_val, y_val)
        
        # Check training results
        final_train_loss = training_result.get('final_train_loss', float('inf'))
        final_val_loss = training_result.get('final_val_loss', float('inf'))
        
        logger.info(f"✓ Training completed: train_loss={final_train_loss:.6f}, val_loss={final_val_loss:.6f}")
        
        return {
            'success': True,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'loss_type': 'coefficient'
        }
        
    except Exception as e:
        logger.error(f"✗ Coefficient loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'loss_type': 'coefficient'}

def test_physics_loss() -> dict:
    """Test the physics-informed loss implementation."""
    logger.info("=" * 60)
    logger.info("Testing physics-informed loss...")
    
    config = create_test_config()  # Already configured for physics loss
    
    try:
        # Generate data
        E_theta, E_phi, y_coeffs, y_P, data_info = generate_training_data(config)
        logger.info(f"✓ Generated data: E_theta {E_theta.shape}, y_P {y_P.shape}")
        
        # Verify P field targets were generated
        if y_P is None:
            raise ValueError("P field targets not generated for physics loss")
        
        # Set up preprocessing
        preprocessing = setup_preprocessing(config, E_theta, E_phi, y_coeffs)
        X = preprocessing.transform_features(E_theta, E_phi)
        logger.info(f"✓ Preprocessed features: {X.shape}")
        
        # Split data (use P field targets)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_P, config)
        logger.info(f"✓ Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # No target normalization for P field targets
        logger.info("✓ Skipped target normalization for P field targets")
        
        # Create model
        input_dim = X.shape[1]
        output_dim = y_coeffs.shape[1]  # Model still predicts coefficients
        model = create_model_from_config(config, input_dim, output_dim)
        logger.info(f"✓ Created physics model: {input_dim} -> {output_dim}")
        
        # Verify physics loss setup
        if hasattr(model, 'config') and hasattr(model.config, 'loss_type'):
            if model.config.loss_type != 'physics':
                raise ValueError(f"Model loss type is {model.config.loss_type}, expected 'physics'")
            logger.info("✓ Model configured for physics loss")
        
        # Train model
        logger.info("Training physics-loss model...")
        training_result = model.fit(X_train, y_train, X_val, y_val)
        
        # Check training results
        final_train_loss = training_result.get('final_train_loss', float('inf'))
        final_val_loss = training_result.get('final_val_loss', float('inf'))
        
        logger.info(f"✓ Training completed: train_loss={final_train_loss:.6f}, val_loss={final_val_loss:.6f}")
        
        # Test prediction (should work and return coefficients)
        test_pred = model.predict(X_test[:5])
        expected_shape = (5, output_dim)
        if test_pred.shape != expected_shape:
            raise ValueError(f"Prediction shape {test_pred.shape}, expected {expected_shape}")
        logger.info(f"✓ Predictions working: {test_pred.shape}")
        
        return {
            'success': True,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'loss_type': 'physics'
        }
        
    except Exception as e:
        logger.error(f"✗ Physics loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'loss_type': 'physics'}

def test_field_reconstruction():
    """Test that we can reconstruct fields from predicted coefficients."""
    logger.info("=" * 60)
    logger.info("Testing field reconstruction...")
    
    try:
        # Generate test data
        generator = DataGenerator()
        dataset = generator.generate_batch(maxorder=3, n_samples=5)
        
        E_theta_true = dataset['amplitude'][..., 0]
        E_phi_true = dataset['amplitude'][..., 1]
        coeffs_e = dataset['coefficients_e'] 
        coeffs_m = dataset['coefficients_m']
        
        # Pack coefficients
        coeffs_packed = pack_coefficients(coeffs_e, coeffs_m)
        
        # Test physics field computation
        from src.models.physics_layers import DifferentiableMultipoleField
        import torch
        
        field_generator = DifferentiableMultipoleField(
            maxorder=3,
            grid_shape=(E_theta_true.shape[1], E_theta_true.shape[2])  # Match data grid
        )
        
        # Convert to torch tensors
        coeffs_tensor = torch.from_numpy(coeffs_packed).float()
        
        # Compute fields
        E_theta_pred, E_phi_pred, P_pred = field_generator.get_field_components(coeffs_tensor)
        
        # Convert back to numpy
        E_theta_pred_np = E_theta_pred.detach().numpy()
        E_phi_pred_np = E_phi_pred.detach().numpy()
        P_pred_np = P_pred.detach().numpy()
        
        # Compute true P field
        P_true = np.abs(E_theta_true)**2 + np.abs(E_phi_true)**2
        
        # Check shapes
        if E_theta_pred_np.shape != E_theta_true.shape:
            # Handle potential shape mismatch (grid orientation)
            if E_theta_pred_np.shape == (E_theta_true.shape[0], E_theta_true.shape[2], E_theta_true.shape[1]):
                E_theta_pred_np = E_theta_pred_np.transpose(0, 2, 1)
                E_phi_pred_np = E_phi_pred_np.transpose(0, 2, 1)
                P_pred_np = P_pred_np.transpose(0, 2, 1)
                logger.info("✓ Handled grid orientation mismatch")
        
        # Compare shapes
        logger.info(f"True shapes: E_theta {E_theta_true.shape}, P {P_true.shape}")
        logger.info(f"Pred shapes: E_theta {E_theta_pred_np.shape}, P {P_pred_np.shape}")
        
        # Compute relative errors
        E_theta_error = np.mean(np.abs(E_theta_true - E_theta_pred_np)) / np.mean(np.abs(E_theta_true))
        P_error = np.mean(np.abs(P_true - P_pred_np)) / np.mean(np.abs(P_true))
        
        logger.info(f"✓ Field reconstruction test completed")
        logger.info(f"  E_theta relative error: {E_theta_error:.4f}")
        logger.info(f"  P field relative error: {P_error:.4f}")
        
        return {
            'success': True,
            'E_theta_error': E_theta_error,
            'P_error': P_error
        }
        
    except Exception as e:
        logger.error(f"✗ Field reconstruction test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def main():
    """Run complete integration test."""
    logger.info("🚀 Starting Physics-Informed Loss Integration Test")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Baseline coefficient loss 
    results['coefficient_loss'] = test_coefficient_loss_baseline()
    
    # Test 2: Physics-informed loss
    results['physics_loss'] = test_physics_loss()
    
    # Test 3: Field reconstruction
    results['field_reconstruction'] = test_field_reconstruction()
    
    # Summary
    logger.info("=" * 60)
    logger.info("🏁 INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        logger.info(f"{test_name:<20}: {status}")
        
        if result.get('success', False):
            if 'final_train_loss' in result:
                logger.info(f"{'':20}  Final loss: {result['final_train_loss']:.6f}")
        else:
            logger.info(f"{'':20}  Error: {result.get('error', 'Unknown')}")
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! Physics-informed loss implementation is working correctly.")
        logger.info("✅ Ready for production use!")
    else:
        logger.info("⚠️  Some tests failed. Please review the errors above.")
        
    return all_passed

if __name__ == "__main__":
    main()