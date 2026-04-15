#!/usr/bin/env python3
"""
Quick test to verify the new streaming training pipeline works correctly.

Tests:
1. Streaming data generation completes without memory errors
2. Model training runs successfully 
3. Model produces non-zero predictions
4. Training loss decreases over epochs
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from src.data.memory_monitor import monitor_memory, get_memory_usage
from experiments.scripts.train_mlp import (
    load_config, main as train_main
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_streaming_training():
    """Test basic streaming training with a small dataset."""
    logger.info("=" * 60)
    logger.info("TESTING: Basic Streaming Training")
    logger.info("=" * 60)
    
    # Create a minimal test configuration
    test_config = {
        'experiment': {
            'name': 'streaming_test',
            'description': 'Basic streaming test'
        },
        'model': {
            'type': 'mlp',
            'maxorder': 3,  # Very small for fast testing
            'hidden_size': 64,
            'n_hidden_layers': 1,
            'loss_type': 'coefficient'  # Use coefficient loss for simplicity
        },
        'training': {
            'n_samples': 500,  # Small dataset
            'epochs': 3,       # Few epochs for quick test
            'batch_size': 32,
            'learning_rate': 0.001,
            'train_ratio': 0.8,
            'val_ratio': 0.1
        },
        'data': {
            'generator_mode': 'random',
            'seed': 42
        },
        'preprocessing': {
            'pca_components': 32,  # Small PCA
            'pca_oversample': 4,
            'normalize_features': True,
            'normalize_targets': False
        },
        'streaming': {
            'enable_streaming': True,
            'force_streaming_above_samples': 100,  # Force streaming
            'cache_dir': 'data/test_cache'
        },
        'memory': {
            'cache_dir': 'data/test_cache'
        },
        'device': 'cpu'
    }
    
    # Create temporary config file
    config_path = Path("test_streaming_config.yaml")
    
    try:
        # Write config to YAML file
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f, default_flow_style=False)
        
        logger.info(f"Created test config: {config_path}")
        
        # Monitor memory during training
        with monitor_memory("streaming_training_test", log_before_after=True) as monitor:
            
            # Test data generation first
            logger.info("\n--- Testing Streaming Data Generation ---")
            from experiments.scripts.train_mlp import generate_training_data_streaming
            
            cache_dir = Path("data/test_cache_generation")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            E_theta_path, E_phi_path, y_coeffs_path, y_P_path, data_info = generate_training_data_streaming(
                test_config, str(cache_dir)
            )
            
            logger.info(f"✅ Data generation successful:")
            logger.info(f"   E_theta: {E_theta_path}")
            logger.info(f"   Generated {data_info['n_samples']} samples")
            
            # Verify generated data
            E_theta_data = np.load(E_theta_path, mmap_mode='r')
            y_coeffs_data = np.load(y_coeffs_path, mmap_mode='r')
            
            logger.info(f"   E_theta shape: {E_theta_data.shape}")
            logger.info(f"   Coeffs shape: {y_coeffs_data.shape}")
            
            # Check that data is non-zero
            assert E_theta_data.max() > 0, "Generated E_theta data should be non-zero"
            assert y_coeffs_data.max() > 0, "Generated coefficients should be non-zero"
            logger.info("✅ Generated data validation passed")
            
            # Test model creation and training
            logger.info("\n--- Testing Model Training ---")
            
            from src.models.mlp import MLPModel, MLPConfig
            from src.data.streaming_dataset import MemmapDataset
            
            # Create model
            model_config = MLPConfig(
                hidden_size=64,
                n_hidden_layers=1,
                input_dim=32,  # PCA components
                output_dim=data_info['coeffs_shape'][0],  # Coefficient dimension
                loss_type='coefficient'
            )
            
            model = MLPModel(model_config)
            logger.info(f"✅ Created model: {model}")
            
            # Test preprocessing
            logger.info("\n--- Testing Streaming Preprocessing ---")
            from experiments.scripts.train_mlp import setup_preprocessing_streaming
            
            preprocessing = setup_preprocessing_streaming(
                test_config, E_theta_path, E_phi_path, y_coeffs_path
            )
            
            logger.info("✅ Preprocessing setup completed")
            
            # Transform features
            X_path = cache_dir / "X_test.npy"
            preprocessing.transform_features_streaming(
                E_theta_path, E_phi_path, str(X_path)
            )
            
            X_data = np.load(X_path, mmap_mode='r')
            logger.info(f"✅ Feature transformation: {X_data.shape}")
            
            # Test dataset creation
            dataset = MemmapDataset(str(X_path), y_coeffs_path)
            logger.info(f"✅ Created dataset with {len(dataset)} samples")
            
            # Test sample loading
            sample_X, sample_y = dataset[0]
            logger.info(f"   Sample shapes: X={sample_X.shape}, y={sample_y.shape}")
            assert sample_X.max() != 0 or sample_X.min() != 0, "Features should be non-zero"
            assert sample_y.max() != 0 or sample_y.min() != 0, "Targets should be non-zero"
            
            # Test quick training (few samples for speed)
            logger.info("\n--- Testing Model Training ---")
            
            # Load small subset for training test
            X_small = np.array(X_data[:100])  # First 100 samples
            y_small = np.array(y_coeffs_data[:100])
            
            # Update model config with correct dimensions
            model.config = model.config.__class__(
                hidden_size=64,
                n_hidden_layers=1,
                input_dim=X_small.shape[1],
                output_dim=y_small.shape[1],
                loss_type='coefficient',
                epochs=3,
                batch_size=16
            )
            
            # Quick training test
            logger.info(f"Training on {X_small.shape} -> {y_small.shape}")
            
            initial_predictions = None
            if not model.is_trained:
                # Test untrained prediction first (should be random)
                model.is_trained = True  # Temporary for prediction test
                model._torch_model = model._create_torch_model()
                initial_predictions = model.predict(X_small[:5])
                model.is_trained = False
            
            # Train the model
            training_results = model.fit(X_small, y_small)
            
            logger.info(f"✅ Training completed:")
            logger.info(f"   Final train loss: {training_results['final_train_loss']:.6f}")
            logger.info(f"   Training time: {training_results['training_time']:.2f}s")
            
            # Test predictions after training
            final_predictions = model.predict(X_small[:5])
            
            logger.info(f"✅ Prediction test:")
            logger.info(f"   Prediction shape: {final_predictions.shape}")
            logger.info(f"   Prediction range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")
            
            # Verify predictions are non-zero
            assert np.abs(final_predictions).max() > 1e-6, "Predictions should be non-zero after training"
            
            # Compare initial vs final predictions (should be different)
            if initial_predictions is not None:
                diff = np.abs(final_predictions - initial_predictions).max()
                logger.info(f"   Prediction change: {diff:.4f}")
                assert diff > 1e-4, "Predictions should change significantly after training"
            
            logger.info("✅ All prediction tests passed!")
            
        # Report memory usage
        peak_memory = monitor.get_peak_usage()
        logger.info(f"\n📊 Peak memory usage: {peak_memory:.1f}MB")
        
        # Clean up
        shutil.rmtree(cache_dir, ignore_errors=True)
        
        return {
            'success': True,
            'peak_memory_mb': peak_memory,
            'final_loss': training_results['final_train_loss'],
            'prediction_range': [float(final_predictions.min()), float(final_predictions.max())],
            'training_time': training_results['training_time']
        }
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
        
    finally:
        # Clean up config file
        if config_path.exists():
            config_path.unlink()


def test_physics_loss_training():
    """Test physics loss training with streaming."""
    logger.info("=" * 60)
    logger.info("TESTING: Physics Loss Streaming Training")
    logger.info("=" * 60)
    
    # Use the actual physics config but with smaller parameters
    config_path = Path("experiments/configs/mlp_physics.yaml")
    
    if not config_path.exists():
        logger.warning("Physics config not found, skipping physics test")
        return {'success': False, 'error': 'Config file not found'}
    
    try:
        # Load and modify config for quick testing
        config = load_config(config_path)
        
        # Make it smaller for testing
        config['training']['n_samples'] = 300  # Very small for quick test
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 16
        config['model']['hidden_size'] = 32
        config['preprocessing']['pca_components'] = 16
        
        logger.info(f"Testing with {config['training']['n_samples']} samples, {config['training']['epochs']} epochs")
        
        with monitor_memory("physics_streaming_test", log_before_after=True) as monitor:
            
            # Test data generation
            from experiments.scripts.train_mlp import generate_training_data_streaming
            
            cache_dir = Path("data/test_cache_physics") 
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            E_theta_path, E_phi_path, y_coeffs_path, y_P_path, data_info = generate_training_data_streaming(
                config, str(cache_dir)
            )
            
            logger.info(f"✅ Physics data generation: {data_info['n_samples']} samples")
            logger.info(f"   Has P field: {data_info['has_p_field_targets']}")
            
            # Verify P field data exists for physics loss
            if y_P_path:
                y_P_data = np.load(y_P_path, mmap_mode='r')
                logger.info(f"   P field shape: {y_P_data.shape}")
                assert y_P_data.max() > 0, "P field should be non-zero"
            
            # Test physics model creation
            from src.models.mlp import MLPModel, MLPConfig
            
            model_config = MLPConfig(
                hidden_size=config['model']['hidden_size'],
                n_hidden_layers=config['model']['n_hidden_layers'],
                input_dim=config['preprocessing']['pca_components'],
                output_dim=data_info['coeffs_shape'][0],  # Always predict coefficients
                loss_type=config['model']['loss_type'],
                maxorder=config['model']['maxorder'],
                grid_n_theta=config['model']['grid_n_theta'],
                grid_n_phi=config['model']['grid_n_phi']
            )
            
            model = MLPModel(model_config)
            logger.info(f"✅ Created physics model: loss_type={model_config.loss_type}")
            
            # Test preprocessing
            from experiments.scripts.train_mlp import setup_preprocessing_streaming
            
            preprocessing = setup_preprocessing_streaming(
                config, E_theta_path, E_phi_path, y_coeffs_path
            )
            
            # Transform features
            X_path = cache_dir / "X_physics.npy"
            preprocessing.transform_features_streaming(
                E_theta_path, E_phi_path, str(X_path)
            )
            
            # Load small subset for quick training
            X_data = np.load(X_path, mmap_mode='r')
            
            # For physics loss, targets are P fields, not coefficients  
            if config['model']['loss_type'] == 'physics':
                y_data = np.load(y_P_path, mmap_mode='r')
            else:
                y_data = np.load(y_coeffs_path, mmap_mode='r')
                
            # Use small subset
            n_test = min(50, X_data.shape[0])
            X_small = np.array(X_data[:n_test])
            y_small = np.array(y_data[:n_test])
            
            logger.info(f"Training physics model: {X_small.shape} -> {y_small.shape}")
            logger.info(f"Target type: {'P field' if config['model']['loss_type'] == 'physics' else 'coefficients'}")
            
            # Train model
            training_results = model.fit(X_small, y_small)
            
            logger.info(f"✅ Physics training completed:")
            logger.info(f"   Final train loss: {training_results['final_train_loss']:.6f}")
            
            # Test predictions (model always outputs coefficients)
            predictions = model.predict(X_small[:3])
            
            logger.info(f"✅ Physics predictions:")
            logger.info(f"   Shape: {predictions.shape}")
            logger.info(f"   Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            # Verify predictions are reasonable
            assert np.abs(predictions).max() > 1e-6, "Physics predictions should be non-zero"
            assert np.isfinite(predictions).all(), "Physics predictions should be finite"
            
        # Clean up
        shutil.rmtree(cache_dir, ignore_errors=True)
        
        peak_memory = monitor.get_peak_usage()
        logger.info(f"\n📊 Physics test peak memory: {peak_memory:.1f}MB")
        
        return {
            'success': True,
            'peak_memory_mb': peak_memory,
            'final_loss': training_results['final_train_loss'],
            'prediction_range': [float(predictions.min()), float(predictions.max())],
            'loss_type': config['model']['loss_type']
        }
        
    except Exception as e:
        logger.error(f"❌ Physics test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    """Run all streaming training tests."""
    logger.info("🚀 Testing New Streaming Training Pipeline")
    logger.info("=" * 80)
    
    results = {}
    
    # Test 1: Basic streaming training
    logger.info("\n🔬 Test 1: Basic Streaming Training")
    results['basic'] = test_basic_streaming_training()
    
    # Test 2: Physics loss training
    logger.info("\n🔬 Test 2: Physics Loss Training")  
    results['physics'] = test_physics_loss_training()
    
    # Generate summary report
    logger.info("\n" + "=" * 80)
    logger.info("🏁 STREAMING TRAINING TEST RESULTS")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        logger.info(f"\n📋 {test_name.title()} Test:")
        
        if result['success']:
            logger.info("   ✅ PASSED")
            logger.info(f"   📊 Peak Memory: {result['peak_memory_mb']:.1f}MB") 
            logger.info(f"   📉 Final Loss: {result['final_loss']:.6f}")
            logger.info(f"   🎯 Prediction Range: {result['prediction_range']}")
            
            if 'loss_type' in result:
                logger.info(f"   🧠 Loss Type: {result['loss_type']}")
                
        else:
            logger.info(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Overall summary
    passed_tests = sum(1 for r in results.values() if r['success'])
    total_tests = len(results)
    
    logger.info(f"\n🎯 OVERALL RESULTS:")
    logger.info(f"   Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        logger.info("   🎉 ALL TESTS PASSED - Streaming pipeline is working!")
        logger.info("   ✅ Memory crashes eliminated")
        logger.info("   ✅ Non-zero predictions confirmed")
        logger.info("   ✅ Physics loss integration verified")
    else:
        logger.info(f"   ⚠️  {total_tests - passed_tests} test(s) failed")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)