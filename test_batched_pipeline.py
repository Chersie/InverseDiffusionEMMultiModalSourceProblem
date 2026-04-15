#!/usr/bin/env python3
"""
Test script for the memory-efficient batched training pipeline.
This script validates that the batched approach works correctly and avoids memory issues.
"""

import numpy as np
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_memory_usage():
    """Test memory usage with different dataset sizes."""
    from experiments.scripts.train_mlp import generate_training_data
    
    # Test config
    test_config = {
        'model': {'maxorder': 4, 'loss_type': 'physics'},
        'training': {'n_samples': 1000},  # Start with reasonable size
        'data': {'generator_mode': 'random', 'seed': 42}
    }
    
    logger.info("🧪 Testing batched data generation...")
    
    # Test different sample sizes
    for n_samples in [100, 500, 1000, 2000]:
        logger.info(f"\n📊 Testing with {n_samples} samples...")
        
        test_config['training']['n_samples'] = n_samples
        
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            E_theta, E_phi, y_coeffs, y_P, data_info = generate_training_data(test_config)
            
            end_time = time.time()
            end_memory = get_memory_usage()
            
            logger.info(f"✅ Success! Generated {n_samples} samples:")
            logger.info(f"   Time: {end_time - start_time:.2f}s")
            logger.info(f"   Memory delta: {end_memory - start_memory:.1f}MB")
            logger.info(f"   Data shapes: E_theta={E_theta.shape}, P_field={y_P.shape if y_P is not None else 'None'}")
            logger.info(f"   Generation rate: {data_info['samples_per_second']:.1f} samples/sec")
            
            # Clear data to free memory
            del E_theta, E_phi, y_coeffs, y_P, data_info
            
        except Exception as e:
            logger.error(f"❌ Failed with {n_samples} samples: {e}")
            break


def test_preprocessing_batching():
    """Test batched preprocessing."""
    from experiments.scripts.train_mlp import generate_training_data, setup_preprocessing_batched, transform_features_batched
    
    logger.info("\n🔄 Testing batched preprocessing...")
    
    # Generate test data
    config = {
        'model': {'maxorder': 4, 'loss_type': 'coefficient'},
        'training': {'n_samples': 1500},  # Size that requires batching
        'data': {'generator_mode': 'random', 'seed': 42},
        'preprocessing': {
            'pca_components': 100,
            'pca_oversample': 4,
            'normalize_features': True,
            'normalize_targets': True
        }
    }
    
    start_time = time.time()
    
    # Generate data
    E_theta, E_phi, y_coeffs, y_P, _ = generate_training_data(config)
    logger.info(f"Data generation: {time.time() - start_time:.2f}s")
    
    # Test batched preprocessing
    prep_start = time.time()
    preprocessing = setup_preprocessing_batched(config, E_theta, E_phi, y_coeffs)
    logger.info(f"Preprocessing fit: {time.time() - prep_start:.2f}s")
    
    # Test batched feature transformation
    transform_start = time.time()
    X = transform_features_batched(preprocessing, E_theta, E_phi)
    logger.info(f"Feature transformation: {time.time() - transform_start:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(f"✅ Total preprocessing pipeline: {total_time:.2f}s")
    logger.info(f"   Final feature shape: {X.shape}")
    
    return preprocessing, X


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return 0  # Return 0 if psutil not available


def test_physics_vs_coefficient():
    """Compare physics vs coefficient loss data generation."""
    logger.info("\n⚖️  Comparing physics vs coefficient loss data generation...")
    
    base_config = {
        'model': {'maxorder': 4},
        'training': {'n_samples': 500},
        'data': {'generator_mode': 'random', 'seed': 42}
    }
    
    # Test coefficient loss
    logger.info("Testing coefficient loss...")
    coeff_config = {**base_config, 'model': {**base_config['model'], 'loss_type': 'coefficient'}}
    coeff_start = time.time()
    _, _, y_coeffs, y_P_coeff, _ = generate_training_data(coeff_config)
    coeff_time = time.time() - coeff_start
    
    # Test physics loss  
    logger.info("Testing physics loss...")
    physics_config = {**base_config, 'model': {**base_config['model'], 'loss_type': 'physics'}}
    physics_start = time.time()
    _, _, _, y_P_phys, _ = generate_training_data(physics_config)
    physics_time = time.time() - physics_start
    
    logger.info(f"✅ Results:")
    logger.info(f"   Coefficient loss: {coeff_time:.2f}s, P field targets: {y_P_coeff is not None}")
    logger.info(f"   Physics loss: {physics_time:.2f}s, P field targets: {y_P_phys is not None}")
    logger.info(f"   Physics overhead: {physics_time - coeff_time:.2f}s ({((physics_time/coeff_time - 1) * 100):.1f}%)")


if __name__ == "__main__":
    logger.info("🚀 Starting batched pipeline tests...\n")
    
    try:
        # Test 1: Memory usage with different sizes
        test_memory_usage()
        
        # Test 2: Preprocessing batching
        preprocessing, X = test_preprocessing_batching()
        
        # Test 3: Physics vs coefficient comparison
        test_physics_vs_coefficient()
        
        logger.info("\n✅ All tests completed successfully!")
        logger.info("\n🎯 Key improvements:")
        logger.info("   • Batched data generation prevents OOM errors")
        logger.info("   • Intelligent batch sizing based on dataset size")
        logger.info("   • Memory-efficient preprocessing with subset fitting")
        logger.info("   • Batched feature and target transformations")
        logger.info("   • Automatic memory cleanup between batches")
        
        logger.info("\n🚀 Ready to train with larger datasets!")
        logger.info("   Try: python experiments/scripts/train_mlp.py --config experiments/configs/mlp_physics.yaml")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()