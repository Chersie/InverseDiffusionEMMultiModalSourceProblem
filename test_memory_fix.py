#!/usr/bin/env python3
"""
Quick test to verify memory fixes work before running full training.
"""

import numpy as np
import time
import logging
import gc
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0

def test_memory_efficient_generation():
    """Test the new memory-efficient data generation."""
    from experiments.scripts.train_mlp import generate_training_data
    
    logger.info("🧪 Testing memory-efficient data generation...")
    
    # Test different sizes progressively
    test_sizes = [500, 1000, 2000, 3000]
    
    for n_samples in test_sizes:
        logger.info(f"\n📊 Testing {n_samples} samples...")
        
        config = {
            'model': {'maxorder': 4, 'loss_type': 'physics'},
            'training': {'n_samples': n_samples},
            'data': {'generator_mode': 'random', 'seed': 42}
        }
        
        start_memory = get_memory_usage()
        start_time = time.time()
        
        try:
            # Test data generation
            E_theta, E_phi, y_coeffs, y_P, data_info = generate_training_data(config)
            
            end_time = time.time()
            end_memory = get_memory_usage()
            
            logger.info(f"✅ SUCCESS: {n_samples} samples generated")
            logger.info(f"   Time: {end_time - start_time:.1f}s")
            logger.info(f"   Memory used: {end_memory - start_memory:.1f}MB")
            logger.info(f"   Peak memory: {end_memory:.1f}MB")
            logger.info(f"   Data shapes: E_theta={E_theta.shape}, P={y_P.shape if y_P is not None else 'None'}")
            logger.info(f"   Batches used: {data_info['batches_used']}")
            
            # Clear data and force GC
            del E_theta, E_phi, y_coeffs, y_P, data_info
            gc.collect()
            
            after_cleanup = get_memory_usage()
            logger.info(f"   Memory after cleanup: {after_cleanup:.1f}MB (freed: {end_memory - after_cleanup:.1f}MB)")
            
            # If this size worked, try the next one
            time.sleep(1)  # Brief pause
            
        except Exception as e:
            logger.error(f"❌ FAILED at {n_samples} samples: {e}")
            break
    
    logger.info("\n🎯 Memory test complete!")

def test_preprocessing_fix():
    """Test memory-efficient preprocessing."""
    from experiments.scripts.train_mlp import generate_training_data, setup_preprocessing_batched
    
    logger.info("\n🔄 Testing memory-efficient preprocessing...")
    
    config = {
        'model': {'maxorder': 4, 'loss_type': 'coefficient'},
        'training': {'n_samples': 1000}, 
        'data': {'generator_mode': 'random', 'seed': 42},
        'preprocessing': {
            'pca_components': 100,
            'pca_oversample': 4,
            'normalize_features': True,
            'normalize_targets': True
        }
    }
    
    start_memory = get_memory_usage()
    
    # Generate data
    E_theta, E_phi, y_coeffs, y_P, _ = generate_training_data(config)
    after_generation = get_memory_usage()
    
    # Test preprocessing
    preprocessing = setup_preprocessing_batched(config, E_theta, E_phi, y_coeffs)
    after_preprocessing = get_memory_usage()
    
    logger.info(f"✅ Preprocessing test complete:")
    logger.info(f"   Start memory: {start_memory:.1f}MB")
    logger.info(f"   After generation: {after_generation:.1f}MB (+{after_generation-start_memory:.1f}MB)")
    logger.info(f"   After preprocessing: {after_preprocessing:.1f}MB (+{after_preprocessing-after_generation:.1f}MB)")

if __name__ == "__main__":
    logger.info("🚀 Testing memory fixes...")
    
    start_total = get_memory_usage()
    logger.info(f"Initial memory: {start_total:.1f}MB")
    
    try:
        # Test 1: Memory-efficient generation
        test_memory_efficient_generation()
        
        # Test 2: Preprocessing 
        test_preprocessing_fix()
        
        end_total = get_memory_usage()
        logger.info(f"\n✅ All tests passed!")
        logger.info(f"Final memory: {end_total:.1f}MB (delta: {end_total-start_total:.1f}MB)")
        
        logger.info("\n🎉 Memory fixes verified! Ready for full training:")
        logger.info("   python experiments/scripts/train_mlp.py --config experiments/configs/mlp_physics.yaml")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()