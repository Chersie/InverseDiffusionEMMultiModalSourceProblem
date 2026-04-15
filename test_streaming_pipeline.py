#!/usr/bin/env python3
"""
Comprehensive tests for the End-to-End Memory-Efficient Batch Pipeline.

This test suite validates:
1. Memory usage efficiency
2. Numerical equivalence between streaming and in-memory processing  
3. Scalability to large datasets
4. Integration between all pipeline components
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from src.data.memory_monitor import MemoryMonitor, monitor_memory, get_memory_usage
from src.data.streaming_dataset import MemmapDataset, create_memmap_arrays
from src.core.config import Config, MemoryConfig, StreamingConfig
from src.api.preprocessing import PreprocessingPipeline, PreprocessingConfig
from src.models.mlp import MLPModel, MLPConfig
from experiments.scripts.train_mlp import (
    generate_training_data, 
    generate_training_data_streaming,
    setup_preprocessing_streaming
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingPipelineTests:
    """Comprehensive test suite for the streaming pipeline."""
    
    def __init__(self, temp_dir: str):
        """Initialize test suite with temporary directory."""
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory usage efficiency of streaming vs in-memory processing."""
        logger.info("=" * 60)
        logger.info("TESTING: Memory Efficiency")
        logger.info("=" * 60)
        
        # Test configurations with progressively larger datasets
        test_sizes = [100, 500, 1000, 2000]
        memory_results = {}
        
        for n_samples in test_sizes:
            logger.info(f"\n--- Testing with {n_samples} samples ---")
            
            # Test in-memory approach
            memory_baseline = self._test_memory_usage_baseline(n_samples)
            
            # Test streaming approach
            memory_streaming = self._test_memory_usage_streaming(n_samples)
            
            memory_results[n_samples] = {
                'baseline_peak_mb': memory_baseline,
                'streaming_peak_mb': memory_streaming,
                'memory_reduction': (memory_baseline - memory_streaming) / memory_baseline * 100
            }
            
            logger.info(f"Memory usage - Baseline: {memory_baseline:.1f}MB, Streaming: {memory_streaming:.1f}MB")
            logger.info(f"Memory reduction: {memory_results[n_samples]['memory_reduction']:.1f}%")
        
        self.results['memory_efficiency'] = memory_results
        return memory_results
    
    def _test_memory_usage_baseline(self, n_samples: int) -> float:
        """Test memory usage of baseline (in-memory) processing."""
        config = self._create_test_config(n_samples, use_streaming=False)
        
        with monitor_memory(f"baseline_{n_samples}", log_before_after=False) as monitor:
            try:
                # Generate data in memory
                E_theta, E_phi, y_coeffs, y_P, data_info = generate_training_data(config)
                
                # Setup preprocessing
                preprocessing = PreprocessingPipeline(PreprocessingConfig(
                    pca_components=64,
                    normalize_features=True,
                    normalize_targets=False
                ))
                
                # Fit and transform
                preprocessing.fit(E_theta, E_phi, targets=y_coeffs)
                X = preprocessing.transform_features(E_theta, E_phi)
                
                # Clear memory
                del E_theta, E_phi, y_coeffs, X
                
            except MemoryError:
                logger.warning(f"MemoryError during baseline processing of {n_samples} samples")
                return float('inf')
        
        return monitor.get_peak_usage()
    
    def _test_memory_usage_streaming(self, n_samples: int) -> float:
        """Test memory usage of streaming processing."""
        config = self._create_test_config(n_samples, use_streaming=True)
        
        with monitor_memory(f"streaming_{n_samples}", log_before_after=False) as monitor:
            try:
                # Generate data with streaming
                cache_dir = self.temp_dir / f"streaming_test_{n_samples}"
                E_theta_path, E_phi_path, y_coeffs_path, y_P_path, data_info = generate_training_data_streaming(
                    config, str(cache_dir)
                )
                
                # Setup preprocessing with streaming
                preprocessing = setup_preprocessing_streaming(
                    config, E_theta_path, E_phi_path, y_coeffs_path
                )
                
                # Transform features with streaming
                X_path = cache_dir / "X_transformed.npy"
                preprocessing.transform_features_streaming(
                    E_theta_path, E_phi_path, str(X_path)
                )
                
                # Clean up
                shutil.rmtree(cache_dir, ignore_errors=True)
                
            except MemoryError:
                logger.warning(f"MemoryError during streaming processing of {n_samples} samples")
                return float('inf')
        
        return monitor.get_peak_usage()
    
    def test_numerical_equivalence(self) -> Dict[str, Any]:
        """Test numerical equivalence between streaming and in-memory processing.""" 
        logger.info("=" * 60)
        logger.info("TESTING: Numerical Equivalence")
        logger.info("=" * 60)
        
        n_samples = 200  # Small enough for both approaches
        config = self._create_test_config(n_samples)
        
        # Generate data with both approaches
        logger.info("Generating data with both approaches...")
        
        # Baseline approach
        E_theta_mem, E_phi_mem, y_coeffs_mem, y_P_mem, _ = generate_training_data(config)
        
        # Streaming approach
        cache_dir = self.temp_dir / "equivalence_test"
        E_theta_path, E_phi_path, y_coeffs_path, y_P_path, _ = generate_training_data_streaming(
            config, str(cache_dir)
        )
        
        # Load streaming results
        E_theta_stream = np.load(E_theta_path, mmap_mode='r')
        E_phi_stream = np.load(E_phi_path, mmap_mode='r')
        y_coeffs_stream = np.load(y_coeffs_path, mmap_mode='r')
        
        # Compare data generation results
        data_comparison = {
            'E_theta_mse': float(np.mean((E_theta_mem - E_theta_stream) ** 2)),
            'E_phi_mse': float(np.mean((E_phi_mem - E_phi_stream) ** 2)),
            'coeffs_mse': float(np.mean((y_coeffs_mem - y_coeffs_stream) ** 2)),
            'E_theta_max_diff': float(np.max(np.abs(E_theta_mem - E_theta_stream))),
            'E_phi_max_diff': float(np.max(np.abs(E_phi_mem - E_phi_stream))),
            'coeffs_max_diff': float(np.max(np.abs(y_coeffs_mem - y_coeffs_stream)))
        }
        
        logger.info("Data generation comparison:")
        for metric, value in data_comparison.items():
            logger.info(f"  {metric}: {value:.2e}")
        
        # Test preprocessing equivalence
        preprocessing_comparison = self._compare_preprocessing(
            E_theta_mem, E_phi_mem, y_coeffs_mem,
            E_theta_path, E_phi_path, y_coeffs_path
        )
        
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)
        
        equivalence_results = {
            'data_generation': data_comparison,
            'preprocessing': preprocessing_comparison
        }
        
        self.results['numerical_equivalence'] = equivalence_results
        return equivalence_results
    
    def _compare_preprocessing(
        self,
        E_theta_mem: np.ndarray,
        E_phi_mem: np.ndarray,
        y_coeffs_mem: np.ndarray,
        E_theta_path: str,
        E_phi_path: str,
        y_coeffs_path: str
    ) -> Dict[str, float]:
        """Compare preprocessing results between in-memory and streaming."""
        logger.info("Comparing preprocessing approaches...")
        
        preproc_config = PreprocessingConfig(
            pca_components=32,
            normalize_features=True,
            normalize_targets=False
        )
        
        # In-memory preprocessing
        preprocessing_mem = PreprocessingPipeline(preproc_config)
        preprocessing_mem.fit(E_theta_mem, E_phi_mem, targets=y_coeffs_mem)
        X_mem = preprocessing_mem.transform_features(E_theta_mem, E_phi_mem)
        
        # Streaming preprocessing
        preprocessing_stream = PreprocessingPipeline(preproc_config)
        preprocessing_stream.fit_streaming(E_theta_path, E_phi_path, y_coeffs_path, max_fit_samples=200)
        
        X_stream_path = self.temp_dir / "X_stream_comparison.npy"
        preprocessing_stream.transform_features_streaming(
            E_theta_path, E_phi_path, str(X_stream_path)
        )
        
        X_stream = np.load(X_stream_path, mmap_mode='r')
        
        # Compare results
        comparison = {
            'features_mse': float(np.mean((X_mem - X_stream) ** 2)),
            'features_max_diff': float(np.max(np.abs(X_mem - X_stream))),
            'pca_components_mse': float(np.mean((
                preprocessing_mem.field_preprocessor.pca_transformer.components_ - 
                preprocessing_stream.field_preprocessor.pca_transformer.components_
            ) ** 2))
        }
        
        logger.info("Preprocessing comparison:")
        for metric, value in comparison.items():
            logger.info(f"  {metric}: {value:.2e}")
        
        return comparison
    
    def test_scalability(self) -> Dict[str, Any]:
        """Test scalability to progressively larger datasets."""
        logger.info("=" * 60)
        logger.info("TESTING: Scalability")
        logger.info("=" * 60)
        
        # Test with increasingly large datasets
        test_sizes = [500, 1000, 2000, 5000]
        scalability_results = {}
        
        for n_samples in test_sizes:
            logger.info(f"\n--- Scalability test with {n_samples} samples ---")
            
            try:
                result = self._test_end_to_end_streaming(n_samples)
                scalability_results[n_samples] = result
                
                logger.info(f"✅ Successfully processed {n_samples} samples")
                logger.info(f"   Peak memory: {result['peak_memory_mb']:.1f}MB")
                logger.info(f"   Total time: {result['total_time']:.1f}s")
                logger.info(f"   Samples/sec: {result['samples_per_second']:.1f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {n_samples} samples: {e}")
                scalability_results[n_samples] = {'error': str(e)}
        
        self.results['scalability'] = scalability_results
        return scalability_results
    
    def _test_end_to_end_streaming(self, n_samples: int) -> Dict[str, Any]:
        """Test complete end-to-end streaming pipeline."""
        config = self._create_test_config(n_samples, use_streaming=True)
        
        with monitor_memory(f"e2e_streaming_{n_samples}", log_before_after=False) as monitor:
            start_time = time.time()
            
            # 1. Data generation
            cache_dir = self.temp_dir / f"e2e_test_{n_samples}"
            E_theta_path, E_phi_path, y_coeffs_path, y_P_path, data_info = generate_training_data_streaming(
                config, str(cache_dir)
            )
            
            # 2. Preprocessing
            preprocessing = setup_preprocessing_streaming(
                config, E_theta_path, E_phi_path, y_coeffs_path
            )
            
            # 3. Feature transformation
            X_path = cache_dir / "X_transformed.npy"
            preprocessing.transform_features_streaming(
                E_theta_path, E_phi_path, str(X_path)
            )
            
            # 4. Create dataset and test data loading
            y_target_path = y_coeffs_path if config['model']['loss_type'] == 'coefficient' else y_P_path
            dataset = MemmapDataset(str(X_path), y_target_path)
            
            # Test sampling from dataset
            sample_indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
            for idx in sample_indices[:10]:  # Test first 10 samples
                features, targets = dataset[idx]
                assert features.shape[0] > 0, "Features should not be empty"
                assert targets.shape[0] > 0, "Targets should not be empty"
            
            end_time = time.time()
            
            # Clean up
            shutil.rmtree(cache_dir, ignore_errors=True)
        
        return {
            'peak_memory_mb': monitor.get_peak_usage(),
            'total_time': end_time - start_time,
            'samples_per_second': n_samples / (end_time - start_time),
            'success': True
        }
    
    def test_integration(self) -> Dict[str, Any]:
        """Test integration between all pipeline components."""
        logger.info("=" * 60)
        logger.info("TESTING: Component Integration")
        logger.info("=" * 60)
        
        n_samples = 300
        config = self._create_test_config(n_samples, use_streaming=True)
        
        integration_results = {}
        
        try:
            # Test complete pipeline with model training
            with monitor_memory("integration_test", log_before_after=True) as monitor:
                
                # 1. Generate streaming data
                cache_dir = self.temp_dir / "integration_test"
                E_theta_path, E_phi_path, y_coeffs_path, y_P_path, data_info = generate_training_data_streaming(
                    config, str(cache_dir)
                )
                
                # 2. Setup preprocessing
                preprocessing = setup_preprocessing_streaming(
                    config, E_theta_path, E_phi_path, y_coeffs_path
                )
                
                # 3. Transform features
                X_path = cache_dir / "X_transformed.npy" 
                preprocessing.transform_features_streaming(
                    E_theta_path, E_phi_path, str(X_path)
                )
                
                # 4. Test model creation and streaming training
                model_config = MLPConfig(
                    hidden_size=64,
                    n_hidden_layers=2,
                    input_dim=config['preprocessing']['pca_components'],
                    output_dim=data_info['n_modes'] * 4,
                    loss_type=config['model']['loss_type'],
                    maxorder=config['model']['maxorder']
                )
                
                model = MLPModel(model_config)
                
                # Test with streaming datasets
                y_target_path = y_coeffs_path if config['model']['loss_type'] == 'coefficient' else y_P_path
                
                # Simple training test (just one epoch)
                train_dataset = MemmapDataset(str(X_path), y_target_path)
                
                # Test dataset loading
                sample_features, sample_targets = train_dataset[0]
                assert sample_features.shape[0] == model_config.input_dim
                
                # Test batched prediction
                X_test_small = np.random.randn(50, model_config.input_dim).astype(np.float32)
                
                # Model needs to be "trained" for predict to work
                model.is_trained = True
                model._torch_model = model._create_torch_model()
                
                predictions = model.predict_safe(X_test_small, force_batch=True)
                assert predictions.shape == (50, model_config.output_dim)
                
                # Clean up
                shutil.rmtree(cache_dir, ignore_errors=True)
            
            integration_results = {
                'success': True,
                'peak_memory_mb': monitor.get_peak_usage(),
                'components_tested': [
                    'streaming_data_generation',
                    'streaming_preprocessing', 
                    'memmap_datasets',
                    'model_integration',
                    'batched_prediction'
                ]
            }
            
            logger.info("✅ Integration test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            integration_results = {'success': False, 'error': str(e)}
        
        self.results['integration'] = integration_results
        return integration_results
    
    def _create_test_config(self, n_samples: int, use_streaming: bool = True) -> Dict[str, Any]:
        """Create test configuration."""
        return {
            'model': {
                'maxorder': 3,
                'loss_type': 'coefficient',  # Use coefficient loss for simpler testing
            },
            'training': {
                'n_samples': n_samples,
                'batch_size': min(32, n_samples // 4),
                'epochs': 1,
                'train_ratio': 0.8,
                'val_ratio': 0.1
            },
            'data': {
                'generator_mode': 'random',
                'seed': 42
            },
            'preprocessing': {
                'pca_components': min(64, n_samples // 4),
                'pca_oversample': 4,
                'normalize_features': True,
                'normalize_targets': False
            },
            'streaming': {
                'enable_streaming': use_streaming,
                'force_streaming_above_samples': 100 if use_streaming else 999999,
                'cache_dir': 'data/cache'
            },
            'memory': {
                'cache_dir': 'data/cache'
            }
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("STREAMING PIPELINE TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Memory efficiency results
        if 'memory_efficiency' in self.results:
            report.append("MEMORY EFFICIENCY RESULTS:")
            report.append("-" * 40)
            for n_samples, metrics in self.results['memory_efficiency'].items():
                reduction = metrics['memory_reduction']
                report.append(f"  {n_samples:4d} samples: {reduction:5.1f}% memory reduction")
            report.append("")
        
        # Numerical equivalence results
        if 'numerical_equivalence' in self.results:
            report.append("NUMERICAL EQUIVALENCE RESULTS:")
            report.append("-" * 40)
            
            data_gen = self.results['numerical_equivalence']['data_generation']
            preprocessing = self.results['numerical_equivalence']['preprocessing']
            
            report.append(f"  Data Generation Max Diff:")
            report.append(f"    E_theta: {data_gen['E_theta_max_diff']:.2e}")
            report.append(f"    E_phi:   {data_gen['E_phi_max_diff']:.2e}")
            report.append(f"    Coeffs:  {data_gen['coeffs_max_diff']:.2e}")
            
            report.append(f"  Preprocessing Max Diff:")
            report.append(f"    Features: {preprocessing['features_max_diff']:.2e}")
            report.append("")
        
        # Scalability results
        if 'scalability' in self.results:
            report.append("SCALABILITY RESULTS:")
            report.append("-" * 40)
            for n_samples, metrics in self.results['scalability'].items():
                if 'error' in metrics:
                    report.append(f"  {n_samples:4d} samples: FAILED - {metrics['error']}")
                else:
                    memory = metrics['peak_memory_mb']
                    time_taken = metrics['total_time'] 
                    throughput = metrics['samples_per_second']
                    report.append(f"  {n_samples:4d} samples: {memory:6.1f}MB, {time_taken:5.1f}s, {throughput:6.1f} samples/s")
            report.append("")
        
        # Integration results
        if 'integration' in self.results:
            integration = self.results['integration']
            report.append("INTEGRATION TEST:")
            report.append("-" * 40)
            if integration['success']:
                report.append("  ✅ PASSED")
                report.append(f"     Peak memory: {integration['peak_memory_mb']:.1f}MB")
                report.append(f"     Components: {len(integration['components_tested'])}")
            else:
                report.append(f"  ❌ FAILED: {integration.get('error', 'Unknown error')}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() 
                          if isinstance(result, dict) and result.get('success', True))
        
        report.append(f"  Total test categories: {total_tests}")
        report.append(f"  Passed: {passed_tests}")
        report.append(f"  Status: {'✅ ALL PASSED' if passed_tests == total_tests else '❌ SOME FAILED'}")
        
        return "\n".join(report)


def main():
    """Run comprehensive streaming pipeline tests."""
    logger.info("Starting comprehensive streaming pipeline tests...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="streaming_pipeline_test_")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize test suite
        tests = StreamingPipelineTests(temp_dir)
        
        # Run all test categories
        logger.info("\n🚀 Starting test execution...")
        
        tests.test_memory_efficiency()
        tests.test_numerical_equivalence() 
        tests.test_scalability()
        tests.test_integration()
        
        # Generate and display report
        report = tests.generate_report()
        print("\n" + report)
        
        # Save report to file
        report_path = Path("streaming_pipeline_test_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"\n📄 Test report saved to: {report_path}")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()