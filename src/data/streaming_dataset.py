"""
Memory-mapped dataset classes for streaming data processing.

This module provides PyTorch-compatible dataset classes that can efficiently
work with large datasets stored as memory-mapped numpy arrays on disk.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MemmapDataset(Dataset):
    """
    PyTorch Dataset that reads from memory-mapped numpy arrays.
    
    This allows processing arbitrarily large datasets without loading
    them entirely into memory. The dataset reads from .npy files that
    are memory-mapped for efficient access.
    """
    
    def __init__(
        self,
        features_path: Union[str, Path],
        targets_path: Union[str, Path], 
        mmap_mode: str = 'r',
        dtype: torch.dtype = torch.float32,
        transform_features: Optional[callable] = None,
        transform_targets: Optional[callable] = None
    ):
        """
        Initialize memory-mapped dataset.
        
        Args:
            features_path: Path to memory-mapped features file (.npy)
            targets_path: Path to memory-mapped targets file (.npy)
            mmap_mode: Memory mapping mode ('r', 'r+', 'w+', 'c')
            dtype: PyTorch dtype for tensors
            transform_features: Optional transform for features
            transform_targets: Optional transform for targets
        """
        self.features_path = Path(features_path)
        self.targets_path = Path(targets_path)
        self.mmap_mode = mmap_mode
        self.dtype = dtype
        self.transform_features = transform_features
        self.transform_targets = transform_targets
        
        # Verify files exist
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        if not self.targets_path.exists():
            raise FileNotFoundError(f"Targets file not found: {self.targets_path}")
        
        # Load memory-mapped arrays
        self._load_arrays()
        
        logger.info(f"Initialized MemmapDataset with {len(self)} samples")
        logger.info(f"Features shape: {self.features.shape}, dtype: {self.features.dtype}")
        logger.info(f"Targets shape: {self.targets.shape}, dtype: {self.targets.dtype}")
    
    def _load_arrays(self):
        """Load memory-mapped arrays from disk."""
        try:
            self.features = np.load(self.features_path, mmap_mode=self.mmap_mode)
            self.targets = np.load(self.targets_path, mmap_mode=self.mmap_mode)
            
            # Verify compatible shapes
            if self.features.shape[0] != self.targets.shape[0]:
                raise ValueError(
                    f"Mismatch in number of samples: "
                    f"features={self.features.shape[0]}, targets={self.targets.shape[0]}"
                )
                
        except Exception as e:
            logger.error(f"Failed to load memory-mapped arrays: {e}")
            raise
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.features.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, targets) as PyTorch tensors
        """
        # Get data from memory-mapped arrays (efficient, only loads needed data)
        features = self.features[idx].copy()  # Copy to avoid memory-mapping issues
        targets = self.targets[idx].copy()
        
        # Apply transforms if provided
        if self.transform_features is not None:
            features = self.transform_features(features)
        if self.transform_targets is not None:
            targets = self.transform_targets(targets)
        
        # Convert to PyTorch tensors
        features_tensor = torch.from_numpy(features).to(self.dtype)
        targets_tensor = torch.from_numpy(targets).to(self.dtype)
        
        return features_tensor, targets_tensor
    
    def get_batch(self, indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently get a batch of samples.
        
        Args:
            indices: Array of sample indices
            
        Returns:
            Tuple of (batch_features, batch_targets) as PyTorch tensors
        """
        # Use fancy indexing to get batch (still memory-efficient with mmap)
        batch_features = self.features[indices].copy()
        batch_targets = self.targets[indices].copy()
        
        # Apply transforms if provided
        if self.transform_features is not None:
            batch_features = self.transform_features(batch_features)
        if self.transform_targets is not None:
            batch_targets = self.transform_targets(batch_targets)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(batch_features).to(self.dtype)
        targets_tensor = torch.from_numpy(batch_targets).to(self.dtype)
        
        return features_tensor, targets_tensor
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'n_samples': len(self),
            'features_shape': self.features.shape,
            'targets_shape': self.targets.shape,
            'features_dtype': str(self.features.dtype),
            'targets_dtype': str(self.targets.dtype),
            'features_path': str(self.features_path),
            'targets_path': str(self.targets_path),
            'mmap_mode': self.mmap_mode
        }


class StreamingDataLoader:
    """
    Custom DataLoader optimized for memory-mapped datasets.
    
    Provides additional memory monitoring and adaptive batch sizing
    capabilities beyond standard PyTorch DataLoader.
    """
    
    def __init__(
        self,
        dataset: MemmapDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        memory_limit_gb: Optional[float] = None,
        adaptive_batching: bool = False
    ):
        """
        Initialize streaming data loader.
        
        Args:
            dataset: MemmapDataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop incomplete last batch
            memory_limit_gb: Optional memory limit for adaptive batching
            adaptive_batching: Whether to use adaptive batch sizing
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.memory_limit_gb = memory_limit_gb
        self.adaptive_batching = adaptive_batching
        
        # Create standard PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        logger.info(f"Created StreamingDataLoader with batch_size={batch_size}")
        if adaptive_batching:
            logger.info(f"Adaptive batching enabled with memory limit: {memory_limit_gb}GB")
    
    def __iter__(self):
        """Iterate over batches."""
        if self.adaptive_batching:
            return self._adaptive_iter()
        else:
            return iter(self.dataloader)
    
    def __len__(self):
        """Return number of batches.""" 
        return len(self.dataloader)
    
    def _adaptive_iter(self):
        """
        Adaptive iteration with memory monitoring.
        
        Adjusts batch size based on memory usage to prevent OOM.
        """
        from .memory_monitor import get_memory_usage
        
        current_batch_size = self.batch_size
        
        for batch_idx, (features, targets) in enumerate(self.dataloader):
            # Monitor memory before yielding batch
            if self.memory_limit_gb is not None:
                current_memory = get_memory_usage()
                
                if current_memory > self.memory_limit_gb * 1024:  # Convert GB to MB
                    # Memory usage too high, reduce batch size for next iteration
                    current_batch_size = max(1, int(current_batch_size * 0.8))
                    logger.warning(
                        f"High memory usage ({current_memory:.1f}MB), "
                        f"reducing batch size to {current_batch_size}"
                    )
                elif current_memory < self.memory_limit_gb * 1024 * 0.5:
                    # Memory usage low, can increase batch size
                    current_batch_size = min(self.batch_size, int(current_batch_size * 1.2))
            
            yield features, targets


def create_memmap_arrays(
    features_shape: Tuple[int, ...],
    targets_shape: Tuple[int, ...],
    features_path: Union[str, Path],
    targets_path: Union[str, Path],
    dtype: np.dtype = np.float32
) -> Tuple[np.memmap, np.memmap]:
    """
    Create memory-mapped arrays for efficient data storage.
    
    Args:
        features_shape: Shape of features array
        targets_shape: Shape of targets array
        features_path: Path for features memory-mapped file
        targets_path: Path for targets memory-mapped file
        dtype: NumPy data type
        
    Returns:
        Tuple of (features_memmap, targets_memmap)
    """
    features_path = Path(features_path)
    targets_path = Path(targets_path)
    
    # Create parent directories if needed
    features_path.parent.mkdir(parents=True, exist_ok=True)
    targets_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create memory-mapped arrays
    features_mm = np.lib.format.open_memmap(
        str(features_path), mode='w+', dtype=dtype, shape=features_shape
    )
    targets_mm = np.lib.format.open_memmap(
        str(targets_path), mode='w+', dtype=dtype, shape=targets_shape
    )
    
    logger.info(f"Created memory-mapped arrays:")
    logger.info(f"  Features: {features_shape} -> {features_path}")
    logger.info(f"  Targets: {targets_shape} -> {targets_path}")
    
    return features_mm, targets_mm


def load_memmap_dataset(
    features_path: Union[str, Path],
    targets_path: Union[str, Path],
    **kwargs
) -> MemmapDataset:
    """
    Convenience function to load a memory-mapped dataset.
    
    Args:
        features_path: Path to features .npy file
        targets_path: Path to targets .npy file  
        **kwargs: Additional arguments for MemmapDataset
        
    Returns:
        MemmapDataset instance
    """
    return MemmapDataset(features_path, targets_path, **kwargs)