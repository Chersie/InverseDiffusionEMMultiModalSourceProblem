"""
Memory monitoring utilities for the streaming pipeline.

This module provides tools to monitor memory usage during training and 
processing to prevent out-of-memory errors and enable adaptive processing.
"""

import psutil
import os
import time
import logging
from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    
    timestamp: float
    process_memory_mb: float
    system_memory_mb: float
    system_memory_percent: float
    available_memory_mb: float
    pid: int
    
    def __str__(self) -> str:
        return (
            f"MemorySnapshot(process={self.process_memory_mb:.1f}MB, "
            f"system={self.system_memory_percent:.1f}%, "
            f"available={self.available_memory_mb:.1f}MB)"
        )


class MemoryMonitor:
    """
    Monitor and track memory usage during processing.
    
    Can be used to:
    - Track peak memory usage
    - Detect memory leaks
    - Trigger adaptive behavior based on memory pressure
    - Log memory usage patterns
    """
    
    def __init__(
        self,
        warning_threshold_mb: Optional[float] = None,
        critical_threshold_mb: Optional[float] = None,
        log_interval_seconds: float = 30.0
    ):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold_mb: Warn when process memory exceeds this
            critical_threshold_mb: Critical alert when process memory exceeds this
            log_interval_seconds: How often to log memory usage
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.log_interval_seconds = log_interval_seconds
        
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[MemorySnapshot] = []
        self.peak_memory_mb = 0.0
        self.last_log_time = 0.0
        
        # Take initial snapshot
        self.take_snapshot()
        
        logger.info(f"Initialized MemoryMonitor (PID: {os.getpid()})")
        if warning_threshold_mb:
            logger.info(f"Warning threshold: {warning_threshold_mb}MB")
        if critical_threshold_mb:
            logger.info(f"Critical threshold: {critical_threshold_mb}MB")
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        try:
            # Process memory
            process_info = self.process.memory_info()
            process_memory_mb = process_info.rss / 1024 / 1024
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_memory_mb = system_memory.total / 1024 / 1024
            system_memory_percent = system_memory.percent
            available_memory_mb = system_memory.available / 1024 / 1024
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                process_memory_mb=process_memory_mb,
                system_memory_mb=system_memory_mb,
                system_memory_percent=system_memory_percent,
                available_memory_mb=available_memory_mb,
                pid=self.process.pid
            )
            
            self.snapshots.append(snapshot)
            
            # Update peak memory
            if process_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = process_memory_mb
            
            # Check thresholds and log if needed
            self._check_thresholds(snapshot)
            self._maybe_log(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            # Return empty snapshot as fallback
            return MemorySnapshot(
                timestamp=time.time(), process_memory_mb=0.0,
                system_memory_mb=0.0, system_memory_percent=0.0,
                available_memory_mb=0.0, pid=os.getpid()
            )
    
    def _check_thresholds(self, snapshot: MemorySnapshot):
        """Check if memory usage exceeds thresholds."""
        if (self.critical_threshold_mb and 
            snapshot.process_memory_mb > self.critical_threshold_mb):
            logger.critical(
                f"CRITICAL: Memory usage {snapshot.process_memory_mb:.1f}MB "
                f"exceeds threshold {self.critical_threshold_mb}MB"
            )
        elif (self.warning_threshold_mb and 
              snapshot.process_memory_mb > self.warning_threshold_mb):
            logger.warning(
                f"WARNING: Memory usage {snapshot.process_memory_mb:.1f}MB "
                f"exceeds threshold {self.warning_threshold_mb}MB"
            )
    
    def _maybe_log(self, snapshot: MemorySnapshot):
        """Log memory usage if enough time has passed."""
        if (time.time() - self.last_log_time) >= self.log_interval_seconds:
            logger.info(f"Memory usage: {snapshot}")
            self.last_log_time = time.time()
    
    def get_current_usage(self) -> float:
        """Get current process memory usage in MB."""
        snapshot = self.take_snapshot()
        return snapshot.process_memory_mb
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage seen so far."""
        return self.peak_memory_mb
    
    def get_available_memory(self) -> float:
        """Get available system memory in MB."""
        snapshot = self.take_snapshot()
        return snapshot.available_memory_mb
    
    def is_memory_pressure(self, threshold_percent: float = 85.0) -> bool:
        """
        Check if system is under memory pressure.
        
        Args:
            threshold_percent: Consider pressure if system memory > this %
            
        Returns:
            True if system memory usage exceeds threshold
        """
        snapshot = self.take_snapshot()
        return snapshot.system_memory_percent > threshold_percent
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        snapshot = self.take_snapshot()
        
        return {
            'current_process_mb': snapshot.process_memory_mb,
            'peak_process_mb': self.peak_memory_mb,
            'system_total_mb': snapshot.system_memory_mb,
            'system_used_percent': snapshot.system_memory_percent,
            'system_available_mb': snapshot.available_memory_mb,
            'num_snapshots': len(self.snapshots)
        }
    
    def reset_peak(self):
        """Reset peak memory tracking."""
        self.peak_memory_mb = self.get_current_usage()
        logger.info(f"Reset peak memory to {self.peak_memory_mb:.1f}MB")
    
    def clear_snapshots(self):
        """Clear stored snapshots to save memory."""
        self.snapshots.clear()
        logger.info("Cleared memory snapshots")


# Global memory monitor instance
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_usage() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Current memory usage in megabytes
    """
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return 0.0


def get_available_memory() -> float:
    """
    Get available system memory in MB.
    
    Returns:
        Available system memory in megabytes  
    """
    try:
        return psutil.virtual_memory().available / 1024 / 1024
    except Exception as e:
        logger.error(f"Failed to get available memory: {e}")
        return 0.0


def initialize_global_monitor(
    warning_threshold_mb: Optional[float] = None,
    critical_threshold_mb: Optional[float] = None,
    log_interval_seconds: float = 30.0
) -> MemoryMonitor:
    """
    Initialize global memory monitor.
    
    Args:
        warning_threshold_mb: Warning threshold in MB
        critical_threshold_mb: Critical threshold in MB
        log_interval_seconds: Logging interval
        
    Returns:
        Global MemoryMonitor instance
    """
    global _global_monitor
    _global_monitor = MemoryMonitor(
        warning_threshold_mb=warning_threshold_mb,
        critical_threshold_mb=critical_threshold_mb,
        log_interval_seconds=log_interval_seconds
    )
    return _global_monitor


def get_global_monitor() -> Optional[MemoryMonitor]:
    """Get the global memory monitor instance."""
    return _global_monitor


@contextmanager
def monitor_memory(
    operation_name: str = "operation",
    log_before_after: bool = True,
    warning_threshold_mb: Optional[float] = None
):
    """
    Context manager to monitor memory usage during an operation.
    
    Args:
        operation_name: Name of operation for logging
        log_before_after: Whether to log before/after memory usage
        warning_threshold_mb: Optional warning threshold for this operation
        
    Yields:
        MemoryMonitor instance for the operation
    """
    monitor = MemoryMonitor(warning_threshold_mb=warning_threshold_mb)
    
    if log_before_after:
        start_memory = monitor.get_current_usage()
        logger.info(f"Starting {operation_name} - Memory: {start_memory:.1f}MB")
    
    try:
        yield monitor
    finally:
        if log_before_after:
            end_memory = monitor.get_current_usage()
            peak_memory = monitor.get_peak_usage()
            logger.info(
                f"Finished {operation_name} - "
                f"Memory: {end_memory:.1f}MB, "
                f"Peak: {peak_memory:.1f}MB, "
                f"Delta: {end_memory - start_memory:+.1f}MB"
            )


def suggest_batch_size(
    base_batch_size: int,
    available_memory_mb: float,
    sample_memory_mb: float,
    safety_factor: float = 0.8
) -> int:
    """
    Suggest an appropriate batch size based on available memory.
    
    Args:
        base_batch_size: Preferred batch size
        available_memory_mb: Available memory in MB
        sample_memory_mb: Estimated memory per sample in MB
        safety_factor: Safety factor (0-1) to avoid OOM
        
    Returns:
        Suggested batch size
    """
    max_samples = int((available_memory_mb * safety_factor) / sample_memory_mb)
    suggested_batch_size = min(base_batch_size, max_samples)
    
    if suggested_batch_size < base_batch_size:
        logger.warning(
            f"Reduced batch size from {base_batch_size} to {suggested_batch_size} "
            f"due to memory constraints (available: {available_memory_mb:.1f}MB, "
            f"per sample: {sample_memory_mb:.2f}MB)"
        )
    
    return max(1, suggested_batch_size)  # Ensure at least batch size 1


def estimate_memory_per_sample(
    feature_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    overhead_factor: float = 2.0
) -> float:
    """
    Estimate memory usage per sample.
    
    Args:
        feature_shape: Shape of feature tensor (excluding batch dimension)
        target_shape: Shape of target tensor (excluding batch dimension)  
        dtype: Data type
        overhead_factor: Factor for gradient/activation overhead
        
    Returns:
        Estimated memory per sample in MB
    """
    bytes_per_element = np.dtype(dtype).itemsize
    
    feature_elements = np.prod(feature_shape)
    target_elements = np.prod(target_shape)
    total_elements = feature_elements + target_elements
    
    base_memory_mb = (total_elements * bytes_per_element) / (1024 * 1024)
    estimated_memory_mb = base_memory_mb * overhead_factor
    
    return estimated_memory_mb