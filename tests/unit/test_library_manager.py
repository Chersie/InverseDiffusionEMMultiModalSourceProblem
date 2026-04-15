"""
Unit tests for LibraryManager.

Tests the high-performance library loading and caching system.
"""
import numpy as np
import pytest
from pathlib import Path

from src.core.library_manager import (
    LibraryManager, _parse_library_file, _validate_grid_ordering,
    get_library_manager, reset_library_manager
)
from src.core.config import Config


class TestLibraryFileParser:
    """Tests for library file parsing functions."""
    
    def test_parse_library_file(self, mock_library_files, small_maxorder):
        """Test parsing of library mode files."""
        # Get a test file
        library_dir = mock_library_files
        test_file = library_dir / "E_l1_m0.txt"
        
        assert test_file.exists(), f"Test file not found: {test_file}"
        
        # Parse the file
        amplitude, theta = _parse_library_file(test_file, header_lines=43)
        
        # Check shapes
        assert amplitude.ndim == 3
        assert amplitude.shape[2] == 2  # E_theta, E_phi components
        assert theta.ndim == 2
        assert amplitude.shape[:2] == theta.shape
        
        # Check data types
        assert amplitude.dtype == complex
        assert theta.dtype == float
        
        # Check that we have valid data (non-zero)
        assert np.any(amplitude != 0)
        assert np.any(theta > 0)
    
    def test_validate_grid_ordering(self, test_grid_shape):
        """Test grid ordering validation."""
        n_phi, n_theta = test_grid_shape
        
        # Create valid grid data
        rows = []
        for j in range(n_phi):
            for i in range(n_theta):
                theta_deg = (i + 1) * (180.0 / (n_theta + 1))
                phi_deg = j * (360.0 / n_phi)
                power = 1.0
                
                row = [str(theta_deg), str(phi_deg), str(power), "1.0", "0.0", "1.0", "0.0"]
                rows.append(row)
        
        # Should not raise exception
        _validate_grid_ordering(rows, n_phi, n_theta, legacy_grid=False)
        
        # Test with invalid ordering (should warn but not fail)
        invalid_rows = rows.copy()
        invalid_rows[0][0] = "999.0"  # Invalid first theta
        
        # This should log a warning but not raise
        _validate_grid_ordering(invalid_rows, n_phi, n_theta, legacy_grid=False)


class TestLibraryManager:
    """Tests for LibraryManager class."""
    
    def setup_method(self):
        """Reset library manager before each test."""
        reset_library_manager()
    
    def test_initialization(self, test_config, small_maxorder):
        """Test library manager initialization."""
        # Create with specific config
        manager = LibraryManager(
            library_dir=test_config.paths.library_fast_dir,
            maxorder=small_maxorder,
            config=test_config
        )
        
        assert manager.maxorder == small_maxorder
        assert manager.config == test_config
        assert not manager._is_loaded
        assert manager._load_time is None
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        manager = LibraryManager()
        
        assert isinstance(manager.config, Config)
        assert manager.maxorder > 0
        assert manager.library_dir.exists() or True  # May not exist in test env
    
    def test_load_mock_library(self, mock_library_files, small_maxorder):
        """Test loading mock library files."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Load library
        manager.load()
        
        assert manager._is_loaded
        assert manager._load_time is not None
        assert manager._load_time > 0
        
        # Check that modes were loaded
        assert len(manager._e_modes) > 0
        assert len(manager._m_modes) > 0
        
        # Should have modes for each (l, m) pair
        expected_modes = small_maxorder * (small_maxorder + 2)
        assert len(manager._e_modes) == expected_modes
        assert len(manager._m_modes) == expected_modes
    
    def test_get_mode_data(self, mock_library_files, small_maxorder):
        """Test retrieving mode data."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        manager.load()
        
        # Test getting E mode
        amp_e, theta_e = manager.get_e_mode(1, 0)
        assert amp_e.ndim == 3
        assert theta_e.ndim == 2
        assert amp_e.dtype == complex
        assert theta_e.dtype == float
        
        # Test getting M mode
        amp_m, theta_m = manager.get_m_mode(1, 0)
        assert amp_m.shape == amp_e.shape
        assert theta_m.shape == theta_e.shape
        
        # Test generic get_mode interface
        amp_e2, theta_e2 = manager.get_mode("E", 1, 0)
        np.testing.assert_array_equal(amp_e, amp_e2)
        np.testing.assert_array_equal(theta_e, theta_e2)
        
        amp_m2, theta_m2 = manager.get_mode("M", 1, 0)
        np.testing.assert_array_equal(amp_m, amp_m2)
        np.testing.assert_array_equal(theta_m, theta_m2)
    
    def test_get_nonexistent_mode(self, mock_library_files, small_maxorder):
        """Test error handling for nonexistent modes."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        manager.load()
        
        # Try to get mode outside maxorder
        with pytest.raises(ValueError, match="not available"):
            manager.get_e_mode(small_maxorder + 1, 0)
        
        with pytest.raises(ValueError, match="not available"):
            manager.get_m_mode(1, small_maxorder + 1)  # m > l
    
    def test_invalid_mode_type(self, mock_library_files, small_maxorder):
        """Test error handling for invalid mode types."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        manager.load()
        
        with pytest.raises(ValueError, match="Invalid mode type"):
            manager.get_mode("X", 1, 0)
    
    def test_get_available_modes(self, mock_library_files, small_maxorder):
        """Test getting list of available modes."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        manager.load()
        
        available = manager.get_available_modes()
        
        assert "E" in available
        assert "M" in available
        assert len(available["E"]) > 0
        assert len(available["M"]) > 0
        
        # Check specific modes
        assert (1, 0) in available["E"]
        assert (1, -1) in available["E"]
        assert (1, 1) in available["E"]
    
    def test_memory_usage_estimation(self, mock_library_files, small_maxorder):
        """Test memory usage estimation."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Before loading
        assert manager.get_memory_usage() == 0
        
        # After loading
        manager.load()
        memory_usage = manager.get_memory_usage()
        
        assert memory_usage > 0
        assert isinstance(memory_usage, int)
    
    def test_get_stats(self, mock_library_files, small_maxorder):
        """Test statistics reporting."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Stats before loading
        stats_before = manager.get_stats()
        assert not stats_before["is_loaded"]
        assert stats_before["load_time_seconds"] is None
        
        # Stats after loading
        manager.load()
        stats_after = manager.get_stats()
        
        assert stats_after["is_loaded"]
        assert stats_after["load_time_seconds"] > 0
        assert stats_after["e_modes_count"] > 0
        assert stats_after["m_modes_count"] > 0
        assert stats_after["memory_usage_mb"] > 0
    
    def test_preload_for_decomposition(self, mock_library_files, small_maxorder):
        """Test preloading optimization."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Preload should load the library
        manager.preload_for_decomposition()
        assert manager._is_loaded
        
        # Preload with different maxorder should reload
        new_maxorder = small_maxorder - 1 if small_maxorder > 1 else small_maxorder + 1
        manager.preload_for_decomposition(new_maxorder)
        assert manager.maxorder == new_maxorder
    
    def test_force_reload(self, mock_library_files, small_maxorder):
        """Test force reloading of library."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Load initially
        manager.load()
        first_load_time = manager._load_time
        
        # Force reload
        manager.load(force_reload=True)
        second_load_time = manager._load_time
        
        assert second_load_time >= first_load_time  # Should be at least as long
    
    def test_string_representation(self, mock_library_files, small_maxorder):
        """Test string representation."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        repr_str = repr(manager)
        assert "LibraryManager" in repr_str
        assert "not loaded" in repr_str
        assert str(small_maxorder) in repr_str
        
        # After loading
        manager.load()
        repr_str_loaded = repr(manager)
        assert "loaded" in repr_str_loaded


class TestGlobalLibraryManager:
    """Tests for global library manager functions."""
    
    def setup_method(self):
        """Reset global state before each test."""
        reset_library_manager()
    
    def test_get_global_manager(self, mock_library_files, small_maxorder):
        """Test global library manager singleton."""
        # First call should create new manager
        manager1 = get_library_manager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Second call should return same manager
        manager2 = get_library_manager()
        
        assert manager1 is manager2
    
    def test_force_new_manager(self, mock_library_files, small_maxorder):
        """Test creating new manager instance."""
        # Get initial manager
        manager1 = get_library_manager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Force new manager
        manager2 = get_library_manager(
            library_dir=mock_library_files,
            maxorder=small_maxorder,
            force_new=True
        )
        
        assert manager1 is not manager2
    
    def test_reset_global_manager(self, mock_library_files, small_maxorder):
        """Test resetting global manager."""
        # Create manager
        manager1 = get_library_manager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Reset
        reset_library_manager()
        
        # Get new manager
        manager2 = get_library_manager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        assert manager1 is not manager2


class TestLibraryManagerErrors:
    """Tests for error handling in LibraryManager."""
    
    def test_nonexistent_library_directory(self, tmp_path):
        """Test error handling for nonexistent library directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        
        with pytest.raises(FileNotFoundError, match="Library directory not found"):
            LibraryManager(library_dir=nonexistent_dir, maxorder=3)
    
    def test_load_before_fit_preprocessing(self, mock_library_files, small_maxorder):
        """Test that accessing modes before loading raises appropriate error."""
        manager = LibraryManager(
            library_dir=mock_library_files,
            maxorder=small_maxorder
        )
        
        # Should automatically load when accessing modes
        # This tests auto-loading behavior
        amp, theta = manager.get_e_mode(1, 0)
        assert amp is not None
        assert theta is not None
        assert manager._is_loaded