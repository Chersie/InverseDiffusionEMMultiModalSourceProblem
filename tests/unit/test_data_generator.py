"""
Unit tests for DataGenerator and related components.

Tests the unified data generation system including Latin square generators
and field computation.
"""
import numpy as np
import pytest

from src.core.data_generator import (
    LatinSquareGenerator, FieldGenerator, DataGenerator,
    LatinSquareConfig, GridConfig,
    get_mode_list, get_n_modes, pack_coefficients, unpack_coefficients,
    generate_pipeline_fields
)
from src.core.config import Config


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_mode_list(self):
        """Test mode list generation."""
        # Test small maxorder
        modes = get_mode_list(2)
        expected = [(1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
        assert modes == expected
        
        # Test single order
        modes = get_mode_list(1)
        expected = [(1, -1), (1, 0), (1, 1)]
        assert modes == expected
    
    def test_get_n_modes(self):
        """Test mode count calculation."""
        assert get_n_modes(1) == 3    # 1*3 = 3
        assert get_n_modes(2) == 8    # 2*4 = 8  
        assert get_n_modes(3) == 15   # 3*5 = 15
        assert get_n_modes(15) == 255 # 15*17 = 255
    
    def test_pack_unpack_coefficients(self):
        """Test coefficient packing and unpacking."""
        n_samples, n_modes = 10, 6
        
        # Create test coefficients
        a_e = np.random.randn(n_samples, n_modes) + 1j * np.random.randn(n_samples, n_modes)
        a_m = np.random.randn(n_samples, n_modes) + 1j * np.random.randn(n_samples, n_modes)
        
        # Pack coefficients
        packed = pack_coefficients(a_e, a_m)
        
        # Check packed shape
        assert packed.shape == (n_samples, 4 * n_modes)
        assert packed.dtype == np.float32
        
        # Unpack coefficients
        a_e_recovered, a_m_recovered = unpack_coefficients(packed)
        
        # Check shapes and types
        assert a_e_recovered.shape == a_e.shape
        assert a_m_recovered.shape == a_m.shape
        assert a_e_recovered.dtype == np.complex64
        assert a_m_recovered.dtype == np.complex64
        
        # Check values (within float32 precision)
        pytest.assert_complex_arrays_close(a_e_recovered, a_e.astype(np.complex64), rtol=1e-6)
        pytest.assert_complex_arrays_close(a_m_recovered, a_m.astype(np.complex64), rtol=1e-6)
    
    def test_pack_unpack_single_sample(self):
        """Test packing/unpacking single sample."""
        n_modes = 6
        
        a_e = np.random.randn(n_modes) + 1j * np.random.randn(n_modes)
        a_m = np.random.randn(n_modes) + 1j * np.random.randn(n_modes)
        
        # Need to add batch dimension for packing
        a_e_batch = a_e[np.newaxis, :]
        a_m_batch = a_m[np.newaxis, :]
        
        packed = pack_coefficients(a_e_batch, a_m_batch)
        a_e_recovered, a_m_recovered = unpack_coefficients(packed)
        
        # Remove batch dimension for comparison
        pytest.assert_complex_arrays_close(a_e_recovered[0], a_e.astype(np.complex64), rtol=1e-6)
        pytest.assert_complex_arrays_close(a_m_recovered[0], a_m.astype(np.complex64), rtol=1e-6)


class TestLatinSquareConfig:
    """Tests for LatinSquareConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LatinSquareConfig()
        
        assert config.mode == "fixed"
        assert config.scale == 1.0
        assert config.base_seed == 42
        assert config.enable_permutations is True
    
    def test_invalid_mode(self):
        """Test error handling for invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            LatinSquareConfig(mode="invalid")


class TestGridConfig:
    """Tests for GridConfig."""
    
    def test_default_config(self):
        """Test default grid configuration."""
        config = GridConfig()
        
        assert config.n_phi == 360
        assert config.n_theta == 179
        assert config.angle_step_deg == 1.0
        assert config.n_points == 360 * 179
    
    def test_validate_consistent_config(self):
        """Test validation with consistent configuration."""
        config = GridConfig(n_phi=36, n_theta=17, angle_step_deg=10.0)
        
        # Should not raise error
        config.validate()
    
    def test_validate_inconsistent_config(self):
        """Test validation with inconsistent configuration."""
        config = GridConfig(n_phi=37, n_theta=17, angle_step_deg=10.0)  # 37 != 360/10
        
        with pytest.raises(ValueError, match="n_phi.*inconsistent"):
            config.validate()


class TestLatinSquareGenerator:
    """Tests for LatinSquareGenerator."""
    
    def test_fixed_mode_generation(self):
        """Test fixed mode coefficient generation."""
        config = LatinSquareConfig(mode="fixed", scale=0.5)
        generator = LatinSquareGenerator(config)
        
        maxorder = 2
        
        # Generate coefficients in dict format
        a_e_dict, a_m_dict = generator.generate_coefficients_dict(maxorder)
        
        # Check structure
        assert isinstance(a_e_dict, dict)
        assert isinstance(a_m_dict, dict)
        
        for l in range(1, maxorder + 1):
            assert l in a_e_dict
            assert l in a_m_dict
            for m in range(-l, l + 1):
                assert m in a_e_dict[l]
                assert m in a_m_dict[l]
                assert isinstance(a_e_dict[l][m], complex)
                assert isinstance(a_m_dict[l][m], complex)
        
        # Test reproducibility
        a_e_dict2, a_m_dict2 = generator.generate_coefficients_dict(maxorder)
        
        for l in range(1, maxorder + 1):
            for m in range(-l, l + 1):
                assert a_e_dict[l][m] == a_e_dict2[l][m]
                assert a_m_dict[l][m] == a_m_dict2[l][m]
    
    def test_random_mode_generation(self):
        """Test random mode coefficient generation."""
        config = LatinSquareConfig(mode="random", scale=0.5, base_seed=123)
        generator = LatinSquareGenerator(config)
        
        maxorder = 2
        
        # Generate for different sample IDs
        a_e_1, a_m_1 = generator.generate_coefficients_array(maxorder, sample_id=0)
        a_e_2, a_m_2 = generator.generate_coefficients_array(maxorder, sample_id=1)
        
        # Check shapes
        expected_n_modes = get_n_modes(maxorder)
        assert a_e_1.shape == (expected_n_modes,)
        assert a_m_1.shape == (expected_n_modes,)
        assert a_e_1.dtype == np.complex64
        assert a_m_1.dtype == np.complex64
        
        # Different sample IDs should give different results
        assert not np.allclose(a_e_1, a_e_2)
        assert not np.allclose(a_m_1, a_m_2)
        
        # Same sample ID should give same results
        a_e_1b, a_m_1b = generator.generate_coefficients_array(maxorder, sample_id=0)
        pytest.assert_complex_arrays_close(a_e_1, a_e_1b)
        pytest.assert_complex_arrays_close(a_m_1, a_m_1b)
    
    def test_array_dict_consistency(self):
        """Test consistency between array and dict formats."""
        config = LatinSquareConfig(mode="fixed")
        generator = LatinSquareGenerator(config)
        
        maxorder = 2
        
        # Generate in both formats
        a_e_dict, a_m_dict = generator.generate_coefficients_dict(maxorder)
        a_e_array, a_m_array = generator.generate_coefficients_array(maxorder)
        
        # Convert dict to array format for comparison
        mode_list = get_mode_list(maxorder)
        
        for k, (l, m) in enumerate(mode_list):
            assert np.isclose(a_e_array[k], a_e_dict[l][m])
            assert np.isclose(a_m_array[k], a_m_dict[l][m])
    
    def test_batch_generation(self):
        """Test batch coefficient generation."""
        config = LatinSquareConfig(mode="random", base_seed=42)
        generator = LatinSquareGenerator(config)
        
        maxorder = 2
        n_samples = 5
        
        # Generate batch
        a_e_batch, a_m_batch = generator.generate_batch_arrays(maxorder, n_samples)
        
        # Check shapes
        expected_n_modes = get_n_modes(maxorder)
        assert a_e_batch.shape == (n_samples, expected_n_modes)
        assert a_m_batch.shape == (n_samples, expected_n_modes)
        
        # Check that samples are different
        for i in range(1, n_samples):
            assert not np.allclose(a_e_batch[i], a_e_batch[0])
        
        # Check consistency with individual generation
        a_e_single, a_m_single = generator.generate_coefficients_array(maxorder, sample_id=0)
        pytest.assert_complex_arrays_close(a_e_batch[0], a_e_single)
        pytest.assert_complex_arrays_close(a_m_batch[0], a_m_single)


class TestFieldGenerator:
    """Tests for FieldGenerator."""
    
    def test_initialization(self):
        """Test field generator initialization."""
        grid_config = GridConfig(n_phi=8, n_theta=6, angle_step_deg=45.0)
        generator = FieldGenerator(grid_config)
        
        assert generator.grid_config.n_phi == 8
        assert generator.grid_config.n_theta == 6
        assert generator._mpfield_module is None
    
    def test_build_grid(self):
        """Test angular grid construction."""
        grid_config = GridConfig(n_phi=4, n_theta=3, angle_step_deg=90.0)
        generator = FieldGenerator(grid_config)
        
        theta, phi = generator.build_grid()
        
        # Check shapes
        assert theta.shape == (4, 3)
        assert phi.shape == (4, 3)
        
        # Check value ranges
        assert np.all(theta >= 0) and np.all(theta <= np.pi)
        assert np.all(phi >= 0) and np.all(phi < 2 * np.pi)
        
        # Check grid structure
        assert theta[0, 0] == theta[1, 0]  # Same theta for different phi
        assert phi[0, 0] != phi[1, 0]      # Different phi for same theta
    
    @pytest.mark.skip(reason="Requires MPField module which may not be available")
    def test_compute_field_from_dict(self):
        """Test field computation from coefficient dictionaries."""
        # This test would require the actual MPField module
        # Skip for now since it's an external dependency
        pass


class TestDataGenerator:
    """Tests for unified DataGenerator."""
    
    def test_pipeline_mode_initialization(self):
        """Test initialization for pipeline mode."""
        generator = DataGenerator.for_pipeline()
        
        assert generator.latin_generator.config.mode == "fixed"
        assert isinstance(generator.field_generator, FieldGenerator)
    
    def test_ml_training_mode_initialization(self):
        """Test initialization for ML training mode."""
        generator = DataGenerator.for_ml_training()
        
        assert generator.latin_generator.config.mode == "random"
        assert isinstance(generator.field_generator, FieldGenerator)
    
    @pytest.mark.skip(reason="Requires MPField module")
    def test_generate_sample(self):
        """Test single sample generation."""
        # Skip due to MPField dependency
        pass
    
    def test_initialization_with_custom_configs(self):
        """Test initialization with custom configurations."""
        latin_config = LatinSquareConfig(mode="fixed", scale=0.1)
        grid_config = GridConfig(n_phi=8, n_theta=6)
        
        generator = DataGenerator(latin_config=latin_config, grid_config=grid_config)
        
        assert generator.latin_generator.config.scale == 0.1
        assert generator.field_generator.grid_config.n_phi == 8


class TestDataGeneratorIntegration:
    """Integration tests for DataGenerator components."""
    
    def test_coefficient_generation_reproducibility(self):
        """Test that pipeline mode generates reproducible coefficients."""
        generator1 = DataGenerator.for_pipeline()
        generator2 = DataGenerator.for_pipeline()
        
        maxorder = 2
        
        # Generate coefficients with both generators
        a_e_1, a_m_1 = generator1.latin_generator.generate_coefficients_array(maxorder, sample_id=0)
        a_e_2, a_m_2 = generator2.latin_generator.generate_coefficients_array(maxorder, sample_id=0)
        
        # Should be identical
        pytest.assert_complex_arrays_close(a_e_1, a_e_2)
        pytest.assert_complex_arrays_close(a_m_1, a_m_2)
    
    def test_ml_mode_diversity(self):
        """Test that ML mode generates diverse coefficients."""
        generator = DataGenerator.for_ml_training()
        
        maxorder = 2
        n_samples = 5
        
        # Generate multiple samples
        samples = []
        for i in range(n_samples):
            a_e, a_m = generator.latin_generator.generate_coefficients_array(maxorder, sample_id=i)
            samples.append((a_e, a_m))
        
        # Check that samples are different
        for i in range(1, n_samples):
            a_e_0, a_m_0 = samples[0]
            a_e_i, a_m_i = samples[i]
            
            # Should not be identical
            assert not np.allclose(a_e_0, a_e_i, rtol=1e-10)
            assert not np.allclose(a_m_0, a_m_i, rtol=1e-10)


@pytest.mark.skip(reason="Requires MPField module and proper file system setup")
class TestPipelineFieldGeneration:
    """Tests for pipeline field generation function."""
    
    def test_generate_pipeline_fields(self, temp_data_dir):
        """Test pipeline field file generation."""
        output_path = temp_data_dir / "test_fields.txt"
        
        result_path = generate_pipeline_fields(
            output_path=output_path,
            maxorder=2
        )
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Check file content structure
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) > 0
        
        # Check first line format (should be: theta phi power |E_theta| phase |E_phi| phase)
        first_line = lines[0].strip().split()
        assert len(first_line) == 7  # Expected number of columns