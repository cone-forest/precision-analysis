"""
Tests for noise generation functions
"""
import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from perlin_noise import process_perlin_file, PerlinNoise
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False

try:
    from gause_noise import process_gaussian_file
    HAS_GAUSSIAN = True
except ImportError:
    HAS_GAUSSIAN = False


@pytest.fixture
def sample_data_file():
    """Create a temporary file with sample pose data"""
    content = """0 100.0 200.0 300.0 45.0 30.0 15.0
1 101.0 201.0 301.0 46.0 31.0 16.0
2 102.0 202.0 302.0 47.0 32.0 17.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.skipif(not HAS_PERLIN, reason="Perlin noise module not available")
class TestPerlinNoise:
    """Tests for Perlin noise generation"""
    
    def test_perlin_noise_class(self):
        """Test PerlinNoise class initialization"""
        noise = PerlinNoise(seed=42)
        assert noise.seed == 42
    
    def test_perlin_noise_fbm(self):
        """Test fractional Brownian motion function"""
        noise = PerlinNoise(seed=42)
        result = noise.fbm(0.5, 0.5, 0.5, octaves=4, scale=1.0)
        assert isinstance(result, (int, float))
        # Result should be in reasonable range
        assert -2.0 <= result <= 2.0
    
    def test_process_perlin_file_zero_noise(self, sample_data_file):
        """Test processing file with zero noise (should be identical)"""
        output_file = sample_data_file + "_output.txt"
        try:
            process_perlin_file(
                input_file=sample_data_file,
                output_file=output_file,
                pos_scale=0.0,
                rot_scale=0.0
            )
            
            # Read both files
            with open(sample_data_file, 'r') as f:
                original = f.readlines()
            with open(output_file, 'r') as f:
                processed = f.readlines()
            
            # Should have same number of lines
            assert len(original) == len(processed)
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_process_perlin_file_creates_output(self, sample_data_file):
        """Test that process_perlin_file creates output file"""
        output_file = sample_data_file + "_output.txt"
        try:
            process_perlin_file(
                input_file=sample_data_file,
                output_file=output_file,
                pos_scale=5.0,
                rot_scale=0.1
            )
            
            assert os.path.exists(output_file)
            
            # Check that output has same number of lines
            with open(output_file, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 3
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_process_perlin_file_format(self, sample_data_file):
        """Test that output file has correct format"""
        output_file = sample_data_file + "_output.txt"
        try:
            process_perlin_file(
                input_file=sample_data_file,
                output_file=output_file,
                pos_scale=1.0,
                rot_scale=0.05
            )
            
            with open(output_file, 'r') as f:
                line = f.readline()
                parts = line.strip().split()
                # Should have 7 parts: id, x, y, z, rx, ry, rz
                assert len(parts) == 7
                # First part should be numeric (id)
                assert parts[0].isdigit()
                # Rest should be floats
                for part in parts[1:]:
                    float(part)  # Should not raise
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


@pytest.mark.skipif(not HAS_GAUSSIAN, reason="Gaussian noise module not available")
class TestGaussianNoise:
    """Tests for Gaussian noise generation"""
    
    def test_process_gaussian_file_zero_noise(self, sample_data_file):
        """Test processing file with very small noise (zero causes singular matrix)"""
        output_file = sample_data_file + "_output.txt"
        try:
            # Use very small but non-zero values to avoid singular matrix
            process_gaussian_file(
                input_file=sample_data_file,
                output_file=output_file,
                pos_std=1e-6,
                rot_std=1e-6
            )
            
            assert os.path.exists(output_file)
            
            # Read both files
            with open(sample_data_file, 'r') as f:
                original_lines = f.readlines()
            with open(output_file, 'r') as f:
                processed_lines = f.readlines()
            
            assert len(original_lines) == len(processed_lines)
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_process_gaussian_file_creates_output(self, sample_data_file):
        """Test that process_gaussian_file creates output file"""
        output_file = sample_data_file + "_output.txt"
        try:
            process_gaussian_file(
                input_file=sample_data_file,
                output_file=output_file,
                pos_std=1.0,
                rot_std=0.1
            )
            
            assert os.path.exists(output_file)
            
            with open(output_file, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 3
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_process_gaussian_file_format(self, sample_data_file):
        """Test that output file has correct format"""
        output_file = sample_data_file + "_output.txt"
        try:
            process_gaussian_file(
                input_file=sample_data_file,
                output_file=output_file,
                pos_std=2.0,
                rot_std=0.2
            )
            
            with open(output_file, 'r') as f:
                line = f.readline()
                parts = line.strip().split()
                assert len(parts) == 7
                assert parts[0].isdigit()
                for part in parts[1:]:
                    float(part)  # Should not raise
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_gaussian_noise_different_seeds(self, sample_data_file):
        """Test that different seeds produce different results"""
        output1 = sample_data_file + "_output1.txt"
        output2 = sample_data_file + "_output2.txt"
        try:
            process_gaussian_file(
                input_file=sample_data_file,
                output_file=output1,
                pos_std=5.0,
                rot_std=0.5,
                seed=42
            )
            process_gaussian_file(
                input_file=sample_data_file,
                output_file=output2,
                pos_std=5.0,
                rot_std=0.5,
                seed=123
            )
            
            with open(output1, 'r') as f:
                lines1 = f.readlines()
            with open(output2, 'r') as f:
                lines2 = f.readlines()
            
            # Results should be different
            assert lines1 != lines2
            
        finally:
            for f in [output1, output2]:
                if os.path.exists(f):
                    os.unlink(f)

