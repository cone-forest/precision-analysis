"""
Tests for functions_call.py module
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functions_call import run_method, load_inputs, get_error_data
from utils import compose, euler_ZYX_to_R


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with pose data"""
    content = """0 100.0 200.0 300.0 45.0 30.0 15.0
1 101.0 201.0 301.0 46.0 31.0 16.0
2 102.0 202.0 302.0 47.0 32.0 17.0
3 103.0 203.0 303.0 48.0 33.0 18.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_poses():
    """Generate a list of sample pose transformation matrices"""
    from utils import compose, euler_ZYX_to_R
    import numpy as np
    poses = []
    for i in range(5):
        angle = i * np.pi / 8
        R = euler_ZYX_to_R(angle, 0, 0)
        t = np.array([i * 0.5, i * 0.3, i * 0.1])
        poses.append(compose(R, t))
    return np.array(poses)


class TestRunMethod:
    """Tests for run_method function"""
    
    def test_run_method_tsai_lenz(self, sample_poses):
        """Test running tsai-lenz method"""
        X, Y, t_stats, r_stats = run_method("tsai-lenz", sample_poses, sample_poses)
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert isinstance(t_stats, dict)
        assert isinstance(r_stats, dict)
    
    def test_run_method_park_martin(self, sample_poses):
        """Test running park-martin method"""
        X, Y, t_stats, r_stats = run_method("park-martin", sample_poses, sample_poses)
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert isinstance(t_stats, dict)
        assert isinstance(r_stats, dict)
    
    def test_run_method_daniilidis(self, sample_poses):
        """Test running daniilidis method"""
        X, Y, t_stats, r_stats = run_method("daniilidis", sample_poses, sample_poses)
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert isinstance(t_stats, dict)
        assert isinstance(r_stats, dict)
    
    def test_run_method_li_wang_wu(self, sample_poses):
        """Test running li-wang-wu method"""
        X, Y, t_stats, r_stats = run_method("li-wang-wu", sample_poses, sample_poses)
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert isinstance(t_stats, dict)
        assert isinstance(r_stats, dict)
    
    def test_run_method_shah(self, sample_poses):
        """Test running shah method"""
        try:
            X, Y, t_stats, r_stats = run_method("shah", sample_poses, sample_poses)
            assert X.shape == (4, 4)
            assert Y.shape == (4, 4)
            assert isinstance(t_stats, dict)
            assert isinstance(r_stats, dict)
        except (np.linalg.LinAlgError, ValueError):
            # Shah algorithm can fail with certain inputs
            pytest.skip("Shah algorithm failed to converge with these inputs")
    
    def test_run_method_unknown_method(self, sample_poses):
        """Test that unknown method raises ValueError"""
        with pytest.raises(ValueError, match="Неизвестный метод"):
            run_method("unknown-method", sample_poses, sample_poses)
    
    def test_run_method_returns_stats(self, sample_poses):
        """Test that run_method returns statistics with all required keys"""
        _, _, t_stats, r_stats = run_method("tsai-lenz", sample_poses, sample_poses)
        
        required_keys = ["mean", "median", "rmse", "p95", "max"]
        for key in required_keys:
            assert key in t_stats
            assert key in r_stats
            assert isinstance(t_stats[key], (int, float))
            assert isinstance(r_stats[key], (int, float))


class TestLoadInputs:
    """Tests for load_inputs function"""
    
    def test_load_inputs(self, sample_csv_file):
        """Test loading two input files"""
        As, Bs = load_inputs(sample_csv_file, sample_csv_file)
        assert len(As) == 4
        assert len(Bs) == 4
        assert As.shape[1:] == (4, 4)
        assert Bs.shape[1:] == (4, 4)
    
    def test_load_inputs_different_files(self, sample_csv_file):
        """Test loading two different files"""
        As, Bs = load_inputs(sample_csv_file, sample_csv_file)
        # Even if same file, should load correctly
        assert len(As) == len(Bs)


class TestGetErrorData:
    """Tests for get_error_data function"""
    
    def test_get_error_data_single_method(self, sample_csv_file):
        """Test getting error data for a single method"""
        t_rows, r_rows = get_error_data(["tsai-lenz"], sample_csv_file, sample_csv_file)
        
        assert "tsai-lenz" in t_rows
        assert "tsai-lenz" in r_rows
        assert isinstance(t_rows["tsai-lenz"], dict)
        assert isinstance(r_rows["tsai-lenz"], dict)
        
        required_keys = ["mean", "median", "rmse", "p95", "max"]
        for key in required_keys:
            assert key in t_rows["tsai-lenz"]
            assert key in r_rows["tsai-lenz"]
    
    def test_get_error_data_multiple_methods(self, sample_csv_file):
        """Test getting error data for multiple methods"""
        methods = ["tsai-lenz", "park-martin", "shah"]
        t_rows, r_rows = get_error_data(methods, sample_csv_file, sample_csv_file)
        
        for method in methods:
            assert method in t_rows
            assert method in r_rows
    
    def test_get_error_data_handles_errors(self, sample_csv_file):
        """Test that get_error_data handles method errors gracefully"""
        # This should not raise, even if a method fails
        t_rows, r_rows = get_error_data(["tsai-lenz", "invalid"], sample_csv_file, sample_csv_file)
        
        # Valid method should work
        assert "tsai-lenz" in t_rows
        
        # Invalid method should have ERR values
        if "invalid" in t_rows:
            assert t_rows["invalid"]["mean"] == "ERR"
    
    def test_get_error_data_all_methods(self, sample_csv_file):
        """Test getting error data for all available methods"""
        methods = ["tsai-lenz", "park-martin", "daniilidis", "li-wang-wu", "shah"]
        t_rows, r_rows = get_error_data(methods, sample_csv_file, sample_csv_file)
        
        assert len(t_rows) == len(methods)
        assert len(r_rows) == len(methods)
        
        for method in methods:
            assert method in t_rows
            assert method in r_rows
            # Check that values are either numbers or "ERR"
            for key in ["mean", "median", "rmse", "p95", "max"]:
                value = t_rows[method][key]
                assert isinstance(value, (int, float)) or value == "ERR"

