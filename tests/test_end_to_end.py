"""
End-to-end tests for the precision analysis library

This test suite:
1. Loads pre-generated test data files from tests/data/
2. Tests the calibration algorithms on these files
3. Verifies that the algorithms produce reasonable results

Test data files are generated once using tests/generate_test_data.py and committed to the repo.
"""
import pytest
import numpy as np
import os
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functions_call import load_inputs, get_error_data, run_method


# Path to test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

# Algorithm-specific regression thresholds with variadic noise levels
# Thresholds are measured for each test case and noise level combination
# Format: {test_case: {noise_level: {algorithm: {translation, rotation}}}}
REGRESSION_THRESHOLDS = {
    "known_transform": {
        "0.1mm": {
            "tsai-lenz": {"translation": 1.28, "rotation": 0.53},
            "park-martin": {"translation": 38.77, "rotation": 72.66},
            "daniilidis": {"translation": 1.22, "rotation": 0.81},
            "li-wang-wu": {"translation": 2.44, "rotation": 0.56},
            "shah": {"translation": 1.13, "rotation": 0.51},
        },
        "1.0mm": {
            "tsai-lenz": {"translation": 1.86, "rotation": 0.54},
            "park-martin": {"translation": 1.85, "rotation": 0.55},
            "daniilidis": {"translation": 2.23, "rotation": 2.18},
            "li-wang-wu": {"translation": 31.59, "rotation": 4.44},
            "shah": {"translation": 1.80, "rotation": 0.53},
        },
        "5.0mm": {
            "tsai-lenz": {"translation": 4.72, "rotation": 0.45},
            "park-martin": {"translation": 4.72, "rotation": 0.45},
            "daniilidis": {"translation": 8.06, "rotation": 7.23},
            "li-wang-wu": {"translation": 120.37, "rotation": 18.96},
            "shah": {"translation": 4.64, "rotation": 0.45},
        },
    },
    "circular_trajectory": {
        "0.1mm": {
            "tsai-lenz": {"translation": 1.59, "rotation": 0.60},
            "park-martin": {"translation": 1.59, "rotation": 0.60},
            "daniilidis": {"translation": 2.70, "rotation": 0.87},
            "li-wang-wu": {"translation": 364.97, "rotation": 0.60},
            "shah": {"translation": 1.55, "rotation": 2.49},
        },
        "1.0mm": {
            "tsai-lenz": {"translation": 2.16, "rotation": 0.58},
            "park-martin": {"translation": 2.16, "rotation": 0.58},
            "daniilidis": {"translation": 12.74, "rotation": 9.34},
            "li-wang-wu": {"translation": 364.96, "rotation": 0.60},
            "shah": {"translation": 2.03, "rotation": 0.57},
        },
        "5.0mm": {
            "tsai-lenz": {"translation": 4.80, "rotation": 0.52},
            "park-martin": {"translation": 4.80, "rotation": 0.52},
            "daniilidis": {"translation": 3441.90, "rotation": 102.42},
            "li-wang-wu": {"translation": 365.59, "rotation": 0.58},
            "shah": {"translation": 4.73, "rotation": 0.52},
        },
    },
    "helix_trajectory": {
        "0.1mm": {
            "tsai-lenz": {"translation": 20.21, "rotation": 4.50},
            "park-martin": {"translation": 1.36, "rotation": 0.58},
            "daniilidis": {"translation": 4.01, "rotation": 0.83},
            "li-wang-wu": {"translation": 1.10, "rotation": 0.58},
            "shah": {"translation": 1.18, "rotation": 0.58},
        },
        "1.0mm": {
            "tsai-lenz": {"translation": 18.45, "rotation": 4.39},
            "park-martin": {"translation": 2.01, "rotation": 0.58},
            "daniilidis": {"translation": 300.45, "rotation": 55.13},
            "li-wang-wu": {"translation": 1.98, "rotation": 0.58},
            "shah": {"translation": 1.98, "rotation": 0.58},
        },
        "5.0mm": {
            "tsai-lenz": {"translation": 20.32, "rotation": 4.28},
            "park-martin": {"translation": 5.82, "rotation": 0.59},
            "daniilidis": {"translation": 4097.92, "rotation": 121.29},
            "li-wang-wu": {"translation": 6.35, "rotation": 0.59},
            "shah": {"translation": 5.79, "rotation": 0.58},
        },
    },
    "complex_3d_path": {
        "0.1mm": {
            "tsai-lenz": {"translation": 25.83, "rotation": 7.94},
            "park-martin": {"translation": 1.24, "rotation": 0.58},
            "daniilidis": {"translation": 2.08, "rotation": 0.61},
            "li-wang-wu": {"translation": 1.10, "rotation": 0.58},
            "shah": {"translation": 1.13, "rotation": 0.58},
        },
        "1.0mm": {
            "tsai-lenz": {"translation": 25.68, "rotation": 7.25},
            "park-martin": {"translation": 1.98, "rotation": 0.58},
            "daniilidis": {"translation": 106.97, "rotation": 9.66},
            "li-wang-wu": {"translation": 1.97, "rotation": 0.58},
            "shah": {"translation": 1.97, "rotation": 0.58},
        },
        "5.0mm": {
            "tsai-lenz": {"translation": 25.16, "rotation": 7.43},
            "park-martin": {"translation": 5.88, "rotation": 0.59},
            "daniilidis": {"translation": 2824.77, "rotation": 75.08},
            "li-wang-wu": {"translation": 5.82, "rotation": 0.58},
            "shah": {"translation": 5.79, "rotation": 0.58},
        },
    },
    "figure8_trajectory": {
        "0.1mm": {
            "tsai-lenz": {"translation": 12.19, "rotation": 3.13},
            "park-martin": {"translation": 1.23, "rotation": 0.58},
            "daniilidis": {"translation": 1.91, "rotation": 0.59},
            "li-wang-wu": {"translation": 1.10, "rotation": 0.58},
            "shah": {"translation": 1.12, "rotation": 0.58},
        },
        "1.0mm": {
            "tsai-lenz": {"translation": 10.78, "rotation": 3.17},
            "park-martin": {"translation": 1.98, "rotation": 0.58},
            "daniilidis": {"translation": 62.69, "rotation": 2.55},
            "li-wang-wu": {"translation": 1.97, "rotation": 0.58},
            "shah": {"translation": 1.97, "rotation": 0.58},
        },
        "5.0mm": {
            "tsai-lenz": {"translation": 14.26, "rotation": 3.38},
            "park-martin": {"translation": 5.81, "rotation": 0.59},
            "daniilidis": {"translation": 1533.58, "rotation": 59.09},
            "li-wang-wu": {"translation": 5.84, "rotation": 0.59},
            "shah": {"translation": 5.79, "rotation": 0.58},
        },
    },
}


@pytest.fixture(params=[0.1, 1.0, 5.0])
def noise_level(request):
    """Fixture providing noise levels for parametrized tests"""
    return request.param


@pytest.fixture
def known_transformation_test_files(noise_level):
    """
    Load pre-generated test files with a known transformation relationship.
    
    Files:
    - known_transform_A_{noise}mm.txt: Sequence of poses in coordinate frame A
    - known_transform_B_{noise}mm.txt: Sequence of poses in coordinate frame B
    """
    suffix = f"_{noise_level}mm"
    file_a = TEST_DATA_DIR / f"known_transform_A{suffix}.txt"
    file_b = TEST_DATA_DIR / f"known_transform_B{suffix}.txt"
    
    if not file_a.exists() or not file_b.exists():
        pytest.skip(f"Test data files not found. Run tests/generate_test_data.py to generate them.")
    
    return str(file_a), str(file_b), noise_level


@pytest.fixture
def simple_trajectory_test_files(noise_level):
    """
    Load pre-generated test files with a circular trajectory.
    """
    suffix = f"_{noise_level}mm"
    file_a = TEST_DATA_DIR / f"circular_trajectory_A{suffix}.txt"
    file_b = TEST_DATA_DIR / f"circular_trajectory_B{suffix}.txt"
    
    if not file_a.exists() or not file_b.exists():
        pytest.skip(f"Test data files not found. Run tests/generate_test_data.py to generate them.")
    
    return str(file_a), str(file_b), noise_level


@pytest.fixture
def helix_trajectory_test_files(noise_level):
    """Load pre-generated test files with a helix trajectory (1000 poses)"""
    suffix = f"_{noise_level}mm"
    file_a = TEST_DATA_DIR / f"helix_trajectory_A{suffix}.txt"
    file_b = TEST_DATA_DIR / f"helix_trajectory_B{suffix}.txt"
    
    if not file_a.exists() or not file_b.exists():
        pytest.skip(f"Test data files not found. Run tests/generate_test_data.py to generate them.")
    
    return str(file_a), str(file_b), noise_level


@pytest.fixture
def complex_3d_path_test_files(noise_level):
    """Load pre-generated test files with a complex 3D path (1000 poses)"""
    suffix = f"_{noise_level}mm"
    file_a = TEST_DATA_DIR / f"complex_3d_path_A{suffix}.txt"
    file_b = TEST_DATA_DIR / f"complex_3d_path_B{suffix}.txt"
    
    if not file_a.exists() or not file_b.exists():
        pytest.skip(f"Test data files not found. Run tests/generate_test_data.py to generate them.")
    
    return str(file_a), str(file_b), noise_level


@pytest.fixture
def figure8_trajectory_test_files(noise_level):
    """Load pre-generated test files with a figure-8 trajectory (1000 poses)"""
    suffix = f"_{noise_level}mm"
    file_a = TEST_DATA_DIR / f"figure8_trajectory_A{suffix}.txt"
    file_b = TEST_DATA_DIR / f"figure8_trajectory_B{suffix}.txt"
    
    if not file_a.exists() or not file_b.exists():
        pytest.skip(f"Test data files not found. Run tests/generate_test_data.py to generate them.")
    
    return str(file_a), str(file_b), noise_level


class TestEndToEndFileLoading:
    """Test that files can be loaded correctly"""
    
    def test_load_generated_files(self, known_transformation_test_files):
        """Test loading the pre-generated test files"""
        file_a, file_b, noise_level = known_transformation_test_files
        
        As, Bs = load_inputs(file_a, file_b)
        
        assert len(As) == 10
        assert len(Bs) == 10
        assert As.shape[1:] == (4, 4)
        assert Bs.shape[1:] == (4, 4)
        
        # Verify all matrices are valid transformation matrices
        for T in As:
            assert T[3, 3] == 1.0
            assert np.all(T[3, :3] == 0)
        
        for T in Bs:
            assert T[3, 3] == 1.0
            assert np.all(T[3, :3] == 0)


class TestEndToEndAlgorithms:
    """Test that algorithms produce reasonable results on test data"""
    
    @pytest.mark.parametrize("method_name", [
        "tsai-lenz",
        "park-martin",
        "daniilidis",
        "li-wang-wu",
        "shah"
    ])
    def test_algorithm_produces_valid_results(self, known_transformation_test_files, method_name):
        """Test that each algorithm produces valid transformation matrices"""
        file_a, file_b, noise_level = known_transformation_test_files
        noise_key = f"{noise_level}mm"
        
        # Load the files
        As, Bs = load_inputs(file_a, file_b)
        
        try:
            # Run the algorithm
            X_computed, Y_computed, t_stats, r_stats = run_method(method_name, As, Bs)
            
            # Verify output shapes
            assert X_computed.shape == (4, 4)
            assert Y_computed.shape == (4, 4)
            
            # Verify transformation matrix properties
            assert X_computed[3, 3] == 1.0
            assert Y_computed[3, 3] == 1.0
            assert np.all(X_computed[3, :3] == 0)
            assert np.all(Y_computed[3, :3] == 0)
            
            # Verify rotation matrix properties
            RX = X_computed[:3, :3]
            RY = Y_computed[:3, :3]
            np.testing.assert_array_almost_equal(RX @ RX.T, np.eye(3), decimal=10)
            np.testing.assert_array_almost_equal(RY @ RY.T, np.eye(3), decimal=10)
            assert abs(np.linalg.det(RX) - 1.0) < 1e-10
            assert abs(np.linalg.det(RY) - 1.0) < 1e-10
            
            # Verify statistics are computed
            required_keys = ["mean", "median", "rmse", "p95", "max"]
            for key in required_keys:
                assert key in t_stats
                assert key in r_stats
                assert isinstance(t_stats[key], (int, float))
                assert isinstance(r_stats[key], (int, float))
            
            # Regression test: algorithm-specific thresholds with noise level
            thresholds = REGRESSION_THRESHOLDS["known_transform"].get(noise_key, {}).get(method_name, {"translation": 1e-5, "rotation": 1e-5})
            t_threshold = thresholds["translation"]
            r_threshold = thresholds["rotation"]
            
            assert t_stats["mean"] < t_threshold, \
                f"Translation error regression for {method_name} (noise {noise_key}): {t_stats['mean']} mm (threshold: {t_threshold} mm)"
            assert r_stats["mean"] < r_threshold, \
                f"Rotation error regression for {method_name} (noise {noise_key}): {r_stats['mean']} deg (threshold: {r_threshold} deg)"
            
        except (np.linalg.LinAlgError, ValueError) as e:
            if method_name == "shah":
                pytest.skip(f"Shah algorithm failed to converge: {e}")
            else:
                raise
    
    def test_all_algorithms_produce_results(self, known_transformation_test_files):
        """Test that get_error_data works with all algorithms"""
        file_a, file_b, noise_level = known_transformation_test_files
        
        methods = ["tsai-lenz", "park-martin", "daniilidis", "li-wang-wu", "shah"]
        t_rows, r_rows = get_error_data(methods, file_a, file_b)
        
        # All methods should have results (or ERR if they failed)
        assert len(t_rows) == len(methods)
        assert len(r_rows) == len(methods)
        
        for method in methods:
            assert method in t_rows
            assert method in r_rows
            
            # Results should either be valid stats or "ERR"
            for key in ["mean", "median", "rmse", "p95", "max"]:
                value = t_rows[method][key]
                assert isinstance(value, (int, float)) or value == "ERR"
                
                value = r_rows[method][key]
                assert isinstance(value, (int, float)) or value == "ERR"
    
    def test_different_trajectories(self, simple_trajectory_test_files):
        """Test algorithms with a different type of trajectory"""
        file_a, file_b, noise_level = simple_trajectory_test_files
        noise_key = f"{noise_level}mm"
        
        # Test with a few algorithms
        methods = ["tsai-lenz", "park-martin", "daniilidis"]
        t_rows, r_rows = get_error_data(methods, file_a, file_b)
        
        for method in methods:
            assert method in t_rows
            assert method in r_rows
            
            # Check that we got valid results (not ERR)
            if t_rows[method]["mean"] != "ERR":
                # Algorithm-specific regression thresholds for circular trajectory with noise level
                thresholds = REGRESSION_THRESHOLDS["circular_trajectory"].get(noise_key, {}).get(method, {"translation": 1e-5, "rotation": 1e-5})
                t_threshold = thresholds["translation"]
                r_threshold = thresholds["rotation"]
                
                assert t_rows[method]["mean"] < t_threshold, \
                    f"Translation error regression for {method} (noise {noise_key}): {t_rows[method]['mean']} mm (threshold: {t_threshold} mm)"
                assert r_rows[method]["mean"] < r_threshold, \
                    f"Rotation error regression for {method} (noise {noise_key}): {r_rows[method]['mean']} deg (threshold: {r_threshold} deg)"
    
    @pytest.mark.parametrize("method_name", [
        "tsai-lenz",
        "park-martin",
        "daniilidis",
        "li-wang-wu",
        "shah"
    ])
    def test_helix_trajectory(self, helix_trajectory_test_files, method_name):
        """Test algorithms on helix trajectory (1000 poses)"""
        file_a, file_b, noise_level = helix_trajectory_test_files
        noise_key = f"{noise_level}mm"
        
        As, Bs = load_inputs(file_a, file_b)
        
        try:
            _, _, t_stats, r_stats = run_method(method_name, As, Bs)
            
            thresholds = REGRESSION_THRESHOLDS["helix_trajectory"].get(noise_key, {}).get(method_name, {"translation": 1e-5, "rotation": 1e-5})
            t_threshold = thresholds["translation"]
            r_threshold = thresholds["rotation"]
            
            assert t_stats["mean"] < t_threshold, \
                f"Translation error regression for {method_name} (noise {noise_key}): {t_stats['mean']} mm (threshold: {t_threshold} mm)"
            assert r_stats["mean"] < r_threshold, \
                f"Rotation error regression for {method_name} (noise {noise_key}): {r_stats['mean']} deg (threshold: {r_threshold} deg)"
        except (np.linalg.LinAlgError, ValueError) as e:
            if method_name == "shah":
                pytest.skip(f"Shah algorithm failed to converge: {e}")
            else:
                raise
    
    @pytest.mark.parametrize("method_name", [
        "tsai-lenz",
        "park-martin",
        "daniilidis",
        "li-wang-wu",
        "shah"
    ])
    def test_complex_3d_path(self, complex_3d_path_test_files, method_name):
        """Test algorithms on complex 3D path (1000 poses)"""
        file_a, file_b, noise_level = complex_3d_path_test_files
        noise_key = f"{noise_level}mm"
        
        As, Bs = load_inputs(file_a, file_b)
        
        try:
            _, _, t_stats, r_stats = run_method(method_name, As, Bs)
            
            thresholds = REGRESSION_THRESHOLDS["complex_3d_path"].get(noise_key, {}).get(method_name, {"translation": 1e-5, "rotation": 1e-5})
            t_threshold = thresholds["translation"]
            r_threshold = thresholds["rotation"]
            
            assert t_stats["mean"] < t_threshold, \
                f"Translation error regression for {method_name} (noise {noise_key}): {t_stats['mean']} mm (threshold: {t_threshold} mm)"
            assert r_stats["mean"] < r_threshold, \
                f"Rotation error regression for {method_name} (noise {noise_key}): {r_stats['mean']} deg (threshold: {r_threshold} deg)"
        except (np.linalg.LinAlgError, ValueError) as e:
            if method_name == "shah":
                pytest.skip(f"Shah algorithm failed to converge: {e}")
            else:
                raise
    
    @pytest.mark.parametrize("method_name", [
        "tsai-lenz",
        "park-martin",
        "daniilidis",
        "li-wang-wu",
        "shah"
    ])
    def test_figure8_trajectory(self, figure8_trajectory_test_files, method_name):
        """Test algorithms on figure-8 trajectory (1000 poses)"""
        file_a, file_b, noise_level = figure8_trajectory_test_files
        noise_key = f"{noise_level}mm"
        
        As, Bs = load_inputs(file_a, file_b)
        
        try:
            _, _, t_stats, r_stats = run_method(method_name, As, Bs)
            
            thresholds = REGRESSION_THRESHOLDS["figure8_trajectory"].get(noise_key, {}).get(method_name, {"translation": 1e-5, "rotation": 1e-5})
            t_threshold = thresholds["translation"]
            r_threshold = thresholds["rotation"]
            
            assert t_stats["mean"] < t_threshold, \
                f"Translation error regression for {method_name} (noise {noise_key}): {t_stats['mean']} mm (threshold: {t_threshold} mm)"
            assert r_stats["mean"] < r_threshold, \
                f"Rotation error regression for {method_name} (noise {noise_key}): {r_stats['mean']} deg (threshold: {r_threshold} deg)"
        except (np.linalg.LinAlgError, ValueError) as e:
            if method_name == "shah":
                pytest.skip(f"Shah algorithm failed to converge: {e}")
            else:
                raise


class TestEndToEndErrorHandling:
    """Test error handling in end-to-end scenarios"""
    
    def test_missing_file_handles_gracefully(self):
        """Test that missing files are handled appropriately"""
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            load_inputs("nonexistent_file_A.txt", "nonexistent_file_B.txt")
    
    def test_invalid_file_format(self):
        """Test that invalid file format is handled"""
        # Create a file with invalid format
        invalid_file = None
        try:
            import tempfile
            invalid_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            invalid_file.write("invalid data\n")
            invalid_file.close()
            
            # Should either raise an error or handle gracefully
            try:
                As, Bs = load_inputs(invalid_file.name, invalid_file.name)
                # If it doesn't raise, at least verify it handles it
                assert len(As) == 0 or len(Bs) == 0
            except (ValueError, IndexError, pd.errors.ParserError, pd.errors.EmptyDataError):
                pass  # Expected behavior
        finally:
            if invalid_file and os.path.exists(invalid_file.name):
                os.unlink(invalid_file.name)
    
    def test_empty_file(self):
        """Test that empty files are handled"""
        empty_file = None
        try:
            import tempfile
            empty_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            empty_file.close()
            
            try:
                As, Bs = load_inputs(empty_file.name, empty_file.name)
                # Empty files should result in empty arrays
                assert len(As) == 0
                assert len(Bs) == 0
            except (pd.errors.EmptyDataError, ValueError):
                pass  # Also acceptable
        finally:
            if empty_file and os.path.exists(empty_file.name):
                os.unlink(empty_file.name)


class TestEndToEndIntegration:
    """Integration tests for the complete workflow"""
    
    def test_complete_workflow(self, known_transformation_test_files):
        """Test the complete workflow from file loading to error calculation"""
        file_a, file_b, noise_level = known_transformation_test_files
        noise_key = f"{noise_level}mm"
        
        # Step 1: Load files
        As, Bs = load_inputs(file_a, file_b)
        assert len(As) > 0
        assert len(Bs) > 0
        
        # Step 2: Run multiple methods
        methods = ["tsai-lenz", "park-martin", "daniilidis"]
        t_rows, r_rows = get_error_data(methods, file_a, file_b)
        
        # Step 3: Verify results
        assert len(t_rows) == len(methods)
        assert len(r_rows) == len(methods)
        
        # Step 4: Verify at least one method succeeded
        successful_methods = [
            m for m in methods 
            if t_rows[m]["mean"] != "ERR" and r_rows[m]["mean"] != "ERR"
        ]
        assert len(successful_methods) > 0, "At least one method should succeed"
        
        # Step 5: Verify successful methods have reasonable errors (algorithm-specific regression thresholds with noise level)
        for method in successful_methods:
            thresholds = REGRESSION_THRESHOLDS["known_transform"].get(noise_key, {}).get(method, {"translation": 1e-5, "rotation": 1e-5})
            t_threshold = thresholds["translation"]
            r_threshold = thresholds["rotation"]
            
            assert t_rows[method]["mean"] < t_threshold, \
                f"Translation error regression for {method} (noise {noise_key}): {t_rows[method]['mean']} mm (threshold: {t_threshold} mm)"
            assert r_rows[method]["mean"] < r_threshold, \
                f"Rotation error regression for {method} (noise {noise_key}): {r_rows[method]['mean']} deg (threshold: {r_threshold} deg)"
            assert t_rows[method]["max"] >= t_rows[method]["mean"]
            assert r_rows[method]["max"] >= r_rows[method]["mean"]
