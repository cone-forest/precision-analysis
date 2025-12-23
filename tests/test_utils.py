"""
Tests for utility functions in utils.py
"""
import pytest
import numpy as np
from utils import (
    invert_T, compose, euler_ZYX_to_R, log_SO3, hat,
    load_poses_csv, df_to_Ts, summarize_errors, calculate_Z
)


class TestInvertT:
    """Tests for invert_T function"""
    
    def test_invert_identity(self):
        """Test inverting identity matrix"""
        T = np.eye(4)
        Ti = invert_T(T)
        np.testing.assert_array_almost_equal(Ti, np.eye(4))
    
    def test_invert_simple_transform(self, sample_transform_matrix):
        """Test inverting a simple transformation"""
        Ti = invert_T(sample_transform_matrix)
        T_back = invert_T(Ti)
        np.testing.assert_array_almost_equal(T_back, sample_transform_matrix)
    
    def test_invert_composition(self, sample_transform_matrix):
        """Test that T @ invert_T(T) = identity"""
        Ti = invert_T(sample_transform_matrix)
        result = sample_transform_matrix @ Ti
        np.testing.assert_array_almost_equal(result, np.eye(4), decimal=10)


class TestCompose:
    """Tests for compose function"""
    
    def test_compose_identity_rotation(self, sample_translation):
        """Test composing with identity rotation"""
        R = np.eye(3)
        T = compose(R, sample_translation)
        np.testing.assert_array_almost_equal(T[:3, 3], sample_translation)
        np.testing.assert_array_almost_equal(T[:3, :3], R)
        assert T[3, 3] == 1.0
        assert np.all(T[3, :3] == 0)
    
    def test_compose_zero_translation(self, sample_rotation_matrix):
        """Test composing with zero translation"""
        t = np.zeros(3)
        T = compose(sample_rotation_matrix, t)
        np.testing.assert_array_almost_equal(T[:3, :3], sample_rotation_matrix)
        np.testing.assert_array_almost_equal(T[:3, 3], t)


class TestEulerZYXToR:
    """Tests for euler_ZYX_to_R function"""
    
    def test_zero_angles(self):
        """Test with zero angles"""
        R = euler_ZYX_to_R(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_90_degree_z_rotation(self):
        """Test 90 degree rotation around Z axis"""
        R = euler_ZYX_to_R(np.pi / 2, 0, 0)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(R, expected)
    
    def test_rotation_properties(self):
        """Test that rotation matrix has correct properties"""
        R = euler_ZYX_to_R(0.5, 0.3, 0.2)
        # Should be orthogonal
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        # Determinant should be 1
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestLogSO3:
    """Tests for log_SO3 function"""
    
    def test_log_identity(self):
        """Test logarithm of identity matrix"""
        R = np.eye(3)
        w = log_SO3(R)
        np.testing.assert_array_almost_equal(w, np.zeros(3))
    
    def test_log_small_rotation(self):
        """Test logarithm of small rotation"""
        angle = 0.01
        R = euler_ZYX_to_R(angle, 0, 0)
        w = log_SO3(R)
        # Should be approximately the rotation vector
        assert np.linalg.norm(w) < 0.1


class TestHat:
    """Tests for hat function"""
    
    def test_hat_zero_vector(self):
        """Test hat of zero vector"""
        w = np.zeros(3)
        W = hat(w)
        np.testing.assert_array_almost_equal(W, np.zeros((3, 3)))
    
    def test_hat_skew_symmetric(self):
        """Test that hat produces skew-symmetric matrix"""
        w = np.array([1.0, 2.0, 3.0])
        W = hat(w)
        np.testing.assert_array_almost_equal(W, -W.T)


class TestLoadPosesCSV:
    """Tests for load_poses_csv function"""
    
    def test_load_valid_file(self, sample_csv_file):
        """Test loading a valid CSV file"""
        df = load_poses_csv(sample_csv_file)
        assert len(df) == 5
        assert list(df.columns) == ["id", "X", "Y", "Z", "RZ", "RY", "RX"]
        assert df.iloc[0]["X"] == 100.0
        assert df.iloc[0]["RZ"] == 45.0


class TestDfToTs:
    """Tests for df_to_Ts function"""
    
    def test_df_to_ts_conversion(self, sample_csv_file):
        """Test converting dataframe to transformation matrices"""
        df = load_poses_csv(sample_csv_file)
        Ts = df_to_Ts(df)
        assert len(Ts) == 5
        assert Ts.shape[1:] == (4, 4)
        # Check that all matrices are valid transformation matrices
        for T in Ts:
            assert T[3, 3] == 1.0
            assert np.all(T[3, :3] == 0)


class TestSummarizeErrors:
    """Tests for summarize_errors function"""
    
    def test_summarize_errors_identical(self, sample_poses):
        """Test error calculation with identical poses"""
        # Use same poses for A and B, with identity X and Y
        X = np.eye(4)
        Y = np.eye(4)
        t_stats, r_stats = summarize_errors(sample_poses, sample_poses, X, Y)
        
        # Errors should be very small (numerical precision)
        assert t_stats["mean"] < 1e-10
        assert r_stats["mean"] < 1e-10
    
    def test_summarize_errors_metrics(self, sample_poses):
        """Test that all required metrics are present"""
        X = np.eye(4)
        Y = np.eye(4)
        t_stats, r_stats = summarize_errors(sample_poses, sample_poses, X, Y)
        
        required_metrics = ["mean", "median", "rmse", "p95", "max"]
        for metric in required_metrics:
            assert metric in t_stats
            assert metric in r_stats
            assert isinstance(t_stats[metric], (int, float))
            assert isinstance(r_stats[metric], (int, float))


class TestCalculateZ:
    """Tests for calculate_Z function"""
    
    def test_calculate_z_shape(self, sample_poses):
        """Test that calculate_Z returns correct shape"""
        X = np.eye(4)
        Z = calculate_Z(sample_poses, sample_poses, X)
        assert Z.shape == (4, 4)
        assert Z[3, 3] == 1.0
    
    def test_calculate_z_properties(self, sample_poses):
        """Test that Z has correct transformation matrix properties"""
        X = np.eye(4)
        Z = calculate_Z(sample_poses, sample_poses, X)
        # Check rotation matrix properties
        R = Z[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

