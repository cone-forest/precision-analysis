"""
Tests for calibration algorithms
"""
import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import compose, euler_ZYX_to_R, invert_T
from tsai_lenz import tsai_lenz
from park_martin import park_martin
from daniilidis import daniilidis


@pytest.fixture
def simple_poses():
    """Create simple test poses for algorithm testing"""
    poses = []
    for i in range(3):
        angle = i * np.pi / 6
        R = euler_ZYX_to_R(angle, 0, 0)
        t = np.array([i * 1.0, 0, 0])
        poses.append(compose(R, t))
    return np.array(poses)


@pytest.fixture
def known_transformation_poses():
    """Create poses with a known transformation relationship"""
    # Create poses A
    As = []
    for i in range(5):
        angle = i * np.pi / 8
        R = euler_ZYX_to_R(angle, 0, 0)
        t = np.array([i * 0.5, i * 0.3, 0])
        As.append(compose(R, t))
    
    # Create known transformation X
    X = compose(euler_ZYX_to_R(np.pi / 4, 0, 0), np.array([1.0, 2.0, 3.0]))
    
    # Create poses B such that A @ X = Y @ B (with Y = identity for simplicity)
    Y = np.eye(4)
    Bs = []
    for A in As:
        # B = Y^-1 @ A @ X = A @ X (since Y = I)
        B = invert_T(Y) @ A @ X
        Bs.append(B)
    
    return np.array(As), np.array(Bs), X, Y


class TestTsaiLenz:
    """Tests for tsai-lenz algorithm"""
    
    def test_tsai_lenz_returns_transforms(self, simple_poses):
        """Test that tsai_lenz returns two transformation matrices"""
        X, Y = tsai_lenz(simple_poses, simple_poses)
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert X[3, 3] == 1.0
        assert Y[3, 3] == 1.0
    
    def test_tsai_lenz_rotation_properties(self, simple_poses):
        """Test that returned rotations have correct properties"""
        X, Y = tsai_lenz(simple_poses, simple_poses)
        RX = X[:3, :3]
        RY = Y[:3, :3]
        
        # Should be orthogonal
        np.testing.assert_array_almost_equal(RX @ RX.T, np.eye(3))
        np.testing.assert_array_almost_equal(RY @ RY.T, np.eye(3))
        
        # Determinant should be 1
        assert abs(np.linalg.det(RX) - 1.0) < 1e-10
        assert abs(np.linalg.det(RY) - 1.0) < 1e-10


class TestParkMartin:
    """Tests for park-martin algorithm"""
    
    def test_park_martin_returns_transforms(self, simple_poses):
        """Test that park_martin returns two transformation matrices"""
        X, Y = park_martin(simple_poses, simple_poses)
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert X[3, 3] == 1.0
        assert Y[3, 3] == 1.0
    
    def test_park_martin_rotation_properties(self, simple_poses):
        """Test that returned rotations have correct properties"""
        X, Y = park_martin(simple_poses, simple_poses)
        RX = X[:3, :3]
        RY = Y[:3, :3]
        
        np.testing.assert_array_almost_equal(RX @ RX.T, np.eye(3))
        np.testing.assert_array_almost_equal(RY @ RY.T, np.eye(3))
        assert abs(np.linalg.det(RX) - 1.0) < 1e-10
        assert abs(np.linalg.det(RY) - 1.0) < 1e-10


class TestDaniilidis:
    """Tests for daniilidis algorithm"""
    
    def test_daniilidis_returns_transforms(self, simple_poses):
        """Test that daniilidis returns two transformation matrices"""
        X, Y = daniilidis(simple_poses, simple_poses)
        assert X.shape == (4, 4)
        assert Y.shape == (4, 4)
        assert X[3, 3] == 1.0
        assert Y[3, 3] == 1.0
    
    def test_daniilidis_rotation_properties(self, simple_poses):
        """Test that returned rotations have correct properties"""
        X, Y = daniilidis(simple_poses, simple_poses)
        RX = X[:3, :3]
        RY = Y[:3, :3]
        
        np.testing.assert_array_almost_equal(RX @ RX.T, np.eye(3))
        np.testing.assert_array_almost_equal(RY @ RY.T, np.eye(3))
        assert abs(np.linalg.det(RX) - 1.0) < 1e-10
        assert abs(np.linalg.det(RY) - 1.0) < 1e-10
