"""
Pytest configuration and fixtures for testing precision-analysis library
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import compose, euler_ZYX_to_R, invert_T


@pytest.fixture
def sample_rotation_matrix():
    """Generate a sample 3x3 rotation matrix"""
    angle = np.pi / 4
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return R


@pytest.fixture
def sample_translation():
    """Generate a sample 3D translation vector"""
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def sample_transform_matrix(sample_rotation_matrix, sample_translation):
    """Generate a sample 4x4 transformation matrix"""
    return compose(sample_rotation_matrix, sample_translation)


@pytest.fixture
def sample_poses():
    """Generate a list of sample pose transformation matrices"""
    poses = []
    for i in range(5):
        angle = i * np.pi / 8
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        t = np.array([i * 0.5, i * 0.3, i * 0.1])
        poses.append(compose(R, t))
    return np.array(poses)


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with pose data"""
    content = """0 100.0 200.0 300.0 45.0 30.0 15.0
1 101.0 201.0 301.0 46.0 31.0 16.0
2 102.0 202.0 302.0 47.0 32.0 17.0
3 103.0 203.0 303.0 48.0 33.0 18.0
4 104.0 204.0 304.0 49.0 34.0 19.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_csv_file_minimal():
    """Create a minimal temporary CSV file with pose data (2 poses)"""
    content = """0 100.0 200.0 300.0 45.0 30.0 15.0
1 101.0 201.0 301.0 46.0 31.0 16.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)

