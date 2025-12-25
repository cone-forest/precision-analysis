"""
Script to generate test data files for end-to-end tests.

Run this script once to generate the test data files that will be committed to the repo.
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import compose, euler_ZYX_to_R, invert_T


def add_noise_to_poses(poses, pos_noise_range=5.0, rot_noise_range=0.5, seed=42):
    """
    Add noise to a list of poses.
    
    Args:
        poses: List of 4x4 transformation matrices
        pos_noise_range: Maximum position noise in mm (±range)
        rot_noise_range: Maximum rotation noise in degrees (±range)
        seed: Random seed for reproducibility
    
    Returns:
        List of noisy poses
    """
    np.random.seed(seed)  # Fixed seed for reproducibility
    noisy_poses = []
    
    for T in poses:
        # Extract position and rotation
        R = T[:3, :3]
        t = T[:3, 3]
        
        # Add position noise (±pos_noise_range mm)
        pos_noise = np.random.uniform(-pos_noise_range, pos_noise_range, 3)
        t_noisy = t + pos_noise
        
        # Add rotation noise (±rot_noise_range degrees)
        rot_noise_deg = np.random.uniform(-rot_noise_range, rot_noise_range, 3)
        rot_noise_rad = np.deg2rad(rot_noise_deg)
        R_noise = euler_ZYX_to_R(rot_noise_rad[0], rot_noise_rad[1], rot_noise_rad[2])
        R_noisy = R @ R_noise
        
        # Reconstruct noisy transformation
        T_noisy = compose(R_noisy, t_noisy)
        noisy_poses.append(T_noisy)
    
    return noisy_poses


def generate_pose_file(poses, filepath):
    """
    Generate a pose file in the expected format: id X Y Z RZ RY RX
    
    Args:
        poses: List of 4x4 transformation matrices
        filepath: Path to save the file
    """
    with open(filepath, 'w') as f:
        for i, T in enumerate(poses):
            # Extract position
            x, y, z = T[:3, 3]
            
            # Extract rotation matrix
            R = T[:3, :3]
            
            # Convert rotation matrix to Euler angles (ZYX order)
            # Using atan2 for robust angle extraction
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                rz = np.arctan2(R[1, 0], R[0, 0])  # Z rotation
                ry = np.arctan2(-R[2, 0], sy)      # Y rotation
                rx = np.arctan2(R[2, 1], R[2, 2])  # X rotation
            else:
                rz = np.arctan2(-R[0, 1], R[1, 1])
                ry = np.arctan2(-R[2, 0], sy)
                rx = 0
            
            # Convert to degrees
            rz_deg = np.rad2deg(rz)
            ry_deg = np.rad2deg(ry)
            rx_deg = np.rad2deg(rx)
            
            # Write in format: id X Y Z RZ RY RX
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {rz_deg:.6f} {ry_deg:.6f} {rx_deg:.6f}\n")


def generate_known_transformation_files(pos_noise_range=5.0, rot_noise_range=0.5, suffix=""):
    """Generate test files with a known transformation relationship"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create a known transformation X (from frame A to frame B)
    X_known = compose(
        euler_ZYX_to_R(np.deg2rad(30), np.deg2rad(15), np.deg2rad(10)),
        np.array([50.0, 100.0, 25.0])  # Translation in mm
    )
    
    # Set Y to identity for simplicity (AX = XB case)
    Y_known = np.eye(4)
    
    # Generate a sequence of poses in frame A
    # These represent a trajectory (e.g., robot end-effector moving)
    As = []
    for i in range(10):
        # Create a trajectory: rotation and translation
        angle = i * np.pi / 10
        R_A = euler_ZYX_to_R(angle, 0.1 * i, 0.05 * i)
        t_A = np.array([i * 20.0, i * 15.0, i * 5.0])
        A = compose(R_A, t_A)
        As.append(A)
    
    # Generate corresponding poses in frame B
    # B = Y^-1 @ A @ X = A @ X (since Y = I)
    Bs = []
    for A in As:
        B = invert_T(Y_known) @ A @ X_known
        Bs.append(B)
    
    # Add noise to B poses with specified noise level
    # Use different seed for different noise levels to ensure different noise patterns
    seed = 42 + int(pos_noise_range * 10)
    Bs_noisy = add_noise_to_poses(Bs, pos_noise_range=pos_noise_range, rot_noise_range=rot_noise_range, seed=seed)
    
    # Generate the pose files with suffix
    file_a = data_dir / f"known_transform_A{suffix}.txt"
    file_b = data_dir / f"known_transform_B{suffix}.txt"
    
    generate_pose_file(As, file_a)
    generate_pose_file(Bs_noisy, file_b)
    
    print(f"Generated: {file_a} (noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")
    print(f"Generated: {file_b} (noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")


def generate_circular_trajectory_files(pos_noise_range=5.0, rot_noise_range=0.5, suffix=""):
    """Generate test files with a circular trajectory"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create a known transformation
    X_known = compose(
        euler_ZYX_to_R(np.deg2rad(45), 0, 0),
        np.array([100.0, 50.0, 0.0])
    )
    Y_known = np.eye(4)
    
    # Generate circular trajectory
    As = []
    radius = 200.0
    for i in range(8):
        angle = i * 2 * np.pi / 8
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = i * 10.0
        
        R_A = euler_ZYX_to_R(angle, 0, 0)
        t_A = np.array([x, y, z])
        A = compose(R_A, t_A)
        As.append(A)
    
    # Generate corresponding B poses
    Bs = [invert_T(Y_known) @ A @ X_known for A in As]
    
    # Add noise to B poses with specified noise level
    seed = 42 + int(pos_noise_range * 10)
    Bs_noisy = add_noise_to_poses(Bs, pos_noise_range=pos_noise_range, rot_noise_range=rot_noise_range, seed=seed)
    
    # Generate the pose files with suffix
    file_a = data_dir / f"circular_trajectory_A{suffix}.txt"
    file_b = data_dir / f"circular_trajectory_B{suffix}.txt"
    
    generate_pose_file(As, file_a)
    generate_pose_file(Bs_noisy, file_b)
    
    print(f"Generated: {file_a} (noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")
    print(f"Generated: {file_b} (noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")


def generate_helix_trajectory_files(pos_noise_range=5.0, rot_noise_range=0.5, suffix=""):
    """Generate test files with a 3D helix trajectory (1000 poses)"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create a known transformation
    X_known = compose(
        euler_ZYX_to_R(np.deg2rad(25), np.deg2rad(15), np.deg2rad(10)),
        np.array([75.0, 125.0, 50.0])
    )
    Y_known = np.eye(4)
    
    # Generate 3D helix trajectory with varying rotations
    As = []
    n_poses = 1000
    radius = 300.0
    height_scale = 0.5
    
    for i in range(n_poses):
        # Helix parameters
        t = i * 4 * np.pi / n_poses  # Multiple full rotations
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = i * height_scale
        
        # Varying rotations: follow the helix and add additional rotations
        rz = t  # Rotation around Z follows the helix
        ry = np.sin(t * 0.5) * np.pi / 6  # Pitch variation
        rx = np.cos(t * 0.3) * np.pi / 8  # Roll variation
        
        R_A = euler_ZYX_to_R(rz, ry, rx)
        t_A = np.array([x, y, z])
        A = compose(R_A, t_A)
        As.append(A)
    
    # Generate corresponding B poses
    Bs = [invert_T(Y_known) @ A @ X_known for A in As]
    
    # Add noise to B poses with specified noise level
    seed = 42 + int(pos_noise_range * 10)
    Bs_noisy = add_noise_to_poses(Bs, pos_noise_range=pos_noise_range, rot_noise_range=rot_noise_range, seed=seed)
    
    # Generate the pose files with suffix
    file_a = data_dir / f"helix_trajectory_A{suffix}.txt"
    file_b = data_dir / f"helix_trajectory_B{suffix}.txt"
    
    generate_pose_file(As, file_a)
    generate_pose_file(Bs_noisy, file_b)
    
    print(f"Generated: {file_a} ({n_poses} poses, noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")
    print(f"Generated: {file_b} ({n_poses} poses, noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")


def generate_complex_3d_path_files(pos_noise_range=5.0, rot_noise_range=0.5, suffix=""):
    """Generate test files with a complex 3D path with significant position and rotation variations (1000 poses)"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create a known transformation
    X_known = compose(
        euler_ZYX_to_R(np.deg2rad(30), np.deg2rad(20), np.deg2rad(15)),
        np.array([100.0, 200.0, 75.0])
    )
    Y_known = np.eye(4)
    
    # Generate complex 3D path
    As = []
    n_poses = 1000
    
    for i in range(n_poses):
        t = i * 2 * np.pi / n_poses
        
        # Complex position: combination of multiple frequencies
        x = 200.0 * np.cos(t) + 100.0 * np.cos(3 * t) + 50.0 * np.sin(5 * t)
        y = 200.0 * np.sin(t) + 100.0 * np.sin(3 * t) + 50.0 * np.cos(5 * t)
        z = 150.0 * np.sin(2 * t) + i * 0.3
        
        # Complex rotations: varying in all three axes
        rz = t + 0.5 * np.sin(2 * t)  # Z rotation with variation
        ry = 0.3 * np.cos(3 * t) + 0.2 * np.sin(4 * t)  # Y rotation
        rx = 0.4 * np.sin(2.5 * t) + 0.3 * np.cos(3.5 * t)  # X rotation
        
        R_A = euler_ZYX_to_R(rz, ry, rx)
        t_A = np.array([x, y, z])
        A = compose(R_A, t_A)
        As.append(A)
    
    # Generate corresponding B poses
    Bs = [invert_T(Y_known) @ A @ X_known for A in As]
    
    # Add noise to B poses with specified noise level
    seed = 42 + int(pos_noise_range * 10)
    Bs_noisy = add_noise_to_poses(Bs, pos_noise_range=pos_noise_range, rot_noise_range=rot_noise_range, seed=seed)
    
    # Generate the pose files with suffix
    file_a = data_dir / f"complex_3d_path_A{suffix}.txt"
    file_b = data_dir / f"complex_3d_path_B{suffix}.txt"
    
    generate_pose_file(As, file_a)
    generate_pose_file(Bs_noisy, file_b)
    
    print(f"Generated: {file_a} ({n_poses} poses, noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")
    print(f"Generated: {file_b} ({n_poses} poses, noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")


def generate_figure8_trajectory_files(pos_noise_range=5.0, rot_noise_range=0.5, suffix=""):
    """Generate test files with a figure-8 (lemniscate) trajectory with rotations (1000 poses)"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create a known transformation
    X_known = compose(
        euler_ZYX_to_R(np.deg2rad(35), np.deg2rad(10), np.deg2rad(20)),
        np.array([150.0, 100.0, 80.0])
    )
    Y_known = np.eye(4)
    
    # Generate figure-8 trajectory
    As = []
    n_poses = 1000
    scale = 250.0
    
    for i in range(n_poses):
        t = i * 4 * np.pi / n_poses  # Multiple figure-8s
        
        # Figure-8 (lemniscate) parametric equations
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)
        z = 100.0 * np.sin(t * 0.5) + i * 0.4
        
        # Rotations that follow the path orientation
        rz = t  # Follow the path
        ry = np.sin(t) * np.pi / 4  # Pitch follows the figure-8
        rx = np.cos(t * 2) * np.pi / 6  # Roll variation
        
        R_A = euler_ZYX_to_R(rz, ry, rx)
        t_A = np.array([x, y, z])
        A = compose(R_A, t_A)
        As.append(A)
    
    # Generate corresponding B poses
    Bs = [invert_T(Y_known) @ A @ X_known for A in As]
    
    # Add noise to B poses with specified noise level
    seed = 42 + int(pos_noise_range * 10)
    Bs_noisy = add_noise_to_poses(Bs, pos_noise_range=pos_noise_range, rot_noise_range=rot_noise_range, seed=seed)
    
    # Generate the pose files with suffix
    file_a = data_dir / f"figure8_trajectory_A{suffix}.txt"
    file_b = data_dir / f"figure8_trajectory_B{suffix}.txt"
    
    generate_pose_file(As, file_a)
    generate_pose_file(Bs_noisy, file_b)
    
    print(f"Generated: {file_a} ({n_poses} poses, noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")
    print(f"Generated: {file_b} ({n_poses} poses, noise: ±{pos_noise_range}mm, ±{rot_noise_range}°)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test data files with variadic noise levels")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.1, 1.0, 5.0],
                        help="Position noise levels in mm (default: 0.1 1.0 5.0)")
    parser.add_argument("--rot-noise", type=float, default=0.5,
                        help="Rotation noise in degrees (default: 0.5)")
    parser.add_argument("--all-levels", action="store_true",
                        help="Generate all test cases with all noise levels")
    parser.add_argument("--test-case", type=str, choices=["known_transform", "circular_trajectory", 
                        "helix_trajectory", "complex_3d_path", "figure8_trajectory", "all"],
                        default="all", help="Which test case to generate")
    
    args = parser.parse_args()
    
    noise_levels = args.noise_levels
    rot_noise = args.rot_noise
    
    print("Generating test data files with variadic noise levels...")
    print(f"Noise levels: {noise_levels} mm")
    print(f"Rotation noise: ±{rot_noise}°")
    print("="*80)
    
    test_cases = {
        "known_transform": generate_known_transformation_files,
        "circular_trajectory": generate_circular_trajectory_files,
        "helix_trajectory": generate_helix_trajectory_files,
        "complex_3d_path": generate_complex_3d_path_files,
        "figure8_trajectory": generate_figure8_trajectory_files,
    }
    
    if args.test_case == "all" or args.all_levels:
        # Generate all test cases with all noise levels
        for case_name, case_func in test_cases.items():
            print(f"\nGenerating {case_name} with multiple noise levels...")
            for noise in noise_levels:
                suffix = f"_{noise}mm"
                case_func(pos_noise_range=noise, rot_noise_range=rot_noise, suffix=suffix)
    else:
        # Generate specific test case with all noise levels
        if args.test_case in test_cases:
            case_func = test_cases[args.test_case]
            print(f"\nGenerating {args.test_case} with multiple noise levels...")
            for noise in noise_levels:
                suffix = f"_{noise}mm"
                case_func(pos_noise_range=noise, rot_noise_range=rot_noise, suffix=suffix)
    
    print("\n" + "="*80)
    print("Test data generation complete!")
    print("="*80)

