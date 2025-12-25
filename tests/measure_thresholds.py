"""
Script to measure thresholds for each test case and noise level combination.

This script runs all algorithms on all test data files and collects error statistics
to determine appropriate regression thresholds.
"""
import numpy as np
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from functions_call import load_inputs, run_method

# Test cases and their noise levels
TEST_CASES = {
    "known_transform": [0.1, 1.0, 5.0],
    "circular_trajectory": [0.1, 1.0, 5.0],
    "helix_trajectory": [0.1, 1.0, 5.0],
    "complex_3d_path": [0.1, 1.0, 5.0],
    "figure8_trajectory": [0.1, 1.0, 5.0],
}

ALGORITHMS = ["tsai-lenz", "park-martin", "daniilidis", "li-wang-wu", "shah"]

TEST_DATA_DIR = Path(__file__).parent / "data"


def measure_thresholds():
    """Measure thresholds for all test case and noise level combinations"""
    results = {}
    
    for test_case, noise_levels in TEST_CASES.items():
        results[test_case] = {}
        
        for noise_level in noise_levels:
            suffix = f"_{noise_level}mm"
            file_a = TEST_DATA_DIR / f"{test_case}_A{suffix}.txt"
            file_b = TEST_DATA_DIR / f"{test_case}_B{suffix}.txt"
            
            if not file_a.exists() or not file_b.exists():
                print(f"Warning: Files not found for {test_case} with noise {noise_level}mm")
                continue
            
            print(f"\n{'='*80}")
            print(f"Testing: {test_case} with noise ±{noise_level}mm")
            print(f"{'='*80}")
            
            # Load test data
            try:
                As, Bs = load_inputs(str(file_a), str(file_b))
            except Exception as e:
                print(f"Error loading files: {e}")
                continue
            
            results[test_case][noise_level] = {}
            
            for algorithm in ALGORITHMS:
                try:
                    _, _, t_stats, r_stats = run_method(algorithm, As, Bs)
                    
                    results[test_case][noise_level][algorithm] = {
                        "translation": {
                            "mean": float(t_stats["mean"]),
                            "median": float(t_stats["median"]),
                            "rmse": float(t_stats["rmse"]),
                            "p95": float(t_stats["p95"]),
                            "max": float(t_stats["max"]),
                        },
                        "rotation": {
                            "mean": float(r_stats["mean"]),
                            "median": float(r_stats["median"]),
                            "rmse": float(r_stats["rmse"]),
                            "p95": float(r_stats["p95"]),
                            "max": float(r_stats["max"]),
                        }
                    }
                    
                    print(f"{algorithm:15s} | T_mean: {t_stats['mean']:8.3f} mm | "
                          f"R_mean: {r_stats['mean']:8.3f} deg")
                    
                except Exception as e:
                    results[test_case][noise_level][algorithm] = {
                        "translation": {"mean": None, "error": str(e)},
                        "rotation": {"mean": None, "error": str(e)}
                    }
                    print(f"{algorithm:15s} | ERROR: {e}")
    
    return results


def calculate_thresholds(results):
    """Calculate regression thresholds based on measured results"""
    thresholds = {}
    
    for test_case, noise_data in results.items():
        thresholds[test_case] = {}
        
        for noise_level, algorithm_data in noise_data.items():
            noise_key = f"{noise_level}mm"
            thresholds[test_case][noise_key] = {}
            
            for algorithm, stats in algorithm_data.items():
                if stats.get("translation", {}).get("mean") is None:
                    # Algorithm failed, set high thresholds
                    thresholds[test_case][noise_key][algorithm] = {
                        "translation": 1e6,
                        "rotation": 1e6
                    }
                else:
                    # Set threshold as 1.5x the mean error (with some margin)
                    t_mean = stats["translation"]["mean"]
                    r_mean = stats["rotation"]["mean"]
                    
                    # Add 20% margin above the mean
                    thresholds[test_case][noise_key][algorithm] = {
                        "translation": max(t_mean * 1.2, t_mean + 1.0),  # At least 1mm above
                        "rotation": max(r_mean * 1.2, r_mean + 0.1)      # At least 0.1deg above
                    }
    
    return thresholds


def print_summary(results, thresholds):
    """Print a summary of results and thresholds"""
    print("\n" + "="*80)
    print("SUMMARY OF THRESHOLDS")
    print("="*80)
    
    for test_case, noise_data in thresholds.items():
        print(f"\n{test_case.upper()}:")
        print("-" * 80)
        
        for noise_key, algorithm_data in noise_data.items():
            print(f"\n  Noise level: {noise_key}")
            print(f"  {'Algorithm':<15} {'Translation (mm)':<20} {'Rotation (deg)':<20}")
            print(f"  {'-'*15} {'-'*20} {'-'*20}")
            
            for algorithm, thresh in algorithm_data.items():
                t_thresh = thresh["translation"]
                r_thresh = thresh["rotation"]
                
                # Get actual mean error for comparison
                noise_level = float(noise_key.replace("mm", ""))
                actual = results[test_case].get(noise_level, {}).get(algorithm, {})
                t_actual = actual.get("translation", {}).get("mean", "N/A")
                r_actual = actual.get("rotation", {}).get("mean", "N/A")
                
                if isinstance(t_actual, (int, float)) and isinstance(r_actual, (int, float)):
                    print(f"  {algorithm:<15} {t_thresh:>8.2f} (actual: {t_actual:>6.3f})  "
                          f"{r_thresh:>8.2f} (actual: {r_actual:>6.3f})")
                else:
                    print(f"  {algorithm:<15} {t_thresh:>8.2f} (FAILED)      "
                          f"{r_thresh:>8.2f} (FAILED)")


if __name__ == "__main__":
    print("Measuring thresholds for all test cases and noise levels...")
    print("This may take a few minutes...")
    
    results = measure_thresholds()
    thresholds = calculate_thresholds(results)
    
    # Print summary
    print_summary(results, thresholds)
    
    # Save results to JSON file
    output_file = Path(__file__).parent / "threshold_measurements.json"
    with open(output_file, 'w') as f:
        json.dump({
            "results": results,
            "thresholds": thresholds
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80)


