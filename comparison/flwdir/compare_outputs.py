#!/usr/bin/env python3
"""
Compare Rust JSON outputs with Python outputs.
"""

import json
import subprocess
import numpy as np
import pyflwdir

def get_rust_results():
    """Get results from Rust implementation."""
    result = subprocess.run(['cargo', 'run'], 
                          capture_output=True, text=True, cwd='.')
    
    if result.returncode != 0:
        print(f"Rust execution failed: {result.stderr}")
        return None
    
    # Parse the JSON output (should be a single JSON array)
    try:
        json_results = json.loads(result.stdout.strip())
        return json_results
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Output: {result.stdout}")
        return None

def get_python_results():
    """Get results from Python implementation."""
    test_cases = [
        # Original test cases
        ("Simple 2x2", np.array([[2, 4], [1, 0]], dtype=np.uint8)),
        ("4x4 Array", np.array([
            [1, 1, 2, 4],
            [1, 2, 4, 4],
            [64, 1, 2, 4],
            [64, 64, 1, 0]
        ], dtype=np.uint8)),
        ("Complex 3x3", np.array([
            [2, 4, 8],
            [1, 0, 16],
            [128, 64, 0]
        ], dtype=np.uint8)),
        
        # New larger test cases
        ("Watershed 8x8", create_watershed_8x8()),
        ("River Network 10x10", create_river_network_10x10()),
        ("Mountainous 12x12", create_mountainous_12x12()),
        ("Large Drainage 15x15", create_large_drainage_15x15()),
        
        # Very large test cases for performance testing
        ("Complex Watershed 50x50", create_complex_watershed_50x50()),
        ("Mega Drainage 100x100", create_mega_drainage_100x100()),
    ]
    
    results = []
    for test_name, d8_array in test_cases:
        print(f"  Processing {test_name} ({d8_array.shape})...")
        flwdir = pyflwdir.from_array(d8_array, ftype='d8')
        
        result = {
            "test_name": test_name,
            "shape": list(flwdir.shape),
            "size": flwdir.size,
            "nnodes": flwdir.nnodes,
            "rank": flwdir.rank.flatten().tolist(),
            "n_upstream": flwdir.n_upstream.flatten().tolist(),
            "idxs_pit": flwdir.idxs_pit.tolist()
        }
        results.append(result)
    
    return results

def create_watershed_8x8():
    """Create a watershed with flow converging to a central outlet."""
    return np.array([
        [4, 4, 4, 4, 8, 8, 8, 8],
        [2, 2, 4, 4, 8, 8, 16, 16],
        [2, 2, 2, 4, 8, 16, 16, 16],
        [1, 1, 2, 4, 8, 16, 32, 32],
        [1, 1, 1, 2, 4, 16, 32, 32],
        [64, 1, 1, 1, 0, 16, 32, 64],
        [64, 64, 1, 1, 1, 2, 4, 64],
        [64, 64, 64, 1, 1, 2, 4, 4],
    ], dtype=np.uint8)

def create_river_network_10x10():
    """Create a river network with multiple tributaries."""
    return np.array([
        [4, 4, 2, 4, 4, 8, 8, 8, 16, 16],
        [4, 2, 2, 4, 4, 8, 8, 16, 16, 16],
        [2, 2, 2, 2, 4, 8, 16, 16, 16, 32],
        [1, 1, 2, 2, 4, 8, 16, 16, 32, 32],
        [1, 1, 1, 2, 4, 8, 16, 32, 32, 32],
        [64, 1, 1, 1, 2, 4, 8, 16, 32, 64],
        [64, 64, 1, 1, 1, 2, 4, 8, 16, 64],
        [64, 64, 64, 1, 1, 1, 2, 4, 8, 64],
        [128, 64, 64, 64, 1, 1, 1, 2, 4, 0],
        [128, 128, 64, 64, 64, 1, 1, 1, 2, 4],
    ], dtype=np.uint8)

def create_mountainous_12x12():
    """Create mountainous terrain with multiple peaks and valleys."""
    return np.array([
        [4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32],
        [2, 4, 4, 8, 8, 16, 16, 16, 32, 32, 32, 64],
        [2, 2, 4, 4, 8, 16, 16, 32, 32, 32, 64, 64],
        [1, 2, 2, 4, 8, 8, 16, 32, 32, 64, 64, 64],
        [1, 1, 2, 4, 4, 8, 16, 16, 32, 64, 64, 128],
        [64, 1, 1, 2, 4, 8, 8, 16, 32, 32, 64, 128],
        [64, 64, 1, 1, 2, 4, 8, 16, 16, 32, 64, 128],
        [128, 64, 64, 1, 1, 2, 4, 8, 16, 32, 64, 128],
        [128, 128, 64, 64, 1, 1, 2, 4, 8, 16, 32, 0],
        [128, 128, 128, 64, 64, 1, 1, 2, 4, 8, 16, 32],
        [128, 128, 128, 128, 64, 64, 1, 1, 2, 4, 8, 16],
        [0, 128, 128, 128, 128, 64, 64, 1, 1, 2, 4, 8],
    ], dtype=np.uint8)

def create_large_drainage_15x15():
    """Create a large flat area with organized drainage patterns."""
    d8_data = np.zeros((15, 15), dtype=np.uint8)
    
    # Create a proper drainage pattern that flows toward the center outlet
    for i in range(15):
        for j in range(15):
            if i == 7 and j == 7:
                # Outlet at center
                flow_dir = 0
            elif i < 7 and j < 7:
                # Upper left quadrant - flow toward center
                flow_dir = 4 if i < j else 1  # Flow south or east
            elif i < 7 and j > 7:
                # Upper right quadrant - flow toward center
                flow_dir = 4 if i < 14 - j else 16  # Flow south or west
            elif i > 7 and j < 7:
                # Lower left quadrant - flow toward center
                flow_dir = 64 if 14 - i < j else 1  # Flow north or east
            elif i > 7 and j > 7:
                # Lower right quadrant - flow toward center
                flow_dir = 64 if 14 - i < 14 - j else 16  # Flow north or west
            elif i == 7 and j < 7:
                # Middle row, left side - flow east
                flow_dir = 1
            elif i == 7 and j > 7:
                # Middle row, right side - flow west
                flow_dir = 16
            elif i < 7 and j == 7:
                # Middle column, top side - flow south
                flow_dir = 4
            elif i > 7 and j == 7:
                # Middle column, bottom side - flow north
                flow_dir = 64
            else:
                # Default case
                flow_dir = 4
            
            d8_data[i, j] = flow_dir
    
    return d8_data

def create_complex_watershed_50x50():
    """Create a complex 50x50 watershed with multiple sub-basins."""
    d8_data = np.zeros((50, 50), dtype=np.uint8)
    
    for i in range(50):
        for j in range(50):
            if i == 45 and j == 25:
                # Main outlet near bottom center
                flow_dir = 0
            elif i < 10:
                # Top section - multiple ridges flowing down
                if j < 15:
                    flow_dir = 2 if (i + j) % 3 == 0 else 4  # SE or S
                elif j < 35:
                    flow_dir = 4  # S
                else:
                    flow_dir = 8 if (i + j) % 3 == 0 else 4  # SW or S
            elif i < 25:
                # Upper middle - converging flows
                if j < 20:
                    flow_dir = 1 if j < i - 5 else 2  # E or SE
                elif j < 30:
                    flow_dir = 4  # S
                else:
                    flow_dir = 16 if j > 55 - i else 8  # W or SW
            elif i < 40:
                # Lower middle - main channel formation
                if j < 10:
                    flow_dir = 1  # E
                elif j < 20:
                    flow_dir = 2 if i > 30 else 1  # SE or E
                elif j < 30:
                    flow_dir = 4  # S
                elif j < 40:
                    flow_dir = 8 if i > 30 else 16  # SW or W
                else:
                    flow_dir = 16  # W
            else:
                # Bottom section - final convergence
                if j < 25:
                    flow_dir = 1 if i == 45 and j > 20 else 64  # E or N
                elif j == 25:
                    flow_dir = 4 if i < 45 else 0  # S or pit
                else:
                    flow_dir = 16 if i == 45 and j < 30 else 64  # W or N
            
            d8_data[i, j] = flow_dir
    
    return d8_data

def create_mega_drainage_100x100():
    """Create a mega 100x100 drainage network with realistic flow patterns."""
    d8_data = np.zeros((100, 100), dtype=np.uint8)
    
    for i in range(100):
        for j in range(100):
            if i == 85 and j == 50:
                # Main outlet
                flow_dir = 0
            elif i < 20:
                # Northern highlands - multiple drainage divides
                sector = j // 20
                if sector == 0:
                    flow_dir = 4 if i < j//2 else 2  # S or SE
                elif sector == 1:
                    flow_dir = 4 if (i + j) % 4 < 2 else 2  # S or SE
                elif sector == 2:
                    flow_dir = 4  # S
                elif sector == 3:
                    flow_dir = 4 if (i + j) % 4 < 2 else 8  # S or SW
                else:
                    flow_dir = 4 if i < (100-j)//2 else 8  # S or SW
            elif i < 40:
                # Upper valleys - tributary formation
                dist_from_center = abs(j - 50) + abs(i - 30)
                if dist_from_center < 10:
                    flow_dir = 4  # Main channel - flow south
                elif j < 50:
                    flow_dir = 2 if i > 30 + (50 - j) // 3 else 1  # SE or E
                else:
                    flow_dir = 8 if i > 30 + (j - 50) // 3 else 16  # SW or W
            elif i < 70:
                # Middle reaches - major tributaries
                if j < 25:
                    flow_dir = 2 if i > 50 + j//2 else 1  # SE or E
                elif j < 40:
                    flow_dir = 2 if i > 55 else 1  # SE or E
                elif j < 60:
                    flow_dir = 4  # S - main stem
                elif j < 75:
                    flow_dir = 8 if i > 55 else 16  # SW or W
                else:
                    flow_dir = 8 if i > 50 + (100-j)//2 else 16  # SW or W
            else:
                # Lower reaches - final convergence
                dist_to_outlet = abs(j - 50) + abs(i - 85)
                if dist_to_outlet < 3:
                    if i < 85:
                        flow_dir = 4  # Toward outlet
                    elif j < 50:
                        flow_dir = 1
                    else:
                        flow_dir = 16
                elif j < 50:
                    flow_dir = 64 if i > 80 else 1  # N or E
                else:
                    flow_dir = 64 if i > 80 else 16  # N or W
            
            d8_data[i, j] = flow_dir
    
    return d8_data

def compare_results(rust_results, python_results):
    """Compare Rust and Python results."""
    if len(rust_results) != len(python_results):
        print(f"❌ Different number of test cases: Rust {len(rust_results)} vs Python {len(python_results)}")
        return False
    
    all_match = True
    
    for i, (rust, python) in enumerate(zip(rust_results, python_results)):
        test_name = rust.get("test_name", f"Test {i+1}")
        print(f"\n=== Comparing {test_name} ===")
        
        # Compare each field
        for field in ["shape", "size", "nnodes", "rank", "n_upstream", "idxs_pit"]:
            rust_val = rust.get(field)
            python_val = python.get(field)
            
            if rust_val != python_val:
                print(f"❌ {field} mismatch:")
                print(f"  Rust:   {rust_val}")
                print(f"  Python: {python_val}")
                all_match = False
            else:
                print(f"✓ {field}: matches (size: {len(rust_val) if isinstance(rust_val, list) else 'N/A'})")
    
    return all_match

def main():
    """Main comparison function."""
    print("Comparing Rust vs Python pyflwdir implementations...")
    print("Testing with larger, more realistic DEM-like inputs...")
    
    # Get results from both implementations
    print("\nGetting Rust results...")
    rust_results = get_rust_results()
    
    if rust_results is None:
        print("Failed to get Rust results")
        return 1
    
    print(f"Got {len(rust_results)} Rust results")
    
    print("\nGetting Python results...")
    python_results = get_python_results()
    print(f"Got {len(python_results)} Python results")
    
    # Compare results
    success = compare_results(rust_results, python_results)
    
    if success:
        print("\n🎉 All tests passed! Rust and Python implementations match exactly.")
        print("✅ Validation complete for all test cases including large DEM-like inputs!")
        return 0
    else:
        print("\n❌ Some tests failed. There are differences between implementations.")
        return 1

if __name__ == "__main__":
    exit(main()) 