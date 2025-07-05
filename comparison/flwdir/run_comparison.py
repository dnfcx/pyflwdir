#!/usr/bin/env python3
"""
Comparison script to test Rust vs Python pyflwdir implementations.
"""

import numpy as np
import pyflwdir
import subprocess
import json
import tempfile
import os
import sys

def test_rust_implementation(d8_array, test_name):
    """Test the Rust implementation by calling the Rust binary."""
    # Create a temporary file to pass the D8 array
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Convert numpy array to list for JSON serialization
        data = {
            'test_name': test_name,
            'd8_array': d8_array.tolist(),
            'shape': d8_array.shape
        }
        json.dump(data, f)
        temp_file = f.name
    
    try:
        # Call the Rust binary (we'll create a simple one)
        result = subprocess.run([
            'cargo', 'run', '--manifest-path', '../pyflwdir-rs/Cargo.toml',
            '--bin', 'test_compare', '--', temp_file
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode != 0:
            print(f"Rust test failed: {result.stderr}")
            return None
        
        # Parse the result
        return json.loads(result.stdout)
    finally:
        os.unlink(temp_file)

def test_python_implementation(d8_array, test_name):
    """Test the Python implementation."""
    print(f"\n=== Testing {test_name} with Python ===")
    
    # Create FlwdirRaster
    flwdir = pyflwdir.from_array(d8_array, ftype='d8')
    
    # Get results
    results = {
        'test_name': test_name,
        'shape': flwdir.shape,
        'size': flwdir.size,
        'nnodes': flwdir.nnodes,
        'rank': flwdir.rank.flatten().tolist(),
        'n_upstream': flwdir.n_upstream.flatten().tolist(),
        'idxs_pit': flwdir.idxs_pit.tolist()
    }
    
    print(f"  Shape: {results['shape']}")
    print(f"  Size: {results['size']}")
    print(f"  nnodes: {results['nnodes']}")
    print(f"  Rank: {results['rank']}")
    print(f"  Upstream count: {results['n_upstream']}")
    print(f"  Pit indices: {results['idxs_pit']}")
    
    return results

def compare_results(python_results, rust_results):
    """Compare Python and Rust results."""
    if rust_results is None:
        print("❌ Rust test failed - cannot compare")
        return False
    
    print(f"\n=== Comparing {python_results['test_name']} ===")
    
    success = True
    
    # Compare basic properties
    for prop in ['shape', 'size', 'nnodes']:
        if python_results[prop] != rust_results[prop]:
            print(f"❌ {prop} mismatch: Python {python_results[prop]} vs Rust {rust_results[prop]}")
            success = False
        else:
            print(f"✓ {prop}: {python_results[prop]}")
    
    # Compare arrays
    for arr_name in ['rank', 'n_upstream', 'idxs_pit']:
        py_arr = python_results[arr_name]
        rust_arr = rust_results[arr_name]
        
        if py_arr != rust_arr:
            print(f"❌ {arr_name} mismatch:")
            print(f"  Python: {py_arr}")
            print(f"  Rust:   {rust_arr}")
            success = False
        else:
            print(f"✓ {arr_name}: {py_arr}")
    
    return success

def main():
    """Main comparison function."""
    print("Starting pyflwdir Rust vs Python comparison...")
    
    # Test cases
    test_cases = [
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
        ], dtype=np.uint8))
    ]
    
    all_success = True
    
    for test_name, d8_array in test_cases:
        # Test Python implementation
        python_results = test_python_implementation(d8_array, test_name)
        
        # Test Rust implementation
        rust_results = test_rust_implementation(d8_array, test_name)
        
        # Compare results
        success = compare_results(python_results, rust_results)
        all_success = all_success and success
    
    if all_success:
        print("\n🎉 All tests passed! Rust and Python implementations match exactly.")
    else:
        print("\n❌ Some tests failed. There are differences between implementations.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 