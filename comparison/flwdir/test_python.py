#!/usr/bin/env python3
"""Test Python pyflwdir implementation to understand expected outputs."""

import numpy as np
import pyflwdir

def test_simple_2x2():
    print("=== Testing Simple 2x2 D8 Array ===")
    
    # Create test data: 2=SE, 4=S, 1=E, 0=PIT
    d8 = np.array([[2, 4], [1, 0]], dtype=np.uint8)
    print(f"D8 array:\n{d8}")
    
    # Create FlwdirRaster
    flw = pyflwdir.from_array(d8, ftype="d8")
    
    print(f"Shape: {flw.shape}")
    print(f"Size: {flw.size}")
    print(f"nnodes: {flw.nnodes}")
    
    # Test rank
    rank = flw.rank
    print(f"Rank: {rank}")
    print(f"Rank flattened: {rank.flatten()}")
    
    # Test upstream count (n_upstream)
    upstream = flw.n_upstream
    print(f"n_upstream: {upstream}")
    print(f"n_upstream flattened: {upstream.flatten()}")
    
    # Test pit indices (idxs_pit)
    pits = flw.idxs_pit
    print(f"idxs_pit: {pits}")
    
    return flw

def test_4x4_array():
    print("\n=== Testing 4x4 D8 Array ===")
    
    # Create a more complex 4x4 flow pattern
    d8 = np.array([
        [1, 1, 2, 4],   # E, E, SE, S
        [1, 2, 4, 4],   # E, SE, S, S
        [64, 1, 2, 4],  # N, E, SE, S
        [64, 64, 1, 0]  # N, N, E, PIT
    ], dtype=np.uint8)
    print(f"D8 array:\n{d8}")
    
    # Create FlwdirRaster
    flw = pyflwdir.from_array(d8, ftype="d8")
    
    print(f"Shape: {flw.shape}")
    print(f"Size: {flw.size}")
    print(f"nnodes: {flw.nnodes}")
    
    # Test rank
    rank = flw.rank
    print(f"Rank: {rank}")
    print(f"Rank flattened: {rank.flatten()}")
    
    # Test upstream count (n_upstream)
    upstream = flw.n_upstream
    print(f"n_upstream: {upstream}")
    print(f"n_upstream flattened: {upstream.flatten()}")
    
    # Test pit indices (idxs_pit)
    pits = flw.idxs_pit
    print(f"idxs_pit: {pits}")
    
    return flw

def test_complex_flow():
    print("\n=== Testing Complex Flow Pattern ===")
    
    # Create a 3x3 array with multiple pits and complex flow
    d8 = np.array([
        [2, 4, 8],    # SE, S, SW
        [1, 0, 16],   # E, PIT, W
        [128, 64, 0]  # NE, N, PIT
    ], dtype=np.uint8)
    print(f"D8 array:\n{d8}")
    
    # Create FlwdirRaster
    flw = pyflwdir.from_array(d8, ftype="d8")
    
    print(f"Shape: {flw.shape}")
    print(f"Size: {flw.size}")
    print(f"nnodes: {flw.nnodes}")
    
    # Test rank
    rank = flw.rank
    print(f"Rank: {rank}")
    print(f"Rank flattened: {rank.flatten()}")
    
    # Test upstream count (n_upstream)
    upstream = flw.n_upstream
    print(f"n_upstream: {upstream}")
    print(f"n_upstream flattened: {upstream.flatten()}")
    
    # Test pit indices (idxs_pit)
    pits = flw.idxs_pit
    print(f"idxs_pit: {pits}")
    
    return flw

if __name__ == "__main__":
    print("Testing Python pyflwdir implementation...")
    
    try:
        test_simple_2x2()
        test_4x4_array()
        test_complex_flow()
        print("\nAll Python tests completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 