#!/usr/bin/env python3
"""Explore Python pyflwdir API to understand available methods."""

import numpy as np
import pyflwdir

def explore_api():
    print("=== Exploring Python pyflwdir API ===")
    
    # Create test data
    d8 = np.array([[2, 4], [1, 0]], dtype=np.uint8)
    print(f"D8 array:\n{d8}")
    
    # Create FlwdirRaster
    flw = pyflwdir.from_array(d8, ftype="d8")
    
    print(f"\nFlwdirRaster type: {type(flw)}")
    print(f"Available attributes and methods:")
    
    # Get all attributes and methods
    attrs = [attr for attr in dir(flw) if not attr.startswith('_')]
    for attr in sorted(attrs):
        attr_obj = getattr(flw, attr)
        if callable(attr_obj):
            print(f"  {attr}() - method")
        else:
            print(f"  {attr} - attribute")
    
    print(f"\nTesting basic properties:")
    print(f"Shape: {flw.shape}")
    print(f"Size: {flw.size}")
    
    # Test rank
    print(f"\nRank:")
    try:
        rank = flw.rank
        print(f"  Rank (property): {rank}")
        print(f"  Rank shape: {rank.shape}")
        print(f"  Rank dtype: {rank.dtype}")
        print(f"  Rank flattened: {rank.flatten()}")
    except Exception as e:
        print(f"  Error getting rank: {e}")
    
    # Test nnodes
    print(f"\nnnodes:")
    try:
        nnodes = flw.nnodes
        print(f"  nnodes (property): {nnodes}")
    except Exception as e:
        print(f"  Error getting nnodes: {e}")
    
    # Test pit_indices
    print(f"\npit_indices:")
    try:
        pits = flw.pit_indices
        print(f"  pit_indices (property): {pits}")
        print(f"  pit_indices dtype: {pits.dtype}")
    except Exception as e:
        print(f"  Error getting pit_indices: {e}")
    
    # Test upstream methods
    print(f"\nUpstream methods:")
    try:
        # Check if there's an upstream method
        if hasattr(flw, 'upstream'):
            upstream = flw.upstream()
            print(f"  upstream(): {upstream}")
        else:
            print("  No upstream() method")
    except Exception as e:
        print(f"  Error calling upstream(): {e}")
    
    # Check for other upstream-related methods
    upstream_methods = [attr for attr in attrs if 'upstream' in attr.lower()]
    print(f"  Methods with 'upstream' in name: {upstream_methods}")
    
    # Test each upstream method
    for method in upstream_methods:
        try:
            result = getattr(flw, method)
            if callable(result):
                print(f"  {method}() - trying to call...")
                try:
                    res = result()
                    print(f"    Result: {res}")
                except Exception as e:
                    print(f"    Error calling {method}(): {e}")
            else:
                print(f"  {method} - property: {result}")
        except Exception as e:
            print(f"  Error with {method}: {e}")

if __name__ == "__main__":
    explore_api() 