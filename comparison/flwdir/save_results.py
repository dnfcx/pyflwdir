#!/usr/bin/env python3
"""
Save benchmark results as JSON files for visualization
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_comparison import get_rust_results_with_timing, get_python_results_with_timing

def main():
    print("🔬 Saving benchmark results as JSON files...")
    
    # Get results from both implementations
    print("📊 Running Rust implementation...")
    rust_results, rust_timing = get_rust_results_with_timing()
    
    if rust_results is None:
        print("❌ Failed to get Rust results")
        return 1
    
    print(f"✅ Got {len(rust_results)} Rust results in {rust_timing['total_time']:.3f}s")
    
    print("🐍 Running Python implementation...")
    python_results, python_timing = get_python_results_with_timing()
    print(f"✅ Got {len(python_results)} Python results in {python_timing['total']['total_time']:.3f}s")
    
    # Add timing information to each result
    print("⏱️  Adding timing information...")
    
    # Add average timing to Rust results (divide total time by number of tests)
    avg_rust_time = rust_timing['total_time'] / len(rust_results)
    for result in rust_results:
        result['timing_seconds'] = avg_rust_time
    
    # Add individual timing to Python results
    for result in python_results:
        test_name = result['test_name']
        if test_name in python_timing:
            result['timing_seconds'] = python_timing[test_name]['time']
        else:
            # Fallback to average if individual timing not found
            result['timing_seconds'] = python_timing['total']['total_time'] / len(python_results)
    
    # Save results as JSON files
    print("💾 Saving results...")
    
    with open('rust_results.json', 'w') as f:
        json.dump(rust_results, f, indent=2)
    print("✅ Saved rust_results.json")
    
    with open('python_results.json', 'w') as f:
        json.dump(python_results, f, indent=2)
    print("✅ Saved python_results.json")
    
    # Print timing summary
    print("\n📊 Timing Summary:")
    print(f"Rust average: {avg_rust_time:.6f}s per test")
    print(f"Python total: {python_timing['total']['total_time']:.6f}s")
    print(f"Overall speedup: {python_timing['total']['total_time'] / rust_timing['total_time']:.1f}x")
    
    print("🎉 Results saved successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 