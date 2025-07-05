use serde_json;
use ndarray::Array2;
use pyflwdir_rs::FlwdirRaster;
use std::time::Instant;

fn main() {
    // WARMUP: Run a small computation to eliminate cold start overhead
    let _ = test_case_internal("Warmup", &[&[1u8, 2u8], &[4u8, 0u8]], false); // Don't include timing
    
    let mut all_results = Vec::new();
    
    // Original test cases
    if let Some(result) = test_case("Simple 2x2", &[&[2u8, 4u8], &[1u8, 0u8]]) {
        all_results.push(result);
    }
    
    if let Some(result) = test_case("4x4 Array", &[
        &[1u8, 1u8, 2u8, 4u8],
        &[1u8, 2u8, 4u8, 4u8],
        &[64u8, 1u8, 2u8, 4u8],
        &[64u8, 64u8, 1u8, 0u8]
    ]) {
        all_results.push(result);
    }
    
    if let Some(result) = test_case("Complex 3x3", &[
        &[2u8, 4u8, 8u8],
        &[1u8, 0u8, 16u8],
        &[128u8, 64u8, 0u8]
    ]) {
        all_results.push(result);
    }
    
    // New larger test cases
    if let Some(result) = test_watershed_8x8() {
        all_results.push(result);
    }
    
    if let Some(result) = test_river_network_10x10() {
        all_results.push(result);
    }
    
    if let Some(result) = test_mountainous_12x12() {
        all_results.push(result);
    }
    
    if let Some(result) = test_large_drainage_15x15() {
        all_results.push(result);
    }
    
    // Medium test cases for performance testing (max 20x20)
    if let Some(result) = test_complex_watershed_18x18() {
        all_results.push(result);
    }
    
    if let Some(result) = test_mega_drainage_20x20() {
        all_results.push(result);
    }
    
    // Output all results as a single JSON array
    println!("{}", serde_json::to_string_pretty(&all_results).unwrap());
}

fn test_case(name: &str, d8_data: &[&[u8]]) -> Option<serde_json::Value> {
    test_case_internal(name, d8_data, true)
}

fn test_case_internal(name: &str, d8_data: &[&[u8]], include_timing: bool) -> Option<serde_json::Value> {
    // Convert to ndarray
    let rows = d8_data.len();
    let cols = d8_data[0].len();
    let mut flat_data = Vec::new();
    
    for row in d8_data {
        for &val in *row {
            flat_data.push(val);
        }
    }
    
    let d8_array = Array2::from_shape_vec((rows, cols), flat_data).unwrap();
    
    // Run the computation once to get the actual results
    let mut flwdir = FlwdirRaster::from_array(d8_array.view());
    let shape = flwdir.shape;
    let size = shape.0 * shape.1;
    let rank = flwdir.rank().clone();
    let nnodes = rank.iter().filter(|&&r| r >= 0).count();
    let upstream_count = flwdir.upstream_counts.clone();
    let pit_indices = flwdir.pit_indices.clone();
    
    let elapsed = if include_timing {
        // Run timing tests 30 times and take the median
        let mut timings = Vec::new();
        
        for _ in 0..30 {
            let start_time = Instant::now();
            
            // Create FlwdirRaster and compute results
            let mut flwdir_timing = FlwdirRaster::from_array(d8_array.view());
            let _ = flwdir_timing.rank(); // Ensure computation is done
            let _ = flwdir_timing.upstream_counts; // Ensure computation is done
            let _ = flwdir_timing.pit_indices; // Ensure computation is done
            
            let elapsed_time = start_time.elapsed().as_secs_f64();
            timings.push(elapsed_time);
        }
        
        // Sort timings and take median
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        timings[timings.len() / 2] // Median
    } else {
        0.0 // Warmup run
    };
    
    if !include_timing {
        return None; // Don't return results for warmup
    }
    
    // Convert ndarray results to vectors for JSON serialization
    let rank_vec: Vec<i32> = rank.to_vec();
    let upstream_vec: Vec<i8> = upstream_count.to_vec();
    let pit_vec: Vec<usize> = pit_indices.to_vec();
    
    // Create JSON output
    let json_output = serde_json::json!({
        "test_name": name,
        "shape": [shape.0, shape.1],
        "size": size,
        "nnodes": nnodes,
        "rank": rank_vec,
        "n_upstream": upstream_vec,
        "idxs_pit": pit_vec,
        "timing_seconds": elapsed
    });
    
    Some(json_output)
}

fn test_watershed_8x8() -> Option<serde_json::Value> {
    // Simulate a watershed with flow converging to a central outlet
    let d8_data = [
        [4u8, 4u8, 4u8, 4u8, 8u8, 8u8, 8u8, 8u8],
        [2u8, 2u8, 4u8, 4u8, 8u8, 8u8, 16u8, 16u8],
        [2u8, 2u8, 2u8, 4u8, 8u8, 16u8, 16u8, 16u8],
        [1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 32u8, 32u8],
        [1u8, 1u8, 1u8, 2u8, 4u8, 16u8, 32u8, 32u8],
        [64u8, 1u8, 1u8, 1u8, 0u8, 16u8, 32u8, 64u8],
        [64u8, 64u8, 1u8, 1u8, 1u8, 2u8, 4u8, 64u8],
        [64u8, 64u8, 64u8, 1u8, 1u8, 2u8, 4u8, 4u8],
    ];
    
    test_case_from_array("Watershed 8x8", &d8_data)
}

fn test_river_network_10x10() -> Option<serde_json::Value> {
    // Simulate a river network with multiple tributaries
    let d8_data = [
        [4u8, 4u8, 2u8, 4u8, 4u8, 8u8, 8u8, 8u8, 16u8, 16u8],
        [4u8, 2u8, 2u8, 4u8, 4u8, 8u8, 8u8, 16u8, 16u8, 16u8],
        [2u8, 2u8, 2u8, 2u8, 4u8, 8u8, 16u8, 16u8, 16u8, 32u8],
        [1u8, 1u8, 2u8, 2u8, 4u8, 8u8, 16u8, 16u8, 32u8, 32u8],
        [1u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 32u8, 32u8, 32u8],
        [64u8, 1u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 32u8, 64u8],
        [64u8, 64u8, 1u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 64u8],
        [64u8, 64u8, 64u8, 1u8, 1u8, 1u8, 2u8, 4u8, 8u8, 64u8],
        [128u8, 64u8, 64u8, 64u8, 1u8, 1u8, 1u8, 2u8, 4u8, 0u8],
        [128u8, 128u8, 64u8, 64u8, 64u8, 1u8, 1u8, 1u8, 2u8, 4u8],
    ];
    
    test_case_from_array("River Network 10x10", &d8_data)
}

fn test_mountainous_12x12() -> Option<serde_json::Value> {
    // Simulate mountainous terrain with multiple peaks and valleys
    let d8_data = [
        [4u8, 4u8, 4u8, 8u8, 8u8, 8u8, 16u8, 16u8, 16u8, 32u8, 32u8, 32u8],
        [2u8, 4u8, 4u8, 8u8, 8u8, 16u8, 16u8, 16u8, 32u8, 32u8, 32u8, 64u8],
        [2u8, 2u8, 4u8, 4u8, 8u8, 16u8, 16u8, 32u8, 32u8, 32u8, 64u8, 64u8],
        [1u8, 2u8, 2u8, 4u8, 8u8, 8u8, 16u8, 32u8, 32u8, 64u8, 64u8, 64u8],
        [1u8, 1u8, 2u8, 4u8, 4u8, 8u8, 16u8, 16u8, 32u8, 64u8, 64u8, 128u8],
        [64u8, 1u8, 1u8, 2u8, 4u8, 8u8, 8u8, 16u8, 32u8, 32u8, 64u8, 128u8],
        [64u8, 64u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 16u8, 32u8, 64u8, 128u8],
        [128u8, 64u8, 64u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 32u8, 64u8, 128u8],
        [128u8, 128u8, 64u8, 64u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 32u8, 0u8],
        [128u8, 128u8, 128u8, 64u8, 64u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8, 32u8],
        [128u8, 128u8, 128u8, 128u8, 64u8, 64u8, 1u8, 1u8, 2u8, 4u8, 8u8, 16u8],
        [0u8, 128u8, 128u8, 128u8, 128u8, 64u8, 64u8, 1u8, 1u8, 2u8, 4u8, 8u8],
    ];
    
    test_case_from_array("Mountainous 12x12", &d8_data)
}

fn test_large_drainage_15x15() -> Option<serde_json::Value> {
    // Create a large flat area with organized drainage patterns
    let mut d8_data = [[0u8; 15]; 15];
    
    // Create a proper drainage pattern that flows toward the center outlet
    for i in 0..15 {
        for j in 0..15 {
            let flow_dir = if i == 7 && j == 7 {
                // Outlet at center
                0
            } else if i < 7 && j < 7 {
                // Upper left quadrant - flow toward center
                if i < j { 4 } else { 1 }  // Flow south or east
            } else if i < 7 && j > 7 {
                // Upper right quadrant - flow toward center
                if i < 14 - j { 4 } else { 16 }  // Flow south or west
            } else if i > 7 && j < 7 {
                // Lower left quadrant - flow toward center
                if 14 - i < j { 64 } else { 1 }  // Flow north or east
            } else if i > 7 && j > 7 {
                // Lower right quadrant - flow toward center
                if 14 - i < 14 - j { 64 } else { 16 }  // Flow north or west
            } else if i == 7 && j < 7 {
                // Middle row, left side - flow east
                1
            } else if i == 7 && j > 7 {
                // Middle row, right side - flow west
                16
            } else if i < 7 && j == 7 {
                // Middle column, top side - flow south
                4
            } else if i > 7 && j == 7 {
                // Middle column, bottom side - flow north
                64
            } else {
                // Default case
                4
            };
            
            d8_data[i][j] = flow_dir;
        }
    }
    
    test_case_from_array("Large Drainage 15x15", &d8_data)
}

fn test_case_from_array<const R: usize, const C: usize>(name: &str, d8_data: &[[u8; C]; R]) -> Option<serde_json::Value> {
    test_case_from_array_internal(name, d8_data, true)
}

fn test_case_from_array_internal<const R: usize, const C: usize>(name: &str, d8_data: &[[u8; C]; R], include_timing: bool) -> Option<serde_json::Value> {
    // Convert to ndarray
    let mut flat_data = Vec::new();
    
    for row in d8_data {
        for &val in row {
            flat_data.push(val);
        }
    }
    
    let d8_array = Array2::from_shape_vec((R, C), flat_data).unwrap();
    
    // Run the computation once to get the actual results
    let mut flwdir = FlwdirRaster::from_array(d8_array.view());
    let shape = flwdir.shape;
    let size = shape.0 * shape.1;
    let rank = flwdir.rank().clone();
    let nnodes = rank.iter().filter(|&&r| r >= 0).count();
    let upstream_count = flwdir.upstream_counts.clone();
    let pit_indices = flwdir.pit_indices.clone();
    
    let elapsed = if include_timing {
        // Run timing tests 30 times and take the median
        let mut timings = Vec::new();
        
        for _ in 0..30 {
            let start_time = Instant::now();
            
            // Create FlwdirRaster and compute results
            let mut flwdir_timing = FlwdirRaster::from_array(d8_array.view());
            let _ = flwdir_timing.rank(); // Ensure computation is done
            let _ = flwdir_timing.upstream_counts; // Ensure computation is done
            let _ = flwdir_timing.pit_indices; // Ensure computation is done
            
            let elapsed_time = start_time.elapsed().as_secs_f64();
            timings.push(elapsed_time);
        }
        
        // Sort timings and take median
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        timings[timings.len() / 2] // Median
    } else {
        0.0 // Warmup run
    };
    
    if !include_timing {
        return None; // Don't return results for warmup
    }
    
    // Convert ndarray results to vectors for JSON serialization
    let rank_vec: Vec<i32> = rank.to_vec();
    let upstream_vec: Vec<i8> = upstream_count.to_vec();
    let pit_vec: Vec<usize> = pit_indices.to_vec();
    
    // Create JSON output
    let json_output = serde_json::json!({
        "test_name": name,
        "shape": [shape.0, shape.1],
        "size": size,
        "nnodes": nnodes,
        "rank": rank_vec,
        "n_upstream": upstream_vec,
        "idxs_pit": pit_vec,
        "timing_seconds": elapsed
    });
    
    Some(json_output)
}

fn test_complex_watershed_18x18() -> Option<serde_json::Value> {
    // Create a complex 18x18 watershed with multiple sub-basins
    let mut d8_data = [[0u8; 18]; 18];
    
    for i in 0..18 {
        for j in 0..18 {
            let flow_dir = if i == 15 && j == 9 {
                // Main outlet near bottom center
                0
            } else if i < 3 {
                // Top section - multiple ridges flowing down
                if j < 6 {
                    if (i + j) % 3 == 0 { 2 } else { 4 }  // SE or S
                } else if j < 12 {
                    4  // S
                } else {
                    if (i + j) % 3 == 0 { 8 } else { 4 }  // SW or S
                }
            } else if i < 9 {
                // Upper middle - converging flows
                if j < 6 {
                    if j < i - 2 { 1 } else { 2 }  // E or SE
                } else if j < 12 {
                    4  // S
                } else {
                    if j > 15 - i { 16 } else { 8 }  // W or SW
                }
            } else if i < 15 {
                // Lower middle - main channel formation
                if j < 3 {
                    1  // E
                } else if j < 6 {
                    if i > 12 { 2 } else { 1 }  // SE or E
                } else if j < 12 {
                    4  // S
                } else if j < 15 {
                    if i > 12 { 8 } else { 16 }  // SW or W
                } else {
                    16  // W
                }
            } else {
                // Bottom section - final convergence
                if j < 9 {
                    if i == 15 && j > 6 { 1 } else { 64 }  // E or N
                } else if j == 9 {
                    if i < 15 { 4 } else { 0 }  // S or pit
                } else {
                    if i == 15 && j < 12 { 16 } else { 64 }  // W or N
                }
            };
            
            d8_data[i][j] = flow_dir;
        }
    }
    
    test_case_from_array("Complex Watershed 18x18", &d8_data)
}

fn test_mega_drainage_20x20() -> Option<serde_json::Value> {
    // Create a mega 20x20 drainage network with realistic flow patterns
    let mut d8_data = [[0u8; 20]; 20];
    
    for i in 0..20 {
        for j in 0..20 {
            let flow_dir = if i == 15 && j == 10 {
                // Main outlet
                0
            } else if i < 5 {
                // Northern highlands - multiple drainage divides
                let sector = j / 5;
                match sector {
                    0 => if i < j/3 { 4 } else { 2 },  // S or SE
                    1 => if (i + j) % 4 < 2 { 4 } else { 2 },  // S or SE
                    2 => 4,  // S
                    3 => if (i + j) % 4 < 2 { 4 } else { 8 },  // S or SW
                    _ => if i < (20-j)/3 { 4 } else { 8 },  // S or SW
                }
            } else if i < 10 {
                // Upper valleys - tributary formation
                let dist_from_center = ((j as i32 - 10).abs() + (i as i32 - 7).abs()) as usize;
                if dist_from_center < 3 {
                    4  // Main channel - flow south
                } else if j < 10 {
                    if i > 7 + (10 - j) / 3 { 2 } else { 1 }  // SE or E
                } else {
                    if i > 7 + (j - 10) / 3 { 8 } else { 16 }  // SW or W
                }
            } else if i < 15 {
                // Middle reaches - major tributaries
                if j < 5 {
                    if i > 12 + j/3 { 2 } else { 1 }  // SE or E
                } else if j < 10 {
                    if i > 12 { 2 } else { 1 }  // SE or E
                } else if j < 15 {
                    4  // S - main stem
                } else if j < 17 {
                    if i > 12 { 8 } else { 16 }  // SW or W
                } else {
                    if i > 12 + (20-j)/3 { 8 } else { 16 }  // SW or W
                }
            } else {
                // Lower reaches - final convergence
                let dist_to_outlet = ((j as i32 - 10).abs() + (i as i32 - 15).abs()) as usize;
                if dist_to_outlet < 3 {
                    if i < 15 { 4 } else if j < 10 { 1 } else { 16 }  // Toward outlet
                } else if j < 10 {
                    if i > 13 { 64 } else { 1 }  // N or E
                } else {
                    if i > 13 { 64 } else { 16 }  // N or W
                }
            };
            
            d8_data[i][j] = flow_dir;
        }
    }
    
    test_case_from_array("Mega Drainage 20x20", &d8_data)
} 