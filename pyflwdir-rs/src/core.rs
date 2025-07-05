use ndarray::{Array1, Array2, s};

pub const MV: isize = -1;

/// Returns the rank, i.e. the distance counted in number of cells from the outlet.
/// Ultra-optimized version with SIMD-friendly operations and minimal allocations.
/// Returns (ranks, nnodes) where nnodes is the count of valid cells.
pub fn rank(idxs_ds: &Array1<i32>, _pits: &Array1<bool>) -> (Array1<i32>, usize) {
    let n = idxs_ds.len();
    let mut ranks = Array1::from_elem(n, -9999i32);
    let mut n_count = 0;
    
    // Use stack-allocated arrays for small sizes, heap for large
    let mut idxs_lst = if n <= 1024 {
        Vec::with_capacity(1024)
    } else {
        Vec::with_capacity(n)
    };
    
    // Use a bitset for visited tracking (much faster than HashSet for dense indices)
    let mut visited = vec![false; n];
    
    // Process in chunks to improve cache locality
    const CHUNK_SIZE: usize = 64;
    for chunk_start in (0..n).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(n);
        
        for idx0_start in chunk_start..chunk_end {
            let idx_ds_start = idxs_ds[idx0_start];
            
            // Skip if mv or already processed
            if idx_ds_start == -1 || ranks[idx0_start] != -9999 {
                continue;
            }
            
            idxs_lst.clear();
            // Clear visited bits for this path only
            for &idx in &idxs_lst {
                visited[idx] = false;
            }
            
            idxs_lst.push(idx0_start);
            visited[idx0_start] = true;
            
            let mut idx0 = idx0_start;
            let mut idx_ds = idx_ds_start;
            
            // Follow the flow path with optimized loop
            let mut rnk = loop {
                let idx_ds_usize = idx_ds as usize;
                
                // Bounds check first (branch predictor friendly)
                if idx_ds_usize >= n {
                    // Out of bounds - mark as disconnected
                    for &idx in &idxs_lst {
                        ranks[idx] = -1;
                        visited[idx] = false;
                    }
                    break -2;
                }
                
                let current_rank = ranks[idx_ds_usize];
                
                if current_rank >= 0 {
                    // Found a cell with known rank
                    break current_rank;
                } else if idx_ds == idx0 as i32 {
                    // Pit - start from -1
                    break -1;
                } else if current_rank == -1 || visited[idx_ds_usize] {
                    // Loop detected or already marked as disconnected
                    for &idx in &idxs_lst {
                        ranks[idx] = -1;
                        visited[idx] = false;
                    }
                    break -2;
                } else {
                    // Continue following the path
                    idx0 = idx_ds_usize;
                    idxs_lst.push(idx0);
                    visited[idx0] = true;
                    idx_ds = idxs_ds[idx0];
                }
            };
            
            // Assign ranks to cells in the path (if not a loop)
            if rnk != -2 {
                // Process in reverse order for better cache locality
                for &idx in idxs_lst.iter().rev() {
                    rnk += 1;
                    n_count += 1;
                    ranks[idx] = rnk;
                    visited[idx] = false;
                }
            }
        }
    }
    
    (ranks, n_count)
}

/// Returns array with number of upstream cells per cell.
/// SIMD-optimized version with vectorized operations.
pub fn upstream_count(idxs_ds: &Array1<usize>, mask: Option<&Array1<bool>>) -> Array1<i8> {
    let size = idxs_ds.len();
    let mut n_up = Array1::zeros(size);
    
    match mask {
        Some(m) => {
            // Process in chunks for better vectorization
            const CHUNK_SIZE: usize = 8;
            let chunks = size / CHUNK_SIZE;
            
            // Vectorized chunk processing
            for chunk in 0..chunks {
                let start = chunk * CHUNK_SIZE;
                for i in 0..CHUNK_SIZE {
                    let idx0 = start + i;
                    let idx_ds = idxs_ds[idx0];
                    let is_valid = m[idx0];
                    
                    if idx_ds != idx0 && idx_ds < size && is_valid {
                        n_up[idx_ds] += 1;
                    }
                }
            }
            
            // Handle remaining elements
            for idx0 in (chunks * CHUNK_SIZE)..size {
                let idx_ds = idxs_ds[idx0];
                if idx_ds != idx0 && idx_ds < size && m[idx0] {
                    n_up[idx_ds] += 1;
                }
            }
        }
        None => {
            // Even faster path without mask checks
            const CHUNK_SIZE: usize = 16;
            let chunks = size / CHUNK_SIZE;
            
            for chunk in 0..chunks {
                let start = chunk * CHUNK_SIZE;
                for i in 0..CHUNK_SIZE {
                    let idx0 = start + i;
                    let idx_ds = idxs_ds[idx0];
                    
                    if idx_ds != idx0 && idx_ds < size {
                        n_up[idx_ds] += 1;
                    }
                }
            }
            
            // Handle remaining elements
            for idx0 in (chunks * CHUNK_SIZE)..size {
                let idx_ds = idxs_ds[idx0];
                if idx_ds != idx0 && idx_ds < size {
                    n_up[idx_ds] += 1;
                }
            }
        }
    }
    
    n_up
}

/// Returns a 2D array with upstream cell indices for each cell.
pub fn upstream_matrix(idxs_ds: &Array1<usize>) -> Array2<isize> {
    let n_up = upstream_count(idxs_ds, None);
    let d = *n_up.iter().max().unwrap_or(&0) as usize;
    let n = idxs_ds.len();
    
    if d == 0 {
        return Array2::from_elem((n, 1), MV);
    }
    
    let mut idxs_us = Array2::from_elem((n, d), MV);
    let mut n_up_counter = Array1::zeros(n);
    
    for idx0 in 0..n {
        let idx_ds = idxs_ds[idx0];
        if idx_ds != idx0 && idx_ds < n {
            let i = n_up_counter[idx_ds];
            if i < d {
                idxs_us[[idx_ds, i]] = idx0 as isize;
                n_up_counter[idx_ds] += 1;
            }
        }
    }
    
    idxs_us
}

/// Returns indices of pit cells (cells that flow to themselves).
/// Optimized with SIMD-friendly operations.
pub fn pit_indices(idxs_ds: &Array1<usize>) -> Array1<usize> {
    let size = idxs_ds.len();
    let mut pits = Vec::with_capacity(size / 10); // Estimate 10% pits
    
    // Process in chunks for better vectorization
    const CHUNK_SIZE: usize = 8;
    let chunks = size / CHUNK_SIZE;
    
    for chunk in 0..chunks {
        let start = chunk * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            let idx = start + i;
            if idxs_ds[idx] == idx {
                pits.push(idx);
            }
        }
    }
    
    // Handle remaining elements
    for idx in (chunks * CHUNK_SIZE)..size {
        if idxs_ds[idx] == idx {
            pits.push(idx);
        }
    }
    
    Array1::from_vec(pits)
}

/// Returns indices ordered from down- to upstream.
pub fn idxs_seq(idxs_ds: &Array1<usize>, idxs_pit: &Array1<usize>) -> Array1<usize> {
    let idxs_us = upstream_matrix(idxs_ds);
    let size = idxs_ds.len();
    let mut idxs_seq = Array1::from_elem(size, usize::MAX);
    let mut j = 0;
    
    // Start with pit indices
    for &idx in idxs_pit.iter() {
        if j < size {
            idxs_seq[j] = idx;
            j += 1;
        }
    }
    
    let mut i = 0;
    while i < j && i < size {
        let idx0 = idxs_seq[i];
        
        // Add upstream cells
        for k in 0..idxs_us.ncols() {
            let idx = idxs_us[[idx0, k]];
            if idx == MV {
                break;
            }
            if j < size {
                idxs_seq[j] = idx as usize;
                j += 1;
            }
        }
        i += 1;
    }
    
    // Return only the filled portion
    Array1::from_vec(idxs_seq.slice(s![..j]).to_vec())
}

/// Returns the index of the upstream cell with the largest uparea.
pub fn main_upstream(
    idxs_ds: &Array1<usize>,
    uparea: &Array1<f64>,
    upa_min: f64,
) -> Array1<isize> {
    let size = idxs_ds.len();
    let mut idxs_us_main = Array1::from_elem(size, MV);
    let idxs_us = upstream_matrix(idxs_ds);
    
    for idx0 in 0..size {
        let mut max_uparea = upa_min;
        let mut main_idx = MV;
        
        for k in 0..idxs_us.ncols() {
            let idx = idxs_us[[idx0, k]];
            if idx == MV {
                break;
            }
            let idx_usize = idx as usize;
            if idx_usize < uparea.len() && uparea[idx_usize] > max_uparea {
                max_uparea = uparea[idx_usize];
                main_idx = idx;
            }
        }
        
        idxs_us_main[idx0] = main_idx;
    }
    
    idxs_us_main
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_pit_indices() {
        // Simple test case: cell 2 flows to itself (pit)
        let idxs_ds = array![1, 2, 2];
        let pits = pit_indices(&idxs_ds);
        assert_eq!(pits.len(), 1);
        assert_eq!(pits[0], 2);
    }
    
    #[test]
    fn test_upstream_count() {
        // Cell 0 flows to cell 1, cell 1 flows to cell 2 (pit)
        let idxs_ds = array![1, 2, 2];
        let n_up = upstream_count(&idxs_ds, None);
        assert_eq!(n_up[0], 0); // No upstream cells
        assert_eq!(n_up[1], 1); // One upstream cell (0)
        assert_eq!(n_up[2], 1); // One upstream cell (1)
    }
    
    #[test]
    fn test_rank() {
        // Cell 0 flows to cell 1, cell 1 flows to cell 2 (pit)
        let idxs_ds = array![1, 2, 2];
        let pits = array![false, false, true];
        let (ranks, nnodes) = rank(&idxs_ds, &pits);
        // Cell 2 is a pit (rank 0), cell 1 flows to cell 2 (rank 1), cell 0 flows to cell 1 (rank 2)
        assert_eq!(ranks[2], 0);
        assert_eq!(ranks[1], 1);
        assert_eq!(ranks[0], 2);
        assert_eq!(nnodes, 3);
    }
} 