use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use crate::core::{rank, upstream_count, pit_indices};
use crate::core_d8::d8_from_array;

/// Flow direction type
#[derive(Debug, Clone, PartialEq)]
pub enum FlowType {
    D8,
    // Add other flow types as needed
}

/// A struct to hold flow direction data and related methods
/// Ultra-optimized version with pre-computed data and vectorized operations
#[derive(Debug, Clone)]
pub struct FlwdirRaster {
    /// 1D array of downstream indices
    pub idxs_ds: Array1<usize>,
    /// Shape of the original 2D raster (nrows, ncols)
    pub shape: (usize, usize),
    /// Pre-computed valid mask for faster operations
    valid_mask: Array1<bool>,
    /// Pre-computed pit indices for faster access
    pub pit_indices: Array1<usize>,
    /// Pre-computed upstream counts for faster access
    pub upstream_counts: Array1<i8>,
    /// Cache for expensive operations
    rank_cache: Option<Array1<i32>>,
}

/// Cached data variants
#[derive(Debug, Clone)]
pub enum CachedData {
    Rank(Array1<i32>),
    Area(Array1<f64>),
    Distance(Array1<f64>),
    UpstreamMain(Array1<isize>),
}

impl FlwdirRaster {
    /// Create a new FlwdirRaster from a 2D D8 flow direction array
    /// Ultra-optimized constructor with pre-computation
    pub fn from_array(d8: ArrayView2<u8>) -> Self {
        let shape = d8.dim();
        let size = shape.0 * shape.1;
        
        // Convert D8 to downstream indices
        let idxs_ds = d8_from_array(&d8);
        
        // Pre-compute valid mask using vectorized operations
        let mut valid_mask = Array1::from_elem(size, false);
        const CHUNK_SIZE: usize = 64;
        let chunks = size / CHUNK_SIZE;
        
        for chunk in 0..chunks {
            let start = chunk * CHUNK_SIZE;
            for i in 0..CHUNK_SIZE {
                let idx = start + i;
                valid_mask[idx] = idxs_ds[idx] != idx || {
                    let row = idx / shape.1;
                    let col = idx % shape.1;
                    d8[[row, col]] == 0 // pit
                };
            }
        }
        
        // Handle remaining elements
        for idx in (chunks * CHUNK_SIZE)..size {
            valid_mask[idx] = idxs_ds[idx] != idx || {
                let row = idx / shape.1;
                let col = idx % shape.1;
                d8[[row, col]] == 0 // pit
            };
        }
        
        // Pre-compute pit indices
        let pit_indices = pit_indices(&idxs_ds);
        
        // Pre-compute upstream counts
        let upstream_counts = upstream_count(&idxs_ds, Some(&valid_mask));
        
        Self {
            idxs_ds,
            shape,
            valid_mask,
            pit_indices,
            upstream_counts,
            rank_cache: None,
        }
    }
    
    /// Get downstream indices as i32 array
    /// Ultra-fast conversion using unsafe code for maximum performance
    pub fn get_idxs_ds_i32(&self) -> Array1<i32> {
        let size = self.idxs_ds.len();
        let mut result = Array1::uninit(size);
        
        // Process in chunks for better vectorization
        const CHUNK_SIZE: usize = 8;
        let chunks = size / CHUNK_SIZE;
        
        for chunk in 0..chunks {
            let start = chunk * CHUNK_SIZE;
            for i in 0..CHUNK_SIZE {
                let idx = start + i;
                let val = self.idxs_ds[idx] as i32;
                unsafe {
                    result[idx].write(val);
                }
            }
        }
        
        // Handle remaining elements
        for idx in (chunks * CHUNK_SIZE)..size {
            let val = self.idxs_ds[idx] as i32;
            unsafe {
                result[idx].write(val);
            }
        }
        
        unsafe { result.assume_init() }
    }
    
    /// Calculate flow accumulation
    /// Ultra-optimized using topological order and pre-computed upstream counts
    pub fn accuflux(&self, weights: Option<&Array1<f64>>) -> Array1<f64> {
        let size = self.idxs_ds.len();
        let mut flux = Array1::ones(size);
        
        // Apply weights if provided
        if let Some(w) = weights {
            for i in 0..size {
                flux[i] = w[i];
            }
        }
        
        // Process cells in topological order based on upstream counts
        let mut cells_by_count: HashMap<i8, Vec<usize>> = HashMap::new();
        let mut max_count = 0i8;
        
        for (idx, &count) in self.upstream_counts.iter().enumerate() {
            cells_by_count.entry(count).or_insert_with(Vec::new).push(idx);
            max_count = max_count.max(count);
        }
        
        // Process from highest upstream count to lowest
        for count in (0..=max_count).rev() {
            if let Some(cells) = cells_by_count.get(&count) {
                for &idx in cells {
                    let idx_ds = self.idxs_ds[idx];
                    if idx_ds != idx && idx_ds < size {
                        flux[idx_ds] += flux[idx];
                    }
                }
            }
        }
        
        flux
    }
    
    /// Get flow ranking with caching
    /// Cached version to avoid redundant calculations
    pub fn rank(&mut self) -> &Array1<i32> {
        if self.rank_cache.is_none() {
            let idxs_ds_i32 = self.get_idxs_ds_i32();
            let pits = Array1::from_elem(self.idxs_ds.len(), false);
            let (ranks, _) = rank(&idxs_ds_i32, &pits);
            self.rank_cache = Some(ranks);
        }
        self.rank_cache.as_ref().unwrap()
    }
    
    /// Get valid cell indices
    pub fn valid_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        for (idx, &valid) in self.valid_mask.iter().enumerate() {
            if valid {
                indices.push(idx);
            }
        }
        indices
    }
    
    /// Check if a cell is valid
    #[inline(always)]
    pub fn is_valid(&self, idx: usize) -> bool {
        idx < self.valid_mask.len() && self.valid_mask[idx]
    }
    
    /// Convert linear index to row, col coordinates
    #[inline(always)]
    pub fn idx_to_rowcol(&self, idx: usize) -> (usize, usize) {
        (idx / self.shape.1, idx % self.shape.1)
    }
    
    /// Convert row, col coordinates to linear index
    #[inline(always)]
    pub fn rowcol_to_idx(&self, row: usize, col: usize) -> usize {
        row * self.shape.1 + col
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::core_d8::{D8_E, D8_S, D8_N, D8_PIT};
    
    #[test]
    fn test_flwdir_from_d8() {
        let d8 = array![
            [D8_E, D8_S],
            [D8_N, D8_PIT]
        ];
        
        let flwdir = FlwdirRaster::from_array(d8.view());
        assert_eq!(flwdir.shape, (2, 2));
        assert_eq!(flwdir.size(), 4);
        assert_eq!(flwdir.ftype, FlowType::D8);
        assert_eq!(flwdir.nnodes(), 4);
    }
    
    #[test]
    fn test_flwdir_to_array() {
        let d8 = array![
            [D8_E, D8_S],
            [D8_N, D8_PIT]
        ];
        
        let flwdir = FlwdirRaster::from_array(d8.view());
        let d8_out = flwdir.to_array();
        
        assert_eq!(d8_out, d8);
    }
    
    #[test]
    fn test_pit_indices() {
        let d8 = array![
            [D8_E, D8_S],
            [D8_N, D8_PIT]
        ];
        
        let flwdir = FlwdirRaster::from_array(d8.view());
        let pits = flwdir.pit_indices();
        
        assert_eq!(pits.len(), 1);
        assert_eq!(pits[0], 3); // Bottom-right cell (index 3)
    }
    
    #[test]
    fn test_rank() {
        // Create a simple 2x2 flow direction array
        // D8 values: 2=SE, 4=S, 1=E, 0=PIT
        let d8 = array![[2u8, 4u8], [1u8, 0u8]];
        let flwdir = FlwdirRaster::from_array(d8.view());
        
        let ranks = flwdir.rank();
        
        // Count valid cells (not -9999)
        let n = ranks.iter().filter(|&&r| r != -9999).count();
        
        // All cells should be processed
        assert_eq!(n, 4);
        
        // Check specific ranks
        // Cell 0 (top-left, value 2=SE) flows to cell 3 (bottom-right)
        // Cell 1 (top-right, value 4=S) flows to cell 3 (bottom-right)  
        // Cell 2 (bottom-left, value 1=E) flows to cell 3 (bottom-right)
        // Cell 3 (bottom-right, value 0=PIT) is a pit
        assert_eq!(ranks[3], 0); // Cell 3 is a pit
        assert_eq!(ranks[0], 1); // Cell 0 flows to cell 3
        assert_eq!(ranks[1], 1); // Cell 1 flows to cell 3
        assert_eq!(ranks[2], 1); // Cell 2 flows to cell 3
    }
    
    #[test]
    fn test_upstream_count() {
        // Use the same D8 array as in the rank test for consistency
        let d8 = array![[2u8, 4u8], [1u8, 0u8]];
        let flwdir = FlwdirRaster::from_array(d8.view());
        let n_up = flwdir.upstream_count(None);
        
        // With this flow pattern:
        // Cell 0 (top-left, value 2=SE) flows to cell 3 (bottom-right)
        // Cell 1 (top-right, value 4=S) flows to cell 3 (bottom-right)  
        // Cell 2 (bottom-left, value 1=E) flows to cell 3 (bottom-right)
        // Cell 3 (bottom-right, value 0=PIT) is a pit
        
        assert_eq!(n_up[0], 0); // No upstream cells
        assert_eq!(n_up[1], 0); // No upstream cells
        assert_eq!(n_up[2], 0); // No upstream cells
        assert_eq!(n_up[3], 3); // Three upstream cells (0, 1, 2)
    }
} 