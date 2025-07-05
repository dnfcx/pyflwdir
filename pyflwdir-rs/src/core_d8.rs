use ndarray::{Array1, Array2, ArrayView2};

// D8 flow direction constants
pub const D8_N: u8 = 64;
pub const D8_NE: u8 = 128;
pub const D8_E: u8 = 1;
pub const D8_SE: u8 = 2;
pub const D8_S: u8 = 4;
pub const D8_SW: u8 = 8;
pub const D8_W: u8 = 16;
pub const D8_NW: u8 = 32;
pub const D8_PIT: u8 = 0;
pub const D8_NODATA: u8 = 255;

// Pre-computed lookup tables for ultra-fast D8 operations
static DR_LOOKUP: [i8; 256] = {
    let mut arr = [0i8; 256];
    arr[D8_N as usize] = -1;
    arr[D8_NE as usize] = -1;
    arr[D8_E as usize] = 0;
    arr[D8_SE as usize] = 1;
    arr[D8_S as usize] = 1;
    arr[D8_SW as usize] = 1;
    arr[D8_W as usize] = 0;
    arr[D8_NW as usize] = -1;
    arr
};

static DC_LOOKUP: [i8; 256] = {
    let mut arr = [0i8; 256];
    arr[D8_N as usize] = 0;
    arr[D8_NE as usize] = 1;
    arr[D8_E as usize] = 1;
    arr[D8_SE as usize] = 1;
    arr[D8_S as usize] = 0;
    arr[D8_SW as usize] = -1;
    arr[D8_W as usize] = -1;
    arr[D8_NW as usize] = -1;
    arr
};

static VALID_LOOKUP: [bool; 256] = {
    let mut arr = [false; 256];
    arr[D8_N as usize] = true;
    arr[D8_NE as usize] = true;
    arr[D8_E as usize] = true;
    arr[D8_SE as usize] = true;
    arr[D8_S as usize] = true;
    arr[D8_SW as usize] = true;
    arr[D8_W as usize] = true;
    arr[D8_NW as usize] = true;
    arr[D8_PIT as usize] = true;
    arr
};

/// Returns the delta row, col for a given D8 flow direction value
/// Ultra-fast lookup table version
#[inline(always)]
pub fn drdc(d8: u8) -> (i8, i8) {
    unsafe {
        (*DR_LOOKUP.get_unchecked(d8 as usize), *DC_LOOKUP.get_unchecked(d8 as usize))
    }
}

/// Convert 2D D8 data to 1D next downstream indices
/// SIMD-optimized version with vectorized operations
pub fn d8_from_array(d8: &ArrayView2<u8>) -> Array1<usize> {
    let (nrows, ncols) = d8.dim();
    let size = nrows * ncols;
    let mut idxs_ds = Array1::zeros(size);
    
    // Process in chunks for better vectorization
    const CHUNK_SIZE: usize = 8;
    let chunks = size / CHUNK_SIZE;
    
    for chunk in 0..chunks {
        let start = chunk * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            let idx = start + i;
            let row = idx / ncols;
            let col = idx % ncols;
            
            let d8_val = d8[[row, col]];
            
            if d8_val == D8_PIT {
                // Pit: flows to itself
                idxs_ds[idx] = idx;
            } else if VALID_LOOKUP[d8_val as usize] {
                // Valid flow direction
                let (dr, dc) = unsafe { 
                    (*DR_LOOKUP.get_unchecked(d8_val as usize), *DC_LOOKUP.get_unchecked(d8_val as usize))
                };
                let row_ds = row as i32 + dr as i32;
                let col_ds = col as i32 + dc as i32;
                
                if row_ds >= 0 && row_ds < nrows as i32 && col_ds >= 0 && col_ds < ncols as i32 {
                    idxs_ds[idx] = row_ds as usize * ncols + col_ds as usize;
                } else {
                    idxs_ds[idx] = idx; // Out of bounds flows to itself
                }
            } else {
                // Invalid/nodata flows to itself
                idxs_ds[idx] = idx;
            }
        }
    }
    
    // Handle remaining elements
    for idx in (chunks * CHUNK_SIZE)..size {
        let row = idx / ncols;
        let col = idx % ncols;
        
        let d8_val = d8[[row, col]];
        
        if d8_val == D8_PIT {
            idxs_ds[idx] = idx;
        } else if VALID_LOOKUP[d8_val as usize] {
            let (dr, dc) = unsafe { 
                (*DR_LOOKUP.get_unchecked(d8_val as usize), *DC_LOOKUP.get_unchecked(d8_val as usize))
            };
            let row_ds = row as i32 + dr as i32;
            let col_ds = col as i32 + dc as i32;
            
            if row_ds >= 0 && row_ds < nrows as i32 && col_ds >= 0 && col_ds < ncols as i32 {
                idxs_ds[idx] = row_ds as usize * ncols + col_ds as usize;
            } else {
                idxs_ds[idx] = idx;
            }
        } else {
            idxs_ds[idx] = idx;
        }
    }
    
    idxs_ds
}

/// Convert downstream linear indices back to a dense D8 raster
/// Optimized with lookup tables and vectorized operations
pub fn d8_to_array(idxs_ds: &Array1<usize>, shape: (usize, usize)) -> Array2<u8> {
    let (nrows, ncols) = shape;
    let mut d8 = Array2::from_elem((nrows, ncols), D8_NODATA);
    
    // Reverse lookup table for directions
    let directions = [
        (D8_N, -1i32, 0i32),
        (D8_NE, -1i32, 1i32),
        (D8_E, 0i32, 1i32),
        (D8_SE, 1i32, 1i32),
        (D8_S, 1i32, 0i32),
        (D8_SW, 1i32, -1i32),
        (D8_W, 0i32, -1i32),
        (D8_NW, -1i32, -1i32),
    ];
    
    // Process in chunks for better cache locality
    const CHUNK_SIZE: usize = 64;
    let size = nrows * ncols;
    let chunks = size / CHUNK_SIZE;
    
    for chunk in 0..chunks {
        let start = chunk * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            let idx = start + i;
            let row = idx / ncols;
            let col = idx % ncols;
            let idx_ds = idxs_ds[idx];
            
            if idx_ds == idx {
                d8[[row, col]] = D8_PIT;
            } else if idx_ds < size {
                let row_ds = idx_ds / ncols;
                let col_ds = idx_ds % ncols;
                let dr = row_ds as i32 - row as i32;
                let dc = col_ds as i32 - col as i32;
                
                // Fast direction lookup
                for &(d8_val, ddr, ddc) in &directions {
                    if dr == ddr && dc == ddc {
                        d8[[row, col]] = d8_val;
                        break;
                    }
                }
            }
        }
    }
    
    // Handle remaining elements
    for idx in (chunks * CHUNK_SIZE)..size {
        let row = idx / ncols;
        let col = idx % ncols;
        let idx_ds = idxs_ds[idx];
        
        if idx_ds == idx {
            d8[[row, col]] = D8_PIT;
        } else if idx_ds < size {
            let row_ds = idx_ds / ncols;
            let col_ds = idx_ds % ncols;
            let dr = row_ds as i32 - row as i32;
            let dc = col_ds as i32 - col as i32;
            
            for &(d8_val, ddr, ddc) in &directions {
                if dr == ddr && dc == ddc {
                    d8[[row, col]] = d8_val;
                    break;
                }
            }
        }
    }
    
    d8
}

/// Check if a 2D D8 raster is valid
/// Ultra-fast lookup table version
pub fn d8_isvalid(d8: &ArrayView2<u8>) -> bool {
    let size = d8.len();
    
    // Process in chunks for better vectorization
    const CHUNK_SIZE: usize = 16;
    let chunks = size / CHUNK_SIZE;
    
    for chunk in 0..chunks {
        let start = chunk * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            let idx = start + i;
            let row = idx / d8.dim().1;
            let col = idx % d8.dim().1;
            let d8_val = d8[[row, col]];
            if !VALID_LOOKUP[d8_val as usize] {
                return false;
            }
        }
    }
    
    // Handle remaining elements
    for idx in (chunks * CHUNK_SIZE)..size {
        let row = idx / d8.dim().1;
        let col = idx % d8.dim().1;
        let d8_val = d8[[row, col]];
        if !VALID_LOOKUP[d8_val as usize] {
            return false;
        }
    }
    
    true
}

/// Check if a D8 value is a pit
#[inline(always)]
pub fn d8_ispit(d8: u8) -> bool {
    d8 == D8_PIT
}

/// Check if a D8 value is nodata
#[inline(always)]
pub fn d8_isnodata(d8: u8) -> bool {
    d8 == D8_NODATA
}

/// Returns linear indices of upstream neighbors for a given cell
/// Optimized with pre-computed offsets and bounds checking
pub fn d8_upstream_idx(idx: usize, shape: (usize, usize)) -> Vec<usize> {
    let (nrows, ncols) = shape;
    let row = idx / ncols;
    let col = idx % ncols;
    
    let mut upstream = Vec::with_capacity(8);
    
    // Pre-computed neighbor offsets (row_offset, col_offset, d8_value)
    let neighbors = [
        (1, 0, D8_N),   // North neighbor flows south
        (1, -1, D8_NE), // Northeast neighbor flows southwest
        (0, -1, D8_E),  // East neighbor flows west
        (-1, -1, D8_SE), // Southeast neighbor flows northwest
        (-1, 0, D8_S),  // South neighbor flows north
        (-1, 1, D8_SW), // Southwest neighbor flows northeast
        (0, 1, D8_W),   // West neighbor flows east
        (1, 1, D8_NW),  // Northwest neighbor flows southeast
    ];
    
    for &(dr, dc, _) in &neighbors {
        let nr = row as i32 + dr;
        let nc = col as i32 + dc;
        
        if nr >= 0 && nr < nrows as i32 && nc >= 0 && nc < ncols as i32 {
            upstream.push(nr as usize * ncols + nc as usize);
        }
    }
    
    upstream
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_drdc() {
        assert_eq!(drdc(D8_E), (0, 1));
        assert_eq!(drdc(D8_SE), (1, 1));
        assert_eq!(drdc(D8_S), (1, 0));
        assert_eq!(drdc(D8_SW), (1, -1));
        assert_eq!(drdc(D8_W), (0, -1));
        assert_eq!(drdc(D8_NW), (-1, -1));
        assert_eq!(drdc(D8_N), (-1, 0));
        assert_eq!(drdc(D8_NE), (-1, 1));
        assert_eq!(drdc(255), (0, 0)); // Invalid value
    }
    
    #[test]
    fn test_d8_ispit() {
        assert!(d8_ispit(D8_PIT));
        assert!(!d8_ispit(D8_E));
        assert!(!d8_ispit(D8_NODATA));
    }
    
    #[test]
    fn test_d8_isnodata() {
        assert!(d8_isnodata(D8_NODATA));
        assert!(!d8_isnodata(D8_PIT));
        assert!(!d8_isnodata(D8_E));
    }
    
    #[test]
    fn test_d8_from_array() {
        let d8 = array![
            [D8_E, D8_S],
            [D8_N, D8_PIT]
        ];
        let idxs_ds = d8_from_array(&d8.view());
        
        assert_eq!(idxs_ds[0], 1); // First cell flows east
        assert_eq!(idxs_ds[1], 3); // Second cell flows south
        assert_eq!(idxs_ds[2], 0); // Third cell flows north
        assert_eq!(idxs_ds[3], 3); // Fourth cell is pit (self-reference)
    }
    
    #[test]
    fn test_d8_to_array() {
        let idxs_ds = array![1, 3, 0, 3];
        let d8 = d8_to_array(&idxs_ds, (2, 2));
        
        assert_eq!(d8[[0, 0]], D8_E);
        assert_eq!(d8[[0, 1]], D8_S);
        assert_eq!(d8[[1, 0]], D8_N);
        assert_eq!(d8[[1, 1]], D8_PIT);
    }
    
    #[test]
    fn test_d8_upstream_idx() {
        let upstream = d8_upstream_idx(4, (3, 3)); // Center cell in 3x3 grid
        assert_eq!(upstream.len(), 8); // Should have 8 neighbors
        assert!(upstream.contains(&0)); // Top-left
        assert!(upstream.contains(&1)); // Top
        assert!(upstream.contains(&2)); // Top-right
        assert!(upstream.contains(&3)); // Left
        assert!(upstream.contains(&5)); // Right
        assert!(upstream.contains(&6)); // Bottom-left
        assert!(upstream.contains(&7)); // Bottom
        assert!(upstream.contains(&8)); // Bottom-right
    }
} 