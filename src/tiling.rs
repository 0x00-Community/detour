/// How a 2D surface's bytes are laid out in memory
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Tiling {
    #[default]
    Linear,
    /// 4 KB tiles of 128 bytes × 32 rows, row-major; inside one, 16-byte columns of 4 rows interleaved (`Tiling::within`)
    Tile4,
}

/// The bytes the CPU moves at a time: a tile's rows are 16-byte runs in every tiling here
const RUN: u32 = 16;

impl Tiling {
    pub const TILE4_WIDTH: u32 = 128;
    pub const TILE4_ROWS: u32 = 32;
    const TILE4_BYTES: usize = 4096;

    /// A row's bytes in memory for `row_bytes` of pixels
    pub const fn pitch(self, row_bytes: u32) -> u32 {
        match self {
            Tiling::Linear => row_bytes,
            Tiling::Tile4 => row_bytes.next_multiple_of(Self::TILE4_WIDTH),
        }
    }

    /// The rows in memory for `height` rows of pixels
    pub const fn rows(self, height: u32) -> u32 {
        match self {
            Tiling::Linear => height,
            Tiling::Tile4 => height.next_multiple_of(Self::TILE4_ROWS),
        }
    }

    /// The bytes a surface of `height` rows of `row_bytes` takes
    pub const fn size(self, row_bytes: u32, height: u32) -> usize {
        self.pitch(row_bytes) as usize * self.rows(height) as usize
    }

    /// Where byte `x` of row `y` is (`x` a multiple of 16 in a tiling)
    pub fn offset(self, x: u32, y: u32, pitch: u32) -> usize {
        match self {
            Tiling::Linear => (y * pitch + x) as usize,
            Tiling::Tile4 => {
                let tile = (y / Self::TILE4_ROWS) as usize * (pitch / Self::TILE4_WIDTH) as usize + (x / Self::TILE4_WIDTH) as usize;
                tile * Self::TILE4_BYTES + Self::within(x % Self::TILE4_WIDTH, y % Self::TILE4_ROWS)
            }
        }
    }

    /// Where byte `x` of row `y` is, any `x`
    pub fn byte(self, x: u32, y: u32, pitch: u32) -> usize {
        self.offset(x & !(RUN - 1), y, pitch) + (x & (RUN - 1)) as usize
    }

    /// A byte's place in its Tile4 tile: 16-byte runs of 4 rows, 4 of them across, then rows 4..8 under them, then
    /// the next 64 bytes of the row, then the next 8 rows twice
    const fn within(x: u32, y: u32) -> usize {
        ((x & 15) | (y & 3) << 4 | (x >> 4 & 3) << 6 | (y >> 2 & 1) << 8 | (x >> 6 & 1) << 9 | (y >> 3 & 3) << 10) as usize
    }

    /// `linear` (rows of `row_bytes`, `height` of them) laid out in `out`
    pub fn tile(self, linear: &[u8], row_bytes: u32, height: u32, out: &mut [u8]) {
        let pitch = self.pitch(row_bytes);
        if self == Tiling::Linear {
            out[..linear.len()].copy_from_slice(linear);
            return;
        }
        for y in 0..height {
            for x in (0..row_bytes).step_by(RUN as usize) {
                let n = RUN.min(row_bytes - x) as usize;
                let from = (y * row_bytes + x) as usize;
                let to = self.offset(x, y, pitch);
                out[to..to + n].copy_from_slice(&linear[from..from + n]);
            }
        }
    }

    /// `tiled` back in rows of `row_bytes` (`height` of them) in `out`: reads `tiled` in order a tile at a time (it is
    /// usually write-combined memory, slow to read but in whole lines), puts the runs where they go in `out`
    pub fn untile(self, tiled: &[u8], row_bytes: u32, height: u32, out: &mut [u8]) {
        if self == Tiling::Linear {
            stream(&tiled[..out.len()], out);
            return;
        }
        #[repr(align(64))]
        struct Tile([u8; Tiling::TILE4_BYTES]);
        let across = (self.pitch(row_bytes) / Self::TILE4_WIDTH) as usize;
        let tiles = across * (self.rows(height) / Self::TILE4_ROWS) as usize;
        let mut tile = Tile([0; Self::TILE4_BYTES]);
        for (i, from) in tiled.chunks_exact(Self::TILE4_BYTES).take(tiles).enumerate() {
            let (x0, y0) = ((i % across) as u32 * Self::TILE4_WIDTH, (i / across) as u32 * Self::TILE4_ROWS);
            stream(from, &mut tile.0);
            for (r, run) in tile.0.chunks_exact(RUN as usize).enumerate() {
                let (x, y) = (x0 + RUNS[r].0, y0 + RUNS[r].1);
                if x >= row_bytes || y >= height {
                    continue;
                }
                let n = RUN.min(row_bytes - x) as usize;
                let to = (y * row_bytes + x) as usize;
                out[to..to + n].copy_from_slice(&run[..n]);
            }
        }
    }
}

/// Byte `x` (a multiple of 16) and row `y` of each 16-byte run of a Tile4 tile, in memory order
const RUNS: [(u32, u32); 256] = {
    let mut runs = [(0, 0); 256];
    let mut r = 0;
    while r < 256 {
        let r32 = r as u32;
        runs[r] = (((r32 >> 2 & 3) | (r32 >> 5 & 1) << 2) << 4, (r32 & 3) | (r32 >> 4 & 1) << 2 | (r32 >> 6 & 3) << 3);
        r += 1;
    }
    runs
};

/// `from` into `out` (same length), with streaming loads where there are: reading write-combined memory a line at a
/// time instead of a load at a time
fn stream(from: &[u8], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse4.1") && (from.as_ptr() as usize).is_multiple_of(16) {
        // SAFETY: SSE4.1 is there; whole 16-byte loads stay inside `from`, which is 16-byte aligned
        unsafe { stream_sse41(from, out) };
        return;
    }
    out.copy_from_slice(from);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn stream_sse41(from: &[u8], out: &mut [u8]) {
    use core::arch::x86_64::{__m128i, _mm_storeu_si128, _mm_stream_load_si128};
    let whole = from.len() / 64 * 64;
    for at in (0..whole).step_by(64) {
        // SAFETY: `at + 64 <= from.len() == out.len()`, `from` 16-byte aligned (the caller)
        unsafe {
            let p = from.as_ptr().add(at) as *const __m128i;
            let (a, b, c, d) = (_mm_stream_load_si128(p), _mm_stream_load_si128(p.add(1)), _mm_stream_load_si128(p.add(2)), _mm_stream_load_si128(p.add(3)));
            let q = out.as_mut_ptr().add(at) as *mut __m128i;
            _mm_storeu_si128(q, a);
            _mm_storeu_si128(q.add(1), b);
            _mm_storeu_si128(q.add(2), c);
            _mm_storeu_si128(q.add(3), d);
        }
    }
    out[whole..].copy_from_slice(&from[whole..]);
}

#[cfg(test)]
mod tests {
    use super::Tiling;

    #[test]
    fn tile4_round_trips_and_fills_its_tiles() {
        let (row, height) = (200 * 4, 45);
        let linear: Vec<u8> = (0..row * height).map(|i| (i * 7 + i / 251) as u8).collect();
        let mut tiled = vec![0; Tiling::Tile4.size(row, height)];
        Tiling::Tile4.tile(&linear, row, height, &mut tiled);
        let mut back = vec![0; linear.len()];
        Tiling::Tile4.untile(&tiled, row, height, &mut back);
        assert_eq!(back, linear);
        for (r, &(x, y)) in super::RUNS.iter().enumerate() {
            assert_eq!(Tiling::within(x, y), r * 16);
        }
        let mut seen = vec![false; Tiling::Tile4.size(Tiling::TILE4_WIDTH, Tiling::TILE4_ROWS)];
        for y in 0..Tiling::TILE4_ROWS {
            for x in 0..Tiling::TILE4_WIDTH {
                let at = Tiling::Tile4.offset(x & !15, y, Tiling::TILE4_WIDTH) + (x & 15) as usize;
                assert!(!seen[at], "({x}, {y}) lands on a byte already taken");
                seen[at] = true;
            }
        }
    }
}
