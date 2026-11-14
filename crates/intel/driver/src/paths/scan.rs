//! Prefix sums on the GPU (`scan.mlir`): counts into exclusive offsets, level by level.

use crate::base::{Gpu, Host};

use crate::xe3::{Buffer, Context, Error, Kernel, Recorder};

/// What records a dispatch: a Recorder, or something that also keeps it (to record it again).
pub trait Dispatch<'k> {
    fn dispatch(&mut self, k: &'k Kernel, groups: [u32; 2], params: &[u32]) -> Result<(), Error>;
}

impl<'k, 'd, G: Gpu<Cmd = u64> + 'd, H: Host> Dispatch<'k> for Recorder<'_, 'd, G, H> {
    fn dispatch(&mut self, k: &'k Kernel, groups: [u32; 2], params: &[u32]) -> Result<(), Error> {
        Recorder::dispatch(self, k, groups, params)
    }
}

/// A workgroup's block of counts (scan.mlir: 16 threads, 4 counts an invocation, scanned in shared memory).
pub const BLOCK: usize = 1024;

/// An exclusive prefix sum of n counts into offsets, planned: blocks of BLOCK a workgroup, their sums scanned the same
/// way, level by level, until one block holds them (scanned in place, the total after them). Each level: (counts,
/// offsets, its blocks' sums, n).
pub struct ScanPlan<'d, G: Gpu<Cmd = u64> + 'd> {
    levels: Vec<(u64, u64, u64, usize)>,
    total: u64,
    bufs: Vec<Buffer<'d, G>>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> ScanPlan<'d, G> {
    /// ⭐ The plan for n counts at `counts` into offsets at `offsets` (GPU addresses), its blocks' sums in buffers of
    /// its own.
    pub fn new<H: Host>(cx: &mut Context<'d, G, H>, counts: u64, offsets: u64, n: usize) -> Result<Self, Error> {
        let mut plan = ScanPlan { levels: Vec::new(), total: 0, bufs: Vec::new() };
        let (mut c, mut o, mut n) = (counts, offsets, n);
        loop {
            let nb = n.div_ceil(BLOCK).max(1);
            let sums = cx.buffer(&vec![0; (nb + 1) * 4])?;
            let s = sums.addr;
            plan.bufs.push(sums);
            plan.levels.push((c, o, s, n));
            if nb <= BLOCK {
                plan.total = s + (nb * 4) as u64;
                return Ok(plan);
            }
            let next = cx.buffer(&vec![0; nb * 4])?;
            let no = next.addr;
            plan.bufs.push(next);
            (c, o, n) = (s, no, nb);
        }
    }

    /// Each level's blocks scanned, the deepest one's sums scanned in place (the total after them), then each level's
    /// blocks' starts added, deepest first.
    pub fn record<'k>(&self, r: &mut impl Dispatch<'k>, k: &'k Kernel) -> Result<(), Error> {
        let blocks = |n: usize| [n.div_ceil(BLOCK).max(1) as u32, 1];
        for &(c, o, s, n) in &self.levels {
            r.dispatch(k, blocks(n), &[c as u32, o as u32, s as u32, n as u32, 0])?;
        }
        let &(_, _, s, n) = self.levels.last().expect("a level");
        let nb = n.div_ceil(BLOCK).max(1);
        r.dispatch(k, [1, 1], &[s as u32, s as u32, (s + 4 * nb as u64) as u32, nb as u32, 0])?;
        for (i, &(_, o, s, n)) in self.levels.iter().enumerate().rev() {
            // the starts: the next level's offsets, or the deepest's sums scanned
            let starts = self.levels.get(i + 1).map_or(s, |next| next.1);
            r.dispatch(k, blocks(n), &[0, o as u32, starts as u32, n as u32, 2])?;
        }
        Ok(())
    }

    /// Where the total is on the GPU.
    pub fn total_addr(&self) -> u64 {
        self.total
    }

    /// The total, read back (once the recorded scan is done).
    pub fn total(&mut self) -> Result<u32, Error> {
        let at = self.total;
        let b = self.bufs.iter_mut().find(|b| b.addr <= at && at < b.addr + b.size).ok_or_else(|| Error::msg("scan: no total"))?;
        let o = (at - b.addr) as usize;
        Ok(u32::from_le_bytes(b.bytes()?[o..][..4].try_into().unwrap()))
    }
}

