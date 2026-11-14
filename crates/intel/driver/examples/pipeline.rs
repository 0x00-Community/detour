//! vello_common's fill path on Xe3 end to end against vello_common's own, bit for bit, and timed (roadmap R8): a page
//! of small glyph-like paths (quadratics and cubics, non-zero and even-odd) flattened (`flatten/`), tiled (`tiles/`)
//! and rendered into strips (`stripgen/`) on the GPU, then each given a solid colour and coarse rasterized into the
//! strip rows' commands against vello_cpu's (`rowcoarse/`, a path a lane), binned by bucket (`rowcoarse/bins.mlir`)
//! and drawn by fine.rs's kernel, the pixels against vello_cpu's (only the paints' attrs packed on the CPU); the
//! stages joined there too
//! (`pipeline/glue.mlir`, the prefix sums `flatten/scan.mlir`); the CPU reads back only the totals the next buffers
//! are sized by.

use detour::base::vello_common::fearless_simd::Level;
use detour::base::vello_common::flatten::{self, FlattenCtx, Line};
use detour::base::vello_common::geometry::RectU16;
use detour::base::vello_common::kurbo::{Affine, BezPath, PathEl, Point as KPoint};
use detour::base::vello_common::peniko::Fill;
use detour::base::vello_common::strip::{self, Strip};
use detour::base::vello_common::tile::Tiles;
use detour::base::{Gpu, Native};
use detour::xe3::{Buffer, Codegen, Context, Fine, Kernel, Tiles as FineTiles};
use detour::xe::Xe;
use std::time::Instant;
use detour::vello_cpu::detour::Cmd;
use detour::vello_cpu::peniko::Color;
use detour::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};

use detour::xe3::{Dispatch, ScanPlan};

const PATHS: usize = 4000;
const WIDTH: u16 = 1280;
const HEIGHT: u16 = 800;

/// What the scene does, in order: a fill of path p in a colour, a layer pushed (its clip path, its opacity), popped.
enum Op {
    Fill(usize, Color),
    Push(Option<usize>, f32),
    Pop,
}

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A glyph-like path: one or two closed contours of lines, quadratics and cubics within a 14 px box; now and then
    /// a big one (100–600 px: an opaque one's gaps make depth commands; shifted up and left a quarter of its size).
    fn glyph(&mut self) -> BezPath {
        let size = if self.next() % 100 == 0 { 100.0 + self.unit() * 500.0 } else { 14.0 };
        self.glyph_sized(size)
    }

    /// A glyph-like path of `size` px.
    fn glyph_sized(&mut self, size: f64) -> BezPath {
        // (a big one now and then past the left and the top: windings of what lies left of the viewport)
        let (ox, oy) = (self.unit() * (f64::from(WIDTH) - 10.0) - (size - 14.0) * 0.25, self.unit() * (f64::from(HEIGHT) - 10.0) - (size - 14.0) * 0.25);
        let mut p = BezPath::new();
        let contours = 1 + (self.next() % 2);
        for _ in 0..contours {
            p.move_to(self.near(ox, oy, size));
            for _ in 0..(4 + self.next() % 8) {
                match self.next() % 3 {
                    0 => p.line_to(self.near(ox, oy, size)),
                    1 => p.quad_to(self.near(ox, oy, size), self.near(ox, oy, size)),
                    _ => p.curve_to(self.near(ox, oy, size), self.near(ox, oy, size), self.near(ox, oy, size)),
                }
            }
            p.close_path();
        }
        p
    }

    fn near(&mut self, ox: f64, oy: f64, size: f64) -> KPoint {
        KPoint::new(ox + self.unit() * size, oy + self.unit() * size)
    }
}

/// The GPU elements of a path (as examples/flatten's); its moves.
fn elements(path: &BezPath) -> (Vec<(u32, [KPoint; 4])>, Vec<KPoint>) {
    let mut last = KPoint::ZERO;
    let mut start = KPoint::ZERO;
    let mut out = Vec::new();
    let mut moves = Vec::new();
    for (k, el) in path.elements().iter().enumerate() {
        match *el {
            PathEl::MoveTo(p) => {
                if k > 0 {
                    out.push((3, [last, start, start, start]));
                }
                last = p;
                start = p;
                moves.push(p);
            }
            PathEl::LineTo(p) => {
                out.push((0, [last, p, p, p]));
                last = p;
            }
            PathEl::QuadTo(p1, p2) => {
                out.push((1, [last, p1, p2, p2]));
                last = p2;
            }
            PathEl::CurveTo(p1, p2, p3) => {
                out.push((2, [last, p1, p2, p3]));
                last = p3;
            }
            PathEl::ClosePath => {
                out.push((3, [last, start, start, start]));
                last = start;
            }
        }
    }
    if !path.elements().is_empty() {
        out.push((3, [last, start, start, start]));
    }
    (out, moves)
}

/// A dispatch kept: its kernel, workgroups, parameters.
type Kept<'k> = (&'k Kernel, [u32; 2], Vec<u32>);

/// A Recorder that keeps what it records.
struct Keeping<'a, 'r, 'd, 'k> {
    r: &'a mut detour::xe3::Recorder<'r, 'd, Xe, Native>,
    kept: &'a mut Vec<Kept<'k>>,
}

impl<'k> Dispatch<'k> for Keeping<'_, '_, '_, 'k> {
    fn dispatch(&mut self, k: &'k Kernel, groups: [u32; 2], params: &[u32]) -> Result<(), detour::xe3::Error> {
        self.kept.push((k, groups, params.to_vec()));
        self.r.dispatch(k, groups, params)
    }
}

struct Gpu3<'d, 'k> {
    cx: Context<'d, Xe, Native>,
    scan: &'k Kernel,
    ms: Vec<(&'static str, f64)>,
    /// every stage's dispatches, in order (all of them again in one submission: [`Gpu3::again`])
    kept: Vec<Kept<'k>>,
}

impl<'d, 'k> Gpu3<'d, 'k> {
    fn words(&mut self, v: &[u32]) -> Result<Buffer<'d, Xe>, Box<dyn std::error::Error>> {
        Ok(self.cx.buffer(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?)
    }

    fn zeros(&mut self, bytes: usize) -> Result<Buffer<'d, Xe>, Box<dyn std::error::Error>> {
        Ok(self.cx.buffer(&vec![0; bytes.max(16)])?)
    }

    /// The dispatches `f` records, submitted and waited for: their time as `name`.
    fn run(&mut self, name: &'static str, f: impl FnOnce(&mut Keeping<'_, '_, 'd, 'k>, &'k Kernel) -> Result<(), detour::xe3::Error>) -> Result<(), Box<dyn std::error::Error>> {
        let mut r = self.cx.record();
        f(&mut Keeping { r: &mut r, kept: &mut self.kept }, self.scan)?;
        let t = Instant::now();
        let n = self.cx.submit(r.finish()).map_err(|e| format!("{name} submit: {e:?}"))?;
        self.cx.wait(n).map_err(|e| format!("{name}: {e:?}"))?;
        self.ms.push((name, t.elapsed().as_secs_f64() * 1e3));
        Ok(())
    }

    /// Every dispatch kept, then `last`'s, in one submission (the buffers as they are: sized by the stages' totals
    /// already read back): its time.
    fn again(&mut self, last: impl FnOnce(&mut detour::xe3::Recorder<'_, 'd, Xe, Native>) -> Result<(), detour::xe3::Error>) -> Result<f64, Box<dyn std::error::Error>> {
        let mut r = self.cx.record();
        for (k, groups, params) in &self.kept {
            r.dispatch(k, *groups, params)?;
        }
        last(&mut r)?;
        let t = Instant::now();
        let n = self.cx.submit(r.finish())?;
        self.cx.wait(n).map_err(|e| format!("again: {e:?}"))?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let flatten_k = cx.kernel(&codegen, include_str!("../src/paths/flatten.mlir"), "flatten")?;
    let scan_k = cx.kernel(&codegen, include_str!("../src/paths/scan.mlir"), "scan")?;
    let tiles_k = cx.kernel(&codegen, include_str!("../src/paths/tiles.mlir"), "tiles")?;
    let cells_src = include_str!("../src/paths/cells.mlir");
    let [cpaths2_k, bbox_k, ncells_k, ccount_k, place_k, crank_k] = ["paths", "bbox", "cells", "count", "place", "rank"].map(|f| cx.kernel(&codegen, cells_src, f));
    let (cpaths2_k, bbox_k, ncells_k, ccount_k, place_k, crank_k) = (cpaths2_k?, bbox_k?, ncells_k?, ccount_k?, place_k?, crank_k?);

    let runs_src = [include_str!("../src/paths/stripgen.mlir"), include_str!("../src/paths/runs.mlir")].concat();
    let (runs_k, strips_k) = (cx.kernel(&codegen, &runs_src, "runs")?, cx.kernel(&codegen, &runs_src, "strips_run")?);
    let glue = include_str!("../src/paths/glue.mlir");
    let zero_k = cx.kernel(&codegen, glue, "zero")?;
    let (ranges_k, stripin_k, split_k, pairs_k) = (cx.kernel(&codegen, glue, "ranges")?, cx.kernel(&codegen, glue, "stripin")?, cx.kernel(&codegen, glue, "split")?, cx.kernel(&codegen, glue, "pairs")?);
    let bins_k = cx.kernel(&codegen, include_str!("../src/paths/bins.mlir"), "bins")?;
    let pairs_src = include_str!("../src/paths/pairs.mlir");
    let [pfill_k, pplace_k, prank_k, pgather_k] = ["pfill", "pplace", "prank", "pgather"].map(|f| cx.kernel(&codegen, pairs_src, f));
    let (pfill_k, pplace_k, prank_k, pgather_k) = (pfill_k?, pplace_k?, prank_k?, pgather_k?);
    let (cpaths_k, fill_k) = (cx.kernel(&codegen, glue, "cpaths")?, cx.kernel(&codegen, glue, "fill")?);
    let coarse_k = cx.kernel(&codegen, &[include_str!("../src/paths/rowcoarse.mlir"), include_str!("../src/paths/layers.mlir")].concat(), "coarse_ev")?;
    let ev_src = include_str!("../src/paths/events.mlir");
    let [evflags_k, lclip_k, lbox_k, evbuild_k] = ["evflags", "lclip", "lbox", "evbuild"].map(|f| cx.kernel(&codegen, ev_src, f));
    let (evflags_k, lclip_k, lbox_k, evbuild_k) = (evflags_k?, lclip_k?, lbox_k?, evbuild_k?);
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut rng = Rng(0x7f4a_7c15_9e37_79b9);
    // the scene: PATHS glyph fills, each its own solid colour; now and then inside nested clip and opacity layers (a
    // clip's path among the paths, at its push, as vello_cpu makes its strips). The GPU's items (kind (0 fill, 1 pop),
    // path, layer, opaque or the pop's ordinal) and layers (parent, the box made on the GPU, the clip path)
    let mut lrng = Rng(0x0dd_ba11_cafe_f00d);
    let mut crng = Rng(0x51ed_270b_27a6_12b4);
    let mut paths: Vec<(BezPath, Fill)> = Vec::new();
    let mut ops: Vec<Op> = Vec::new();
    let (mut items, mut layers): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut open: Vec<u32> = Vec::new();
    let (mut pops, mut clips, mut max_depth) = (0u32, 0, 0);
    let pop = |open: &mut Vec<u32>, ops: &mut Vec<Op>, items: &mut Vec<u32>, pops: &mut u32, layers: &[u32]| {
        let m = open.pop().expect("an open layer");
        ops.push(Op::Pop);
        items.extend([1, layers[8 * m as usize + 5], m, *pops]);
        *pops += 1;
    };
    for _ in 0..PATHS {
        if open.len() < 3 && lrng.next() % 60 == 0 {
            let parent = open.last().map_or(u32::MAX, |m| *m);
            let opacity = if lrng.next() % 2 == 0 { 1.0 } else { 0.3 + lrng.unit() as f32 * 0.6 };
            let clip = (lrng.next() % 3 != 0).then(|| {
                let size = 150.0 + lrng.unit() * 450.0;
                paths.push((lrng.glyph_sized(size), Fill::NonZero));
                clips += 1;
                paths.len() - 1
            });
            ops.push(Op::Push(clip, opacity));
            layers.extend([parent, 0, 0, 0, 0, clip.map_or(u32::MAX, |c| c as u32), 0, 0]);
            open.push((layers.len() / 8 - 1) as u32);
            max_depth = max_depth.max(open.len());
        }
        if !open.is_empty() && lrng.next() % 40 == 0 {
            pop(&mut open, &mut ops, &mut items, &mut pops, &layers);
        }
        let path = rng.glyph();
        let rule = if rng.next() % 2 == 0 { Fill::NonZero } else { Fill::EvenOdd };
        let a = if crng.next() % 2 == 0 { 255 } else { 64 + (crng.next() % 128) as u8 };
        let colour = Color::from_rgba8((crng.next() % 256) as u8, (crng.next() % 256) as u8, (crng.next() % 256) as u8, a);
        paths.push((path, rule));
        ops.push(Op::Fill(paths.len() - 1, colour));
        items.extend([0, (paths.len() - 1) as u32, open.last().map_or(u32::MAX, |m| *m), u32::from(a == 255)]);
    }
    while !open.is_empty() {
        pop(&mut open, &mut ops, &mut items, &mut pops, &layers);
    }
    let npaths = paths.len();
    let nitems = items.len() / 4;
    let nlayers = layers.len() / 8;
    let bbox = RectU16 { x0: 0, y0: 0, x1: WIDTH, y1: HEIGHT };

    // vello_common on the CPU, timed
    let t = Instant::now();
    let mut tiles = Tiles::new(level, WIDTH, HEIGHT);
    let mut ctx = FlattenCtx::default();
    let mut lines: Vec<Line> = Vec::new();
    let mut want = Vec::with_capacity(npaths);
    for (path, rule) in &paths {
        flatten::fill(level, path.iter(), Affine::IDENTITY, &mut lines, &mut ctx, bbox);
        tiles.make_tiles_analytic_aa(level, &lines, WIDTH, HEIGHT);
        tiles.sort_tiles();
        let (mut s, mut a): (Vec<Strip>, Vec<u8>) = (Vec::new(), Vec::new());
        strip::render(level, &tiles, &mut s, &mut a, *rule, None, &lines);
        want.push((s, a));
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
    // vello_cpu's coarse rasterization of the scene, timed
    let t = Instant::now();
    let mut rctx = RenderContext::new(WIDTH, HEIGHT);
    let mut resources = Resources::new();
    for op in &ops {
        match *op {
            Op::Fill(p, colour) => {
                rctx.set_fill_rule(paths[p].1);
                rctx.set_paint(colour);
                rctx.fill_path(&paths[p].0);
            }
            Op::Push(clip, opacity) => {
                rctx.set_fill_rule(Fill::NonZero);
                rctx.push_layer(clip.map(|c| &paths[c].0), None, Some(opacity), None, None);
            }
            Op::Pop => rctx.pop_layer(),
        }
    }
    let cpu_coarse_ms0 = t.elapsed().as_secs_f64() * 1e3;
    let mut reference = Pixmap::new(WIDTH, HEIGHT);
    rctx.render_with(&mut reference, &mut Resources::new(), RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let t = Instant::now();
    let coarse = rctx.coarse(&mut resources).ok_or("no coarse rasterization")?;
    let cpu_coarse_ms = cpu_coarse_ms0 + t.elapsed().as_secs_f64() * 1e3;

    // the GPU's inputs: the elements, each path's (first element, count), whether a move is NaN, its fill rule
    let mut bytes = Vec::new();
    let (mut pel, mut flags, mut rules) = (Vec::new(), Vec::new(), Vec::new());
    for (path, rule) in &paths {
        let (els, moves) = elements(path);
        pel.extend([(bytes.len() / 128) as u32, els.len() as u32]);
        flags.push(u32::from(moves.iter().any(|m| m.is_nan())));
        rules.push(u32::from(*rule == Fill::EvenOdd));
        for (kind, p) in els {
            let mut e = [0u8; 128];
            e[..4].copy_from_slice(&kind.to_le_bytes());
            for (j, v) in p.iter().flat_map(|q| [q.x, q.y]).chain(Affine::IDENTITY.as_coeffs()).enumerate() {
                e[8 + 8 * j..][..8].copy_from_slice(&v.to_le_bytes());
            }
            bytes.extend(e);
        }
    }
    let n = bytes.len() / 128;
    let (cols, rows) = (u32::from(WIDTH.div_ceil(4)), u32::from(HEIGHT.div_ceil(4)));
    let words = rows.div_ceil(32);
    let wstride = (rows * 20 + words * 4 + 4).next_multiple_of(16);
    let vw = u32::from(WIDTH.next_multiple_of(4));
    let mut g = Gpu3 { scan: &scan_k, cx, ms: Vec::new(), kept: Vec::new() };
    let elems = g.cx.buffer(&bytes)?;
    let top = f64::from((bbox.y0 / 4) * 4);
    let consts = g.cx.buffer(&[f64::from(bbox.x0), top, f64::from(bbox.x1), f64::from(bbox.y1)].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let pel_b = g.words(&pel)?;
    let flags_b = g.words(&flags)?;
    let rules_b = g.words(&rules)?;
    let ecounts = g.zeros(n * 4)?;
    let eoffsets = g.zeros(n * 4)?;
    let mut eplan = ScanPlan::new(&mut g.cx, ecounts.addr, eoffsets.addr, n)?;
    let fscratch = g.zeros(n * 640)?;
    let nans = g.zeros(n * 4)?;
    // the lanes' elements: sorted by kind (a lane's work is its kind's)
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_by_key(|&e| bytes[e as usize * 128]);
    let perm_b = g.words(&perm)?;
    let fparams = |pts: u64, lines: u64| [elems.addr, consts.addr, ecounts.addr, eoffsets.addr, pts, fscratch.addr, lines, nans.addr, perm_b.addr].map(|a| a as u32);
    let fp0 = g.words(&fparams(0, 0))?;
    let eg = [n.div_ceil(16) as u32, 1];
    // the GPU's clock up first (the count and its scan again and again: the same counts): cold, every stage takes
    // two or three times as long
    for _ in 0..20 {
        let mut r = g.cx.record();
        r.dispatch(&flatten_k, eg, &[fp0.addr as u32, n as u32, 0])?;
        eplan.record(&mut r, g.scan)?;
        let n = g.cx.submit(r.finish())?;
        g.cx.wait(n)?;
    }
    g.run("flatten: count + scan", |r, s| {
        r.dispatch(&flatten_k, eg, &[fp0.addr as u32, n as u32, 0])?;
        eplan.record(r, s)
    })?;
    let nlines = eplan.total()? as usize;
    let pts = g.zeros(nlines * 8)?;
    let lines_b = g.zeros(nlines * 16)?;
    let ranges = g.zeros(npaths * 8)?;
    let fp1 = g.words(&fparams(pts.addr, lines_b.addr))?;
    let pg = [npaths.div_ceil(16) as u32, 1];
    let ltotal = eplan.total_addr();
    g.run("flatten: lines + ranges", |r, _| {
        r.dispatch(&flatten_k, eg, &[fp1.addr as u32, n as u32, 1])?;
        r.dispatch(&ranges_k, pg, &[pel_b.addr as u32, flags_b.addr as u32, eoffsets.addr as u32, nans.addr as u32, ltotal as u32, ranges.addr as u32, npaths as u32, n as u32])
    })?;

    // the windings of what lies left of the viewport: only the paths with lines there have theirs (the count pass
    // flags them; block 1 + their prefix sum), the others share block 0 (zeros)
    let (wflags, wpre) = (g.zeros(npaths * 4)?, g.zeros(npaths * 4)?);
    let mut wplan = ScanPlan::new(&mut g.cx, wflags.addr, wpre.addr, npaths)?;
    let tconsts = g.words(&[cols, rows, wstride, wflags.addr as u32, wpre.addr as u32])?;
    let tcounts = g.zeros(npaths * 4)?;
    let toffsets = g.zeros(npaths * 4)?;
    let mut tplan = ScanPlan::new(&mut g.cx, tcounts.addr, toffsets.addr, npaths)?;
    let tparams = |tiles: u64, windings: u64| [lines_b.addr, ranges.addr, tconsts.addr, tcounts.addr, toffsets.addr, tiles, windings].map(|a| a as u32);
    let tp0 = g.words(&tparams(0, 0))?;
    g.run("tiles: count + scan", |r, s| {
        r.dispatch(&tiles_k, pg, &[tp0.addr as u32, npaths as u32, 0])?;
        tplan.record(r, s)?;
        wplan.record(r, s)
    })?;
    let (ntiles, nculled) = (tplan.total()? as usize, wplan.total()? as usize);
    let tiles_b = g.zeros(ntiles * 8)?;
    let wbytes = (1 + nculled) * wstride as usize;
    let windings = g.zeros(wbytes)?;
    let tp1 = g.words(&tparams(tiles_b.addr, windings.addr))?;
    let spaths = g.zeros(npaths * 16)?;
    // written unsorted, then counted into cells (tiles/cells.mlir): each path's box, its cells, their prefix sum; the
    // cells' counts, theirs; the tiles placed, ranked in their cells
    let tmp = g.zeros(ntiles * 16)?;
    let tgs = [ntiles.div_ceil(16) as u32, 1];
    g.run("tiles: write", |r, _| {
        // (the windings are added into: zeroed first, for the frame recorded again)
        r.dispatch(&zero_k, [wbytes.div_ceil(256) as u32, 1], &[windings.addr as u32, (wbytes / 4) as u32])?;
        r.dispatch(&tiles_k, pg, &[tp1.addr as u32, npaths as u32, 2])
    })?;
    println!("  ({nculled} paths with lines left of the viewport)");
    let (tpaths, bbox, pcells, cbase) = (g.zeros(ntiles * 4)?, g.zeros(npaths * 16)?, g.zeros(npaths * 4)?, g.zeros(npaths * 4)?);
    let mut bplan = ScanPlan::new(&mut g.cx, pcells.addr, cbase.addr, npaths)?;
    let cp0 = g.words(&[tiles_b.addr, toffsets.addr, tcounts.addr, tpaths.addr, bbox.addr, pcells.addr, cbase.addr].map(|a| a as u32))?;
    g.run("tiles: sort, boxes", |r, s| {
        r.dispatch(&cpaths2_k, pg, &[cp0.addr as u32, npaths as u32])?;
        r.dispatch(&bbox_k, [ntiles.div_ceil(32 * 16) as u32, 1], &[cp0.addr as u32, ntiles as u32])?;
        r.dispatch(&ncells_k, pg, &[cp0.addr as u32, npaths as u32])?;
        bplan.record(r, s)
    })?;
    let ncells = bplan.total()? as usize;
    // (whole 16 bytes: @zero's)
    let cbytes = ncells.next_multiple_of(4) * 4;
    let (cellcount, celloff, cellfill) = (g.zeros(cbytes)?, g.zeros(cbytes)?, g.zeros(cbytes)?);
    let cplan2 = ScanPlan::new(&mut g.cx, cellcount.addr, celloff.addr, ncells)?;
    let cp1 = g.words(&[tiles_b.addr, toffsets.addr, tcounts.addr, tpaths.addr, bbox.addr, pcells.addr, cbase.addr, cellcount.addr, celloff.addr, cellfill.addr, tmp.addr].map(|a| a as u32))?;
    let zg = [cbytes.div_ceil(256) as u32, 1];
    g.run("tiles: sort, cells", |r, s| {
        r.dispatch(&zero_k, zg, &[cellcount.addr as u32, ncells.next_multiple_of(4) as u32])?;
        r.dispatch(&zero_k, zg, &[cellfill.addr as u32, ncells.next_multiple_of(4) as u32])?;
        r.dispatch(&ccount_k, tgs, &[cp1.addr as u32, ntiles as u32])?;
        cplan2.record(r, s)?;
        r.dispatch(&place_k, tgs, &[cp1.addr as u32, ntiles as u32])?;
        r.dispatch(&crank_k, tgs, &[cp1.addr as u32, ntiles as u32])
    })?;
    println!("  ({ncells} cells)");
    g.run("tiles: stripin", |r, _| r.dispatch(&stripin_k, pg, &[ranges.addr as u32, toffsets.addr as u32, tcounts.addr as u32, rules_b.addr as u32, spaths.addr as u32, npaths as u32]))?;

    let sconsts = g.words(&[rows, vw, wstride, wflags.addr as u32, wpre.addr as u32])?;
    // strip rendering a row run a lane: each path's runs counted, their offsets, written densely
    let rcounts = g.zeros(npaths * 4)?;
    let roffsets = g.zeros(npaths * 4)?;
    let mut rplan = ScanPlan::new(&mut g.cx, rcounts.addr, roffsets.addr, npaths)?;
    let rparams = |runs: u64| [spaths.addr, windings.addr, sconsts.addr, tiles_b.addr, runs, rcounts.addr, roffsets.addr].map(|a| a as u32);
    let rp0 = g.words(&rparams(0))?;
    g.run("strips: runs count + scan", |r, s| {
        r.dispatch(&runs_k, pg, &[rp0.addr as u32, npaths as u32, 0])?;
        rplan.record(r, s)
    })?;
    let nslots = rplan.total()? as usize;
    let runs = g.zeros(nslots * 32)?;
    let rp1 = g.words(&rparams(runs.addr))?;
    let scounts = g.zeros(nslots * 8)?;
    let soffsets = g.zeros(nslots * 8)?;
    let (cs_s, cs_a, os_s, os_a) = (g.zeros(nslots * 4)?, g.zeros(nslots * 4)?, g.zeros(nslots * 4)?, g.zeros(nslots * 4)?);
    let mut splan = ScanPlan::new(&mut g.cx, cs_s.addr, os_s.addr, nslots)?;
    let mut aplan = ScanPlan::new(&mut g.cx, cs_a.addr, os_a.addr, nslots)?;
    let sscratch = g.zeros(nslots * 80)?;
    let sparams = |strips: u64, alphas: u64| [lines_b.addr, spaths.addr, tiles_b.addr, windings.addr, sconsts.addr, scounts.addr, soffsets.addr, strips, alphas, sscratch.addr, runs.addr, roffsets.addr].map(|a| a as u32);
    let sp0 = g.words(&sparams(0, 0))?;
    let sg = [nslots.div_ceil(16) as u32, 1];
    g.run("strips: runs + count + scan", |r, s| {
        r.dispatch(&runs_k, pg, &[rp1.addr as u32, npaths as u32, 1])?;
        r.dispatch(&strips_k, sg, &[sp0.addr as u32, nslots as u32, 0])?;
        r.dispatch(&split_k, sg, &[scounts.addr as u32, cs_s.addr as u32, cs_a.addr as u32, nslots as u32])?;
        splan.record(r, s)?;
        aplan.record(r, s)?;
        r.dispatch(&pairs_k, sg, &[os_s.addr as u32, os_a.addr as u32, soffsets.addr as u32, nslots as u32])
    })?;
    let (nstrips, nalphas) = (splan.total()? as usize, aplan.total()? as usize);
    let mut strips_b = g.zeros(nstrips * 8)?;
    let mut alphas_b = g.zeros(nalphas)?;
    let sp1 = g.words(&sparams(strips_b.addr, alphas_b.addr))?;
    g.run("strips: write", |r, _| r.dispatch(&strips_k, sg, &[sp1.addr as u32, nslots as u32, 1]))?;

    // coarse rasterization an event a lane (rowcoarse/layers.mlir): the paths' strips (rowcoarse's path table, @cpaths);
    // the scene's events and its layers' boxes made from them (pipeline/events.mlir); each (row, layer)'s first fill
    // marked; the events' (row, event) pairs in (row, event) order (rowcoarse/pairs.mlir); their commands' counts,
    // scanned in that order (one longer: the total last); written
    let cpaths_b = g.zeros(npaths * 32)?;
    let nrows = rows as usize;
    let nl = nlayers.max(1);
    let (pflags, pattrs, popaque) = (g.zeros(npaths * 4)?, g.zeros(npaths * 4)?, g.zeros(npaths * 4)?);
    let cpp = g.words(&[soffsets.addr, roffsets.addr, splan.total_addr(), aplan.total_addr(), popaque.addr, cpaths_b.addr, pflags.addr, pattrs.addr].map(|a| a as u32))?;
    let items_b = g.words(&items)?;
    let mut lw = layers.clone();
    lw.resize(lw.len().max(8), 0);
    let layers_b = g.words(&lw)?;
    let (eflags, eattrs) = (g.zeros(nitems * 4)?, g.zeros(nitems * 4)?);
    let mut paintplan = ScanPlan::new(&mut g.cx, eflags.addr, eattrs.addr, nitems)?;
    let events_b = g.zeros(nitems * 32)?;
    let (prow, prcount, pairbase) = (g.zeros(nitems * 8)?, g.zeros(nitems * 4)?, g.zeros(nitems * 4)?);
    let mut prplan = ScanPlan::new(&mut g.cx, prcount.addr, pairbase.addr, nitems)?;
    let cconsts0 = g.words(&[cols, rows])?;
    let own = g.zeros(nl * 16)?;
    let evp = g.words(&[cpaths_b.addr, items_b.addr, layers_b.addr, eflags.addr, eattrs.addr, events_b.addr, prow.addr, prcount.addr, strips_b.addr, cconsts0.addr, own.addr].map(|a| a as u32))?;
    let (ig, lg) = ([nitems.div_ceil(16) as u32, 1], [nl.div_ceil(16) as u32, 1]);
    g.run("coarse: events, layers", |r, s| {
        r.dispatch(&cpaths_k, pg, &[cpp.addr as u32, npaths as u32, nslots as u32, 0])?;
        r.dispatch(&evflags_k, ig, &[evp.addr as u32, nitems as u32])?;
        paintplan.record(r, s)?;
        r.dispatch(&lclip_k, lg, &[evp.addr as u32, nlayers as u32])?;
        r.dispatch(&lbox_k, lg, &[evp.addr as u32, nlayers as u32])?;
        r.dispatch(&evbuild_k, ig, &[evp.addr as u32, nitems as u32])?;
        prplan.record(r, s)
    })?;
    let (npaints, npairs) = (paintplan.total()?, prplan.total()? as usize);
    // (dwords in whole 16 bytes: @zero's)
    let dwords = |n: usize| n.next_multiple_of(4) * 4;
    let (rowcount, rowstart, rowfill) = (g.zeros(dwords(nrows + 1))?, g.zeros(dwords(nrows + 1))?, g.zeros(dwords(nrows + 1))?);
    let rsplan = ScanPlan::new(&mut g.cx, rowcount.addr, rowstart.addr, nrows + 1)?;
    let first = g.zeros(dwords(nrows * nl))?;
    let (pairpath, bucket, sorted) = (g.zeros(npairs * 4)?, g.zeros(npairs * 4)?, g.zeros(npairs * 4)?);
    let (ctab, dtab) = (g.zeros(dwords(npairs))?, g.zeros(dwords(npairs))?);
    let (sc, sd, coffs, doffs) = (g.zeros((npairs + 1) * 4)?, g.zeros((npairs + 1) * 4)?, g.zeros((npairs + 1) * 4)?, g.zeros((npairs + 1) * 4)?);
    let mut cplan = ScanPlan::new(&mut g.cx, sc.addr, coffs.addr, npairs + 1)?;
    let mut dplan = ScanPlan::new(&mut g.cx, sd.addr, doffs.addr, npairs + 1)?;
    let cconsts = g.words(&[cols, rows, nitems as u32, nl as u32, 0, prow.addr as u32, pairbase.addr as u32, sorted.addr as u32])?;
    let pp = g.words(&[strips_b.addr, cpaths_b.addr, cconsts.addr, prow.addr, prcount.addr, pairbase.addr, pairpath.addr, rowcount.addr, rowstart.addr, rowfill.addr, bucket.addr, sorted.addr, ctab.addr, dtab.addr, sc.addr, sd.addr].map(|a| a as u32))?;
    let cparams = |cmds: u64, depth: u64| [strips_b.addr, events_b.addr, layers_b.addr, cconsts.addr, first.addr, ctab.addr, dtab.addr, coffs.addr, doffs.addr, cmds, depth].map(|a| a as u32);
    let cq0 = g.words(&cparams(0, 0))?;
    let prg = [npairs.div_ceil(16) as u32, 1];
    // @zero's / @fill's workgroups and parameters for n dwords at a
    let zp = |a: u64, n: usize| ([dwords(n).div_ceil(256) as u32, 1], [a as u32, (dwords(n) / 4) as u32]);
    g.run("coarse: pairs + count + scans", |r, s| {
        for (a, n) in [(rowcount.addr, nrows + 1), (rowfill.addr, nrows + 1), (ctab.addr, npairs), (dtab.addr, npairs)] {
            let (zg, zargs) = zp(a, n);
            r.dispatch(&zero_k, zg, &zargs)?;
        }
        let (fg, fargs) = zp(first.addr, nrows * nl);
        r.dispatch(&fill_k, fg, &[fargs[0], fargs[1], u32::MAX])?;
        r.dispatch(&coarse_k, ig, &[cq0.addr as u32, nitems as u32, 2])?;
        r.dispatch(&pfill_k, ig, &[pp.addr as u32, nitems as u32])?;
        rsplan.record(r, s)?;
        r.dispatch(&pplace_k, prg, &[pp.addr as u32, npairs as u32])?;
        r.dispatch(&prank_k, prg, &[pp.addr as u32, npairs as u32])?;
        r.dispatch(&coarse_k, ig, &[cq0.addr as u32, nitems as u32, 0])?;
        r.dispatch(&pgather_k, prg, &[pp.addr as u32, npairs as u32])?;
        cplan.record(r, s)?;
        dplan.record(r, s)
    })?;
    println!("  ({nitems} events, {nlayers} layers ({clips} clipped, {max_depth} deep at most), {npairs} (row, event) pairs)");
    let (ncmds, ndepth) = (cplan.total()?, dplan.total()?);
    let mut cmds_b = g.zeros(ncmds as usize * 16)?;
    let mut depth_b = g.zeros(ndepth as usize * 8)?;
    let cq1 = g.words(&cparams(cmds_b.addr, depth_b.addr))?;
    g.run("coarse: write", |r, _| r.dispatch(&coarse_k, ig, &[cq1.addr as u32, nitems as u32, 1]))?;

    // the rows' commands binned by 128 px bucket as the fine kernel reads them, a (row, bucket) a lane
    let nb = usize::from(WIDTH).div_ceil(128);
    let nbl = nrows * nb;
    let (bcc, bdc, bco, bdo) = (g.zeros(nbl * 4)?, g.zeros(nbl * 4)?, g.zeros(nbl * 4)?, g.zeros(nbl * 4)?);
    let mut bcplan = ScanPlan::new(&mut g.cx, bcc.addr, bco.addr, nbl)?;
    let mut bdplan = ScanPlan::new(&mut g.cx, bdc.addr, bdo.addr, nbl)?;
    let bconsts = g.words(&[rows, nitems as u32, nb as u32, rowstart.addr as u32, npaints])?;
    let (ctot, dtot) = (cplan.total_addr(), dplan.total_addr());
    let bparams = |cmds: u64, depth: u64, rows: u64| [cmds_b.addr, depth_b.addr, coffs.addr, doffs.addr, ctot, dtot, bconsts.addr, bcc.addr, bdc.addr, bco.addr, bdo.addr, cmds, depth, rows].map(|a| a as u32);
    let bq0 = g.words(&bparams(0, 0, 0))?;
    let bg = [nb.div_ceil(16) as u32, rows];
    g.run("bins: count + scans", |r, s| {
        r.dispatch(&bins_k, bg, &[bq0.addr as u32, nbl as u32, 0])?;
        bcplan.record(r, s)?;
        bdplan.record(r, s)
    })?;
    let (nbc, nbd) = (bcplan.total()? as usize, bdplan.total()? as usize);
    let (mut bcmds, mut bdepth, mut brows) = (g.zeros(nbc * 16)?, g.zeros(nbd * 8)?, g.zeros(nbl * 16)?);
    let bq1 = g.words(&bparams(bcmds.addr, bdepth.addr, brows.addr))?;
    g.run("bins: write", |r, _| r.dispatch(&bins_k, bg, &[bq1.addr as u32, nbl as u32, 1]))?;

    // the pixels: fine.rs's kernel on the GPU's bins and alphas (only the paints' attrs packed on the CPU)
    let fine = Fine::new(&mut g.cx, &codegen)?;
    let ftiles = FineTiles::new(&coarse, &[], &[])?;
    let mut fbuf = g.zeros(ftiles.bytes())?;
    let faddr = fbuf.addr;
    let at = ftiles.write(fbuf.bytes()?, faddr, &[])?.with_gpu(brows.addr, bcmds.addr, bdepth.addr, alphas_b.addr);
    g.run("fine", |r, _| fine.record(r.r, &at))?;
    // the whole frame again in one submission (the buffers sized already): no reading back between the stages
    let one_ms = g.again(|r| fine.record(r, &at))?;
    // fine again (the same pixels), its fastest of ten: the GPU's clock and the machine's load vary a lot
    let mut fine_ms = Vec::new();
    for _ in 0..10 {
        let mut r = g.cx.record();
        fine.record(&mut r, &at)?;
        let t = Instant::now();
        let n = g.cx.submit(r.finish())?;
        g.cx.wait(n).map_err(|e| format!("fine ten: {e:?}"))?;
        fine_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    fine_ms.sort_by(f64::total_cmp);
    println!("fine: fastest of ten {:.2} ms, median {:.2} ms", fine_ms[0], fine_ms[5]);
    let got_px = fbuf.bytes()?[(at.out - faddr) as usize..][..ftiles.pixels()].to_vec();
    let want_px = reference.data_as_u8_slice();
    let px_bad = got_px.chunks(4).zip(want_px.chunks(4)).filter(|(a, b)| a != b).count();
    println!("{} pixels, {px_bad} differ from vello_cpu's", got_px.len() / 4);

    // bit for bit, path by path
    let mut soff = soffsets;
    let mut roff = roffsets;
    let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..][..4].try_into().unwrap());
    let (ob, tob, sb, ab) = (soff.bytes()?.to_vec(), roff.bytes()?.to_vec(), strips_b.bytes()?.to_vec(), alphas_b.bytes()?.to_vec());
    // a path's strips and alphas: its slots' in turn, from its first slot's offsets to the next path's
    let slot0 = |p: usize| if p < npaths { word(&tob, 4 * p) as usize } else { nslots };
    let at = |slot: usize| if slot < nslots { (word(&ob, 8 * slot) as usize, word(&ob, 8 * slot + 4) as usize) } else { (nstrips, nalphas) };
    let mut ok = 0;
    let mut first_bad = None;
    for (p, (ws, wa)) in want.iter().enumerate() {
        let ((so, ao), (se, ae)) = (at(slot0(p)), at(slot0(p + 1)));
        let (ns, na) = (se - so, ae - ao);
        let got: Vec<(u16, u16, u32, bool)> = (0..ns).map(|i| {
            let (xy, pk) = (word(&sb, (so + i) * 8), word(&sb, (so + i) * 8 + 4));
            (xy as u16, (xy >> 16) as u16, pk & 0x7fff_ffff, pk >> 31 != 0)
        }).collect();
        let w: Vec<(u16, u16, u32, bool)> = ws.iter().map(|s| (s.x, s.y, s.alpha_idx(), s.fill_gap())).collect();
        if got == w && &ab[ao..][..na] == wa.as_slice() {
            ok += 1;
        } else if first_bad.is_none() {
            first_bad = Some((p, ns, w.len(), na, wa.len()));
        }
    }
    // the rows' commands bit for bit (a push's span left out: fine does not read it); a row's: its pairs rowstart[r] ..
    // rowstart[r + 1] (the offsets one longer)
    let (mut coffs, mut doffs, mut rowstart) = (coffs, doffs, rowstart);
    let (cob, dob, rsb, cb, db) = (coffs.bytes()?.to_vec(), doffs.bytes()?.to_vec(), rowstart.bytes()?.to_vec(), cmds_b.bytes()?.to_vec(), depth_b.bytes()?.to_vec());
    let kind = |c: &Cmd| match *c {
        Cmd::PaintFill { span, alpha, attrs } => (0, u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), attrs),
        Cmd::LayerFill { span, alpha, attrs } => (1, u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), attrs),
        Cmd::PushBuf(_) => (2, 0, u32::MAX, 0),
        Cmd::PopBuf => (3, 0, u32::MAX, 0),
    };
    let (mut rows_ok, mut kinds) = (0, [0usize; 4]);
    let mut first_bad_row = None;
    for (r, row) in coarse.rows.iter().enumerate().take(nrows) {
        let want_c: Vec<(u32, u32, u32, u32)> = row.cmds.iter().map(kind).collect();
        let want_d: Vec<(u32, u32)> = row.depth.iter().map(|d| (u32::from(d.buckets.start) | u32::from(d.buckets.end) << 16, d.attrs)).collect();
        let (p0, p1) = (word(&rsb, 4 * r) as usize, word(&rsb, 4 * (r + 1)) as usize);
        let (ca, cz, da, dz) = (word(&cob, 4 * p0) as usize, word(&cob, 4 * p1) as usize, word(&dob, 4 * p0) as usize, word(&dob, 4 * p1) as usize);
        let got_c: Vec<(u32, u32, u32, u32)> = (ca..cz).map(|i| (word(&cb, 16 * i + 12), word(&cb, 16 * i), word(&cb, 16 * i + 4), word(&cb, 16 * i + 8))).collect();
        let got_d: Vec<(u32, u32)> = (da..dz).map(|i| (word(&db, 8 * i), word(&db, 8 * i + 4))).collect();
        for c in &want_c {
            kinds[c.0 as usize] += 1;
        }
        if got_c == want_c && got_d == want_d {
            rows_ok += 1;
        } else if first_bad_row.is_none() {
            let at = got_c.iter().zip(&want_c).position(|(g, w)| g != w);
            first_bad_row = Some((r, got_c.len(), want_c.len(), at.map(|i| (got_c[i], want_c[i])), got_d.len(), want_d.len()));
        }
    }
    println!("{rows_ok} of {nrows} rows' coarse commands bit for bit ({} paint fills, {} layer fills, {} pushes, {} pops; {ndepth} depth commands)", kinds[0], kinds[1], kinds[2], kinds[3]);
    if let Some(b) = first_bad_row {
        println!("  first wrong: row {}: {} commands (want {}), first differing {:?}; {} depth (want {})", b.0, b.1, b.2, b.3, b.4, b.5);
    }
    // the bins bit for bit: fine.rs's packing of vello_cpu's rows
    let (mut want_rows, mut want_cmds, mut want_depth) = (Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new());
    for row in coarse.rows.iter().take(nrows) {
        for b in 0..nb as u32 {
            let (f0, d0) = (want_cmds.len() / 4, want_depth.len() / 2);
            let ov = |x: u16, w: u16| u32::from(x) < (b + 1) * 128 && u32::from(x) + u32::from(w) > b * 128;
            for c in &row.cmds {
                match *c {
                    Cmd::PaintFill { span, alpha, attrs } if ov(span.x, span.width) => want_cmds.extend([0, u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), attrs]),
                    Cmd::LayerFill { span, alpha, attrs } if ov(span.x, span.width) => want_cmds.extend([1, u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), npaints + attrs]),
                    Cmd::PushBuf(_) => want_cmds.extend([2, 0, u32::MAX, 0]),
                    Cmd::PopBuf if want_cmds.len() > f0 * 4 && want_cmds[want_cmds.len() - 4] == 2 => want_cmds.truncate(want_cmds.len() - 4),
                    Cmd::PopBuf => want_cmds.extend([3, 0, u32::MAX, 0]),
                    _ => {}
                }
            }
            for d in row.depth.iter().filter(|d| u32::from(d.buckets.start) <= b && b < u32::from(d.buckets.end)) {
                want_depth.extend([u32::from(d.buckets.start) | u32::from(d.buckets.end) << 16, d.attrs]);
            }
            want_rows.extend([f0 as u32, (want_cmds.len() / 4 - f0) as u32, d0 as u32, (want_depth.len() / 2 - d0) as u32]);
        }
    }
    let dwords = |b: &[u8], n: usize| b[..n * 4].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
    let bins_ok = dwords(brows.bytes()?, nbl * 4) == want_rows && dwords(bcmds.bytes()?, nbc * 4) == want_cmds && dwords(bdepth.bytes()?, nbd * 2) == want_depth;
    println!("bins {}: {nbl} (row, bucket)s, {nbc} commands, {nbd} depth commands (want {}, {})", if bins_ok { "bit for bit" } else { "WRONG" }, want_cmds.len() / 4, want_depth.len() / 2);
    let gpu_ms: f64 = g.ms.iter().map(|(_, m)| m).sum();
    println!("{ok} of {npaths} paths' strips and alphas bit for bit ({n} elements, {nlines} lines, {ntiles} tiles, {nstrips} strips, {nalphas} alpha bytes)");
    if let Some(b) = first_bad {
        println!("  first wrong: path {}: {} strips (want {}), {} alpha bytes (want {})", b.0, b.1, b.2, b.3, b.4);
    }
    for (name, ms) in &g.ms {
        println!("  {name}: {ms:.2} ms");
    }
    println!("the frame again in one submission: {one_ms:.2} ms (all checked after it)");
    println!("GPU {gpu_ms:.2} ms ({} submissions), to strips: vello_common on one CPU thread {cpu_ms:.2} ms; to coarse: vello_cpu {cpu_coarse_ms:.2} ms", g.ms.len());
    let pass = ok == npaths && rows_ok == nrows && bins_ok && px_bad == 0;
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
