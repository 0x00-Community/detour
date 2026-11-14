//! ⭐ A scene's paths rasterized into fine's rows on the GPU (roadmap R8): flattened, tiled, sorted, stripped, coarse
//! rasterized with its layers, binned, each stage bit for bit vello_common's / vello_cpu's (the examples `flatten`,
//! `tiles`, `stripgen`, `rowcoarse`, `layercoarse`, `pipeline` check them). The kernels are in `paths/`.
//!
//! [`PathRaster::bin`] takes a scene recorded by `RenderContext::new_recording` ([`Recording`]: its draws' geometry, not
//! stripped); [`Tiles::recorded`](crate::xe3::Tiles::recorded) packs what it paints with; [`TilesAt::with_gpu`](crate::xe3::TilesAt::with_gpu)
//! puts the [`Binned`] rows, commands and alphas in, and [`Fine`](crate::xe3::Fine) draws them.

mod scan;

pub use scan::*;

use crate::base::{Gpu, Host};
use crate::vello_cpu::detour::{Geometry, Item, Recording};
use crate::vello_cpu::kurbo::{Affine, BezPath, PathEl, Point};
use crate::vello_cpu::peniko::{BlendMode, Fill};
use crate::base::vello_common::encode::EncodedPaint;

use crate::xe3::{Buffer, Codegen, Context, Error, Kernel, Recorder};

const FLATTEN: &str = include_str!("paths/flatten.mlir");
const SCAN: &str = include_str!("paths/scan.mlir");
const TILES: &str = include_str!("paths/tiles.mlir");
const CELLS: &str = include_str!("paths/cells.mlir");
const STRIPGEN: &str = include_str!("paths/stripgen.mlir");
const RUNS: &str = include_str!("paths/runs.mlir");
const GLUE: &str = include_str!("paths/glue.mlir");
const EVENTS: &str = include_str!("paths/events.mlir");
const ROWCOARSE: &str = include_str!("paths/rowcoarse.mlir");
const LAYERS: &str = include_str!("paths/layers.mlir");
const PAIRS: &str = include_str!("paths/pairs.mlir");
const BINS: &str = include_str!("paths/bins.mlir");

/// The width of a bucket fine bins a strip row's commands by (fine.rs's BUCKET).
const BUCKET: usize = 128;

/// ⭐ The path rasterization's kernels.
pub struct PathRaster {
    flatten: Kernel,
    scan: Kernel,
    tiles: Kernel,
    /// tiles.mlir's tiles into cells: @paths, @bbox, @cells, @count, @place, @rank
    cells: [Kernel; 6],
    runs: Kernel,
    strips: Kernel,
    /// glue.mlir: @ranges, @stripin, @split, @pairs, @cpaths, @zero, @fill
    glue: [Kernel; 7],
    /// events.mlir: @lclip, @lbox, @evbuild
    events: [Kernel; 3],
    coarse: Kernel,
    /// pairs.mlir: @pfill, @pplace, @prank, @pgather
    pairs: [Kernel; 4],
    bins: Kernel,
}

/// ⭐ A scene's rows of commands binned for fine on the GPU, and its alphas: their addresses (fine's rows, cmds, depth
/// and alphas arrays) and the buffers they are in (kept until fine has drawn them).
pub struct Binned<'d, G: Gpu<Cmd = u64> + 'd> {
    pub rows: u64,
    pub cmds: u64,
    pub depth: u64,
    pub alphas: u64,
    _bufs: Vec<Buffer<'d, G>>,
    _plans: Vec<ScanPlan<'d, G>>,
}

/// A frame of the path rasterization: its buffers (by address), its scans, the stages submitted in turn.
struct Frame<'c, 'd, G: Gpu<Cmd = u64> + 'd, H: Host> {
    cx: &'c mut Context<'d, G, H>,
    bufs: Vec<Buffer<'d, G>>,
    plans: Vec<ScanPlan<'d, G>>,
}

impl<'c, 'd, G: Gpu<Cmd = u64> + 'd, H: Host> Frame<'c, 'd, G, H> {
    /// a buffer of `b` (16 bytes at least: the rest zeros), kept; its address
    fn bytes(&mut self, b: &[u8]) -> Result<u64, Error> {
        let mut v = b.to_vec();
        v.resize(v.len().max(16), 0);
        let buf = self.cx.buffer(&v)?;
        let a = buf.addr;
        self.bufs.push(buf);
        Ok(a)
    }

    fn words(&mut self, v: &[u32]) -> Result<u64, Error> {
        self.bytes(&v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>())
    }

    /// n bytes of zeros (16 at least; whole 16 bytes, as @zero / @fill write them)
    fn zeros(&mut self, n: usize) -> Result<u64, Error> {
        self.bytes(&vec![0; n.next_multiple_of(16).max(16)])
    }

    fn plan(&mut self, counts: u64, offsets: u64, n: usize) -> Result<usize, Error> {
        let p = ScanPlan::new(self.cx, counts, offsets, n)?;
        self.plans.push(p);
        Ok(self.plans.len() - 1)
    }

    fn total(&mut self, plan: usize) -> Result<usize, Error> {
        Ok(self.plans[plan].total()? as usize)
    }

    /// The dispatches `f` records (with the scans' plans), submitted and waited for.
    fn run(&mut self, f: impl FnOnce(&mut Recorder<'_, 'd, G, H>, &[ScanPlan<'d, G>]) -> Result<(), Error>) -> Result<(), Error> {
        let mut r = self.cx.record();
        f(&mut r, &self.plans)?;
        let n = self.cx.submit(r.finish())?;
        self.cx.wait(n)
    }
}

/// A path's GPU elements (examples/flatten's): kind (0 line, 1 quadratic, 2 cubic, 3 close), p0 (the point before
/// it), p1, p2, p3; a close (p0 the last point, p1 the subpath's start) at each move but the first, each close and the
/// end, as `flatten_simd::flatten` closes; its moves.
struct Elements {
    els: Vec<(u32, [Point; 4])>,
    moves: Vec<Point>,
}

impl From<&BezPath> for Elements {
    fn from(path: &BezPath) -> Self {
        let (mut last, mut start) = (Point::ZERO, Point::ZERO);
        let (mut els, mut moves) = (Vec::new(), Vec::new());
        for (k, el) in path.elements().iter().enumerate() {
            match *el {
                PathEl::MoveTo(p) => {
                    if k > 0 {
                        els.push((3, [last, start, start, start]));
                    }
                    (last, start) = (p, p);
                    moves.push(p);
                }
                PathEl::LineTo(p) => {
                    els.push((0, [last, p, p, p]));
                    last = p;
                }
                PathEl::QuadTo(p1, p2) => {
                    els.push((1, [last, p1, p2, p2]));
                    last = p2;
                }
                PathEl::CurveTo(p1, p2, p3) => {
                    els.push((2, [last, p1, p2, p3]));
                    last = p3;
                }
                PathEl::ClosePath => {
                    els.push((3, [last, start, start, start]));
                    last = start;
                }
            }
        }
        if !path.elements().is_empty() {
            els.push((3, [last, start, start, start]));
        }
        Elements { els, moves }
    }
}

/// A scene's paths and events as the GPU takes them.
#[derive(Default)]
struct Scene {
    /// 128 bytes an element: kind, then its points and the path's transform (f64)
    elements: Vec<u8>,
    /// a path's first element and count
    pel: Vec<u32>,
    /// a path's moves NaN
    nan: Vec<u32>,
    /// a path's fill rule (1 even-odd)
    rules: Vec<u32>,
    /// four i32 an item: kind (0 fill, 1 pop), path, layer (-1 none), opaque (a fill's) or the pop's ordinal
    items: Vec<u32>,
    /// a fill's paint attrs (its place among the fills)
    attrs: Vec<u32>,
    /// eight i32 a layer: parent, its box (made on the GPU), its clip path at word 5 (-1 none)
    layers: Vec<u32>,
    fills: u32,
}

impl Scene {
    fn path(&mut self, path: &BezPath, rule: Fill, transform: Affine) -> u32 {
        let e = Elements::from(path);
        self.pel.extend([(self.elements.len() / 128) as u32, e.els.len() as u32]);
        self.nan.push(u32::from(e.moves.iter().any(|m| (transform * *m).is_nan())));
        self.rules.push(u32::from(rule == Fill::EvenOdd));
        for (kind, p) in e.els {
            let mut b = [0u8; 128];
            b[..4].copy_from_slice(&kind.to_le_bytes());
            for (j, v) in p.iter().flat_map(|q| [q.x, q.y]).chain(transform.as_coeffs()).enumerate() {
                b[8 + 8 * j..][..8].copy_from_slice(&v.to_le_bytes());
            }
            self.elements.extend(b);
        }
        (self.pel.len() / 2 - 1) as u32
    }

    fn paths(&self) -> usize {
        self.pel.len() / 2
    }

    /// `rec` as the GPU takes it: each fill's path and each clip's, in order (a clip's at its push: vello_cpu makes
    /// its strips there); a fill opaque as vello_cpu's depth wants it (its own blend, no mask, a paint without
    /// transparency; outside any layer, which the GPU checks).
    fn of(rec: &Recording, encoded: &[EncodedPaint]) -> Result<Self, Error> {
        if let Some(what) = rec.refused {
            return Err(Error::msg(format!("paths: {what} (not yet)")));
        }
        let mut sc = Scene::default();
        let (mut open, mut pops): (Vec<u32>, u32) = (Vec::new(), 0);
        let geometry = |sc: &mut Scene, g: &Geometry| match g {
            Geometry::Path { path, fill_rule, transform } => Ok(sc.path(path, *fill_rule, *transform)),
            Geometry::Rect(_) => Err(Error::msg("paths: a fast path rectangle (not yet)")),
        };
        for item in &rec.items {
            match item {
                Item::Fill { geometry: g, attrs } => {
                    let p = geometry(&mut sc, g)?;
                    let opaque = attrs.blend_mode == BlendMode::default() && attrs.mask.is_none() && !attrs.paint.may_have_transparency(encoded);
                    sc.items.extend([0, p, open.last().copied().unwrap_or(u32::MAX), u32::from(opaque)]);
                    sc.attrs.push(sc.fills);
                    sc.fills += 1;
                }
                Item::Push { clip } => {
                    let c = clip.as_ref().map(|g| geometry(&mut sc, g)).transpose()?;
                    sc.layers.extend([open.last().copied().unwrap_or(u32::MAX), 0, 0, 0, 0, c.unwrap_or(u32::MAX), 0, 0]);
                    open.push((sc.layers.len() / 8 - 1) as u32);
                }
                Item::Pop(_) => {
                    let m = open.pop().ok_or_else(|| Error::msg("paths: a pop without a push"))?;
                    sc.items.extend([1, sc.layers[8 * m as usize + 5], m, pops]);
                    sc.attrs.push(0);
                    pops += 1;
                }
            }
        }
        Ok(sc)
    }
}

impl PathRaster {
    pub fn new<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>) -> Result<Self, Error> {
        let runs_src = [STRIPGEN, RUNS].concat();
        let coarse_src = [ROWCOARSE, LAYERS].concat();
        let mut k = |src: &str, f: &str| cx.kernel(codegen, src, f);
        Ok(PathRaster {
            flatten: k(FLATTEN, "flatten")?,
            scan: k(SCAN, "scan")?,
            tiles: k(TILES, "tiles")?,
            cells: [k(CELLS, "paths")?, k(CELLS, "bbox")?, k(CELLS, "cells")?, k(CELLS, "count")?, k(CELLS, "place")?, k(CELLS, "rank")?],
            runs: k(&runs_src, "runs")?,
            strips: k(&runs_src, "strips_run")?,
            glue: [k(GLUE, "ranges")?, k(GLUE, "stripin")?, k(GLUE, "split")?, k(GLUE, "pairs")?, k(GLUE, "cpaths")?, k(GLUE, "zero")?, k(GLUE, "fill")?],
            events: [k(EVENTS, "lclip")?, k(EVENTS, "lbox")?, k(EVENTS, "evbuild")?],
            coarse: k(&coarse_src, "coarse_ev")?,
            pairs: [k(PAIRS, "pfill")?, k(PAIRS, "pplace")?, k(PAIRS, "prank")?, k(PAIRS, "pgather")?],
            bins: k(BINS, "bins")?,
        })
    }

    /// ⭐ `rec`'s rows binned for fine (its paints `encoded`, its size a multiple of 4 each): its paths flattened,
    /// tiled, stripped, its events coarse rasterized with its layers, binned, each stage a submission (the totals the
    /// next one's buffers are sized by read back). Its paint attrs are each fill's place among the fills and each
    /// layer's among the pops ([`Tiles::recorded`](crate::xe3::Tiles::recorded)'s).
    pub fn bin<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, cx: &mut Context<'d, G, H>, rec: &Recording, encoded: &[EncodedPaint]) -> Result<Binned<'d, G>, Error> {
        let sc = Scene::of(rec, encoded)?;
        let (width, height) = (rec.width, rec.height);
        let mut f = Frame { cx, bufs: Vec::new(), plans: Vec::new() };
        let npaths = sc.paths();
        let n = sc.elements.len() / 128;
        let nitems = sc.items.len() / 4;
        let nlayers = sc.layers.len() / 8;
        let (cols, rows) = (u32::from(width.div_ceil(4)), u32::from(height.div_ceil(4)));
        let nrows = rows as usize;
        let wwords = rows.div_ceil(32);
        let wstride = (rows * 20 + wwords * 4 + 4).next_multiple_of(16);
        let vw = u32::from(width.next_multiple_of(4));
        let lanes = |n: usize| [n.div_ceil(16).max(1) as u32, 1];
        let (eg, pg, ig) = (lanes(n), lanes(npaths), lanes(nitems));
        let [ranges_k, stripin_k, split_k, pairs_k, cpaths_k, _zero, fill_k] = &self.glue;
        let [paths_k, bbox_k, ncells_k, ccount_k, place_k, crank_k] = &self.cells;
        let [lclip_k, lbox_k, evbuild_k] = &self.events;
        let [pfill_k, pplace_k, prank_k, pgather_k] = &self.pairs;
        // @zero's / @fill's workgroups and dwords for n dwords
        let dwords = |n: usize| n.next_multiple_of(4) * 4;
        let zp = |n: usize| ([dwords(n).div_ceil(256).max(1) as u32, 1], (dwords(n) / 4) as u32);

        // flatten: each element's lines counted, their offsets, written; each path's lines
        let mut elements = sc.elements.clone();
        elements.resize(elements.len().max(128), 0);
        let elems = f.bytes(&elements)?;
        let consts = f.bytes(&[0.0, 0.0, f64::from(width), f64::from(height)].iter().flat_map(|v: &f64| v.to_le_bytes()).collect::<Vec<_>>())?;
        let (pel, nan, rules) = (f.words(&sc.pel)?, f.words(&sc.nan)?, f.words(&sc.rules)?);
        let (ecounts, eoffsets) = (f.zeros(n * 4)?, f.zeros(n * 4)?);
        let eplan = f.plan(ecounts, eoffsets, n)?;
        let (fscratch, nans) = (f.zeros(n * 640)?, f.zeros(n * 4)?);
        // the lanes' elements sorted by kind (a lane's work is its kind's)
        let mut perm: Vec<u32> = (0..n as u32).collect();
        perm.sort_by_key(|&e| elements[e as usize * 128]);
        let perm = f.words(&perm)?;
        let fparams = |pts: u64, lines: u64| [elems, consts, ecounts, eoffsets, pts, fscratch, lines, nans, perm].map(|a| a as u32);
        let fp0 = f.words(&fparams(0, 0))?;
        f.run(|r, p| {
            r.dispatch(&self.flatten, eg, &[fp0 as u32, n as u32, 0])?;
            p[eplan].record(r, &self.scan)
        })?;
        let nlines = f.total(eplan)?;
        let (pts, lines) = (f.zeros(nlines * 8)?, f.zeros(nlines * 16)?);
        let ranges = f.zeros(npaths * 8)?;
        let fp1 = f.words(&fparams(pts, lines))?;
        let ltotal = f.plans[eplan].total_addr();
        f.run(|r, _| {
            r.dispatch(&self.flatten, eg, &[fp1 as u32, n as u32, 1])?;
            r.dispatch(ranges_k, pg, &[pel as u32, nan as u32, eoffsets as u32, nans as u32, ltotal as u32, ranges as u32, npaths as u32, n as u32])
        })?;

        // tiles: counted (and the paths with lines left of the viewport flagged: their own windings), written
        // unsorted, sorted by counting into cells
        let (wflags, wpre) = (f.zeros(npaths * 4)?, f.zeros(npaths * 4)?);
        let wplan = f.plan(wflags, wpre, npaths)?;
        let tconsts = f.words(&[cols, rows, wstride, wflags as u32, wpre as u32])?;
        let (tcounts, toffsets) = (f.zeros(npaths * 4)?, f.zeros(npaths * 4)?);
        let tplan = f.plan(tcounts, toffsets, npaths)?;
        let tparams = |tiles: u64, windings: u64| [lines, ranges, tconsts, tcounts, toffsets, tiles, windings].map(|a| a as u32);
        let tp0 = f.words(&tparams(0, 0))?;
        f.run(|r, p| {
            r.dispatch(&self.tiles, pg, &[tp0 as u32, npaths as u32, 0])?;
            p[tplan].record(r, &self.scan)?;
            p[wplan].record(r, &self.scan)
        })?;
        let (ntiles, nculled) = (f.total(tplan)?, f.total(wplan)?);
        let tiles = f.zeros(ntiles * 8)?;
        let wbytes = (1 + nculled) * wstride as usize;
        let windings = f.zeros(wbytes)?;
        let tp1 = f.words(&tparams(tiles, windings))?;
        let spaths = f.zeros(npaths * 16)?;
        let tmp = f.zeros(ntiles * 16)?;
        let tg = lanes(ntiles);
        f.run(|r, _| r.dispatch(&self.tiles, pg, &[tp1 as u32, npaths as u32, 2]))?;
        let (tpaths, bbox, pcells, cbase) = (f.zeros(ntiles * 4)?, f.zeros(npaths * 16)?, f.zeros(npaths * 4)?, f.zeros(npaths * 4)?);
        let bplan = f.plan(pcells, cbase, npaths)?;
        let cp0 = f.words(&[tiles, toffsets, tcounts, tpaths, bbox, pcells, cbase].map(|a| a as u32))?;
        f.run(|r, p| {
            r.dispatch(paths_k, pg, &[cp0 as u32, npaths as u32])?;
            r.dispatch(bbox_k, [ntiles.div_ceil(32 * 16).max(1) as u32, 1], &[cp0 as u32, ntiles as u32])?;
            r.dispatch(ncells_k, pg, &[cp0 as u32, npaths as u32])?;
            p[bplan].record(r, &self.scan)
        })?;
        let ncells = f.total(bplan)?;
        let (cellcount, celloff, cellfill) = (f.zeros(dwords(ncells))?, f.zeros(dwords(ncells))?, f.zeros(dwords(ncells))?);
        let cplan2 = f.plan(cellcount, celloff, ncells)?;
        let cp1 = f.words(&[tiles, toffsets, tcounts, tpaths, bbox, pcells, cbase, cellcount, celloff, cellfill, tmp].map(|a| a as u32))?;
        f.run(|r, p| {
            r.dispatch(ccount_k, tg, &[cp1 as u32, ntiles as u32])?;
            p[cplan2].record(r, &self.scan)?;
            r.dispatch(place_k, tg, &[cp1 as u32, ntiles as u32])?;
            r.dispatch(crank_k, tg, &[cp1 as u32, ntiles as u32])?;
            r.dispatch(stripin_k, pg, &[ranges as u32, toffsets as u32, tcounts as u32, rules as u32, spaths as u32, npaths as u32])
        })?;

        // strips: each path's row runs counted, their offsets; each run's strips and alphas counted, scanned, written
        let sconsts = f.words(&[rows, vw, wstride, wflags as u32, wpre as u32])?;
        let (rcounts, roffsets) = (f.zeros(npaths * 4)?, f.zeros(npaths * 4)?);
        let rplan = f.plan(rcounts, roffsets, npaths)?;
        let rparams = |runs: u64| [spaths, windings, sconsts, tiles, runs, rcounts, roffsets].map(|a| a as u32);
        let rp0 = f.words(&rparams(0))?;
        f.run(|r, p| {
            r.dispatch(&self.runs, pg, &[rp0 as u32, npaths as u32, 0])?;
            p[rplan].record(r, &self.scan)
        })?;
        let nslots = f.total(rplan)?;
        let runs = f.zeros(nslots * 32)?;
        let rp1 = f.words(&rparams(runs))?;
        let (scounts, soffsets) = (f.zeros(nslots * 8)?, f.zeros(nslots * 8)?);
        let (cs_s, cs_a, os_s, os_a) = (f.zeros(nslots * 4)?, f.zeros(nslots * 4)?, f.zeros(nslots * 4)?, f.zeros(nslots * 4)?);
        let (splan, aplan) = (f.plan(cs_s, os_s, nslots)?, f.plan(cs_a, os_a, nslots)?);
        let sscratch = f.zeros(nslots * 80)?;
        let sparams = |strips: u64, alphas: u64| [lines, spaths, tiles, windings, sconsts, scounts, soffsets, strips, alphas, sscratch, runs, roffsets].map(|a| a as u32);
        let sp0 = f.words(&sparams(0, 0))?;
        let sg = lanes(nslots);
        f.run(|r, p| {
            r.dispatch(&self.runs, pg, &[rp1 as u32, npaths as u32, 1])?;
            r.dispatch(&self.strips, sg, &[sp0 as u32, nslots as u32, 0])?;
            r.dispatch(split_k, sg, &[scounts as u32, cs_s as u32, cs_a as u32, nslots as u32])?;
            p[splan].record(r, &self.scan)?;
            p[aplan].record(r, &self.scan)?;
            r.dispatch(pairs_k, sg, &[os_s as u32, os_a as u32, soffsets as u32, nslots as u32])
        })?;
        let (nstrips, nalphas) = (f.total(splan)?, f.total(aplan)?);
        let strips = f.zeros(nstrips * 8)?;
        let alphas = f.zeros(nalphas)?;
        let sp1 = f.words(&sparams(strips, alphas))?;
        f.run(|r, _| r.dispatch(&self.strips, sg, &[sp1 as u32, nslots as u32, 1]))?;

        // coarse: the paths' strip table; the layers' boxes and the events from them; each (row, layer)'s first fill
        // marked; the events' (row, event) pairs in (row, event) order; their commands counted, scanned, written
        let nl = nlayers.max(1);
        let cpaths = f.zeros(npaths * 32)?;
        let (pflags, pattrs, popaque) = (f.zeros(npaths * 4)?, f.zeros(npaths * 4)?, f.zeros(npaths * 4)?);
        let cpp = f.words(&[soffsets as u32, roffsets as u32, f.plans[splan].total_addr() as u32, f.plans[aplan].total_addr() as u32, popaque as u32, cpaths as u32, pflags as u32, pattrs as u32])?;
        let items = f.words(&sc.items)?;
        let mut lw = sc.layers.clone();
        lw.resize(lw.len().max(8), 0);
        let layers = f.words(&lw)?;
        let eattrs = f.words(&sc.attrs)?;
        let events = f.zeros(nitems * 32)?;
        let (prow, prcount, pairbase) = (f.zeros(nitems * 8)?, f.zeros(nitems * 4)?, f.zeros(nitems * 4)?);
        let prplan = f.plan(prcount, pairbase, nitems)?;
        let cconsts0 = f.words(&[cols, rows])?;
        let own = f.zeros(nl * 16)?;
        let evp = f.words(&[cpaths, items, layers, 0, eattrs, events, prow, prcount, strips, cconsts0, own].map(|a| a as u32))?;
        let lg = lanes(nl);
        f.run(|r, p| {
            r.dispatch(cpaths_k, pg, &[cpp as u32, npaths as u32, nslots as u32, 0])?;
            if nlayers > 0 {
                r.dispatch(lclip_k, lg, &[evp as u32, nlayers as u32])?;
                r.dispatch(lbox_k, lg, &[evp as u32, nlayers as u32])?;
            }
            r.dispatch(evbuild_k, ig, &[evp as u32, nitems as u32])?;
            p[prplan].record(r, &self.scan)
        })?;
        let npairs = f.total(prplan)?;
        let (rowcount, rowstart, rowfill) = (f.zeros(dwords(nrows + 1))?, f.zeros(dwords(nrows + 1))?, f.zeros(dwords(nrows + 1))?);
        let rsplan = f.plan(rowcount, rowstart, nrows + 1)?;
        let first = f.zeros(dwords(nrows * nl))?;
        let (pairpath, bucket, sorted) = (f.zeros(npairs * 4)?, f.zeros(npairs * 4)?, f.zeros(npairs * 4)?);
        let (ctab, dtab) = (f.zeros(dwords(npairs))?, f.zeros(dwords(npairs))?);
        let (sc_, sd, coffs, doffs) = (f.zeros((npairs + 1) * 4)?, f.zeros((npairs + 1) * 4)?, f.zeros((npairs + 1) * 4)?, f.zeros((npairs + 1) * 4)?);
        let (cplan, dplan) = (f.plan(sc_, coffs, npairs + 1)?, f.plan(sd, doffs, npairs + 1)?);
        let cconsts = f.words(&[cols, rows, nitems as u32, nl as u32, 0, prow as u32, pairbase as u32, sorted as u32])?;
        let pp = f.words(&[strips, cpaths, cconsts, prow, prcount, pairbase, pairpath, rowcount, rowstart, rowfill, bucket, sorted, ctab, dtab, sc_, sd].map(|a| a as u32))?;
        let cparams = |cmds: u64, depth: u64| [strips, events, layers, cconsts, first, ctab, dtab, coffs, doffs, cmds, depth].map(|a| a as u32);
        let cq0 = f.words(&cparams(0, 0))?;
        let prg = lanes(npairs);
        f.run(|r, p| {
            let (fg, fn_) = zp(nrows * nl);
            r.dispatch(fill_k, fg, &[first as u32, fn_, u32::MAX])?;
            r.dispatch(&self.coarse, ig, &[cq0 as u32, nitems as u32, 2])?;
            r.dispatch(pfill_k, ig, &[pp as u32, nitems as u32])?;
            p[rsplan].record(r, &self.scan)?;
            r.dispatch(pplace_k, prg, &[pp as u32, npairs as u32])?;
            r.dispatch(prank_k, prg, &[pp as u32, npairs as u32])?;
            r.dispatch(&self.coarse, ig, &[cq0 as u32, nitems as u32, 0])?;
            r.dispatch(pgather_k, prg, &[pp as u32, npairs as u32])?;
            p[cplan].record(r, &self.scan)?;
            p[dplan].record(r, &self.scan)
        })?;
        let (ncmds, ndepth) = (f.total(cplan)?, f.total(dplan)?);
        let (cmds, depth) = (f.zeros(ncmds * 16)?, f.zeros(ndepth * 8)?);
        let cq1 = f.words(&cparams(cmds, depth))?;
        f.run(|r, _| r.dispatch(&self.coarse, ig, &[cq1 as u32, nitems as u32, 1]))?;

        // binned by bucket, a (row, bucket) a lane: counted, scanned, written
        let nb = usize::from(width).div_ceil(BUCKET);
        let nbl = nrows * nb;
        let (bcc, bdc, bco, bdo) = (f.zeros(nbl * 4)?, f.zeros(nbl * 4)?, f.zeros(nbl * 4)?, f.zeros(nbl * 4)?);
        let (bcplan, bdplan) = (f.plan(bcc, bco, nbl)?, f.plan(bdc, bdo, nbl)?);
        let bconsts = f.words(&[rows, nitems as u32, nb as u32, rowstart as u32, sc.fills])?;
        let bparams = |c: u64, d: u64, rw: u64| [cmds, depth, coffs, doffs, 0, 0, bconsts, bcc, bdc, bco, bdo, c, d, rw].map(|a| a as u32);
        let bq0 = f.words(&bparams(0, 0, 0))?;
        let bg = [nb.div_ceil(16) as u32, rows];
        f.run(|r, p| {
            r.dispatch(&self.bins, bg, &[bq0 as u32, nbl as u32, 0])?;
            p[bcplan].record(r, &self.scan)?;
            p[bdplan].record(r, &self.scan)
        })?;
        let (nbc, nbd) = (f.total(bcplan)?, f.total(bdplan)?);
        let (bcmds, bdepth, brows) = (f.zeros(nbc * 16)?, f.zeros(nbd * 8)?, f.zeros(nbl * 16)?);
        let bq1 = f.words(&bparams(bcmds, bdepth, brows))?;
        f.run(|r, _| r.dispatch(&self.bins, bg, &[bq1 as u32, nbl as u32, 1]))?;
        Ok(Binned { rows: brows, cmds: bcmds, depth: bdepth, alphas, _bufs: f.bufs, _plans: f.plans })
    }
}
