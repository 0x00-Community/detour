//! vello_cpu's coarse rasterization of plain fills on Xe3 against vello_cpu's own, bit for bit (roadmap R8.4, its
//! first part): a page of glyph-like paths, each filled with its own solid colour (opaque or not; non-zero or
//! even-odd), drawn by a `detour::vello_cpu::RenderContext` whose `coarse()` is the reference; the GPU takes each path's strips
//! and alphas (vello_common's, as that context makes them) and buckets them into the strip rows' render and depth
//! commands (`rowcoarse/rowcoarse.mlir`, a row a lane, every path in draw order).

use detour::base::vello_common::fearless_simd::Level;
use detour::base::vello_common::flatten::{self, FlattenCtx, Line};
use detour::base::vello_common::geometry::RectU16;
use detour::base::vello_common::kurbo::{Affine, BezPath, Point as KPoint};
use detour::base::vello_common::peniko::Fill;
use detour::base::vello_common::strip::{self, Strip};
use detour::base::vello_common::tile::Tiles;
use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;
use detour::vello_cpu::detour::Cmd;
use detour::vello_cpu::peniko::Color;
use detour::vello_cpu::{RenderContext, Resources};

use detour::xe3::ScanPlan;

const PATHS: usize = 4000;
const WIDTH: u16 = 1280;
const HEIGHT: u16 = 800;

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

    fn near(&mut self, ox: f64, oy: f64, size: f64) -> KPoint {
        KPoint::new(ox + self.unit() * size, oy + self.unit() * size)
    }

    /// A glyph-like path (14 px), or now and then a big one (up to 600 px: its gaps span depth buckets).
    fn glyph(&mut self) -> BezPath {
        let size = if self.next() % 40 == 0 { 100.0 + self.unit() * 500.0 } else { 14.0 };
        let (ox, oy) = (self.unit() * (f64::from(WIDTH) - size * 0.5) - size * 0.25, self.unit() * (f64::from(HEIGHT) - size * 0.5) - size * 0.25);
        let mut p = BezPath::new();
        for _ in 0..(1 + self.next() % 2) {
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("../src/paths/rowcoarse.mlir"), "coarse")?;
    let per_path = cx.kernel(&codegen, include_str!("../src/paths/rowcoarse.mlir"), "coarse_path")?;
    let scan = cx.kernel(&codegen, include_str!("../src/paths/scan.mlir"), "scan")?;
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut rng = Rng(0x51ed_270b_27a6_12b4);
    let bbox = RectU16 { x0: 0, y0: 0, x1: WIDTH, y1: HEIGHT };

    // vello_cpu's: the paths drawn, their coarse rasterization; and each path's strips and alphas as it makes them
    let mut ctx = RenderContext::new(WIDTH, HEIGHT);
    let mut resources = Resources::new();
    let mut tiles = Tiles::new(level, WIDTH, HEIGHT);
    let (mut sbytes, mut paths) = (Vec::new(), Vec::new());
    let mut alpha_base = 0u32;
    let (mut nstrip_total, mut opaque_paths) = (0, 0);
    for _ in 0..PATHS {
        let path = rng.glyph();
        let rule = if rng.next() % 2 == 0 { Fill::NonZero } else { Fill::EvenOdd };
        let opaque = rng.next() % 2 == 0;
        let a = if opaque { 255 } else { 64 + (rng.next() % 128) as u8 };
        let colour = Color::from_rgba8((rng.next() % 256) as u8, (rng.next() % 256) as u8, (rng.next() % 256) as u8, a);
        ctx.set_fill_rule(rule);
        ctx.set_paint(colour);
        ctx.fill_path(&path);

        let mut lines: Vec<Line> = Vec::new();
        flatten::fill(level, path.iter(), Affine::IDENTITY, &mut lines, &mut FlattenCtx::default(), bbox);
        tiles.make_tiles_analytic_aa(level, &lines, WIDTH, HEIGHT);
        tiles.sort_tiles();
        let (mut s, mut al): (Vec<Strip>, Vec<u8>) = (Vec::new(), Vec::new());
        strip::render(level, &tiles, &mut s, &mut al, rule, None, &lines);
        paths.extend([(sbytes.len() / 8) as u32, s.len() as u32, alpha_base, paths.len() as u32 / 8, u32::from(opaque), 0, 0, 0]);
        opaque_paths += usize::from(opaque);
        nstrip_total += s.len();
        for st in &s {
            sbytes.extend((u32::from(st.x) | u32::from(st.y) << 16).to_le_bytes());
            sbytes.extend((st.alpha_idx() | u32::from(st.fill_gap()) << 31).to_le_bytes());
        }
        alpha_base += al.len() as u32;
    }
    let coarse = ctx.coarse(&mut resources).ok_or("no coarse rasterization")?;
    sbytes.resize(sbytes.len().max(8), 0);

    let (cols, rows) = (u32::from(WIDTH.div_ceil(4)), u32::from(HEIGHT.div_ceil(4)));
    let nrows = rows as usize;
    let sbuf = cx.buffer(&sbytes)?;
    let pbuf = cx.buffer(&paths.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    const CHUNK: usize = 32;
    let nchunks = PATHS.div_ceil(CHUNK);
    let nlanes = nrows * nchunks;
    let cbuf = cx.buffer(&[cols, rows, PATHS as u32, CHUNK as u32, nchunks as u32, 0, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let mut counts = cx.buffer(&vec![0; nlanes * 8])?;
    let offsets = cx.buffer(&vec![0; nlanes * 8])?;
    let (ccounts, dcounts, coffs, doffs) = (cx.buffer(&vec![0; nrows * 4])?, cx.buffer(&vec![0; nrows * 4])?, cx.buffer(&vec![0; nrows * 4])?, cx.buffer(&vec![0; nrows * 4])?);
    let mut cblocks = cx.buffer(&vec![0; (nrows.div_ceil(256) + 1) * 4])?;
    let mut dblocks = cx.buffer(&vec![0; (nrows.div_ceil(256) + 1) * 4])?;
    let groups = [nchunks.div_ceil(16) as u32, nrows as u32];
    let fixed = [sbuf.addr, pbuf.addr, cbuf.addr, counts.addr, offsets.addr];
    let addrs = |cmds: u64, depth: u64| fixed.iter().chain(&[cmds, depth]).flat_map(|a| (*a as u32).to_le_bytes()).collect::<Vec<_>>();
    let p0 = cx.buffer(&addrs(0, 0))?;
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p0.addr as u32, nrows as u32, 0])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    // the counts split, scanned, paired back (on the CPU here: a few hundred rows)
    let cv: Vec<(u32, u32)> = counts.bytes()?[..nlanes * 8].chunks(8).map(|c| (u32::from_le_bytes(c[..4].try_into().unwrap()), u32::from_le_bytes(c[4..].try_into().unwrap()))).collect();
    let _ = (&ccounts, &dcounts, &coffs, &doffs, &mut cblocks, &mut dblocks, &scan);
    let mut offs = Vec::with_capacity(nlanes);
    let (mut nc, mut nd) = (0u32, 0u32);
    for (c, d) in &cv {
        offs.push((nc, nd));
        nc += c;
        nd += d;
    }
    let obuf = cx.buffer(&offs.iter().flat_map(|(c, d)| [c.to_le_bytes(), d.to_le_bytes()].concat()).collect::<Vec<_>>())?;
    let mut cmds = cx.buffer(&vec![0xa5; nc.max(1) as usize * 16])?;
    let mut depth = cx.buffer(&vec![0xa5; nd.max(1) as usize * 8])?;
    let fixed1 = [sbuf.addr, pbuf.addr, cbuf.addr, counts.addr, obuf.addr, cmds.addr, depth.addr];
    let p1 = cx.buffer(&fixed1.iter().flat_map(|a| (*a as u32).to_le_bytes()).collect::<Vec<_>>())?;
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p1.addr as u32, nrows as u32, 1])?;
    let t = std::time::Instant::now();
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let cb = cmds.bytes()?.to_vec();
    let db = depth.bytes()?.to_vec();
    let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..][..4].try_into().unwrap());

    // a path a lane: its commands' slots from a (row, path) table's prefix sum, on the GPU
    let nt = nrows * PATHS;
    let (mut ctab, mut dtab) = (cx.buffer(&vec![0; nt * 4])?, cx.buffer(&vec![0; nt * 4])?);
    let (mut coffs, mut doffs) = (cx.buffer(&vec![0; nt * 4])?, cx.buffer(&vec![0; nt * 4])?);
    let mut cplan = ScanPlan::new(&mut cx, ctab.addr, coffs.addr, nt)?;
    let mut dplan = ScanPlan::new(&mut cx, dtab.addr, doffs.addr, nt)?;
    let pg = [PATHS.div_ceil(16) as u32, 1];
    let q0 = cx.buffer(&[sbuf.addr, pbuf.addr, cbuf.addr, ctab.addr, dtab.addr, coffs.addr, doffs.addr, 0, 0].iter().flat_map(|a| (*a as u32).to_le_bytes()).collect::<Vec<_>>())?;
    let mut path_count_ms = 0.0;
    let mut parts = Vec::new();
    for step in 0..3 {
        let mut r = cx.record();
        match step {
            0 => r.dispatch(&per_path, pg, &[q0.addr as u32, PATHS as u32, 0])?,
            1 => cplan.record(&mut r, &scan)?,
            _ => dplan.record(&mut r, &scan)?,
        }
        let t = std::time::Instant::now();
        let s = cx.submit(r.finish())?;
        cx.wait(s)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        parts.push(ms);
        path_count_ms += ms;
    }
    println!("  count {:.2} ms, commands' table scan {:.2} ms, depth's {:.2} ms", parts[0], parts[1], parts[2]);
    let (pc, pd) = (cplan.total()?, dplan.total()?);
    let mut pcmds = cx.buffer(&vec![0xa5; pc.max(1) as usize * 16])?;
    let mut pdepth = cx.buffer(&vec![0xa5; pd.max(1) as usize * 8])?;
    let q1 = cx.buffer(&[sbuf.addr, pbuf.addr, cbuf.addr, ctab.addr, dtab.addr, coffs.addr, doffs.addr, pcmds.addr, pdepth.addr].iter().flat_map(|a| (*a as u32).to_le_bytes()).collect::<Vec<_>>())?;
    let mut r = cx.record();
    r.dispatch(&per_path, pg, &[q1.addr as u32, PATHS as u32, 1])?;
    let t = std::time::Instant::now();
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let path_write_ms = t.elapsed().as_secs_f64() * 1e3;
    let (pcb, pdb) = (pcmds.bytes()?.to_vec(), pdepth.bytes()?.to_vec());
    let (cob, dob) = (coffs.bytes()?.to_vec(), doffs.bytes()?.to_vec());
    let _ = (&mut ctab, &mut dtab);

    let (mut ok, mut path_ok, mut cmds_total, mut depth_total) = (0, 0, 0, 0);
    let mut first_bad = None;
    for (r, row) in coarse.rows.iter().enumerate().take(nrows) {
        let want_c: Vec<(u16, u16, u32, u32)> = row
            .cmds
            .iter()
            .filter_map(|c| match *c {
                Cmd::PaintFill { span, alpha, attrs } => Some((span.x, span.width, alpha.unwrap_or(u32::MAX), attrs)),
                _ => None,
            })
            .collect();
        let want_d: Vec<(u16, u16, u32)> = row.depth.iter().map(|d| (d.buckets.start, d.buckets.end, d.attrs)).collect();
        // the row's commands: its chunks' in turn
        let (co, dn) = offs[r * nchunks];
        let (rc, rd) = cv[r * nchunks..][..nchunks].iter().fold((0, 0), |(a, b), (c, d)| (a + c, b + d));
        let got_c: Vec<(u16, u16, u32, u32)> = (0..rc as usize).map(|i| {
            let at = (co as usize + i) * 16;
            let xw = word(&cb, at);
            (xw as u16, (xw >> 16) as u16, word(&cb, at + 4), word(&cb, at + 8))
        }).collect();
        let got_d: Vec<(u16, u16, u32)> = (0..rd as usize).map(|i| {
            let at = (dn as usize + i) * 8;
            let se = word(&db, at);
            (se as u16, (se >> 16) as u16, word(&db, at + 4))
        }).collect();
        // the per-path version's row: its table row's slots, from (row, 0) to (row + 1, 0)
        let span = |o: &[u8], t: u32| -> (usize, usize) {
            let a = word(o, 4 * r * PATHS) as usize;
            let b = if r + 1 < nrows { word(o, 4 * (r + 1) * PATHS) as usize } else { t as usize };
            (a, b)
        };
        let ((ca, cz), (da, dz)) = (span(&cob, pc), span(&dob, pd));
        let pgot_c: Vec<(u16, u16, u32, u32)> = (ca..cz).map(|i| {
            let xw = word(&pcb, i * 16);
            (xw as u16, (xw >> 16) as u16, word(&pcb, i * 16 + 4), word(&pcb, i * 16 + 8))
        }).collect();
        let pgot_d: Vec<(u16, u16, u32)> = (da..dz).map(|i| {
            let se = word(&pdb, i * 8);
            (se as u16, (se >> 16) as u16, word(&pdb, i * 8 + 4))
        }).collect();
        if pgot_c == want_c && pgot_d == want_d {
            path_ok += 1;
        }
        cmds_total += want_c.len();
        depth_total += want_d.len();
        let only_fills = want_c.len() == row.cmds.len();
        if only_fills && got_c == want_c && got_d == want_d {
            ok += 1;
        } else if first_bad.is_none() {
            first_bad = Some((r, got_c.len(), want_c.len(), row.cmds.len(), got_c.iter().zip(&want_c).position(|(g, w)| g != w), got_d.len(), want_d.len()));
        }
    }
    println!("{ok} of {nrows} rows' commands bit for bit ({cmds_total} paint fills, {depth_total} depth commands; {PATHS} paths, {opaque_paths} opaque, {nstrip_total} strips)");
    if let Some(b) = first_bad {
        println!("  first wrong: row {}: {} paint fills (want {}, of {} commands), first differing at {:?}; {} depth commands (want {})", b.0, b.1, b.2, b.3, b.4, b.5, b.6);
    }
    println!("a path a lane: {path_ok} of {nrows} rows bit for bit");
    println!("coarse: a row and a chunk a lane, pass 1 {gpu_ms:.2} ms; a path a lane, count + table scans {path_count_ms:.2} ms, write {path_write_ms:.2} ms");
    let pass = ok == nrows && path_ok == nrows;
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
