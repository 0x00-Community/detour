//! vello_common's strip rendering on Xe3 against vello_common's own, bit for bit (roadmap R8.3): random paths (many
//! past the viewport's left and top), flattened and tiled by vello_common, their strips and alpha bytes
//! (`strip::render`, non-zero or even-odd) from the sorted tiles and the culled windings. A lane renders one path
//! (`stripgen/stripgen.mlir`: pass 0 counts its strips and alpha bytes, pass 1 writes them at a prefix sum of those).

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

const PATHS: usize = 2000;
const WIDTH: u16 = 1000;
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

    fn point(&mut self) -> KPoint {
        KPoint::new(self.unit() * 1400.0 - 300.0, self.unit() * 1200.0 - 300.0)
    }

    fn path(&mut self) -> BezPath {
        let mut p = BezPath::new();
        let mut last = self.point();
        p.move_to(last);
        for _ in 0..(1 + self.next() % 10) {
            let to = match self.next() % 6 {
                0 => KPoint::new(last.x, self.unit() * 1200.0 - 300.0),
                1 => KPoint::new(self.unit() * 1400.0 - 300.0, last.y),
                2 => KPoint::new(last.x + self.unit() * 3.0, last.y + self.unit() * 3.0),
                _ => self.point(),
            };
            match self.next() % 4 {
                0 => p.quad_to(self.point(), to),
                1 => p.curve_to(self.point(), self.point(), to),
                _ => p.line_to(to),
            }
            last = to;
            if self.next() % 7 == 0 {
                p.close_path();
            }
        }
        p
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("../src/paths/stripgen.mlir"), "strips")?;
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let bbox = RectU16 { x0: 0, y0: 0, x1: WIDTH, y1: HEIGHT };
    let (rows, vw) = (u32::from(HEIGHT.div_ceil(4)), u32::from(WIDTH.next_multiple_of(4)));
    let words = rows.div_ceil(32);
    let stride = (rows * 20 + words * 4 + 4).next_multiple_of(16) as usize;

    // vello_common's own: each path's lines, tiles, windings, strips and alphas
    let mut tiles = Tiles::new(level, WIDTH, HEIGHT);
    let (mut lbytes, mut tbytes, mut paths) = (Vec::new(), Vec::new(), Vec::new());
    let mut wbytes = vec![0u8; PATHS * stride];
    let mut want = Vec::new();
    let mut cpu_ms = 0.0;
    for p in 0..PATHS {
        let mut lines: Vec<Line> = Vec::new();
        flatten::fill(level, rng.path().iter(), Affine::IDENTITY, &mut lines, &mut FlattenCtx::default(), bbox);
        let rule = if rng.next() % 2 == 0 { Fill::NonZero } else { Fill::EvenOdd };
        let culled = tiles.make_tiles_analytic_aa(level, &lines, WIDTH, HEIGHT);
        tiles.sort_tiles();
        let (mut strips, mut alphas): (Vec<Strip>, Vec<u8>) = (Vec::new(), Vec::new());
        let t = std::time::Instant::now();
        strip::render(level, &tiles, &mut strips, &mut alphas, rule, None, &lines);
        cpu_ms += t.elapsed().as_secs_f64() * 1e3;
        want.push((strips, alphas));
        paths.extend([(lbytes.len() / 16) as u32, (tbytes.len() / 8) as u32, tiles.len(), u32::from(rule == Fill::EvenOdd)]);
        for l in &lines {
            for v in [l.p0.x, l.p0.y, l.p1.x, l.p1.y] {
                lbytes.extend(v.to_le_bytes());
            }
        }
        for t in tiles.iter() {
            tbytes.extend(t.packed_winding_line_idx.to_le_bytes());
            tbytes.extend((u32::from(t.y) << 16 | u32::from(t.x)).to_le_bytes());
        }
        let w = &tiles.windings;
        let base = p * stride;
        let mut put = |at: usize, v: u32| wbytes[base + at..][..4].copy_from_slice(&v.to_le_bytes());
        for r in 0..rows as usize {
            for j in 0..4 {
                put(16 * r + 4 * j, w.partial[r][j].to_bits());
            }
            put(16 * rows as usize + 4 * r, i32::from(w.coarse[r]) as u32);
        }
        for k in 0..words as usize {
            put(20 * rows as usize + 4 * k, w.active[k]);
        }
        put(20 * rows as usize + 4 * words as usize, u32::from(culled));
    }
    lbytes.resize(lbytes.len().max(16), 0);
    tbytes.resize(tbytes.len().max(8), 0);

    let lbuf = cx.buffer(&lbytes)?;
    let pbuf = cx.buffer(&paths.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let tbuf = cx.buffer(&tbytes)?;
    let wbuf = cx.buffer(&wbytes)?;
    let cbuf = cx.buffer(&[rows, vw, stride as u32, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let mut counts = cx.buffer(&vec![0; PATHS * 8])?;
    let scratch = cx.buffer(&vec![0; PATHS * 80])?;
    let groups = [PATHS.div_ceil(16) as u32, 1];
    let fixed = [lbuf.addr, pbuf.addr, tbuf.addr, wbuf.addr, cbuf.addr, counts.addr];
    let addrs = |offsets: u64, strips: u64, alphas: u64| fixed.iter().chain(&[offsets, strips, alphas, scratch.addr]).flat_map(|a| (*a as u32).to_le_bytes()).collect::<Vec<_>>();
    let p0 = cx.buffer(&addrs(0, 0, 0))?;
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p0.addr as u32, PATHS as u32, 0])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let cs: Vec<(u32, u32)> = counts.bytes()?[..PATHS * 8].chunks(8).map(|c| (u32::from_le_bytes(c[..4].try_into().unwrap()), u32::from_le_bytes(c[4..].try_into().unwrap()))).collect();
    let mut offsets = Vec::with_capacity(PATHS);
    let (mut ns, mut na) = (0u32, 0u32);
    for (s, a) in &cs {
        offsets.push((ns, na));
        ns += s;
        na += a;
    }
    let obuf = cx.buffer(&offsets.iter().flat_map(|(s, a)| [s.to_le_bytes(), a.to_le_bytes()].concat()).collect::<Vec<_>>())?;
    let mut sbuf = cx.buffer(&vec![0xa5; ns.max(1) as usize * 8])?;
    let mut abuf = cx.buffer(&vec![0xa5; na.max(16) as usize])?;
    let p1 = cx.buffer(&addrs(obuf.addr, sbuf.addr, abuf.addr))?;
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p1.addr as u32, PATHS as u32, 1])?;
    let t = std::time::Instant::now();
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let sb = sbuf.bytes()?.to_vec();
    let ab = abuf.bytes()?.to_vec();
    let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..][..4].try_into().unwrap());

    let (mut ok, mut strips_total, mut alphas_total) = (0, 0, 0);
    let mut first_bad = None;
    for (p, (strips, alphas)) in want.iter().enumerate() {
        let (so, ao) = offsets[p];
        let got: Vec<(u16, u16, u32, bool)> = (0..cs[p].0 as usize)
            .map(|i| {
                let at = (so as usize + i) * 8;
                let (xy, pk) = (word(&sb, at), word(&sb, at + 4));
                (xy as u16, (xy >> 16) as u16, pk & 0x7fff_ffff, pk >> 31 != 0)
            })
            .collect();
        let want_s: Vec<(u16, u16, u32, bool)> = strips.iter().map(|s| (s.x, s.y, s.alpha_idx(), s.fill_gap())).collect();
        let got_a = &ab[ao as usize..][..cs[p].1 as usize];
        strips_total += strips.len();
        alphas_total += alphas.len();
        if got == want_s && got_a == alphas.as_slice() {
            ok += 1;
        } else if first_bad.is_none() {
            let sat = got.iter().zip(&want_s).position(|(g, w)| g != w);
            let aat = got_a.iter().zip(alphas).position(|(g, w)| g != w);
            first_bad = Some((p, got.len(), want_s.len(), sat.map(|i| (got[i], want_s[i])), got_a.len(), alphas.len(), aat));
        }
    }
    println!("{ok} of {PATHS} paths' strips and alphas bit for bit ({strips_total} strips, {alphas_total} alpha bytes)");
    println!("strips: GPU pass 1 {gpu_ms:.2} ms, vello_common on the CPU {cpu_ms:.2} ms");
    if let Some(b) = first_bad {
        println!("  first wrong: path {}: {} strips, want {}, first differing {:?}; {} alpha bytes, want {}, first differing at {:?}", b.0, b.1, b.2, b.3, b.4, b.5, b.6);
    }
    let pass = ok == PATHS;
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
