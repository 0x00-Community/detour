use detour::base::vello_common::fearless_simd::Level;
use detour::base::vello_common::flatten::{self, FlattenCtx, Line};
use detour::base::vello_common::geometry::RectU16;
use detour::base::vello_common::kurbo::{Affine, BezPath, Point as KPoint};
use detour::base::vello_common::tile::Tiles;
use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const PATHS: usize = 2000;
const WIDTH: u16 = 1000;
const HEIGHT: u16 = 800;

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
    let kernel = cx.kernel(&codegen, include_str!("../src/paths/tiles.mlir"), "tiles")?;
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let bbox = RectU16 { x0: 0, y0: 0, x1: WIDTH, y1: HEIGHT };
    let lines: Vec<Vec<Line>> = (0..PATHS)
        .map(|_| {
            let mut l = Vec::new();
            flatten::fill(level, rng.path().iter(), Affine::IDENTITY, &mut l, &mut FlattenCtx::default(), bbox);
            l
        })
        .collect();
    let (cols, rows) = (u32::from(WIDTH.div_ceil(4)), u32::from(HEIGHT.div_ceil(4)));
    let words = rows.div_ceil(32);
    let stride = (rows * 16 + rows * 4 + words * 4 + 4).next_multiple_of(16);

    let mut all = Vec::new();
    let mut paths = Vec::new();
    for l in &lines {
        paths.extend([(all.len() / 16) as u32, l.len() as u32]);
        for line in l {
            for v in [line.p0.x, line.p0.y, line.p1.x, line.p1.y] {
                all.extend(v.to_le_bytes());
            }
        }
    }
    all.resize(all.len().max(16), 0);
    let lbuf = cx.buffer(&all)?;
    let pbuf = cx.buffer(&paths.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let cbuf = cx.buffer(&[cols, rows, stride, 0, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let mut counts = cx.buffer(&vec![0; PATHS * 4])?;
    let mut wbuf = cx.buffer(&vec![0; PATHS * stride as usize])?;
    let groups = [PATHS.div_ceil(16) as u32, 1];
    let (la, pa, ca, na, wa) = (lbuf.addr, pbuf.addr, cbuf.addr, counts.addr, wbuf.addr);
    let addrs = |offsets: u64, tiles: u64| [la, pa, ca, na, offsets, tiles, wa].iter().flat_map(|a| (*a as u32).to_le_bytes()).collect::<Vec<_>>();
    let p0 = cx.buffer(&addrs(0, 0))?;
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p0.addr as u32, PATHS as u32, 0])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let cs: Vec<u32> = counts.bytes()?[..PATHS * 4].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut offsets = Vec::with_capacity(PATHS);
    let mut total = 0u32;
    for c in &cs {
        offsets.push(total);
        total += c;
    }
    let obuf = cx.buffer(&offsets.iter().flat_map(|o| o.to_le_bytes()).collect::<Vec<_>>())?;
    let mut tbuf = cx.buffer(&vec![0xa5; total.max(1) as usize * 8])?;
    let p1 = cx.buffer(&addrs(obuf.addr, tbuf.addr))?;
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p1.addr as u32, PATHS as u32, 1])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let tb = tbuf.bytes()?.to_vec();
    let wb = wbuf.bytes()?.to_vec();
    let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..][..4].try_into().unwrap());

    let (mut ok, mut tiles_total, mut culled_paths) = (0, 0, 0);
    let mut first_bad = None;
    let mut tiles = Tiles::new(level, WIDTH, HEIGHT);
    for (p, l) in lines.iter().enumerate() {
        let culled = tiles.make_tiles_analytic_aa(level, l, WIDTH, HEIGHT);
        tiles.sort_tiles();
        let want: Vec<(u32, u32)> = tiles.iter().map(|t| (t.packed_winding_line_idx, u32::from(t.y) << 16 | u32::from(t.x))).collect();
        let got: Vec<(u32, u32)> = (0..cs[p] as usize).map(|i| {
            let at = (offsets[p] as usize + i) * 8;
            (word(&tb, at), word(&tb, at + 4))
        }).collect();
        let base = p * stride as usize;
        let w = &tiles.windings;
        let partial_ok = (0..rows as usize).all(|r| (0..4).all(|j| word(&wb, base + 16 * r + 4 * j) == w.partial[r][j].to_bits()));
        let coarse_ok = (0..rows as usize).all(|r| word(&wb, base + 16 * rows as usize + 4 * r) as i16 == w.coarse[r]);
        let active_ok = (0..words as usize).all(|k| word(&wb, base + 20 * rows as usize + 4 * k) == w.active[k]);
        let culled_ok = (word(&wb, base + 20 * rows as usize + 4 * words as usize) != 0) == culled;
        tiles_total += want.len();
        culled_paths += usize::from(culled);
        if got == want && partial_ok && coarse_ok && active_ok && culled_ok {
            ok += 1;
        } else if first_bad.is_none() {
            first_bad = Some((p, got.len(), want.len(), got.iter().zip(&want).position(|(g, w)| g != w), partial_ok, coarse_ok, active_ok, culled_ok));
        }
    }
    println!("{ok} of {PATHS} paths' tiles and windings bit for bit ({tiles_total} tiles; {culled_paths} paths with lines left of the viewport)");
    if let Some(b) = first_bad {
        println!("  first wrong: path {}: {} tiles, want {}; first differing at {:?}; partial {} coarse {} active {} culled {}", b.0, b.1, b.2, b.3, b.4, b.5, b.6, b.7);
    }
    let pass = ok == PATHS;
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
