use detour::base::vello_common::flatten::{self, FlattenCtx, Line, Point};
use detour::base::vello_common::fearless_simd::Level;
use detour::base::vello_common::geometry::RectU16;
use detour::base::vello_common::kurbo::{Affine, BezPath, PathEl, Point as KPoint};
use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

use detour::xe3::ScanPlan;

const PATHS: usize = 2000;
const BOX: RectU16 = RectU16 { x0: 0, y0: 0, x1: 1000, y1: 800 };

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
        KPoint::new(self.unit() * 1300.0 - 150.0, self.unit() * 1100.0 - 150.0)
    }

    fn path(&mut self) -> BezPath {
        let mut p = BezPath::new();
        let mut last = self.point();
        p.move_to(last);
        for _ in 0..(1 + self.next() % 12) {
            let to = self.point();
            match self.next() % 10 {
                0 => {
                    last = self.point();
                    p.move_to(last);
                }
                1 => p.close_path(),
                2 | 3 => p.line_to(to),
                4 => p.quad_to(last.lerp(to, self.unit()), to),
                5 => p.quad_to(last, to),
                6 => p.quad_to(self.point(), to),
                7 => p.curve_to(last.lerp(to, self.unit()), last.lerp(to, self.unit()), to),
                8 => p.curve_to(last, to, to),
                _ => p.curve_to(self.point(), self.point(), to),
            }
            if !matches!(p.elements().last(), Some(PathEl::MoveTo(_))) {
                last = to;
            }
        }
        p
    }

    fn affine(&mut self) -> Affine {
        let s = 0.25 + self.unit() * 3.0;
        Affine::rotate(self.unit() * 6.3).then_scale(s).then_translate((self.unit() * 400.0 - 200.0, self.unit() * 400.0 - 200.0).into())
    }
}

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

fn bits(l: &Line) -> [u32; 4] {
    [l.p0.x.to_bits(), l.p0.y.to_bits(), l.p1.x.to_bits(), l.p1.y.to_bits()]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("../src/paths/flatten.mlir"), "flatten")?;
    let scan = cx.kernel(&codegen, include_str!("../src/paths/scan.mlir"), "scan")?;
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let paths: Vec<(BezPath, Affine)> = (0..PATHS).map(|_| (rng.path(), rng.affine())).collect();
    let mut bytes = Vec::new();
    let mut firsts = Vec::new();
    for (path, affine) in &paths {
        let (els, moves) = elements(path);
        firsts.push((bytes.len() / 128, els.len(), moves.iter().any(|m| (*affine * *m).is_nan())));
        for (kind, p) in els {
            let mut e = [0u8; 128];
            e[..4].copy_from_slice(&kind.to_le_bytes());
            for (j, v) in p.iter().flat_map(|q| [q.x, q.y]).chain(affine.as_coeffs()).enumerate() {
                e[8 + 8 * j..][..8].copy_from_slice(&v.to_le_bytes());
            }
            bytes.extend(e);
        }
    }
    let n = bytes.len() / 128;
    let top = ((BOX.y0 / 4) * 4) as f64;
    let consts: Vec<u8> = [BOX.x0 as f64, top, BOX.x1 as f64, BOX.y1 as f64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let elems = cx.buffer(&bytes)?;
    let cbuf = cx.buffer(&consts)?;
    let mut counts = cx.buffer(&vec![0; n * 4])?;
    let scratch = cx.buffer(&vec![0; n * 640])?;
    let mut nans = cx.buffer(&vec![0; n * 4])?;
    let groups = [n.div_ceil(16) as u32, 1];
    let mut perm: Vec<u32> = (0..n as u32).collect();
    perm.sort_by_key(|&e| bytes[e as usize * 128]);
    let pbuf_perm = cx.buffer(&perm.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<_>>())?;
    let (ea, ca, na, sa, za, pa) = (elems.addr, cbuf.addr, counts.addr, scratch.addr, nans.addr, pbuf_perm.addr);
    let addrs = |offsets: u64, pts: u64, lines: u64| [ea, ca, na, offsets, pts, sa, lines, za, pa].iter().flat_map(|a| (*a as u32).to_le_bytes()).collect::<Vec<_>>();
    let obuf = cx.buffer(&vec![0; n * 4])?;
    let mut plan = ScanPlan::new(&mut cx, counts.addr, obuf.addr, n)?;
    let p0 = cx.buffer(&addrs(obuf.addr, 0, 0))?;
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p0.addr as u32, n as u32, 0])?;
    plan.record(&mut r, &scan)?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let total = plan.total()? as usize;
    let pbuf = cx.buffer(&vec![0xa5; total.max(1) * 8])?;
    let mut lbuf = cx.buffer(&vec![0xa5; total.max(1) * 16])?;
    let p1 = cx.buffer(&addrs(obuf.addr, pbuf.addr, lbuf.addr))?;
    let cs: Vec<i32> = counts.bytes()?[..n * 4].chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let mut r = cx.record();
    r.dispatch(&kernel, groups, &[p1.addr as u32, n as u32, 1])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let all: Vec<Line> = lbuf.bytes()?[..total * 16]
        .chunks(16)
        .map(|c| {
            let f = |k: usize| f32::from_le_bytes(c[4 * k..][..4].try_into().unwrap());
            Line::new(Point::new(f(0), f(1)), Point::new(f(2), f(3)))
        })
        .collect();
    let nan: Vec<bool> = nans.bytes()?[..n * 4].chunks(4).map(|c| c != [0; 4]).collect();
    let mut obuf = obuf;
    let offsets: Vec<u32> = obuf.bytes()?[..n * 4].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();

    let (mut paths_ok, mut lines_total) = (0, 0);
    let points_total = total;
    let mut first_bad = None;
    for (k, ((path, affine), (first, len, moved_nan))) in paths.iter().zip(&firsts).enumerate() {
        let mut want = Vec::new();
        flatten::fill(level, path.iter(), *affine, &mut want, &mut FlattenCtx::default(), BOX);
        let els = *first..first + len;
        let got: Vec<Line> = if *moved_nan || els.clone().any(|e| nan[e]) { Vec::new() } else { els.flat_map(|e| all[offsets[e] as usize..][..cs[e].max(0) as usize].to_vec()).collect() };
        lines_total += want.len();
        if got.len() == want.len() && got.iter().zip(&want).all(|(g, w)| bits(g) == bits(w)) {
            paths_ok += 1;
        } else if first_bad.is_none() {
            let at = got.iter().zip(&want).position(|(g, w)| bits(g) != bits(w));
            first_bad = Some((k, got.len(), want.len(), at.map(|i| (got[i], want[i]))));
        }
    }
    println!("{paths_ok} of {PATHS} paths' lines bit for bit ({lines_total} lines, {points_total} lines from the GPU)");
    if let Some((k, g, w, at)) = first_bad {
        println!("  first wrong: path {k}: {g} lines, want {w}; first differing {at:?}");
    }
    let ok = paths_ok == PATHS;
    println!("{} (device {:#x})", if ok { "ok" } else { "FAILED" }, xe.info().device_id);
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
