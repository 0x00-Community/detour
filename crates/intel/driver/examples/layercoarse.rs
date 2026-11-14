use detour::base::vello_common::fearless_simd::Level;
use detour::base::vello_common::flatten::{self, FlattenCtx, Line};
use detour::base::vello_common::geometry::RectU16;
use detour::base::vello_common::kurbo::{Affine, BezPath, Point as KPoint};
use detour::base::vello_common::peniko::Fill;
use detour::base::vello_common::strip::{self, Strip};
use detour::base::vello_common::tile::Tiles;
use detour::base::vello_common::util::strip_bbox;
use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context, Fine, Tiles as FineTiles};
use detour::xe::Xe;
use detour::vello_cpu::detour::Cmd;
use detour::vello_cpu::peniko::Color;
use detour::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};

use detour::xe3::ScanPlan;

const FILLS: usize = 3000;
const WIDTH: u16 = 1280;
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

    fn near(&mut self, ox: f64, oy: f64, size: f64) -> KPoint {
        KPoint::new(ox + self.unit() * size, oy + self.unit() * size)
    }

    fn glyph(&mut self, size: f64) -> BezPath {
        let (ox, oy) = (self.unit() * f64::from(WIDTH) - size * 0.25, self.unit() * f64::from(HEIGHT) - size * 0.25);
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

#[derive(Default)]
struct Scene {
    strips: Vec<u32>,
    alphas: Vec<u8>,
    events: Vec<u32>,
    layers: Vec<u32>,
    prow: Vec<u32>,
    prcount: Vec<u32>,
    paints: u32,
    pops: u32,
}

impl Scene {
    fn strips(&mut self, level: Level, tiles: &mut Tiles, path: &BezPath, rule: Fill) -> (u32, u32, u32, Option<RectU16>) {
        let bbox = RectU16 { x0: 0, y0: 0, x1: WIDTH, y1: HEIGHT };
        let mut lines: Vec<Line> = Vec::new();
        flatten::fill(level, path.iter(), Affine::IDENTITY, &mut lines, &mut FlattenCtx::default(), bbox);
        tiles.make_tiles_analytic_aa(level, &lines, WIDTH, HEIGHT);
        tiles.sort_tiles();
        let (mut s, mut al): (Vec<Strip>, Vec<u8>) = (Vec::new(), Vec::new());
        strip::render(level, tiles, &mut s, &mut al, rule, None, &lines);
        let first = (self.strips.len() / 2) as u32;
        for st in &s {
            self.strips.extend([u32::from(st.x) | u32::from(st.y) << 16, st.alpha_idx() | u32::from(st.fill_gap()) << 31]);
        }
        let base = self.alphas.len() as u32;
        self.alphas.extend(&al);
        (first, s.len() as u32, base, strip_bbox(&s))
    }

    fn bx(&self, m: i32) -> [u32; 4] {
        if m < 0 {
            return [0, 0, u32::from(WIDTH.div_ceil(4)), u32::from(HEIGHT.div_ceil(4))];
        }
        let l = &self.layers[8 * m as usize..];
        [l[1], l[2], l[3], l[4]]
    }

    fn rows(&mut self, first: u32, last: u32) {
        let n = if last >= first { last - first + 1 } else { 0 };
        self.prow.extend([first, n]);
        self.prcount.push(n);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let src = [include_str!("../src/paths/rowcoarse.mlir"), include_str!("../src/paths/layers.mlir")].concat();
    let coarse_k = cx.kernel(&codegen, &src, "coarse_ev")?;
    let scan = cx.kernel(&codegen, include_str!("../src/paths/scan.mlir"), "scan")?;
    let pairs_src = include_str!("../src/paths/pairs.mlir");
    let [pfill_k, pplace_k, prank_k, pgather_k] = ["pfill", "pplace", "prank", "pgather"].map(|f| cx.kernel(&codegen, pairs_src, f));
    let (pfill_k, pplace_k, prank_k, pgather_k) = (pfill_k?, pplace_k?, prank_k?, pgather_k?);
    let bins_k = cx.kernel(&codegen, include_str!("../src/paths/bins.mlir"), "bins")?;
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut rng = Rng(0x0dd_ba11_cafe_f00d);

    let mut ctx = RenderContext::new(WIDTH, HEIGHT);
    let mut tiles = Tiles::new(level, WIDTH, HEIGHT);
    let mut sc = Scene::default();
    let mut open: Vec<(i32, (u32, u32, u32))> = Vec::new();
    let (mut clips, mut max_depth) = (0, 0);
    let pop = |sc: &mut Scene, ctx: &mut RenderContext, (m, (cf, cn, ca)): (i32, (u32, u32, u32))| {
        ctx.pop_layer();
        sc.events.extend([1, cf, cn, ca, sc.pops, 0, m as u32, 0]);
        sc.pops += 1;
        let b = sc.bx(m);
        sc.rows(b[1], b[3].wrapping_sub(1).min(b[3].max(1) - 1));
        if b[1] >= b[3] {
            let n = sc.prcount.len() - 1;
            sc.prcount[n] = 0;
            sc.prow[2 * n + 1] = 0;
        }
    };
    for _ in 0..FILLS {
        if open.len() < 3 && rng.next() % 40 == 0 {
            let parent = open.last().map_or(-1, |o| o.0);
            let clip = rng.next() % 3 != 0;
            let opacity = if rng.next() % 2 == 0 { 1.0 } else { 0.3 + rng.unit() as f32 * 0.6 };
            let m = (sc.layers.len() / 8) as i32;
            let pb = sc.bx(parent);
            let (strips, own) = if clip {
                let size = 150.0 + rng.unit() * 450.0;
                let path = rng.glyph(size);
                ctx.set_fill_rule(Fill::NonZero);
                ctx.push_layer(Some(&path), None, Some(opacity), None, None);
                let (f, n, a, bb) = sc.strips(level, &mut tiles, &path, Fill::NonZero);
                let bb = bb.unwrap_or(RectU16::ZERO);
                clips += 1;
                ((f, n, a), Some([u32::from(bb.x0) / 4, u32::from(bb.y0) / 4, u32::from(bb.x1) / 4, u32::from(bb.y1) / 4]))
            } else {
                ctx.push_layer(None, None, Some(opacity), None, None);
                ((0, 0, 0), None)
            };
            let b = own.map_or(pb, |o| [o[0].max(pb[0]), o[1].max(pb[1]), o[2].min(pb[2]), o[3].min(pb[3])]);
            sc.layers.extend([parent as u32, b[0], b[1], b[2], b[3], 0, 0, 0]);
            open.push((m, strips));
            max_depth = max_depth.max(open.len());
        }
        if !open.is_empty() && rng.next() % 30 == 0 {
            let o = open.pop().expect("open");
            pop(&mut sc, &mut ctx, o);
        }
        let big = rng.next() % 40 == 0;
        let size = if big { 100.0 + rng.unit() * 400.0 } else { 14.0 };
        let path = rng.glyph(size);
        let rule = if rng.next() % 2 == 0 { Fill::NonZero } else { Fill::EvenOdd };
        let opaque = rng.next() % 2 == 0;
        let a = if opaque { 255 } else { 64 + (rng.next() % 128) as u8 };
        ctx.set_fill_rule(rule);
        ctx.set_paint(Color::from_rgba8((rng.next() % 256) as u8, (rng.next() % 256) as u8, (rng.next() % 256) as u8, a));
        ctx.fill_path(&path);
        let (f, n, ab, _) = sc.strips(level, &mut tiles, &path, rule);
        let m = open.last().map_or(-1, |o| o.0);
        sc.events.extend([0, f, n, ab, sc.paints, u32::from(opaque && m < 0), m as u32, 0]);
        sc.paints += u32::from(n > 0);
        let b = sc.bx(m);
        let row = |i: u32| sc.strips[2 * (f + i) as usize] >> 18;
        let (r0, r1) = if n >= 2 { (row(0).max(b[1]), row(n - 2).min(b[3].max(1) - 1)) } else { (1, 0) };
        let (r0, r1) = if b[1] >= b[3] { (1, 0) } else { (r0, r1) };
        sc.rows(r0, r1);
    }
    while let Some(o) = open.pop() {
        pop(&mut sc, &mut ctx, o);
    }
    let mut reference = Pixmap::new(WIDTH, HEIGHT);
    ctx.render_with(&mut reference, &mut Resources::new(), RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let mut resources = Resources::new();
    let coarse = ctx.coarse(&mut resources).ok_or("no coarse rasterization")?;
    let nev = sc.events.len() / 8;
    let nl = (sc.layers.len() / 8).max(1);
    sc.layers.resize(sc.layers.len().max(8), 0);
    sc.strips.resize(sc.strips.len().max(2), 0);
    println!("{FILLS} fills, {} layers ({clips} clipped, {max_depth} deep at most), {nev} events", sc.layers.len() / 8);

    let (cols, rows) = (u32::from(WIDTH.div_ceil(4)), u32::from(HEIGHT.div_ceil(4)));
    let nrows = rows as usize;
    let words = |v: &[u32]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    let zeros = |n: usize| vec![0u8; n.next_multiple_of(4).max(4) * 4];
    let strips_b = cx.buffer(&words(&sc.strips))?;
    let events_b = cx.buffer(&words(&sc.events))?;
    let layers_b = cx.buffer(&words(&sc.layers))?;
    let prow = cx.buffer(&words(&sc.prow))?;
    let prcount = cx.buffer(&words(&sc.prcount))?;
    let pairbase = cx.buffer(&zeros(nev))?;
    let mut prplan = ScanPlan::new(&mut cx, prcount.addr, pairbase.addr, nev)?;
    let first = cx.buffer(&vec![0xff; nrows * nl * 4])?;
    let (rowcount, rowstart, rowfill) = (cx.buffer(&zeros(nrows + 1))?, cx.buffer(&zeros(nrows + 1))?, cx.buffer(&zeros(nrows + 1))?);
    let rsplan = ScanPlan::new(&mut cx, rowcount.addr, rowstart.addr, nrows + 1)?;
    let mut r = cx.record();
    prplan.record(&mut r, &scan)?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let npairs = prplan.total()? as usize;
    let (pairpath, bucket, sorted) = (cx.buffer(&zeros(npairs))?, cx.buffer(&zeros(npairs))?, cx.buffer(&zeros(npairs))?);
    let (ctab, dtab) = (cx.buffer(&zeros(npairs))?, cx.buffer(&zeros(npairs))?);
    let (scb, sdb, coffs, doffs) = (cx.buffer(&zeros(npairs + 1))?, cx.buffer(&zeros(npairs + 1))?, cx.buffer(&zeros(npairs + 1))?, cx.buffer(&zeros(npairs + 1))?);
    let mut cplan = ScanPlan::new(&mut cx, scb.addr, coffs.addr, npairs + 1)?;
    let mut dplan = ScanPlan::new(&mut cx, sdb.addr, doffs.addr, npairs + 1)?;
    let consts = cx.buffer(&words(&[cols, rows, nev as u32, nl as u32, 0, prow.addr as u32, pairbase.addr as u32, sorted.addr as u32]))?;
    let pp = cx.buffer(&words(&[strips_b.addr, events_b.addr, consts.addr, prow.addr, prcount.addr, pairbase.addr, pairpath.addr, rowcount.addr, rowstart.addr, rowfill.addr, bucket.addr, sorted.addr, ctab.addr, dtab.addr, scb.addr, sdb.addr].map(|a| a as u32)))?;
    let cparams = |cmds: u64, depth: u64| words(&[strips_b.addr, events_b.addr, layers_b.addr, consts.addr, first.addr, ctab.addr, dtab.addr, coffs.addr, doffs.addr, cmds, depth].map(|a| a as u32));
    let cq0 = cx.buffer(&cparams(0, 0))?;
    let eg = [nev.div_ceil(16) as u32, 1];
    let prg = [npairs.div_ceil(16) as u32, 1];
    let t = std::time::Instant::now();
    let mut r = cx.record();
    r.dispatch(&coarse_k, eg, &[cq0.addr as u32, nev as u32, 2])?;
    r.dispatch(&pfill_k, eg, &[pp.addr as u32, nev as u32])?;
    rsplan.record(&mut r, &scan)?;
    r.dispatch(&pplace_k, prg, &[pp.addr as u32, npairs as u32])?;
    r.dispatch(&prank_k, prg, &[pp.addr as u32, npairs as u32])?;
    r.dispatch(&coarse_k, eg, &[cq0.addr as u32, nev as u32, 0])?;
    r.dispatch(&pgather_k, prg, &[pp.addr as u32, npairs as u32])?;
    cplan.record(&mut r, &scan)?;
    dplan.record(&mut r, &scan)?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let (ncmds, ndepth) = (cplan.total()? as usize, dplan.total()? as usize);
    let mut cmds = cx.buffer(&vec![0xa5; ncmds.max(1) * 16])?;
    let mut depth = cx.buffer(&vec![0xa5; ndepth.max(1) * 8])?;
    let cq1 = cx.buffer(&cparams(cmds.addr, depth.addr))?;
    let mut r = cx.record();
    r.dispatch(&coarse_k, eg, &[cq1.addr as u32, nev as u32, 1])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let coarse_ms = t.elapsed().as_secs_f64() * 1e3;

    let word = |b: &[u8], at: usize| u32::from_le_bytes(b[at..][..4].try_into().unwrap());
    let (mut coffs, mut doffs, mut rowstart) = (coffs, doffs, rowstart);
    let (cob, dob, rsb, cb, db) = (coffs.bytes()?.to_vec(), doffs.bytes()?.to_vec(), rowstart.bytes()?.to_vec(), cmds.bytes()?.to_vec(), depth.bytes()?.to_vec());
    let (mut ok, mut ncmd, mut kinds) = (0, 0, [0usize; 4]);
    let mut first_bad = None;
    for (r, row) in coarse.rows.iter().enumerate().take(nrows) {
        let want_c: Vec<(u32, u32, u32, u32)> = row
            .cmds
            .iter()
            .map(|c| match *c {
                Cmd::PaintFill { span, alpha, attrs } => (u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), attrs, 0),
                Cmd::LayerFill { span, alpha, attrs } => (u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), attrs, 1),
                Cmd::PushBuf(_) => (0, u32::MAX, 0, 2),
                Cmd::PopBuf => (0, u32::MAX, 0, 3),
            })
            .collect();
        let want_d: Vec<(u32, u32)> = row.depth.iter().map(|d| (u32::from(d.buckets.start) | u32::from(d.buckets.end) << 16, d.attrs)).collect();
        let (p0, p1) = (word(&rsb, 4 * r) as usize, word(&rsb, 4 * (r + 1)) as usize);
        let (ca, cz, da, dz) = (word(&cob, 4 * p0) as usize, word(&cob, 4 * p1) as usize, word(&dob, 4 * p0) as usize, word(&dob, 4 * p1) as usize);
        let got_c: Vec<(u32, u32, u32, u32)> = (ca..cz).map(|i| (word(&cb, 16 * i), word(&cb, 16 * i + 4), word(&cb, 16 * i + 8), word(&cb, 16 * i + 12))).collect();
        let got_d: Vec<(u32, u32)> = (da..dz).map(|i| (word(&db, 8 * i), word(&db, 8 * i + 4))).collect();
        for c in &want_c {
            kinds[c.3 as usize] += 1;
        }
        ncmd += want_c.len();
        if got_c == want_c && got_d == want_d {
            ok += 1;
        } else if first_bad.is_none() {
            let at = got_c.iter().zip(&want_c).position(|(g, w)| g != w);
            first_bad = Some((r, got_c.len(), want_c.len(), at.map(|i| (got_c[i], want_c[i])), got_d.len(), want_d.len()));
        }
    }
    println!("{ok} of {nrows} rows' commands bit for bit ({ncmd}: {} paint fills, {} layer fills, {} pushes, {} pops; {} depth; {npairs} (row, event) pairs) in {coarse_ms:.2} ms", kinds[0], kinds[1], kinds[2], kinds[3], ndepth);
    if let Some(b) = first_bad {
        println!("  first wrong: row {}: {} commands (want {}), first differing {:?}; {} depth (want {})", b.0, b.1, b.2, b.3, b.4, b.5);
    }

    let nb = usize::from(WIDTH).div_ceil(128);
    let nbl = nrows * nb;
    let (bcc, bdc, bco, bdo) = (cx.buffer(&zeros(nbl))?, cx.buffer(&zeros(nbl))?, cx.buffer(&zeros(nbl))?, cx.buffer(&zeros(nbl))?);
    let mut bcplan = ScanPlan::new(&mut cx, bcc.addr, bco.addr, nbl)?;
    let mut bdplan = ScanPlan::new(&mut cx, bdc.addr, bdo.addr, nbl)?;
    let bconsts = cx.buffer(&words(&[rows, nev as u32, nb as u32, rowstart.addr as u32, sc.paints]))?;
    let bparams = |c: u64, d: u64, rw: u64| words(&[cmds.addr, depth.addr, coffs.addr, doffs.addr, 0, 0, bconsts.addr, bcc.addr, bdc.addr, bco.addr, bdo.addr, c, d, rw].map(|a| a as u32));
    let bq0 = cx.buffer(&bparams(0, 0, 0))?;
    let bg = [nb.div_ceil(16) as u32, rows];
    let mut r = cx.record();
    r.dispatch(&bins_k, bg, &[bq0.addr as u32, nbl as u32, 0])?;
    bcplan.record(&mut r, &scan)?;
    bdplan.record(&mut r, &scan)?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let (nbc, nbd) = (bcplan.total()? as usize, bdplan.total()? as usize);
    let (bcmds, bdepth, brows) = (cx.buffer(&zeros(nbc * 4))?, cx.buffer(&zeros(nbd * 2))?, cx.buffer(&zeros(nbl * 4))?);
    let bq1 = cx.buffer(&bparams(bcmds.addr, bdepth.addr, brows.addr))?;
    let fine = Fine::new(&mut cx, &codegen)?;
    let ftiles = FineTiles::new(&coarse, &[], &[])?;
    let mut fbuf = cx.buffer(&vec![0; ftiles.bytes()])?;
    let faddr = fbuf.addr;
    let alphas_same = sc.alphas == coarse.alphas;
    let mut ab = sc.alphas.clone();
    ab.resize(ab.len().next_multiple_of(4).max(4), 0);
    let alphas_b = cx.buffer(&ab)?;
    let at = ftiles.write(fbuf.bytes()?, faddr, &[])?.with_gpu(brows.addr, bcmds.addr, bdepth.addr, alphas_b.addr);
    let mut r = cx.record();
    r.dispatch(&bins_k, bg, &[bq1.addr as u32, nbl as u32, 1])?;
    fine.record(&mut r, &at)?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let got = fbuf.bytes()?[(at.out - faddr) as usize..][..ftiles.pixels()].to_vec();
    let (mut want_rows, mut want_cmds, mut want_depth) = (Vec::<u32>::new(), Vec::<u32>::new(), Vec::<u32>::new());
    for row in coarse.rows.iter().take(nrows) {
        for b in 0..nb as u32 {
            let (f0, d0) = (want_cmds.len() / 4, want_depth.len() / 2);
            let ov = |x: u16, w: u16| u32::from(x) < (b + 1) * 128 && u32::from(x) + u32::from(w) > b * 128;
            for c in &row.cmds {
                match *c {
                    Cmd::PaintFill { span, alpha, attrs } if ov(span.x, span.width) => want_cmds.extend([0, u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), attrs]),
                    Cmd::LayerFill { span, alpha, attrs } if ov(span.x, span.width) => want_cmds.extend([1, u32::from(span.x) | u32::from(span.width) << 16, alpha.unwrap_or(u32::MAX), sc.paints + attrs]),
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
    let dw = |b: &[u8], n: usize| b[..n * 4].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
    let (mut brows, mut bcmds) = (brows, bcmds);
    let (gr, gc) = (dw(brows.bytes()?, nbl * 4), dw(bcmds.bytes()?, nbc * 4));
    let rows_at = gr.chunks(4).zip(want_rows.chunks(4)).position(|(g, w)| g != w);
    let cmd_at = gc.chunks(4).zip(want_cmds.chunks(4)).position(|(g, w)| g != w);
    if let Some(i) = cmd_at {
        for j in i.saturating_sub(4)..(i + 6).min(nbc) {
            println!("  {j}: {:?} want {:?}", &gc[4 * j..][..4], &want_cmds[4 * j..][..4]);
        }
        let r = gr.chunks(4).position(|w| (w[0] as usize) <= i && i < (w[0] + w[1]) as usize);
        println!("  in (row, bucket) {r:?}: {:?}", r.map(|r| &gr[4 * r..][..4]));
    }
    println!("bins: {} commands (want {}), rows {}, commands {}", nbc, want_cmds.len() / 4, rows_at.map_or("same".into(), |i| format!("differ at {i}: {:?} want {:?}", &gr[4 * i..][..4], &want_rows[4 * i..][..4])), cmd_at.map_or("same".into(), |i| format!("differ at {i}: {:?} want {:?}", &gc[4 * i..][..4], &want_cmds[4 * i..][..4])));
    let want = reference.data_as_u8_slice();
    let px_bad = got.chunks(4).zip(want.chunks(4)).filter(|(a, b)| a != b).count();
    println!("{} pixels, {px_bad} differ from vello_cpu's ({} alpha bytes, {})", got.len() / 4, sc.alphas.len(), if alphas_same { "vello_cpu's" } else { "NOT vello_cpu's" });
    let pass = ok == nrows && px_bad == 0 && alphas_same;
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
