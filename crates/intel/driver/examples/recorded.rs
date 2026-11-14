use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context, Fine, PathRaster, Tiles};
use detour::xe::Xe;
use detour::vello_cpu::kurbo::{Affine, BezPath, Cap, Join, Point, Stroke};
use detour::vello_cpu::peniko::{Color, Fill, Gradient};
use detour::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};

const WIDTH: u16 = 1024;
const HEIGHT: u16 = 768;

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

    fn glyph(&mut self, size: f64) -> BezPath {
        let mut p = BezPath::new();
        let near = |r: &mut Rng| Point::new(r.unit() * size, r.unit() * size);
        for _ in 0..(1 + self.next() % 2) {
            p.move_to(near(self));
            for _ in 0..(3 + self.next() % 6) {
                match self.next() % 3 {
                    0 => p.line_to(near(self)),
                    1 => p.quad_to(near(self), near(self)),
                    _ => p.curve_to(near(self), near(self), near(self)),
                }
            }
            p.close_path();
        }
        p
    }

    fn colour(&mut self) -> Color {
        let a = if self.next() % 2 == 0 { 255 } else { 64 + (self.next() % 160) as u8 };
        Color::from_rgba8((self.next() % 256) as u8, (self.next() % 256) as u8, (self.next() % 256) as u8, a)
    }
}

fn scene(ctx: &mut RenderContext) {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut depth = 0;
    for i in 0..1500 {
        if depth < 3 && rng.next() % 50 == 0 {
            depth += 1;
            let opacity = if rng.next() % 2 == 0 { 1.0 } else { 0.4 + rng.unit() as f32 * 0.5 };
            if rng.next() % 3 == 0 {
                ctx.push_layer(None, None, Some(opacity), None, None);
            } else {
                let size = 200.0 + rng.unit() * 400.0;
                let clip = rng.glyph(size);
                ctx.set_transform(Affine::translate((rng.unit() * f64::from(WIDTH) - size / 3.0, rng.unit() * f64::from(HEIGHT) - size / 3.0)));
                ctx.set_fill_rule(Fill::NonZero);
                ctx.push_layer(Some(&clip), None, Some(opacity), None, None);
            }
        }
        if depth > 0 && rng.next() % 40 == 0 {
            ctx.pop_layer();
            depth -= 1;
        }
        let size = if rng.next() % 30 == 0 { 80.0 + rng.unit() * 300.0 } else { 6.0 + rng.unit() * 20.0 };
        let path = rng.glyph(size);
        let t = Affine::translate((rng.unit() * f64::from(WIDTH), rng.unit() * f64::from(HEIGHT))) * Affine::rotate(rng.unit() * 6.3) * Affine::scale(0.5 + rng.unit() * 1.5);
        ctx.set_transform(t);
        if i % 7 == 0 {
            let g = Gradient::new_linear(Point::new(0.0, 0.0), Point::new(size, size)).with_stops([rng.colour(), rng.colour()].as_slice());
            ctx.set_paint(g);
        } else {
            ctx.set_paint(rng.colour());
        }
        if i % 5 == 0 {
            let stroke = Stroke::new(0.5 + rng.unit() * 4.0).with_join(Join::Round).with_caps(Cap::Round);
            ctx.set_stroke(stroke);
            ctx.stroke_path(&path);
        } else {
            ctx.set_fill_rule(if rng.next() % 2 == 0 { Fill::NonZero } else { Fill::EvenOdd });
            ctx.fill_path(&path);
        }
    }
    while depth > 0 {
        ctx.pop_layer();
        depth -= 1;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reference = RenderContext::new(WIDTH, HEIGHT);
    scene(&mut reference);
    let mut want = Pixmap::new(WIDTH, HEIGHT);
    let t = std::time::Instant::now();
    reference.render_with(&mut want, &mut Resources::new(), RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut rec = RenderContext::new_recording(WIDTH, HEIGHT);
    scene(&mut rec);
    let (recording, encoded) = rec.recording().ok_or("not a recording")?;
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let paths = PathRaster::new(&mut cx, &codegen)?;
    let fine = Fine::new(&mut cx, &codegen)?;
    let mut times = Vec::new();
    let mut got = Vec::new();
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let binned = paths.bin(&mut cx, recording, encoded)?;
        let tiles = Tiles::recorded(recording, encoded, &[])?;
        let mut buf = cx.buffer(&vec![0; tiles.bytes()])?;
        let addr = buf.addr;
        let at = tiles.write(buf.bytes()?, addr, &[])?.with_gpu(binned.rows, binned.cmds, binned.depth, binned.alphas);
        let mut r = cx.record();
        fine.record(&mut r, &at)?;
        let s = cx.submit(r.finish())?;
        cx.wait(s)?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
        got = buf.bytes()?[(at.out - addr) as usize..][..tiles.pixels()].to_vec();
    }
    let wantb = want.data_as_u8_slice();
    let bad: Vec<usize> = (0..got.len() / 4).filter(|&i| got[4 * i..][..4] != wantb[4 * i..][..4]).collect();
    println!("{} items recorded; GPU {:.2} / {:.2} / {:.2} ms (cold, then warm), vello_cpu {cpu_ms:.2} ms", recording.items.len(), times[0], times[1], times[2]);
    println!("{} pixels, {} differ from vello_cpu's", got.len() / 4, bad.len());
    if let Some(&i) = bad.first() {
        println!("  first at ({}, {}): {:?} want {:?}", i % usize::from(WIDTH), i / usize::from(WIDTH), &got[4 * i..][..4], &wantb[4 * i..][..4]);
    }
    let pass = bad.is_empty();
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
