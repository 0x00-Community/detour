//! Blur effects (`Frost`) on the GPU against a Rust model of the same definition (`detour::xe3::Frost`'s module
//! docs), byte for byte. A 512 x 320 premultiplied canvas (gradients, stripes, noise, translucent patches) under, in
//! order: frosted glass of 1 to 5 Kawase levels (fractional and off-canvas rects, radii past half the height, a region
//! whose deep levels are 1 x 1, saturation past the unorm range, two overlapping, one over the wallpaper layer under a
//! window), CSS filters of a straight and a premultiplied layer (with and without blur, colour matrices with offsets
//! past the range, clipped, partly off the canvas) and drop shadows (fractional offsets, clipped, off the canvas).
//! Then 1200 x 700 panels (σ 24) on 1920 x 1080: checked, and the GPU's time a panel (11 a frame against 1).
//! `frost K` runs effect K of the first scene alone; `frost --mlir` prints the shaders.

use detour::base::{Backdrop, Effect, Shadow};
use detour::xe3::{Buffer, Canvas, Codegen, Context, Frost, Layer};
use detour::xe::Xe;
use detour::base::{Gpu, Native};

#[path = "common/frost_model.rs"]
mod model;
use model::*;

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// Gradients, stripes and noise, opaque but for translucent patches (premultiplied: no channel above alpha); with
/// `seed`'s noise.
fn scene(w: usize, h: usize, seed: u64) -> Image {
    let mut rng = Rng(seed);
    let px = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let noise = (rng.next() >> 58) as usize;
            let stripe = if (x / 24 + y / 40) % 2 == 0 { 60 } else { 0 };
            let a = if (x / 64) % 3 == 1 && (y / 48) % 2 == 1 { 96 + (x % 64) as u32 * 2 } else { 255 };
            let c = |v: usize| ((v.min(255) as u32) * a / 255) as u8;
            [c(x * 255 / w + noise), c(y * 255 / h + stripe), c((x + y) % 256), a as u8]
        })
        .collect();
    Image { w, h, px }
}

/// A layer: a disc, a ring and a translucent band on transparency, straight alpha (or premultiplied).
fn layer(w: usize, h: usize, premultiplied: bool) -> Image {
    let px = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 - w as f32 * 0.4, (i / w) as f32 - h as f32 * 0.5);
            let d = (x * x + y * y).sqrt();
            let c = if d < h as f32 * 0.3 {
                [230, 120, 40, 255]
            } else if (d - h as f32 * 0.42).abs() < 3.0 {
                [40, 90, 230, 160]
            } else if (i % w) > w * 3 / 4 {
                [20, 200, 120, 90 + (i / w % 100) as u8]
            } else {
                [255, 255, 255, 0]
            };
            if premultiplied { premultiply(c.map(f32::from)).map(pack) } else { c }
        })
        .collect();
    Image { w, h, px }
}

/// One effect of a scene.
#[derive(Clone, Copy)]
enum Op {
    /// over the wallpaper under a window when true
    Backdrop(Backdrop, bool),
    Filter(usize, Effect, [u32; 4], [u32; 4]),
    Shadow(usize, Shadow, [u32; 4], [u32; 4]),
}

impl Image {
    /// `ops` over this canvas on the GPU `times` times, each checked against the model; the times, sorted.
    fn frost<'d>(&self, cx: &mut Context<'d, Xe, Native>, frost: &mut Frost<'d, Xe>, layers: &[(Image, bool)], under: &Image, ops: &[Op], times: usize) -> Result<Vec<std::time::Duration>, Box<dyn std::error::Error>> {
        let mut want = self.clone();
        for op in ops {
            match *op {
                Op::Backdrop(b, u) => want.backdrop(u.then_some(under), &b, &[[0, 0, self.w as u32, self.h as u32]]),
                Op::Filter(l, e, rect, clip) => want.filtered(&layers[l].0, layers[l].1, &e, rect, clip),
                Op::Shadow(l, s, rect, clip) => want.shadow(&layers[l].0, &s, rect, clip),
            }
        }
        let mut canvas_buf = cx.buffer(&self.px.concat())?;
        let layer_bufs: Vec<Buffer<'d, Xe>> = layers.iter().map(|(l, _)| cx.buffer(&l.px.concat())).collect::<Result<_, _>>()?;
        let under_buf = cx.buffer(&under.px.concat())?;
        let canvas = Canvas { addr: canvas_buf.addr, size: [self.w as u32, self.h as u32], stride: self.w as u32 };
        let under = Layer { addr: under_buf.addr, stride: under.w as u32, premultiplied: true };
        let layer: Vec<Layer> = layers.iter().zip(&layer_bufs).map(|((l, p), b)| Layer { addr: b.addr, stride: l.w as u32, premultiplied: *p }).collect();
        let mut took = Vec::new();
        for _ in 0..times {
            canvas_buf.bytes()?.copy_from_slice(&self.px.concat());
            let slot = cx.slot();
            let mut r = cx.record();
            let mut f = frost.frame(slot);
            for op in ops {
                match op {
                    Op::Backdrop(b, u) => f.backdrop(&mut r, &canvas, u.then_some(&under), b, &[canvas.size].map(|[w, h]| [0, 0, w, h]))?,
                    Op::Filter(l, e, rect, clip) => f.filtered(&mut r, &canvas, &layer[*l], e, *rect, *clip)?,
                    Op::Shadow(l, s, rect, clip) => f.shadow(&mut r, &canvas, &layer[*l], s, *rect, *clip)?,
                }
            }
            let t = std::time::Instant::now();
            let n = cx.submit(r.finish())?;
            cx.wait(n)?;
            took.push(t.elapsed());
            let got = canvas_buf.bytes()?;
            let (mut wrong, mut changed) = (0, 0);
            let mut first = Vec::new();
            for (i, (g, w)) in got.as_chunks::<4>().0.iter().zip(&want.px).enumerate() {
                changed += usize::from(w != &self.px[i]);
                if g != w {
                    wrong += 1;
                    if first.len() < 6 {
                        first.push(((i % self.w, i / self.w), *g, *w));
                    }
                }
            }
            if wrong > 0 || took.len() == 1 {
                println!("  {} x {}: {} pixels, {changed} changed by the effects, {wrong} wrong {first:?}", self.w, self.h, got.len() / 4);
            }
            assert_eq!(wrong, 0, "pixels off the model");
        }
        took.sort();
        Ok(took)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `--mlir`: the shaders' source (for `--example lower`), no GPU
    if std::env::args().any(|a| a == "--mlir") {
        let codegen = Codegen::default();
        Frost::<Xe>::shaders(&codegen.0).iter().for_each(|(_, body)| println!("{}", codegen.0.show(body)));
        return Ok(());
    }
    let grey = [0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let warm = [1.1, 0.1, 0.0, 0.0, 0.05, 0.0, 0.9, 0.1, 0.0, -0.02, 0.0, 0.0, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 0.8, 0.0];
    let ops = [
        Op::Backdrop(Backdrop { rect: [40.0, 30.0, 200.0, 120.0], radius: 16.0, effect: model::frosted(12.0, 1.8, 1.1) }, false),
        Op::Backdrop(Backdrop { rect: [300.5, 20.25, 180.75, 90.5], radius: 60.0, effect: model::frosted(3.0, 0.5, 0.9) }, false),
        Op::Backdrop(Backdrop { rect: [-20.0, 200.0, 150.0, 140.0], radius: 0.0, effect: model::frosted(64.0, 1.0, 1.0) }, false),
        Op::Backdrop(Backdrop { rect: [400.0, 250.0, 9.0, 7.0], radius: 3.0, effect: model::frosted(30.0, 2.0, 1.3) }, false),
        Op::Backdrop(Backdrop { rect: [180.0, 100.0, 101.0, 77.0], radius: 12.5, effect: model::frosted(8.0, 1.4, 1.05) }, false),
        Op::Backdrop(Backdrop { rect: [330.0, 140.0, 150.0, 100.0], radius: 10.0, effect: model::frosted(10.0, 1.6, 1.0) }, true),
        Op::Filter(0, Effect { blur: 0.0, matrix: Some(grey) }, [20, 170, 140, 90], [0, 0, 512, 320]),
        Op::Filter(0, Effect { blur: 6.0, matrix: Some(warm) }, [190, 200, 140, 90], [200, 210, 100, 200]),
        Op::Filter(1, Effect { blur: 4.0, matrix: None }, [420, 260, 140, 90], [0, 0, 512, 320]),
        Op::Filter(1, Effect { blur: 0.0, matrix: Some(warm) }, [60, 20, 140, 90], [70, 25, 120, 70]),
        Op::Shadow(0, Shadow { dx: 6.5, dy: 4.25, blur: 8.0, color: [0.0, 0.0, 0.1, 0.6] }, [250, 20, 140, 90], [0, 0, 512, 320]),
        Op::Shadow(1, Shadow { dx: -3.0, dy: 10.0, blur: 0.0, color: [0.9, 0.2, 0.1, 1.0] }, [10, 250, 140, 90], [0, 260, 512, 50]),
        Op::Shadow(0, Shadow { dx: 30.0, dy: -12.75, blur: 20.0, color: [0.2, 0.2, 0.2, 0.8] }, [430, 200, 140, 90], [0, 0, 512, 320]),
    ];
    // `frost K`: effect K alone
    let only: Option<usize> = std::env::args().nth(1).and_then(|a| a.parse().ok());
    let ops: Vec<Op> = ops.into_iter().enumerate().filter(|(k, _)| only.is_none_or(|o| o == *k)).map(|(_, b)| b).collect();
    let layers = [(layer(140, 90, false), false), (layer(140, 90, true), true)];

    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let t = std::time::Instant::now();
    let mut frost = Frost::new(&mut cx, &codegen, [1920, 1080], 16)?;
    println!("{} shaders compiled in {:?} (once)", Frost::<Xe>::shaders(&codegen.0).len(), t.elapsed());

    let took = scene(512, 320, 1).frost(&mut cx, &mut frost, &layers, &scene(512, 320, 2), &ops, 1)?;
    println!("{} effects, submit to done {:?}", ops.len(), took[0]);

    let panel = Op::Backdrop(Backdrop { rect: [360.0, 190.0, 1200.0, 700.0], radius: 24.0, effect: model::frosted(24.0, 1.8, 1.05) }, false);
    let (canvas, under) = (scene(1920, 1080, 1), scene(1920, 1080, 2));
    canvas.frost(&mut cx, &mut frost, &layers, &under, &[panel; 3], 1)?;
    // the GPU's time a panel: frames of 1 and of 11 back to back (no checking between them: the GPU stays clocked
    // up), the best of 9 each
    let Op::Backdrop(b, _) = panel else { unreachable!() };
    let c = cx.buffer(&canvas.px.concat())?;
    let target = Canvas { addr: c.addr, size: [1920, 1080], stride: 1920 };
    let mut best = [std::time::Duration::MAX; 2];
    for _ in 0..9 {
        for (k, n) in [1, 11].into_iter().enumerate() {
            let slot = cx.slot();
            let mut r = cx.record();
            let mut f = frost.frame(slot);
            for _ in 0..n {
                f.backdrop(&mut r, &target, None, &b, &[[0, 0, 1920, 1080]])?;
            }
            let t = std::time::Instant::now();
            let s = cx.submit(r.finish())?;
            cx.wait(s)?;
            best[k] = best[k].min(t.elapsed());
        }
    }
    println!("1200 x 700 panel ({} levels): submit to done {:?} for one a frame; {:?} a panel (11 a frame)", b.levels(), best[0], (best[1] - best[0]) / 10);
    println!("ok");
    Ok(())
}
