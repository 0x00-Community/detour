use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 128;
const CLEAR: [f64; 4] = [0.2, 0.4, 0.6, 1.0];

type Rect = ((u32, u32, u32, u32), [f64; 3], f64);

const RECTS: [Rect; 3] = [
    ((10, 10, 70, 60), [1.0, 0.0, 0.0], 0.5),
    ((40, 30, 70, 70), [0.0, 1.0, 0.0], 0.25),
    ((25, 50, 50, 60), [0.0, 0.0, 1.0], 0.75),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let rects = cx.pipeline(&codegen, include_str!("blend/rects.mlir"), "rects")?;
    let clear = cx.clear_pipeline(&codegen, CLEAR.map(|c| c as f32))?;
    let corners = cx.clear_vertices()?;
    let quad = cx.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
    let mut inst = Vec::new();
    for ((x, y, w, h), c, a) in RECTS {
        let f = |v: u32| v as f32 / W as f32 * 2.0;
        let rect = [f(x) - 1.0, f(y) - 1.0, f(w), f(h)];
        let color = [c[0] * a, c[1] * a, c[2] * a, a].map(|v| v as f32);
        inst.extend(rect.iter().chain(&color).flat_map(|v| v.to_le_bytes()));
    }
    let inst = cx.buffer(&inst)?;
    let mut target = cx.target(W, W, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.draw(&rects, &[&quad, &inst], &[], 4, RECTS.len() as u32)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let (mut bad, mut worst, mut covered) = (Vec::new(), 0.0f64, 0);
    for y in 0..W {
        for x in 0..W {
            let mut want = CLEAR;
            let mut n = 0;
            for ((rx, ry, rw, rh), c, a) in RECTS {
                if x >= rx && x < rx + rw && y >= ry && y < ry + rh {
                    n += 1;
                    let src = [c[0] * a, c[1] * a, c[2] * a, a];
                    want = std::array::from_fn(|i| src[i] + (want[i] * 255.0).round() / 255.0 * (1.0 - a));
                }
            }
            covered += usize::from(n > 1);
            let got = &px[((y * W + x) * 4) as usize..][..4];
            for i in 0..4 {
                let d = (f64::from(got[i]) - want[i] * 255.0).abs();
                worst = worst.max(d);
                if d > 1.0 {
                    bad.push((x, y, i, got[i], want[i] * 255.0));
                }
            }
        }
    }
    println!("{} pixels, {covered} under two rectangles or more, worst difference {worst:.2} (unorm8 steps), {} wrong", W * W, bad.len());
    assert!(bad.is_empty(), "wrong: {:?}", &bad[..bad.len().min(8)]);
    println!("ok");
    Ok(())
}
