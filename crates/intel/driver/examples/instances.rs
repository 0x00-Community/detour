use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 256;
const H: u32 = 256;
const CLEAR: [u8; 4] = [0, 0, 0, 255];

fn rect(i: u32) -> (u32, u32, u32, u32) {
    (i % 4 * 64 + 8, i / 4 * 64 + 16, 40, 24)
}

fn color(i: u32) -> [u8; 4] {
    [(i * 16) as u8, (255 - i * 16) as u8, (i * 37 % 256) as u8, 255]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let rects = cx.pipeline(&codegen, include_str!("instances/rects.mlir"), "rects")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 0.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let quad = cx.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
    let mut inst = Vec::new();
    for i in 0..16 {
        let (x, y, w, h) = rect(i);
        let ndc = [x as f32 / W as f32 * 2.0 - 1.0, y as f32 / H as f32 * 2.0 - 1.0, w as f32 / W as f32 * 2.0, h as f32 / H as f32 * 2.0];
        let c = color(i).map(|v| f32::from(v) / 255.0);
        inst.extend(ndc.iter().chain(&c).flat_map(|v| v.to_le_bytes()));
    }
    let inst = cx.buffer(&inst)?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.draw(&rects, &[&quad, &inst], &[], 4, 16)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let mut bad = Vec::new();
    let mut counts = [0u32; 17];
    for y in 0..H {
        for x in 0..W {
            let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into().unwrap();
            let inside = (0..16).find(|i| {
                let (rx, ry, rw, rh) = rect(*i);
                x >= rx && x < rx + rw && y >= ry && y < ry + rh
            });
            let want = inside.map_or(CLEAR, color);
            counts[inside.map_or(16, |i| i as usize)] += u32::from(got == want);
            if got != want {
                bad.push((x, y, got, want));
            }
        }
    }
    println!("per rectangle right pixels {:?} (of {} each), background {}", &counts[..16], 40 * 24, counts[16]);
    println!("{} wrong", bad.len());
    assert!(bad.is_empty(), "wrong: {:?}", &bad[..bad.len().min(8)]);
    println!("ok");
    Ok(())
}
