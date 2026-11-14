//! Each color target its own state (`GPUColorTargetState`): one draw over the whole of two targets cleared to
//! (0, 64, 255, 255). Target 0 is opaque with write mask red | alpha: only those channels change, to (255, 64, 255, 255).
//! Target 1 blends premultiplied; its source (100, 0, 0, 0) has alpha 0, so the blend adds it: (100, 64, 255, 255).
//! Checked pixel by pixel in both.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 160;
const H: u32 = 96;
const WANT: [[u8; 4]; 2] = [[255, 64, 255, 255], [100, 64, 255, 255]];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let two = cx.pipeline(&codegen, include_str!("masks/two.mlir"), "two")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 64.0 / 255.0, 1.0, 1.0])?;
    let all = cx.clear_vertices()?;
    let mut targets = [cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?, cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?];

    let mut r = cx.record();
    for t in &targets {
        r.target(t);
        r.draw(&clear, &[&all], &[], 3, 1)?;
    }
    r.targets(&[&targets[0], &targets[1]]);
    r.draw(&two, &[&all], &[], 3, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let mut bad = Vec::new();
    for (k, t) in targets.iter_mut().enumerate() {
        let px = t.bytes()?;
        bad.extend(px.as_chunks::<4>().0.iter().enumerate().filter(|(_, p)| **p != WANT[k]).map(|(i, p)| (k, i as u32 % W, i as u32 / W, *p)));
    }
    println!("{} pixels in each of 2 targets, {} wrong", W * H, bad.len());
    assert!(bad.is_empty(), "wrong pixels (target, x, y, got) {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
