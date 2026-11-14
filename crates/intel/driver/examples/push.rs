//! Push constants (WGSL `var<immediate>`, `setImmediates`; Vulkan push constants: `Recorder::push_constants`,
//! `detour.push_constant {offset}`) on Xe3: a quarter-wide quad drawn four times in 1920x1080, each draw pushing its own
//! translation (vertex shader) and color. Quarters 0 and 1: the color pushed (fragment shader, byte 16); 2 and 3: a
//! pipeline reading a color from a bound buffer as well, scaled by a pushed scalar (byte 8). Checked pixel by pixel; a
//! draw whose pipeline reads more than was pushed is refused.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;

#[rustfmt::skip]
const QUAD: [f32; 12] = [-1.0, -1.0, -0.5, -1.0, -1.0, 1.0,   -1.0, 1.0, -0.5, -1.0, -0.5, 1.0];

fn bytes(words: &[f32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let pushed = cx.pipeline(&codegen, include_str!("push/pushed.mlir"), "pushed")?;
    let mixed = cx.pipeline(&codegen, include_str!("push/mixed.mlir"), "mixed")?;
    println!("push bytes read: pushed {}, mixed {}", pushed.layout.push, mixed.layout.push);
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let vb = cx.floats(&QUAD)?;
    let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.75, 1.0]];
    let white = cx.floats(&[1.0, 1.0, 1.0, 1.0])?;
    let scales = [1.0f32, 0.5];
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    for k in 0..2 {
        let mut words = vec![0.5 * k as f32, 0.0, 0.0, 0.0];
        words.extend(colors[k]);
        r.push_constants(&bytes(&words))?;
        r.draw(&pushed, &[&vb], &[], 6, 1)?;
    }
    r.bind(&[&white]);
    for (k, s) in [(2usize, scales[0]), (3, scales[1])] {
        r.push_constants(&bytes(&[0.5 * k as f32, 0.0, s]))?;
        r.draw(&mixed, &[&vb], &[], 6, 1)?;
    }
    r.push_constants(&bytes(&[0.0, 0.0]))?;
    let refused = r.draw(&pushed, &[&vb], &[], 6, 1).err().map(|e| e.to_string());
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    println!("8 bytes pushed for a pipeline reading 32: {}", refused.as_deref().unwrap_or("drawn"));

    let byte = |c: f32| (c * 255.0).round() as u8;
    let want: Vec<[u8; 4]> = vec![colors[0].map(byte), colors[1].map(byte), [255, 255, 255, 255], [128, 128, 128, 128]];
    let px = target.bytes()?;
    let mut bad = 0;
    for y in 0..H {
        for x in 0..W {
            let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            bad += usize::from(p != want[(x / (W / 4)) as usize]);
        }
    }
    println!("quarters {:?}; {bad} wrong of {}", (0..4).map(|k| &px[(k * W / 4 * 4) as usize..][..4]).collect::<Vec<_>>(), W * H);
    assert_eq!(bad, 0);
    assert!(refused.is_some());
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
