//! Conditional rendering (Vulkan `VK_EXT_conditional_rendering`: `Recorder::begin_conditional` / `end_conditional`, an
//! MI_PREDICATE the draws test) on Xe3: a quarter-wide quad pushed into each quarter of 1920x1080 under the conditions
//! [0, 1, 7, 0] at bytes 0, 4, 8, 12: q0 (0) skipped, q1 (1) drawn, q2 (7, inverted) skipped, q3 (0, inverted) drawn.
//! Then over q1 a draw under 0 again (skipped), and after `end_conditional` one over q2 (drawn: the predicate no longer
//! tested). Pixel by pixel.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;

#[rustfmt::skip]
const QUAD: [f32; 12] = [-1.0, -1.0, -0.5, -1.0, -1.0, 1.0,   -1.0, 1.0, -0.5, -1.0, -0.5, 1.0];

fn push(k: usize, color: [f32; 4]) -> Vec<u8> {
    [0.5 * k as f32, 0.0, 0.0, 0.0].iter().chain(&color).flat_map(|w| w.to_le_bytes()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let pushed = cx.pipeline(&codegen, include_str!("push/pushed.mlir"), "pushed")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let vb = cx.floats(&QUAD)?;
    let conditions = cx.buffer(&[0u32, 1, 7, 0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    let (red, green, yellow) = ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]);
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    for (k, inverted, color) in [(0usize, false, red), (1, false, red), (2, true, red), (3, true, yellow)] {
        r.begin_conditional(&conditions, 4 * k as u64, inverted)?;
        r.push_constants(&push(k, color))?;
        r.draw(&pushed, &[&vb], &[], 6, 1)?;
    }
    r.begin_conditional(&conditions, 0, false)?;
    r.push_constants(&push(1, green))?;
    r.draw(&pushed, &[&vb], &[], 6, 1)?;
    r.end_conditional();
    r.push_constants(&push(2, green))?;
    r.draw(&pushed, &[&vb], &[], 6, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let want: [[u8; 4]; 4] = [[0, 0, 255, 255], [255, 0, 0, 255], [0, 255, 0, 255], [255, 255, 0, 255]];
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
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
