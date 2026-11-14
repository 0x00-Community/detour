//! Depth bias (`GPUDepthStencilState.depthBias*`) on Xe3, by what it lets through: a red quad at z 0.5 writes the depth
//! (compare less), then a green one at the same z with a bias. Unbiased (or biased +1) it fails everywhere (0.5 is not
//! less than 0.5); biased −1 (−1 × 2^-24, the resolution of depth32float at 0.5) it passes everywhere; biased −1 but
//! clamped to −2^-26 it fails again (0.5 − 2^-26 rounds back to 0.5): the clamp holds.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 256;
const H: u32 = 128;
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];

/// The whole target at depth `z` in one color: two triangles, each vertex x y z w r g b a.
fn quad(z: f32, rgba: [f32; 4]) -> Vec<f32> {
    let corners = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)];
    corners.iter().flat_map(|&(x, y)| [x, y, z, 1.0].into_iter().chain(rgba)).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let src = include_str!("depth/flat.mlir");
    let plain = cx.pipeline(&codegen, &src.replace("WRITE", "true"), "flat")?;
    let clear = cx.clear_depth_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_depth_vertices(1.0)?;
    let red = cx.floats(&quad(0.5, [1.0, 0.0, 0.0, 1.0]))?;
    let green = cx.floats(&quad(0.5, [0.0, 1.0, 0.0, 1.0]))?;
    let clamp = 2f32.powi(-26);
    let cases = [
        ("unbiased", "true".to_owned(), RED),
        ("biased +1", "true, depth_bias = 1".to_owned(), RED),
        ("biased -1", "true, depth_bias = -1".to_owned(), GREEN),
        ("biased -1, clamped to -2^-26", format!("true, depth_bias = -1, depth_bias_clamp = {:?}", -clamp), RED),
    ];
    let mut ok = true;
    for (name, depth, want) in cases {
        let biased = cx.pipeline(&codegen, &src.replace("WRITE", &depth), "flat")?;
        let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let buffer = cx.depth_buffer(W, H)?;
        let mut r = cx.record();
        r.target(&target);
        r.depth(&buffer);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        r.draw(&plain, &[&red], &[], 6, 1)?;
        r.draw(&biased, &[&green], &[], 6, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let bad = (0..(W * H) as usize).filter(|i| px[i * 4..][..4] != want).count();
        println!("{name}: {} of {} pixels {}", W * H - bad as u32, W * H, if want == GREEN { "green (passed)" } else { "red (failed)" });
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
