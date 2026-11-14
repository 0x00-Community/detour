//! One clear pipeline for every color (`Context::clear_any_pipeline`, `Recorder::clear`: the color is the draw's blend
//! constant) on Xe3, byte for byte against a pipeline compiled for each color (`Context::clear_pipeline`, the color in
//! its shader) over colors that round every way (halves of a byte step, thirds, the ends, ties); with a depth
//! attachment, the depth cleared too (a quad at z 0.5 passes after a clear to 1, fails after one to 0.25); and the draws
//! after a clear blending with the constant they had.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{ClearDepth, Codegen, Context};
use detour::xe::Xe;

const N: u32 = 16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let any = cx.clear_any_pipeline(&codegen, 1, ClearDepth::None)?;
    let corners = cx.clear_vertices()?;
    let half = 0.5 / 255.0;
    let colors: [[f32; 4]; 14] = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.25, 0.75, 1.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.1, 0.9],
        [half, 1.5 / 255.0, 2.5 / 255.0, 254.5 / 255.0],
        [half + 1e-6, half - 1e-6, 127.5 / 255.0, 128.5 / 255.0],
        [0.123456, 0.654321, 0.999, 0.001],
        [0.2, 0.4, 0.6, 0.8],
        [0.3, 0.7, 0.05, 0.95],
        [1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 4.0 / 255.0],
        [0.0019, 0.0021, 0.9979, 0.9981],
        [0.25, 0.5, 0.75, 0.125],
        [0.6, 0.1, 0.3, 0.45],
        [1.0, 0.0, 1.0, 0.5],
    ];
    let mut ok = true;
    for rgba in colors {
        let own = cx.clear_pipeline(&codegen, rgba)?;
        let constant = cx.blend_constant(rgba)?;
        let mut a = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut b = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&a);
        r.draw(&own, &[&corners], &[], 3, 1)?;
        r.target(&b);
        r.clear(&any, &corners, &constant)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let (pa, pb) = (a.bytes()?[..(N * N * 4) as usize].to_vec(), b.bytes()?[..(N * N * 4) as usize].to_vec());
        let same = pa == pb;
        println!("{rgba:?}: {:?} {}", &pb[..4], if same { "as its own pipeline" } else { "differs from its own pipeline" });
        ok &= same && pb.chunks(4).all(|p| p == &pb[..4]);
    }

    // with a depth attachment: the depth cleared to the corners' z
    let any_depth = cx.clear_any_pipeline(&codegen, 1, ClearDepth::Depth)?;
    let flat = cx.pipeline(&codegen, &include_str!("depth/flat.mlir").replace("WRITE", "true"), "flat")?;
    let quad: Vec<f32> = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)].iter().flat_map(|&(x, y)| [x, y, 0.5, 1.0, 0.0, 1.0, 0.0, 1.0]).collect();
    let quad = cx.floats(&quad)?;
    let blue = cx.blend_constant([0.0, 0.0, 1.0, 1.0])?;
    for (z, passes) in [(1.0, true), (0.25, false)] {
        let corners = cx.clear_depth_vertices(z)?;
        let mut target = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let depth = cx.depth_buffer(N, N)?;
        let mut r = cx.record();
        r.target(&target);
        r.depth(&depth);
        r.clear(&any_depth, &corners, &blue)?;
        r.draw(&flat, &[&quad], &[], 6, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let want: [u8; 4] = if passes { [0, 255, 0, 255] } else { [0, 0, 255, 255] };
        let px = target.bytes()?;
        let good = (0..(N * N) as usize).all(|i| px[i * 4..][..4] == want);
        println!("depth cleared to {z}: a quad at 0.5 {} ({})", if passes { "passes" } else { "fails" }, if good { "every pixel" } else { "wrong" });
        ok &= good;
    }
    // after a clear, the draws blend with the constant they had (0 here): the clear pipeline drawn as a plain draw (its
    // fragment 1, blended 1 · constant) writes 0, not the clear's color
    let red = cx.blend_constant([1.0, 0.0, 0.0, 1.0])?;
    let mut target = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&target);
    r.clear(&any, &corners, &red)?;
    r.draw(&any, &[&corners], &[], 3, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let after = target.bytes()?[..4].to_vec();
    println!("a draw after a clear to red: {after:?} (its constant 0, not the clear's)");
    ok &= after == [0, 0, 0, 0];
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
