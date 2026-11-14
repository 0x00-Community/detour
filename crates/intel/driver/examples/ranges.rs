//! Buffer bindings of part of a buffer (`GPUBufferBinding` offset and size, dynamic offsets: `Recorder::bind_ranges`)
//! on Xe3: one buffer of uniform blocks 256 bytes apart (a translation, then a color), a quarter-wide quad drawn four
//! times, each binding block k's translation and color: quarter k in color k, checked pixel by pixel in 1920x1080.
//! The last binds 8 bytes of its color: what the shader reads past them is 0 (blue and alpha).

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;

#[rustfmt::skip]
const QUAD: [f32; 12] = [-1.0, -1.0, -0.5, -1.0, -1.0, 1.0,   -1.0, 1.0, -0.5, -1.0, -0.5, 1.0];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let moved = cx.pipeline(&codegen, include_str!("uniform/moved.mlir"), "moved")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let vb = cx.floats(&QUAD)?;
    // blocks k (256 bytes each): the translation at 256 k, the color at 256 k + 16
    let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 0.5, 0.75, 1.0]];
    let mut words = vec![0f32; 4 * 64];
    for (k, c) in colors.iter().enumerate() {
        words[k * 64] = 0.5 * k as f32;
        words[k * 64 + 4..][..4].copy_from_slice(c);
    }
    let blocks = cx.floats(&words)?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    for k in 0..4u64 {
        let color = if k == 3 { 8 } else { 16 };
        r.bind_ranges(&[(&blocks, 256 * k, 8), (&blocks, 256 * k + 16, color)])?;
        r.draw(&moved, &[&vb], &[], 6, 1)?;
    }
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let byte = |c: f32| (c * 255.0).round() as u8;
    let want: Vec<[u8; 4]> = colors.iter().enumerate().map(|(k, c)| if k == 3 { [byte(c[0]), byte(c[1]), 0, 0] } else { c.map(byte) }).collect();
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
