//! Texture sampling: a 16x16 checkerboard (each texel its own color) stretched over a 128x128 target with nearest
//! filtering, so each texel covers exactly 8x8 pixels. The texture coordinates are a varying; the fragment shader
//! samples binding 0. Through the runtime; every pixel is checked against its texel.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const T: u32 = 16;
const W: u32 = 128;

fn texel(x: u32, y: u32) -> [u8; 4] {
    if (x + y).is_multiple_of(2) { [(x * 16) as u8, (y * 16) as u8, 200, 255] } else { [255 - (y * 16) as u8, 40, (x * 16) as u8, 255] }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let textured = cx.pipeline(&codegen, include_str!("texture/checker.mlir"), "textured")?;
    let texels: Vec<u8> = (0..T).flat_map(|y| (0..T).flat_map(move |x| texel(x, y))).collect();
    let image = cx.texture(T, T, TextureFormat::Rgba8Unorm, &texels)?;
    // x, y, u, v: v = 0 at the first row in memory (y = -1 in clip space maps to row 0)
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])?;
    let mut target = cx.target(W, W, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&textured, &[&quad], &[&image], 4, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let mut bad = Vec::new();
    for y in 0..W {
        for x in 0..W {
            let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into().unwrap();
            let want = texel(x / (W / T), y / (W / T));
            if got != want {
                bad.push((x, y, got, want));
            }
        }
    }
    let at = |x: u32, y: u32| px[((y * W + x) * 4) as usize..][..4].to_vec();
    println!("(0,0) {:?} (8,0) {:?} (127,127) {:?}", at(0, 0), at(8, 0), at(127, 127));
    println!("{} of {} pixels wrong", bad.len(), W * W);
    assert!(bad.is_empty(), "wrong: {:?}", &bad[..bad.len().min(8)]);
    println!("ok");
    Ok(())
}
