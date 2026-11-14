//! Multiple render targets: one draw into two targets (`Recorder::targets`), its fragment shader's result 0 a constant
//! into the first, result 1 a 16x16 checkerboard sampled with nearest filtering into the second (the texture after
//! the two targets in the binding table). The quad covers the left half of each 256x128 target (cleared blue first),
//! so every pixel is known: checked one by one in both.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const ORANGE: [u8; 4] = [252, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const T: u32 = 16;
const W: u32 = 256;
const H: u32 = 128;

fn texel(x: u32, y: u32) -> [u8; 4] {
    if (x + y).is_multiple_of(2) { [(x * 16) as u8, (y * 16) as u8, 200, 255] } else { [255 - (y * 16) as u8, 40, (x * 16) as u8, 255] }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let two = cx.pipeline(&codegen, include_str!("mrt/two.mlir"), "two")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let texels: Vec<u8> = (0..T).flat_map(|y| (0..T).flat_map(move |x| texel(x, y))).collect();
    let image = cx.texture(T, T, TextureFormat::Rgba8Unorm, &texels)?;
    // x, y, u, v over the left half: u, v 0 ..= 1 across it
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])?;
    let mut a = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut b = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    for t in [&a, &b] {
        r.target(t);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
    }
    r.targets(&[&a, &b]);
    r.draw(&two, &[&quad], &[&image], 4, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let (half, mut bad) = (W / 2, Vec::new());
    let want_a = |x: u32, _: u32| if x < half { ORANGE } else { BLUE };
    let want_b = |x: u32, y: u32| if x < half { texel(x / (half / T), y / (H / T)) } else { BLUE };
    for (k, (t, want)) in [(&mut a, &want_a as &dyn Fn(u32, u32) -> [u8; 4]), (&mut b, &want_b)].into_iter().enumerate() {
        let px = t.bytes()?;
        for y in 0..H {
            for x in 0..W {
                let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
                if got != want(x, y) {
                    bad.push((k, x, y, got, want(x, y)));
                }
            }
        }
    }
    println!("{} pixels in each of 2 targets, {} wrong", W * H, bad.len());
    assert!(bad.is_empty(), "wrong pixels (target, x, y, got, want) {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
