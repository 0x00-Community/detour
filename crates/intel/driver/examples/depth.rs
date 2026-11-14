//! The depth test, 1920x1080, checked pixel by pixel. Clear: blue, depth 1. A red quad (left two thirds) at z 0.25,
//! then a green one (right two thirds) at z 0.75 drawn after it: where they overlap the nearer red stays. Then a
//! yellow full-target quad at z 0.5 that does not write depth (passes only over the green), then a cyan one at z 0.6:
//! it passes where yellow went only if yellow left the depth at 0.75. So: left two thirds red, the rest cyan.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;
const RED: [u8; 4] = [255, 0, 0, 255];
const CYAN: [u8; 4] = [0, 255, 255, 255];

/// A rectangle `x0..x1` (clip space, full height) at depth `z` in one color: two triangles, each vertex x y z w r g b a.
fn quad(x0: f32, x1: f32, z: f32, rgba: [f32; 4]) -> Vec<f32> {
    let corners = [(x0, -1.0), (x1, -1.0), (x0, 1.0), (x0, 1.0), (x1, -1.0), (x1, 1.0)];
    corners.iter().flat_map(|&(x, y)| [x, y, z, 1.0].into_iter().chain(rgba)).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let src = include_str!("depth/flat.mlir");
    let writes = cx.pipeline(&codegen, &src.replace("WRITE", "true"), "flat")?;
    let keeps = cx.pipeline(&codegen, &src.replace("WRITE", "false"), "flat")?;
    let clear = cx.clear_depth_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_depth_vertices(1.0)?;
    let red = cx.floats(&quad(-1.0, 1.0 / 3.0, 0.25, [1.0, 0.0, 0.0, 1.0]))?;
    let green = cx.floats(&quad(-1.0 / 3.0, 1.0, 0.75, [0.0, 1.0, 0.0, 1.0]))?;
    let yellow = cx.floats(&quad(-1.0, 1.0, 0.5, [1.0, 1.0, 0.0, 1.0]))?;
    let cyan = cx.floats(&quad(-1.0, 1.0, 0.6, [0.0, 1.0, 1.0, 1.0]))?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let depth = cx.depth_buffer(W, H)?;

    let mut r = cx.record();
    r.target(&target);
    r.depth(&depth);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.draw(&writes, &[&red], &[], 6, 1)?;
    r.draw(&writes, &[&green], &[], 6, 1)?;
    r.draw(&keeps, &[&yellow], &[], 6, 1)?;
    r.draw(&writes, &[&cyan], &[], 6, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let mut bad = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            // clip x = 1/3 is the pixel edge at 2 W / 3
            let want = if x < W * 2 / 3 { RED } else { CYAN };
            if p != want {
                bad.push((x, y, p));
            }
        }
    }
    println!("{} wrong of {}", bad.len(), W * H);
    assert!(bad.is_empty(), "wrong pixels {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
