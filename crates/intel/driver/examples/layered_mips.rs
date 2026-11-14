//! Layered textures with mip levels (`Context::texture_layers`) on Xe3, pixel by pixel: a 2D array of 3 layers x 7
//! levels of solid colors, each layer drawn as squares of 64, 32, 16 and 8 pixels (levels 0, 1, 2, 3 from the
//! derivatives); and a cube of 6 faces x 7 levels sampled at each face's center with lod_min_clamp 2 (level 2).

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const T: u32 = 64;
const LEVELS: u32 = 7;

fn pipeline(sampler: &str, cube: bool) -> String {
    format!(
        r#"detour.pipeline @p {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], sampler = {sampler},
  vertex_buffers = [{{stride = 20, attributes = [{{location = 0, format = "float32x2", offset = 0}}, {{location = 1, format = "float32x3", offset = 8}}]}}]}}
func.func @vs(%p: vector<2xf32>, %c: vector<3xf32>) -> (vector<4xf32>, vector<3xf32>) attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %c : vector<4xf32>, vector<3xf32>
}}
func.func @fs(%c: vector<3xf32>) -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  %s = "detour.sample"(%c) {{binding = 0 : i64{}}} : (vector<3xf32>) -> vector<4xf32>
  return %s : vector<4xf32>
}}
"#,
        if cube { ", cube = true" } else { "" }
    )
}

fn color(layer: u32, level: u32) -> [u8; 4] {
    [(layer * 40 + 20) as u8, (level * 35 + 10) as u8, (200 - layer * 20 - level * 10) as u8, 255]
}

/// The levels of `layer`: level l solid `color(layer, l)`.
fn levels(layer: u32) -> Vec<Vec<u8>> {
    (0..LEVELS).map(|l| color(layer, l).repeat(((T >> l).max(1) * (T >> l).max(1)) as usize)).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mut ok = true;

    // a 2D array: layer k's squares in row k of a 120 x 192 target
    let (w, h) = (120u32, 192u32);
    let data: Vec<Vec<Vec<u8>>> = (0..3).map(levels).collect();
    let refs: Vec<Vec<&[u8]>> = data.iter().map(|l| l.iter().map(Vec::as_slice).collect()).collect();
    let layers: Vec<&[&[u8]]> = refs.iter().map(Vec::as_slice).collect();
    let array = cx.texture_layers(T, T, TextureFormat::Rgba8Unorm, false, &layers)?;
    let squares = [(0u32, 64u32, 0u32), (64, 32, 1), (96, 16, 2), (112, 8, 3)];
    let clip = |px: u32, py: u32| [px as f32 / w as f32 * 2.0 - 1.0, py as f32 / h as f32 * 2.0 - 1.0];
    let mut v = Vec::new();
    for layer in 0..3u32 {
        for &(x, n, _) in &squares {
            let (a, b) = (clip(x, 64 * layer), clip(x + n, 64 * layer + n));
            for (cx_, cy_, u, vv) in [(a[0], a[1], 0.0, 0.0), (b[0], a[1], 1.0, 0.0), (a[0], b[1], 0.0, 1.0), (a[0], b[1], 0.0, 1.0), (b[0], a[1], 1.0, 0.0), (b[0], b[1], 1.0, 1.0)] {
                v.extend([cx_, cy_, u, vv, layer as f32]);
            }
        }
    }
    let vb = cx.floats(&v)?;
    let p = cx.pipeline(&codegen, &pipeline(r#"{mipmap_filter = "nearest"}"#, false), "p")?;
    let mut target = cx.target(w, h, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&target);
    r.draw(&p, &[&vb], &[&array], v.len() as u32 / 5, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = target.bytes()?;
    for layer in 0..3 {
        for &(x0, size, level) in &squares {
            let bad = (64 * layer..64 * layer + size).flat_map(|y| (x0..x0 + size).map(move |x| (x, y))).filter(|&(x, y)| px[((y * w + x) * 4) as usize..][..4] != color(layer, level)).count();
            println!("array layer {layer}, a {size}-pixel square: level {level}, {bad} pixels wrong");
            ok &= bad == 0;
        }
    }

    // a cube with levels: each face's center at level 2 (lod_min_clamp)
    let data: Vec<Vec<Vec<u8>>> = (0..6).map(levels).collect();
    let refs: Vec<Vec<&[u8]>> = data.iter().map(|l| l.iter().map(Vec::as_slice).collect()).collect();
    let faces: Vec<&[&[u8]]> = refs.iter().map(Vec::as_slice).collect();
    let cube = cx.texture_layers(T, T, TextureFormat::Rgba8Unorm, true, &faces)?;
    let dirs = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    let mut v = Vec::new();
    for (f, d) in dirs.iter().enumerate() {
        let (x0, x1) = (f as f32 / 6.0 * 2.0 - 1.0, (f as f32 + 1.0) / 6.0 * 2.0 - 1.0);
        for (x, y) in [(x0, -1.0), (x1, -1.0), (x0, 1.0), (x0, 1.0), (x1, -1.0), (x1, 1.0)] {
            v.extend([x, y, d[0], d[1], d[2]]);
        }
    }
    let vb = cx.floats(&v)?;
    let p = cx.pipeline(&codegen, &pipeline(r#"{mipmap_filter = "nearest", lod_min_clamp = 2.0}"#, true), "p")?;
    let mut target = cx.target(24, 4, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&target);
    r.draw(&p, &[&vb], &[&cube], 36, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = target.bytes()?;
    for f in 0..6u32 {
        let bad = (0..4).flat_map(|y| (4 * f..4 * f + 4).map(move |x| (x, y))).filter(|&(x, y)| px[((y * 24 + x) * 4) as usize..][..4] != color(f, 2)).count();
        println!("cube face {f}, lod_min_clamp 2: level 2, {bad} pixels wrong");
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
