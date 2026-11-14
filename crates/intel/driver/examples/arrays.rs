//! 2D array textures (`texture_2d_array`, `Context::texture_array`) on Xe3, pixel by pixel: three 4 x 4 layers of
//! distinct texels, a 96 x 16 target in six strips, each sampling (nearest) one layer index: 0, 1, 2, then 1.4 and 1.6
//! (rounded: 1, 2), and -1 / 7 (clamped to the first and last layer).

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const PIPELINE: &str = r#"detour.pipeline @layered {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], filter = "nearest",
  vertex_buffers = [{stride = 20, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x3", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %uvl: vector<3xf32>) -> (vector<4xf32>, vector<3xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uvl : vector<4xf32>, vector<3xf32>
}
func.func @fs(%uvl: vector<3xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = "detour.sample"(%uvl) {binding = 0 : i64} : (vector<3xf32>) -> vector<4xf32>
  return %c : vector<4xf32>
}
"#;

const W: u32 = 96;
const H: u32 = 16;
const STRIPS: [(f32, u32); 6] = [(0.0, 0), (1.0, 1), (2.0, 2), (1.4, 1), (1.6, 2), (-1.0, 0)];

fn texel(layer: u32, x: u32, y: u32) -> [u8; 4] {
    [(layer * 80 + 20) as u8, (x * 60) as u8, (y * 60) as u8, 255]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let p = cx.pipeline(&codegen, PIPELINE, "layered")?;
    let layers: Vec<Vec<u8>> = (0..3).map(|l| (0..4).flat_map(|y| (0..4).flat_map(move |x| texel(l, x, y))).collect()).collect();
    let refs: Vec<&[u8]> = layers.iter().map(Vec::as_slice).collect();
    let tex = cx.texture_array(4, 4, TextureFormat::Rgba8Unorm, &refs)?;
    let mut ok = true;
    for (extra, cases) in [("", STRIPS.to_vec()), ("past the last", vec![(7.0, 2)])] {
        let n = cases.len() as u32;
        let strip = W / 6;
        // strip k: x over its 16 pixels (u 0 .. 1), the whole height (v 0 .. 1), its layer index
        let mut v = Vec::new();
        for (k, &(layer, _)) in cases.iter().enumerate() {
            let x0 = (k as u32 * strip) as f32 / W as f32 * 2.0 - 1.0;
            let x1 = ((k as u32 + 1) * strip) as f32 / W as f32 * 2.0 - 1.0;
            for (x, y, u, vv) in [(x0, -1.0, 0.0, 0.0), (x1, -1.0, 1.0, 0.0), (x0, 1.0, 0.0, 1.0), (x0, 1.0, 0.0, 1.0), (x1, -1.0, 1.0, 0.0), (x1, 1.0, 1.0, 1.0)] {
                v.extend([x, y, u, vv, layer]);
            }
        }
        let vb = cx.floats(&v)?;
        let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&p, &[&vb], &[&tex], 6 * n, 1)?;
        let done = cx.submit(r.finish())?;
        cx.wait(done)?;
        let px = target.bytes()?;
        for (k, &(layer, want_layer)) in cases.iter().enumerate() {
            let mut bad = 0;
            for y in 0..H {
                for x in k as u32 * strip..(k as u32 + 1) * strip {
                    let want = texel(want_layer, (x - k as u32 * strip) / 4, y / 4);
                    bad += usize::from(px[((y * W + x) * 4) as usize..][..4] != want);
                }
            }
            println!("layer index {layer}{}: layer {want_layer}, {bad} pixels wrong", if extra.is_empty() { String::new() } else { format!(" ({extra})") });
            ok &= bad == 0;
        }
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
