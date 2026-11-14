//! Dual-source blending (WebGPU `dual-source-blending`: a fragment function with `detour.dual_source` returns two colors
//! for its one target; blend factors `src1`, `one-minus-src1`, `src1-alpha`, `one-minus-src1-alpha`) on Xe3: one SIMD16
//! dual-source render target write of both colors (Xe2+ keeps them SIMD16). A whole-target triangle, color 0 (1, 0.5, 0.25, 1) and color 1
//! (0.5, 0.5, 1, 0.5), over (0.25, 0.75, 0.5, 1): src · src1 + 0 ⇒ (128, 64, 64, 128); src · 1 + dst · (1 − src1 α)
//! ⇒ (255, 223, 128, 255) (red and alpha clamped). Pixel by pixel, 1920x1080.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;

fn pipeline(color: &str, alpha: &str) -> String {
    format!(
        r#"detour.pipeline @dual {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], color_blends = [{{color = {color}, alpha = {alpha}}}],
  vertex_buffers = [{{stride = 8, attributes = [{{location = 0, format = "float32x2", offset = 0}}]}}]}}
func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}}
func.func @fs() -> (vector<4xf32>, vector<4xf32>) attributes {{detour.stage = "fragment", detour.dual_source = true}} {{
  %c0 = arith.constant dense<[1.0, 0.5, 0.25, 1.0]> : vector<4xf32>
  %c1 = arith.constant dense<[0.5, 0.5, 1.0, 0.5]> : vector<4xf32>
  return %c0, %c1 : vector<4xf32>, vector<4xf32>
}}
"#
    )
}

fn component(src: &str, dst: &str) -> String {
    format!(r#"{{src_factor = "{src}", dst_factor = "{dst}", operation = "add"}}"#)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let clear = cx.clear_pipeline(&codegen, [0.25, 0.75, 0.5, 1.0])?;
    let corners = cx.clear_vertices()?;
    let tri = cx.floats(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0])?;
    let mut ok = true;
    for (name, color, alpha, want) in [
        ("src · src1", component("src1", "zero"), component("src1-alpha", "zero"), [128u8, 64, 64, 128]),
        ("src + dst · (1 − src1 α)", component("one", "one-minus-src1-alpha"), component("one", "one-minus-src1-alpha"), [255, 223, 128, 255]),
    ] {
        let p = cx.pipeline(&codegen, &pipeline(&color, &alpha), "dual")?;
        let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        r.draw(&p, &[&tri], &[], 3, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let bad = px[..(W * H * 4) as usize].chunks(4).filter(|c| *c != want).count();
        println!("{name}: {bad} of {} pixels not {want:?}, e.g. {:?}", W * H, &px[..4]);
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
