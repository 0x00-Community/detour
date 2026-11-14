use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 128;
const H: u32 = 64;

fn bounded(min: f32, max: f32) -> String {
    format!(
        r#"detour.pipeline @bounded {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  depth = {{compare = "always", write = false, depth_bounds_min = {min:?}, depth_bounds_max = {max:?}}},
  vertex_buffers = [{{stride = 32, attributes = [{{location = 0, format = "float32x4", offset = 0}}, {{location = 1, format = "float32x4", offset = 16}}]}}]}}
func.func @vs(%p: vector<4xf32>, %c: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {{detour.stage = "vertex"}} {{
  return %p, %c : vector<4xf32>, vector<4xf32>
}}
func.func @fs(%c: vector<4xf32>) -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  %w = arith.constant dense<[1.0, 1.0, 1.0, 1.0]> : vector<4xf32>
  return %w : vector<4xf32>
}}
"#
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let flat = cx.pipeline(&codegen, &include_str!("depth/flat.mlir").replace("WRITE", "true"), "flat")?;
    let clear = cx.clear_depth_pipeline(&codegen, [0.0, 0.0, 0.0, 1.0])?;
    let corners = cx.clear_depth_vertices(1.0)?;
    let rect = |x0: f32, x1: f32, z: f32| -> Vec<f32> { [(x0, -1.0), (x1, -1.0), (x0, 1.0), (x0, 1.0), (x1, -1.0), (x1, 1.0)].iter().flat_map(|&(x, y)| [x, y, z, 1.0, 0.0, 0.0, 0.0, 1.0]).collect() };
    let (left, right, all) = (cx.floats(&rect(-1.0, 0.0, 0.25))?, cx.floats(&rect(0.0, 1.0, 0.75))?, cx.floats(&rect(-1.0, 1.0, 0.5))?);
    let mut ok = true;
    for ([min, max], [l, r]) in [([0.0, 0.5], [true, false]), ([0.5, 1.0], [false, true]), ([0.3, 0.7], [false, false]), ([0.25, 0.25], [true, false])] {
        let pipeline = cx.pipeline(&codegen, &bounded(min, max), "bounded")?;
        let depth = cx.depth_buffer(W, H)?;
        let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut rec = cx.record();
        rec.target(&target);
        rec.depth(&depth);
        rec.draw(&clear, &[&corners], &[], 3, 1)?;
        rec.draw(&flat, &[&left], &[], 6, 1)?;
        rec.draw(&flat, &[&right], &[], 6, 1)?;
        rec.draw(&pipeline, &[&all], &[], 6, 1)?;
        let n = cx.submit(rec.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let mut bad = 0;
        for y in 0..H {
            for x in 0..W {
                let white = px[((y * W + x) * 4) as usize..][..4] == [255; 4];
                bad += usize::from(white != if x < W / 2 { l } else { r });
            }
        }
        println!("bounds [{min}, {max}]: left {}, right {}; {bad} of {} pixels wrong", if l { "drawn" } else { "kept" }, if r { "drawn" } else { "kept" }, W * H);
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
