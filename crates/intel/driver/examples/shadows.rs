use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 64;
const H: u32 = 32;

fn pipeline(sampler: &str, fs: &str) -> String {
    format!(
        r#"detour.pipeline @p {{vertex = @vs, fragment = @fs, topology = "triangle-strip", targets = ["rgba32float"], sampler = {sampler},
  vertex_buffers = [{{stride = 16, attributes = [{{location = 0, format = "float32x2", offset = 0}}, {{location = 1, format = "float32x2", offset = 8}}]}}]}}
func.func @vs(%p: vector<2xf32>, %uv: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uv : vector<4xf32>, vector<2xf32>
}}
func.func @fs(%uv: vector<2xf32>) -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
{fs}
  return %out : vector<4xf32>
}}
"#
    )
}

const PLAIN: &str = r#"  %out = "detour.sample"(%uv) {binding = 0 : i64} : (vector<2xf32>) -> vector<4xf32>"#;

fn compare(r: &str) -> String {
    format!(
        r#"  %ref = arith.constant {r} : f32
  %one = arith.constant 1.0 : f32
  %c = "detour.sample_compare"(%uv, %ref) {{binding = 0 : i64}} : (vector<2xf32>, f32) -> f32
  %out = "vector.from_elements"(%c, %c, %c, %one) : (f32, f32, f32, f32) -> vector<4xf32>"#
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let flat = cx.pipeline(&codegen, &include_str!("depth/flat.mlir").replace("WRITE", "true"), "flat")?;
    let clear = cx.clear_depth_pipeline(&codegen, [0.0, 0.0, 0.0, 1.0])?;
    let corners = cx.clear_depth_vertices(1.0)?;
    let half = |x0: f32, x1: f32, z: f32| -> Vec<f32> { [(x0, -1.0), (x1, -1.0), (x0, 1.0), (x0, 1.0), (x1, -1.0), (x1, 1.0)].iter().flat_map(|&(x, y)| [x, y, z, 1.0, 1.0, 1.0, 1.0, 1.0]).collect() };
    let (left, right) = (cx.floats(&half(-1.0, 0.0, 0.25))?, cx.floats(&half(0.0, 1.0, 0.75))?);
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])?;
    let mut cases = vec![("the depths".to_owned(), pipeline("{}", PLAIN), [0.25, 0.75])];
    for (f, r, want) in [
        ("less", "0.5", [0.0, 1.0]),
        ("greater", "0.5", [1.0, 0.0]),
        ("equal", "0.25", [1.0, 0.0]),
        ("not-equal", "0.25", [0.0, 1.0]),
        ("less-equal", "0.25", [1.0, 1.0]),
        ("greater-equal", "0.25", [1.0, 0.0]),
        ("always", "0.5", [1.0, 1.0]),
        ("never", "0.5", [0.0, 0.0]),
    ] {
        cases.push((format!("compare {f}, ref {r}"), pipeline(&format!(r#"{{compare = "{f}"}}"#), &compare(r)), want));
    }
    let mut ok = true;
    for (name, src, [l, r_]) in cases {
        let sampling = cx.pipeline(&codegen, &src, "p")?;
        let scratch = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let depth = cx.depth_buffer(W, H)?;
        let mut out = cx.target(W, H, TextureFormat::Rgba32Float, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&scratch);
        r.depth(&depth);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        r.draw(&flat, &[&left], &[], 6, 1)?;
        r.draw(&flat, &[&right], &[], 6, 1)?;
        r.depth_barrier()?;
        r.target(&out);
        r.draw(&sampling, &[&quad], &[&depth], 4, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = out.bytes()?;
        let mut bad = 0;
        for y in 0..H {
            for x in 0..W {
                let red = f32::from_le_bytes(px[((y * W + x) * 16) as usize..][..4].try_into()?);
                bad += usize::from(red != if x < W / 2 { l } else { r_ });
            }
        }
        let at = |x: u32| f32::from_le_bytes(px[(x * 16) as usize..][..4].try_into().expect("4"));
        println!("{name}: left {}, right {}; {bad} of {} pixels wrong", at(0), at(W - 1), W * H);
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
