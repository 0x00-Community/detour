use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const N: u32 = 8;

fn pipeline(samples: u32, alpha: f32) -> String {
    format!(
        r#"detour.pipeline @p {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], samples = {samples} : i32, alpha_to_coverage = true,
  vertex_buffers = [{{stride = 8, attributes = [{{location = 0, format = "float32x2", offset = 0}}]}}]}}
func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}}
func.func @fs() -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  %c = arith.constant dense<[1.0, 1.0, 1.0, {alpha:?}]> : vector<4xf32>
  return %c : vector<4xf32>
}}
"#
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let corners = cx.clear_vertices()?;
    let quad = cx.floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])?;
    let rv = cx.resolve_vertices(N, N)?;
    let mut ok = true;
    for samples in [4u32, 8] {
        let clear = cx.msaa_clear_pipeline_samples(&codegen, [0.0, 0.0, 0.0, 1.0], samples)?;
        let resolve = cx.resolve_pipeline_samples(&codegen, samples)?;
        let mut wrong = Vec::new();
        for k in 0..=256 {
            let alpha = k as f32 / 256.0;
            let p = cx.pipeline(&codegen, &pipeline(samples, alpha), "p")?;
            let msaa = cx.msaa_target_samples(N, N, TextureFormat::Rgba8Unorm, samples)?;
            let mut out = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
            let mut r = cx.record();
            r.target(&msaa);
            r.draw(&clear, &[&corners], &[], 3, 1)?;
            r.draw(&p, &[&quad], &[], 6, 1)?;
            r.target(&out);
            r.resolve(&resolve, &rv, &msaa)?;
            let n = cx.submit(r.finish())?;
            cx.wait(n)?;
            let px = out.bytes()?;
            let reds: Vec<u8> = (0..(N * N) as usize).map(|i| px[i * 4]).collect();
            let same = reds.iter().all(|&r| r == reds[0]);
            let covered = (f32::from(reds[0]) / 255.0 * samples as f32).round() as u32;
            let want = (f64::from(k) / 256.0 * f64::from(samples) - 0.5).ceil().max(0.0) as u32;
            if !same || covered != want || u32::from(reds[0]) != (255 * covered + samples / 2) / samples {
                wrong.push((k, covered, want));
            }
        }
        println!("{samples}x: alpha k / 256 for k 0 ..= 256, {} off ceil(a n - 1/2) {:?}", wrong.len(), &wrong[..wrong.len().min(6)]);
        ok &= wrong.is_empty();
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
