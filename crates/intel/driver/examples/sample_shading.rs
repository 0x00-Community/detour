//! Sample shading (pipeline `sample_shading = true`: Vulkan `sampleShadingEnable`, WGSL `@interpolate(..., sample)`)
//! on Xe3: a whole-target quad into a 4x target, its varying the pixel position (x, y), the fragment shader one color
//! where x − 0.4 y > 300.1 and another elsewhere, resolved. Shaded per sample, a pixel the edge crosses is the mean of
//! its four samples' colors, each decided at the sample's position (Mesa's 4x positions); shaded per pixel (the same
//! pipeline without it, the control), every sample takes the pixel center's color. Pixel by pixel, both; pixels with
//! a sample within 0.01 of the edge are not checked (interpolation rounding). The colors are multiples of 8, so every
//! mean is a whole unorm8 value. Then `detour.sample_index`: shaded per sample, the color (s · 64, s² · 16, 248 where s
//! is 0) of sample s, resolved: every pixel (96, 56, 62), the means over s = 0, 1, 2, 3 (each once).

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;
const IN: [u8; 4] = [248, 128, 64, 255];
const OUT: [u8; 4] = [0, 0, 248, 255];
const SAMPLES_4X: [(f64, f64); 4] = [(6.0, 2.0), (14.0, 6.0), (2.0, 10.0), (10.0, 14.0)];

fn pipeline(per_sample: bool) -> String {
    format!(
        r#"detour.pipeline @split {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], samples = 4 : i32, sample_shading = {per_sample},
  vertex_buffers = [{{stride = 16, attributes = [{{location = 0, format = "float32x2", offset = 0}}, {{location = 1, format = "float32x2", offset = 8}}]}}]}}
func.func @vs(%p: vector<2xf32>, %xy: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %xy : vector<4xf32>, vector<2xf32>
}}
func.func @fs(%xy: vector<2xf32>) -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  %x = vector.extract %xy[0] : f32 from vector<2xf32>
  %y = vector.extract %xy[1] : f32 from vector<2xf32>
  %k = arith.constant 0.4 : f32
  %ky = arith.mulf %y, %k : f32
  %d = arith.subf %x, %ky : f32
  %t = arith.constant 300.1 : f32
  %inside = arith.cmpf ogt, %d, %t : f32
  %a = arith.constant dense<[0.97254902, 0.50196078, 0.25098039, 1.0]> : vector<4xf32>
  %b = arith.constant dense<[0.0, 0.0, 0.97254902, 1.0]> : vector<4xf32>
  %c = arith.select %inside, %a, %b : vector<4xf32>
  return %c : vector<4xf32>
}}
"#
    )
}

/// Whether the point (x, y) of the target is on the first color's side; `None` within 0.01 of the edge.
fn side(x: f64, y: f64) -> Option<bool> {
    let d = x - 0.4 * y - 300.1;
    (d.abs() > 0.01).then_some(d > 0.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let corners = cx.clear_vertices()?;
    let rv = cx.resolve_vertices(W, H)?;
    let (w, h) = (W as f32, H as f32);
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, w, 0.0, -1.0, 1.0, 0.0, h, -1.0, 1.0, 0.0, h, 1.0, -1.0, w, 0.0, 1.0, 1.0, w, h])?;
    let clear = cx.msaa_clear_pipeline_samples(&codegen, [0.0, 0.0, 0.0, 1.0], 4)?;
    let resolve = cx.resolve_pipeline_samples(&codegen, 4)?;
    let mut ok = true;
    for per_sample in [true, false] {
        let p = cx.pipeline(&codegen, &pipeline(per_sample), "split")?;
        let msaa = cx.msaa_target_samples(W, H, TextureFormat::Rgba8Unorm, 4)?;
        let mut out = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&msaa);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        r.draw(&p, &[&quad], &[], 6, 1)?;
        r.target(&out);
        r.resolve(&resolve, &rv, &msaa)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = out.bytes()?;
        let (mut mixed, mut skipped, mut bad) = (0usize, 0usize, Vec::new());
        for y in 0..H {
            for x in 0..W {
                let at = |sx: f64, sy: f64| side(f64::from(x) + sx / 16.0, f64::from(y) + sy / 16.0);
                let sides: Option<Vec<bool>> = if per_sample { SAMPLES_4X.iter().map(|&(sx, sy)| at(sx, sy)).collect() } else { at(8.0, 8.0).map(|s| vec![s; 4]) };
                let Some(sides) = sides else {
                    skipped += 1;
                    continue;
                };
                let k = sides.iter().filter(|s| **s).count() as u32;
                mixed += usize::from(k != 0 && k != 4);
                let want: [u8; 4] = std::array::from_fn(|c| ((u32::from(IN[c]) * k + u32::from(OUT[c]) * (4 - k)) / 4) as u8);
                let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
                if got != want {
                    bad.push((x, y, k, got, want));
                }
            }
        }
        println!("{}: {mixed} pixels mixed, {skipped} near the edge skipped, {} wrong", if per_sample { "per sample" } else { "per pixel (control)" }, bad.len());
        if !bad.is_empty() {
            println!("   first (x, y, samples in, got, want): {:?}", &bad[..bad.len().min(6)]);
        }
        ok &= bad.is_empty() && (mixed > 0) == per_sample;
    }
    let index = r#"detour.pipeline @index {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], samples = 4 : i32, sample_shading = true,
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}
func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}
func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %s = "detour.sample_index"() : () -> i32
  %f = arith.sitofp %s : i32 to f32
  %k = arith.constant 0.25098039 : f32
  %r = arith.mulf %f, %k : f32
  %ff = arith.mulf %f, %f : f32
  %k2 = arith.constant 0.062745098 : f32
  %g = arith.mulf %ff, %k2 : f32
  %zero = arith.constant 0 : i32
  %first = arith.cmpi eq, %s, %zero : i32
  %hi = arith.constant 0.97254902 : f32
  %lo = arith.constant 0.0 : f32
  %b = arith.select %first, %hi, %lo : f32
  %one = arith.constant 1.0 : f32
  %c = vector.from_elements %r, %g, %b, %one : vector<4xf32>
  return %c : vector<4xf32>
}
"#;
    let p = cx.pipeline(&codegen, index, "index")?;
    let msaa = cx.msaa_target_samples(W, H, TextureFormat::Rgba8Unorm, 4)?;
    let mut out = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&msaa);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.draw(&p, &[&quad], &[], 6, 1)?;
    r.target(&out);
    r.resolve(&resolve, &rv, &msaa)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = out.bytes()?;
    let bad = px[..(W * H * 4) as usize].chunks(4).filter(|c| *c != [96, 56, 62, 255]).count();
    println!("sample_index: {bad} of {} pixels not (96, 56, 62), e.g. {:?}", W * H, &px[..4]);
    ok &= bad == 0;
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
