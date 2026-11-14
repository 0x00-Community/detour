//! The sample mask a fragment shader writes (WGSL `@builtin(sample_mask)` out: the last result of a fragment function
//! with `detour.sample_mask`; the render target write's oMask) on Xe3: a whole-target quad into a 4x target cleared
//! blue, its mask (pixel x) & 15, resolved. Pixel (x, y) is the quad's color on popcount(x & 15) of its 4 samples and
//! the clear color on the rest: their mean, pixel by pixel in 1920x1080 (colors multiples of 8: whole unorm8 means).
//! Then the input coverage mask (`detour.coverage_mask`, WGSL `@builtin(sample_mask)` in) of one triangle over the
//! whole target (a quad's diagonal pixels would be two invocations of partial masks), red mask · 16: 15 (240) without
//! varyings and with one (its planes past the mask: green and blue from the varying (0.25, 0.5019608)); with
//! `sample_shading` the mask ANDed with 1 << sample_index (the hardware gives the pixel's coverage; Mesa ANDs too),
//! red resolving to (16 + 32 + 64 + 128) / 4 = 60. Last, the `msaa` example's triangle: its edge pixels' masks worked
//! out exactly from the 4x positions (bit k: sample k), red mask · 16 on the covered samples and black on the rest.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;
const IN: [u8; 4] = [248, 128, 64, 255];
const OUT: [u8; 4] = [0, 0, 248, 255];

const PIPELINE: &str = r#"detour.pipeline @masked {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], samples = 4 : i32,
  vertex_buffers = [{stride = 12, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %x: f32) -> (vector<4xf32>, f32) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %x : vector<4xf32>, f32
}
func.func @fs(%x: f32) -> (vector<4xf32>, i32) attributes {detour.stage = "fragment", detour.sample_mask = true} {
  %c = arith.constant dense<[0.97254902, 0.50196078, 0.25098039, 1.0]> : vector<4xf32>
  %xi = arith.fptosi %x : f32 to i32
  %m15 = arith.constant 15 : i32
  %mask = arith.andi %xi, %m15 : i32
  return %c, %mask : vector<4xf32>, i32
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let corners = cx.clear_vertices()?;
    let rv = cx.resolve_vertices(W, H)?;
    let w = W as f32;
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 1.0, -1.0, w, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, -1.0, w, 1.0, 1.0, w])?;
    let clear = cx.msaa_clear_pipeline_samples(&codegen, OUT.map(|c| f32::from(c) / 255.0), 4)?;
    let resolve = cx.resolve_pipeline_samples(&codegen, 4)?;
    let p = cx.pipeline(&codegen, PIPELINE, "masked")?;
    println!("pipeline writes the sample mask: {}", p.layout.omask);
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
    let mut bad = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let k = (x & 15).count_ones();
            let want: [u8; 4] = std::array::from_fn(|c| ((u32::from(IN[c]) * k + u32::from(OUT[c]) * (4 - k)) / 4) as u8);
            let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            if got != want {
                bad.push((x, y, got, want));
            }
        }
    }
    println!("{} pixels, {} wrong", W * H, bad.len());
    if !bad.is_empty() {
        println!("   first (x, y, got, want): {:?}", &bad[..bad.len().min(6)]);
    }
    let mut ok = bad.is_empty() && p.layout.omask;
    let reader = |varying: bool, per_sample: bool| {
        let (vin, vty, ret, gb) = if varying {
            (", %v: vector<2xf32>", "vector<2xf32>", "%pos, %v : vector<4xf32>, vector<2xf32>", "%g = vector.extract %v[0] : f32 from vector<2xf32>
  %b = vector.extract %v[1] : f32 from vector<2xf32>")
        } else {
            ("", "", "%pos : vector<4xf32>", "%g = arith.constant 0.0 : f32
  %b = arith.constant 0.0 : f32")
        };
        let vs_out = if varying { "(vector<4xf32>, vector<2xf32>)".to_owned() } else { "vector<4xf32>".to_owned() };
        let attrs = if varying { r#"{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x2", offset = 8}"# } else { r#"{location = 0, format = "float32x2", offset = 0}"# };
        let fs_in = if varying { format!("%v: {vty}") } else { String::new() };
        let mask = if per_sample {
            "%cov = \"detour.coverage_mask\"() : () -> i32\n  %s = \"detour.sample_index\"() : () -> i32\n  %bit1 = arith.constant 1 : i32\n  %bit = arith.shli %bit1, %s : i32\n  %m = arith.andi %cov, %bit : i32"
        } else {
            "%m = \"detour.coverage_mask\"() : () -> i32"
        };
        format!(
            r#"detour.pipeline @reads {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], samples = 4 : i32, sample_shading = {per_sample},
  vertex_buffers = [{{stride = 16, attributes = [{attrs}]}}]}}
func.func @vs(%p: vector<2xf32>{vin}) -> {vs_out} attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return {ret}
}}
func.func @fs({fs_in}) -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  {mask}
  %mf = arith.sitofp %m : i32 to f32
  %k = arith.constant 0.062745098 : f32
  %r = arith.mulf %mf, %k : f32
  {gb}
  %one = arith.constant 1.0 : f32
  %c = vector.from_elements %r, %g, %b, %one : vector<4xf32>
  return %c : vector<4xf32>
}}
"#
        )
    };
    let full = cx.floats(&[-1.0, -1.0, 0.25, 0.5019608, 3.0, -1.0, 0.25, 0.5019608, -1.0, 3.0, 0.25, 0.5019608])?;
    let black = cx.msaa_clear_pipeline_samples(&codegen, [0.0, 0.0, 0.0, 1.0], 4)?;
    for (name, varying, per_sample, want) in [
        ("coverage mask, no varyings", false, false, [240u8, 0, 0, 255]),
        ("coverage mask, a varying", true, false, [240, 64, 128, 255]),
        ("coverage mask per sample, a varying", true, true, [60, 64, 128, 255]),
    ] {
        let p = cx.pipeline(&codegen, &reader(varying, per_sample), "reads")?;
        let msaa = cx.msaa_target_samples(W, H, TextureFormat::Rgba8Unorm, 4)?;
        let mut out = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&msaa);
        r.draw(&black, &[&corners], &[], 3, 1)?;
        r.draw(&p, &[&full], &[], 3, 1)?;
        r.target(&out);
        r.resolve(&resolve, &rv, &msaa)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = out.bytes()?;
        let wrong = px[..(W * H * 4) as usize].chunks(4).filter(|c| *c != want).count();
        println!("{name}: {wrong} of {} pixels not {want:?}, e.g. {:?}", W * H, &px[..4]);
        ok &= wrong == 0;
    }
    const CORNERS: [(f64, f64); 3] = [(100.25, 60.5), (1800.75, 300.125), (500.5, 1000.25)];
    const SAMPLES_4X: [(i64, i64); 4] = [(6, 2), (14, 6), (2, 10), (10, 14)];
    let mask_of = |x: u32, y: u32| -> Option<u32> {
        let v = CORNERS.map(|(px, py)| ((px * 256.0) as i64, (py * 256.0) as i64));
        let mut m = 0;
        for (k, &(sx, sy)) in SAMPLES_4X.iter().enumerate() {
            let p = (i64::from(x) * 256 + sx * 16, i64::from(y) * 256 + sy * 16);
            let e: Vec<i64> = (0..3).map(|i| (v[(i + 1) % 3].0 - v[i].0) * (p.1 - v[i].1) - (v[(i + 1) % 3].1 - v[i].1) * (p.0 - v[i].0)).collect();
            if e.contains(&0) {
                return None;
            }
            if e.iter().all(|e| *e > 0) || e.iter().all(|e| *e < 0) {
                m |= 1 << k;
            }
        }
        Some(m)
    };
    let clip: Vec<f32> = CORNERS.iter().flat_map(|(x, y)| [(x / f64::from(W) * 2.0 - 1.0) as f32, (y / f64::from(H) * 2.0 - 1.0) as f32, 0.25, 0.5019608]).collect();
    let tri = cx.floats(&clip)?;
    let p = cx.pipeline(&codegen, &reader(true, false), "reads")?;
    let msaa = cx.msaa_target_samples(W, H, TextureFormat::Rgba8Unorm, 4)?;
    let mut out = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&msaa);
    r.draw(&black, &[&corners], &[], 3, 1)?;
    r.draw(&p, &[&tri], &[], 3, 1)?;
    r.target(&out);
    r.resolve(&resolve, &rv, &msaa)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = out.bytes()?;
    let (mut partial, mut skipped, mut wrong) = (0, 0, Vec::new());
    for y in 0..H {
        for x in 0..W {
            let Some(m) = mask_of(x, y) else {
                skipped += 1;
                continue;
            };
            let k = m.count_ones();
            partial += usize::from(k != 0 && k != 4);
            let want = [m * 16 * k / 4, 64 * k / 4, 128 * k / 4, 255].map(|c| c as u8);
            let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            if got != want && k != 0 && k != 4 {
                wrong.push((x, y, m, got, want));
            }
        }
    }
    println!("coverage mask on a triangle's edge: {partial} partly covered pixels, {skipped} skipped, {} wrong", wrong.len());
    if !wrong.is_empty() {
        println!("   first (x, y, mask, got, want): {:?}", &wrong[..wrong.len().min(6)]);
    }
    ok &= wrong.is_empty() && partial > 0;
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
