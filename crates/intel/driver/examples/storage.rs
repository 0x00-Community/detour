//! Storage textures (`texture_storage_2d`, `textureStore`, read-write `textureLoad`) on Xe3, written by compute:
//! `@fill` over a 37 x 23 grid (the lanes past x 36 outside, dropped) stores (x, y, x y, 1) into an rgba32float
//! texture, x y + 0.5 into an r32float one, (x, y, x + y) / 255 into an rgba8unorm one; `@double` reads the rgba32float
//! texture and stores it doubled into another; after `Recorder::texture_barrier` a draw samples the rgba8unorm texture
//! (nearest, texel centers) into a target. Every texel against its definition, bit for bit; the bytes past each
//! texture untouched.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 37;
const H: u32 = 23;

const SAMPLE: &str = r#"detour.pipeline @p {vertex = @vs, fragment = @fs, topology = "triangle-strip", targets = ["rgba8unorm"], filter = "nearest",
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x2", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %uv: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uv : vector<4xf32>, vector<2xf32>
}
func.func @fs(%uv: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %out = "detour.sample"(%uv) {binding = 0 : i64} : (vector<2xf32>) -> vector<4xf32>
  return %out : vector<4xf32>
}
"#;

fn unorm(v: u32) -> u8 {
    ((v as f32 * 0.003_921_568_6_f32) * 255.0).round() as u8
}

fn report(name: &str, got: &[u8], want: &[u8]) -> bool {
    let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
    println!("{name}: {bad} of {} bytes wrong", want.len());
    if bad > 0 && let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        println!("  first at byte {i}: {} (want {})", got[i], want[i]);
    }
    bad == 0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let src = include_str!("storage/kernels.mlir");
    let (fill, double) = (cx.kernel(&codegen, src, "fill")?, cx.kernel(&codegen, src, "double")?);
    let sampling = cx.pipeline(&codegen, SAMPLE, "p")?;
    let size = |f: TextureFormat| (W * H * f.bytes()) as usize;
    let mut rgba = cx.texture(W, H, TextureFormat::Rgba32Float, &vec![0; size(TextureFormat::Rgba32Float)])?;
    let mut red = cx.texture(W, H, TextureFormat::R32Float, &vec![0; size(TextureFormat::R32Float)])?;
    let mut bytes = cx.texture(W, H, TextureFormat::Rgba8Unorm, &vec![0; size(TextureFormat::Rgba8Unorm)])?;
    let mut doubled = cx.texture(W, H, TextureFormat::Rgba32Float, &vec![0; size(TextureFormat::Rgba32Float)])?;
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])?;
    let mut out = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let groups = [W.div_ceil(16), H];
    let mut r = cx.record();
    r.dispatch_with(&fill, groups, &[W], &[&rgba, &red, &bytes])?;
    r.dispatch_with(&double, groups, &[W], &[&rgba, &doubled])?;
    r.texture_barrier()?;
    r.target(&out);
    r.draw(&sampling, &[&quad], &[&bytes], 4, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let texels = |f: &dyn Fn(u32, u32) -> Vec<u8>| -> Vec<u8> { (0..H).flat_map(|y| (0..W).flat_map(move |x| f(x, y))).collect() };
    let floats = |v: [f32; 4]| v.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<u8>>();
    let mut ok = true;
    let want = texels(&|x, y| floats([x as f32, y as f32, (x * y) as f32, 1.0]));
    ok &= report("rgba32float (x, y, x y, 1)", &rgba.bytes()?[..want.len()], &want);
    let want = texels(&|x, y| ((x * y) as f32 + 0.5).to_le_bytes().to_vec());
    ok &= report("r32float x y + 0.5", &red.bytes()?[..want.len()], &want);
    let want = texels(&|x, y| vec![unorm(x), unorm(y), unorm(x + y), 255]);
    ok &= report("rgba8unorm (x, y, x + y) / 255", &bytes.bytes()?[..want.len()], &want);
    let want2 = texels(&|x, y| floats([2.0 * x as f32, 2.0 * y as f32, 2.0 * (x * y) as f32, 2.0]));
    ok &= report("read-write: rgba32float doubled", &doubled.bytes()?[..want2.len()], &want2);
    ok &= report("sampled after the barrier", &out.bytes()?[..want.len()], &want);
    for (name, t, n) in [("rgba32float", &mut rgba, size(TextureFormat::Rgba32Float)), ("r32float", &mut red, size(TextureFormat::R32Float)), ("rgba8unorm", &mut bytes, size(TextureFormat::Rgba8Unorm))] {
        let past = &t.bytes()?[n..];
        let touched = past.iter().filter(|b| **b != 0).count();
        println!("{name}: {touched} of the {} bytes past it written", past.len());
        ok &= touched == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
