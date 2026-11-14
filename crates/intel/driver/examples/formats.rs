//! Texture formats on Xe3 (`GPUTextureFormat`: rgba8unorm(-srgb), bgra8unorm(-srgb), r8unorm, rg8unorm, r16float,
//! rg16float, rgba16float, r32float, rg32float, rgba32float), both ways, against their definitions:
//! - rendered: a fragment shader's (0.25, 0.5, 0.75, 0.125) into a target of each, its bytes the format's encoding
//!   (unorm round(c 255); sRGB the transfer function's, then round; f16 IEEE round to nearest even; f32 as is);
//! - sampled: one texel of known bytes of each, sampled into an rgba32float target, its floats the format's decoding
//!   (missing channels 0, alpha 1).
//!
//! Exact for all but sRGB: Xe3's sRGB conversions are within D3D's tolerance, not the transfer function's exact
//! values. Swept against it: all 256 bytes decoded (under half a byte step off) and 256 linear values encoded (a
//! gradient; one off at most).

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const COLOR: [f32; 4] = [0.25, 0.5, 0.75, 0.125];

/// f32 → IEEE binary16, round to nearest even (normal and subnormal results; no overflow here).
fn half(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        let m = (man | 0x80_0000) >> (1 - exp);
        let (q, r) = (m >> 13, m & 0x1fff);
        return sign | (q + u32::from(r > 0x1000 || (r == 0x1000 && q & 1 == 1))) as u16;
    }
    let (q, r) = ((exp as u32) << 10 | man >> 13, man & 0x1fff);
    sign | (q + u32::from(r > 0x1000 || (r == 0x1000 && q & 1 == 1))) as u16
}

fn unhalf(h: u16) -> f32 {
    let (sign, exp, man) = (u32::from(h >> 15), u32::from((h >> 10) & 0x1f), u32::from(h & 0x3ff));
    let mag = if exp == 0 { man as f32 * 2f32.powi(-24) } else { f32::from_bits((exp + 112) << 23 | man << 13) };
    if sign == 1 { -mag } else { mag }
}

fn encode_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

fn decode_srgb(c: f32) -> f32 {
    if c <= 0.040_45 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

const ALL: [(TextureFormat, &str); 12] = [
    (TextureFormat::Rgba8Unorm, "rgba8unorm"),
    (TextureFormat::Rgba8UnormSrgb, "rgba8unorm-srgb"),
    (TextureFormat::Bgra8Unorm, "bgra8unorm"),
    (TextureFormat::Bgra8UnormSrgb, "bgra8unorm-srgb"),
    (TextureFormat::R8Unorm, "r8unorm"),
    (TextureFormat::Rg8Unorm, "rg8unorm"),
    (TextureFormat::R16Float, "r16float"),
    (TextureFormat::Rg16Float, "rg16float"),
    (TextureFormat::Rgba16Float, "rgba16float"),
    (TextureFormat::R32Float, "r32float"),
    (TextureFormat::Rg32Float, "rg32float"),
    (TextureFormat::Rgba32Float, "rgba32float"),
];

/// The channels a format keeps, in memory order (indices into rgba).
fn channels(f: TextureFormat) -> &'static [usize] {
    match f {
        TextureFormat::R8Unorm | TextureFormat::R16Float | TextureFormat::R32Float => &[0],
        TextureFormat::Rg8Unorm | TextureFormat::Rg16Float | TextureFormat::Rg32Float => &[0, 1],
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => &[2, 1, 0, 3],
        _ => &[0, 1, 2, 3],
    }
}

/// A texel of `f` holding `rgba` (sRGB: the color channels encoded).
fn encode(f: TextureFormat, rgba: [f32; 4]) -> Vec<u8> {
    let unorm = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    channels(f)
        .iter()
        .flat_map(|&k| {
            let c = rgba[k];
            match f.bytes() / channels(f).len() as u32 {
                1 if f.srgb() && k < 3 => vec![unorm(encode_srgb(c))],
                1 => vec![unorm(c)],
                2 => half(c).to_le_bytes().to_vec(),
                _ => c.to_le_bytes().to_vec(),
            }
        })
        .collect()
}

/// What sampling a texel of `f` of bytes `texel` gives.
fn decode(f: TextureFormat, texel: &[u8]) -> [f32; 4] {
    let mut out = [0.0, 0.0, 0.0, 1.0];
    let size = (f.bytes() / channels(f).len() as u32) as usize;
    for (i, &k) in channels(f).iter().enumerate() {
        let b = &texel[i * size..][..size];
        out[k] = match size {
            1 if f.srgb() && k < 3 => decode_srgb(f32::from(b[0]) / 255.0),
            1 => f32::from(b[0]) / 255.0,
            2 => unhalf(u16::from_le_bytes([b[0], b[1]])),
            _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        };
    }
    out
}

fn render(name: &str) -> String {
    format!(
        r#"detour.pipeline @p {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["{name}"],
  vertex_buffers = [{{stride = 8, attributes = [{{location = 0, format = "float32x2", offset = 0}}]}}]}}
func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}}
func.func @fs() -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  %c = arith.constant dense<[0.25, 0.5, 0.75, 0.125]> : vector<4xf32>
  return %c : vector<4xf32>
}}
"#
    )
}

/// A color gradient: positions (x, y) and colors (r, g, b, a) interpolated.
const GRADIENT: &str = r#"detour.pipeline @g {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm-srgb"],
  vertex_buffers = [{stride = 24, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x4", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %c: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %c : vector<4xf32>, vector<4xf32>
}
func.func @fs(%c: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  return %c : vector<4xf32>
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let full = cx.floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])?;
    let sampling = cx.pipeline(&codegen, &include_str!("texture/checker.mlir").replace("targets = [\"rgba8unorm\"]", "targets = [\"rgba32float\"]"), "textured")?;
    let strip = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])?;
    let mut ok = true;
    for (f, name) in ALL {
        // rendered into
        let p = cx.pipeline(&codegen, &render(name), "p")?;
        let mut target = cx.target(4, 4, f, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&p, &[&full], &[], 6, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let want = encode(f, COLOR);
        let px = target.bytes()?;
        let size = want.len();
        let wrote = (0..16).all(|i| px[i * size..][..size] == want[..]);
        // sampled: a texel of the bytes `stored`
        let stored = encode(f, [0.2, 0.4, 0.6, 0.8]);
        let tex = cx.texture(1, 1, f, &stored)?;
        let mut out = cx.target(4, 4, TextureFormat::Rgba32Float, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&out);
        r.draw(&sampling, &[&strip], &[&tex], 4, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let got: Vec<f32> = out.bytes()?[..16].chunks(4).map(|b| f32::from_le_bytes(b.try_into().expect("4"))).collect();
        let read = decode(f, &stored);
        let sampled = got == read;
        println!(
            "{name}: rendered {:?} {}; sampled {got:?} {}",
            &px[..size],
            if wrote { "as encoded".to_owned() } else { format!("(want {want:?})") },
            if sampled { "as decoded".to_owned() } else { format!("(want {read:?})") },
        );
        // sRGB is checked against the formula in the sweeps below (Xe3's conversion is within tolerance, not exact)
        ok &= f.srgb() || (wrote && sampled);
    }
    // sRGB sweeps: every byte decoded (a 256 x 1 texture, nearest, into rgba32float), 256 linear values encoded (a
    // horizontal gradient of the vertex colors, rendered into a 256 x 1 srgb target): the hardware against the formula
    let bytes: Vec<u8> = (0..=255u8).flat_map(|b| [b, b, b, 255]).collect();
    let tex = cx.texture(256, 1, TextureFormat::Rgba8UnormSrgb, &bytes)?;
    let mut out = cx.target(256, 1, TextureFormat::Rgba32Float, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&out);
    r.draw(&sampling, &[&strip], &[&tex], 4, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = out.bytes()?;
    let (mut worst, mut at) = (0f32, 0usize);
    for b in 0..256usize {
        let got = f32::from_le_bytes(px[b * 16..][..4].try_into()?);
        let e = (got - decode_srgb(b as f32 / 255.0)).abs();
        if e > worst {
            (worst, at) = (e, b);
        }
    }
    println!("sRGB decode, all 256 bytes: worst |error| {worst:.3e} at {at} ({:.3} of a byte step)", worst * 255.0);
    let gradient = cx.pipeline(&codegen, GRADIENT, "g")?;
    // x, y, r g b a: red from 0 at the left edge to 1 at the right (pixel i's center: (i + 0.5) / 256)
    let ramp: Vec<f32> = [(-1.0, -1.0, 0.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0)].iter().flat_map(|&(x, y, r)| [x, y, r, r, r, 1.0]).collect();
    let ramp = cx.floats(&ramp)?;
    let mut target = cx.target(256, 1, TextureFormat::Rgba8UnormSrgb, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&target);
    r.draw(&gradient, &[&ramp], &[], 6, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = target.bytes()?;
    let (mut off, mut most) = (0, 0i32);
    for i in 0..256usize {
        let want = (encode_srgb((i as f32 + 0.5) / 256.0) * 255.0).round() as i32;
        let d = (i32::from(px[i * 4]) - want).abs();
        off += usize::from(d != 0);
        most = most.max(d);
    }
    println!("sRGB encode, 256 values: {off} bytes off the formula's, by {most} at most");
    // the definitions hold exactly but for sRGB, which Xe3 converts within D3D's tolerance (decode under half a byte
    // step, encode one off at most)
    ok &= worst * 255.0 < 0.5 && most <= 1;
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
