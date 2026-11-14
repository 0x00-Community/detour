//! The shader math against Rust's f32, bit for bit: 65536 operand sets (a, b, c, d), one a pixel, each the product of
//! two rgba32float textures' texels (the sampler flushes denormals, so a denormal is a normal float times 2^-100); per pipeline four results a pixel in an rgba32float target. `a / b` and `sqrt` are the EU's approximate
//! inv / rsq refined (`Exact` in the codegen), the rest single instructions. The operands: random normal floats of
//! every exponent, denormals, equal and adjacent pairs, and rows of zeros, infinities and NaNs. Any NaN equals any NaN; min / max
//! of -0 and +0 may be either (minnumf / maxnumf leave it open, as Rust's f32::min / max).

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 256;

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A normal float of exponent within ±`spread` of 0, either sign.
    fn float(&mut self, spread: i32) -> f32 {
        let r = self.next();
        let e = (127 + (r % (2 * spread as u64 + 1)) as i32 - spread) as u32;
        f32::from_bits(((r >> 40) as u32 & 0x8000_0000) | (e << 23) | (r >> 8) as u32 & 0x7f_ffff)
    }
}

fn operands() -> Vec<[f32; 4]> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let special = [0.0, -0.0, 1.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.5, 3.0, 1e-30, 1e30];
    (0..W * W)
        .map(|i| {
            let mut v = [rng.float(126), rng.float(126), rng.float(126), rng.float(126)];
            match i % 16 {
                // equal operands, and close ones
                0 => v[1] = v[0],
                1 => v[1] = f32::from_bits(v[0].to_bits() + 1),
                // integers and halves for floor
                2 => v[2] = (rng.next() % 2001) as f32 / 2.0 - 500.0,
                // narrow exponents: quotients and roots near 1
                3 => v = [rng.float(2), rng.float(2), rng.float(2), rng.float(2)],
                // denormal operands
                4 => v[0] = f32::from_bits(rng.next() as u32 & 0x807f_ffff),
                5 => v[1] = f32::from_bits(rng.next() as u32 & 0x807f_ffff),
                6 => v[2] = f32::from_bits(rng.next() as u32 & 0x807f_ffff),
                _ => {}
            }
            if i < 2 * W {
                let k = special.len() as u32;
                v[0] = special[(i % k) as usize];
                v[1] = special[(i / k % k) as usize];
            }
            v
        })
        .collect()
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// For min / max: either zero of two zeros.
fn same_or_zero(a: f32, b: f32) -> bool {
    same(a, b) || (a == 0.0 && b == 0.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let ops = operands();
    // a denormal v as (v * 2^100) * 2^-100, exact both ways
    let split = |v: f32| if v.is_subnormal() { (v * 2f32.powi(100), 2f32.powi(-100)) } else { (v, 1.0) };
    let texels = |part: fn((f32, f32)) -> f32| -> Vec<u8> { ops.iter().flatten().flat_map(|v| part(split(*v)).to_le_bytes()).collect() };
    let values = cx.texture(W, W, TextureFormat::Rgba32Float, &texels(|p| p.0))?;
    let scales = cx.texture(W, W, TextureFormat::Rgba32Float, &texels(|p| p.1))?;
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])?;

    type Reference = fn([f32; 4]) -> [f32; 4];
    let pipelines: [(&str, [&str; 4], Reference); 2] = [
        ("arith", ["a / b", "sqrt(|a|)", "fma(a, b, c)", "floor(c)"], |[a, b, c, _]| [a / b, a.abs().sqrt(), a.mul_add(b, c), c.floor()]),
        ("select", ["min(a, b)", "max(a, b)", "a < b ? c : d", "a >= b ? c : d"], |[a, b, c, d]| [a.min(b), a.max(b), if a < b { c } else { d }, if a >= b { c } else { d }]),
    ];
    let mut wrong = 0;
    for (name, what, reference) in pipelines {
        let pipeline = cx.pipeline(&codegen, include_str!("math/ops.mlir"), name)?;
        let mut target = cx.target(W, W, TextureFormat::Rgba32Float, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&pipeline, &[&quad], &[&values, &scales], 4, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px: Vec<f32> = target.bytes()?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        for (k, what) in what.iter().enumerate() {
            let bad: Vec<_> = ops
                .iter()
                .zip(px.as_chunks::<4>().0)
                .filter(|(o, got)| !if name == "select" && k < 2 && o[0] == 0.0 && o[1] == 0.0 { same_or_zero } else { same }(got[k], reference(**o)[k]))
                .map(|(o, got)| (*o, got[k], reference(*o)[k]))
                .collect();
            println!("{what}: {} of {} wrong {:?}", bad.len(), ops.len(), &bad[..bad.len().min(4)]);
            wrong += bad.len();
        }
    }
    assert_eq!(wrong, 0, "results off Rust's f32");
    println!("ok");
    Ok(())
}
