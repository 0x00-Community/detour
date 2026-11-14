//! f64 arithmetic on Xe3 against Rust's, bit for bit (roadmap R8.0): 65536 invocations, each reading (a, b, c, d) and
//! writing a + b, a - b, a b, fma(a, b, c), -a, |a|, a 0.1, (a as f32) as f64 + (d as i32) as f64, a < b ? a : b,
//! a == b ? c : a, max(a, b), a / b, √a, floor, ceil, trunc and round-to-even of a (no f64 in the math unit: see the codegen's `Exact64`). The operands:
//! random doubles of every exponent, denormals, equal and adjacent pairs, zeros, infinities and NaNs; d within i32's
//! range. Any NaN equals any NaN.

use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const N: usize = 1 << 16;
const RESULTS: usize = 17;
const NAMES: [&str; RESULTS] = ["addf", "subf", "mulf", "fma", "negf", "absf", "mulf(0.1)", "truncf/extf + fptosi/sitofp", "select(olt)", "select(oeq)", "maxnumf", "divf", "sqrt", "floor", "ceil", "trunc", "roundeven"];

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A double of exponent within ±`spread` of 0 (a denormal below), either sign.
    fn double(&mut self, spread: i64) -> f64 {
        let r = self.next();
        let e = (1023 + (r % (2 * spread as u64 + 1)) as i64 - spread).max(0) as u64;
        f64::from_bits((r & 0x8000_0000_0000_0000) | (e << 52) | (self.next() & 0x000f_ffff_ffff_ffff))
    }
}

fn operands() -> Vec<[f64; 4]> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let special = [0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.5, 3.0, 1e-310, f64::MAX, f64::MIN_POSITIVE];
    (0..N)
        .map(|i| {
            let mut v = [rng.double(1100), rng.double(1100), rng.double(1100), 0.0];
            match i % 8 {
                0 => v[1] = v[0],
                1 => v[1] = f64::from_bits(v[0].to_bits() + 1),
                2 => v = [rng.double(40), rng.double(40), rng.double(40), 0.0],
                3 => v[2] = -v[0] * v[1],
                // a / b next to a tie of a subnormal quotient's rounding: (2m + 1) 2^-1075 b
                4 => {
                    let b = f64::from_bits(0x3ff0_0000_0000_0000 | (rng.next() & 0x000f_ffff_ffff_ffff)) * 2f64.powi((rng.next() % 64) as i32 - 32);
                    let m = (rng.next() % (1 << 40)) as f64;
                    v[1] = b;
                    v[0] = (2.0 * m + 1.0) * b * 2f64.powi(-1000) * 2f64.powi(-75);
                }
                // b's mantissa all ones
                5 => v[1] = f64::from_bits((v[1].to_bits() & 0xfff0_0000_0000_0000) | 0x000f_ffff_ffff_ffff),
                // a a square (its root next to a representable one), and one ulp above it
                6 => {
                    let s = rng.double(500).abs();
                    v[0] = f64::from_bits((s * s).to_bits() + (rng.next() & 1));
                }
                _ => {}
            }
            if i < special.len() * special.len() {
                v[0] = special[i % special.len()];
                v[1] = special[i / special.len()];
            }
            v[3] = ((rng.next() as i64) % 4_000_000_000) as f64 * 0.5;
            v
        })
        .collect()
}

fn reference([a, b, c, d]: [f64; 4]) -> [f64; RESULTS] {
    [a + b, a - b, a * b, a.mul_add(b, c), -a, a.abs(), a * 0.1, (a as f32) as f64 + (d as i32) as f64, if a < b { a } else { b }, if a == b { c } else { a }, a.max(b), a / b, a.sqrt(), a.floor(), a.ceil(), a.trunc(), a.round_ties_even()]
}

fn same(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("math64/ops.mlir"), "ops")?;
    let ops = operands();
    let src = cx.buffer(&ops.iter().flatten().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let mut dst = cx.buffer(&vec![0xa5; N * RESULTS * 8])?;
    let mut r = cx.record();
    r.dispatch(&kernel, [(N / 16) as u32, 1], &[src.addr as u32, dst.addr as u32])?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let got: Vec<f64> = dst.bytes()?[..N * RESULTS * 8].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    let mut ok = true;
    for (k, name) in NAMES.iter().enumerate() {
        // max of -0 and +0 may be either (maxnumf leaves it open, as Rust's f64::max)
        let zeros = |i: usize| k == 10 && ops[i][0] == 0.0 && ops[i][1] == 0.0 && got[i * RESULTS + k] == 0.0;
        let bad: Vec<usize> = (0..N).filter(|i| !same(got[i * RESULTS + k], reference(ops[*i])[k]) && !zeros(*i)).collect();
        println!("{name}: {} of {N} wrong", bad.len());
        if let Some(&i) = bad.first() {
            let o = ops[i];
            println!("  first: invocation {i}, operands {:?} ({:x?}): got {:e} ({:#x}), want {:e} ({:#x})", o, o.map(f64::to_bits), got[i * RESULTS + k], got[i * RESULTS + k].to_bits(), reference(o)[k], reference(o)[k].to_bits());
        }
        ok &= bad.is_empty();
    }
    println!("{} (device {:#x})", if ok { "ok" } else { "FAILED" }, xe.info().device_id);
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
