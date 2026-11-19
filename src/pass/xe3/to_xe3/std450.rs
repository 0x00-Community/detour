use alloc::vec;
use alloc::vec::Vec;

use super::{Cr, Exact, GpuToXe3};
use crate::{A, Lanes, OpId, ValueId};

/// SPIR-V's extended math instructions (the GLSL.std.450 set, `spirv.GL.*`) and the core ops mlirpass's SPIR-V
/// lowering leaves alone (dot, integer division, FMod, SNegate), lowered on f32 and i32 lanes the way Mesa's NIR
/// builds them: transcendental functions on the EU's math unit, the rest from adds, multiplies and selects
pub(super) struct Std450;

impl Std450 {
    const LOG2_E: f32 = core::f32::consts::LOG2_E;
    const LN_2: f32 = core::f32::consts::LN_2;
    const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
    const PI: f32 = core::f32::consts::PI;
    /// atan on [0, 1] (NIR's `nir_atan`): u · (c0 + u²(c1 + u²(c2 + …)))
    const ATAN: [f32; 6] = [0.999_979_3, -0.332_675_64, 0.193_892_5, -0.117_350_32, 0.053_681_378, -0.012_132_321];

    fn f32s(cr: &Cr<'_, '_, '_>, op: OpId) -> bool {
        GpuToXe3::result_elem(cr, op) == Some("f32")
    }

    fn f(cr: &mut Cr<'_, '_, '_>, at: OpId, name: &str, x: &[ValueId], attrs: Vec<(&str, A)>) -> ValueId {
        Exact::op(cr, at, name, x, attrs)
    }

    fn i(cr: &mut Cr<'_, '_, '_>, at: OpId, name: &str, x: &[ValueId], attrs: Vec<(&str, A)>) -> ValueId {
        GpuToXe3::eu(cr, at, name, x, attrs, "i32")
    }

    fn sub(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.add", &[a, b], vec![("neg1", A::Bool(true))])
    }

    fn mul(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.mul", &[a, b], vec![])
    }

    /// a + b·c
    fn mad(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId, c: ValueId) -> ValueId {
        Exact::mad(cr, at, a, b, c, false)
    }

    /// `cond(c) ? a : b`
    fn csel(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId, c: ValueId, cond: &str) -> ValueId {
        Self::f(cr, at, "xe3.eu.csel", &[a, b, c], vec![("cond", A::str(cond))])
    }

    fn min(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.sel", &[a, b], vec![("cond", A::str("lt"))])
    }

    fn max(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.sel", &[a, b], vec![("cond", A::str("ge"))])
    }

    fn abs(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.mov", &[x], vec![("abs0", A::Bool(true))])
    }

    fn neg(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.mov", &[x], vec![("neg0", A::Bool(true))])
    }

    fn imm(cr: &mut Cr<'_, '_, '_>, at: OpId, v: f32) -> ValueId {
        Exact::imm(cr, at, v)
    }

    fn dot(cr: &mut Cr<'_, '_, '_>, at: OpId, a: &[ValueId], b: &[ValueId]) -> ValueId {
        let mut sum = Self::mul(cr, at, a[0], b[0]);
        for (x, y) in a.iter().zip(b).skip(1) {
            sum = Self::mad(cr, at, sum, *x, *y);
        }
        sum
    }

    /// 2^x
    fn exp2(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        Exact::math(cr, at, "exp", x)
    }

    fn log2(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        Exact::math(cr, at, "log", x)
    }

    fn floor(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.rndd", &[x], vec![])
    }

    /// `p` with `x`'s sign
    fn signed_like(cr: &mut Cr<'_, '_, '_>, at: OpId, p: ValueId, x: ValueId) -> ValueId {
        let n = Self::neg(cr, at, p);
        Self::csel(cr, at, n, p, x, "lt")
    }

    /// atan(t) for t in [0, 1]
    fn atan_unit(cr: &mut Cr<'_, '_, '_>, at: OpId, t: ValueId) -> ValueId {
        let t2 = Self::mul(cr, at, t, t);
        let mut p = Self::imm(cr, at, Self::ATAN[5]);
        for c in Self::ATAN[..5].iter().rev() {
            let c = Self::imm(cr, at, *c);
            p = Self::mad(cr, at, c, t2, p);
        }
        Self::mul(cr, at, t, p)
    }

    /// atan(|n| / |d|) over the first quadrant: min / max reduced to [0, 1], then π/2 - p where |n| > |d|
    fn atan_ratio(cr: &mut Cr<'_, '_, '_>, at: OpId, n: ValueId, d: ValueId) -> ValueId {
        let (an, ad) = (Self::abs(cr, at, n), Self::abs(cr, at, d));
        let lo = Self::min(cr, at, an, ad);
        let hi = Self::max(cr, at, an, ad);
        let q = Exact::div(cr, at, lo, hi);
        let zero = Self::imm(cr, at, 0.0);
        let t = Self::csel(cr, at, zero, q, hi, "z");
        let p = Self::atan_unit(cr, at, t);
        let half_pi = Self::imm(cr, at, Self::FRAC_PI_2);
        let other = Self::sub(cr, at, half_pi, p);
        let over = Self::sub(cr, at, ad, an);
        Self::csel(cr, at, other, p, over, "lt")
    }

    /// Each lane of a result `f` of the operands' lanes, on f32
    fn each(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], mut f: impl FnMut(&mut Cr<'_, '_, '_>, &[ValueId]) -> ValueId) -> bool {
        Self::f32s(cr, op) && cr.lanes_each(op, o, |cr, x| f(cr, x))
    }

    pub(super) fn unary(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], name: &str) -> bool {
        Self::each(cr, op, o, |cr, x| {
            let x = x[0];
            match name {
                "spirv.GL.Exp" => {
                    let e = Self::imm(cr, op, Self::LOG2_E);
                    let y = Self::mul(cr, op, x, e);
                    Self::exp2(cr, op, y)
                }
                "spirv.GL.Exp2" => Self::exp2(cr, op, x),
                "spirv.GL.Log" => {
                    let l = Self::log2(cr, op, x);
                    let ln2 = Self::imm(cr, op, Self::LN_2);
                    Self::mul(cr, op, l, ln2)
                }
                "spirv.GL.Log2" => Self::log2(cr, op, x),
                "spirv.GL.InverseSqrt" => Exact::math(cr, op, "rsq", x),
                "spirv.GL.Sin" => Exact::math(cr, op, "sin", x),
                "spirv.GL.Cos" => Exact::math(cr, op, "cos", x),
                "spirv.GL.Tan" => {
                    let (s, c) = (Exact::math(cr, op, "sin", x), Exact::math(cr, op, "cos", x));
                    Exact::div(cr, op, s, c)
                }
                "spirv.GL.Atan" => {
                    let one = Self::imm(cr, op, 1.0);
                    let p = Self::atan_ratio(cr, op, x, one);
                    Self::signed_like(cr, op, p, x)
                }
                "spirv.GL.Fract" => {
                    let fl = Self::floor(cr, op, x);
                    Self::sub(cr, op, x, fl)
                }
                "spirv.GL.Trunc" => Self::f(cr, op, "xe3.eu.rndz", &[x], vec![]),
                "spirv.GL.Radians" => {
                    let k = Self::imm(cr, op, Self::PI / 180.0);
                    Self::mul(cr, op, x, k)
                }
                "spirv.GL.Degrees" => {
                    let k = Self::imm(cr, op, 180.0 / Self::PI);
                    Self::mul(cr, op, x, k)
                }
                _ => {
                    let (zero, one, minus) = (Self::imm(cr, op, 0.0), Self::imm(cr, op, 1.0), Self::imm(cr, op, -1.0));
                    let below = Self::csel(cr, op, minus, zero, x, "lt");
                    Self::csel(cr, op, one, below, x, "gt")
                }
            }
        })
    }

    pub(super) fn binary(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], name: &str) -> bool {
        Self::each(cr, op, o, |cr, x| {
            let (a, b) = (x[0], x[1]);
            match name {
                "spirv.GL.Pow" => {
                    let l = Self::log2(cr, op, a);
                    let y = Self::mul(cr, op, b, l);
                    Self::exp2(cr, op, y)
                }
                "spirv.GL.Atan2" => {
                    let p = Self::atan_ratio(cr, op, a, b);
                    let pi = Self::imm(cr, op, Self::PI);
                    let back = Self::sub(cr, op, pi, p);
                    let p = Self::csel(cr, op, back, p, b, "lt");
                    Self::signed_like(cr, op, p, a)
                }
                "spirv.GL.Step" => {
                    let (zero, one) = (Self::imm(cr, op, 0.0), Self::imm(cr, op, 1.0));
                    let d = Self::sub(cr, op, b, a);
                    Self::csel(cr, op, zero, one, d, "lt")
                }
                _ => {
                    let r = Exact::math(cr, op, "inv", b);
                    let q = Self::mul(cr, op, a, r);
                    let fl = Self::floor(cr, op, q);
                    let m = Self::mul(cr, op, b, fl);
                    Self::sub(cr, op, a, m)
                }
            }
        })
    }

    pub(super) fn ternary(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], name: &str) -> bool {
        Self::each(cr, op, o, |cr, x| {
            let (a, b, c) = (x[0], x[1], x[2]);
            match name {
                "spirv.GL.FClamp" => {
                    let lo = Self::max(cr, op, a, b);
                    Self::min(cr, op, lo, c)
                }
                "spirv.GL.FMix" => {
                    let d = Self::sub(cr, op, b, a);
                    Self::mad(cr, op, a, c, d)
                }
                _ => {
                    let span = Self::sub(cr, op, b, a);
                    let from = Self::sub(cr, op, c, a);
                    let t = Exact::div(cr, op, from, span);
                    let (zero, one) = (Self::imm(cr, op, 0.0), Self::imm(cr, op, 1.0));
                    let t = Self::max(cr, op, t, zero);
                    let t = Self::min(cr, op, t, one);
                    let (two, three) = (Self::imm(cr, op, 2.0), Self::imm(cr, op, 3.0));
                    let u = Exact::mad(cr, op, three, t, two, true);
                    let t2 = Self::mul(cr, op, t, t);
                    Self::mul(cr, op, t2, u)
                }
            }
        })
    }

    /// dot, length, distance: a scalar of the operands' lanes
    pub(super) fn reduce(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], name: &str) -> bool {
        if !Self::f32s(cr, op) || o.iter().any(|v| v.len() != o[0].len()) {
            return false;
        }
        let v = match name {
            "spirv.Dot" => Self::dot(cr, op, &o[0], &o[1]),
            _ => {
                let d: Vec<ValueId> = match name {
                    "spirv.GL.Distance" => o[0].iter().zip(&o[1]).map(|(a, b)| Self::sub(cr, op, *a, *b)).collect(),
                    _ => o[0].clone(),
                };
                match d[..] {
                    [x] => Self::abs(cr, op, x),
                    _ => {
                        let s = Self::dot(cr, op, &d, &d);
                        Exact::sqrt(cr, op, s)
                    }
                }
            }
        };
        cr.replace(op, vec![vec![v]]);
        true
    }

    /// normalize, cross, reflect: a vector of the operands' lanes
    pub(super) fn vector(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], name: &str) -> bool {
        if !Self::f32s(cr, op) || o.iter().any(|v| v.len() != o[0].len()) {
            return false;
        }
        let values: Vec<ValueId> = match name {
            "spirv.GL.Normalize" => {
                let s = Self::dot(cr, op, &o[0], &o[0]);
                let r = Exact::math(cr, op, "rsq", s);
                o[0].iter().map(|x| Self::mul(cr, op, *x, r)).collect()
            }
            "spirv.GL.Cross" => {
                let ([ax, ay, az], [bx, by, bz]) = (&o[0][..], &o[1][..]) else { return false };
                let one = |cr: &mut Cr<'_, '_, '_>, p: ValueId, q: ValueId, r: ValueId, t: ValueId| {
                    let pq = Self::mul(cr, op, p, q);
                    Exact::mad(cr, op, pq, r, t, true)
                };
                let x = one(cr, *ay, *bz, *az, *by);
                let y = one(cr, *az, *bx, *ax, *bz);
                let z = one(cr, *ax, *by, *ay, *bx);
                vec![x, y, z]
            }
            _ => {
                let d = Self::dot(cr, op, &o[1], &o[0]);
                let two = Self::imm(cr, op, 2.0);
                let k = Self::mul(cr, op, d, two);
                o[0].iter().zip(&o[1]).map(|(i, n)| Exact::mad(cr, op, *i, k, *n, true)).collect()
            }
        };
        cr.replace(op, vec![values]);
        true
    }

    /// x · y for i32 lanes, the low 32 bits
    fn imul(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId, y: ValueId) -> ValueId {
        let lo = Self::i(cr, at, "xe3.eu.mul", &[x, y], vec![("half1", A::str("lo"))]);
        let hi = Self::i(cr, at, "xe3.eu.mul", &[x, y], vec![("half1", A::str("hi"))]);
        let sixteen = GpuToXe3::imm(cr, at, 16, "i32");
        let up = Self::i(cr, at, "xe3.eu.shl", &[hi, sixteen], vec![("unsigned", A::Bool(true))]);
        Self::i(cr, at, "xe3.eu.add", &[lo, up], vec![])
    }

    /// All ones where `x cond y` holds, zero elsewhere
    fn mask(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId, y: ValueId, cond: &str, unsigned: bool) -> ValueId {
        let mut attrs = vec![("cond", A::str(cond)), ("dst_as", A::str(if unsigned { "ud" } else { "d" }))];
        if unsigned {
            attrs.push(("unsigned", A::Bool(true)));
        }
        Self::i(cr, at, "xe3.eu.cmp", &[x, y], attrs)
    }

    fn and(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId, y: ValueId) -> ValueId {
        Self::i(cr, at, "xe3.eu.and", &[x, y], vec![("as", A::str("ud"))])
    }

    /// Unsigned n / d and n % d: a quotient from f32 reciprocals, refined once, then corrected by one
    fn udivrem(cr: &mut Cr<'_, '_, '_>, at: OpId, n: ValueId, d: ValueId) -> (ValueId, ValueId) {
        let u = || vec![("unsigned", A::Bool(true))];
        let nf = Self::f(cr, at, "xe3.eu.mov", &[n], u());
        let df = Self::f(cr, at, "xe3.eu.mov", &[d], u());
        let rcp = Exact::math(cr, at, "inv", df);
        let qf = Self::mul(cr, at, nf, rcp);
        let q = Self::i(cr, at, "xe3.eu.mov", &[qf], u());
        let qd = Self::imul(cr, at, q, d);
        let r = Self::i(cr, at, "xe3.eu.add", &[n, qd], vec![("neg1", A::Bool(true))]);
        let rf = Self::f(cr, at, "xe3.eu.mov", &[r], vec![]);
        let ef = Self::mul(cr, at, rf, rcp);
        let e = Self::i(cr, at, "xe3.eu.mov", &[ef], vec![]);
        let q = Self::i(cr, at, "xe3.eu.add", &[q, e], vec![]);
        let qd = Self::imul(cr, at, q, d);
        let r = Self::i(cr, at, "xe3.eu.add", &[n, qd], vec![("neg1", A::Bool(true))]);
        let zero = GpuToXe3::imm(cr, at, 0, "i32");
        let under = Self::mask(cr, at, r, zero, "lt", false);
        let q = Self::i(cr, at, "xe3.eu.add", &[q, under], vec![]);
        let back = Self::and(cr, at, d, under);
        let r = Self::i(cr, at, "xe3.eu.add", &[r, back], vec![]);
        let over = Self::mask(cr, at, r, d, "ge", true);
        let q = Self::i(cr, at, "xe3.eu.add", &[q, over], vec![("neg1", A::Bool(true))]);
        let back = Self::and(cr, at, d, over);
        let r = Self::i(cr, at, "xe3.eu.add", &[r, back], vec![("neg1", A::Bool(true))]);
        (q, r)
    }

    /// x with the sign `s` (0 or all ones) applied: (x ^ s) - s
    fn apply_sign(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId, s: ValueId) -> ValueId {
        let flipped = Self::i(cr, at, "xe3.eu.xor", &[x, s], vec![("as", A::str("ud"))]);
        Self::i(cr, at, "xe3.eu.add", &[flipped, s], vec![("neg1", A::Bool(true))])
    }

    fn sign_of(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        let k = GpuToXe3::imm(cr, at, 31, "i32");
        Self::i(cr, at, "xe3.eu.asr", &[x, k], vec![])
    }

    /// Integer division and remainders: UDiv, UMod, SDiv, SRem (the dividend's sign), SMod (the divisor's)
    pub(super) fn divide(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], name: &str) -> bool {
        if GpuToXe3::result_elem(cr, op) != Some("i32") {
            return false;
        }
        cr.lanes_each(op, o, |cr, x| {
            let (n, d) = (x[0], x[1]);
            if matches!(name, "spirv.UDiv" | "spirv.UMod") {
                let (q, r) = Self::udivrem(cr, op, n, d);
                return if name == "spirv.UDiv" { q } else { r };
            }
            let an = Self::i(cr, op, "xe3.eu.mov", &[n], vec![("abs0", A::Bool(true))]);
            let ad = Self::i(cr, op, "xe3.eu.mov", &[d], vec![("abs0", A::Bool(true))]);
            let (q, r) = Self::udivrem(cr, op, an, ad);
            let sn = Self::sign_of(cr, op, n);
            match name {
                "spirv.SDiv" => {
                    let both = Self::i(cr, op, "xe3.eu.xor", &[n, d], vec![("as", A::str("ud"))]);
                    let s = Self::sign_of(cr, op, both);
                    Self::apply_sign(cr, op, q, s)
                }
                "spirv.SRem" => Self::apply_sign(cr, op, r, sn),
                _ => {
                    let r = Self::apply_sign(cr, op, r, sn);
                    let both = Self::i(cr, op, "xe3.eu.xor", &[r, d], vec![("as", A::str("ud"))]);
                    let zero = GpuToXe3::imm(cr, op, 0, "i32");
                    let differ = Self::mask(cr, op, both, zero, "lt", false);
                    let some = Self::mask(cr, op, r, zero, "nz", false);
                    let fix = Self::and(cr, op, differ, some);
                    let add = Self::and(cr, op, d, fix);
                    Self::i(cr, op, "xe3.eu.add", &[r, add], vec![])
                }
            }
        })
    }

    /// SNegate, SAbs and SSign on i32 lanes
    pub(super) fn integer(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], name: &str) -> bool {
        if GpuToXe3::result_elem(cr, op) != Some("i32") {
            return false;
        }
        cr.lanes_each(op, o, |cr, x| match name {
            "spirv.SNegate" => Self::i(cr, op, "xe3.eu.mov", &[x[0]], vec![("neg0", A::Bool(true))]),
            "spirv.GL.SAbs" => Self::i(cr, op, "xe3.eu.mov", &[x[0]], vec![("abs0", A::Bool(true))]),
            _ => {
                let zero = GpuToXe3::imm(cr, op, 0, "i32");
                let neg = Self::mask(cr, op, x[0], zero, "lt", false);
                let pos = Self::mask(cr, op, x[0], zero, "gt", false);
                Self::i(cr, op, "xe3.eu.add", &[neg, pos], vec![("neg1", A::Bool(true))])
            }
        })
    }

    /// SClamp and UClamp on i32 lanes
    pub(super) fn clamp(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], unsigned: bool) -> bool {
        if GpuToXe3::result_elem(cr, op) != Some("i32") {
            return false;
        }
        cr.lanes_each(op, o, |cr, x| {
            let attrs = |cond: &'static str| {
                let mut a = vec![("cond", A::str(cond))];
                if unsigned {
                    a.push(("unsigned", A::Bool(true)));
                }
                a
            };
            let lo = Self::i(cr, op, "xe3.eu.sel", &[x[0], x[1]], attrs("ge"));
            Self::i(cr, op, "xe3.eu.sel", &[lo, x[2]], attrs("lt"))
        })
    }
}
