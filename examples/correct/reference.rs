//! The CPU reference: what GL ES / Vulkan say a draw writes, computed exactly. Coordinates are the target's memory
//! coordinates (row 0 first), which are detour's window coordinates (`ColorTarget`: row 0 is the viewport's y = 0).

use detour::{Blend, BlendEquation, BlendFactor, CullMode};

/// Sub-pixel precision: window coordinates snap to 1/256 of a pixel (Xe's 8-bit sub-pixel precision)
pub const SUB: i64 = 256;

/// Which edges own the pixel centres lying exactly on them
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillRule {
    /// a horizontal edge owns them when the triangle lies at larger y (else at smaller y)
    pub top_low: bool,
    /// a slanted or vertical edge owns them when the triangle lies at larger x (else at smaller x)
    pub left: bool,
}

impl FillRule {
    pub const ALL: [FillRule; 4] = [
        FillRule { top_low: true, left: true },
        FillRule { top_low: false, left: true },
        FillRule { top_low: true, left: false },
        FillRule { top_low: false, left: false },
    ];
}

/// The viewport transform as the GPU does it: ndc × half extent + centre, in f32
pub fn window(clip: [f32; 4], viewport: [f32; 4]) -> [f32; 2] {
    let [vx, vy, vw, vh] = viewport;
    let (sx, sy) = (vw * 0.5, vh * 0.5);
    let (tx, ty) = (vx + sx, vy + sy);
    let (x, y) = (clip[0] / clip[3], clip[1] / clip[3]);
    [x * sx + tx, y * sy + ty]
}

pub fn snap(v: f32) -> i64 {
    (f64::from(v) * SUB as f64).round_ties_even() as i64
}

/// Twice the signed area of a triangle in window coordinates (positive: counter-clockwise with y up)
pub fn area(p: [[i64; 2]; 3]) -> i64 {
    (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0])
}

/// Whether a triangle (window coordinates, snapped) is culled
pub fn culled(p: [[i64; 2]; 3], cull: CullMode, front_ccw: bool) -> bool {
    let a = area(p);
    let front = (a > 0) == front_ccw;
    match cull {
        CullMode::None => false,
        CullMode::Front => front,
        CullMode::Back => !front,
    }
}

/// The pixels a triangle covers, in a `w`×`h` target, by `rule`
pub fn triangle(p: [[i64; 2]; 3], w: u32, h: u32, rule: FillRule, mut hit: impl FnMut(u32, u32)) {
    let mut p = p;
    let a = area(p);
    if a == 0 {
        return;
    }
    if a < 0 {
        p.swap(1, 2);
    }
    let edges: [([i64; 2], [i64; 2]); 3] = [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])];
    let owns = |d: [i64; 2]| {
        let (dx, dy) = (d[0], d[1]);
        // interior on the left of the direction (y up): its normal is (-dy, dx)
        if dy == 0 { (dx > 0) == rule.top_low } else { (dy < 0) == rule.left }
    };
    let xs = p.map(|v| v[0]);
    let ys = p.map(|v| v[1]);
    let lo = |m: i64| ((m - SUB / 2).div_euclid(SUB)).max(0);
    let x0 = lo(*xs.iter().min().unwrap());
    let y0 = lo(*ys.iter().min().unwrap());
    let x1 = (xs.iter().max().unwrap().div_euclid(SUB) + 1).min(i64::from(w) - 1);
    let y1 = (ys.iter().max().unwrap().div_euclid(SUB) + 1).min(i64::from(h) - 1);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let s = [x * SUB + SUB / 2, y * SUB + SUB / 2];
            let inside = edges.iter().all(|&(a, b)| {
                let d = [b[0] - a[0], b[1] - a[1]];
                let e = d[0] * (s[1] - a[1]) - d[1] * (s[0] - a[0]);
                e > 0 || (e == 0 && owns(d))
            });
            if inside {
                hit(x as u32, y as u32);
            }
        }
    }
}

fn factor(f: BlendFactor, s: [f32; 4], d: [f32; 4], c: [f32; 4], ch: usize) -> f32 {
    match f {
        BlendFactor::Zero => 0.0,
        BlendFactor::One => 1.0,
        BlendFactor::SrcColor => s[ch],
        BlendFactor::OneMinusSrcColor => 1.0 - s[ch],
        BlendFactor::DstColor => d[ch],
        BlendFactor::OneMinusDstColor => 1.0 - d[ch],
        BlendFactor::SrcAlpha => s[3],
        BlendFactor::OneMinusSrcAlpha => 1.0 - s[3],
        BlendFactor::DstAlpha => d[3],
        BlendFactor::OneMinusDstAlpha => 1.0 - d[3],
        BlendFactor::ConstantColor => c[ch],
        BlendFactor::OneMinusConstantColor => 1.0 - c[ch],
        BlendFactor::ConstantAlpha => c[3],
        BlendFactor::OneMinusConstantAlpha => 1.0 - c[3],
        BlendFactor::SrcAlphaSaturate if ch == 3 => 1.0,
        BlendFactor::SrcAlphaSaturate => s[3].min(1.0 - d[3]),
    }
}

/// `s` blended over `d` (GL ES 3.0 §4.1.7); a fixed-point target clamps the source, the constant and the result
pub fn blend(b: &Blend, s: [f32; 4], d: [f32; 4], fixed: bool) -> [f32; 4] {
    let clamp = |v: [f32; 4]| if fixed { v.map(|c| c.clamp(0.0, 1.0)) } else { v };
    let (s, c) = (clamp(s), clamp(b.color));
    core::array::from_fn(|ch| {
        let i = usize::from(ch == 3);
        let (fs, fd) = (factor(b.src[i], s, d, c, ch), factor(b.dst[i], s, d, c, ch));
        let v = match b.equation[i] {
            BlendEquation::Add => s[ch] * fs + d[ch] * fd,
            BlendEquation::Subtract => s[ch] * fs - d[ch] * fd,
            BlendEquation::ReverseSubtract => d[ch] * fd - s[ch] * fs,
            BlendEquation::Min => s[ch].min(d[ch]),
            BlendEquation::Max => s[ch].max(d[ch]),
        };
        if fixed { v.clamp(0.0, 1.0) } else { v }
    })
}

/// A float in [0, 1] as unorm8: round to nearest (GL ES 3.0 §2.1.6.1); NaN is 0
pub fn unorm8(v: f32) -> u8 {
    if v.is_nan() { 0 } else { (v.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8 }
}

/// Units in the last place between two floats (same sign assumed close; NaNs and infinities equal only themselves)
pub fn ulps(a: f32, b: f32) -> u64 {
    if a == b || (a.is_nan() && b.is_nan()) {
        return 0;
    }
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return u64::MAX;
    }
    let key = |f: f32| {
        let i = f.to_bits() as i32;
        if i < 0 { i64::from(i32::MIN) - i64::from(i) } else { i64::from(i) }
    };
    (key(a) - key(b)).unsigned_abs()
}
