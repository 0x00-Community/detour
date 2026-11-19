//! Cases of what programs compute: vertex fetch (formats, strides, offsets, defaults), interpolation, std140
//! uniform blocks, and math lowered from SPIR-V, read back from an RGBA32F target and checked against the CPU.

use detour::{CullMode, DrawPrimitive, VertexElement, VertexFormat};

use crate::gpu::{Op, Target, bytes};
use crate::reference::ulps;

pub type Check = Box<dyn Fn(&[[f32; 4]], u32) -> Result<String, String>>;

pub struct Case {
    pub name: String,
    pub target: Target,
    pub op: Op,
    pub check: Check,
}

const QUAD_VS: &str = "#version 310 es\nlayout(location = 0) in vec2 a_pos;\nvoid main() { gl_Position = vec4(a_pos, 0.0, 1.0); }\n";

/// The full target as two triangles in clip coordinates
fn quad_op(vs: &str, fs: &str, w: u32, h: u32) -> Op {
    let pos = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
    Op {
        program: (vs.into(), fs.into()),
        buffers: vec![(bytes(&pos), 8)],
        elements: vec![VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 }],
        uniforms: Vec::new(),
        primitive: DrawPrimitive::Triangles,
        first: 0,
        count: 6,
        instances: 1,
        viewport: [0.0, 0.0, w as f32, h as f32],
        scissor: None,
        cull: CullMode::None,
        front_ccw: true,
        blend: None,
        mask: [true; 4],
    }
}

fn exact(name: &str, got: [f32; 4], want: [f32; 4]) -> Result<(), String> {
    match got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits() || (*a == b)) {
        true => Ok(()),
        false => Err(format!("{name}: {got:?}, want {want:?}")),
    }
}

/// Vertex fetch: every format, interleaved with strides and offsets, two buffers, the defaults of missing components;
/// constant attributes must reach the fragments exactly
pub fn vertex_fetch() -> Vec<Case> {
    let vs = "#version 310 es
layout(location = 0) in vec4 a_pos;
layout(location = 1) in float a_one;
layout(location = 2) in vec3 a_three;
layout(location = 3) in vec4 a_four;
layout(location = 4) in vec4 a_short;
out vec4 v_a;
out vec4 v_b;
out vec4 v_c;
void main() { gl_Position = a_pos; v_a = vec4(a_one, a_three); v_b = a_four; v_c = a_short; }
";
    let fs = "#version 310 es
precision highp float;
in vec4 v_a;
in vec4 v_b;
in vec4 v_c;
layout(location = 0) out vec4 o;
void main() { int x = int(gl_FragCoord.x); o = x == 0 ? v_a : x == 1 ? v_b : v_c; }
";
    let pos = [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let (one, three, four, short) = (0.3f32, [1.25f32, -7.5, 1e-3], [0.1f32, 0.2, 0.3, 0.4], [3.5f32, -2.25]);
    // buffer 0, stride 44: pad 4 | pos.xy | one | pad 8 | three.xyz | pad 4; buffer 1, stride 28: pad 8 | four | short.xy (Float32x2 into a vec4)
    let mut b0 = Vec::new();
    let mut b1 = Vec::new();
    for p in pos {
        b0.extend(bytes(&[f32::NAN, p[0], p[1], one, 9.0, 9.0, three[0], three[1], three[2], f32::NAN, 7.0]));
        b1.extend(bytes(&[5.0, 5.0, four[0], four[1], four[2], four[3], short[0], short[1]]));
    }
    let op = Op {
        buffers: vec![(b0, 44), (b1, 32)],
        elements: vec![
            VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 4 },
            VertexElement { buffer: 0, format: VertexFormat::Float32, offset: 12 },
            VertexElement { buffer: 0, format: VertexFormat::Float32x3, offset: 24 },
            VertexElement { buffer: 1, format: VertexFormat::Float32x4, offset: 8 },
            VertexElement { buffer: 1, format: VertexFormat::Float32x2, offset: 24 },
        ],
        ..quad_op(vs, fs, 3, 4)
    };
    let check: Check = Box::new(move |px, w| {
        for (i, p) in px.iter().enumerate() {
            let x = i as u32 % w;
            let want = match x {
                0 => [one, three[0], three[1], three[2]],
                1 => four,
                _ => [short[0], short[1], 0.0, 1.0],
            };
            exact(&format!("pixel ({x}, {})", i as u32 / w), *p, want)?;
        }
        Ok("exact".into())
    });
    let mut cases = vec![Case { name: "vertex formats, strides, offsets, defaults".into(), target: Target::rgba32f(3, 4), op, check }];
    // a gradient: an attribute linear in x and y interpolated to pixel centres
    let vs = "#version 310 es
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); v_uv = a_uv; }
";
    let fs = "#version 310 es
precision highp float;
in vec2 v_uv;
layout(location = 0) out vec4 o;
void main() { o = vec4(v_uv, gl_FragCoord.xy); }
";
    let (w, h) = (64u32, 32u32);
    let mut data = Vec::new();
    for p in pos {
        data.extend([p[0], p[1], (p[0] + 1.0) * 0.5, (p[1] + 1.0) * 0.5]);
    }
    let op = Op {
        buffers: vec![(bytes(&data), 16)],
        elements: vec![VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 }, VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 8 }],
        ..quad_op(vs, fs, w, h)
    };
    let check: Check = Box::new(move |px, w| {
        let mut worst = 0.0f32;
        let h = px.len() as u32 / w;
        for (i, p) in px.iter().enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            if p[2] != fx || p[3] != fy {
                return Err(format!("gl_FragCoord at ({x}, {y}): {:?}, want {fx}, {fy}", &p[2..]));
            }
            let want = [fx / w as f32, fy / h as f32];
            worst = worst.max((p[0] - want[0]).abs()).max((p[1] - want[1]).abs());
        }
        match worst <= 2e-6 {
            true => Ok(format!("gl_FragCoord exact, interpolation within {worst:e}")),
            false => Err(format!("interpolation off by {worst:e}")),
        }
    });
    cases.push(Case { name: "interpolation and gl_FragCoord".into(), target: Target::rgba32f(w, h), op, check });
    // equal colors at each vertex (what acdes's solid program gets) reach every pixel unchanged
    let vs = "#version 310 es
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_color;
out vec4 v_color;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); v_color = a_color; }
";
    let fs = "#version 310 es\nprecision highp float;\nin vec4 v_color;\nlayout(location = 0) out vec4 o;\nvoid main() { o = v_color; }\n";
    let color = [0.1f32, 1.0 / 3.0, 0.7, 0.123_456_79];
    let tris = [
        [[-1.0f32, -1.0], [0.9, -0.73], [0.17, 1.0]],
        [[0.9, -0.73], [1.0, 1.0], [0.17, 1.0]],
        [[-1.0, -1.0], [0.17, 1.0], [-1.0, 1.0]],
        [[-1.0, -1.0], [1.0, -1.0], [0.9, -0.73]],
        [[0.9, -0.73], [1.0, -1.0], [1.0, 1.0]],
    ];
    let mut data = Vec::new();
    for t in tris {
        for p in t {
            data.extend([p[0], p[1]]);
            data.extend(color);
        }
    }
    let op = Op {
        buffers: vec![(bytes(&data), 24)],
        elements: vec![VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 }, VertexElement { buffer: 0, format: VertexFormat::Float32x4, offset: 8 }],
        count: 15,
        ..quad_op(vs, fs, 61, 37)
    };
    let check: Check = Box::new(move |px, w| {
        let off: Vec<(usize, u64)> = px.iter().map(|p| p.iter().zip(color).map(|(a, b)| ulps(*a, b)).max().unwrap()).enumerate().filter(|(_, u)| *u > 0).collect();
        match off.iter().max_by_key(|(_, u)| *u) {
            None => Ok("exact".into()),
            Some((i, u)) => Err(format!("{} pixels off, up to {u} ulp at ({}, {}): {:?}", off.len(), *i as u32 % w, *i as u32 / w, px[*i])),
        }
    });
    cases.push(Case { name: "constant varying".into(), target: Target::rgba32f(61, 37), op, check });
    cases
}

/// std140 blocks in both stages: scalars, vectors, vec3 padding, arrays, matrices, and a block of 32 vec4s
pub fn uniforms() -> Vec<Case> {
    let mut cases = Vec::new();
    // member: GLSL type, name, std140 offset, components read (as floats)
    let members: &[(&str, &str, usize, usize)] = &[
        ("float", "a", 0, 1),
        ("vec2", "b", 8, 2),
        ("vec3", "c", 16, 3),
        ("float", "d", 28, 1),
        ("vec4", "e", 32, 4),
        ("vec3", "f", 48, 3),
        ("vec2", "g", 64, 2),
        ("float", "h", 72, 1),
        ("vec3", "i", 80, 3),
        ("vec3", "j", 96, 3),
        ("float", "k", 108, 1),
        ("vec4", "l", 112, 4),
    ];
    let value = |offset: usize, c: usize| 1.0 + offset as f32 + 0.25 * c as f32;
    let mut block = String::from("layout(std140, binding = 1) uniform U {\n");
    let mut chain = String::new();
    for (n, (ty, name, _, comps)) in members.iter().enumerate() {
        block += &format!("  {ty} {name};\n");
        let v = match comps {
            1 => format!("vec4(u.{name}, 0.0, 0.0, 0.0)"),
            2 => format!("vec4(u.{name}, 0.0, 0.0)"),
            3 => format!("vec4(u.{name}, 0.0)"),
            _ => format!("u.{name}"),
        };
        chain += &format!("  if (x == {n}) o = {v};\n");
    }
    block += "} u;\n";
    let fs = format!("#version 310 es\nprecision highp float;\n{block}layout(location = 0) out vec4 o;\nvoid main() {{\n  int x = int(gl_FragCoord.x);\n  o = vec4(-1.0);\n{chain}}}\n");
    let size = 128;
    let mut data = vec![f32::NAN; size / 4];
    for (_, _, offset, comps) in members {
        for c in 0..*comps {
            data[offset / 4 + c] = value(*offset, c);
        }
    }
    let mut op = quad_op(QUAD_VS, &fs, members.len() as u32, 2);
    op.uniforms = vec![(1, bytes(&data))];
    let want: Vec<[f32; 4]> = members.iter().map(|(_, _, offset, comps)| core::array::from_fn(|c| if c < *comps { value(*offset, c) } else { 0.0 })).collect();
    let check: Check = Box::new(move |px, w| {
        for (i, p) in px.iter().enumerate() {
            exact(&format!("member {}", members[i % w as usize].1), *p, want[i % w as usize])?;
        }
        Ok("exact".into())
    });
    cases.push(Case { name: "std140 scalars, vectors, vec3 padding".into(), target: Target::rgba32f(members.len() as u32, 2), op, check });
    // arrays and matrices
    let fs = "#version 310 es
precision highp float;
layout(std140, binding = 1) uniform U {
  float s[3];
  vec2 t[2];
  mat2 m;
  mat3 n;
  vec4 q;
} u;
layout(location = 0) out vec4 o;
void main() {
  int x = int(gl_FragCoord.x);
  o = vec4(-1.0);
  if (x == 0) o = vec4(u.s[0], u.s[1], u.s[2], 0.0);
  if (x == 1) o = vec4(u.t[0], u.t[1]);
  if (x == 2) o = vec4(u.m[0], u.m[1]);
  if (x == 3) o = vec4(u.n[0], 0.0);
  if (x == 4) o = vec4(u.n[1], u.n[2].z);
  if (x == 5) o = u.q;
}
";
    let mut data = [f32::NAN; 192 / 4];
    let mut set = |at: usize, v: &[f32]| v.iter().enumerate().for_each(|(i, f)| data[at / 4 + i] = *f);
    set(0, &[1.0]);
    set(16, &[2.0]);
    set(32, &[3.0]);
    set(48, &[4.0, 5.0]);
    set(64, &[6.0, 7.0]);
    set(80, &[8.0, 9.0]);
    set(96, &[10.0, 11.0]);
    set(112, &[12.0, 13.0, 14.0]);
    set(128, &[15.0, 16.0, 17.0]);
    set(144, &[18.0, 19.0, 20.0]);
    set(160, &[21.0, 22.0, 23.0, 24.0]);
    let want = [[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 0.0], [15.0, 16.0, 17.0, 20.0], [21.0, 22.0, 23.0, 24.0]];
    let mut op = quad_op(QUAD_VS, fs, 6, 1);
    op.uniforms = vec![(1, bytes(&data[..176 / 4]))];
    let check: Check = Box::new(move |px, w| {
        for (i, p) in px.iter().enumerate() {
            exact(&format!("pixel {}", i as u32 % w), *p, want[i % w as usize])?;
        }
        Ok("exact".into())
    });
    cases.push(Case { name: "std140 arrays and matrices".into(), target: Target::rgba32f(6, 1), op, check });
    // many members: 32 vec4s, each read, and a sum of them all
    for n in [9usize, 16, 32, 48] {
        let mut block = String::from("layout(std140, binding = 1) uniform U {\n");
        let mut chain = String::new();
        let mut sum = String::from("vec4(0.0)");
        for i in 0..n {
            let ty = if i % 3 == 1 { "vec3" } else { "vec4" };
            block += &format!("  {ty} m{i};\n");
            let v = if ty == "vec3" { format!("vec4(u.m{i}, 1.0)") } else { format!("u.m{i}") };
            chain += &format!("  if (x == {i}) o = {v};\n");
            sum = format!("{sum} + {v}");
        }
        block += "} u;\n";
        let fs = format!(
            "#version 310 es\nprecision highp float;\n{block}layout(location = 0) out vec4 o;\nvoid main() {{\n  int x = int(gl_FragCoord.x);\n  o = {sum};\n{chain}}}\n"
        );
        let mut data = Vec::new();
        let mut want = Vec::new();
        for i in 0..n {
            let v: [f32; 4] = core::array::from_fn(|c| (i * 4 + c) as f32 * 0.5 + 0.125);
            data.extend(v);
            want.push(if i % 3 == 1 { [v[0], v[1], v[2], 1.0] } else { v });
        }
        let total: [f32; 4] = core::array::from_fn(|c| want.iter().fold(0.0f32, |s, v| s + v[c]));
        let mut op = quad_op(QUAD_VS, &fs, n as u32 + 1, 1);
        op.uniforms = vec![(1, bytes(&data))];
        let check: Check = Box::new(move |px, _| {
            for (i, p) in px.iter().enumerate() {
                exact(&format!("member {i}"), *p, if i < n { want[i] } else { total })?;
            }
            Ok("exact".into())
        });
        cases.push(Case { name: format!("std140 block of {n} vectors"), target: Target::rgba32f(n as u32 + 1, 1), op, check });
    }
    // a block in each stage, as acdes's programs have
    let vs = "#version 310 es
layout(location = 0) in vec2 a_pos;
layout(std140, binding = 0) uniform V { vec2 size; vec3 tint; float k; } v;
out vec4 v_t;
void main() { gl_Position = vec4(a_pos * v.size, 0.0, 1.0); v_t = vec4(v.tint, v.k); }
";
    let fs = "#version 310 es
precision highp float;
layout(std140, binding = 1) uniform U { vec3 a; vec3 b; vec4 c; } u;
in vec4 v_t;
layout(location = 0) out vec4 o;
void main() { o = gl_FragCoord.x < 2.0 ? v_t : gl_FragCoord.x < 3.0 ? vec4(u.a, u.b.y) : u.c; }
";
    let mut op = quad_op(vs, fs, 4, 2);
    let mut vblock = [0.0f32; 8];
    vblock[..2].copy_from_slice(&[1.0, 1.0]);
    vblock[4..8].copy_from_slice(&[0.5, 0.25, 0.125, 2.5]);
    let mut fblock = [f32::NAN; 12];
    fblock[..3].copy_from_slice(&[1.5, 2.5, 3.5]);
    fblock[4..7].copy_from_slice(&[4.5, 5.5, 6.5]);
    fblock[8..12].copy_from_slice(&[7.5, 8.5, 9.5, 10.5]);
    op.uniforms = vec![(0, bytes(&vblock)), (1, bytes(&fblock))];
    let check: Check = Box::new(move |px, w| {
        for (i, p) in px.iter().enumerate() {
            let want = match i as u32 % w {
                0 | 1 => [0.5, 0.25, 0.125, 2.5],
                2 => [1.5, 2.5, 3.5, 5.5],
                _ => [7.5, 8.5, 9.5, 10.5],
            };
            exact(&format!("pixel {i}"), *p, want)?;
        }
        Ok("exact".into())
    });
    cases.push(Case { name: "a block in each stage".into(), target: Target::rgba32f(4, 2), op, check });
    cases
}

/// How far from the exact value a result may be
#[derive(Clone, Copy)]
pub enum Bound {
    /// units in the last place
    Ulp(f64),
    /// GLSL's exp / exp2: 3 + 2|x| ulp
    ExpUlp,
    /// log / log2: 3 ulp outside [0.5, 2], 2^-21 absolute inside
    Log,
    /// an absolute error
    Abs(f64),
    /// exactly IEEE's (or a flushed denormal)
    Exact,
}

/// One function: its GLSL (of `x`, and `y` when it takes two), the exact value in f64, the bound, and the inputs:
/// ranges of bit patterns (start, step) for x (and y)
pub struct Func {
    pub name: &'static str,
    pub glsl: &'static str,
    pub exact: fn(f64, f64) -> f64,
    pub bound: Bound,
    pub x: (u32, u32),
    pub y: (u32, u32),
    /// inputs GLSL leaves the result undefined at, or gives no precision for
    pub undefined: fn(f32, f32) -> bool,
}

/// vello's erf approximation, in f32 as the shader computes it
fn erf7(x: f32) -> f32 {
    let xx = x * x;
    let x2 = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
    x2 / (1.0 + x2 * x2).sqrt()
}

pub fn funcs() -> Vec<Func> {
    let f = |v: f32| v.to_bits();
    vec![
        Func { name: "x / y", glsl: "x / y", exact: |x, y| x / y, bound: Bound::Ulp(2.5), x: (f(-3e5), 0x00f3_1a7b), y: (f(1e-30), 0x0003_c1d1), undefined: |_, y| !(1.0 / 8.5e37..=8.5e37).contains(&y.abs()) },
        Func { name: "x / y, near 1", glsl: "x / y", exact: |x, y| x / y, bound: Bound::Ulp(2.5), x: (f(0.9), 0x113), y: (f(1.1), 0x2f1), undefined: |_, y| !(1.0 / 8.5e37..=8.5e37).contains(&y.abs()) },
        Func { name: "1 / x", glsl: "1.0 / x", exact: |x, _| 1.0 / x, bound: Bound::Ulp(2.5), x: (f(1e-37), 0x0004_6f31), y: (0, 0), undefined: |x, _| !(1.0 / 8.5e37..=8.5e37).contains(&x.abs()) },
        Func { name: "sqrt", glsl: "sqrt(x)", exact: |x, _| x.sqrt(), bound: Bound::Ulp(3.0), x: (f(1e-37), 0x0004_6f31), y: (0, 0), undefined: |x, _| x < 0.0 },
        Func { name: "inversesqrt", glsl: "inversesqrt(x)", exact: |x, _| 1.0 / x.sqrt(), bound: Bound::Ulp(2.0), x: (f(1e-37), 0x0004_6f31), y: (0, 0), undefined: |x, _| x <= 0.0 },
        Func { name: "exp", glsl: "exp(x)", exact: |x, _| x.exp(), bound: Bound::ExpUlp, x: (f(-87.0), 0x0000_25f1), y: (0, 0), undefined: |_, _| false },
        Func { name: "exp, small", glsl: "exp(x)", exact: |x, _| x.exp(), bound: Bound::ExpUlp, x: (f(-2.0), 0x0000_1a1f), y: (0, 0), undefined: |_, _| false },
        Func { name: "exp2", glsl: "exp2(x)", exact: |x, _| x.exp2(), bound: Bound::ExpUlp, x: (f(-125.0), 0x0000_25f1), y: (0, 0), undefined: |_, _| false },
        Func { name: "log", glsl: "log(x)", exact: |x, _| x.ln(), bound: Bound::Log, x: (f(1e-37), 0x0004_6f31), y: (0, 0), undefined: |x, _| x <= 0.0 },
        Func { name: "log2", glsl: "log2(x)", exact: |x, _| x.log2(), bound: Bound::Log, x: (f(1e-37), 0x0004_6f31), y: (0, 0), undefined: |x, _| x <= 0.0 },
        Func { name: "pow", glsl: "pow(x, y)", exact: |x, y| x.powf(y), bound: Bound::Ulp(0.0), x: (f(0.01), 0x0000_7a13), y: (f(-6.0), 0x0000_2b1f), undefined: |x, y| x <= 0.0 || !x.is_finite() || !y.is_finite() },
        Func { name: "sin", glsl: "sin(x)", exact: |x, _| x.sin(), bound: Bound::Abs(1.0 / 2048.0), x: (f(-core::f32::consts::PI), 0x0000_2001), y: (0, 0), undefined: |x, _| x.is_nan() || x.abs() > core::f32::consts::PI },
        Func { name: "cos", glsl: "cos(x)", exact: |x, _| x.cos(), bound: Bound::Abs(1.0 / 2048.0), x: (f(0.0), 0x0000_3c11), y: (0, 0), undefined: |x, _| x.is_nan() || x.abs() > core::f32::consts::PI },
        Func { name: "floor", glsl: "floor(x)", exact: |x, _| x.floor(), bound: Bound::Exact, x: (f(-3e3), 0x0001_1fd3), y: (0, 0), undefined: |_, _| false },
        Func { name: "fract", glsl: "fract(x)", exact: |x, _| x - x.floor(), bound: Bound::Ulp(1.0), x: (f(-3e3), 0x0001_1fd3), y: (0, 0), undefined: |_, _| false },
        Func { name: "x * y + 1", glsl: "x * y + 1.0", exact: |x, y| f64::from((x as f32 * y as f32) + 1.0), bound: Bound::Ulp(0.5), x: (f(-3.0), 0x0000_4f13), y: (f(0.3), 0x0000_3a91), undefined: |_, _| false },
        Func { name: "mix", glsl: "mix(x, y, 0.3)", exact: |x, y| x * 0.7 + y * 0.3, bound: Bound::Ulp(4.0), x: (f(-3.0), 0x0000_4f13), y: (f(0.3), 0x0000_3a91), undefined: |x, y| x.is_infinite() || y.is_infinite() },
        Func { name: "mix at 1", glsl: "mix(x, y, 1.0)", exact: |_, y| y, bound: Bound::Exact, x: (f(-3.0), 0x0000_4f13), y: (f(0.3), 0x0000_3a91), undefined: |x, _| !x.is_finite() },
        Func {
            name: "erf7 (vello's)",
            glsl: "erf7(x)",
            exact: |x, _| f64::from(erf7(x as f32)),
            bound: Bound::Ulp(4.0),
            x: (f(-4.0), 0x0000_4007),
            y: (0, 0),
            undefined: |_, _| false,
        },
        Func {
            name: "acdes's shadow sigmoid",
            glsl: "0.5 + 0.5 * x / sqrt(x * x + 1.0)",
            exact: |x, _| {
                let x = x as f32;
                f64::from(0.5 + 0.5 * x / (x * x + 1.0).sqrt())
            },
            bound: Bound::Abs(1.0 / 4194304.0),
            x: (f(-40.0), 0x0000_5a11),
            y: (0, 0),
            undefined: |_, _| false,
        },
    ]
}

/// Special values for the second row: NaN, infinities, zeros, denormals, extremes, ±1
pub const SPECIALS: [u32; 16] = [
    0x7fc0_0000,
    0x7f80_0000,
    0xff80_0000,
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x3f80_0000,
    0xbf80_0000,
    0x0080_0000,
    0x4049_0fdb,
    0xc2b0_0000,
    0x42b0_0000,
];

pub const MATH_W: u32 = 64;
pub const MATH_H: u32 = 33;

/// The program of `f`: rows 0..32 its ranges, row 32 x and y from `SPECIALS`
pub fn math_fs(f: &Func) -> String {
    let mut special = String::from("0u");
    for (i, b) in SPECIALS.iter().enumerate().rev() {
        special = format!("(k == {i}u ? {b:#x}u : {special})");
    }
    let mut special_y = String::from("0u");
    for (i, b) in SPECIALS.iter().enumerate().rev() {
        special_y = format!("(k == {i}u ? {b:#x}u : {special_y})");
    }
    format!(
        "#version 310 es
precision highp float;
precision highp int;
layout(location = 0) out vec4 o;
float erf7(float x) {{
  float xx = x * x;
  float x2 = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
  return x2 / sqrt(1.0 + x2 * x2);
}}
void main() {{
  uint i = uint(gl_FragCoord.y) * {MATH_W}u + uint(gl_FragCoord.x);
  uint bx = {:#x}u + i * {:#x}u;
  uint by = {:#x}u + i * {:#x}u;
  if (uint(gl_FragCoord.y) == {}u) {{
    uint k = uint(gl_FragCoord.x) % 16u;
    uint j = uint(gl_FragCoord.x) / 16u;
    bx = {special};
    k = (k + j * 5u) % 16u;
    by = {special_y};
  }}
  float x = uintBitsToFloat(bx);
  float y = uintBitsToFloat(by);
  o = vec4({}, x, y, 1.0);
}}
",
        f.x.0,
        f.x.1,
        f.y.0,
        f.y.1,
        MATH_H - 1,
        f.glsl
    )
}

pub fn math_op(f: &Func) -> Op {
    quad_op(QUAD_VS, &math_fs(f), MATH_W, MATH_H)
}

fn denormal(v: f64) -> bool {
    v != 0.0 && v.abs() < f64::from(f32::MIN_POSITIVE)
}

/// The result `got` of `f` at `x`, `y` within its bound: Ok(ulps off), Err(why). Denormal inputs may be kept or
/// flushed to zero (of their sign) and either result is right (Vulkan without float controls); inputs where GLSL
/// leaves the result undefined, or the spec gives no precision, are not judged
pub fn judge(f: &Func, x: f32, y: f32, got: f32) -> Result<f64, String> {
    if (f.undefined)(x, y) {
        return Ok(0.0);
    }
    let flush = |v: f32| if denormal(v.into()) { 0.0f32.copysign(v) } else { v };
    let mut tries = vec![(x, y)];
    if (flush(x), flush(y)) != (x, y) || flush(x).to_bits() != x.to_bits() || flush(y).to_bits() != y.to_bits() {
        tries.push((flush(x), flush(y)));
    }
    let mut first = None;
    for (x, y) in tries {
        match judge_one(f, x, y, got) {
            Ok(u) => return Ok(u),
            Err(e) => {
                first.get_or_insert(e);
            }
        }
    }
    Err(first.unwrap_or_default())
}

fn judge_one(f: &Func, x: f32, y: f32, got: f32) -> Result<f64, String> {
    let exact = (f.exact)(x.into(), y.into());
    let rounded = exact as f32;
    if got.to_bits() == rounded.to_bits() || (got.is_nan() && exact.is_nan()) || (got == 0.0 && rounded == 0.0) {
        return Ok(0.0);
    }
    if denormal(exact) && got == 0.0 {
        return Ok(0.0);
    }
    if exact.is_nan() || got.is_nan() || exact.is_infinite() || got.is_infinite() || rounded.is_infinite() {
        return Err(format!("{got:e}, want {rounded:e}"));
    }
    let ulp = |v: f64| {
        let a = (v.abs() as f32).max(f32::MIN_POSITIVE);
        f64::from(f32::from_bits(a.to_bits() + 1) - a)
    };
    let off = (f64::from(got) - exact).abs() / ulp(exact);
    let ok = match f.bound {
        Bound::Ulp(0.0) => {
            let inherited = 3.0 + 2.0 * (f64::from(y) * f64::from(x).log2()).abs();
            off <= inherited + f64::from(y).abs() * 2f64.powi(-21) * core::f64::consts::LN_2 * exact.abs() / ulp(exact)
        }
        Bound::Ulp(n) => off <= n,
        Bound::ExpUlp => off <= 3.0 + 2.0 * f64::from(x).abs(),
        Bound::Log => match (0.5..=2.0).contains(&f64::from(x)) {
            true => (f64::from(got) - exact).abs() <= 2f64.powi(-21),
            false => off <= 3.0,
        },
        Bound::Abs(e) => (f64::from(got) - exact).abs() <= e,
        Bound::Exact => false,
    };
    match ok {
        true => Ok(off),
        false => Err(format!("{got:e}, want {exact:e} ({off:.1} ulp)")),
    }
}

/// gl_VertexID (with a first vertex), gl_InstanceID, gl_FrontFacing, and n varyings
pub fn builtins() -> Vec<Case> {
    let mut cases = Vec::new();
    let vs = "#version 310 es
out vec4 v_id;
void main() {
  int id = gl_VertexID % 6;
  float x = (id == 1 || id == 2 || id == 4) ? 1.0 : -1.0;
  float y = (id == 2 || id == 4 || id == 5) ? 1.0 : -1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
  v_id = vec4(float(gl_VertexID / 6), float(gl_InstanceID), 0.25, 1.0);
}
";
    let fs = "#version 310 es\nprecision highp float;\nin vec4 v_id;\nlayout(location = 0) out vec4 o;\nvoid main() { o = v_id; }\n";
    let mut op = quad_op(vs, fs, 8, 8);
    (op.buffers, op.elements, op.first, op.count) = (Vec::new(), Vec::new(), 18, 6);
    let check: Check = Box::new(|px, _| {
        for p in px {
            exact("gl_VertexID / 6, gl_InstanceID", *p, [3.0, 0.0, 0.25, 1.0])?;
        }
        Ok("exact".into())
    });
    cases.push(Case { name: "gl_VertexID from first 18, no attributes".into(), target: Target::rgba32f(8, 8), op, check });
    let fs = "#version 310 es
precision highp float;
layout(location = 0) out vec4 o;
void main() { o = gl_FrontFacing ? vec4(1.0, gl_FragCoord.xy, 0.0) : vec4(0.5, gl_FragCoord.xy, 0.0); }
";
    // the left half counter-clockwise (front), the right half clockwise
    let pos = [-1.0f32, -1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0, -1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0];
    let mut op = quad_op(QUAD_VS, fs, 16, 4);
    op.buffers = vec![(bytes(&pos), 8)];
    op.count = 12;
    let check: Check = Box::new(|px, w| {
        for (i, p) in px.iter().enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            let facing = if x < w / 2 { 1.0 } else { 0.5 };
            exact(&format!("pixel ({x}, {y})"), *p, [facing, x as f32 + 0.5, y as f32 + 0.5, 0.0])?;
        }
        Ok("exact".into())
    });
    cases.push(Case { name: "gl_FrontFacing and gl_FragCoord".into(), target: Target::rgba32f(16, 4), op, check });
    for n in [3usize, 4, 8, 15] {
        let mut vs = String::from("#version 310 es\nlayout(location = 0) in vec2 a_pos;\n");
        let mut fs = String::from("#version 310 es\nprecision highp float;\n");
        let mut sum = String::from("vec4(0.0)");
        let mut body = String::new();
        for k in 0..n {
            vs += &format!("out vec4 v{k};\n");
            fs += &format!("in vec4 v{k};\n");
            let p = (1u32 << k) as f32;
            body += &format!("  v{k} = vec4({p:?}, {:?}, {:?}, 1.0);\n", p + 0.5, 3.0 * p);
            sum = format!("{sum} + v{k} * {:?}", (k + 1) as f32);
        }
        vs += &format!("void main() {{\n  gl_Position = vec4(a_pos, 0.0, 1.0);\n{body}}}\n");
        fs += &format!("layout(location = 0) out vec4 o;\nvoid main() {{ o = {sum}; }}\n");
        let want: [f32; 4] = core::array::from_fn(|c| {
            (0..n).fold(0.0f32, |s, k| {
                let p = (1u32 << k) as f32;
                s + [p, p + 0.5, 3.0 * p, 1.0][c] * (k + 1) as f32
            })
        });
        let check: Check = Box::new(move |px, _| {
            for p in px {
                exact("the varyings' weighted sum", *p, want)?;
            }
            Ok("exact".into())
        });
        cases.push(Case { name: format!("{n} varyings"), target: Target::rgba32f(5, 3), op: quad_op(QUAD_VS, "", 5, 3), check });
        let last = cases.last_mut().unwrap();
        last.op.program = (vs, fs);
    }
    cases
}

/// A vertex shader indexing a local array with a computed index (GLSL ES 3.00 §12.30: required in every stage)
pub fn arrays() -> Vec<Case> {
    let vs = "#version 310 es
out vec4 v_c;
void main() {
  int id = gl_VertexID % 6;
  vec2 c[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
  gl_Position = vec4(c[id], 0.0, 1.0);
  v_c = vec4(0.25, 0.5, 0.75, 1.0);
}
";
    let fs = "#version 310 es
precision highp float;
in vec4 v_c;
layout(location = 0) out vec4 o;
void main() {
  float k[4] = float[4](1.0, 2.0, 4.0, 8.0);
  int i = int(gl_FragCoord.x) % 4;
  k[i] += 16.0;
  k[(i + 1) % 4] *= 3.0;
  o = v_c * (k[i] + k[(i + 1) % 4] * 0.5);
}
";
    let mut op = quad_op(vs, fs, 8, 2);
    (op.buffers, op.elements) = (Vec::new(), Vec::new());
    let check: Check = Box::new(|px, w| {
        for (i, p) in px.iter().enumerate() {
            let x = i as u32 % w;
            let base = [1.0f32, 2.0, 4.0, 8.0];
            let i = x as usize % 4;
            let k = base[i] + 16.0 + base[(i + 1) % 4] * 3.0 * 0.5;
            exact(&format!("pixel {x}"), *p, [0.25 * k, 0.5 * k, 0.75 * k, k])?;
        }
        Ok("exact".into())
    });
    vec![Case { name: "local arrays, computed indices".into(), target: Target::rgba32f(8, 2), op, check }]
}
