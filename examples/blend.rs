//! Blending and the color mask on Xe3, checked against GL ES 3.0's equations (§4.1.7) computed on the CPU (no
//! clamping in a floating-point target): a
//! triangle covering the target, of one uniform color, drawn over a cleared target with each blend state; draws
//! stacked; RGBA32F and RGBA8 targets.
//!
//!   cargo run --release --example blend

use detour::{Attrib, BlendEquation, BlendFactor, BufferTarget, Capability, ClearMask, ColorFormat, Context, Device, DrawMode, ShaderType, Toolchain, UniformValue, Usage};

const SIZE: u32 = 16;
const VERT: &str = "#version 300 es\nin vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
const FRAG: &str = "#version 300 es\nprecision highp float;\nuniform vec4 color;\nout vec4 o;\nvoid main() { o = color; }\n";
/// One triangle over the whole viewport
const COVER: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];

/// A blend state as GL sets it, and what it computes
#[derive(Clone, Copy)]
struct Mode {
    src: [BlendFactor; 2],
    dst: [BlendFactor; 2],
    equation: [BlendEquation; 2],
    color: [f32; 4],
}

impl Mode {
    fn of(src: BlendFactor, dst: BlendFactor) -> Mode {
        Mode { src: [src; 2], dst: [dst; 2], equation: [BlendEquation::Add; 2], color: [0.0; 4] }
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

    /// `s` blended over `d`: clamped to [0, 1] in a fixed-point target, not in a floating-point one
    fn apply(&self, s: [f32; 4], d: [f32; 4], fixed: bool) -> [f32; 4] {
        core::array::from_fn(|ch| {
            let i = usize::from(ch == 3);
            let (fs, fd) = (Mode::factor(self.src[i], s, d, self.color, ch), Mode::factor(self.dst[i], s, d, self.color, ch));
            let v = match self.equation[i] {
                BlendEquation::Add => s[ch] * fs + d[ch] * fd,
                BlendEquation::Subtract => s[ch] * fs - d[ch] * fd,
                BlendEquation::ReverseSubtract => d[ch] * fd - s[ch] * fs,
                BlendEquation::Min => s[ch].min(d[ch]),
                BlendEquation::Max => s[ch].max(d[ch]),
            };
            if fixed { v.clamp(0.0, 1.0) } else { v }
        })
    }

    fn set(&self, gl: &mut Context<'_>) {
        gl.blend_func_separate(self.src[0], self.dst[0], self.src[1], self.dst[1]);
        gl.blend_equation_separate(self.equation[0], self.equation[1]);
        gl.blend_color(self.color);
    }
}

/// A case: the target's clear color, then draws (a color and a blend state, None: blending off), with a color mask
struct Case {
    name: &'static str,
    format: ColorFormat,
    clear: [f32; 4],
    draws: Vec<([f32; 4], Option<Mode>)>,
    mask: [bool; 4],
}

impl Case {
    fn expected(&self) -> [f32; 4] {
        let mut d = self.clear;
        for &(s, mode) in &self.draws {
            let out = mode.map_or(s, |m| m.apply(s, d, self.format == ColorFormat::Rgba8Unorm));
            d = core::array::from_fn(|c| if self.mask[c] { out[c] } else { d[c] });
        }
        match self.format {
            ColorFormat::Rgba8Unorm => d.map(|c| (c * 255.0).round() / 255.0),
            ColorFormat::Rgba32Float => d,
        }
    }

    fn drawn(&self, tools: &Toolchain<'_>, device: &Device<'_>) -> Result<Vec<[f32; 4]>, String> {
        let mut gl = Context::new(tools, SIZE, SIZE, self.format).map_err(|e| e.0)?;
        let mut shaders = Vec::new();
        for (ty, src) in [(ShaderType::Vertex, VERT), (ShaderType::Fragment, FRAG)] {
            let s = gl.create_shader(ty);
            gl.shader_source(s, src);
            gl.compile_shader(s);
            if !gl.shader_compiled(s) {
                return Err(gl.shader_log(s));
            }
            shaders.push(s);
        }
        let p = gl.create_program();
        shaders.iter().for_each(|&s| gl.attach_shader(p, s));
        gl.link_program(p);
        if !gl.link_status(p) {
            return Err(gl.program_log(p));
        }
        gl.use_program(p);
        let vao = gl.gen_vertex_arrays(1)[0];
        gl.bind_vertex_array(vao);
        let b = gl.gen_buffers(1)[0];
        gl.bind_buffer(BufferTarget::Array, b);
        let bytes: Vec<u8> = COVER.iter().flat_map(|f| f.to_le_bytes()).collect();
        gl.buffer_data(BufferTarget::Array, bytes.len(), Some(&bytes), Usage::StaticDraw);
        let pos = gl.attrib_location(p, "pos") as u32;
        gl.vertex_attrib_pointer(pos, Attrib { size: 2, ..Attrib::default() });
        gl.enable_vertex_attrib_array(pos, true);
        let color = gl.uniform_location(p, "color");
        gl.viewport(0, 0, SIZE as i32, SIZE as i32);
        gl.clear_color(self.clear);
        gl.clear(ClearMask { color: true, ..Default::default() });
        let [r, g, b, a] = self.mask;
        gl.color_mask(r, g, b, a);
        for (s, mode) in &self.draws {
            gl.uniform(color, UniformValue::Float(4, s));
            gl.enable(Capability::Blend, mode.is_some());
            if let Some(m) = mode {
                m.set(&mut gl);
            }
            gl.draw_arrays(device, DrawMode::Triangles, 0, 3).map_err(|e| e.0)?;
        }
        if let Some(e) = gl.error() {
            return Err(format!("GL error {e:?}"));
        }
        let pixels = gl.framebuffer.pixels(device).map_err(|e| e.0)?;
        Ok(match self.format {
            ColorFormat::Rgba32Float => pixels.chunks_exact(16).map(|p| core::array::from_fn(|c| f32::from_le_bytes(p[c * 4..c * 4 + 4].try_into().unwrap()))).collect(),
            ColorFormat::Rgba8Unorm => pixels.chunks_exact(4).map(|p| core::array::from_fn(|c| f32::from(p[c]) / 255.0)).collect(),
        })
    }

    /// "ok", or the first pixel that differs
    fn run(&self, tools: &Toolchain<'_>, device: &Device<'_>) -> Result<String, String> {
        let want = self.expected();
        let tolerance = match self.format {
            ColorFormat::Rgba8Unorm => 1.0 / 255.0,
            ColorFormat::Rgba32Float => 1e-6,
        };
        let pixels = self.drawn(tools, device)?;
        match pixels.iter().enumerate().find(|(_, p)| p.iter().zip(want).any(|(a, b)| (a - b).abs() > tolerance)) {
            None => Ok(format!("ok {want:?}")),
            Some((i, p)) => Err(format!("({}, {}) {p:?}, expected {want:?}", i as u32 % SIZE, i as u32 / SIZE)),
        }
    }
}

fn main() {
    use BlendEquation as E;
    use BlendFactor as F;
    let red = [1.0, 0.0, 0.0, 1.0];
    let half_blue = [0.0, 0.0, 0.5, 0.5];
    let over = Mode::of(F::One, F::OneMinusSrcAlpha);
    let case = |name, clear, draws| Case { name, format: ColorFormat::Rgba32Float, clear, draws, mask: [true; 4] };
    let mut cases = vec![
        case("replace", red, vec![(half_blue, None)]),
        case("premultiplied over", red, vec![(half_blue, Some(over))]),
        case("straight over", red, vec![([0.0, 0.0, 1.0, 0.5], Some(Mode::of(F::SrcAlpha, F::OneMinusSrcAlpha)))]),
        case("separate alpha", [0.2, 0.4, 0.6, 0.5], vec![([0.8, 0.6, 0.2, 0.25], Some(Mode { src: [F::SrcAlpha, F::One], dst: [F::OneMinusSrcAlpha, F::OneMinusSrcAlpha], ..over }))]),
        case("multiply", [0.5, 0.8, 1.0, 1.0], vec![([0.5, 0.5, 0.25, 1.0], Some(Mode::of(F::DstColor, F::Zero)))]),
        case("additive", [0.25, 0.5, 0.0, 0.5], vec![([0.5, 0.25, 0.75, 0.25], Some(Mode::of(F::One, F::One)))]),
        case("subtract", [0.25, 0.5, 0.75, 1.0], vec![([0.5, 0.75, 1.0, 1.0], Some(Mode { equation: [E::Subtract; 2], ..Mode::of(F::One, F::One) }))]),
        case("reverse subtract", [0.5, 0.75, 1.0, 1.0], vec![([0.25, 0.5, 0.25, 0.5], Some(Mode { equation: [E::ReverseSubtract; 2], ..Mode::of(F::One, F::One) }))]),
        case("min", [0.2, 0.8, 0.5, 0.6], vec![([0.6, 0.4, 0.5, 0.2], Some(Mode { equation: [E::Min; 2], ..Mode::of(F::Zero, F::Zero) }))]),
        case("max", [0.2, 0.8, 0.5, 0.6], vec![([0.6, 0.4, 0.5, 0.2], Some(Mode { equation: [E::Max; 2], ..Mode::of(F::Zero, F::Zero) }))]),
        case("constant", red, vec![([0.0, 1.0, 0.0, 1.0], Some(Mode { color: [0.25, 0.5, 0.75, 0.5], ..Mode::of(F::ConstantColor, F::OneMinusConstantAlpha) }))]),
        case("alpha saturate", [0.5, 0.5, 0.5, 0.75], vec![([1.0, 0.5, 0.25, 0.5], Some(Mode::of(F::SrcAlphaSaturate, F::One)))]),
        case("stacked", red, vec![(half_blue, Some(over)), ([0.0, 0.25, 0.0, 0.25], Some(over)), ([0.1, 0.1, 0.1, 0.5], Some(over))]),
        case("off again", red, vec![(half_blue, Some(over)), ([0.0, 1.0, 0.0, 1.0], None)]),
    ];
    cases.push(Case { mask: [true, false, true, false], ..case("color mask", red, vec![([0.0, 1.0, 1.0, 0.0], None)]) });
    cases.push(Case { mask: [false, true, true, true], ..case("color mask blended", red, vec![(half_blue, Some(over))]) });
    cases.push(Case { format: ColorFormat::Rgba8Unorm, ..case("rgba8 over", red, vec![(half_blue, Some(over)), ([0.0, 0.25, 0.0, 0.25], Some(over))]) });
    let tools = Toolchain::new();
    let xe = <detour::Xe as detour::Gpu>::open_first().unwrap();
    let device = Device::new(&xe);
    let mut failed = 0;
    for c in &cases {
        let r = c.run(&tools, &device);
        failed += usize::from(r.is_err());
        println!("{}: {}", c.name, r.unwrap_or_else(|e| format!("FAIL {e}")));
    }
    std::process::exit(i32::from(failed > 0));
}
