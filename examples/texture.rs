//! Sampled textures drawn with `Device::draw`, checked against a CPU reference: a quad over the whole target samples
//! a texture (a surface) at the pixel centres of the coordinates given, nearest or linear, clamped, repeated or
//! mirrored, from RGBA8 and R8 textures in both tilings; a texture drawn into and then sampled in the same batch (render
//! to texture, nothing read back between) and drawn over after a draw sampled it; a part of a texture written
//! (`Device::write`) and sampled.
//!
//!   cargo run --release --example texture [FILTER...]      (cases whose name contains a filter)

use detour::{Isa, 
    ColorFormat, ColorTarget, Compiled, Device, Draw, Filter, Gpu, Sampled, Surface, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat,
    Wrap, Xe,
};
use mlirpass::{GlslReader, GlslStage};

const SHADERS: [&str; 2] = [
    "#version 300 es\nlayout(location = 0) in vec2 pos;\nlayout(location = 1) in vec2 uv;\nout vec2 t;\nvoid main() { t = uv; gl_Position = vec4(pos, 0.0, 1.0); }\n",
    "#version 300 es\nprecision highp float;\nlayout(binding = 2) uniform highp sampler2D tex;\nin vec2 t;\nout vec4 o;\nvoid main() { o = texture(tex, t); }\n",
];
const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";
const BINDING: u32 = 2;

/// A texture: its size, format, tiling and texels (rows)
struct Texture {
    width: u32,
    height: u32,
    format: ColorFormat,
    tiling: Tiling,
    texels: Vec<u8>,
}

impl Texture {
    fn new(width: u32, height: u32, format: ColorFormat, tiling: Tiling, seed: u32) -> Texture {
        let n = (width * height * format.bytes()) as usize;
        let texels = (0..n as u32).map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40_503)) >> 24) as u8).collect();
        Texture { width, height, format, tiling, texels }
    }

    fn target(&self) -> ColorTarget<'_> {
        ColorTarget { width: self.width, height: self.height, format: self.format, tiling: self.tiling, load: TargetLoad::Pixels(&self.texels, &[]) }
    }

    /// Texel (x, y) as RGBA in [0, 1] (R8: red, 0, 0, 1)
    fn texel(&self, x: u32, y: u32) -> [f32; 4] {
        let b = self.format.bytes() as usize;
        let at = (y * self.width + x) as usize * b;
        let t = &self.texels[at..at + b];
        match self.format {
            ColorFormat::R8Unorm => [f32::from(t[0]) / 255.0, 0.0, 0.0, 1.0],
            _ => core::array::from_fn(|c| f32::from(t[c]) / 255.0),
        }
    }
}

/// Coordinate `i` of a texture `n` texels across, as `wrap` takes it
fn wrapped(i: i64, n: u32, wrap: Wrap) -> u32 {
    let n = i64::from(n);
    (match wrap {
        Wrap::ClampToEdge => i.clamp(0, n - 1),
        Wrap::Repeat => i.rem_euclid(n),
        Wrap::MirroredRepeat => {
            let m = i.rem_euclid(2 * n);
            if m < n { m } else { 2 * n - 1 - m }
        }
    }) as u32
}

/// What sampling `t` at (u, v) gives
fn sample(t: &Texture, u: f32, v: f32, filter: Filter, wrap: Wrap) -> [f32; 4] {
    let (x, y) = (u * t.width as f32, v * t.height as f32);
    match filter {
        Filter::Nearest => t.texel(wrapped(x.floor() as i64, t.width, wrap), wrapped(y.floor() as i64, t.height, wrap)),
        Filter::Linear => {
            let (x, y) = (x - 0.5, y - 0.5);
            let (x0, y0) = (x.floor(), y.floor());
            let (a, b) = (x - x0, y - y0);
            let at = |dx: i64, dy: i64| t.texel(wrapped(x0 as i64 + dx, t.width, wrap), wrapped(y0 as i64 + dy, t.height, wrap));
            let [p00, p10, p01, p11] = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)];
            core::array::from_fn(|c| (p00[c] * (1.0 - a) + p10[c] * a) * (1.0 - b) + (p01[c] * (1.0 - a) + p11[c] * a) * b)
        }
    }
}

/// A sampling draw: a `width`×`height` RGBA8 target covered by coordinates `uv` (u0, v0, u1, v1)
struct Case {
    name: &'static str,
    texture: Texture,
    width: u32,
    height: u32,
    uv: [f32; 4],
    filter: Filter,
    wrap: Wrap,
    tolerance: u8,
}

impl Case {
    fn reference(&self) -> Vec<u8> {
        let [u0, v0, u1, v1] = self.uv;
        let mut out = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let u = u0 + (u1 - u0) * (x as f32 + 0.5) / self.width as f32;
                let v = v0 + (v1 - v0) * (y as f32 + 0.5) / self.height as f32;
                out.extend(sample(&self.texture, u, v, self.filter, self.wrap).map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8));
            }
        }
        out
    }
}

/// The quad's vertices: two triangles, position then coordinates
fn quad([u0, v0, u1, v1]: [f32; 4]) -> Vec<u8> {
    let corners = [(-1.0, -1.0, u0, v0), (1.0, -1.0, u1, v0), (-1.0, 1.0, u0, v1), (1.0, -1.0, u1, v0), (1.0, 1.0, u1, v1), (-1.0, 1.0, u0, v1)];
    corners.iter().flat_map(|&(x, y, u, v): &(f32, f32, f32, f32)| [x, y, u, v]).flat_map(f32::to_le_bytes).collect()
}

/// `texture` sampled over the whole of `surface` (a `width`×`height` RGBA8 target of `tiling`), recorded
#[allow(clippy::too_many_arguments, reason = "a draw's few knobs")]
fn draw(
    device: &Device<'_>,
    program: &Compiled,
    texture: Surface,
    filter: Filter,
    wrap: Wrap,
    uv: [f32; 4],
    surface: Surface,
    size: [u32; 2],
    tiling: Tiling,
) -> Result<(), String> {
    let vertices = quad(uv);
    let d = Draw {
        buffers: vec![VertexBuffer { bytes: &vertices, stride: 16 }],
        elements: vec![
            VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 },
            VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 8 },
        ],
        textures: vec![Sampled { binding: BINDING, surface: texture, filter, wrap }],
        count: 6,
        viewport: [0.0, 0.0, size[0] as f32, size[1] as f32],
        target: ColorTarget { width: size[0], height: size[1], format: ColorFormat::Rgba8Unorm, tiling, load: TargetLoad::Keep(&[]) },
        ..Draw::default()
    };
    device.draw(program, &d, surface).map_err(|e| e.0)
}

/// A new RGBA8 surface of `size` and `tiling`, zero
fn target(device: &Device<'_>, size: [u32; 2], tiling: Tiling) -> Result<Surface, String> {
    let t = ColorTarget { width: size[0], height: size[1], format: ColorFormat::Rgba8Unorm, tiling, load: TargetLoad::Keep(&[]) };
    device.surface(&t).map_err(|e| e.0)
}

/// `got` against `want`, each byte within `tolerance`
fn compare(got: &[u8], want: &[u8], width: u32, tolerance: u8) -> Result<(), String> {
    let wrong: Vec<String> = got
        .as_chunks::<4>()
        .0
        .iter()
        .zip(want.as_chunks::<4>().0)
        .enumerate()
        .filter(|(_, (g, w))| g.iter().zip(*w).any(|(a, b)| a.abs_diff(*b) > tolerance))
        .map(|(i, (g, w))| format!("({}, {}) {g:?} not {w:?}", i as u32 % width, i as u32 / width))
        .collect();
    match wrong.len() {
        0 => Ok(()),
        n => Err(format!("{n} pixels wrong: {}", wrong.into_iter().take(4).collect::<Vec<_>>().join("; "))),
    }
}

fn run(device: &Device<'_>, program: &Compiled, c: &Case) -> Result<(), String> {
    let texture = device.surface(&c.texture.target()).map_err(|e| e.0)?;
    let size = [c.width, c.height];
    let surface = target(device, size, Tiling::Tile4)?;
    draw(device, program, texture, c.filter, c.wrap, c.uv, surface, size, Tiling::Tile4)?;
    let got = device.read(surface, (c.width * c.height * 4) as usize).map_err(|e| e.0)?;
    device.free(texture).and(device.free(surface)).map_err(|e| e.0)?;
    compare(&got, &c.reference(), c.width, c.tolerance)
}

/// A texture drawn into surface `b`, `b` sampled into `c`, in one batch: `c` is the texture
fn render_to_texture(device: &Device<'_>, program: &Compiled) -> Result<(), String> {
    let t = Texture::new(16, 16, ColorFormat::Rgba8Unorm, Tiling::Linear, 7);
    let a = device.surface(&t.target()).map_err(|e| e.0)?;
    let size = [16, 16];
    let (b, c) = (target(device, size, Tiling::Tile4)?, target(device, size, Tiling::Linear)?);
    for _ in 0..3 {
        draw(device, program, a, Filter::Nearest, Wrap::ClampToEdge, [0.0, 0.0, 1.0, 1.0], b, size, Tiling::Tile4)?;
        draw(device, program, b, Filter::Nearest, Wrap::ClampToEdge, [0.0, 0.0, 1.0, 1.0], c, size, Tiling::Linear)?;
    }
    let got = device.read(c, 16 * 16 * 4).map_err(|e| e.0)?;
    [a, b, c].into_iter().try_for_each(|s| device.free(s)).map_err(|e| e.0)?;
    compare(&got, &t.texels, 16, 0)
}

/// Texture `a` sampled into `c`, then `a` drawn over (sampling `d`) in the same batch: `c` holds `a` as it was
fn sample_then_overwrite(device: &Device<'_>, program: &Compiled) -> Result<(), String> {
    let (ta, td) = (Texture::new(64, 64, ColorFormat::Rgba8Unorm, Tiling::Tile4, 21), Texture::new(64, 64, ColorFormat::Rgba8Unorm, Tiling::Tile4, 22));
    let (a, d) = (device.surface(&ta.target()).map_err(|e| e.0)?, device.surface(&td.target()).map_err(|e| e.0)?);
    let size = [64, 64];
    let c = target(device, size, Tiling::Linear)?;
    let whole = [0.0, 0.0, 1.0, 1.0];
    draw(device, program, a, Filter::Nearest, Wrap::ClampToEdge, whole, c, size, Tiling::Linear)?;
    draw(device, program, d, Filter::Nearest, Wrap::ClampToEdge, whole, a, size, Tiling::Tile4)?;
    let got = device.read(c, 64 * 64 * 4).map_err(|e| e.0)?;
    let overwritten = device.read(a, 64 * 64 * 4).map_err(|e| e.0)?;
    [a, c, d].into_iter().try_for_each(|s| device.free(s)).map_err(|e| e.0)?;
    compare(&got, &ta.texels, 64, 0).and_then(|()| compare(&overwritten, &td.texels, 64, 0))
}

/// A texture's part written, then sampled: the part new, the rest as it was
fn write_part(device: &Device<'_>, program: &Compiled) -> Result<(), String> {
    let mut t = Texture::new(32, 16, ColorFormat::R8Unorm, Tiling::Tile4, 3);
    let texture = device.surface(&t.target()).map_err(|e| e.0)?;
    let size = [32, 16];
    let surface = target(device, size, Tiling::Linear)?;
    draw(device, program, texture, Filter::Nearest, Wrap::ClampToEdge, [0.0, 0.0, 1.0, 1.0], surface, size, Tiling::Linear)?;
    let [x, y, w, h] = [5, 3, 19, 9];
    let part: Vec<u8> = (0..w * h).map(|i| (i * 7 + 1) as u8).collect();
    device.write(texture, [x, y, w, h], &part).map_err(|e| e.0)?;
    for r in 0..h {
        let at = ((y + r) * t.width + x) as usize;
        t.texels[at..at + w as usize].copy_from_slice(&part[(r * w) as usize..][..w as usize]);
    }
    draw(device, program, texture, Filter::Nearest, Wrap::ClampToEdge, [0.0, 0.0, 1.0, 1.0], surface, size, Tiling::Linear)?;
    let got = device.read(surface, 32 * 16 * 4).map_err(|e| e.0)?;
    device.free(texture).and(device.free(surface)).map_err(|e| e.0)?;
    let case = Case { name: "", texture: t, width: 32, height: 16, uv: [0.0, 0.0, 1.0, 1.0], filter: Filter::Nearest, wrap: Wrap::ClampToEdge, tolerance: 0 };
    compare(&got, &case.reference(), 32, 0)
}

fn cases() -> Vec<Case> {
    let rgba = |w, h, tiling, seed| Texture::new(w, h, ColorFormat::Rgba8Unorm, tiling, seed);
    let whole = [0.0, 0.0, 1.0, 1.0];
    let case = |name, texture, width, height, uv, filter, wrap, tolerance| Case { name, texture, width, height, uv, filter, wrap, tolerance };
    vec![
        case("nearest 1:1, linear texture", rgba(8, 8, Tiling::Linear, 1), 8, 8, whole, Filter::Nearest, Wrap::ClampToEdge, 0),
        case("nearest 1:1, Tile4 texture", rgba(64, 48, Tiling::Tile4, 2), 64, 48, whole, Filter::Nearest, Wrap::ClampToEdge, 0),
        case("linear 1:1", rgba(16, 16, Tiling::Tile4, 3), 16, 16, whole, Filter::Linear, Wrap::ClampToEdge, 1),
        case("nearest magnified", rgba(4, 4, Tiling::Linear, 4), 16, 16, whole, Filter::Nearest, Wrap::ClampToEdge, 0),
        case("linear magnified", rgba(4, 4, Tiling::Linear, 5), 16, 16, whole, Filter::Linear, Wrap::ClampToEdge, 2),
        case("linear minified", rgba(32, 32, Tiling::Tile4, 6), 16, 16, whole, Filter::Linear, Wrap::ClampToEdge, 2),
        case("repeat", rgba(4, 4, Tiling::Linear, 7), 16, 16, [0.0, 0.0, 2.0, 2.0], Filter::Nearest, Wrap::Repeat, 0),
        case("mirrored repeat", rgba(4, 4, Tiling::Linear, 8), 16, 16, [0.0, 0.0, 2.0, 2.0], Filter::Nearest, Wrap::MirroredRepeat, 0),
        case("clamp to edge", rgba(4, 4, Tiling::Linear, 9), 16, 16, [-0.5, -0.5, 1.5, 1.5], Filter::Nearest, Wrap::ClampToEdge, 0),
        case("linear repeat", rgba(4, 4, Tiling::Tile4, 10), 16, 16, [-1.0, 0.0, 1.0, 2.0], Filter::Linear, Wrap::Repeat, 2),
        case("R8, nearest", Texture::new(16, 8, ColorFormat::R8Unorm, Tiling::Linear, 11), 16, 8, whole, Filter::Nearest, Wrap::ClampToEdge, 0),
        case("R8, linear magnified", Texture::new(8, 8, ColorFormat::R8Unorm, Tiling::Tile4, 12), 32, 32, whole, Filter::Linear, Wrap::ClampToEdge, 2),
    ]
}

fn main() {
    let only: Vec<String> = std::env::args().skip(1).collect();
    let picked = |name: &str| only.is_empty() || only.iter().any(|f| name.contains(f.as_str()));
    let tools = Toolchain::new(Isa::Xe3);
    let mut body = tools.body();
    GlslReader::read(&mut body, SHADERS[0], GlslStage::Vertex).unwrap();
    GlslReader::read(&mut body, SHADERS[1], GlslStage::Fragment).unwrap();
    let root = body.root;
    tools.pipeline(TO_SPIRV).and_then(|p| p.run(&mut body, root)).unwrap();
    let program = tools.compile(body).unwrap_or_else(|e| panic!("{}", e.0));
    let xe = Xe::open_first().unwrap();
    let device = Device::new(&xe);
    let mut failed = 0;
    let mut report = |name: &str, r: Result<(), String>| {
        failed += usize::from(r.is_err());
        println!("{}   {name}{}", if r.is_ok() { "ok" } else { "FAIL" }, r.err().map(|e| format!(": {e}")).unwrap_or_default());
    };
    for c in cases().iter().filter(|c| picked(c.name)) {
        report(c.name, run(&device, &program, c));
    }
    if picked("render to texture") {
        report("render to texture", render_to_texture(&device, &program));
    }
    if picked("sample then overwrite") {
        report("sample then overwrite", sample_then_overwrite(&device, &program));
    }
    if picked("write a part") {
        report("write a part", write_part(&device, &program));
    }
    println!("{} lowered shape(s)", device.lowered());
    std::process::exit(i32::from(failed > 0));
}
