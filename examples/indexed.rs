//! Indexed draws and colors in four bytes, against the plain draw: quads (overlapping, each later one over the last)
//! drawn as six vertices each with Float32x4 colors make the reference; the same quads with Unorm8x4 colors, as four
//! vertices and six u16 or u32 indices each, from an index other than the first, and indexed and plain draws taking
//! turns in one batch must give it byte for byte.
//!
//!   cargo run --release --example indexed

#[path = "support/glsl.rs"]
mod glsl;

use detour::{Isa, ColorFormat, ColorTarget, Compiled, Device, Draw, IndexBuffer, IndexFormat, Surface, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat, Xe};

const SHADERS: [&str; 2] = [
    "#version 300 es\nlayout(location = 0) in vec2 pos;\nlayout(location = 1) in vec4 color;\nout vec4 c;\nvoid main() { c = color; gl_Position = vec4(pos, 0.0, 1.0); }\n",
    "#version 300 es\nprecision highp float;\nin vec4 c;\nout vec4 o;\nvoid main() { o = c; }\n",
];
const SIZE: u32 = 256;
const QUADS: u32 = 3000;

/// Quad `k`: its corners in pixels and its color
fn quad(k: u32) -> ([f32; 4], [u8; 4]) {
    let h = |m: u32| k.wrapping_mul(m).wrapping_add(0x9e37_79b9) >> 7;
    let (x, y) = ((h(2_654_435_761) % (SIZE - 24)) as f32 + 0.25, (h(40_503) % (SIZE - 24)) as f32 + 0.5);
    let (w, hh) = ((h(97) % 20 + 2) as f32, (h(131) % 20 + 2) as f32);
    let c = [(h(3) & 255) as u8, (h(5) & 255) as u8, (h(7) & 255) as u8, (h(11) & 255) as u8];
    ([x, y, x + w, y + hh], c)
}

fn ndc(x: f32, y: f32) -> [f32; 2] {
    [x / SIZE as f32 * 2.0 - 1.0, y / SIZE as f32 * 2.0 - 1.0]
}

/// The quads' corners (x0 y0, x1 y0, x0 y1, x1 y1), each with its color as `colors` says
fn corners(quads: &[([f32; 4], [u8; 4])], bytes_color: bool) -> Vec<Vec<u8>> {
    quads
        .iter()
        .map(|([x0, y0, x1, y1], c)| {
            let mut v = Vec::new();
            for (x, y) in [(*x0, *y0), (*x1, *y0), (*x0, *y1), (*x1, *y1)] {
                v.extend(ndc(x, y).iter().flat_map(|f| f.to_le_bytes()));
                match bytes_color {
                    true => v.extend(c),
                    false => v.extend(c.iter().flat_map(|b| (f32::from(*b) / 255.0).to_le_bytes())),
                }
            }
            v
        })
        .collect()
}

/// A vertex's bytes: position, then its color as floats or bytes
fn stride(bytes_color: bool) -> u32 {
    if bytes_color { 12 } else { 24 }
}

fn elements(bytes_color: bool) -> Vec<VertexElement> {
    let color = if bytes_color { VertexFormat::Unorm8x4 } else { VertexFormat::Float32x4 };
    vec![VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 }, VertexElement { buffer: 0, format: color, offset: 8 }]
}

/// Six vertices a quad (two triangles), from its four corners
fn plain(corners: &[Vec<u8>], stride: u32) -> Vec<u8> {
    let s = stride as usize;
    corners.iter().flat_map(|q| [0, 1, 2, 1, 3, 2].into_iter().flat_map(move |i| q[i * s..(i + 1) * s].to_vec())).collect()
}

/// Six indices a quad into its four corners, in `format`
fn indices(quads: u32, format: IndexFormat) -> Vec<u8> {
    let all = (0..quads).flat_map(|q| [0, 1, 2, 1, 3, 2].map(|i| 4 * q + i));
    match format {
        IndexFormat::U16 => all.flat_map(|i| (i as u16).to_le_bytes()).collect(),
        IndexFormat::U32 => all.flat_map(u32::to_le_bytes).collect(),
    }
}

fn target() -> ColorTarget<'static> {
    ColorTarget { width: SIZE, height: SIZE, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Tile4, load: TargetLoad::Keep(&[]) }
}

struct Run<'a, 't> {
    device: &'a Device<'t>,
    program: &'a Compiled,
}

impl Run<'_, '_> {
    fn draw(&self, s: Surface, vertices: &[u8], bytes_color: bool, indices: Option<IndexBuffer<'_>>, first: u32, count: u32) -> Result<(), String> {
        let d = Draw {
            buffers: vec![VertexBuffer { bytes: vertices, stride: stride(bytes_color) }],
            elements: elements(bytes_color),
            indices,
            first,
            count,
            viewport: [0.0, 0.0, SIZE as f32, SIZE as f32],
            target: target(),
            ..Draw::default()
        };
        self.device.draw(self.program, &d, s).map_err(|e| e.0)
    }

    /// A cleared surface, `draws` recorded into it, its pixels
    fn image(&self, draws: impl FnOnce(Surface) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let zero = [0u8; 4];
        let t = ColorTarget { load: TargetLoad::Fill(&zero, &[]), ..target() };
        let s = self.device.surface(&t).map_err(|e| e.0)?;
        draws(s)?;
        let pixels = self.device.read(s, (SIZE * SIZE * 4) as usize).map_err(|e| e.0)?;
        self.device.free(s).map_err(|e| e.0)?;
        Ok(pixels)
    }
}

/// A case: its name, its pixels, and the pixels it must give
type Case<'a> = (&'static str, Result<Vec<u8>, String>, &'a [u8]);

fn same(got: &[u8], want: &[u8]) -> Result<(), String> {
    let wrong = got.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0).filter(|(a, b)| a != b).count();
    if wrong == 0 { Ok(()) } else { Err(format!("{wrong} pixels differ from the plain draw")) }
}

fn main() {
    let tools = Toolchain::new(Isa::Xe3);
    let program = tools.compile(&glsl::spirv(SHADERS[0], SHADERS[1]).unwrap()).unwrap_or_else(|e| panic!("{}", e.0));
    let xe = Xe::open_first().unwrap();
    let device = Device::new(&xe);
    let run = Run { device: &device, program: &program };
    let quads: Vec<_> = (0..QUADS).map(quad).collect();
    let (floats, bytes) = (corners(&quads, false), corners(&quads, true));
    let (plain_floats, plain_bytes) = (plain(&floats, 24), plain(&bytes, 12));
    let (all_floats, all_bytes): (Vec<u8>, Vec<u8>) = (floats.concat(), bytes.concat());
    let (u16s, u32s) = (indices(QUADS, IndexFormat::U16), indices(QUADS, IndexFormat::U32));
    let ib16 = IndexBuffer { bytes: &u16s, format: IndexFormat::U16 };
    let ib32 = IndexBuffer { bytes: &u32s, format: IndexFormat::U32 };
    let n = 6 * QUADS;
    let reference = run.image(|s| run.draw(s, &plain_floats, false, None, 0, n)).unwrap();
    let half = QUADS / 2;
    let late = run.image(|s| run.draw(s, &plain_floats, false, None, 6 * half, 6 * (QUADS - half))).unwrap();
    let cases: Vec<Case<'_>> = vec![
        ("Unorm8x4 colors", run.image(|s| run.draw(s, &plain_bytes, true, None, 0, n)), &reference),
        ("u16 indices, Unorm8x4 colors", run.image(|s| run.draw(s, &all_bytes, true, Some(ib16), 0, n)), &reference),
        ("u32 indices, Float32x4 colors", run.image(|s| run.draw(s, &all_floats, false, Some(ib32), 0, n)), &reference),
        ("u16 indices from the middle", run.image(|s| run.draw(s, &all_bytes, true, Some(ib16), 6 * half, 6 * (QUADS - half))), &late),
        (
            "indexed and plain draws taking turns",
            run.image(|s| {
                for k in 0..10 {
                    let (first, count) = (k * QUADS / 10 * 6, QUADS / 10 * 6);
                    match k % 3 {
                        0 => run.draw(s, &plain_floats, false, None, first, count)?,
                        1 => run.draw(s, &all_bytes, true, Some(ib16), first, count)?,
                        _ => run.draw(s, &all_floats, false, Some(ib32), first, count)?,
                    }
                }
                Ok(())
            }),
            &reference,
        ),
    ];
    let mut failed = 0;
    for (name, got, want) in cases {
        let r = got.and_then(|g| same(&g, want));
        failed += usize::from(r.is_err());
        println!("{}   {name}{}", if r.is_ok() { "ok" } else { "FAIL" }, r.err().map(|e| format!(": {e}")).unwrap_or_default());
    }
    let bad = run.image(|s| run.draw(s, &all_bytes, true, Some(IndexBuffer { bytes: &u16s[..10], format: IndexFormat::U16 }), 0, 6));
    println!("{}   indices past the buffer refused", if bad.is_err() { "ok" } else { "FAIL" });
    failed += usize::from(bad.is_ok());
    println!("{} lowered shape(s)", device.lowered());
    std::process::exit(i32::from(failed > 0));
}
