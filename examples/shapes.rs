//! Draws of one program whose vertices, counts, first vertex and scissor all differ, drawn with `Device::draw` into
//! one surface: they share one lowered shape (`Device::lowered`), each lands where it should, and a frame of them
//! takes the time printed. Then the surface is freed and a bigger one drawn into.
//!
//!   cargo run --release --example shapes [FRAMES]

use std::time::Instant;

use detour::{ColorFormat, ColorTarget, CullMode, Device, Draw, DrawPrimitive, Surface, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat};
use mlirpass::{GlslReader, GlslStage};

const VERT: &str = "#version 300 es\nlayout(location = 0) in vec2 pos;\nlayout(location = 1) in vec4 color;\nout vec4 c;\nvoid main() { c = color; gl_Position = vec4(pos, 0.0, 1.0); }\n";
const FRAG: &str = "#version 300 es\nprecision highp float;\nin vec4 c;\nout vec4 o;\nvoid main() { o = c; }\n";
const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";
/// Cells of a grid, each a draw
const GRID: u32 = 8;
const CELL: u32 = 32;
const FLOATS: usize = 6;

fn program(tools: &Toolchain<'_>) -> Vec<u8> {
    let mut body = tools.body();
    GlslReader::read(&mut body, VERT, GlslStage::Vertex).unwrap();
    GlslReader::read(&mut body, FRAG, GlslStage::Fragment).unwrap();
    let root = body.root;
    tools.pipeline(TO_SPIRV).and_then(|p| p.run(&mut body, root)).unwrap();
    tools.program(&tools.show(&body)).unwrap()
}

/// Cell `k`'s color
fn color(k: u32) -> [u8; 4] {
    [(k * 37 % 200 + 40) as u8, (k * 91 % 200 + 40) as u8, (k * 53 % 200 + 40) as u8, 255]
}

/// Two triangles over pixels x0..x1, y0..y1 of a `size` target, in one color
fn quad(out: &mut Vec<f32>, [x0, y0, x1, y1]: [u32; 4], size: u32, rgba: [u8; 4]) {
    let n = |p: u32| p as f32 / size as f32 * 2.0 - 1.0;
    let c = rgba.map(|v| v as f32 / 255.0);
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y0), (x1, y1), (x0, y1)] {
        out.extend([n(x), n(y)]);
        out.extend(c);
    }
}

/// Cell `k`'s draw: a quad over the whole target first (skipped by `first`) for every fourth, then the cell's quad
/// one to three times; the left half of the cell as the scissor for even ones
fn cell(k: u32, size: u32, floats: &mut Vec<f32>) -> (u32, u32, Option<[u32; 4]>) {
    floats.clear();
    let (x, y) = (k % GRID * CELL, k / GRID * CELL);
    let skipped = k % 4 == 3;
    if skipped {
        quad(floats, [0, 0, size, size], size, [255, 0, 0, 255]);
    }
    for _ in 0..k % 3 + 1 {
        quad(floats, [x, y, x + CELL, y + CELL], size, color(k));
    }
    let first = if skipped { 6 } else { 0 };
    let count = (floats.len() / FLOATS) as u32 - first;
    (first, count, k.is_multiple_of(2).then_some([x, y, CELL / 2, CELL]))
}

fn target(size: u32, load: TargetLoad<'_>) -> ColorTarget<'_> {
    ColorTarget { width: size, height: size, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Tile4, load }
}

/// One frame: the target cleared, a draw a cell
fn frame(device: &Device<'_>, program: &detour::Compiled<'_>, surface: Surface, size: u32) {
    let clear = [0u8; 4];
    let mut floats = Vec::new();
    let mut bytes = Vec::new();
    for k in 0..GRID * GRID {
        let (first, count, scissor) = cell(k, size, &mut floats);
        bytes.clear();
        bytes.extend(floats.iter().flat_map(|f| f.to_le_bytes()));
        let load = if k == 0 { TargetLoad::Fill(&clear, &[]) } else { TargetLoad::Keep(&[]) };
        let d = Draw {
            buffers: vec![VertexBuffer { bytes: &bytes, stride: (FLOATS * 4) as u32 }],
            elements: vec![
                VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 },
                VertexElement { buffer: 0, format: VertexFormat::Float32x4, offset: 8 },
            ],
            uniforms: Vec::new(),
            primitive: DrawPrimitive::Triangles,
            first,
            count,
            instances: 1,
            viewport: [0.0, 0.0, size as f32, size as f32],
            depth_range: [0.0, 1.0],
            scissor,
            cull: CullMode::None,
            front_ccw: true,
            blend: None,
            color_mask: [true; 4],
            target: target(size, load),
        };
        device.draw(program, &d, surface).unwrap();
    }
}

/// Each cell where its draw put it: the whole cell, or its left half (the right one clear)
fn check(pixels: &[u8], size: u32) -> Result<(), String> {
    let at = |x: u32, y: u32| -> [u8; 4] { pixels[((y * size + x) * 4) as usize..][..4].try_into().unwrap() };
    let mut wrong = Vec::new();
    for k in 0..GRID * GRID {
        let (x, y) = (k % GRID * CELL, k / GRID * CELL);
        let left = at(x + CELL / 4, y + CELL / 2);
        let right = at(x + 3 * CELL / 4, y + CELL / 2);
        let want_right = if k.is_multiple_of(2) { [0; 4] } else { color(k) };
        if left != color(k) || right != want_right {
            wrong.push(format!("cell {k}: {left:?} {right:?}, not {:?} {want_right:?}", color(k)));
        }
    }
    if wrong.is_empty() { Ok(()) } else { Err(format!("{} of {} cells: {}", wrong.len(), GRID * GRID, wrong.join("; "))) }
}

fn main() {
    let frames: u32 = std::env::args().nth(1).map_or(50, |n| n.parse().unwrap());
    let tools = Toolchain::new();
    let bytes = program(&tools);
    let compiled = tools.load(&bytes).unwrap();
    let xe = <detour::Xe as detour::Gpu>::open_first().unwrap();
    let device = Device::new(&xe);
    let size = GRID * CELL;
    let surface = device.surface(&target(size, TargetLoad::Keep(&[]))).unwrap();
    let started = Instant::now();
    frame(&device, &compiled, surface, size);
    let pixels = device.read(surface, (size * size * 4) as usize).unwrap();
    let first = started.elapsed();
    check(&pixels, size).unwrap();
    let started = Instant::now();
    for _ in 0..frames {
        frame(&device, &compiled, surface, size);
        device.read(surface, (size * size * 4) as usize).unwrap();
    }
    let each = started.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
    check(&device.read(surface, (size * size * 4) as usize).unwrap(), size).unwrap();
    println!("{} draws a frame: {} shape(s) lowered; the first frame {:.1} ms, then {each:.2} ms a frame ({frames} frames, read back)", GRID * GRID, device.lowered(), first.as_secs_f64() * 1e3);

    device.free(surface).unwrap();
    assert!(device.read(surface, 4).is_err(), "a freed surface read");
    let bigger = 2 * size;
    let surface = device.surface(&target(bigger, TargetLoad::Keep(&[]))).unwrap();
    frame(&device, &compiled, surface, bigger);
    let pixels = device.read(surface, (bigger * bigger * 4) as usize).unwrap();
    let inside: Vec<u8> = (0..size).flat_map(|y| pixels[(y * bigger * 4) as usize..][..(size * 4) as usize].to_vec()).collect();
    check(&inside, size).unwrap();
    println!("freed, then a {bigger}x{bigger} surface drawn: {} shape(s) lowered", device.lowered());
}
