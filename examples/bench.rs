//! Draw throughput: the triangle case drawn again and again into one framebuffer, timed per draw.
//!
//!   cargo run --release --example bench -- [DRAWS] [WIDTHxHEIGHT]...      (default 200 draws, 64x64 and 1920x1080)

use std::time::Instant;

use detour::ColorFormat;
use detour::{Attrib, BufferTarget, ClearMask, Context, Device, DrawMode, ShaderType, Toolchain, Usage};

const VERT: &str = include_str!("cases/triangle/shader.vert");
const FRAG: &str = include_str!("cases/triangle/shader.frag");
const POS: [f32; 6] = [-0.8, -0.7, 0.9, -0.4, -0.1, 0.85];
const COLOR: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
const WARMUP: usize = 5;

fn attribute(gl: &mut Context<'_>, p: u32, name: &str, size: u32, values: &[f32]) {
    let b = gl.gen_buffers(1)[0];
    gl.bind_buffer(BufferTarget::Array, b);
    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
    gl.buffer_data(BufferTarget::Array, bytes.len(), Some(&bytes), Usage::StaticDraw);
    let at = gl.attrib_location(p, name) as u32;
    gl.vertex_attrib_pointer(at, Attrib { size, ..Attrib::default() });
    gl.enable_vertex_attrib_array(at, true);
}

fn run(tools: &Toolchain<'_>, device: &Device<'_>, width: u32, height: u32, draws: usize) {
    let mut gl = Context::new(tools, width, height, ColorFormat::Rgba8Unorm).unwrap();
    let mut shaders = Vec::new();
    for (ty, src) in [(ShaderType::Vertex, VERT), (ShaderType::Fragment, FRAG)] {
        let s = gl.create_shader(ty);
        gl.shader_source(s, src);
        gl.compile_shader(s);
        assert!(gl.shader_compiled(s), "{}", gl.shader_log(s));
        shaders.push(s);
    }
    let p = gl.create_program();
    shaders.iter().for_each(|&s| gl.attach_shader(p, s));
    gl.link_program(p);
    assert!(gl.link_status(p), "{}", gl.program_log(p));
    gl.use_program(p);
    let vao = gl.gen_vertex_arrays(1)[0];
    gl.bind_vertex_array(vao);
    attribute(&mut gl, p, "pos", 2, &POS);
    attribute(&mut gl, p, "color", 3, &COLOR);
    gl.viewport(0, 0, width as i32, height as i32);
    gl.clear_color([0.0; 4]);
    gl.clear(ClearMask { color: true, ..Default::default() });
    for _ in 0..WARMUP {
        gl.draw_arrays(device, DrawMode::Triangles, 0, 3).unwrap();
    }
    gl.framebuffer.pixels(device).unwrap();
    let t = Instant::now();
    for _ in 0..draws {
        gl.draw_arrays(device, DrawMode::Triangles, 0, 3).unwrap();
    }
    gl.framebuffer.pixels(device).unwrap();
    let queued = t.elapsed().as_secs_f64() * 1e3;
    let mut times = Vec::with_capacity(draws);
    for _ in 0..draws {
        let t = Instant::now();
        gl.draw_arrays(device, DrawMode::Triangles, 0, 3).unwrap();
        gl.framebuffer.pixels(device).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let center = ((height / 2 * width + width / 2) * 4) as usize;
    let px = &gl.framebuffer.pixels(device).unwrap()[center..center + 4];
    assert!(px.iter().any(|&b| b != 0), "nothing drawn at the center");
    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2];
    let p95 = times[times.len() * 95 / 100];
    println!(
        "{width}x{height}: queued {:.4} ms/draw ({:.0} draws/s); draw + read back p50 {p50:.3} ms, p95 {p95:.3}; center {px:?}",
        queued / draws as f64,
        draws as f64 * 1e3 / queued
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let draws = args.next().map_or(200, |a| a.parse().expect("DRAWS"));
    let mut sizes: Vec<(u32, u32)> = args.map(|a| a.split_once('x').map(|(w, h)| (w.parse().unwrap(), h.parse().unwrap())).expect("WIDTHxHEIGHT")).collect();
    if sizes.is_empty() {
        sizes = vec![(64, 64), (1920, 1080)];
    }
    let tools = Toolchain::new();
    let xe = <detour::Xe as detour::Gpu>::open_first().unwrap();
    let device = Device::new(&xe);
    for (w, h) in sizes {
        run(&tools, &device, w, h, draws);
    }
}
