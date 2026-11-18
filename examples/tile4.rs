//! Tile4's arrangement inside a tile, measured: a gradient drawn into a linear and a Tile4 target, the Tile4 memory
//! read back raw, and every arrangement of the 8 address bits above a 16-byte run tried against the linear drawing.
//!
//!   cargo run --release --example tile4

use detour::{Attrib, BufferTarget, ClearMask, ColorFormat, Context, Device, DrawMode, ShaderType, Tiling, Toolchain, Usage};

const VERT: &str = "#version 300 es\nin vec2 pos;\nout vec3 v;\nvoid main() { v = vec3(pos * 0.5 + 0.5, 0.5); gl_Position = vec4(pos, 0.0, 1.0); }\n";
const FRAG: &str = "#version 300 es\nprecision highp float;\nin vec3 v;\nout vec4 o;\nvoid main() { o = vec4(v, 1.0); }\n";
const QUAD: [f32; 12] = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
const W: u32 = 256;
const H: u32 = 64;

fn draw(tools: &Toolchain<'_>, device: &Device<'_>, tiling: Tiling) -> (Vec<u8>, Vec<u8>) {
    let mut gl = Context::new(tools, W, H, ColorFormat::Rgba8Unorm).unwrap();
    gl.framebuffer.tiling = tiling;
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
    let b = gl.gen_buffers(1)[0];
    gl.bind_buffer(BufferTarget::Array, b);
    let bytes: Vec<u8> = QUAD.iter().flat_map(|f| f.to_le_bytes()).collect();
    gl.buffer_data(BufferTarget::Array, bytes.len(), Some(&bytes), Usage::StaticDraw);
    let at = gl.attrib_location(p, "pos") as u32;
    gl.vertex_attrib_pointer(at, Attrib { size: 2, ..Attrib::default() });
    gl.enable_vertex_attrib_array(at, true);
    gl.viewport(0, 0, W as i32, H as i32);
    gl.clear(ClearMask { color: true, ..Default::default() });
    gl.draw_arrays(device, DrawMode::Triangles, 0, 6).unwrap();
    let pixels = gl.framebuffer.pixels(device).unwrap().to_vec();
    let raw = device.raw(gl.framebuffer.surface.unwrap()).unwrap();
    (pixels, raw)
}

/// Bits 4..12 of a byte's offset in its tile, from x bits 4..7 (sources 0..3) and y bits 0..5 (sources 3..8)
fn within(bits: &[usize; 8], x: u32, y: u32) -> usize {
    let src = [x >> 4 & 1, x >> 5 & 1, x >> 6 & 1, y & 1, y >> 1 & 1, y >> 2 & 1, y >> 3 & 1, y >> 4 & 1];
    (x & 15) as usize | bits.iter().enumerate().map(|(i, &b)| (src[b] as usize) << (4 + i)).sum::<usize>()
}

fn permutations(n: usize, f: &mut impl FnMut(&[usize; 8])) {
    fn go(at: usize, p: &mut [usize; 8], used: &mut [bool; 8], n: usize, f: &mut impl FnMut(&[usize; 8])) {
        if at == n {
            return f(p);
        }
        for b in 0..n {
            if !used[b] {
                used[b] = true;
                p[at] = b;
                go(at + 1, p, used, n, f);
                used[b] = false;
            }
        }
    }
    go(0, &mut [0; 8], &mut [false; 8], n, f);
}

fn main() {
    let tools = Toolchain::new();
    let xe = <detour::Xe as detour::Gpu>::open_first().unwrap();
    let device = Device::new(&xe);
    let (linear, _) = draw(&tools, &device, Tiling::Linear);
    let (tiled_pixels, raw) = draw(&tools, &device, Tiling::Tile4);
    println!("Tiling::Tile4 as detour lays it out: {}", if tiled_pixels == linear { "matches the linear drawing" } else { "DIFFERS" });
    let row = W * 4;
    let pitch = Tiling::Tile4.pitch(row) as usize;
    let mut found = 0;
    permutations(8, &mut |bits| {
        let ok = (0..H).all(|y| {
            (0..row).step_by(16).all(|x| {
                let tile = (y / 32) as usize * (pitch / 128) + (x / 128) as usize;
                let at = tile * 4096 + within(bits, x % 128, y % 32);
                raw[at..at + 16] == linear[(y * row + x) as usize..(y * row + x) as usize + 16]
            })
        });
        if ok {
            found += 1;
            let names = ["x4", "x5", "x6", "y0", "y1", "y2", "y3", "y4"];
            println!("offset bits 4..12 = {:?}", bits.iter().map(|&b| names[b]).collect::<Vec<_>>());
        }
    });
    println!("{found} arrangement(s) match");
}
