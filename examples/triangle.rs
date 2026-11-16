//! One triangle through detour's GL on Xe3 and through the local Mesa (mlirpass's glrun), compared byte for byte.
//!
//!   cargo run --example triangle -- GLRUN

use std::fs;
use std::path::Path;
use std::process::Command;

use detour::{Attrib, BufferTarget, ClearMask, Context, Device, DrawMode, ShaderType, SurfaceFormat, Toolchain, Usage};

const SIZE: u32 = 64;
const PIXEL: usize = 16;
const VERT: &str = "#version 300 es
in vec2 pos;
in vec3 color;
out vec3 v;
void main() { v = color; gl_Position = vec4(pos, 0.0, 1.0); }
";
const FRAG: &str = "#version 300 es
precision highp float;
in vec3 v;
out vec4 o;
void main() { o = vec4(v, 1.0); }
";
const POS: [f32; 6] = [-0.8, -0.7, 0.9, -0.4, -0.1, 0.85];
const COLOR: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() {
    let glrun = std::env::args().nth(1).expect("the glrun binary");
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/triangle");
    fs::create_dir_all(&out).unwrap();

    let tools = Toolchain::new().unwrap();
    let device = Device::open().unwrap();
    let mut gl = Context::new(&tools, SIZE, SIZE, SurfaceFormat::R32G32B32A32Float).unwrap();
    let shaders = [(ShaderType::Vertex, VERT), (ShaderType::Fragment, FRAG)].map(|(ty, src)| {
        let s = gl.create_shader(ty);
        gl.shader_source(s, src);
        gl.compile_shader(s);
        assert!(gl.shader_compiled(s), "{}", gl.shader_log(s));
        s
    });
    let p = gl.create_program();
    shaders.iter().for_each(|&s| gl.attach_shader(p, s));
    gl.link_program(p);
    assert!(gl.link_status(p), "{}", gl.program_log(p));
    gl.use_program(p);
    let vao = gl.gen_vertex_arrays(1)[0];
    gl.bind_vertex_array(vao);
    let bufs = gl.gen_buffers(2);
    for (&b, (name, data, size)) in bufs.iter().zip([("pos", &POS[..], 2), ("color", &COLOR[..], 3)]) {
        gl.bind_buffer(BufferTarget::Array, b);
        let bytes = floats(data);
        gl.buffer_data(BufferTarget::Array, bytes.len(), Some(&bytes), Usage::StaticDraw);
        let at = gl.attrib_location(p, name) as u32;
        gl.vertex_attrib_pointer(at, Attrib { size, ..Attrib::default() });
        gl.enable_vertex_attrib_array(at, true);
    }
    gl.viewport(0, 0, SIZE as i32, SIZE as i32);
    gl.clear_color([0.0; 4]);
    gl.clear(ClearMask { color: true, ..Default::default() }).unwrap();
    gl.draw_arrays(&device, DrawMode::Triangles, 0, 3).unwrap();
    assert_eq!(gl.error(), None);
    let ours = gl.framebuffer.pixels.clone();
    fs::write(out.join("detour.raw"), &ours).unwrap();

    let (vert, frag, res, mesa) = (out.join("tri.vert"), out.join("tri.frag"), out.join("tri.res"), out.join("mesa.raw"));
    fs::write(&vert, VERT).unwrap();
    fs::write(&frag, FRAG).unwrap();
    let list = |v: &[f32]| v.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>().join(" ");
    fs::write(&res, format!("attribute pos 0 2 {}\nattribute color 1 3 {}\n", list(&POS), list(&COLOR))).unwrap();
    let o = Command::new(&glrun).arg("glsl").arg(&vert).arg(&frag).args([SIZE.to_string(), SIZE.to_string()]).arg(&mesa).arg(&res).output().unwrap();
    assert!(o.status.success(), "glrun: {}", String::from_utf8_lossy(&o.stderr));
    let theirs = fs::read(&mesa).unwrap();

    let covered = |px: &[u8]| px.chunks(PIXEL).filter(|p| p.iter().any(|b| *b != 0)).count();
    let differing: Vec<usize> = (0..ours.len() / PIXEL).filter(|&i| ours[i * PIXEL..(i + 1) * PIXEL] != theirs[i * PIXEL..(i + 1) * PIXEL]).collect();
    println!("covered: detour {} mesa {}; differing pixels: {}", covered(&ours), covered(&theirs), differing.len());
    for &i in differing.iter().take(8) {
        let px = |b: &[u8]| b[i * PIXEL..(i + 1) * PIXEL].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
        println!("  ({}, {}) detour {:?} mesa {:?}", i as u32 % SIZE, i as u32 / SIZE, px(&ours), px(&theirs));
    }
}
