use detour::{Context, ShaderType, Toolchain};
use detour::ColorFormat;

const VERT: &str = "#version 300 es
in vec2 pos;
in vec3 color;
out vec3 v;
layout(std140) uniform Frame { vec4 offset; float scale; };
void main() { v = color; gl_Position = vec4(pos * scale, 0.0, 1.0) + offset; }
";
const FRAG: &str = "#version 300 es
precision mediump float;
in vec3 v;
out vec4 o;
void main() { o = vec4(v, 1.0); }
";
const LOOSE: &str = "#version 300 es
in vec2 pos;
out vec3 v;
uniform float scale;
void main() { v = vec3(scale); gl_Position = vec4(pos * scale, 0.0, 1.0); }
";

fn main() {
    let tools = Toolchain::new();
    let mut gl = Context::new(&tools, 64, 64, ColorFormat::Rgba8Unorm).unwrap();
    for vert in [VERT, LOOSE] {
        let shaders = [(ShaderType::Vertex, vert), (ShaderType::Fragment, FRAG)].map(|(ty, src)| {
            let s = gl.create_shader(ty);
            gl.shader_source(s, src);
            gl.compile_shader(s);
            assert!(gl.shader_compiled(s), "{}", gl.shader_log(s));
            s
        });
        let p = gl.create_program();
        shaders.iter().for_each(|&s| gl.attach_shader(p, s));
        gl.link_program(p);
        if !gl.link_status(p) {
            println!("link failed: {}", gl.program_log(p).lines().next().unwrap_or(""));
            continue;
        }
        println!("pos {} color {} scale {}", gl.attrib_location(p, "pos"), gl.attrib_location(p, "color"), gl.uniform_location(p, "scale"));
    }
    println!("error {:?}", gl.error());
}
