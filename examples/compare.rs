//! Each case under examples/cases (shader.vert, shader.frag, resources) drawn by detour's GL on Xe3 and by the local
//! Mesa (mlirpass's glrun), compared byte for byte. `resources` is glrun's: one line each,
//!   attribute NAME LOCATION N V...   three vertices of N floats
//!   uniform NAME LOCATION f|i V...   glUniform{1,2,3,4}{f,i}v (16 floats: glUniformMatrix4fv)
//!
//!   cargo run --example compare -- GLRUN [CASE...]      (DETOUR_FULL=1: the whole error, with the IR)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use detour::{Attrib, BufferTarget, ClearMask, Context, Device, DrawMode, ShaderType, Toolchain, UniformValue, Usage};
use detour::ColorFormat;

const SIZE: u32 = 64;
const PIXEL: usize = 16;
const VERTICES: i32 = 3;
const MAT4: usize = 16;

/// One line of `resources`
enum Resource {
    Attribute { name: String, size: u32, values: Vec<f32> },
    Float { name: String, values: Vec<f32> },
    Int { name: String, values: Vec<i32> },
}

impl TryFrom<&str> for Resource {
    type Error = String;

    fn try_from(line: &str) -> Result<Resource, String> {
        let w: Vec<&str> = line.split_whitespace().collect();
        let floats = |v: &[&str]| v.iter().map(|x| x.parse::<f32>().map_err(|e| format!("{line}: {e}"))).collect::<Result<Vec<_>, _>>();
        match w.as_slice() {
            ["attribute", name, _, n, v @ ..] => Ok(Resource::Attribute { name: (*name).into(), size: n.parse().map_err(|e| format!("{line}: {e}"))?, values: floats(v)? }),
            ["uniform", name, _, "f", v @ ..] => Ok(Resource::Float { name: (*name).into(), values: floats(v)? }),
            ["uniform", name, _, "i", v @ ..] => Ok(Resource::Int { name: (*name).into(), values: v.iter().map(|x| x.parse().map_err(|e| format!("{line}: {e}"))).collect::<Result<_, _>>()? }),
            _ => Err(format!("resources: {line}")),
        }
    }
}

/// A case's pixels from both, or why one could not draw
struct Case {
    dir: PathBuf,
}

impl Case {
    fn detour(&self, tools: &Toolchain<'_>, device: &Device) -> Result<Vec<u8>, String> {
        let read = |f: &str| fs::read_to_string(self.dir.join(f)).map_err(|e| format!("{f}: {e}"));
        let (vert, frag, res) = (read("shader.vert")?, read("shader.frag")?, read("resources")?);
        let mut gl = Context::new(tools, SIZE, SIZE, ColorFormat::Rgba32Float).map_err(|e| e.0)?;
        let mut shaders = Vec::new();
        for (ty, src) in [(ShaderType::Vertex, &vert), (ShaderType::Fragment, &frag)] {
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
        for line in res.lines().filter(|l| !l.trim().is_empty()) {
            match Resource::try_from(line)? {
                Resource::Attribute { name, size, values } => {
                    let b = gl.gen_buffers(1)[0];
                    gl.bind_buffer(BufferTarget::Array, b);
                    let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
                    gl.buffer_data(BufferTarget::Array, bytes.len(), Some(&bytes), Usage::StaticDraw);
                    let at = gl.attrib_location(p, &name);
                    if at < 0 {
                        continue;
                    }
                    gl.vertex_attrib_pointer(at as u32, Attrib { size, ..Attrib::default() });
                    gl.enable_vertex_attrib_array(at as u32, true);
                }
                Resource::Float { name, values } => {
                    let at = gl.uniform_location(p, &name);
                    let v = if values.len() == MAT4 { UniformValue::Matrix(4, 4, &values) } else { UniformValue::Float(values.len() as u32, &values) };
                    gl.uniform(at, v);
                }
                Resource::Int { name, values } => {
                    let at = gl.uniform_location(p, &name);
                    gl.uniform(at, UniformValue::Int(values.len() as u32, &values));
                }
            }
        }
        if let Some(e) = gl.error() {
            return Err(format!("GL error {e:?} setting the resources"));
        }
        gl.viewport(0, 0, SIZE as i32, SIZE as i32);
        gl.clear_color([0.0; 4]);
        gl.clear(ClearMask { color: true, ..Default::default() });
        gl.draw_arrays(device, DrawMode::Triangles, 0, VERTICES).map_err(|e| e.0)?;
        Ok(gl.framebuffer.pixels.clone())
    }

    fn mesa(&self, glrun: &Path, out: &Path) -> Result<Vec<u8>, String> {
        let o = Command::new(glrun).arg("glsl").arg(self.dir.join("shader.vert")).arg(self.dir.join("shader.frag")).args([SIZE.to_string(), SIZE.to_string()]).arg(out).arg(self.dir.join("resources")).output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).trim().to_owned());
        }
        fs::read(out).map_err(|e| e.to_string())
    }

    /// "ok", or what differs
    fn run(&self, tools: &Toolchain<'_>, device: &Device, glrun: &Path, out: &Path) -> Result<String, String> {
        let ours = self.detour(tools, device).map_err(|e| if std::env::var_os("DETOUR_FULL").is_some() { format!("detour: {e}") } else { format!("detour: {}", e.lines().next().unwrap_or("")) })?;
        let theirs = self.mesa(glrun, out).map_err(|e| format!("mesa: {e}"))?;
        let px = |b: &[u8], i: usize| b[i * PIXEL..(i + 1) * PIXEL].chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>();
        let covered = theirs.chunks(PIXEL).filter(|p| p.iter().any(|b| *b != 0)).count();
        let differing: Vec<usize> = (0..ours.len() / PIXEL).filter(|&i| ours[i * PIXEL..(i + 1) * PIXEL] != theirs[i * PIXEL..(i + 1) * PIXEL]).collect();
        match differing.first() {
            None => Ok(format!("ok ({covered} pixels)")),
            Some(&i) => Err(format!("{} of {covered} pixels differ; ({}, {}) detour {:?} mesa {:?}", differing.len(), i as u32 % SIZE, i as u32 / SIZE, px(&ours, i), px(&theirs, i))),
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let glrun = PathBuf::from(args.next().expect("the glrun binary"));
    let only: Vec<String> = args.collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("target/compare");
    fs::create_dir_all(&out).unwrap();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("examples/cases")).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let tools = Toolchain::new();
    let device = Device::open().unwrap();
    let mut failed = 0;
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        let r = Case { dir }.run(&tools, &device, &glrun, &out.join(format!("{name}.raw")));
        failed += usize::from(r.is_err());
        println!("{name}: {}", r.unwrap_or_else(|e| format!("FAIL {e}")));
    }
    std::process::exit(i32::from(failed > 0));
}
