#![allow(dead_code, reason = "each example uses part of it")]

//! A GL case of examples/cases (shader.vert, shader.frag, resources) set up on detour's GL: what `compare` and
//! `bench_opengl` share. `resources` is glrun's: one line each,
//!   attribute NAME LOCATION N V...   three vertices of N floats
//!   uniform NAME LOCATION f|i V...   glUniform{1,2,3,4}{f,i}v (16 floats: glUniformMatrix4fv)

use std::fs;
use std::path::PathBuf;

use detour::{Attrib, BufferTarget, ColorFormat, Context, ShaderType, Toolchain, UniformValue, Usage};

/// The vertices a case draws
pub const VERTICES: i32 = 3;
const MAT4: usize = 16;

/// One line of `resources`
pub enum Resource {
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

/// One case: its directory
pub struct Case {
    pub dir: PathBuf,
}

impl Case {
    /// The case's `resources`, a line each
    pub fn resources(&self) -> Result<Vec<Resource>, String> {
        let text = fs::read_to_string(self.dir.join("resources")).map_err(|e| format!("resources: {e}"))?;
        text.lines().filter(|l| !l.trim().is_empty()).map(Resource::try_from).collect()
    }

    /// A context of `size`² with the case's program in use and its resources set, ready to draw
    pub fn context<'t>(&self, tools: &'t Toolchain<'t>, size: u32) -> Result<Context<'t>, String> {
        let read = |f: &str| fs::read_to_string(self.dir.join(f)).map_err(|e| format!("{f}: {e}"));
        let (vert, frag) = (read("shader.vert")?, read("shader.frag")?);
        let mut gl = Context::new(tools, size, size, ColorFormat::Rgba32Float).map_err(|e| e.0)?;
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
        for r in self.resources()? {
            match r {
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
        Ok(gl)
    }
}
