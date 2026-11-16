//! Each script under examples/glcalls runs on detour's GL and on the local Mesa (examples/support/glcalls.c);
//! glGetError and what each call returns must be the same, line by line. A script line is a call and its arguments:
//! GL enum names (GL_ optional), integers, `data` (client memory), `null` or `offset:N` (into a bound buffer).
//!
//!   cargo run --example glcalls -- GLCALLS [SCRIPT...]

use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use detour::{
    AttachPoint, BufferTarget, ColorBuffer, Context, Dims, FramebufferTarget, Gl, GlError, ImageTarget, InternalFormat, PixelFormat, PixelStore, PixelType, Region, TexParam,
    TexelSource, TextureTarget, Toolchain, Usage,
};
use mlirpass::ColorFormat;

/// Client data every `data` argument points at (examples/support/glcalls.c's DATA)
const DATA: usize = 1 << 20;
const SIZE: u32 = 64;

/// One argument of a call
#[derive(Clone, Copy, Debug)]
enum Arg {
    Num(i64),
    Data,
    Null,
    Offset(usize),
}

impl TryFrom<&str> for Arg {
    type Error = String;

    fn try_from(t: &str) -> Result<Arg, String> {
        match t {
            "data" => Ok(Arg::Data),
            "null" => Ok(Arg::Null),
            t if t.starts_with("offset:") => t["offset:".len()..].parse().map(Arg::Offset).map_err(|e| format!("{t}: {e}")),
            t => t.parse::<i64>().ok().or_else(|| Gl::value(t).map(i64::from)).map(Arg::Num).ok_or_else(|| format!("{t}: not a number or a GL enum")),
        }
    }
}

impl Arg {
    /// The argument for glcalls.c
    fn wire(self) -> String {
        match self {
            Arg::Num(v) => v.to_string(),
            Arg::Data => "D".into(),
            Arg::Null => "N".into(),
            Arg::Offset(o) => format!("O{o}"),
        }
    }
}

/// A call's arguments, read as the call's types
struct Args<'a> {
    v: &'a [Arg],
    data: &'a [u8],
}

impl Args<'_> {
    fn n(&self, i: usize) -> i64 {
        match self.v.get(i) {
            Some(Arg::Num(v)) => *v,
            _ => 0,
        }
    }

    fn i(&self, i: usize) -> i32 {
        self.n(i) as i32
    }

    fn u(&self, i: usize) -> u32 {
        self.n(i) as u32
    }

    fn texels(&self, i: usize) -> TexelSource<'_> {
        match self.v.get(i) {
            Some(Arg::Data) => TexelSource::Client(self.data),
            Some(Arg::Offset(o)) => TexelSource::Offset(*o),
            _ => TexelSource::None,
        }
    }
}

/// One call on detour: its results, after glGetError
struct Detour<'c, 't> {
    gl: &'c mut Context<'t>,
}

impl Detour<'_, '_> {
    /// `f`'s GL enum argument, or the error a GL entry point records for it
    fn call(&mut self, f: &str, a: &Args<'_>) -> Result<String, String> {
        let gl = &mut *self.gl;
        macro_rules! enm {
            ($t:ty, $v:expr) => {
                match <$t>::from_gl($v) {
                    Ok(x) => x,
                    Err(e) => {
                        gl.fail(e);
                        return Ok(String::new());
                    }
                }
            };
        }
        let names = |v: Vec<u32>| v.iter().map(|n| format!(" {n}")).collect::<String>();
        Ok(match f {
            "gen_textures" => names(gl.gen_textures(a.n(0) as usize)),
            "gen_buffers" => names(gl.gen_buffers(a.n(0) as usize)),
            "gen_framebuffers" => names(gl.gen_framebuffers(a.n(0) as usize)),
            "gen_renderbuffers" => names(gl.gen_renderbuffers(a.n(0) as usize)),
            "bind_texture" => {
                gl.bind_texture(enm!(TextureTarget, a.u(0)), a.u(1));
                String::new()
            }
            "delete_textures" => {
                gl.delete_textures(&[a.u(0)]);
                String::new()
            }
            "is_texture" => format!(" {}", u8::from(gl.is_texture(a.u(0)))),
            "active_texture" => {
                match a.u(0).checked_sub(Gl::TEXTURE0) {
                    Some(u) => gl.active_texture(u),
                    None => gl.fail(GlError::InvalidEnum),
                }
                String::new()
            }
            "tex_image_2d" | "tex_image_3d" => {
                let three = f == "tex_image_3d";
                let k = usize::from(three);
                let target = enm!(ImageTarget, a.u(0));
                let internal = match InternalFormat::from_gl(a.u(2)) {
                    Ok(i) => i,
                    Err(_) => {
                        gl.fail(GlError::InvalidValue);
                        return Ok(String::new());
                    }
                };
                let size = [a.i(3), a.i(4), if three { a.i(5) } else { 1 }];
                let (format, ty) = (enm!(PixelFormat, a.u(6 + k)), enm!(PixelType, a.u(7 + k)));
                gl.tex_image(if three { Dims::Three } else { Dims::Two }, target, a.i(1), internal, size, a.i(5 + k), format, ty, a.texels(8 + k)).map_err(|e| e.0)?;
                String::new()
            }
            "tex_sub_image_2d" | "tex_sub_image_3d" => {
                let three = f == "tex_sub_image_3d";
                let target = enm!(ImageTarget, a.u(0));
                let region = if three { Region { level: a.i(1), offset: [a.i(2), a.i(3), a.i(4)], size: [a.i(5), a.i(6), a.i(7)] } } else { Region { level: a.i(1), offset: [a.i(2), a.i(3), 0], size: [a.i(4), a.i(5), 1] } };
                let k = if three { 8 } else { 6 };
                let (format, ty) = (enm!(PixelFormat, a.u(k)), enm!(PixelType, a.u(k + 1)));
                gl.tex_sub_image(if three { Dims::Three } else { Dims::Two }, target, region, format, ty, a.texels(k + 2)).map_err(|e| e.0)?;
                String::new()
            }
            "tex_storage_2d" | "tex_storage_3d" => {
                let target = enm!(TextureTarget, a.u(0));
                let internal = enm!(InternalFormat, a.u(2));
                let size = [a.i(3), a.i(4), if f == "tex_storage_3d" { a.i(5) } else { 1 }];
                gl.tex_storage(target, a.i(1), internal, size);
                String::new()
            }
            "tex_parameteri" => {
                let target = enm!(TextureTarget, a.u(0));
                match TexParam::from_gl(a.u(1), a.i(2)) {
                    Ok(p) => gl.tex_parameter(target, p),
                    Err(e) => gl.fail(e),
                }
                String::new()
            }
            "get_tex_parameter" => {
                let target = enm!(TextureTarget, a.u(0));
                format!(" {}", gl.get_tex_parameter(target, a.u(1)))
            }
            "pixel_storei" => {
                let v = a.i(1);
                let p = match a.u(0) {
                    Gl::PACK_ALIGNMENT => PixelStore::PackAlignment(v),
                    Gl::PACK_ROW_LENGTH => PixelStore::PackRowLength(v),
                    Gl::PACK_SKIP_PIXELS => PixelStore::PackSkipPixels(v),
                    Gl::PACK_SKIP_ROWS => PixelStore::PackSkipRows(v),
                    Gl::UNPACK_ALIGNMENT => PixelStore::UnpackAlignment(v),
                    Gl::UNPACK_ROW_LENGTH => PixelStore::UnpackRowLength(v),
                    Gl::UNPACK_IMAGE_HEIGHT => PixelStore::UnpackImageHeight(v),
                    Gl::UNPACK_SKIP_PIXELS => PixelStore::UnpackSkipPixels(v),
                    Gl::UNPACK_SKIP_ROWS => PixelStore::UnpackSkipRows(v),
                    Gl::UNPACK_SKIP_IMAGES => PixelStore::UnpackSkipImages(v),
                    _ => {
                        gl.fail(GlError::InvalidEnum);
                        return Ok(String::new());
                    }
                };
                gl.pixel_store(p);
                String::new()
            }
            "bind_buffer" => {
                gl.bind_buffer(enm!(BufferTarget, a.u(0)), a.u(1));
                String::new()
            }
            "buffer_data" => {
                let target = enm!(BufferTarget, a.u(0));
                let size = a.n(1) as usize;
                let data = matches!(a.v.get(2), Some(Arg::Data)).then(|| &a.data[..size.min(a.data.len())]);
                gl.buffer_data(target, size, data, enm!(Usage, a.u(3)));
                String::new()
            }
            "bind_renderbuffer" => {
                match a.u(0) {
                    Gl::RENDERBUFFER => gl.bind_renderbuffer(a.u(1)),
                    _ => gl.fail(GlError::InvalidEnum),
                }
                String::new()
            }
            "delete_renderbuffers" => {
                gl.delete_renderbuffers(&[a.u(0)]);
                String::new()
            }
            "renderbuffer_storage" | "renderbuffer_storage_multisample" => {
                let k = usize::from(f == "renderbuffer_storage_multisample");
                if a.u(0) != Gl::RENDERBUFFER {
                    gl.fail(GlError::InvalidEnum);
                    return Ok(String::new());
                }
                let samples = if k == 1 { a.i(1) } else { 0 };
                gl.renderbuffer_storage(samples, enm!(InternalFormat, a.u(1 + k)), a.i(2 + k), a.i(3 + k));
                String::new()
            }
            "get_renderbuffer_parameter" => {
                if a.u(0) != Gl::RENDERBUFFER {
                    gl.fail(GlError::InvalidEnum);
                    return Ok(String::new());
                }
                format!(" {}", gl.get_renderbuffer_parameter(a.u(1)))
            }
            "bind_framebuffer" => {
                gl.bind_framebuffer(enm!(FramebufferTarget, a.u(0)), a.u(1));
                String::new()
            }
            "delete_framebuffers" => {
                gl.delete_framebuffers(&[a.u(0)]);
                String::new()
            }
            "framebuffer_texture_2d" => {
                let (target, point) = (enm!(FramebufferTarget, a.u(0)), enm!(AttachPoint, a.u(1)));
                let image = enm!(ImageTarget, a.u(2));
                gl.framebuffer_texture_2d(target, point, image, a.u(3), a.i(4));
                String::new()
            }
            "framebuffer_texture_layer" => {
                let (target, point) = (enm!(FramebufferTarget, a.u(0)), enm!(AttachPoint, a.u(1)));
                gl.framebuffer_texture_layer(target, point, a.u(2), a.i(3), a.i(4));
                String::new()
            }
            "framebuffer_renderbuffer" => {
                let (target, point) = (enm!(FramebufferTarget, a.u(0)), enm!(AttachPoint, a.u(1)));
                if a.u(2) != Gl::RENDERBUFFER {
                    gl.fail(GlError::InvalidEnum);
                    return Ok(String::new());
                }
                gl.framebuffer_renderbuffer(target, point, a.u(3));
                String::new()
            }
            "check_framebuffer_status" => match FramebufferTarget::from_gl(a.u(0)) {
                Ok(t) => format!(" {}", gl.check_framebuffer_status(t).to_gl()),
                Err(e) => {
                    gl.fail(e);
                    " 0".into()
                }
            },
            "get_framebuffer_attachment_parameter" => {
                let (target, point) = (enm!(FramebufferTarget, a.u(0)), enm!(AttachPoint, a.u(1)));
                format!(" {}", gl.get_framebuffer_attachment_parameter(target, point, a.u(2)))
            }
            "draw_buffers" => {
                let mut bufs = Vec::new();
                for i in 0..a.v.len() {
                    bufs.push(enm!(ColorBuffer, a.u(i)));
                }
                gl.draw_buffers(&bufs);
                String::new()
            }
            "read_buffer" => {
                gl.read_buffer(enm!(ColorBuffer, a.u(0)));
                String::new()
            }
            f => return Err(format!("{f}: not a call glcalls knows")),
        })
    }
}

/// A script: its calls, what detour and Mesa say for each
struct Script {
    path: PathBuf,
}

impl Script {
    fn calls(&self) -> Result<Vec<(String, String, Vec<Arg>)>, String> {
        let text = fs::read_to_string(&self.path).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for line in text.lines().map(|l| l.split('#').next().unwrap_or("").trim()).filter(|l| !l.is_empty()) {
            let mut t = line.split_whitespace();
            let f = t.next().unwrap_or_default().to_owned();
            let args = t.map(Arg::try_from).collect::<Result<Vec<_>, _>>()?;
            out.push((line.to_owned(), f, args));
        }
        Ok(out)
    }

    fn mesa(glcalls: &Path, calls: &[(String, String, Vec<Arg>)]) -> Result<Vec<String>, String> {
        let wire: String = calls.iter().map(|(_, f, a)| format!("{f} {}\n", a.iter().map(|x| x.wire()).collect::<Vec<_>>().join(" "))).collect();
        let mut child = Command::new(glcalls).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        child.stdin.take().ok_or("no stdin")?.write_all(wire.as_bytes()).map_err(|e| e.to_string())?;
        let o = child.wait_with_output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).trim().to_owned());
        }
        Ok(String::from_utf8_lossy(&o.stdout).lines().map(str::to_owned).collect())
    }

    /// The differing lines, or how many calls matched
    fn run(&self, tools: &Toolchain<'_>, glcalls: &Path, data: &[u8]) -> Result<usize, String> {
        let calls = self.calls()?;
        let theirs = Script::mesa(glcalls, &calls)?;
        let mut gl = Context::new(tools, SIZE, SIZE, ColorFormat::Rgba8Unorm).map_err(|e| e.0)?;
        let mut bad = String::new();
        for (k, (line, f, args)) in calls.iter().enumerate() {
            let out = Detour { gl: &mut gl }.call(f, &Args { v: args, data })?;
            let err = gl.error().map_or(0, GlError::to_gl);
            let ours = format!("{err}{out}");
            let mesa = theirs.get(k).cloned().unwrap_or_default();
            if ours != mesa {
                let show = |s: &str| s.split_whitespace().map(|w| w.parse::<u32>().ok().filter(|v| *v >= Gl::INVALID_ENUM).and_then(Gl::name).unwrap_or_else(|| w.to_owned())).collect::<Vec<_>>().join(" ");
                writeln!(bad, "  {line}\n    detour {}\n    mesa   {}", show(&ours), show(&mesa)).unwrap();
            }
        }
        if bad.is_empty() { Ok(calls.len()) } else { Err(bad) }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let glcalls = PathBuf::from(args.next().expect("the glcalls binary"));
    let only: Vec<String> = args.collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/glcalls");
    let mut scripts: Vec<PathBuf> = fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "calls")).collect();
    scripts.sort();
    let tools = Toolchain::new();
    let data: Vec<u8> = (0..DATA).map(|i| i as u8).collect();
    let mut failed = false;
    for path in scripts {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        match (Script { path }).run(&tools, &glcalls, &data) {
            Ok(n) => println!("{name}: ok ({n} calls)"),
            Err(e) => {
                failed = true;
                println!("{name}: FAIL\n{e}");
            }
        }
    }
    std::process::exit(i32::from(failed));
}
