//! Programs compiled ahead of time (`Toolchain::program`) and loaded without compiling (`Toolchain::load`), drawn
//! with `Device::draw` and no GL: each case under examples/cases that sets no uniforms, its GLSL made SPIR-V MLIR text
//! by detour's front end, compiled twice (the bytes must be the same), loaded, drawn, and compared byte for byte with
//! the same case drawn through detour's GL. A program for another detour or GPU, and bytes that are not one, must be
//! refused.
//!
//!   cargo run --release --example program [CASE...]      (DETOUR_SPIRV=DIR: the SPIR-V MLIR text of each case there)

use std::fs;
use std::path::{Path, PathBuf};

#[path = "glcase/mod.rs"]
mod glcase;

use detour::{ClearMask, ColorFormat, ColorTarget, CullMode, Device, Draw, DrawMode, DrawPrimitive, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat};
use glcase::{Case, Resource, VERTICES};
use mlirpass::{GlslReader, GlslStage};

const SIZE: u32 = 64;
const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";

/// A case's vertex inputs in location order: floats a vertex, values
type Inputs = Vec<(u32, Vec<f32>)>;

/// A case's SPIR-V modules as MLIR text, and its vertex inputs
fn spirv(case: &Case, tools: &Toolchain<'_>) -> Result<(String, Inputs), String> {
    let resources = case.resources()?;
    if resources.iter().any(|r| !matches!(r, Resource::Attribute { .. })) {
        return Err("sets uniforms: left out".into());
    }
    let read = |f: &str| fs::read_to_string(case.dir.join(f)).map_err(|e| format!("{f}: {e}"));
    let mut body = tools.body();
    GlslReader::read(&mut body, &read("shader.vert")?, GlslStage::Vertex).map_err(|e| e.0)?;
    GlslReader::read(&mut body, &read("shader.frag")?, GlslStage::Fragment).map_err(|e| e.0)?;
    let root = body.root;
    tools.pipeline(TO_SPIRV).and_then(|p| p.run(&mut body, root)).map_err(|e| e.0)?;
    let vert = body.ops(body.top_block(root)).find(|&m| &*body[m].name == "spirv.module").ok_or("no spirv.module")?;
    let mut inputs = Vec::new();
    for g in body.ops(body.top_block(vert)).filter(|&g| &*body[g].name == "spirv.GlobalVariable") {
        let op = &body[g];
        let input = op.get("type").is_some_and(|t| t.to_string().contains("Input"));
        let (Some(name), Some(at), false, true) = (op.get_str("sym_name"), op.get("location"), op.get("built_in").is_some(), input) else { continue };
        let at: u32 = at.to_string().split(':').next().unwrap_or("").trim().parse().map_err(|_| format!("@{name}: location {at}"))?;
        let r = resources.iter().find_map(|r| match r {
            Resource::Attribute { name: n, size, values } if n == name => Some((*size, values.clone())),
            _ => None,
        });
        let (size, values) = r.ok_or_else(|| format!("input {name}: not in resources"))?;
        inputs.push((at, size, values));
    }
    inputs.sort_by_key(|i| i.0);
    Ok((tools.show(&body), inputs.into_iter().map(|(_, n, v)| (n, v)).collect()))
}

/// The case drawn through detour's GL
fn through_gl(case: &Case, tools: &Toolchain<'_>, device: &Device<'_>) -> Result<Vec<u8>, String> {
    let mut gl = case.context(tools, SIZE)?;
    gl.viewport(0, 0, SIZE as i32, SIZE as i32);
    gl.clear_color([0.0; 4]);
    gl.clear(ClearMask { color: true, ..Default::default() });
    gl.draw_arrays(device, DrawMode::Triangles, 0, VERTICES).map_err(|e| e.0)?;
    gl.framebuffer.pixels(device).map_err(|e| e.0)
}

/// The case compiled ahead of time, loaded and drawn with `Device::draw`
fn loaded(text: &str, inputs: &[(u32, Vec<f32>)], tools: &Toolchain<'_>, device: &Device<'_>) -> Result<Vec<u8>, String> {
    let program = tools.program(text).map_err(|e| e.0)?;
    if tools.program(text).map_err(|e| e.0)? != program {
        return Err("compiled twice, two different programs".into());
    }
    let compiled = tools.load(&program).map_err(|e| e.0)?;
    let bytes: Vec<Vec<u8>> = inputs.iter().map(|(_, v)| v.iter().flat_map(|f| f.to_le_bytes()).collect()).collect();
    let mut elements = Vec::new();
    for (i, (n, _)) in inputs.iter().enumerate() {
        elements.push(VertexElement { buffer: i as u32, format: VertexFormat::try_from(*n).map_err(|e| e.0)?, offset: 0 });
    }
    let clear = [0u8; 16];
    let target = ColorTarget { width: SIZE, height: SIZE, format: ColorFormat::Rgba32Float, tiling: Tiling::Tile4, load: TargetLoad::Fill(&clear, &[]) };
    let surface = device.surface(&target).map_err(|e| e.0)?;
    let draw = Draw {
        buffers: bytes.iter().zip(inputs).map(|(b, (n, _))| VertexBuffer { bytes: b, stride: n * 4 }).collect(),
        elements,
        uniforms: Vec::new(),
        primitive: DrawPrimitive::Triangles,
        first: 0,
        count: VERTICES as u32,
        instances: 1,
        viewport: [0.0, 0.0, SIZE as f32, SIZE as f32],
        depth_range: [0.0, 1.0],
        scissor: None,
        cull: CullMode::None,
        front_ccw: true,
        blend: None,
        color_mask: [true; 4],
        target,
    };
    device.draw(&compiled, &draw, surface).map_err(|e| e.0)?;
    device.read(surface, (SIZE * SIZE * 16) as usize).map_err(|e| e.0)
}

/// What `load` must refuse
fn refusals(text: &str, tools: &Toolchain<'_>) -> Result<(), String> {
    let program = String::from_utf8(tools.program(text).map_err(|e| e.0)?).map_err(|e| e.to_string())?;
    let version = program.lines().nth(1).unwrap_or_default().to_owned();
    let others = [program.replacen(&format!("\n{version}\n"), "\n0.0.0-other\n", 1), program.replacen("\nxe3\n", "\nrdna2\n", 1), "not a program".into(), program[..program.len() / 2].to_owned()];
    for (what, bytes) in ["another detour", "another GPU", "not a program", "cut short"].iter().zip(others) {
        if tools.load(bytes.as_bytes()).is_ok() {
            return Err(format!("{what}: loaded"));
        }
    }
    Ok(())
}

fn main() {
    let only: Vec<String> = std::env::args().skip(1).collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("examples/cases")).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let save = std::env::var_os("DETOUR_SPIRV").map(PathBuf::from);
    let tools = Toolchain::new();
    let xe = <detour::Xe as detour::Gpu>::open_first().unwrap();
    let device = Device::new(&xe);
    let mut failed = 0;
    let mut refused = false;
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        let case = Case { dir };
        let r = spirv(&case, &tools).and_then(|(text, inputs)| {
            if let Some(dir) = &save {
                fs::write(dir.join(format!("{name}.mlir")), &text).map_err(|e| e.to_string())?;
            }
            if !refused {
                refusals(&text, &tools)?;
                refused = true;
            }
            let ours = loaded(&text, &inputs, &tools, &device)?;
            let gl = through_gl(&case, &tools, &device)?;
            match ours == gl {
                true => Ok(format!("ok ({} bytes)", ours.len())),
                false => Err("differs from the same case drawn through GL".into()),
            }
        });
        match &r {
            Err(e) if e.ends_with("left out") => println!("{name}: {e}"),
            _ => {
                failed += usize::from(r.is_err());
                println!("{name}: {}", r.unwrap_or_else(|e| format!("FAIL {e}")));
            }
        }
    }
    std::process::exit(i32::from(failed > 0));
}
