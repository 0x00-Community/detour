//! Programs compiled ahead of time (`Toolchain::program`) and loaded without compiling (`Toolchain::load`), drawn
//! with `Device::draw`: each case under examples/cases that sets no uniforms, its GLSL made SPIR-V MLIR text by
//! detour's front end, compiled twice (the bytes must be the same), loaded, drawn, and compared byte for byte with the
//! same SPIR-V compiled in place (`Toolchain::compile`) and drawn. A program for another detour or GPU, and bytes that
//! are not one (cut short, or with bytes past its end), must be refused. A case's `resources` has a line per vertex input, `attribute NAME LOCATION N V...`
//! (three vertices of N floats); a case with a `uniform` line is left out.
//!
//!   cargo run --release --example program [CASE...]      (DETOUR_SPIRV=DIR: the SPIR-V MLIR text of each case there)

use std::fs;
use std::path::{Path, PathBuf};

use detour::{
    ColorFormat, ColorTarget, Compiled, CullMode, Device, Draw, DrawPrimitive, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat,
};
use mlirpass::{Body, GlslReader, GlslStage};

const SIZE: u32 = 64;
/// The vertices a case draws
const VERTICES: u32 = 3;
const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";

/// A case's vertex inputs in location order: floats a vertex, values
type Inputs = Vec<(u32, Vec<f32>)>;

/// A case's vertex inputs by name: floats a vertex, values
fn attributes(dir: &Path) -> Result<Vec<(String, u32, Vec<f32>)>, String> {
    let text = fs::read_to_string(dir.join("resources")).map_err(|e| format!("resources: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.as_slice() {
            ["attribute", name, _, n, v @ ..] => {
                let values = v.iter().map(|x| x.parse::<f32>().map_err(|e| format!("{line}: {e}"))).collect::<Result<_, _>>()?;
                out.push(((*name).to_owned(), n.parse().map_err(|e| format!("{line}: {e}"))?, values));
            }
            ["uniform", ..] => return Err("sets uniforms: left out".into()),
            _ => return Err(format!("resources: {line}")),
        }
    }
    Ok(out)
}

/// A case's SPIR-V modules, as a body and as MLIR text, and its vertex inputs
fn spirv<'t>(dir: &Path, tools: &'t Toolchain<'t>) -> Result<(Body<'t>, String, Inputs), String> {
    let attributes = attributes(dir)?;
    let read = |f: &str| fs::read_to_string(dir.join(f)).map_err(|e| format!("{f}: {e}"));
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
        let (_, size, values) = attributes.iter().find(|a| a.0 == name).cloned().ok_or_else(|| format!("input {name}: not in resources"))?;
        inputs.push((at, size, values));
    }
    inputs.sort_by_key(|i| i.0);
    let text = tools.show(&body);
    Ok((body, text, inputs.into_iter().map(|(_, n, v)| (n, v)).collect()))
}

/// The case compiled ahead of time and loaded
fn loaded(text: &str, tools: &Toolchain<'_>) -> Result<Compiled, String> {
    let program = tools.program(text).map_err(|e| e.0)?;
    if tools.program(text).map_err(|e| e.0)? != program {
        return Err("compiled twice, two different programs".into());
    }
    tools.load(&program).map_err(|e| e.0)
}

/// `compiled` drawn with the case's vertex inputs, read back
fn draw(compiled: &Compiled, inputs: &[(u32, Vec<f32>)], device: &Device<'_>) -> Result<Vec<u8>, String> {
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
        textures: Vec::new(),
        primitive: DrawPrimitive::Triangles,
        indices: None,
        first: 0,
        count: VERTICES,
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
    device.draw(compiled, &draw, surface).map_err(|e| e.0)?;
    device.read(surface, (SIZE * SIZE * 16) as usize).map_err(|e| e.0)
}

/// What `load` must refuse
fn refusals(text: &str, tools: &Toolchain<'_>) -> Result<(), String> {
    let program = tools.program(text).map_err(|e| e.0)?;
    let lines: Vec<&[u8]> = program.splitn(4, |b| *b == b'\n').collect();
    let [magic, _, gpu, rest] = lines[..] else { return Err("a program without its three header lines".into()) };
    let with = |version: &[u8], gpu: &[u8], rest: &[u8]| [magic, version, gpu, rest].join(&b'\n');
    let others = [
        with(b"0.0.0-other", gpu, rest),
        with(env!("CARGO_PKG_VERSION").as_bytes(), b"rdna2", rest),
        b"not a program".to_vec(),
        program[..program.len() - 1].to_vec(),
        [&program[..], b"\0"].concat(),
    ];
    for (what, bytes) in ["another detour", "another GPU", "not a program", "cut short", "bytes past its end"].iter().zip(others) {
        if tools.load(&bytes).is_ok() {
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
        let r = spirv(&dir, &tools).and_then(|(body, text, inputs)| {
            if let Some(dir) = &save {
                fs::write(dir.join(format!("{name}.mlir")), &text).map_err(|e| e.to_string())?;
            }
            if !refused {
                refusals(&text, &tools)?;
                refused = true;
            }
            let ours = draw(&loaded(&text, &tools)?, &inputs, &device)?;
            let compiled = tools.compile(body).map_err(|e| e.0)?;
            match ours == draw(&compiled, &inputs, &device)? {
                true => Ok(format!("ok ({} bytes)", ours.len())),
                false => Err("differs from the same SPIR-V compiled in place".into()),
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
