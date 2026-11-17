//! detour's SPIR-V path on Xe3 against the official Vulkan driver (Mesa's ANV), the same work on both: each case
//! under examples/cases made SPIR-V for Vulkan once (detour's GLSL front end), then on each side the two SPIR-V
//! modules made machine code (detour: SPIR-V read, lowered to Xe3; ANV: shader modules + pipeline) and FRAMES frames
//! of clear + the triangle into a SIZE² RGBA32F target, each waited for. A case setting uniforms is left out (Vulkan
//! takes them in a block, which these cases do not declare).
//!
//! detour has no Vulkan API yet: its side is what one will run, the SPIR-V a Vulkan application hands it drawn
//! through `Device::draw`.
//!
//!   cc -O2 -o vkbench examples/support/vkbench.c -lvulkan
//!   cargo run --release --example bench_vulkan -- VKBENCH [CASE...]      (FRAMES, default 100; SIZE, default 256)

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[path = "glcase/mod.rs"]
mod glcase;

use detour::{ColorFormat, ColorTarget, CullMode, Device, Draw, DrawPrimitive, Toolchain, VertexBuffer, VertexElement, VertexFormat};
use glcase::{Case, Resource, VERTICES};
use mlirformat::Error;
use mlirpass::{Body, Emit, GlslReader, GlslStage, Shift, SpirvBinary};

/// Frames a case draws when FRAMES is not set
const FRAMES: u32 = 100;
/// The target's side when SIZE is not set
const SIZE: u32 = 256;
/// Nanoseconds a millisecond
const NS_MS: f64 = 1e6;
/// GLSL ⇒ SPIR-V for Vulkan
const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";

/// One driver's times for a case
struct Times {
    compile: Duration,
    first: Duration,
    mean: Duration,
    min: Duration,
}

/// A case as Vulkan takes it: its two SPIR-V modules, and its vertex inputs (location, floats a vertex, values)
struct Spirv {
    vert: Vec<u8>,
    frag: Vec<u8>,
    inputs: Vec<(u32, u32, Vec<f32>)>,
}

impl Spirv {
    /// The case's GLSL made SPIR-V for Vulkan, by detour's front end
    fn of(case: &Case, tools: &Toolchain<'_>) -> Result<Spirv, String> {
        let resources = case.resources()?;
        if resources.iter().any(|r| !matches!(r, Resource::Attribute { .. })) {
            return Err("sets uniforms: left out".into());
        }
        let read = |f: &str| std::fs::read_to_string(case.dir.join(f)).map_err(|e| format!("{f}: {e}"));
        let mut body = tools.body();
        GlslReader::read(&mut body, &read("shader.vert")?, GlslStage::Vertex).map_err(|e| e.0)?;
        GlslReader::read(&mut body, &read("shader.frag")?, GlslStage::Fragment).map_err(|e| e.0)?;
        let root = body.root;
        tools.pipeline(TO_SPIRV).and_then(|p| p.run(&mut body, root)).map_err(|e| e.0)?;
        let modules: Vec<_> = body.ops(body.top_block(root)).filter(|&m| &*body[m].name == "spirv.module").collect();
        let [vert, frag] = modules[..] else { return Err("not two spirv.module".into()) };
        let words = |m| SpirvBinary::default().emit(&body, m).map(|w| SpirvBinary::bytes(&w)).map_err(|e| e.0);
        let mut inputs = Vec::new();
        for g in body.ops(body.top_block(vert)).filter(|&g| &*body[g].name == "spirv.GlobalVariable") {
            let op = &body[g];
            let input = op.get("type").is_some_and(|t| t.to_string().contains("Input"));
            let (Some(name), Some(at), false) = (op.get_str("sym_name"), op.get("location"), op.get("built_in").is_some()) else { continue };
            if !input {
                continue;
            }
            let at: u32 = at.to_string().split(':').next().unwrap_or("").trim().parse().map_err(|_| format!("@{name}: location {at}"))?;
            let r = resources.iter().find_map(|r| match r {
                Resource::Attribute { name: n, size, values } if n == name => Some((*size, values.clone())),
                _ => None,
            });
            let (size, values) = r.ok_or_else(|| format!("input {name}: not in resources"))?;
            inputs.push((at, size, values));
        }
        inputs.sort_by_key(|i| i.0);
        Ok(Spirv { vert: words(vert)?, frag: words(frag)?, inputs })
    }

    /// The SPIR-V on detour: read, lowered to Xe3, drawn
    fn detour(&self, tools: &Toolchain<'_>, device: &Device, size: u32, frames: u32) -> Result<Times, String> {
        let start = Instant::now();
        let mut body = tools.body();
        for bytes in [&self.vert, &self.frag] {
            Spirv::absorb(&mut body, bytes).map_err(|e| e.0)?;
        }
        let compiled = tools.compile(body).map_err(|e| e.0)?;
        let compile = start.elapsed();
        let bytes: Vec<Vec<u8>> = self.inputs.iter().map(|(_, _, v)| v.iter().flat_map(|f| f.to_le_bytes()).collect()).collect();
        let pixels = vec![0u8; (size * size * ColorFormat::Rgba32Float.bytes()) as usize];
        let mut elements = Vec::new();
        for (i, (_, n, _)) in self.inputs.iter().enumerate() {
            elements.push(VertexElement { buffer: i as u32, format: VertexFormat::try_from(*n).map_err(|e| e.0)?, offset: 0 });
        }
        let draw = Draw {
            buffers: bytes.iter().zip(&self.inputs).map(|(b, (_, n, _))| VertexBuffer { bytes: b, stride: n * 4 }).collect(),
            elements,
            uniforms: Vec::new(),
            primitive: DrawPrimitive::Triangles,
            first: 0,
            count: VERTICES as u32,
            instances: 1,
            viewport: [0.0, 0.0, size as f32, size as f32],
            depth_range: [0.0, 1.0],
            scissor: None,
            cull: CullMode::None,
            front_ccw: true,
            target: ColorTarget { width: size, height: size, format: ColorFormat::Rgba32Float, pixels: &pixels },
        };
        let mut times = Vec::new();
        for _ in 0..frames {
            let t = Instant::now();
            device.draw(&compiled, &draw).map_err(|e| e.0)?;
            times.push(t.elapsed());
        }
        let rest = &times[1..];
        Ok(Times { compile, first: times[0], mean: rest.iter().sum::<Duration>() / rest.len() as u32, min: rest.iter().copied().min().unwrap_or_default() })
    }

    /// A SPIR-V module read into `body`, at the end of its top block
    fn absorb(body: &mut Body<'_>, bytes: &[u8]) -> Result<(), Error> {
        let sub = SpirvBinary::default().read(body.ctx, bytes)?;
        let m = sub.first_op(sub.top_block(sub.root)).ok_or_else(|| Error::from("an empty SPIR-V module"))?;
        let moved = m.shift(&body.absorb(sub));
        body.detach(moved);
        let top = body.top_block(body.root);
        body.insert(moved, top, None);
        Ok(())
    }

    /// The SPIR-V on ANV, through vkbench
    fn mesa(&self, vkbench: &Path, dir: &Path, size: u32, frames: u32) -> Result<Times, String> {
        let (vert, frag, attrs) = (dir.join("vert.spv"), dir.join("frag.spv"), dir.join("attrs"));
        let lines: Vec<String> = self.inputs.iter().map(|(at, n, v)| format!("{at} {n} {}", v.iter().map(f32::to_string).collect::<Vec<_>>().join(" "))).collect();
        std::fs::write(&vert, &self.vert).and_then(|_| std::fs::write(&frag, &self.frag)).and_then(|_| std::fs::write(&attrs, lines.join("\n") + "\n")).map_err(|e| e.to_string())?;
        let o = Command::new(vkbench).args([&vert, &frag, &attrs]).args([size.to_string(), size.to_string(), frames.to_string()]).output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).trim().to_owned());
        }
        let ns: Vec<u64> = String::from_utf8_lossy(&o.stdout).split_whitespace().map(|v| v.parse().map_err(|e| format!("vkbench: {e}"))).collect::<Result<_, _>>()?;
        let [compile, first, mean, min] = ns[..] else { return Err("vkbench: four times expected".into()) };
        Ok(Times { compile: Duration::from_nanos(compile), first: Duration::from_nanos(first), mean: Duration::from_nanos(mean), min: Duration::from_nanos(min) })
    }
}

impl Times {
    fn ms(d: Duration) -> String {
        format!("{:.3}", d.as_nanos() as f64 / NS_MS)
    }

    /// detour over Mesa (how many times as long)
    fn ratio(ours: Duration, theirs: Duration) -> String {
        format!("{:.2}×", ours.as_secs_f64() / theirs.as_secs_f64().max(f64::MIN_POSITIVE))
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let vkbench = PathBuf::from(args.next().expect("the vkbench binary (examples/support/vkbench.c)"));
    let num = |v: &str, d: u32| std::env::var(v).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let (frames, size) = (num("FRAMES", FRAMES).max(2), num("SIZE", SIZE));
    let only: Vec<String> = args.collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cases");
    let scratch = std::env::temp_dir().join("detour-bench_vulkan");
    std::fs::create_dir_all(&scratch).unwrap();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().filter_map(|e| Some(e.ok()?.path())).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let tools = Toolchain::new();
    let device = Device::open().expect("an Xe3 GPU");
    println!("{frames} frames of {size}×{size} RGBA32F a case; times in ms (detour / mesa), ratio = detour ÷ mesa");
    println!("{:<18} {:>21} {:>8} {:>21} {:>8} {:>21} {:>8}", "case", "compile", "", "first frame", "", "frame (mean)", "");
    for dir in dirs.into_iter().filter(|d| only.is_empty() || only.iter().any(|o| d.ends_with(o))) {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let case = Case { dir };
        let spirv = match Spirv::of(&case, &tools) {
            Ok(s) => s,
            Err(e) => {
                println!("{name:<18} {e}");
                continue;
            }
        };
        match (spirv.detour(&tools, &device, size, frames), spirv.mesa(&vkbench, &scratch, size, frames)) {
            (Ok(d), Ok(m)) => println!(
                "{name:<18} {:>10} / {:<8} {:>8} {:>10} / {:<8} {:>8} {:>10} / {:<8} {:>8}   (fastest {} / {})",
                Times::ms(d.compile), Times::ms(m.compile), Times::ratio(d.compile, m.compile),
                Times::ms(d.first), Times::ms(m.first), Times::ratio(d.first, m.first),
                Times::ms(d.mean), Times::ms(m.mean), Times::ratio(d.mean, m.mean),
                Times::ms(d.min), Times::ms(m.min),
            ),
            (d, m) => println!("{name:<18} detour: {}  mesa: {}", d.err().map_or("ok".into(), |e| e.lines().next().unwrap_or("").to_owned()), m.err().unwrap_or_else(|| "ok".into())),
        }
    }
}
