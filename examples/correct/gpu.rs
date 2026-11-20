//! Drawing through `Device::draw` (what acdes uses): GLSL made SPIR-V by mlirpass's front end, compiled by detour, and
//! draws described with their own vertex bytes.

use std::cell::RefCell;
use std::collections::HashMap;

use detour::{Blend, ColorFormat, ColorTarget, Compiled, CullMode, Device, Draw, DrawPrimitive, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement};
use mlirpass::{GlslReader, GlslStage};

const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";

pub struct Gpu<'t> {
    pub tools: &'t Toolchain<'t>,
    pub device: Device<'t>,
    programs: RefCell<HashMap<(String, String), &'t Compiled<'t>>>,
}

/// One draw: what `Draw` takes, owned
#[derive(Clone)]
pub struct Op {
    pub program: (String, String),
    pub buffers: Vec<(Vec<u8>, u32)>,
    pub elements: Vec<VertexElement>,
    pub uniforms: Vec<(u32, Vec<u8>)>,
    pub primitive: DrawPrimitive,
    pub first: u32,
    pub count: u32,
    pub instances: u32,
    pub viewport: [f32; 4],
    pub scissor: Option<[u32; 4]>,
    pub cull: CullMode,
    pub front_ccw: bool,
    pub blend: Option<Blend>,
    pub mask: [bool; 4],
}

/// A target: its size, format, layout, and its texel before the first draw
#[derive(Clone)]
pub struct Target {
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub tiling: Tiling,
    pub clear: Vec<u8>,
    /// its pixels before the first draw (linear rows), instead of `clear`
    pub pixels: Option<Vec<u8>>,
}

impl Target {
    pub fn rgba8(width: u32, height: u32, clear: [u8; 4]) -> Target {
        Target { width, height, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Tile4, clear: clear.to_vec(), pixels: None }
    }

    pub fn rgba32f(width: u32, height: u32) -> Target {
        Target { width, height, format: ColorFormat::Rgba32Float, tiling: Tiling::Tile4, clear: vec![0; 16], pixels: None }
    }
}

pub fn bytes(f: &[f32]) -> Vec<u8> {
    f.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl<'t> Gpu<'t> {
    pub fn new(tools: &'t Toolchain<'t>, device: Device<'t>) -> Self {
        Gpu { tools, device, programs: RefCell::new(HashMap::new()) }
    }

    /// The program of these shaders, compiled once
    pub fn program(&self, vs: &str, fs: &str) -> Result<&'t Compiled<'t>, String> {
        let key = (vs.to_owned(), fs.to_owned());
        if let Some(p) = self.programs.borrow().get(&key) {
            return Ok(p);
        }
        let mut body = self.tools.body();
        GlslReader::read(&mut body, vs, GlslStage::Vertex).map_err(|e| format!("vertex shader: {}", e.0))?;
        GlslReader::read(&mut body, fs, GlslStage::Fragment).map_err(|e| format!("fragment shader: {}", e.0))?;
        let root = body.root;
        self.tools.pipeline(TO_SPIRV).and_then(|p| p.run(&mut body, root)).map_err(|e| format!("to SPIR-V: {}", e.0))?;
        let text = self.tools.show(&body);
        let bin = self.tools.program(&text).map_err(|e| format!("compiling: {}", e.0))?;
        let bin: &'t [u8] = Box::leak(bin.into_boxed_slice());
        let compiled: &'t Compiled<'t> = Box::leak(Box::new(self.tools.load(bin).map_err(|e| format!("loading: {}", e.0))?));
        self.programs.borrow_mut().insert(key, compiled);
        Ok(compiled)
    }

    /// The draws into a new target, its pixels read back (linear rows)
    pub fn run(&self, t: &Target, ops: &[Op]) -> Result<Vec<u8>, String> {
        let target = |load| ColorTarget { width: t.width, height: t.height, format: t.format, tiling: t.tiling, load };
        let surface = self.device.surface(&target(TargetLoad::Keep(&[]))).map_err(|e| e.0)?;
        let result = (|| {
            for (i, op) in ops.iter().enumerate() {
                let program = self.program(&op.program.0, &op.program.1)?;
                let load = match (i, &t.pixels) {
                    (0, Some(p)) => TargetLoad::Pixels(p, &[]),
                    (0, None) => TargetLoad::Fill(&t.clear, &[]),
                    _ => TargetLoad::Keep(&[]),
                };
                let d = Draw {
                    buffers: op.buffers.iter().map(|(b, stride)| VertexBuffer { bytes: b, stride: *stride }).collect(),
                    elements: op.elements.clone(),
                    uniforms: op.uniforms.iter().map(|(b, u)| (*b, u.as_slice())).collect(),
                    textures: Vec::new(),
                    primitive: op.primitive,
                    indices: None,
                    first: op.first,
                    count: op.count,
                    instances: op.instances,
                    viewport: op.viewport,
                    depth_range: [0.0, 1.0],
                    scissor: op.scissor,
                    cull: op.cull,
                    front_ccw: op.front_ccw,
                    blend: op.blend,
                    color_mask: op.mask,
                    target: target(load),
                };
                self.device.draw(program, &d, surface).map_err(|e| e.0)?;
            }
            let len = (t.width * t.height * t.format.bytes()) as usize;
            if ops.is_empty() {
                return Ok(t.pixels.clone().unwrap_or_else(|| t.clear.repeat(len / t.clear.len())));
            }
            self.device.read(surface, len).map_err(|e| e.0)
        })();
        self.device.free(surface).map_err(|e| e.0)?;
        result
    }
}
