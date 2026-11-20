#![allow(dead_code, reason = "each example uses part of it")]

//! A case of examples/cases as a Vulkan application hands it to its driver: GLSL made SPIR-V for Vulkan (detour's
//! front end; the loose uniforms packed in a block at set 0, binding 0, as glslang's --vulkan-relaxed does), vertex
//! inputs and the block's bytes; then set up on detour's Vulkan call for call as examples/support/vkrun.c sets it up
//! on the local driver. What `compare_vulkan` and `bench_vulkan` share.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use detour::{Toolchain, VertexFormat, VkStorage};
use detour::{VkAttachmentDescription, VkAttachmentLoadOp, VkAttachmentStoreOp, VkBuffer, VkBufferImageCopy, VkClearValue, VkCommandBuffer, VkDescriptorBufferInfo, VkDescriptorSetLayoutBinding};
use detour::{VkDescriptorType, VkDevice, VkDeviceMemory, VkFence, VkFormat, VkGraphicsPipelineCreateInfo, VkImage, VkImageCreateInfo, VkMemoryPropertyFlags, VkPipelineShaderStageCreateInfo};
use detour::{VkPrimitiveTopology, VkQueue, VkRasterization, VkRect2D, VkRenderPassBeginInfo, VkRenderPassCreateInfo, VkShaderStage, VkSubmitInfo, VkVertexInputAttributeDescription};
use detour::{VkVertexInputBindingDescription, VkVertexInputRate, VkViewport, VkWriteDescriptorSet};
use mlirformat::{Attribute, Op};
use mlirformat::spirv_types::SpirvType;
use mlirpass::{Body, Emit, GlslGlobal, GlslReader, GlslStage, Heads, SpirvBinary, Storage, Symbol};

/// The vertices a case draws
pub const VERTICES: i32 = 3;

/// The uniform block's binding (set 0)
pub const UBO_BINDING: u32 = 0;
/// The target's format, and a texel's bytes
const FORMAT: VkFormat = VkFormat::R32g32b32a32Sfloat;
/// std140: a block's size is a multiple of a vec4
const STD140_VEC4: usize = 16;
const WORD: usize = 4;
const LINK: &str = "builtin.module(glsl-link)";
const TO_SPIRV: &str = "builtin.module(convert-glsl-to-spirv{client=vulkan})";
const SPIRV_MODULE: &str = "spirv.module";
const GLOBAL_VARIABLE: &str = "spirv.GlobalVariable";
/// The block glsl-default-block gathers the loose uniforms into, and its location ⇒ byte offset table
const DEFAULT_BLOCK: &str = "gl_DefaultUniformBlock";
const OFFSETS: &str = "offsets";

/// One line of a case's `resources` (glrun's):
///   attribute NAME LOCATION N V...   three vertices of N floats
///   uniform NAME LOCATION f|i V...   glUniform{1,2,3,4}{f,i}v (16 floats: glUniformMatrix4fv)
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

/// One case of examples/cases: its directory (shader.vert, shader.frag, resources)
pub struct Case {
    pub dir: PathBuf,
}

impl Case {
    /// The case's `resources`, a line each
    pub fn resources(&self) -> Result<Vec<Resource>, String> {
        let text = std::fs::read_to_string(self.dir.join("resources")).map_err(|e| format!("resources: {e}"))?;
        text.lines().filter(|l| !l.trim().is_empty()).map(Resource::try_from).collect()
    }
}

/// The program's GLSL globals after glsl-link: the loose uniforms (name ⇒ location) and, once glsl-default-block
/// gathered them, the block's location ⇒ byte offset table
struct Loose;

impl Loose {
    fn globals(body: &Body<'_>) -> Vec<mlirpass::OpId> {
        let mut globals = Vec::new();
        body.walk(body.root, &mut |o| globals.push(o));
        globals.retain(|&g| &*body[g].name == GlslGlobal::NAME);
        globals
    }

    /// The loose uniforms (one used in both stages once), by name: their locations
    fn uniforms(body: &Body<'_>) -> Result<Vec<(String, u32)>, String> {
        let mut out: Vec<(String, u32)> = Vec::new();
        for g in Loose::globals(body) {
            let h = body.head::<GlslGlobal>(g).map_err(|e| e.0)?;
            if h.storage != Storage::Uniform || h.builtin.is_some() {
                continue;
            }
            if let Ok(SpirvType::Struct { .. }) = SpirvType::try_from(&h.ty) {
                continue;
            }
            let name = Symbol::defined(&h.sym_name).map_err(|e| e.0)?.to_owned();
            if out.iter().any(|(n, _)| *n == name) {
                continue;
            }
            out.push((name.clone(), h.location.ok_or_else(|| format!("uniform {name} without a location after glsl-link"))?));
        }
        Ok(out)
    }

    /// The default block's location ⇒ byte offset table
    fn offsets(body: &Body<'_>) -> Result<BTreeMap<u32, u32>, String> {
        let mut offsets = BTreeMap::new();
        for g in Loose::globals(body) {
            let op = &body[g];
            if op.get(Symbol::ATTR).and_then(|s| Symbol::defined(s).ok()) != Some(DEFAULT_BLOCK) {
                continue;
            }
            let Some(Attribute::Array(pairs)) = op.get(OFFSETS) else { return Err(format!("{DEFAULT_BLOCK} without its {OFFSETS}")) };
            for p in pairs {
                let Attribute::Array(lo) = p else { return Err(format!("{OFFSETS}: {p}")) };
                let [l, o] = lo.as_slice() else { return Err(format!("{OFFSETS}: {p}")) };
                let int = |a: &Attribute<'_>| a.int().ok().and_then(|v| u32::try_from(v).ok()).ok_or_else(|| format!("{OFFSETS}: {p}"));
                offsets.insert(int(l)?, int(o)?);
            }
        }
        Ok(offsets)
    }

    /// A vertex shader's application inputs (not built-ins): name and location
    fn inputs(body: &Body<'_>, module: mlirpass::OpId) -> Result<Vec<(String, u32)>, String> {
        let mut out = Vec::new();
        for g in body.ops(body.top_block(module)).filter(|&g| &*body[g].name == GLOBAL_VARIABLE) {
            let op = &body[g];
            let name = op.get_str("sym_name").ok_or("spirv.GlobalVariable without a name")?.to_owned();
            let Some(Attribute::Type(t)) = op.get("type") else { continue };
            let Ok(SpirvType::Ptr { storage, .. }) = SpirvType::try_from(t) else { continue };
            if VkStorage::from(&*storage) != VkStorage::Input || op.get("built_in").is_some() {
                continue;
            }
            let at = match op.get("location") {
                Some(Attribute::Int { lit, .. }) => lit.parse().map_err(|_| format!("input {name}: location {lit}"))?,
                _ => return Err(format!("input {name}: no location")),
            };
            out.push((name, at));
        }
        Ok(out)
    }
}
/// The memory a buffer the host writes or reads takes
const HOST: VkMemoryPropertyFlags = VkMemoryPropertyFlags { device_local: false, host_visible: true, host_coherent: true, host_cached: false };
const ENTRY: &str = "main";

/// A case as Vulkan takes it: its two SPIR-V modules, its vertex inputs (location, floats a vertex, values) and its
/// uniform block (empty: none)
pub struct VkCase {
    pub vert: Vec<u8>,
    pub frag: Vec<u8>,
    pub inputs: Vec<(u32, u32, Vec<f32>)>,
    pub ubo: Vec<u8>,
}

/// One driver's times for a case
pub struct Times {
    pub compile: Duration,
    pub first: Duration,
    pub mean: Duration,
    pub min: Duration,
}

impl VkCase {
    /// The case's GLSL made SPIR-V for Vulkan, by detour's front end
    pub fn of(case: &Case, tools: &Toolchain<'_>) -> Result<VkCase, String> {
        let resources = case.resources()?;
        let read = |f: &str| std::fs::read_to_string(case.dir.join(f)).map_err(|e| format!("{f}: {e}"));
        let mut body = tools.body();
        GlslReader::read(&mut body, &read("shader.vert")?, GlslStage::Vertex).map_err(|e| e.0)?;
        GlslReader::read(&mut body, &read("shader.frag")?, GlslStage::Fragment).map_err(|e| e.0)?;
        let root = body.root;
        let run = |body: &mut mlirpass::Body<'_>, text: &str| tools.pipeline(text).and_then(|p| p.run(body, root)).map_err(|e| e.0);
        run(&mut body, LINK)?;
        let loose = Loose::uniforms(&body)?;
        if !loose.is_empty() {
            run(&mut body, &format!("builtin.module(glsl-default-block{{binding={UBO_BINDING}}})"))?;
        }
        let offsets = Loose::offsets(&body)?;
        run(&mut body, TO_SPIRV)?;
        let modules: Vec<_> = body.ops(body.top_block(root)).filter(|&m| &*body[m].name == SPIRV_MODULE).collect();
        let [vert, frag] = modules[..] else { return Err("not two spirv.module".into()) };
        let words = |m| SpirvBinary::default().emit(&body, m).map(|w| SpirvBinary::bytes(&w)).map_err(|e| e.0);
        let mut inputs = Vec::new();
        for (input, at) in Loose::inputs(&body, vert)? {
            let r = resources.iter().find_map(|r| match r {
                Resource::Attribute { name, size, values } if *name == input => Some((*size, values.clone())),
                _ => None,
            });
            let (size, values) = r.ok_or_else(|| format!("input {input}: not in resources"))?;
            inputs.push((at, size, values));
        }
        inputs.sort_by_key(|i| i.0);
        let mut ubo = Vec::new();
        for r in &resources {
            let (name, words): (&str, Vec<[u8; WORD]>) = match r {
                Resource::Attribute { .. } => continue,
                Resource::Float { name, values } => (name, values.iter().map(|f| f.to_le_bytes()).collect()),
                Resource::Int { name, values } => (name, values.iter().map(|i| i.to_le_bytes()).collect()),
            };
            let Some((_, at)) = loose.iter().find(|(u, _)| u == name) else { continue };
            let off = *offsets.get(at).ok_or_else(|| format!("uniform {name}: no offset in the block"))? as usize;
            let end = (off + words.len() * WORD).next_multiple_of(STD140_VEC4);
            if ubo.len() < end {
                ubo.resize(end, 0);
            }
            words.iter().enumerate().for_each(|(i, w)| ubo[off + i * WORD..off + (i + 1) * WORD].copy_from_slice(w));
        }
        Ok(VkCase { vert: words(vert)?, frag: words(frag)?, inputs, ubo })
    }

    /// The case on the local Vulkan through vkrun: its times, and its pixels written to `out` when given
    pub fn mesa(&self, vkrun: &Path, dir: &Path, size: u32, frames: u32, out: Option<&Path>) -> Result<Times, String> {
        let (vert, frag, attrs, ubo) = (dir.join("vert.spv"), dir.join("frag.spv"), dir.join("attrs"), dir.join("ubo"));
        let lines: Vec<String> = self.inputs.iter().map(|(at, n, v)| format!("{at} {n} {}", v.iter().map(f32::to_string).collect::<Vec<_>>().join(" "))).collect();
        let write = |p: &PathBuf, b: &[u8]| std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()));
        write(&vert, &self.vert)?;
        write(&frag, &self.frag)?;
        write(&attrs, (lines.join("\n") + "\n").as_bytes())?;
        let ubo = match self.ubo.is_empty() {
            true => PathBuf::from("-"),
            false => {
                write(&ubo, &self.ubo)?;
                ubo
            }
        };
        let mut cmd = Command::new(vkrun);
        cmd.args([&vert, &frag, &attrs, &ubo]).args([size.to_string(), size.to_string(), frames.to_string()]);
        if let Some(out) = out {
            cmd.arg(out);
        }
        let o = cmd.output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).trim().to_owned());
        }
        let ns: Vec<u64> = String::from_utf8_lossy(&o.stdout).split_whitespace().map(|v| v.parse().map_err(|e| format!("vkrun: {e}"))).collect::<Result<_, _>>()?;
        let [compile, first, mean, min] = ns[..] else { return Err("vkrun: four times expected".into()) };
        Ok(Times { compile: Duration::from_nanos(compile), first: Duration::from_nanos(first), mean: Duration::from_nanos(mean), min: Duration::from_nanos(min) })
    }
}

/// The case set up on detour's Vulkan: a command buffer of clear + the triangle, a fence, the target image
pub struct VkScene {
    pub queue: VkQueue,
    pub cmd: VkCommandBuffer,
    pub fence: VkFence,
    pub image: VkImage,
    pub size: u32,
    /// shader modules + pipeline
    pub compile: Duration,
}

impl VkScene {
    fn host_type(vk: &VkDevice<'_, '_>, bits: u32) -> Result<u32, String> {
        let types = vk.memory_types();
        (0..types.len() as u32).find(|&i| bits & (1 << i) != 0 && types[i as usize].property_flags.contains(HOST)).ok_or_else(|| "no host-visible memory type".into())
    }

    /// A host-visible buffer of `bytes` bytes holding `data` (when given)
    fn host_buffer(vk: &mut VkDevice<'_, '_>, bytes: u64, data: Option<&[u8]>) -> Result<(VkBuffer, VkDeviceMemory), String> {
        let b = vk.create_buffer(bytes);
        let req = vk.buffer_memory_requirements(b).map_err(|e| e.to_string())?;
        let mem = vk.allocate_memory(req.size, VkScene::host_type(vk, req.memory_type_bits)?).map_err(|e| e.to_string())?;
        vk.bind_buffer_memory(b, mem, 0).map_err(|e| e.to_string())?;
        if let Some(data) = data {
            vk.map_memory(mem, 0, Some(bytes)).map_err(|e| e.to_string())?.copy_from_slice(data);
        }
        Ok((b, mem))
    }

    /// The case set up as vkrun sets it up
    pub fn new(vk: &mut VkDevice<'_, '_>, case: &VkCase, size: u32) -> Result<VkScene, String> {
        let e = |e: detour::VkError| e.to_string();
        let queue = vk.queue(0, 0).map_err(e)?;
        let image = vk.create_image(VkImageCreateInfo { format: FORMAT, extent: [size, size] }).map_err(e)?;
        let req = vk.image_memory_requirements(image).map_err(e)?;
        let device_local = (0..vk.memory_types().len() as u32).find(|&i| req.memory_type_bits & (1 << i) != 0 && vk.memory_types()[i as usize].property_flags.device_local).ok_or("no device-local memory")?;
        let imem = vk.allocate_memory(req.size, device_local).map_err(e)?;
        vk.bind_image_memory(image, imem, 0).map_err(e)?;
        let view = vk.create_image_view(image).map_err(e)?;
        let attachment = VkAttachmentDescription { format: FORMAT, load_op: VkAttachmentLoadOp::Clear, store_op: VkAttachmentStoreOp::Store };
        let pass = vk.create_render_pass(&VkRenderPassCreateInfo { attachments: &[attachment], color_attachments: &[0] }).map_err(e)?;
        let fb = vk.create_framebuffer(pass, &[view], [size, size]).map_err(e)?;

        let mut bufs = Vec::new();
        let mut bindings = Vec::new();
        let mut attributes = Vec::new();
        for (i, (at, n, values)) in case.inputs.iter().enumerate() {
            let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
            debug_assert_eq!(values.len(), *n as usize * VERTICES as usize);
            bufs.push(VkScene::host_buffer(vk, bytes.len() as u64, Some(&bytes))?.0);
            let format = VertexFormat::try_from(*n).map_err(|e| e.0)?.into();
            bindings.push(VkVertexInputBindingDescription { binding: i as u32, stride: n * WORD as u32, input_rate: VkVertexInputRate::Vertex });
            attributes.push(VkVertexInputAttributeDescription { location: *at, binding: i as u32, format, offset: 0 });
        }
        let dsl = vk.create_descriptor_set_layout(&[VkDescriptorSetLayoutBinding { binding: UBO_BINDING, descriptor_type: VkDescriptorType::UniformBuffer, descriptor_count: 1 }]);
        let ubo = (!case.ubo.is_empty()).then(|| VkScene::host_buffer(vk, case.ubo.len() as u64, Some(&case.ubo))).transpose()?;
        let sets = if ubo.is_some() { vec![dsl] } else { Vec::new() };
        let layout = vk.create_pipeline_layout(&sets).map_err(e)?;

        let start = Instant::now();
        let vs = vk.create_shader_module(&case.vert).map_err(e)?;
        let fs = vk.create_shader_module(&case.frag).map_err(e)?;
        let stages = [VkPipelineShaderStageCreateInfo { stage: VkShaderStage::Vertex, module: vs, name: ENTRY }, VkPipelineShaderStageCreateInfo { stage: VkShaderStage::Fragment, module: fs, name: ENTRY }];
        let area = VkRect2D { offset: [0, 0], extent: [size, size] };
        let pipeline = vk
            .create_graphics_pipeline(&VkGraphicsPipelineCreateInfo {
                stages: &stages,
                vertex_bindings: &bindings,
                vertex_attributes: &attributes,
                topology: VkPrimitiveTopology::TriangleList,
                viewports: &[VkViewport { x: 0.0, y: 0.0, width: size as f32, height: size as f32, min_depth: 0.0, max_depth: 1.0 }],
                scissors: &[area],
                rasterization: VkRasterization::default(),
                layout,
                render_pass: pass,
                subpass: 0,
            })
            .map_err(e)?;
        let compile = start.elapsed();

        let set = match ubo {
            Some((buffer, _)) => {
                let pool = vk.create_descriptor_pool();
                let set = vk.allocate_descriptor_sets(pool, &[dsl]).map_err(e)?[0];
                let info = VkDescriptorBufferInfo { buffer, offset: 0, range: None };
                vk.update_descriptor_sets(&[VkWriteDescriptorSet { dst_set: set, dst_binding: UBO_BINDING, descriptor_type: VkDescriptorType::UniformBuffer, buffers: &[info] }]).map_err(e)?;
                Some(set)
            }
            None => None,
        };
        let pool = vk.create_command_pool(0).map_err(e)?;
        let cmd = vk.allocate_command_buffers(pool, 1).map_err(e)?[0];
        vk.begin_command_buffer(cmd).map_err(e)?;
        vk.cmd_begin_render_pass(cmd, &VkRenderPassBeginInfo { render_pass: pass, framebuffer: fb, render_area: area, clear_values: &[VkClearValue::Color([0.0; 4])] }).map_err(e)?;
        vk.cmd_bind_pipeline(cmd, pipeline).map_err(e)?;
        if !bufs.is_empty() {
            vk.cmd_bind_vertex_buffers(cmd, 0, &bufs, &vec![0; bufs.len()]).map_err(e)?;
        }
        if let Some(set) = set {
            vk.cmd_bind_descriptor_sets(cmd, layout, 0, &[set]).map_err(e)?;
        }
        vk.cmd_draw(cmd, VERTICES as u32, 1, 0, 0).map_err(e)?;
        vk.cmd_end_render_pass(cmd).map_err(e)?;
        vk.end_command_buffer(cmd).map_err(e)?;
        let fence = vk.create_fence(false);
        Ok(VkScene { queue, cmd, fence, image, size, compile })
    }

    /// One frame: submitted, waited for, the fence reset
    pub fn frame(&self, vk: &mut VkDevice<'_, '_>) -> Result<(), String> {
        let e = |e: detour::VkError| e.to_string();
        vk.queue_submit(self.queue, &[VkSubmitInfo { command_buffers: &[self.cmd] }], Some(self.fence)).map_err(e)?;
        if !vk.wait_for_fences(&[self.fence], true).map_err(e)? {
            return Err("the fence was not signaled".into());
        }
        vk.reset_fences(&[self.fence]).map_err(e)
    }

    /// `frames` frames, timed as vkrun times them
    pub fn run(&self, vk: &mut VkDevice<'_, '_>, frames: u32) -> Result<Times, String> {
        let mut times = Vec::new();
        for _ in 0..frames.max(1) {
            let t = Instant::now();
            self.frame(vk)?;
            times.push(t.elapsed());
        }
        let rest = if times.len() > 1 { &times[1..] } else { &times[..] };
        Ok(Times { compile: self.compile, first: times[0], mean: rest.iter().sum::<Duration>() / rest.len() as u32, min: rest.iter().copied().min().unwrap_or_default() })
    }

    /// The target's texels, copied into a host buffer as vkrun reads them back
    pub fn pixels(&self, vk: &mut VkDevice<'_, '_>) -> Result<Vec<u8>, String> {
        let e = |e: detour::VkError| e.to_string();
        let bytes = u64::from(self.size) * u64::from(self.size) * u64::from(FORMAT.bytes());
        let (buffer, mem) = VkScene::host_buffer(vk, bytes, None)?;
        let pool = vk.create_command_pool(0).map_err(e)?;
        let copy = vk.allocate_command_buffers(pool, 1).map_err(e)?[0];
        vk.begin_command_buffer(copy).map_err(e)?;
        let region = VkBufferImageCopy { buffer_offset: 0, buffer_row_length: 0, buffer_image_height: 0, image_offset: [0, 0], image_extent: [self.size, self.size] };
        vk.cmd_copy_image_to_buffer(copy, self.image, buffer, &[region]).map_err(e)?;
        vk.end_command_buffer(copy).map_err(e)?;
        vk.queue_submit(self.queue, &[VkSubmitInfo { command_buffers: &[copy] }], Some(self.fence)).map_err(e)?;
        vk.wait_for_fences(&[self.fence], true).map_err(e)?;
        vk.reset_fences(&[self.fence]).map_err(e)?;
        Ok(vk.map_memory(mem, 0, Some(bytes)).map_err(e)?.to_vec())
    }
}
