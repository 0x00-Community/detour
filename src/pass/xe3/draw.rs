use alloc::borrow::ToOwned;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use mlirformat::{Attribute, Error, Op};

use core::result::Result;

use crate::*;
use crate::ImageMemType as MemType;



/// One xe3 op with the fields given (the rest left to the genxml defaults)
macro_rules! xe3 {
    ($ty:ident { $($f:ident: $v:expr),* $(,)? }) => {
        $ty { $($f: Some($v.into()),)* ..Default::default() }
    };
}

/// Memory object control state: the index of the write-back, coherent entry
const MOCS: u32 = 1 << 1;
/// The binding table pool: surface states and binding tables, addressed from its base
const POOL: u32 = 0x10_0000;
/// The unit of the pool's size field
const POOL_PAGE: u32 = 4096;
/// The state base addresses' size fields: the whole 32-bit range
const STATE_SIZE: u32 = 0xf_ffff;
/// MI_LOAD_REGISTER_IMM targets and values Mesa's iris sets once per context on Xe2+
const CS_CHICKEN1: u32 = 0x20d8;
const CS_CHICKEN1_VALUE: u32 = 0x0010_0010;
const CACHE_MODE_SS: u32 = 0x6208;
const CACHE_MODE_SS_VALUE: u32 = 0x0020_0000;
/// The compute mode's variable register size allocation, its enable bit in the mask dword
const VRT_MASK: u32 = 1 << 10;
/// The drawing rectangle covers every pixel 16-bit coordinates reach
const DRAWING_MAX: u32 = 0xffff;

/// Bytes the command streamer may read past a stream's end (its prefetch): left mapped after the batch
const CS_PREFETCH: u64 = 4096;
/// URB: the 64-byte units the VS's entries share, starting after the push constants (four stages of 8 KB)
const URB_SPACE: u32 = 5888;
const URB_START: u32 = 4;
const URB_ENTRY_GRANULE: u32 = 8;
const PUSH_STAGES: u32 = 4;
const PUSH_KB: u32 = 8;
const PUSH_ALLOC_SUB_OPCODE: u32 = 18;
/// Threads: the VS's, and the PS's per pixel shader dispatcher
const VS_THREADS: u32 = 545;
const PS_THREADS: u32 = 127;
/// GRF blocks of 32 registers a thread takes (3: 128 registers)
const REGISTERS_PER_THREAD: u32 = 3;
/// The clip's and the SF's provoking vertices: GL's last vertex
const TRIANGLE_PROVOKING: u32 = 2;
const LINE_PROVOKING: u32 = 1;
const POINT_WIDTH_MAX: f32 = 255.875;
const POINT_WIDTH_MIN: f32 = 0.125;
/// The clip test's guardband around the viewport (in NDC)
const GUARDBAND: f32 = 512.0;
/// Every SBE attribute's active components: all four
const COMPONENTS_XYZW: u32 = 3;
const SBE_ATTRIBUTES: usize = 32;
const SBE_SWIZZLES: u32 = 16;
const POINT_WIDTH_FROM_STATE: u32 = 1;
const AA_LINE_DISTANCE_TRUE: u32 = 1;
const POINT_RASTERIZATION_UPPER_RIGHT: u32 = 1;
const LINE_AA_REGION_1_PIXEL: u32 = 1;
const POINT_SPRITE_ORIGIN_LOWER_LEFT: u32 = 1;
/// The WM's barycentric mode: perspective pixel
const BARYCENTRIC_PERSPECTIVE_PIXEL: u32 = 1;
/// Surface state: no swizzle (SCS_RED, SCS_GREEN, SCS_BLUE, SCS_ALPHA)
const SCS: [u32; 4] = [4, 5, 6, 7];
const HALIGN_128B: u32 = 3;
const VALIGN_4: u32 = 1;
const MIP_TAIL_NONE: u32 = 15;
/// The PS's SIMD width field: SIMD16
const PS_SIMD16: u32 = 0;
/// The color calculator's alpha test format and the blend state's alpha test function GL leaves alone
const ALPHA_TEST_FLOAT: u32 = 1;
const ALPHA_TEST_NEVER_USED: u32 = 1;
/// Everything the 3DSTATE_CONSTANT_ALL updates: VS, HS, DS, GS, PS
const CONSTANT_ALL_STAGES: u32 = 0x1f;
/// The VF's batch sizes, as Mesa sets them on Xe2+
const VFG_LIST_1: u32 = 2;
const VFG_LIST_2: u32 = 1;
const VFG_LIST_3: u32 = 2;
const VFG_STRIP: u32 = 3;
const VFG_PATCH: u32 = 1;
const VFG_PATCH_MULTIPLIER: u32 = 31;
/// Placement: surface states and the first binding table entry on 64 bytes, the following entries on 4,
/// viewports on 32 and 64, the fence on a page
const SURFACE_ALIGN: u32 = 64;
const ENTRY_ALIGN: u32 = 4;
const CC_VIEWPORT_ALIGN: u32 = 32;
const STATE_ALIGN: u32 = 64;
const FENCE_ALIGN: u64 = 4096;
const FENCE_SIZE: u64 = 64;
/// Sampler states: their array on 32 bytes, each 16; the border color on 64; the PS's sampler count in fours
const SAMPLER_ALIGN: u32 = 32;
const SAMPLER_ENTRY_ALIGN: u32 = 16;
const BORDER_ALIGN: u32 = 64;
const SAMPLERS_PER_COUNT: u32 = 4;
const SAMPLERS_MAX: u32 = 16;
/// A sampler's LOD range: the one level a texture has
const LOD_MAX: f32 = 0.0;
/// A buffer surface's size, split over its width, height and depth fields
const BUFFER_WIDTH_BITS: u32 = 7;
const BUFFER_HEIGHT_BITS: u32 = 14;
const BUFFER_DEPTH_BITS: u32 = 11;

/// GL's blending as the blend state's factors and functions
struct XeBlend;

impl XeBlend {
    const fn factor(f: BlendFactor) -> u32 {
        match f {
            BlendFactor::Zero => XE3_BLENDFACTOR_ZERO,
            BlendFactor::One => XE3_BLENDFACTOR_ONE,
            BlendFactor::SrcColor => XE3_BLENDFACTOR_SRC_COLOR,
            BlendFactor::OneMinusSrcColor => XE3_BLENDFACTOR_INV_SRC_COLOR,
            BlendFactor::DstColor => XE3_BLENDFACTOR_DST_COLOR,
            BlendFactor::OneMinusDstColor => XE3_BLENDFACTOR_INV_DST_COLOR,
            BlendFactor::SrcAlpha => XE3_BLENDFACTOR_SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => XE3_BLENDFACTOR_INV_SRC_ALPHA,
            BlendFactor::DstAlpha => XE3_BLENDFACTOR_DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => XE3_BLENDFACTOR_INV_DST_ALPHA,
            BlendFactor::ConstantColor => XE3_BLENDFACTOR_CONST_COLOR,
            BlendFactor::OneMinusConstantColor => XE3_BLENDFACTOR_INV_CONST_COLOR,
            BlendFactor::ConstantAlpha => XE3_BLENDFACTOR_CONST_ALPHA,
            BlendFactor::OneMinusConstantAlpha => XE3_BLENDFACTOR_INV_CONST_ALPHA,
            BlendFactor::SrcAlphaSaturate => XE3_BLENDFACTOR_SRC_ALPHA_SATURATE,
        }
    }

    const fn function(e: BlendEquation) -> u32 {
        match e {
            BlendEquation::Add => XE3_BLENDFUNCTION_ADD,
            BlendEquation::Subtract => XE3_BLENDFUNCTION_SUBTRACT,
            BlendEquation::ReverseSubtract => XE3_BLENDFUNCTION_REVERSE_SUBTRACT,
            BlendEquation::Min => XE3_BLENDFUNCTION_MIN,
            BlendEquation::Max => XE3_BLENDFUNCTION_MAX,
        }
    }

    /// Channel `i`'s (0 RGB, 1 alpha) source and destination factors: MIN and MAX ignore GL's factors, as ONE
    const fn factors(b: &Blend, i: usize) -> [u32; 2] {
        match b.equation[i] {
            BlendEquation::Min | BlendEquation::Max => [XE3_BLENDFACTOR_ONE; 2],
            _ => [XeBlend::factor(b.src[i]), XeBlend::factor(b.dst[i])],
        }
    }
}

/// genxml's SURFACE_FORMAT values the draw uses
struct SurfaceFormat;

impl SurfaceFormat {
    const R32G32B32A32_FLOAT: u32 = 0x000;
    const R32G32B32_FLOAT: u32 = 0x040;
    const R32G32_FLOAT: u32 = 0x085;
    const R32_FLOAT: u32 = 0x0d8;
    const R32_UINT: u32 = 0x0d7;
    const R8G8B8A8_UNORM: u32 = 0x0c7;
    const R8_UNORM: u32 = 0x140;
    const RAW: u32 = 0x1ff;

    const fn vertex(f: VertexFormat) -> u32 {
        match f {
            VertexFormat::Float32 => SurfaceFormat::R32_FLOAT,
            VertexFormat::Float32x2 => SurfaceFormat::R32G32_FLOAT,
            VertexFormat::Float32x3 => SurfaceFormat::R32G32B32_FLOAT,
            VertexFormat::Float32x4 => SurfaceFormat::R32G32B32A32_FLOAT,
            VertexFormat::Unorm8x4 => SurfaceFormat::R8G8B8A8_UNORM,
        }
    }

    const fn color(f: ColorFormat) -> u32 {
        match f {
            ColorFormat::Rgba32Float => SurfaceFormat::R32G32B32A32_FLOAT,
            ColorFormat::Rgba8Unorm => SurfaceFormat::R8G8B8A8_UNORM,
            ColorFormat::R8Unorm => SurfaceFormat::R8_UNORM,
        }
    }
}

/// genxml's values of what the draw describes
struct Genxml;

impl Genxml {
    fn primitive(p: DrawPrimitive) -> u32 {
        match p {
            DrawPrimitive::Points => XE3__3DPRIM_POINTLIST,
            DrawPrimitive::Lines => XE3__3DPRIM_LINELIST,
            DrawPrimitive::LineStrip => XE3__3DPRIM_LINESTRIP,
            DrawPrimitive::Triangles => XE3__3DPRIM_TRILIST,
            DrawPrimitive::TriangleStrip => XE3__3DPRIM_TRISTRIP,
        }
    }

    fn filter(f: Filter) -> u32 {
        match f {
            Filter::Nearest => XE3_MAPFILTER_NEAREST,
            Filter::Linear => XE3_MAPFILTER_LINEAR,
        }
    }

    fn wrap(w: Wrap) -> u32 {
        match w {
            Wrap::ClampToEdge => XE3_TCM_CLAMP,
            Wrap::Repeat => XE3_TCM_WRAP,
            Wrap::MirroredRepeat => XE3_TCM_MIRROR,
        }
    }

    fn cull(c: CullMode) -> u32 {
        match c {
            CullMode::None => XE3_CULLMODE_NONE,
            CullMode::Front => XE3_CULLMODE_FRONT,
            CullMode::Back => XE3_CULLMODE_BACK,
        }
    }
}

/// A draw's image, where the second-level batches of its `BATCH` stream start, and the commands a device writes
/// for each draw (`Xe3DrawCommands`)
pub struct Xe3DrawImage {
    pub image: Image,
    pub state: u64,
    pub flush: u64,
    pub commands: Xe3DrawCommands,
    /// the binding table pool (`SURFACES`): its address, and from which offset in it the device may write
    pub pool: u64,
    pub pool_free: u64,
    /// the null surface's address
    pub null: u64,
}

/// What changes from draw to draw of one image, as templates whose holes a device fills in: the primitive (holes
/// `count`, `first`, `instances`), the vertex buffers (`vb<i>`: an address, `vb<i>_size`; no dwords when the draw
/// reads none), a scissor rectangle (`x0`, `y0`, `x1`, `y1`, inclusive) and the pointer to one (`scissor`, below
/// 4 GiB, on `Xe3Draw::SCISSOR_ALIGN`), the color target's surface state (`target`: its address, `pitch`: a row's
/// bytes less one), a uniform block's (`ubo`: its address, `ubo_w`, `ubo_h`, `ubo_d`: its size less one, split as
/// `Xe3Draw::buffer_size` does), a sampled texture's (`tex`: its address, `tex_format`, `tex_tile`: `Xe3Draw::texture`,
/// `tex_w`, `tex_h`, `tex_pitch`: less one), the binding table (`rt`, `null`, `ubo<binding>`, `tex<binding>`: surface
/// states' addresses, on
/// `SURFACE_ALIGN` in the pool) and the pointers to it (`bt`: its offset in the pool; the vertex shader's: no dwords
/// when it reads no buffers), and the pointers to the viewport states (`cc`, `sf_clip`: `Xe3Draw::viewports`, below
/// 4 GiB). The color target's size is in its surface state (`width`, `height`, less one)
pub struct Xe3DrawCommands {
    pub primitive: Template,
    /// the index buffer (`ib`: its address, `ib_size`: its bytes; no dwords when the draw has none)
    pub index_buffer: Template,
    pub vertex_buffers: Template,
    pub scissor_rect: Template,
    pub scissor_pointer: Template,
    pub target_surface: Template,
    pub uniform_surface: Template,
    pub texture_surface: Template,
    pub binding_table: Template,
    pub table_pointers_vs: Template,
    pub table_pointers_ps: Template,
    pub viewport_cc: Template,
    pub viewport_sf_clip: Template,
    /// between a draw into a surface and one sampling it, or the other way round: the render target written back and
    /// the texture cache emptied, once the draws before it are done (no holes)
    pub barrier: Template,
}

/// A draw on Xe3: the program's kernels (in the module, as the shader pipeline leaves them), the state, the commands
/// and the memories, as one image
pub struct Xe3Draw<'b, 'a> {
    body: &'b mut Body<'a>,
    code: BlockId,
    surfaces: BlockId,
    batch: BlockId,
}

impl<'a> Xe3Draw<'_, 'a> {
    /// SPIR-V vertex and fragment shaders (spirv.module ops at the top of the module) to Xe3 kernels
    pub const SHADERS: &'static str = "builtin.module(xevm-attach-target{chip=ptl}, convert-spirv-entry-points, gpu-inline-calls, convert-spirv-cf-to-cf, sroa, mem2reg, gpu-local-arrays, mem2reg, lift-cf-to-scf, canonicalize, cse, convert-gpu-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad)";
    /// A surface state's alignment, and a binding table's
    pub const SURFACE_ALIGN: u64 = SURFACE_ALIGN as u64;
    /// A scissor rectangle's alignment: 64 bytes. genxml's pointer keeps bits 5 and up, but on Xe3 (Panther Lake) a
    /// rectangle 32 bytes past a 64-byte boundary is not the one the draw uses
    pub const SCISSOR_ALIGN: u64 = 64;
    const CODE: &'static str = "code";
    const SURFACES: &'static str = "surfaces";
    /// The stream of the draw: run for every draw
    pub const BATCH: &'static str = "batch";
    /// The stream setting up the context (base addresses, modes): run once on a queue, before its first draw
    pub const INIT: &'static str = "init";
    const FENCE: &'static str = "fence";
    const NULL: &'static str = "null_surface";
    const BT_NULL: &'static str = "bt_null";
    const BT: &'static str = "bt";
    const VS: &'static str = "vs";
    const FS: &'static str = "fs";
    const CC_VIEWPORT: &'static str = "cc_viewport";
    const SF_CLIP_VIEWPORT: &'static str = "sf_clip_viewport";
    const SCISSOR: &'static str = "scissor";
    const COLOR_CALC: &'static str = "color_calc";
    const BLEND: &'static str = "blend";
    const SAMPLERS: &'static str = "samplers";
    const BORDER: &'static str = "border_color";
    const SHADER_MODULE: &'static str = "shaders";

    /// The module's top-level spirv.module ops moved into a gpu.module (what convert-spirv-entry-points takes),
    /// then `SHADERS` run on it
    pub fn compile(body: &mut Body<'a>, passes: &Passes) -> Result<(), Error> {
        let root = body.root;
        let top = body.top_block(root);
        let modules: Vec<OpId> = body.ops(top).filter(|&m| &*body[m].name == "spirv.module").collect();
        body[root].attributes.push(mlirformat::NamedAttr { name: "gpu.container_module".into(), value: None });
        let mut data = OpData::build("gpu.module").with("sym_name", Attribute::str(Xe3Draw::SHADER_MODULE));
        for &m in &modules {
            if let Some((stage, names)) = Xe3Draw::builtins(body, m) {
                data.set(&GpuToXe3::builtins_attr(stage), Attribute::str(&names.join(",")));
            }
        }
        let shaders = body.create(data, &[]);
        let region = body.add_region(shaders);
        let block = body.add_block(region, &[]);
        body.insert(shaders, top, None);
        for m in modules {
            body.detach(m);
            body.insert(m, block, None);
        }
        PassManager::parse(passes, Xe3Draw::SHADERS)?.run(body, root)
    }

    /// A spirv.module's stage and the built-in inputs of its entry point, in the order convert-spirv-entry-points
    /// gives them its function's first arguments (they have no location: they sort first, in interface order)
    fn builtins(body: &Body<'a>, module: OpId) -> Option<(ShaderStage, Vec<String>)> {
        let top = body.top_block(module);
        let entry = body.ops(top).find(|&e| &*body[e].name == "spirv.EntryPoint")?;
        let stage = match body[entry].get("execution_model")?.to_string() {
            m if m.contains("Vertex") => ShaderStage::Vertex,
            m if m.contains("Fragment") => ShaderStage::Fragment,
            _ => return None,
        };
        let Some(Attribute::Array(interface)) = body[entry].get("interface") else { return None };
        let mut names = Vec::new();
        for i in interface {
            let sym = i.to_string();
            let sym = sym.trim_start_matches('@').trim_matches('"');
            let var = body.ops(top).find(|&g| &*body[g].name == "spirv.GlobalVariable" && body[g].get_str("sym_name") == Some(sym))?;
            let input = body[var].get("type").is_some_and(|t| t.to_string().contains(", Input>"));
            if let (true, Some(Attribute::Str { lit, .. })) = (input, body[var].get("built_in")) {
                names.push(lit.to_string());
            }
        }
        (!names.is_empty()).then_some((stage, names))
    }

    /// The image of a draw shaped as `d`, built in `body` (a module `compile` left one vertex and one fragment
    /// kernel in) and laid out from `base` (below 4 GiB): the same for every draw of that program and state,
    /// whatever its target's memory, vertices, uniforms, counts and scissor. Its stream `INIT` sets the context up;
    /// `BATCH` is two second-level batches: the draw's state (from its start: the whole target as the scissor, every
    /// binding table entry the null surface) and the flush (`flush`). A first-level batch runs the state, then for
    /// each of a run of these draws the commands its target, uniforms, vertices, scissor and counts need
    /// (`commands`), then a flush
    pub fn image(body: &'_ mut Body<'a>, d: &Draw<'_>, base: u64) -> Result<Xe3DrawImage, Error> {
        let root = body.root;
        let kernels = Kernels::all(body, root, Xe3Kernel::NAME);
        let code = Xe3Draw::memory(body, &ImageMemory::named(Xe3Draw::CODE, MemType::Streaming))?;
        let batch_memory = Xe3Draw::memory(body, &ImageMemory::named(Xe3Draw::BATCH, MemType::Streaming))?;
        let init = Xe3Draw::container(body, batch_memory, OpData::head(&ImageStream::named(Xe3Draw::INIT))?);
        let batch = Xe3Draw::container(body, batch_memory, OpData::head(&ImageStream::named(Xe3Draw::BATCH))?);
        let pool = ImageMemory { size: Some(u64::from(POOL)), align: Some(u64::from(POOL)), ..ImageMemory::named(Xe3Draw::SURFACES, MemType::Streaming) };
        let surfaces = Xe3Draw::memory(body, &pool)?;
        let (mut vs, mut fs) = (None, None);
        for k in kernels {
            let h = body.head::<Xe3Kernel>(k)?;
            let name = match h.stage {
                Some(ShaderStage::Vertex) if vs.is_none() => Xe3Draw::VS,
                Some(ShaderStage::Fragment) if fs.is_none() => Xe3Draw::FS,
                s => return Err(Error(format!("a draw runs one vertex and one fragment kernel, not {s:?}"))),
            };
            *if name == Xe3Draw::VS { &mut vs } else { &mut fs } = Some(h);
            body[k].set(Symbol::ATTR, Attribute::str(name));
            body.detach(k);
            body.insert(k, code, None);
        }
        let shader_modules: Vec<OpId> = body.ops(body.top_block(root)).filter(|&o| &*body[o].name == "gpu.module").collect();
        for s in shader_modules {
            body.detach(s);
        }
        let (Some(vs), Some(fs)) = (vs, fs) else { return Err(Error("a draw needs a vertex and a fragment kernel".into())) };
        let mut me = Xe3Draw { body, code, surfaces, batch: init };
        me.put(code, &ImageData { align: Some(FENCE_ALIGN), ..ImageData::zeros(Xe3Draw::FENCE, FENCE_SIZE) });
        let null = me.null_surface();
        me.put(batch_memory, &ImageData::zeros("prefetch", CS_PREFETCH));
        me.context();
        me.batch = batch;
        let vs_buffers = me.draw(d, &vs, &fs)?;
        let flush = me.second_level_end();
        me.pc_flush();
        Xe3Batch::end(body, root);
        ImageLayout(&Xe3Image).run_at(body, root, base)?;
        let ops = body.block_ops(batch);
        let flush = 4 * ops[..flush].iter().map(|c| Pack(XE3).template(&body[*c].name, &body[*c].all_attrs()).map_or(0, |t| t.dw.len() as u64)).sum::<u64>();
        let offset = |o: OpId| body[o].get(ImagePlaced::OFFSET).map(Attrs::as_int).transpose().map(|v| v.unwrap_or(0) as u64);
        let null = offset(null)?;
        let mut pool_free = 0;
        for o in body.block_ops(surfaces) {
            pool_free = pool_free.max(offset(o)? + Xe3Draw::SURFACE_ALIGN);
        }
        let image = XE3_IMAGE.emit(body, root)?;
        let start = image.stream_named(Xe3Draw::BATCH).ok_or_else(|| Error(format!("no stream {} in the image", Xe3Draw::BATCH)))?;
        let pool = image.segments.iter().find(|s| s.name == Xe3Draw::SURFACES).map(|s| s.addr).ok_or_else(|| Error(format!("no memory {}", Xe3Draw::SURFACES)))?;
        if pool_free >= u64::from(POOL) / 2 {
            return Err(Error(format!("the binding table pool's static part takes {pool_free} bytes")));
        }
        let commands = Xe3Draw::commands(d, vs_buffers)?;
        Ok(Xe3DrawImage { state: start, flush: start + flush, image, commands, pool, pool_free: u64::from(POOL) / 2, null: pool + null })
    }

    /// An op's dwords, its symbols left as holes
    fn template(op: &impl TypedOp) -> Result<Template, Error> {
        let (data, _) = op.split();
        Pack(XE3).template(&data.name, &data.all_attrs())
    }

    /// The binding table's entries: the color target, each sampled texture after it at its binding, then each uniform
    /// block at its binding from `XE3_BUFFER_BTI` (`None`: the null surface)
    fn entries(d: &Draw<'_>) -> Vec<Option<String>> {
        let mut entries = vec![Some("rt".to_owned())];
        for t in &d.textures {
            let at = 1 + t.binding as usize;
            entries.resize(entries.len().max(at + 1), None);
            entries[at] = Some(Xe3Draw::texture_binding(t.binding));
        }
        for &(binding, _) in &d.uniforms {
            let at = usize::from(XE3_BUFFER_BTI) + binding as usize;
            entries.resize(entries.len().max(at + 1), None);
            entries[at] = Some(Xe3Draw::uniform_buffer(binding));
        }
        entries
    }

    /// A buffer surface's size fields for `size` bytes: its width, height and depth
    pub fn buffer_size(size: u32) -> [u32; 3] {
        let n = size.max(1).next_multiple_of(4) - 1;
        let bits = |shift: u32, width: u32| (n >> shift) & ((1 << width) - 1);
        [bits(0, BUFFER_WIDTH_BITS), bits(BUFFER_WIDTH_BITS, BUFFER_HEIGHT_BITS), bits(BUFFER_WIDTH_BITS + BUFFER_HEIGHT_BITS, BUFFER_DEPTH_BITS)]
    }

    /// The commands of `d`'s draws (`Xe3DrawCommands`)
    fn commands(d: &Draw<'_>, vs_buffers: bool) -> Result<Xe3DrawCommands, Error> {
        let t = &d.target;
        let target_surface = Xe3Draw::template(&xe3!(StateRenderSurfaceState {
            surface_type: XE3_SURFTYPE_2D,
            surface_format: SurfaceFormat::color(t.format),
            surface_horizontal_alignment: HALIGN_128B,
            surface_vertical_alignment: VALIGN_4,
            width: A::sym("width"),
            height: A::sym("height"),
            surface_pitch: A::sym("pitch"),
            tile_mode: if t.tiling == Tiling::Tile4 { XE3_TILE4 } else { 0 },
            mip_tail_start_lod: MIP_TAIL_NONE,
            mocs: MOCS,
            l1_cache_control: XE3_L1CC_WB,
            shader_channel_select_red: SCS[0],
            shader_channel_select_green: SCS[1],
            shader_channel_select_blue: SCS[2],
            shader_channel_select_alpha: SCS[3],
            surface_base_address: A::sym("target"),
        }))?;
        let uniform_surface = Xe3Draw::template(&xe3!(StateRenderSurfaceState {
            surface_type: XE3_SURFTYPE_BUFFER,
            surface_format: SurfaceFormat::RAW,
            surface_horizontal_alignment: HALIGN_128B,
            surface_vertical_alignment: VALIGN_4,
            width: A::sym("ubo_w"),
            height: A::sym("ubo_h"),
            depth: A::sym("ubo_d"),
            surface_pitch: 0u32,
            mocs: MOCS,
            l1_cache_control: XE3_L1CC_WB,
            surface_base_address: A::sym("ubo"),
        }))?;
        let texture_surface = Xe3Draw::template(&xe3!(StateRenderSurfaceState {
            surface_type: XE3_SURFTYPE_2D,
            surface_format: A::sym("tex_format"),
            surface_horizontal_alignment: HALIGN_128B,
            surface_vertical_alignment: VALIGN_4,
            width: A::sym("tex_w"),
            height: A::sym("tex_h"),
            surface_pitch: A::sym("tex_pitch"),
            tile_mode: A::sym("tex_tile"),
            mip_tail_start_lod: MIP_TAIL_NONE,
            mocs: MOCS,
            l1_cache_control: XE3_L1CC_WB,
            shader_channel_select_red: SCS[0],
            shader_channel_select_green: SCS[1],
            shader_channel_select_blue: SCS[2],
            shader_channel_select_alpha: SCS[3],
            surface_base_address: A::sym("tex"),
        }))?;
        let mut binding_table = Template::default();
        for e in Xe3Draw::entries(d) {
            binding_table.extend(&Xe3Draw::template(&xe3!(StateBindingTableState { surface_state_pointer: A::sym(e.as_deref().unwrap_or("null")) }))?);
        }
        let table_pointers_vs = match vs_buffers {
            true => Xe3Draw::template(&xe3!(Cmd3dstateBindingTablePointersVs { pointer_to_vs_binding_table: A::sym("bt") }))?,
            false => Template::default(),
        };
        let table_pointers_ps = Xe3Draw::template(&xe3!(Cmd3dstateBindingTablePointersPs { pointer_to_ps_binding_table: A::sym("bt") }))?;
        let viewport_cc = Xe3Draw::template(&xe3!(Cmd3dstateViewportStatePointersCc { cc_viewport_pointer: A::sym("cc") }))?;
        let viewport_sf_clip = Xe3Draw::template(&xe3!(Cmd3dstateViewportStatePointersSfClip { sf_clip_viewport_pointer: A::sym("sf_clip") }))?;
        let primitive = match d.indices {
            None => Xe3Draw::template(&xe3!(Cmd3dprimitiveExtended {
                vertex_access_type: XE3_SEQUENTIAL,
                vertex_count_per_instance: A::sym("count"),
                instance_count: A::sym("instances"),
                start_vertex_location: A::sym("first"),
                base_vertex_location: 0u32,
                extended_parameters_present: true,
                extended_parameter_0: A::sym("first"),
            }))?,
            Some(_) => Xe3Draw::template(&xe3!(Cmd3dprimitiveExtended {
                vertex_access_type: XE3_RANDOM,
                vertex_count_per_instance: A::sym("count"),
                instance_count: A::sym("instances"),
                start_vertex_location: A::sym("first"),
                base_vertex_location: 0u32,
                extended_parameters_present: true,
                extended_parameter_0: 0u32,
            }))?,
        };
        let index_buffer = match d.indices {
            None => Template::default(),
            Some(i) => Xe3Draw::template(&xe3!(Cmd3dstateIndexBuffer {
                index_format: match i.format {
                    IndexFormat::U16 => XE3_INDEX_WORD,
                    IndexFormat::U32 => XE3_INDEX_DWORD,
                },
                mocs: MOCS,
                l3_bypass_disable: true,
                buffer_starting_address: A::sym("ib"),
                buffer_size: A::sym("ib_size"),
            }))?,
        };
        let vertex_buffers = match d.buffers.is_empty() {
            true => Template::default(),
            false => {
                let vbs: Vec<_> = d.buffers.iter().enumerate().map(|(i, b)| {
                    let state = StateVertexBufferStateFields {
                        vertex_buffer_index: Some((i as u32).into()),
                        buffer_pitch: Some(b.stride.into()),
                        address_modify_enable: Some(true.into()),
                        mocs: Some(MOCS.into()),
                        l3_bypass_disable: Some(true.into()),
                        buffer_starting_address: Some(A::sym(Xe3Draw::vertex_buffer(i)).into()),
                        buffer_size: Some(A::sym(format!("{}_size", Xe3Draw::vertex_buffer(i))).into()),
                        ..Default::default()
                    };
                    Fields(Cmd3dstateVertexBuffersElement { vertex_buffer_state: Some(Fields(state)) })
                }).collect();
                Xe3Draw::template(&xe3!(Cmd3dstateVertexBuffers { elements: vbs }))?
            }
        };
        let scissor_rect = Xe3Draw::template(&xe3!(StateScissorRect {
            scissor_rectangle_x_min: A::sym("x0"),
            scissor_rectangle_y_min: A::sym("y0"),
            scissor_rectangle_x_max: A::sym("x1"),
            scissor_rectangle_y_max: A::sym("y1"),
        }))?;
        let scissor_pointer = Xe3Draw::template(&xe3!(Cmd3dstateScissorStatePointers { scissor_rect_pointer: A::sym("scissor") }))?;
        let barrier = Xe3Draw::template(&xe3!(CmdPipeControl {
            render_target_cache_flush_enable: true,
            texture_cache_invalidation_enable: true,
            command_streamer_stall_enable: true,
        }))?;
        Ok(Xe3DrawCommands {
            primitive,
            index_buffer,
            vertex_buffers,
            scissor_rect,
            scissor_pointer,
            target_surface,
            uniform_surface,
            texture_surface,
            binding_table,
            table_pointers_vs,
            table_pointers_ps,
            viewport_cc,
            viewport_sf_clip,
            barrier,
        })
    }

    /// A draw's scissor as SCISSOR_RECT's inclusive x0, y0, x1, y1 (an empty one: x0 > x1)
    pub fn scissor(d: &Draw<'_>) -> [u32; 4] {
        let [sx, sy, sw, sh] = d.scissor.unwrap_or([0, 0, d.target.width, d.target.height]);
        if sw == 0 || sh == 0 { [1, 1, 0, 0] } else { [sx, sy, sx + sw - 1, sy + sh - 1] }
    }

    /// The batch so far ended as a second-level batch (MI_BATCH_BUFFER_END, a qword's padding): the number of
    /// commands before the next one
    fn second_level_end(&mut self) -> usize {
        self.cmd(&CmdMiBatchBufferEnd::default());
        let dw: usize = self.body.block_ops(self.batch).iter().map(|c| Pack(XE3).template(&self.body[*c].name, &self.body[*c].all_attrs()).map_or(0, |t| t.dw.len())).sum();
        if !dw.is_multiple_of(2) {
            self.cmd(&CmdMiNoop::default());
        }
        self.body.block_ops(self.batch).len()
    }

    fn container(body: &mut Body<'a>, block: BlockId, data: OpData<'a>) -> BlockId {
        let op = body.create(data, &[]);
        let r = body.add_region(op);
        let inner = body.add_block(r, &[]);
        body.insert(op, block, None);
        inner
    }

    fn memory(body: &mut Body<'a>, m: &ImageMemory<'_>) -> Result<BlockId, Error> {
        let top = body.top_block(body.root);
        Ok(Xe3Draw::container(body, top, OpData::head(m)?))
    }

    fn put(&mut self, block: BlockId, op: &impl TypedOp) -> OpId {
        let (data, _) = op.split();
        let o = self.body.create(data, &[]);
        self.body.insert(o, block, None);
        o
    }

    fn cmd(&mut self, op: &impl TypedOp) {
        self.put(self.batch, op);
    }

    /// A state placed under `name`, aligned to `align` bytes, in the pool (`surfaces`) or with the code
    fn state(&mut self, surfaces: bool, name: &str, align: u32, op: &impl TypedOp) -> OpId {
        let o = self.put(if surfaces { self.surfaces } else { self.code }, op);
        self.body[o].set(Symbol::ATTR, Attribute::str(name));
        self.body[o].set(ImagePlaced::ALIGN, A::from(align).into());
        o
    }

    /// Vertex buffer `i`'s hole in `Xe3DrawCommands::vertex_buffers` (its size's: with `_size`)
    pub fn vertex_buffer(i: usize) -> String {
        format!("vb{i}")
    }

    /// Uniform block `binding`'s hole in `Xe3DrawCommands::binding_table`
    pub fn uniform_buffer(binding: u32) -> String {
        format!("ubo{binding}")
    }

    /// Sampled texture `binding`'s hole in `Xe3DrawCommands::binding_table`
    pub fn texture_binding(binding: u32) -> String {
        format!("tex{binding}")
    }

    /// A texture's `tex_format` and `tex_tile` in `Xe3DrawCommands::texture_surface`
    pub const fn texture(format: ColorFormat, tiling: Tiling) -> [u32; 2] {
        [SurfaceFormat::color(format), if matches!(tiling, Tiling::Tile4) { XE3_TILE4 } else { 0 }]
    }

    /// The fragment shader's sampler states (`SAMPLERS`, one per binding up to its last texture's, with the border
    /// color they share) and the pointer to them: the count of fours the PS states
    fn samplers(&mut self, d: &Draw<'_>, bindings: u32) -> Result<u32, Error> {
        if bindings == 0 {
            return Ok(0);
        }
        if bindings > SAMPLERS_MAX {
            return Err(Error(format!("a draw on Xe3 samples at most {SAMPLERS_MAX} bindings, the shader {bindings}")));
        }
        self.state(false, Xe3Draw::BORDER, BORDER_ALIGN, &StateSamplerBorderColorState::default());
        for b in 0..bindings {
            let t = d.textures.iter().find(|t| t.binding == b);
            let (filter, wrap) = t.map_or((Filter::Nearest, Wrap::ClampToEdge), |t| (t.filter, t.wrap));
            let (f, w, round) = (Genxml::filter(filter), Genxml::wrap(wrap), filter == Filter::Linear);
            let state = xe3!(StateSamplerState {
                min_mode_filter: f,
                mag_mode_filter: f,
                mip_mode_filter: XE3_MIPFILTER_NONE,
                lod_preclamp_mode: XE3_CLAMP_MODE_OGL,
                max_lod: LOD_MAX,
                min_lod: LOD_MAX,
                border_color_pointer: A::sym(Xe3Draw::BORDER),
                tcx_address_control_mode: w,
                tcy_address_control_mode: w,
                tcz_address_control_mode: w,
                u_address_min_filter_rounding_enable: round,
                u_address_mag_filter_rounding_enable: round,
                v_address_min_filter_rounding_enable: round,
                v_address_mag_filter_rounding_enable: round,
                r_address_min_filter_rounding_enable: round,
                r_address_mag_filter_rounding_enable: round,
                sampler_disable: t.is_none(),
            });
            match b {
                0 => {
                    self.state(false, Xe3Draw::SAMPLERS, SAMPLER_ALIGN, &state);
                }
                _ => {
                    let o = self.put(self.code, &state);
                    self.body[o].set(ImagePlaced::ALIGN, A::from(SAMPLER_ENTRY_ALIGN).into());
                }
            }
        }
        self.cmd(&xe3!(Cmd3dstateSamplerStatePointersPs { pointer_to_ps_sampler_state: A::sym(Xe3Draw::SAMPLERS) }));
        Ok(bindings.div_ceil(SAMPLERS_PER_COUNT))
    }

    fn pc_flush(&mut self) {
        self.cmd(&xe3!(CmdPipeControl { depth_cache_flush_enable: true, render_target_cache_flush_enable: true, hdc_pipeline_flush_enable: true, command_streamer_stall_enable: true }));
    }

    /// The context Mesa's iris sets up on Xe2+: base addresses 0 (every pointer an address), the fence, the modes
    fn context(&mut self) {
        self.pc_flush();
        self.cmd(&xe3!(CmdStateBaseAddress {
            general_state_base_address_modify_enable: true,
            general_state_mocs: MOCS,
            general_state_buffer_size_modify_enable: true,
            general_state_buffer_size: STATE_SIZE,
            stateless_data_port_access_mocs: MOCS,
            l1_cache_control: XE3_L1CC_WB,
            surface_state_base_address_modify_enable: true,
            surface_state_mocs: MOCS,
            dynamic_state_base_address_modify_enable: true,
            dynamic_state_mocs: MOCS,
            dynamic_state_buffer_size_modify_enable: true,
            dynamic_state_buffer_size: STATE_SIZE,
            indirect_object_base_address_modify_enable: true,
            indirect_object_mocs: MOCS,
            indirect_object_buffer_size_modify_enable: true,
            indirect_object_buffer_size: STATE_SIZE,
            instruction_base_address_modify_enable: true,
            instruction_mocs: MOCS,
            instruction_buffer_size_modify_enable: true,
            instruction_buffer_size: STATE_SIZE,
            bindless_surface_state_mocs: MOCS,
            bindless_sampler_state_mocs: MOCS,
        }));
        self.cmd(&xe3!(CmdPipeControl {
            state_cache_invalidation_enable: true,
            constant_cache_invalidation_enable: true,
            texture_cache_invalidation_enable: true,
            instruction_cache_invalidate_enable: true,
            command_streamer_stall_enable: true,
        }));
        self.cmd(&xe3!(CmdStateSystemMemFenceAddress { system_memory_fence_address: A::sym(Xe3Draw::FENCE) }));
        self.cmd(&xe3!(CmdMiLoadRegisterImm { register_offset: CS_CHICKEN1, data_dword: CS_CHICKEN1_VALUE }));
        self.cmd(&xe3!(CmdMiLoadRegisterImm { register_offset: CACHE_MODE_SS, data_dword: CACHE_MODE_SS_VALUE }));
        self.cmd(&xe3!(Cmd3dstate3dMode { dx10_ogl_border_mode_for_ycrcb: true, dx10_ogl_border_mode_for_ycrcb_mask: true }));
        self.cmd(&xe3!(CmdStateComputeMode { enable_variable_register_size_allocation: true, mask1: VRT_MASK, enable_variable_register_size_allocation_mask: 1u32 }));
        self.cmd(&xe3!(Cmd3dstateDrawingRectangleFast { clipped_drawing_rectangle_x_max: DRAWING_MAX, clipped_drawing_rectangle_y_max: DRAWING_MAX }));
        self.cmd(&xe3!(Cmd3dstateSamplePattern { _1x_sample0_x_offset: 0.5f32, _1x_sample0_y_offset: 0.5f32 }));
        self.cmd(&Cmd3dstateAaLineParameters::default());
        self.cmd(&Cmd3dstateWmChromakey::default());
        self.cmd(&Cmd3dstateWmHzOp::default());
        self.cmd(&Cmd3dstatePolyStippleOffset::default());
        self.cmd(&Cmd3dstateMeshControl::default());
        self.cmd(&Cmd3dstateTaskControl::default());
        for stage in 0..PUSH_STAGES {
            self.cmd(&xe3!(Cmd3dstatePushConstantAllocDs { _3d_command_sub_opcode: PUSH_ALLOC_SUB_OPCODE + stage, constant_buffer_offset: stage * PUSH_KB, constant_buffer_size: PUSH_KB }));
        }
        self.pc_flush();
    }

    /// The null surface, and a binding table of it alone: the null surface's op
    fn null_surface(&mut self) -> OpId {
        let null = self.state(true, Xe3Draw::NULL, SURFACE_ALIGN, &xe3!(StateRenderSurfaceState { surface_type: XE3_SURFTYPE_NULL, surface_format: SurfaceFormat::R32_UINT, mocs: MOCS }));
        self.state(true, Xe3Draw::BT_NULL, SURFACE_ALIGN, &xe3!(StateBindingTableState { surface_state_pointer: A::sym(Xe3Draw::NULL) }));
        null
    }

    /// A binding table of `entries` entries, each the null surface (a draw's commands bind its own)
    fn binding_table(&mut self, entries: usize) {
        for i in 0..entries {
            let s = xe3!(StateBindingTableState { surface_state_pointer: A::sym(Xe3Draw::NULL) });
            if i == 0 {
                self.state(true, Xe3Draw::BT, SURFACE_ALIGN, &s);
            } else {
                let o = self.put(self.surfaces, &s);
                self.body[o].set(ImagePlaced::ALIGN, A::from(ENTRY_ALIGN).into());
            }
        }
    }

    /// A draw's viewport states: CC_VIEWPORT and SF_CLIP_VIEWPORT
    fn viewport_states(d: &Draw<'_>) -> (StateCcViewport, StateSfClipViewport) {
        let [x, y, w, h] = d.viewport;
        let [near, far] = d.depth_range;
        let (tw, th) = (d.target.width as f32, d.target.height as f32);
        let cc = xe3!(StateCcViewport { minimum_depth: near.min(far), maximum_depth: near.max(far) });
        let sf_clip = xe3!(StateSfClipViewport {
            viewport_matrix_element_m00: w / 2.0,
            viewport_matrix_element_m11: h / 2.0,
            viewport_matrix_element_m22: far - near,
            viewport_matrix_element_m30: x + w / 2.0,
            viewport_matrix_element_m31: y + h / 2.0,
            viewport_matrix_element_m32: near,
            x_min_clip_guardband: -GUARDBAND,
            x_max_clip_guardband: GUARDBAND,
            y_min_clip_guardband: -GUARDBAND,
            y_max_clip_guardband: GUARDBAND,
            x_min_viewport: x.max(0.0),
            x_max_viewport: (x + w).min(tw) - 1.0,
            y_min_viewport: y.max(0.0),
            y_max_viewport: (y + h).min(th) - 1.0,
        });
        (cc, sf_clip)
    }

    /// A draw's viewport states' dwords, for `Xe3DrawCommands::viewport_cc` (on 32 bytes) and `viewport_sf_clip` (on
    /// 64)
    pub fn viewports(d: &Draw<'_>) -> Result<[Vec<u32>; 2], Error> {
        let (cc, sf_clip) = Xe3Draw::viewport_states(d);
        Ok([Xe3Draw::template(&cc)?.dw, Xe3Draw::template(&sf_clip)?.dw])
    }

    fn viewport(&mut self, d: &Draw<'_>) {
        let (cc, sf_clip) = Xe3Draw::viewport_states(d);
        self.state(false, Xe3Draw::CC_VIEWPORT, CC_VIEWPORT_ALIGN, &cc);
        self.state(false, Xe3Draw::SF_CLIP_VIEWPORT, STATE_ALIGN, &sf_clip);
        let (x0, y0, x1, y1) = (0, 0, d.target.width - 1, d.target.height - 1);
        self.state(false, Xe3Draw::SCISSOR, STATE_ALIGN, &xe3!(StateScissorRect { scissor_rectangle_x_min: x0, scissor_rectangle_y_min: y0, scissor_rectangle_x_max: x1, scissor_rectangle_y_max: y1 }));
    }

    /// The draw's state: whether the vertex shader reads buffers (and its binding table is the draw's)
    fn draw(&mut self, d: &Draw<'_>, vh: &Xe3Kernel<'_>, fh: &Xe3Kernel<'_>) -> Result<bool, Error> {
        let inputs = vh.inputs.unwrap_or(0);
        let system = vh.system.unwrap_or(0);
        let fetched = inputs + u32::from(system != 0);
        let varyings = vh.varyings.unwrap_or(0);
        if inputs as usize != d.elements.len() {
            return Err(Error(format!("the vertex shader reads {inputs} inputs, the draw gives {}", d.elements.len())));
        }
        if fh.targets.unwrap_or(1) != 1 {
            return Err(Error("a draw on Xe3: one color target".into()));
        }
        let bindings = fh.bindings.unwrap_or(0);
        if bindings > 0 && d.textures.is_empty() {
            return Err(Error(format!("the fragment shader samples bindings below {bindings}, the draw gives no texture")));
        }
        self.viewport(d);
        let blend = d.blend.unwrap_or_default();
        let [red, green, blue, alpha] = blend.color;
        self.state(false, Xe3Draw::COLOR_CALC, STATE_ALIGN, &xe3!(StateColorCalcState {
            alpha_test_format: ALPHA_TEST_FLOAT,
            blend_constant_color_red: red,
            blend_constant_color_green: green,
            blend_constant_color_blue: blue,
            blend_constant_color_alpha: alpha,
        }));
        let independent = blend.src[0] != blend.src[1] || blend.dst[0] != blend.dst[1] || blend.equation[0] != blend.equation[1];
        self.state(false, Xe3Draw::BLEND, STATE_ALIGN, &xe3!(StateBlendState {
            alpha_test_function: ALPHA_TEST_NEVER_USED,
            independent_alpha_blend_enable: independent,
        }));
        let [rgb, alpha] = [XeBlend::factors(&blend, 0), XeBlend::factors(&blend, 1)];
        let [r, g, b, a] = d.color_mask;
        let entry = self.put(self.code, &xe3!(StateBlendStateEntry {
            write_disable_red: !r,
            write_disable_green: !g,
            write_disable_blue: !b,
            write_disable_alpha: !a,
            color_buffer_blend_enable: d.blend.is_some(),
            source_blend_factor: rgb[0],
            destination_blend_factor: rgb[1],
            color_blend_function: XeBlend::function(blend.equation[0]),
            source_alpha_blend_factor: alpha[0],
            destination_alpha_blend_factor: alpha[1],
            alpha_blend_function: XeBlend::function(blend.equation[1]),
            post_blend_color_clamp_enable: true,
            pre_blend_color_clamp_enable: true,
            color_clamp_range: XE3_COLORCLAMP_RTFORMAT,
            simple_float_blend_enable: true,
        }));
        self.body[entry].set(ImagePlaced::ALIGN, A::from(ENTRY_ALIGN).into());

        let entries = Xe3Draw::entries(d).len();
        let table = entries as u32;
        self.binding_table(entries);
        let bt = || A::within(Xe3Draw::SURFACES, Xe3Draw::BT);
        let null = || A::within(Xe3Draw::SURFACES, Xe3Draw::BT_NULL);
        let vs_buffers = vh.buffers.unwrap_or(0) > 0;

        self.cmd(&xe3!(Cmd3dstateViewportStatePointersCc { cc_viewport_pointer: A::sym(Xe3Draw::CC_VIEWPORT) }));
        self.cmd(&xe3!(Cmd3dstateViewportStatePointersSfClip { sf_clip_viewport_pointer: A::sym(Xe3Draw::SF_CLIP_VIEWPORT) }));
        self.cmd(&xe3!(Cmd3dstateScissorStatePointers { scissor_rect_pointer: A::sym(Xe3Draw::SCISSOR) }));
        let units = fetched.max(2 + varyings).div_ceil(4).max(1);
        let entries_urb = URB_SPACE / units / URB_ENTRY_GRANULE * URB_ENTRY_GRANULE;
        self.cmd(&xe3!(Cmd3dstateUrbAllocVs {
            vs_urb_entry_allocation_size: units - 1,
            vs_urb_starting_address_slice0: URB_START,
            vs_urb_starting_address_slice_n: URB_START,
            vs_number_of_urb_entries_slice0: entries_urb,
            vs_number_of_urb_entries_slice_n: entries_urb,
        }));
        self.cmd(&xe3!(Cmd3dstateUrbAllocHs { hs_urb_starting_address_slice0: URB_START, hs_urb_starting_address_slice_n: URB_START }));
        self.cmd(&xe3!(Cmd3dstateUrbAllocDs { ds_urb_starting_address_slice0: URB_START, ds_urb_starting_address_slice_n: URB_START }));
        self.cmd(&xe3!(Cmd3dstateUrbAllocGs { gs_urb_starting_address_slice0: URB_START, gs_urb_starting_address_slice_n: URB_START }));
        self.cmd(&xe3!(Cmd3dstateBlendStatePointers { blend_state_pointer_valid: true, blend_state_pointer: A::sym(Xe3Draw::BLEND) }));
        self.cmd(&xe3!(Cmd3dstateCcStatePointers { color_calc_state_pointer_valid: true, color_calc_state_pointer: A::sym(Xe3Draw::COLOR_CALC) }));
        self.cmd(&xe3!(Cmd3dstateConstantAll { shader_update_enable: CONSTANT_ALL_STAGES, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePoolAlloc { binding_table_pool_base_address: A::sym(Xe3Draw::SURFACES), binding_table_pool_buffer_size: POOL / POOL_PAGE, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersVs { pointer_to_vs_binding_table: if vs_buffers { bt() } else { null() } }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersHs { pointer_to_hs_binding_table: null() }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersDs { pointer_to_ds_binding_table: null() }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersGs { pointer_to_gs_binding_table: null() }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersPs { pointer_to_ps_binding_table: bt() }));
        let samplers = self.samplers(d, bindings)?;
        self.cmd(&xe3!(CmdPipeControl { stall_at_pixel_scoreboard: true, render_target_cache_flush_enable: true }));
        self.cmd(&xe3!(Cmd3dstateMultisample { number_of_multisamples: 0u32 }));
        self.cmd(&xe3!(Cmd3dstateSampleMask { sample_mask: 1u32 }));
        self.cmd(&xe3!(Cmd3dstateVs {
            kernel_start_pointer: A::sym(Xe3Draw::VS),
            registers_per_thread: REGISTERS_PER_THREAD,
            vertex_urb_entry_read_length: fetched.div_ceil(2).max(1),
            binding_table_entry_count: if vs_buffers { table } else { 0 },
            dispatch_grf_start_register_for_urb_data: u32::from(XE3_VS_ATTR_GRF),
            enable: true,
            statistics_enable: true,
            maximum_number_of_threads: VS_THREADS,
        }));
        self.cmd(&Cmd3dstateHs::default());
        self.cmd(&Cmd3dstateTe::default());
        self.cmd(&Cmd3dstateDs::default());
        self.cmd(&Cmd3dstateGs::default());
        let imask = fh.imask.unwrap_or(false);
        let fragment_inputs = fh.inputs.unwrap_or(0);
        let setup = if fragment_inputs > 0 { Interpolate::setup(imask) } else { 1 + u8::from(imask) };
        self.cmd(&xe3!(Cmd3dstatePs {
            kernel_0_enable: true,
            kernel_0_simd_width: PS_SIMD16,
            kernel_start_pointer_0: A::sym(Xe3Draw::FS),
            registers_per_thread: REGISTERS_PER_THREAD,
            binding_table_entry_count: table,
            sampler_count: samplers,
            maximum_number_of_threads_per_psd: PS_THREADS,
            dispatch_grf_start_register_for_constant_setup_data_0: u32::from(setup),
            kernel_0_poly_packing_policy: XE3_POLY_PACK16_FIXED,
            vector_mask_enable: false,
        }));
        self.cmd(&xe3!(Cmd3dstatePsExtra { pixel_shader_valid: true, omask_present_to_render_target: fh.omask.unwrap_or(false), input_coverage_mask_state: u32::from(imask) }));
        self.cmd(&Cmd3dstateStreamout::default());
        self.cmd(&xe3!(Cmd3dstateClip {
            api_mode: XE3_APIMODE_D3D,
            statistics_enable: true,
            force_user_clip_distance_clip_test_enable_bitmask: true,
            early_cull_enable: true,
            triangle_fan_provoking_vertex_select: TRIANGLE_PROVOKING,
            line_strip_list_provoking_vertex_select: LINE_PROVOKING,
            triangle_strip_list_provoking_vertex_select: TRIANGLE_PROVOKING,
            guardband_clip_test_enable: true,
            viewport_xy_clip_test_enable: true,
            clip_enable: true,
            force_zero_rta_index_enable: true,
            maximum_point_width: POINT_WIDTH_MAX,
            minimum_point_width: POINT_WIDTH_MIN,
        }));
        self.cmd(&xe3!(Cmd3dstateRaster {
            viewport_z_near_clip_test_enable: true,
            scissor_rectangle_enable: true,
            front_winding: u32::from(!d.front_ccw),
            cull_mode: Genxml::cull(d.cull),
            viewport_z_far_clip_test_enable: true,
        }));
        self.cmd(&xe3!(Cmd3dstateSf {
            viewport_transform_enable: true,
            statistics_enable: true,
            line_width: 1.0f32,
            point_width: 1.0f32,
            point_width_source: POINT_WIDTH_FROM_STATE,
            aa_line_distance_mode: AA_LINE_DISTANCE_TRUE,
            triangle_fan_provoking_vertex_select: TRIANGLE_PROVOKING,
            line_strip_list_provoking_vertex_select: LINE_PROVOKING,
            triangle_strip_list_provoking_vertex_select: TRIANGLE_PROVOKING,
        }));
        let bary = if fragment_inputs == 0 { 0 } else { BARYCENTRIC_PERSPECTIVE_PIXEL };
        self.cmd(&xe3!(Cmd3dstateWm { point_rasterization_rule: POINT_RASTERIZATION_UPPER_RIGHT, line_antialiasing_region_width: LINE_AA_REGION_1_PIXEL, statistics_enable: true, barycentric_interpolation_mode: bary }));
        self.cmd(&xe3!(Cmd3dstateSbe {
            number_of_sf_output_attributes: varyings,
            vertex_urb_entry_read_offset: u32::from(varyings > 0),
            vertex_urb_entry_read_length: varyings.div_ceil(2).max(1),
            point_sprite_texture_coordinate_origin: POINT_SPRITE_ORIGIN_LOWER_LEFT,
            attribute_swizzle_enable: true,
            force_vertex_urb_entry_read_offset: true,
            force_vertex_urb_entry_read_length: true,
            attribute_active_component_format: vec![Fill::from(COMPONENTS_XYZW); SBE_ATTRIBUTES],
        }));
        let swiz: Vec<_> = (0..SBE_SWIZZLES).map(|i| Fields(StateSfOutputAttributeDetailFields { source_attribute: (i < varyings).then(|| i.into()), ..Default::default() })).collect();
        self.cmd(&xe3!(Cmd3dstateSbeSwiz { attribute: swiz }));
        self.cmd(&xe3!(Cmd3dstatePsBlend {
            has_writeable_rt: d.color_mask.iter().any(|c| *c),
            color_buffer_blend_enable: d.blend.is_some(),
            independent_alpha_blend_enable: independent,
            source_blend_factor: rgb[0],
            destination_blend_factor: rgb[1],
            source_alpha_blend_factor: alpha[0],
            destination_alpha_blend_factor: alpha[1],
        }));
        self.cmd(&Cmd3dstateWmDepthStencil::default());
        self.cmd(&Cmd3dstateDepthBounds::default());
        self.cmd(&xe3!(Cmd3dstateDepthBuffer { surface_type: XE3_SURFTYPE_NULL, surface_format: XE3_D32_FLOAT, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateStencilBuffer { surface_type: XE3_SURFTYPE_NULL, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateHierDepthBuffer { mocs: MOCS }));
        self.cmd(&xe3!(CmdPipeControl { command_streamer_stall_enable: true }));

        self.cmd(&xe3!(Cmd3dstateVfTopology { primitive_topology_type: Genxml::primitive(d.primitive) }));
        let mut elements: Vec<_> = d.elements.iter().map(|e| {
            let n = e.format.components();
            let control = |c: u32| Some(Fill::from(if c < n { XE3_VFCOMP_STORE_SRC } else if c < 3 { XE3_VFCOMP_STORE_0 } else { XE3_VFCOMP_STORE_1_FP }));
            let state = StateVertexElementStateFields {
                vertex_buffer_index: Some(e.buffer.into()),
                source_element_format: Some(SurfaceFormat::vertex(e.format).into()),
                source_element_offset: Some(e.offset.into()),
                valid: Some(true.into()),
                component_0_control: control(0),
                component_1_control: control(1),
                component_2_control: control(2),
                component_3_control: control(3),
                ..Default::default()
            };
            Fields(Cmd3dstateVertexElementsElement { element: Some(Fields(state)) })
        }).collect();
        if system != 0 {
            // the input after the attributes: zeros the vertex fetcher writes the vertex and instance index into
            let zero = || Some(Fill::from(XE3_VFCOMP_STORE_0));
            let state = StateVertexElementStateFields {
                vertex_buffer_index: Some(0u32.into()),
                source_element_format: Some(SurfaceFormat::R32G32B32A32_FLOAT.into()),
                valid: Some(true.into()),
                component_0_control: zero(),
                component_1_control: zero(),
                component_2_control: zero(),
                component_3_control: zero(),
                ..Default::default()
            };
            elements.push(Fields(Cmd3dstateVertexElementsElement { element: Some(Fields(state)) }));
        }
        self.cmd(&xe3!(Cmd3dstateVertexElements { elements: elements }));
        for i in 0..fetched {
            self.cmd(&xe3!(Cmd3dstateVfInstancing { vertex_element_index: i, instancing_enable: false, instance_data_step_rate: 0u32 }));
        }
        self.cmd(&xe3!(Cmd3dstateVfSgvs {
            vertexid_enable: system & 1 != 0,
            vertexid_component_number: 0u32,
            vertexid_element_offset: inputs,
            instanceid_enable: system & 2 != 0,
            instanceid_component_number: 1u32,
            instanceid_element_offset: inputs,
        }));
        // the vertex fetcher's VertexID counts from 0: the vertex index adds the first vertex (extended parameter 0)
        self.cmd(&xe3!(Cmd3dstateVfSgvs2 { xp0_enable: system & 1 != 0, xp0_source_select: XE3_XP0_PARAMETER, xp0_component_number: 2u32, xp0_element_offset: inputs }));
        self.cmd(&xe3!(Cmd3dstateVfStatistics { statistics_enable: true }));
        self.cmd(&xe3!(Cmd3dstateVf { geometry_distribution_enable: true }));
        self.cmd(&xe3!(Cmd3dstateVfg {
            list_1_batch_size_scale: VFG_LIST_1,
            list_2_batch_size_scale: VFG_LIST_2,
            list_3_batch_size_scale: VFG_LIST_3,
            strip_batch_size_scale: VFG_STRIP,
            patch_batch_size_scale: VFG_PATCH,
            patch_batch_size_multiplier: VFG_PATCH_MULTIPLIER,
        }));
        self.cmd(&Cmd3dstatePrimitiveReplication::default());
        Ok(vs_buffers)
    }
}
