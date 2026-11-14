use mlirformat::{Attribute, Const, Error, Op};
use mlirpass::{BlockId, Body, OpData, OpId, Quiet, Rewriter};

use crate::base::{AddressMode, Fields, Fill, SamplerState, BlendFactor, BlendOperation, BlendState, CullMode, FrontFace, Data, StencilOperation, Heads, MemType, Memory, OpHead, Stream, CompareFunction, DepthState, FilterMode, IndexFormat, Kernel, MipmapFilter, StepMode, Symbol, LoadOp, Stage, TextureFormat, Topology, VertexFormat};
use crate::xe3_dialect::{
    APIMODE_D3D, COLORCLAMP_RTFORMAT, COMPAREFUNCTION_ALWAYS, COMPAREFUNCTION_EQUAL, COMPAREFUNCTION_GEQUAL, COMPAREFUNCTION_GREATER,
    PREFILTEROP_ALWAYS, PREFILTEROP_EQUAL, PREFILTEROP_GEQUAL, PREFILTEROP_GREATER, PREFILTEROP_LEQUAL, PREFILTEROP_LESS, PREFILTEROP_NEVER, PREFILTEROP_NOTEQUAL, SURFTYPE_3D, SURFTYPE_CUBE, COMPAREFUNCTION_LEQUAL, COMPAREFUNCTION_LESS, MIPFILTER_LINEAR, MIPFILTER_NEAREST, MIPFILTER_NONE, COMPAREFUNCTION_NEVER, COMPAREFUNCTION_NOTEQUAL, CULLMODE_BACK, CULLMODE_FRONT, CULLMODE_NONE, D32_FLOAT, INDEX_DWORD, INDEX_WORD, POLY_PACK16_FIXED, RANDOM, SEQUENTIAL, SURFTYPE_2D, SURFTYPE_BUFFER, SURFTYPE_NULL, TILE4, VFCOMP_STORE_0, VFCOMP_STORE_1_FP, VFCOMP_STORE_SRC, _3DPRIM_LINELIST,
    _3DPRIM_LINESTRIP, _3DPRIM_POINTLIST, _3DPRIM_TRILIST, _3DPRIM_TRISTRIP,
};

use crate::xe3_codegen::{A, MOCS, WithAttrs};
use crate::xe3;
use crate::xe3_dialect::{
    BLENDFACTOR_CONST_COLOR, BLENDFACTOR_DST_ALPHA, BLENDFACTOR_DST_COLOR, BLENDFACTOR_INV_CONST_COLOR, BLENDFACTOR_INV_DST_ALPHA, BLENDFACTOR_INV_DST_COLOR, BLENDFACTOR_INV_SRC_ALPHA,
    BLENDFACTOR_INV_SRC_COLOR, BLENDFACTOR_ONE, BLENDFACTOR_SRC_ALPHA, BLENDFACTOR_SRC_ALPHA_SATURATE, BLENDFACTOR_SRC_COLOR, BLENDFACTOR_ZERO, BLENDFUNCTION_ADD, BLENDFUNCTION_MAX,
    BLENDFUNCTION_MIN, BLENDFUNCTION_REVERSE_SUBTRACT, BLENDFUNCTION_SUBTRACT, L1CC_WB,
};
use crate::xe3_dialect::{STENCILOP_DECR, STENCILOP_DECRSAT, STENCILOP_INCR, STENCILOP_INCRSAT, STENCILOP_INVERT, STENCILOP_KEEP, STENCILOP_REPLACE, STENCILOP_ZERO};

struct SurfaceFormat;

impl SurfaceFormat {
    const BC1_UNORM: u32 = 390;
    const BC3_UNORM: u32 = 392;
    const BC7_UNORM: u32 = 418;
    const R32G32B32A32_FLOAT: u32 = 0x000;
    const R32G32B32_FLOAT: u32 = 0x040;
    const R32G32_FLOAT: u32 = 0x085;
    const R32_FLOAT: u32 = 0x0d8;
    const R32_UINT: u32 = 0x0d7;
    const R16G16B16A16_FLOAT: u32 = 0x084;
    const B8G8R8A8_UNORM: u32 = 0x0c0;
    const B8G8R8A8_UNORM_SRGB: u32 = 0x0c1;
    const R8G8B8A8_UNORM: u32 = 0x0c7;
    const R8G8B8A8_UNORM_SRGB: u32 = 0x0c8;
    const R16G16_FLOAT: u32 = 0x0d0;
    const R8G8_UNORM: u32 = 0x106;
    const R16_FLOAT: u32 = 0x10e;
    const R8_UNORM: u32 = 0x140;
    const RAW: u32 = 0x1ff;
}

pub trait Isl {
    fn isl(self) -> u32;
}

impl Isl for VertexFormat {
    fn isl(self) -> u32 {
        match self {
            VertexFormat::Float32 => SurfaceFormat::R32_FLOAT,
            VertexFormat::Float32x2 => SurfaceFormat::R32G32_FLOAT,
            VertexFormat::Float32x3 => SurfaceFormat::R32G32B32_FLOAT,
            VertexFormat::Float32x4 => SurfaceFormat::R32G32B32A32_FLOAT,
        }
    }
}

impl Isl for TextureFormat {
    fn isl(self) -> u32 {
        match self {
            TextureFormat::Rgba8Unorm => SurfaceFormat::R8G8B8A8_UNORM,
            TextureFormat::Rgba8UnormSrgb => SurfaceFormat::R8G8B8A8_UNORM_SRGB,
            TextureFormat::Bgra8Unorm => SurfaceFormat::B8G8R8A8_UNORM,
            TextureFormat::Bgra8UnormSrgb => SurfaceFormat::B8G8R8A8_UNORM_SRGB,
            TextureFormat::R8Unorm => SurfaceFormat::R8_UNORM,
            TextureFormat::Rg8Unorm => SurfaceFormat::R8G8_UNORM,
            TextureFormat::R16Float => SurfaceFormat::R16_FLOAT,
            TextureFormat::Rg16Float => SurfaceFormat::R16G16_FLOAT,
            TextureFormat::Rgba16Float => SurfaceFormat::R16G16B16A16_FLOAT,
            TextureFormat::R32Float => SurfaceFormat::R32_FLOAT,
            TextureFormat::Rg32Float => SurfaceFormat::R32G32_FLOAT,
            TextureFormat::Rgba32Float => SurfaceFormat::R32G32B32A32_FLOAT,
            TextureFormat::Bc1RgbaUnorm => SurfaceFormat::BC1_UNORM,
            TextureFormat::Bc3RgbaUnorm => SurfaceFormat::BC3_UNORM,
            TextureFormat::Bc7RgbaUnorm => SurfaceFormat::BC7_UNORM,
        }
    }
}

impl Isl for BlendFactor {
    fn isl(self) -> u32 {
        match self {
            BlendFactor::Zero => BLENDFACTOR_ZERO,
            BlendFactor::One => BLENDFACTOR_ONE,
            BlendFactor::Src => BLENDFACTOR_SRC_COLOR,
            BlendFactor::OneMinusSrc => BLENDFACTOR_INV_SRC_COLOR,
            BlendFactor::SrcAlpha => BLENDFACTOR_SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => BLENDFACTOR_INV_SRC_ALPHA,
            BlendFactor::Dst => BLENDFACTOR_DST_COLOR,
            BlendFactor::OneMinusDst => BLENDFACTOR_INV_DST_COLOR,
            BlendFactor::DstAlpha => BLENDFACTOR_DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => BLENDFACTOR_INV_DST_ALPHA,
            BlendFactor::SrcAlphaSaturated => BLENDFACTOR_SRC_ALPHA_SATURATE,
            BlendFactor::Constant => BLENDFACTOR_CONST_COLOR,
            BlendFactor::OneMinusConstant => BLENDFACTOR_INV_CONST_COLOR,
            BlendFactor::Src1 => 9,
            BlendFactor::Src1Alpha => 10,
            BlendFactor::OneMinusSrc1 => 25,
            BlendFactor::OneMinusSrc1Alpha => 26,
        }
    }
}

impl Isl for BlendOperation {
    fn isl(self) -> u32 {
        match self {
            BlendOperation::Add => BLENDFUNCTION_ADD,
            BlendOperation::Subtract => BLENDFUNCTION_SUBTRACT,
            BlendOperation::ReverseSubtract => BLENDFUNCTION_REVERSE_SUBTRACT,
            BlendOperation::Min => BLENDFUNCTION_MIN,
            BlendOperation::Max => BLENDFUNCTION_MAX,
        }
    }
}

impl Isl for StencilOperation {
    fn isl(self) -> u32 {
        match self {
            StencilOperation::Keep => STENCILOP_KEEP,
            StencilOperation::Zero => STENCILOP_ZERO,
            StencilOperation::Replace => STENCILOP_REPLACE,
            StencilOperation::Invert => STENCILOP_INVERT,
            StencilOperation::IncrementClamp => STENCILOP_INCRSAT,
            StencilOperation::DecrementClamp => STENCILOP_DECRSAT,
            StencilOperation::IncrementWrap => STENCILOP_INCR,
            StencilOperation::DecrementWrap => STENCILOP_DECR,
        }
    }
}

impl Isl for CompareFunction {
    fn isl(self) -> u32 {
        match self {
            CompareFunction::Never => COMPAREFUNCTION_NEVER,
            CompareFunction::Less => COMPAREFUNCTION_LESS,
            CompareFunction::Equal => COMPAREFUNCTION_EQUAL,
            CompareFunction::LessEqual => COMPAREFUNCTION_LEQUAL,
            CompareFunction::Greater => COMPAREFUNCTION_GREATER,
            CompareFunction::NotEqual => COMPAREFUNCTION_NOTEQUAL,
            CompareFunction::GreaterEqual => COMPAREFUNCTION_GEQUAL,
            CompareFunction::Always => COMPAREFUNCTION_ALWAYS,
        }
    }
}

impl Isl for IndexFormat {
    fn isl(self) -> u32 {
        match self {
            IndexFormat::Uint16 => INDEX_WORD,
            IndexFormat::Uint32 => INDEX_DWORD,
        }
    }
}

impl Isl for Topology {
    fn isl(self) -> u32 {
        match self {
            Topology::PointList => _3DPRIM_POINTLIST,
            Topology::LineList => _3DPRIM_LINELIST,
            Topology::LineStrip => _3DPRIM_LINESTRIP,
            Topology::TriangleList => _3DPRIM_TRILIST,
            Topology::TriangleStrip => _3DPRIM_TRISTRIP,
        }
    }
}

const STATISTICS: [u32; 6] = [0x2310, 0x2318, 0x2320, 0x2338, 0x2340, 0x2348];

pub const POOL: u32 = 0x10_0000;

const CS_THREADS: u32 = 320;

#[derive(Clone, Debug)]
pub struct KernelAt {
    pub addr: A,
    pub threads: u32,
    pub shared: u32,
}

impl KernelAt {
    const SLM_KB: [(u32, u32); 16] = [(0, 0), (1, 1), (2, 2), (4, 3), (8, 4), (16, 5), (24, 8), (32, 6), (48, 9), (64, 7), (96, 10), (128, 11), (192, 12), (256, 13), (384, 14), (u32::MAX, 14)];
    const PREFERRED_KB: [(u32, u32); 11] = [(0, 0), (16, 1), (32, 2), (64, 3), (96, 4), (128, 5), (160, 6), (192, 7), (224, 8), (256, 9), (384, 10)];

    fn slm(bytes: u32) -> u32 {
        Self::SLM_KB.iter().find(|(kb, _)| kb.saturating_mul(1024) >= bytes).map_or(14, |(_, e)| *e)
    }

    fn preferred_slm(bytes: u32, threads: u32) -> u32 {
        if bytes == 0 {
            return 0;
        }
        let groups = (8 * 10 * 16) / (16 * threads);
        let want = (groups * bytes).min(192 * 1024);
        Self::PREFERRED_KB.iter().find(|(kb, _)| kb * 1024 >= want).map_or(10, |(_, e)| *e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub levels: u32,
    pub samples: u32,
    pub layers: u32,
    pub kind: TextureKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextureKind {
    #[default]
    Flat,
    Cube,
    Volume,
}

impl Texture {
    const ALIGN_BYTES: u32 = 128;
    const ALIGN_ROWS: u32 = 4;

    pub fn layout(&self) -> (u32, u32, Vec<(u32, u32)>) {
        let bpp = self.format.bytes();
        if self.samples > 1 {
            let ((tw, th), rows) = self.tile64();
            return (self.width.div_ceil(tw) * tw * bpp, self.height.div_ceil(th) * rows, vec![(0, 0)]);
        }
        if self.levels <= 1 {
            let (bw, bh) = self.format.block();
            return (self.width.div_ceil(bw) * bpp, self.height.div_ceil(bh), vec![(0, 0)]);
        }
        let (aw, ah) = (Self::ALIGN_BYTES / bpp, Self::ALIGN_ROWS);
        let size = |l: u32| ((self.width >> l).max(1).next_multiple_of(aw), (self.height >> l).max(1).next_multiple_of(ah));
        let (mut x, mut y) = (0, 0);
        let mut offsets = Vec::new();
        for l in 0..self.levels {
            offsets.push((x, y));
            if l == 1 {
                x += size(1).0;
            } else {
                y += size(l).1;
            }
        }
        let (top_w, left_h) = (size(0).0, size(0).1 + if self.levels > 1 { size(1).1 } else { 0 });
        let bottom_w = if self.levels > 2 { size(1).0 + size(2).0 } else if self.levels > 1 { size(1).0 } else { 0 };
        let right_h = size(0).1 + (2..self.levels).map(|l| size(l).1).sum::<u32>();
        let (w, h) = (top_w.max(bottom_w), left_h.max(right_h));
        (w * bpp, h, offsets)
    }

    pub fn plain(width: u32, height: u32, format: TextureFormat) -> Texture {
        Texture { width, height, format, levels: 1, samples: 1, layers: 1, kind: TextureKind::Flat }
    }

    pub fn size(&self) -> u64 {
        let (pitch, rows, _) = self.layout();
        let rows = if self.layers > 1 { self.array_pitch() * self.layers } else { rows };
        u64::from(pitch) * u64::from(rows)
    }

    pub fn array_pitch(&self) -> u32 {
        self.layout().1.next_multiple_of(Self::ALIGN_ROWS)
    }

    fn tile64(&self) -> ((u32, u32), u32) {
        let px = match self.samples {
            2 => (64, 128),
            4 => (64, 64),
            8 => (64, 32),
            _ => (32, 32),
        };
        (px, 64 * 1024 / (px.0 * self.format.bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexAttr {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u32,
    pub buffer: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline {
    pub vertex: String,
    pub fragment: String,
    pub topology: Topology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub strides: Vec<u32>,
    pub instanced: Vec<bool>,
    pub attrs: Vec<VertexAttr>,
    pub varyings: u32,
    pub bindings: u32,
    pub buffers: u32,
    pub push: u32,
    pub sampler: SamplerState,
    pub alpha_to_coverage: bool,
    pub sample_shading: bool,
    pub omask: bool,
    pub imask: bool,
    pub fragment_inputs: u32,
    pub colors: Vec<ColorTarget>,
    pub depth: Option<DepthState>,
    pub samples: u32,
    pub targets: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorTarget {
    pub blend: Option<BlendState>,
    pub write_mask: u8,
}

impl ColorTarget {
    pub const ALL: u32 = 0xf;
}

pub struct DepthTarget {
    pub addr: A,
    pub pitch: A,
    pub width: A,
    pub height: A,
    pub stencil: Option<StencilTarget>,
}

pub struct StencilTarget {
    pub addr: A,
    pub pitch: A,
    pub reference: A,
}

impl DepthTarget {
    pub fn layout(width: u32, height: u32) -> (u32, u32) {
        ((width * 4).next_multiple_of(128), height.next_multiple_of(32))
    }

    pub fn stencil_layout(width: u32, height: u32) -> (u32, u32) {
        (width.next_multiple_of(128), height.next_multiple_of(32))
    }
}

pub struct IndexArgs {
    pub format: A,
    pub addr: A,
    pub size: A,
    pub base_vertex: A,
}

pub struct DrawArgs {
    pub blend: A,
    pub bt: A,
    pub samplers: Option<A>,
    pub buffers: Vec<(A, A)>,
    pub vertices: A,
    pub instances: A,
    pub first: A,
    pub index: Option<IndexArgs>,
    pub depth: Option<DepthTarget>,
    pub indirect: Option<Vec<A>>,
    pub predicate: A,
}

const WRITE_PS_DEPTH_COUNT: u32 = 2;
const WRITE_TIMESTAMP: u32 = 3;

pub const IA_VERTICES_COUNT: u32 = 0x2310;
pub const IA_PRIMITIVES_COUNT: u32 = 0x2318;
pub const VS_INVOCATION_COUNT: u32 = 0x2320;
pub const CL_INVOCATION_COUNT: u32 = 0x2338;
pub const CL_PRIMITIVES_COUNT: u32 = 0x2340;
pub const PS_INVOCATION_COUNT: u32 = 0x2348;
pub const CS_INVOCATION_COUNT: u32 = 0x2290;

const GPGPU_DISPATCHDIMX: u32 = 0x2500;

const MI_PREDICATE_SRC0: u32 = 0x2400;
const MI_PREDICATE_SRC1: u32 = 0x2408;

const LOAD_LOAD: u32 = 2;
const LOAD_LOADINV: u32 = 3;
const COMBINE_SET: u32 = 0;
const COMBINE_XOR: u32 = 3;
const COMPARE_SRCS_EQUAL: u32 = 2;

const PRIM_START_VERTEX: u32 = 0x2430;
const PRIM_VERTEX_COUNT: u32 = 0x2434;
const PRIM_INSTANCE_COUNT: u32 = 0x2438;
const PRIM_START_INSTANCE: u32 = 0x243C;
const PRIM_BASE_VERTEX: u32 = 0x2440;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heap {
    Code,
    Surfaces,
}

#[derive(Default)]
pub struct Cmds<'a> {
    pub batch: Vec<OpData<'a>>,
    pub code: Vec<OpData<'a>>,
    pub surfaces: Vec<OpData<'a>>,
    n: usize,
}

impl TryFrom<&crate::base::Pipeline<'_>> for Pipeline {
    type Error = Error;

    fn try_from(p: &crate::base::Pipeline<'_>) -> Result<Pipeline, Error> {
        let layouts = p.layouts()?;
        let mut attrs: Vec<VertexAttr> = layouts
            .iter()
            .enumerate()
            .flat_map(|(i, l)| l.attributes.iter().map(move |a| VertexAttr { location: a.location, format: a.format, offset: a.offset, buffer: i as u32 }))
            .collect();
        attrs.sort_by_key(|v| v.location);
        if attrs.iter().enumerate().any(|(i, v)| v.location != i as u32) {
            return Err(Error("pipeline: vertex attribute locations are not 0, 1, ...".into()));
        }
        Ok(Pipeline {
            vertex: Symbol::name(&p.vertex)?.to_owned(),
            fragment: Symbol::name(&p.fragment)?.to_owned(),
            topology: p.topology.unwrap_or_default(),
            strip_index_format: p.strip_index_format,
            front_face: p.front_face.unwrap_or_default(),
            cull_mode: p.cull_mode.unwrap_or_default(),
            strides: layouts.iter().map(|l| l.stride).collect(),
            instanced: layouts.iter().map(|l| l.step_mode == StepMode::Instance).collect(),
            attrs,
            varyings: 0,
            bindings: 0,
            buffers: 0,
            push: 0,
            sampler: p.sampler_state()?,
            alpha_to_coverage: p.alpha_to_coverage.unwrap_or(false),
            sample_shading: p.sample_shading.unwrap_or(false),
            omask: false,
            imask: false,
            fragment_inputs: 0,
            colors: {
                let n = p.targets.len();
                let blends = p.blend_states(n)?;
                let masks = p.write_masks.clone().unwrap_or_else(|| vec![ColorTarget::ALL; n]);
                if blends.len() != n || masks.len() != n {
                    return Err(Error(format!("pipeline: {n} targets, {} blends, {} write masks", blends.len(), masks.len())));
                }
                blends
                    .into_iter()
                    .zip(masks)
                    .map(|(blend, m)| u8::try_from(m).ok().filter(|m| *m <= ColorTarget::ALL as u8).map(|write_mask| ColorTarget { blend, write_mask }).ok_or_else(|| Error(format!("pipeline: write mask {m}"))))
                    .collect::<Result<_, _>>()?
            },
            depth: p.depth_state()?,
            targets: match p.targets.len() {
                n @ 1..=crate::xe3_codegen::MAX_TARGETS => n as u32,
                n => return Err(Error(format!("pipeline: {n} targets (1 ..= {})", crate::xe3_codegen::MAX_TARGETS))),
            },
            samples: match p.samples.unwrap_or(1) {
                n @ (1 | 2 | 4 | 8) => n,
                n => return Err(Error(format!("pipeline: {n} samples (1, 2, 4 or 8)"))),
            },
        })
    }
}

impl Pipeline {
    pub fn of(body: &Body<'_>, op: OpId) -> Result<Pipeline, Error> {
        Pipeline::try_from(&body.head::<crate::base::Pipeline>(op)?)
    }

    pub fn with_kernels(mut self, body: &Body<'_>, kernels: &[OpId]) -> Result<Pipeline, Error> {
        let get = |name: &str| -> Result<Kernel<'static>, Error> {
            let mut heads = kernels.iter().map(|k| body.head::<Kernel>(*k));
            heads.find(|k| k.as_ref().map_or(true, |k| k.name().is_ok_and(|n| n == name))).unwrap_or_else(|| Err(Error(format!("no kernel @{name}"))))
        };
        let (vertex, fragment) = (get(&self.vertex)?, get(&self.fragment)?);
        if fragment.targets != Some(self.targets) {
            return Err(Error(format!("pipeline: {} targets, its fragment shader returns {:?}", self.targets, fragment.targets)));
        }
        self.varyings = vertex.varyings.unwrap_or(0);
        self.bindings = fragment.bindings.unwrap_or(0);
        self.buffers = vertex.buffers.unwrap_or(0).max(fragment.buffers.unwrap_or(0));
        self.push = vertex.push.unwrap_or(0).max(fragment.push.unwrap_or(0));
        self.omask = fragment.omask == Some(true);
        self.imask = fragment.imask == Some(true);
        self.fragment_inputs = fragment.inputs.unwrap_or(0);
        Ok(self)
    }

    pub fn table_entries(&self) -> u32 {
        match (self.buffers, self.push) {
            (0, 0) => self.targets + self.bindings,
            (n, _) => u32::from(crate::xe3_codegen::BUFFER_BTI) + n,
        }
    }

    pub fn reads_memory(&self) -> bool {
        self.buffers > 0 || self.push > 0
    }
}

impl<'a> Cmds<'a> {
    pub fn cmd(&mut self, data: OpData<'a>) {
        self.batch.push(data);
    }

    pub fn put(&mut self, heap: Heap, data: OpData<'a>) {
        match heap {
            Heap::Code => self.code.push(data),
            Heap::Surfaces => self.surfaces.push(data),
        }
    }

    pub fn state(&mut self, heap: Heap, prefix: &str, align: u32, data: OpData<'a>) -> String {
        self.n += 1;
        let name = format!("{prefix}{}", self.n);
        self.put(heap, data.with_attrs(vec![("sym_name", A::str(name.clone())), ("align", A::from(align))]));
        name
    }

    pub fn null_surface(&mut self) -> String {
        let null = self.state(Heap::Surfaces, "null_surface", 64, xe3!(StateRenderSurfaceState, surface_type = SURFTYPE_NULL, surface_format = SurfaceFormat::R32_UINT, mocs = MOCS));
        self.put(Heap::Surfaces, xe3!(StateBindingTableState, surface_state_pointer = A::sym(null.clone())).with_attrs(vec![("sym_name", A::str("bt_null")), ("align", A::from(64u32))]));
        null
    }

    pub fn pc_flush(&mut self) {
        self.cmd(xe3!(
            CmdPipeControl,
            depth_cache_flush_enable = true,
            render_target_cache_flush_enable = true,
            hdc_pipeline_flush_enable = true,
            command_streamer_stall_enable = true,
        ));
    }

    pub fn compute_context(&mut self) {
        self.cmd(xe3!(CmdPipeControl, hdc_pipeline_flush_enable = true, untyped_data_port_cache_flush_enable = true, command_streamer_stall_enable = true));
        self.base_addresses();
        self.cmd(xe3!(
            CmdPipeControl,
            state_cache_invalidation_enable = true,
            constant_cache_invalidation_enable = true,
            texture_cache_invalidation_enable = true,
            instruction_cache_invalidate_enable = true,
            command_streamer_stall_enable = true,
        ));
        self.cmd(xe3!(CmdStateSystemMemFenceAddress, system_memory_fence_address = A::sym("fence")));
        self.cmd(xe3!(CmdStateComputeMode, enable_variable_register_size_allocation = true, mask1 = 1u32 << 10, enable_variable_register_size_allocation_mask = 1u32));
    }

    #[allow(clippy::too_many_arguments, reason = "XY_BLOCK_COPY_BLT's two surfaces")]
    pub fn block_copy(&mut self, bpp: u32, (src, src_pitch, src_extent): (A, u32, [u32; 2]), [sx, sy]: [u32; 2], (dst, dst_pitch, dst_extent): (A, u32, [u32; 2]), [dx, dy]: [u32; 2], [w, h]: [u32; 2]) -> Result<(), Error> {
        let depth = match bpp {
            1 => 0u32,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 5,
            _ => return Err(Error(format!("block_copy: {bpp}-byte texels"))),
        };
        let mocs = MOCS >> 1;
        self.cmd(xe3!(
            CmdXyBlockCopyBlt,
            color_depth = depth,
            destination_pitch = dst_pitch - 1,
            destination_mocs_index = mocs,
            destination_x1 = dx,
            destination_y1 = dy,
            destination_x2 = dx + w,
            destination_y2 = dy + h,
            destination_base_address = dst,
            destination_target_memory = 1u32,
            source_x1 = sx,
            source_y1 = sy,
            source_pitch = src_pitch - 1,
            source_mocs_index = mocs,
            source_base_address = src,
            source_target_memory = 1u32,
            destination_surface_type = 1u32,
            destination_surface_width = dst_extent[0] - 1,
            destination_surface_height = dst_extent[1] - 1,
            source_surface_type = 1u32,
            source_surface_width = src_extent[0] - 1,
            source_surface_height = src_extent[1] - 1,
        ));
        Ok(())
    }

    pub fn set_event(&mut self, at: A, value: u32, copy: bool) {
        if copy {
            self.cmd(xe3!(CmdMiFlushDw, post_sync_operation = 1u32, address = at, immediate_data = value));
        } else {
            self.cmd(xe3!(
                CmdPipeControl,
                command_streamer_stall_enable = true,
                hdc_pipeline_flush_enable = true,
                untyped_data_port_cache_flush_enable = true,
                render_target_cache_flush_enable = true,
                depth_cache_flush_enable = true,
                post_sync_operation = 1u32,
                address = at,
                immediate_data = value,
            ));
        }
    }

    pub fn wait_event(&mut self, at: A, value: u32, copy: bool) {
        self.cmd(xe3!(CmdMiSemaphoreWait, compare_operation = 4u32, wait_mode = 1u32, semaphore_data_dword = value, semaphore_address = at));
        if !copy {
            self.cmd(xe3!(
                CmdPipeControl,
                command_streamer_stall_enable = true,
                state_cache_invalidation_enable = true,
                constant_cache_invalidation_enable = true,
                texture_cache_invalidation_enable = true,
            ));
        }
    }

    pub fn copy_tail(&mut self) {
        self.cmd(xe3!(CmdMiFlushDw));
        self.cmd(xe3!(CmdMiBatchBufferEnd));
    }

    fn base_addresses(&mut self) {
        let size = 0xf_ffffu32;
        self.cmd(xe3!(
            CmdStateBaseAddress,
            general_state_base_address_modify_enable = true,
            general_state_mocs = MOCS,
            general_state_buffer_size_modify_enable = true,
            general_state_buffer_size = size,
            stateless_data_port_access_mocs = MOCS,
            l1_cache_control = L1CC_WB,
            surface_state_base_address_modify_enable = true,
            surface_state_mocs = MOCS,
            dynamic_state_base_address_modify_enable = true,
            dynamic_state_mocs = MOCS,
            dynamic_state_buffer_size_modify_enable = true,
            dynamic_state_buffer_size = size,
            indirect_object_base_address_modify_enable = true,
            indirect_object_mocs = MOCS,
            indirect_object_buffer_size_modify_enable = true,
            indirect_object_buffer_size = size,
            instruction_base_address_modify_enable = true,
            instruction_mocs = MOCS,
            instruction_buffer_size_modify_enable = true,
            instruction_buffer_size = size,
            bindless_surface_state_mocs = MOCS,
            bindless_sampler_state_mocs = MOCS,
        ));
    }

    pub fn context(&mut self) {
        self.pc_flush();
        self.base_addresses();
        self.cmd(xe3!(
            CmdPipeControl,
            state_cache_invalidation_enable = true,
            constant_cache_invalidation_enable = true,
            texture_cache_invalidation_enable = true,
            instruction_cache_invalidate_enable = true,
            command_streamer_stall_enable = true,
        ));
        self.cmd(xe3!(CmdStateSystemMemFenceAddress, system_memory_fence_address = A::sym("fence")));
        self.cmd(xe3!(CmdMiLoadRegisterImm, register_offset = 0x20d8u32, data_dword = 0x0010_0010u32));
        self.cmd(xe3!(CmdMiLoadRegisterImm, register_offset = 0x6208u32, data_dword = 0x0020_0000u32));
        self.cmd(xe3!(Cmd3dstate3dMode, dx10_ogl_border_mode_for_ycrcb = true, dx10_ogl_border_mode_for_ycrcb_mask = true));
        self.cmd(xe3!(CmdStateComputeMode, enable_variable_register_size_allocation = true, mask1 = 1u32 << 10, enable_variable_register_size_allocation_mask = 1u32));
        self.cmd(xe3!(Cmd3dstateDrawingRectangleFast, clipped_drawing_rectangle_x_max = 0xffffu32, clipped_drawing_rectangle_y_max = 0xffffu32));
        self.cmd(xe3!(
            Cmd3dstateSamplePattern,
            _1x_sample0_x_offset = 0.5f32,
            _1x_sample0_y_offset = 0.5f32,
            _2x_sample0_x_offset = 0.75f32,
            _2x_sample0_y_offset = 0.75f32,
            _2x_sample1_x_offset = 0.25f32,
            _2x_sample1_y_offset = 0.25f32,
            _8x_sample0_x_offset = 0.5625f32,
            _8x_sample0_y_offset = 0.3125f32,
            _8x_sample1_x_offset = 0.4375f32,
            _8x_sample1_y_offset = 0.6875f32,
            _8x_sample2_x_offset = 0.8125f32,
            _8x_sample2_y_offset = 0.5625f32,
            _8x_sample3_x_offset = 0.3125f32,
            _8x_sample3_y_offset = 0.1875f32,
            _8x_sample4_x_offset = 0.1875f32,
            _8x_sample4_y_offset = 0.8125f32,
            _8x_sample5_x_offset = 0.0625f32,
            _8x_sample5_y_offset = 0.4375f32,
            _8x_sample6_x_offset = 0.6875f32,
            _8x_sample6_y_offset = 0.9375f32,
            _8x_sample7_x_offset = 0.9375f32,
            _8x_sample7_y_offset = 0.0625f32,
            _4x_sample0_x_offset = 0.375f32,
            _4x_sample0_y_offset = 0.125f32,
            _4x_sample1_x_offset = 0.875f32,
            _4x_sample1_y_offset = 0.375f32,
            _4x_sample2_x_offset = 0.125f32,
            _4x_sample2_y_offset = 0.625f32,
            _4x_sample3_x_offset = 0.625f32,
            _4x_sample3_y_offset = 0.875f32,
        ));
        for c in [xe3!(Cmd3dstateAaLineParameters), xe3!(Cmd3dstateWmChromakey), xe3!(Cmd3dstateWmHzOp), xe3!(Cmd3dstatePolyStippleOffset), xe3!(Cmd3dstateMeshControl), xe3!(Cmd3dstateTaskControl)] {
            self.cmd(c);
        }
        for (stage, offset) in [0u32, 8, 16, 24].into_iter().enumerate() {
            self.cmd(xe3!(Cmd3dstatePushConstantAllocDs, _3d_command_sub_opcode = 18 + stage as u32, constant_buffer_offset = offset, constant_buffer_size = 8u32));
        }
        self.pc_flush();
    }

    pub fn statistics(&mut self, q: &str, at: u32) {
        self.cmd(xe3!(CmdPipeControl, command_streamer_stall_enable = true, stall_at_pixel_scoreboard = true));
        for (i, reg) in STATISTICS.iter().enumerate() {
            for half in 0..2u32 {
                let to = Fill::at(q, (at + i as u32 * 8 + half * 4).into());
                self.cmd(xe3!(CmdMiStoreRegisterMem, register_address = reg + half * 4, memory_address = to));
            }
        }
    }

    pub fn surface(&mut self, prefix: &str, base: A, t: &Texture) -> String {
        let cube = t.kind == TextureKind::Cube;
        self.state(
            Heap::Surfaces,
            prefix,
            64,
            xe3!(
                StateRenderSurfaceState,
                surface_type = match t.kind { TextureKind::Flat => SURFTYPE_2D, TextureKind::Cube => SURFTYPE_CUBE, TextureKind::Volume => SURFTYPE_3D },
                cube_face_enable_positive_x = cube,
                cube_face_enable_negative_x = cube,
                cube_face_enable_positive_y = cube,
                cube_face_enable_negative_y = cube,
                cube_face_enable_positive_z = cube,
                cube_face_enable_negative_z = cube,
                surface_format = t.format.isl(),
                tile_mode = if t.samples > 1 { 1u32 } else { 0u32 },
                number_of_multisamples = t.samples.trailing_zeros(),
                surface_horizontal_alignment = 3u32,
                surface_vertical_alignment = 1u32,
                width = t.width - 1,
                height = t.height - 1,
                surface_pitch = t.layout().0 - 1,
                mip_count_lod = t.levels - 1,
                surface_array = t.layers > 1 && t.kind != TextureKind::Volume,
                depth = if cube { t.layers / 6 - 1 } else { t.layers - 1 },
                surface_qpitch = if t.layers > 1 { t.array_pitch() / 4 } else { 0 },
                mip_tail_start_lod = 15u32,
                mocs = MOCS,
                l1_cache_control = L1CC_WB,
                shader_channel_select_red = 4u32,
                shader_channel_select_green = 5u32,
                shader_channel_select_blue = 6u32,
                shader_channel_select_alpha = 7u32,
                surface_base_address = base,
            ),
        )
    }

    pub fn depth_surface(&mut self, prefix: &str, base: A, width: u32, height: u32) -> String {
        let (pitch, _) = DepthTarget::layout(width, height);
        self.state(
            Heap::Surfaces,
            prefix,
            64,
            xe3!(
                StateRenderSurfaceState,
                surface_type = SURFTYPE_2D,
                surface_format = SurfaceFormat::R32_FLOAT,
                tile_mode = TILE4,
                depth_stencil_resource = true,
                surface_horizontal_alignment = 1u32,
                surface_vertical_alignment = 1u32,
                width = width - 1,
                height = height - 1,
                surface_pitch = pitch - 1,
                mip_tail_start_lod = 15u32,
                mocs = MOCS,
                l1_cache_control = L1CC_WB,
                shader_channel_select_red = 4u32,
                shader_channel_select_green = 5u32,
                shader_channel_select_blue = 6u32,
                shader_channel_select_alpha = 7u32,
                surface_base_address = base,
            ),
        )
    }

    pub fn depth_barrier(&mut self) {
        self.cmd(xe3!(CmdPipeControl, depth_cache_flush_enable = true, depth_stall_enable = true, command_streamer_stall_enable = true));
        self.cmd(xe3!(CmdPipeControl, texture_cache_invalidation_enable = true, command_streamer_stall_enable = true));
    }

    pub fn buffer_surface(&mut self, prefix: &str, base: A, size: u32) -> String {
        self.buffer_surface_of(prefix, base, Self::buffer_size(size).map(A::from))
    }

    pub fn buffer_size(size: u32) -> [u32; 3] {
        let n = size.max(1).next_multiple_of(4) - 1;
        [n & 0x7f, (n >> 7) & 0x3fff, (n >> 21) & 0x7ff]
    }

    pub fn buffer_surface_of(&mut self, prefix: &str, base: A, [width, height, depth]: [A; 3]) -> String {
        self.state(
            Heap::Surfaces,
            prefix,
            64,
            xe3!(
                StateRenderSurfaceState,
                surface_type = SURFTYPE_BUFFER,
                surface_format = SurfaceFormat::RAW,
                surface_horizontal_alignment = 3u32,
                surface_vertical_alignment = 1u32,
                width = width,
                height = height,
                depth = depth,
                surface_pitch = 0u32,
                mocs = MOCS,
                l1_cache_control = L1CC_WB,
                surface_base_address = base,
            ),
        )
    }

    pub fn binding_table(&mut self, entries: Vec<A>) -> String {
        let mut entries = entries.into_iter();
        let first = entries.next().expect("a binding table has an entry");
        let name = self.state(Heap::Surfaces, "bt", 64, xe3!(StateBindingTableState, surface_state_pointer = first));
        for e in entries {
            self.put(Heap::Surfaces, xe3!(StateBindingTableState, surface_state_pointer = e).with("align", A::from(4u32).into()));
        }
        name
    }

    pub fn samplers(&mut self, n: u32, s: &SamplerState) -> String {
        let border = self.state(Heap::Code, "border_color", 64, xe3!(StateSamplerBorderColorState));
        assert!(n > 0, "at least one sampler");
        let aniso = s.max_anisotropy > 1 && s.mag_filter == FilterMode::Linear && s.min_filter == FilterMode::Linear;
        let map = |f: FilterMode| if aniso { 2u32 } else { u32::from(f == FilterMode::Linear) };
        let (min_linear, mag_linear) = (s.min_filter == FilterMode::Linear, s.mag_filter == FilterMode::Linear);
        let address = s.address.map(|m| match m {
            AddressMode::Repeat => 0u32,
            AddressMode::MirrorRepeat => 1,
            AddressMode::ClampToEdge => 2,
        });
        let lod = |v: f32| v.clamp(0.0, 14.0);
        let (min_lod, max_lod) = match s.mipmap_filter {
            Some(_) => (lod(s.lod_min_clamp), lod(s.lod_max_clamp)),
            None => (0.0, 0.0),
        };
        let sampler = || {
            xe3!(
                StateSamplerState,
                min_mode_filter = map(s.min_filter),
                mag_mode_filter = map(s.mag_filter),
                mip_mode_filter = match s.mipmap_filter { None => MIPFILTER_NONE, Some(MipmapFilter::Nearest) => MIPFILTER_NEAREST, Some(MipmapFilter::Linear) => MIPFILTER_LINEAR },
                lod_preclamp_mode = 2u32,
                cube_surface_control_mode = 1u32,
                shadow_function = match s.compare {
                    None | Some(CompareFunction::Never) => PREFILTEROP_ALWAYS,
                    Some(CompareFunction::Less) => PREFILTEROP_LEQUAL,
                    Some(CompareFunction::Equal) => PREFILTEROP_NOTEQUAL,
                    Some(CompareFunction::LessEqual) => PREFILTEROP_LESS,
                    Some(CompareFunction::Greater) => PREFILTEROP_GEQUAL,
                    Some(CompareFunction::NotEqual) => PREFILTEROP_EQUAL,
                    Some(CompareFunction::GreaterEqual) => PREFILTEROP_GREATER,
                    Some(CompareFunction::Always) => PREFILTEROP_NEVER,
                },
                min_lod = min_lod,
                max_lod = max_lod,
                maximum_anisotropy = if aniso { (u32::from(s.max_anisotropy.min(16)) - 2) / 2 } else { 0 },
                tcx_address_control_mode = address[0],
                tcy_address_control_mode = address[1],
                tcz_address_control_mode = address[2],
                u_address_min_filter_rounding_enable = min_linear,
                v_address_min_filter_rounding_enable = min_linear,
                r_address_min_filter_rounding_enable = min_linear,
                u_address_mag_filter_rounding_enable = mag_linear,
                v_address_mag_filter_rounding_enable = mag_linear,
                r_address_mag_filter_rounding_enable = mag_linear,
                border_color_pointer = A::sym(border.clone()),
            )
        };
        let first = self.state(Heap::Code, "samplers", 32, sampler());
        for _ in 1..n {
            self.put(Heap::Code, sampler().with_attrs(vec![("align", A::from(16u32))]));
        }
        first
    }

    pub fn blend(&mut self, colors: &[ColorTarget], alpha_to_coverage: bool) -> String {
        let dither = colors.iter().all(|c| c.blend.is_none());
        let independent = colors.iter().any(|c| c.blend.is_some_and(|b| b.color != b.alpha));
        let name = self.state(
            Heap::Code,
            "blend",
            64,
            xe3!(StateBlendState, color_dither_enable = dither, alpha_test_function = 1u32, independent_alpha_blend_enable = independent, alpha_to_coverage_enable = alpha_to_coverage),
        );
        for c in colors {
            let mut entry = crate::xe3_value!(
                StateBlendStateEntry,
                post_blend_color_clamp_enable = true,
                pre_blend_color_clamp_enable = true,
                color_clamp_range = COLORCLAMP_RTFORMAT,
                simple_float_blend_enable = true,
                write_disable_red = c.write_mask & 1 == 0,
                write_disable_green = c.write_mask & 2 == 0,
                write_disable_blue = c.write_mask & 4 == 0,
                write_disable_alpha = c.write_mask & 8 == 0,
            );
            if let Some(b) = c.blend {
                let f = |v: u32| Some(Fill::from(v));
                entry.color_buffer_blend_enable = Some(true.into());
                (entry.source_blend_factor, entry.destination_blend_factor) = (f(b.color.src_factor.isl()), f(b.color.dst_factor.isl()));
                (entry.source_alpha_blend_factor, entry.destination_alpha_blend_factor) = (f(b.alpha.src_factor.isl()), f(b.alpha.dst_factor.isl()));
                (entry.color_blend_function, entry.alpha_blend_function) = (f(b.color.operation.isl()), f(b.alpha.operation.isl()));
            }
            self.put(Heap::Code, crate::xe3_codegen::xe3_op(&entry).with("align", A::from(4u32).into()));
        }
        name
    }

    pub fn pass_state(&mut self, target: &str, t: &Texture) -> [String; 5] {
        let (w, h) = (t.width as f32, t.height as f32);
        let [cc, sf] = self.viewport([0.0, 0.0, w, h], [0.0, 1.0], [t.width, t.height]);
        let ccs = self.color_calc([0.0; 4]);
        let rss = self.surface("rt_surface", A::sym(target), t);
        let sc = self.scissor([0, 0, t.width, t.height]);
        [cc, sf, ccs, rss, sc]
    }

    pub fn viewport(&mut self, [x, y, w, h]: [f32; 4], [min, max]: [f32; 2], [tw, th]: [u32; 2]) -> [String; 2] {
        let code = Heap::Code;
        let cc = self.state(code, "cc_viewport", 32, xe3!(StateCcViewport, minimum_depth = min.min(max), maximum_depth = min.max(max)));
        let (x0, y0) = (x.max(0.0), y.max(0.0));
        let (x1, y1) = ((x + w).min(tw as f32), (y + h).min(th as f32));
        let sf = self.state(
            code,
            "sf_clip_viewport",
            64,
            xe3!(
                StateSfClipViewport,
                viewport_matrix_element_m00 = w / 2.0,
                viewport_matrix_element_m11 = h / 2.0,
                viewport_matrix_element_m22 = max - min,
                viewport_matrix_element_m30 = x + w / 2.0,
                viewport_matrix_element_m31 = y + h / 2.0,
                viewport_matrix_element_m32 = min,
                x_min_clip_guardband = -512.0f32,
                x_max_clip_guardband = 512.0f32,
                y_min_clip_guardband = -512.0f32,
                y_max_clip_guardband = 512.0f32,
                x_min_viewport = x0,
                x_max_viewport = x1 - 1.0,
                y_min_viewport = y0,
                y_max_viewport = y1 - 1.0,
            ),
        );
        [cc, sf]
    }

    pub fn color_calc(&mut self, [r, g, b, a]: [f32; 4]) -> String {
        self.state(
            Heap::Code,
            "color_calc",
            64,
            xe3!(StateColorCalcState, alpha_test_format = 1u32, blend_constant_color_red = r, blend_constant_color_green = g, blend_constant_color_blue = b, blend_constant_color_alpha = a),
        )
    }

    pub fn scissor(&mut self, [x, y, w, h]: [u32; 4]) -> String {
        let (x0, y0, x1, y1) = if w == 0 || h == 0 { (1u32, 1u32, 0u32, 0u32) } else { (x, y, x + w - 1, y + h - 1) };
        self.state(
            Heap::Code,
            "scissor",
            64,
            xe3!(StateScissorRect, scissor_rectangle_x_min = x0, scissor_rectangle_y_min = y0, scissor_rectangle_x_max = x1, scissor_rectangle_y_max = y1),
        )
    }

    pub fn compute(&mut self, kernel: &KernelAt, [x, y]: [A; 2], params: Vec<A>) {
        self.walker(kernel, [x, y, A::from(1u32)], false, params, None);
    }

    pub fn compute_bound(&mut self, kernel: &KernelAt, [x, y]: [A; 2], params: Vec<A>, table: A, pool: A) {
        self.cmd(xe3!(Cmd3dstateBindingTablePoolAlloc, binding_table_pool_base_address = pool, binding_table_pool_buffer_size = POOL / 4096, mocs = MOCS));
        self.walker(kernel, [x, y, A::from(1u32)], false, params, Some(table));
    }

    pub fn condition(&mut self, at: A, inverted: bool) {
        self.cmd(xe3!(CmdMiLoadRegisterMem, register_address = MI_PREDICATE_SRC0, memory_address = at));
        for (reg, v) in [(MI_PREDICATE_SRC0 + 4, 0u32), (MI_PREDICATE_SRC1, 0), (MI_PREDICATE_SRC1 + 4, 0)] {
            self.cmd(xe3!(CmdMiLoadRegisterImm, register_offset = reg, data_dword = v));
        }
        let load = if inverted { LOAD_LOAD } else { LOAD_LOADINV };
        self.cmd(xe3!(CmdMiPredicate, load_operation = load, combine_operation = COMBINE_SET, compare_operation = COMPARE_SRCS_EQUAL));
    }

    pub fn draw_count_predicate(&mut self, count: A, index: u32) {
        if index == 0 {
            self.cmd(xe3!(CmdMiLoadRegisterMem, register_address = MI_PREDICATE_SRC0, memory_address = count));
            self.cmd(xe3!(CmdMiLoadRegisterImm, register_offset = MI_PREDICATE_SRC0 + 4, data_dword = 0u32));
            self.cmd(xe3!(CmdMiLoadRegisterImm, register_offset = MI_PREDICATE_SRC1 + 4, data_dword = 0u32));
        }
        self.cmd(xe3!(CmdMiLoadRegisterImm, register_offset = MI_PREDICATE_SRC1, data_dword = index));
        let (load, combine) = if index == 0 { (LOAD_LOADINV, COMBINE_SET) } else { (LOAD_LOAD, COMBINE_XOR) };
        self.cmd(xe3!(CmdMiPredicate, load_operation = load, combine_operation = combine, compare_operation = COMPARE_SRCS_EQUAL));
    }

    pub fn texture_barrier(&mut self) {
        self.cmd(xe3!(CmdPipeControl, untyped_data_port_cache_flush_enable = true, hdc_pipeline_flush_enable = true, texture_cache_invalidation_enable = true, command_streamer_stall_enable = true));
    }

    pub fn timestamp(&mut self, at: A) {
        self.cmd(xe3!(CmdPipeControl, command_streamer_stall_enable = true, post_sync_operation = WRITE_TIMESTAMP, address = at));
    }

    pub fn depth_count(&mut self, at: A) {
        self.cmd(xe3!(CmdPipeControl, depth_stall_enable = true, post_sync_operation = WRITE_PS_DEPTH_COUNT, address = at));
    }

    pub fn statistic(&mut self, reg: u32, at: [A; 2]) {
        self.cmd(xe3!(CmdPipeControl, command_streamer_stall_enable = true, stall_at_pixel_scoreboard = true));
        for (half, to) in at.into_iter().enumerate() {
            self.cmd(xe3!(CmdMiStoreRegisterMem, register_address = reg + half as u32 * 4, memory_address = to));
        }
    }

    pub fn compute_indirect(&mut self, kernel: &KernelAt, at: [A; 3], params: Vec<A>) {
        for (reg, from) in [GPGPU_DISPATCHDIMX, GPGPU_DISPATCHDIMX + 4, GPGPU_DISPATCHDIMX + 8].into_iter().zip(at) {
            self.cmd(xe3!(CmdMiLoadRegisterMem, register_address = reg, memory_address = from));
        }
        self.walker(kernel, [0u32, 0, 0].map(A::from), true, params, None);
    }

    fn walker(&mut self, kernel: &KernelAt, [x, y, z]: [A; 3], indirect: bool, params: Vec<A>, table: Option<A>) {
        self.cmd(xe3!(CmdCfeState, maximum_number_of_threads = CS_THREADS));
        let threads = kernel.threads.max(1);
        let mut idd = crate::xe3_value!(
            StateInterfaceDescriptorDataFields,
            kernel_start_pointer = kernel.addr.clone(),
            registers_per_thread = 3u32,
            number_of_threads_in_gpgpu_thread_group = threads,
            thread_group_dispatch_size = match threads {
                0..=16 => 0u32,
                17..=32 => 1,
                _ => 2,
            },
            shared_local_memory_size = KernelAt::slm(kernel.shared),
            preferred_slm_allocation_size = KernelAt::preferred_slm(kernel.shared, threads),
            number_of_barriers = u32::from(threads > 1),
        );
        idd.binding_table_pointer = table.map(Fill::from);
        let mut inline: Vec<Fill> = params.into_iter().map(Fill::from).collect();
        inline.resize(crate::xe3_codegen::COMPUTE_PARAMS, 0u32.into());
        let body = crate::xe3_value!(
            StateComputeWalkerBodyFields,
            simd_size = 1u32,
            message_simd = 1u32,
            emit_inline_parameter = true,
            generate_local_id = true,
            emit_local = 1u32,
            execution_mask = 0xffffu32,
            local_x_maximum = 16 * threads - 1,
            thread_group_id_x_dimension = x,
            thread_group_id_y_dimension = y,
            thread_group_id_z_dimension = z,
            interface_descriptor = Fields(idd),
            post_sync = Fields(crate::xe3_value!(StatePostsyncDataFields, mocs = MOCS)),
            inline_data = inline,
        );
        self.cmd(xe3!(CmdComputeWalker, indirect_parameter_enable = indirect, body = Fields(body)));
        self.cmd(xe3!(CmdPipeControl, untyped_data_port_cache_flush_enable = true, hdc_pipeline_flush_enable = true, command_streamer_stall_enable = true));
    }

    pub fn draw(&mut self, p: &Pipeline, args: DrawArgs, shared: &[String; 5]) -> Result<(), Error> {
        if args.buffers.len() != p.strides.len() {
            return Err(Error(format!("draw: {} vertex buffers for a pipeline of {}", args.buffers.len(), p.strides.len())));
        }
        let [cc, sf, ccs, _, sc] = shared;
        let inputs = p.attrs.len() as u32;

        self.cmd(xe3!(Cmd3dstateViewportStatePointersCc, cc_viewport_pointer = A::sym(cc)));
        self.cmd(xe3!(Cmd3dstateViewportStatePointersSfClip, sf_clip_viewport_pointer = A::sym(sf)));
        self.cmd(xe3!(Cmd3dstateScissorStatePointers, scissor_rect_pointer = A::sym(sc)));
        let units = inputs.max(2 + p.varyings).div_ceil(4).max(1);
        let entries = 5888 / units / 8 * 8;
        self.cmd(xe3!(
            Cmd3dstateUrbAllocVs,
            vs_urb_entry_allocation_size = units - 1,
            vs_urb_starting_address_slice0 = 4u32,
            vs_urb_starting_address_slice_n = 4u32,
            vs_number_of_urb_entries_slice0 = entries,
            vs_number_of_urb_entries_slice_n = entries,
        ));
        self.cmd(xe3!(Cmd3dstateUrbAllocHs, hs_urb_starting_address_slice0 = 4u32, hs_urb_starting_address_slice_n = 4u32));
        self.cmd(xe3!(Cmd3dstateUrbAllocDs, ds_urb_starting_address_slice0 = 4u32, ds_urb_starting_address_slice_n = 4u32));
        self.cmd(xe3!(Cmd3dstateUrbAllocGs, gs_urb_starting_address_slice0 = 4u32, gs_urb_starting_address_slice_n = 4u32));
        self.cmd(xe3!(Cmd3dstateBlendStatePointers, blend_state_pointer_valid = true, blend_state_pointer = args.blend.clone()));
        self.cmd(xe3!(Cmd3dstateCcStatePointers, color_calc_state_pointer_valid = true, color_calc_state_pointer = A::sym(ccs)));
        self.cmd(xe3!(Cmd3dstateConstantAll, shader_update_enable = 0x1fu32, mocs = MOCS));
        self.cmd(xe3!(Cmd3dstateBindingTablePoolAlloc, binding_table_pool_base_address = A::sym("surfaces"), binding_table_pool_buffer_size = POOL / 4096, mocs = MOCS));
        let vs_bt = if p.reads_memory() { args.bt.clone() } else { A::within("surfaces", "bt_null") };
        self.cmd(xe3!(Cmd3dstateBindingTablePointersVs, pointer_to_vs_binding_table = vs_bt));
        let null = || A::within("surfaces", "bt_null");
        self.cmd(xe3!(Cmd3dstateBindingTablePointersHs, pointer_to_hs_binding_table = null()));
        self.cmd(xe3!(Cmd3dstateBindingTablePointersDs, pointer_to_ds_binding_table = null()));
        self.cmd(xe3!(Cmd3dstateBindingTablePointersGs, pointer_to_gs_binding_table = null()));
        self.cmd(xe3!(Cmd3dstateBindingTablePointersPs, pointer_to_ps_binding_table = args.bt.clone()));
        if let Some(samplers) = args.samplers.clone() {
            self.cmd(xe3!(Cmd3dstateSamplerStatePointersPs, pointer_to_ps_sampler_state = samplers));
        }
        self.cmd(xe3!(CmdPipeControl, stall_at_pixel_scoreboard = true, render_target_cache_flush_enable = true));
        self.cmd(xe3!(Cmd3dstateMultisample, number_of_multisamples = p.samples.trailing_zeros()));
        self.cmd(xe3!(Cmd3dstateSampleMask, sample_mask = (1u32 << p.samples) - 1));
        self.cmd(xe3!(
            Cmd3dstateVs,
            kernel_start_pointer = A::sym(p.vertex.clone()),
            registers_per_thread = 3u32,
            vertex_urb_entry_read_length = inputs.div_ceil(2).max(1),
            binding_table_entry_count = if p.reads_memory() { p.table_entries() } else { 0 },
            dispatch_grf_start_register_for_urb_data = u32::from(crate::xe3_codegen::VS_ATTR_GRF),
            enable = true,
            statistics_enable = true,
            maximum_number_of_threads = 545u32,
        ));
        for c in [xe3!(Cmd3dstateHs), xe3!(Cmd3dstateTe), xe3!(Cmd3dstateDs), xe3!(Cmd3dstateGs)] {
            self.cmd(c);
        }
        self.cmd(xe3!(
            Cmd3dstatePs,
            kernel_0_enable = true,
            kernel_0_simd_width = 0u32,
            kernel_start_pointer_0 = A::sym(p.fragment.clone()),
            registers_per_thread = 3u32,
            binding_table_entry_count = p.table_entries(),
            sampler_count = p.bindings.div_ceil(4),
            maximum_number_of_threads_per_psd = 127u32,
            dispatch_grf_start_register_for_constant_setup_data_0 = u32::from(if p.fragment_inputs > 0 { crate::xe3_codegen::Interpolate::setup(p.imask) } else { 1 + u8::from(p.imask) }),
            kernel_0_poly_packing_policy = POLY_PACK16_FIXED,
            vector_mask_enable = p.bindings > 0,
        ));
        self.cmd(xe3!(Cmd3dstatePsExtra, pixel_shader_valid = true, pixel_shader_is_per_sample = p.sample_shading, omask_present_to_render_target = p.omask, input_coverage_mask_state = u32::from(p.imask)));
        self.cmd(xe3!(Cmd3dstateStreamout));
        self.cmd(xe3!(
            Cmd3dstateClip,
            api_mode = APIMODE_D3D,
            statistics_enable = true,
            force_user_clip_distance_clip_test_enable_bitmask = true,
            early_cull_enable = true,
            triangle_fan_provoking_vertex_select = 2u32,
            line_strip_list_provoking_vertex_select = 1u32,
            triangle_strip_list_provoking_vertex_select = 2u32,
            guardband_clip_test_enable = true,
            viewport_xy_clip_test_enable = true,
            clip_enable = true,
            force_zero_rta_index_enable = true,
            maximum_point_width = 255.875f32,
            minimum_point_width = 0.125f32,
        ));
        let bias = p.depth.filter(DepthState::biased);
        self.cmd(xe3!(
            Cmd3dstateRaster,
            viewport_z_near_clip_test_enable = true,
            scissor_rectangle_enable = true,
            global_depth_offset_enable_solid = bias.is_some(),
            global_depth_offset_enable_wireframe = bias.is_some(),
            global_depth_offset_enable_point = bias.is_some(),
            global_depth_offset_constant = bias.map_or(0.0, |d| d.bias as f32),
            global_depth_offset_scale = bias.map_or(0.0, |d| d.bias_slope_scale),
            global_depth_offset_clamp = bias.map_or(0.0, |d| d.bias_clamp),
            front_winding = u32::from(p.front_face == FrontFace::Ccw),
            cull_mode = match p.cull_mode {
                CullMode::None => CULLMODE_NONE,
                CullMode::Front => CULLMODE_FRONT,
                CullMode::Back => CULLMODE_BACK,
            },
            viewport_z_far_clip_test_enable = true,
            dx_multisample_rasterization_enable = p.samples > 1,
        ));
        self.cmd(xe3!(
            Cmd3dstateSf,
            viewport_transform_enable = true,
            statistics_enable = true,
            line_width = 1.0f32,
            point_width = 1.0f32,
            point_width_source = 1u32,
            aa_line_distance_mode = 1u32,
            triangle_fan_provoking_vertex_select = 2u32,
            line_strip_list_provoking_vertex_select = 1u32,
            triangle_strip_list_provoking_vertex_select = 2u32,
        ));
        let bary = if p.fragment_inputs == 0 { 0 } else if p.sample_shading { 4 } else { 1 };
        self.cmd(xe3!(Cmd3dstateWm, point_rasterization_rule = 1u32, line_antialiasing_region_width = 1u32, statistics_enable = true, barycentric_interpolation_mode = bary));
        self.cmd(xe3!(
            Cmd3dstateSbe,
            number_of_sf_output_attributes = p.varyings,
            vertex_urb_entry_read_offset = u32::from(p.varyings > 0),
            vertex_urb_entry_read_length = p.varyings.div_ceil(2).max(1),
            point_sprite_texture_coordinate_origin = 1u32,
            attribute_swizzle_enable = true,
            force_vertex_urb_entry_read_offset = true,
            force_vertex_urb_entry_read_length = true,
            attribute_active_component_format = vec![Fill::from(3u32); 32],
        ));
        let swiz: Vec<_> = (0..16u32)
            .map(|i| Fields(if i < p.varyings { crate::xe3_value!(StateSfOutputAttributeDetailFields, source_attribute = i) } else { Default::default() }))
            .collect();
        self.cmd(xe3!(Cmd3dstateSbeSwiz, attribute = swiz));
        let mut ps_blend = crate::xe3_value!(Cmd3dstatePsBlend, has_writeable_rt = true, alpha_to_coverage_enable = p.alpha_to_coverage);
        if let Some(b) = p.colors.first().and_then(|c| c.blend) {
            let f = |v: u32| Some(Fill::from(v));
            ps_blend.color_buffer_blend_enable = Some(true.into());
            (ps_blend.source_blend_factor, ps_blend.destination_blend_factor) = (f(b.color.src_factor.isl()), f(b.color.dst_factor.isl()));
            (ps_blend.source_alpha_blend_factor, ps_blend.destination_alpha_blend_factor) = (f(b.alpha.src_factor.isl()), f(b.alpha.dst_factor.isl()));
            ps_blend.independent_alpha_blend_enable = Some((b.color != b.alpha).into());
        }
        self.cmd(crate::xe3_codegen::xe3_op(&ps_blend));
        match (p.depth, &args.depth) {
            (Some(d), Some(t)) => {
                let mut wm = crate::xe3_value!(Cmd3dstateWmDepthStencil, depth_test_enable = true, depth_test_function = d.compare.isl(), depth_buffer_write_enable = d.write);
                match (d.stencil, &t.stencil) {
                    (Some(s), Some(st)) => {
                        let f = |v: u32| Some(Fill::from(v));
                        wm.stencil_test_enable = Some(true.into());
                        wm.stencil_buffer_write_enable = Some((s.write_mask != 0).into());
                        (wm.stencil_test_function, wm.stencil_fail_op) = (f(s.compare.isl()), f(s.fail.isl()));
                        (wm.stencil_pass_depth_fail_op, wm.stencil_pass_depth_pass_op) = (f(s.depth_fail.isl()), f(s.pass.isl()));
                        (wm.stencil_test_mask, wm.stencil_write_mask) = (f(s.read_mask.into()), f(s.write_mask.into()));
                        wm.stencil_reference_value = Some(st.reference.clone().into());
                        if let Some(b) = s.back {
                            wm.double_sided_stencil_enable = Some(true.into());
                            (wm.backface_stencil_test_function, wm.backface_stencil_fail_op) = (f(b.compare.isl()), f(b.fail.isl()));
                            (wm.backface_stencil_pass_depth_fail_op, wm.backface_stencil_pass_depth_pass_op) = (f(b.depth_fail.isl()), f(b.pass.isl()));
                            (wm.backface_stencil_test_mask, wm.backface_stencil_write_mask) = (f(s.read_mask.into()), f(s.write_mask.into()));
                            wm.backface_stencil_reference_value = Some(st.reference.clone().into());
                        }
                    }
                    (Some(_), None) => return Err(Error("draw: the pipeline tests stencil but the depth attachment has no stencil".into())),
                    (None, _) => {}
                }
                self.cmd(crate::xe3_codegen::xe3_op(&wm));
                self.cmd(match d.bounds {
                    Some([min, max]) => xe3!(Cmd3dstateDepthBounds, depth_bounds_test_enable = 1u32, depth_bounds_test_min_value = min, depth_bounds_test_max_value = max),
                    None => xe3!(Cmd3dstateDepthBounds),
                });
                self.cmd(xe3!(
                    Cmd3dstateDepthBuffer,
                    surface_type = SURFTYPE_2D,
                    surface_format = D32_FLOAT,
                    depth_write_enable = true,
                    surface_base_address = t.addr.clone(),
                    surface_pitch = t.pitch.clone(),
                    width = t.width.clone(),
                    height = t.height.clone(),
                    tiled_mode = TILE4,
                    mocs = MOCS,
                ));
            }
            (None, None) => {
                self.cmd(xe3!(Cmd3dstateWmDepthStencil));
                self.cmd(xe3!(Cmd3dstateDepthBounds));
                self.cmd(xe3!(Cmd3dstateDepthBuffer, surface_type = SURFTYPE_NULL, surface_format = D32_FLOAT, mocs = MOCS));
            }
            (Some(_), None) => return Err(Error("draw: the pipeline tests depth but the pass has no depth attachment".into())),
            (None, Some(_)) => return Err(Error("draw: the pass has a depth attachment the pipeline does not describe".into())),
        }
        match args.depth.as_ref().and_then(|t| t.stencil.as_ref().map(|s| (t, s))) {
            Some((t, s)) => self.cmd(xe3!(
                Cmd3dstateStencilBuffer,
                surface_type = SURFTYPE_2D,
                stencil_write_enable = true,
                surface_base_address = s.addr.clone(),
                surface_pitch = s.pitch.clone(),
                width = t.width.clone(),
                height = t.height.clone(),
                tiled_mode = TILE4,
                mocs = MOCS,
            )),
            None => self.cmd(xe3!(Cmd3dstateStencilBuffer, surface_type = SURFTYPE_NULL, mocs = MOCS)),
        }
        self.cmd(xe3!(Cmd3dstateHierDepthBuffer, mocs = MOCS));
        self.cmd(xe3!(CmdPipeControl, command_streamer_stall_enable = true));

        self.cmd(xe3!(Cmd3dstateVfTopology, primitive_topology_type = p.topology.isl()));
        let vbs: Vec<_> = args
            .buffers
            .into_iter()
            .enumerate()
            .map(|(i, (addr, size))| {
                let state = crate::xe3_value!(
                    StateVertexBufferStateFields,
                    vertex_buffer_index = i as u32,
                    buffer_pitch = p.strides[i],
                    address_modify_enable = true,
                    mocs = MOCS,
                    l3_bypass_disable = true,
                    buffer_starting_address = addr,
                    buffer_size = size,
                );
                Fields(crate::xe3_value!(Cmd3dstateVertexBuffersElement, vertex_buffer_state = Fields(state)))
            })
            .collect();
        self.cmd(xe3!(Cmd3dstateVertexBuffers, elements = vbs));
        let elements: Vec<_> = p
            .attrs
            .iter()
            .map(|v| {
                let n = v.format.components();
                let control = |c: u32| if c < n { VFCOMP_STORE_SRC } else if c < 3 { VFCOMP_STORE_0 } else { VFCOMP_STORE_1_FP };
                let element = crate::xe3_value!(
                    StateVertexElementStateFields,
                    vertex_buffer_index = v.buffer,
                    source_element_format = v.format.isl(),
                    source_element_offset = v.offset,
                    valid = true,
                    component_0_control = control(0),
                    component_1_control = control(1),
                    component_2_control = control(2),
                    component_3_control = control(3),
                );
                Fields(crate::xe3_value!(Cmd3dstateVertexElementsElement, element = Fields(element)))
            })
            .collect();
        self.cmd(xe3!(Cmd3dstateVertexElements, elements = elements));
        for (i, v) in p.attrs.iter().enumerate() {
            let on = p.instanced[v.buffer as usize];
            self.cmd(xe3!(Cmd3dstateVfInstancing, vertex_element_index = i as u32, instancing_enable = on, instance_data_step_rate = u32::from(on)));
        }
        self.cmd(xe3!(Cmd3dstateVfSgvs));
        self.cmd(xe3!(Cmd3dstateVfSgvs2));
        self.cmd(xe3!(Cmd3dstateVfStatistics, statistics_enable = true));
        let cut = p.strip_index_format.map(|f| match f {
            IndexFormat::Uint16 => 0xffffu32,
            IndexFormat::Uint32 => u32::MAX,
        });
        self.cmd(xe3!(Cmd3dstateVf, geometry_distribution_enable = true, indexed_draw_cut_index_enable = cut.is_some(), cut_index = cut.unwrap_or(0)));
        self.cmd(xe3!(
            Cmd3dstateVfg,
            list_1_batch_size_scale = 2u32,
            list_2_batch_size_scale = 1u32,
            list_3_batch_size_scale = 2u32,
            strip_batch_size_scale = 3u32,
            patch_batch_size_scale = 1u32,
            patch_batch_size_multiplier = 31u32,
        ));
        self.cmd(xe3!(Cmd3dstatePrimitiveReplication));
        let indexed = args.index.is_some();
        let (access, base) = match args.index {
            Some(ix) => {
                self.cmd(xe3!(
                    Cmd3dstateIndexBuffer,
                    index_format = ix.format,
                    mocs = MOCS,
                    l3_bypass_disable = true,
                    buffer_starting_address = ix.addr,
                    buffer_size = ix.size,
                ));
                (RANDOM, ix.base_vertex)
            }
            None => (SEQUENTIAL, A::from(0u32)),
        };
        if let Some(at) = &args.indirect {
            let regs: &[u32] = if indexed {
                &[PRIM_VERTEX_COUNT, PRIM_INSTANCE_COUNT, PRIM_START_VERTEX, PRIM_BASE_VERTEX, PRIM_START_INSTANCE]
            } else {
                &[PRIM_VERTEX_COUNT, PRIM_INSTANCE_COUNT, PRIM_START_VERTEX, PRIM_START_INSTANCE]
            };
            if at.len() != regs.len() {
                return Err(Error(format!("draw: {} indirect argument dwords, {} wanted", at.len(), regs.len())));
            }
            for (reg, from) in regs.iter().zip(at) {
                self.cmd(xe3!(CmdMiLoadRegisterMem, register_address = *reg, memory_address = from.clone()));
            }
            if !indexed {
                self.cmd(xe3!(CmdMiLoadRegisterImm, register_offset = PRIM_BASE_VERTEX, data_dword = 0u32));
            }
            self.cmd(xe3!(Cmd3dprimitiveExtended, vertex_access_type = access, indirect_parameter_enable = true, predicate_enable = args.predicate.clone()));
            return Ok(());
        }
        self.cmd(xe3!(
            Cmd3dprimitiveExtended,
            vertex_access_type = access,
            vertex_count_per_instance = args.vertices,
            instance_count = args.instances,
            start_vertex_location = args.first,
            base_vertex_location = base,
            predicate_enable = args.predicate.clone(),
        ));
        Ok(())
    }
}

pub struct Render<'r, 'a> {
    rw: Rewriter<'r, 'a>,
    module: OpId,
    cmds: Cmds<'a>,
    textures: Vec<(String, Texture, Option<String>)>,
    samplers: Vec<(String, String)>,
    blends: Vec<(Vec<ColorTarget>, String)>,
}

impl<'a> Render<'_, 'a> {
    fn container(&mut self, block: BlockId, data: OpData<'a>) -> BlockId {
        self.rw.set_end(block);
        let op = self.rw.create_with(data, &[], |b, op| {
            let r = b.add_region(op);
            b.add_block(r, &[]);
        });
        self.rw.body.top_block(op)
    }

    fn memory(&mut self, m: &Memory<'_>) -> Result<BlockId, Error> {
        let top = self.rw.body.top_block(self.module);
        Ok(self.container(top, OpData::head(m)?))
    }

    fn emit(&mut self, block: BlockId, op: &impl mlirpass::TypedOp) -> OpId {
        self.rw.set_end(block);
        self.rw.emit(op)
    }

    fn put(&mut self, block: BlockId, data: OpData<'a>) -> OpId {
        self.rw.set_end(block);
        self.rw.create(data, &[])
    }

    fn named<O: Op>(&self, name: &str) -> Result<OpId, Error> {
        let top = self.rw.body.top_block(self.module);
        let body = &*self.rw.body;
        body.ops(top).find(|o| body[*o].name == O::NAME && body[*o].get(Symbol::ATTR).is_some_and(|s| Symbol::defined(s).is_ok_and(|s| s == name))).ok_or_else(|| Error(format!("no {} @{name}", O::NAME)))
    }

    fn typed<O: Op>(&self, op: OpId) -> Result<O, Error> {
        self.rw.body.head(op)
    }

    fn buffer(&self, a: &Attribute<'_>) -> Result<(String, u32), Error> {
        let name = Symbol::name(a)?.to_owned();
        let size = BufferSize::try_from(&self.typed::<crate::base::Buffer>(self.named::<crate::base::Buffer>(&name)?)?)?.size;
        Ok((name, size))
    }

    fn draw(&mut self, d: &crate::base::Draw<'_>, shared: &[String; 5]) -> Result<(), Error> {
        let pipe = self.named::<crate::base::Pipeline>(Symbol::name(&d.pipeline)?)?;
        let kernels = Kernel::all(self.rw.body, self.module);
        let p = Pipeline::of(self.rw.body, pipe)?.with_kernels(self.rw.body, &kernels)?;
        let buffers = d.vertex_buffers.iter().map(|b| self.buffer(b).map(|(name, size)| (A::sym(name), A::from(size)))).collect::<Result<Vec<_>, _>>()?;
        let mut entries = vec![A::sym(shared[3].clone())];
        for t in d.textures.iter().flatten() {
            let name = Symbol::name(t)?.to_owned();
            let i = self.textures.iter().position(|(n, _, _)| *n == name).ok_or_else(|| Error(format!("draw: no texture @{name}")))?;
            let (_, tex, surface) = &mut self.textures[i];
            entries.push(A::sym(surface.get_or_insert_with(|| self.cmds.surface("texture", A::sym(name), tex)).clone()));
        }
        if entries.len() as u32 != 1 + p.bindings {
            return Err(Error(format!("draw: {} textures for a pipeline that samples {}", entries.len() - 1, p.bindings)));
        }
        let bt = self.cmds.binding_table(entries);
        let samplers = match (p.bindings, self.samplers.iter().find(|(n, _)| *n == p.fragment)) {
            (0, _) => None,
            (_, Some((_, s))) => Some(s.clone()),
            (n, None) => {
                let s = self.cmds.samplers(n, &p.sampler);
                self.samplers.push((p.fragment.clone(), s.clone()));
                Some(s)
            }
        };
        if p.targets != 1 {
            return Err(Error("draw: a module's render pass has one color target (MRT: the runtime's Recorder::targets)".into()));
        }
        let blend = match self.blends.iter().find(|(c, _)| *c == p.colors) {
            Some((_, b)) => b.clone(),
            None => {
                let b = self.cmds.blend(&p.colors, p.alpha_to_coverage);
                self.blends.push((p.colors.clone(), b.clone()));
                b
            }
        };
        let index = match (&d.index_buffer, d.index_format) {
            (None, _) => None,
            (Some(ib), Some(format)) => {
                let (name, size) = self.buffer(ib)?;
                Some(IndexArgs { format: A::from(format.isl()), addr: A::sym(name), size: A::from(size), base_vertex: A::from(d.base_vertex.unwrap_or(0) as u32) })
            }
            (Some(_), None) => return Err(Error("draw: an index buffer without its index_format".into())),
        };
        let (count, first) = match index {
            Some(_) => (d.indices.ok_or_else(|| Error("draw: indexed without indices".into()))?, d.first_index.unwrap_or(0)),
            None => (d.vertices.ok_or_else(|| Error("draw: no vertices".into()))?, d.first_vertex.unwrap_or(0)),
        };
        let args = DrawArgs {
            blend: A::sym(blend),
            bt: A::within("surfaces", bt),
            samplers: samplers.map(A::sym),
            buffers,
            vertices: A::from(count),
            instances: A::from(d.instances.unwrap_or(1)),
            first: A::from(first),
            index,
            depth: None,
            indirect: None,
            predicate: A::from(0u32),
        };
        self.cmds.draw(&p, args, shared)
    }

    pub fn lower(body: &mut Body<'a>, module: OpId) -> Result<(), Error> {
        let top = body.top_block(module);
        let ops = body.block_ops(top);
        let mut quiet = Quiet;
        let mut me = Render { rw: Rewriter::new(body, &mut quiet), module, cmds: Cmds::default(), textures: Vec::new(), samplers: Vec::new(), blends: Vec::new() };
        let code = me.memory(&Memory::named("code", MemType::Streaming))?;
        let batch = me.container(code, OpData::head(&Stream::named("batch"))?);
        let surfaces = me.memory(&Memory { size: Some(u64::from(POOL)), align: Some(u64::from(POOL)), ..Memory::named("surfaces", MemType::Streaming) })?;
        me.cmds.null_surface();
        me.emit(code, &Data { align: Some(4096), ..Data::zeros("fence", 64) });

        for &op in &ops {
            let name = &*me.rw.body[op].name;
            if name == <crate::base::Buffer as Op>::NAME {
                let b = me.typed::<crate::base::Buffer>(op)?;
                let name = Symbol::defined(&b.sym_name)?.to_owned();
                let size = BufferSize::try_from(&b)?;
                let block = me.memory(&Memory::named(name.clone(), if size.readback { MemType::Coherent } else { MemType::Streaming }))?;
                me.emit(block, &Data { bytes: b.data, ..Data::zeros(format!("{name}_data"), u64::from(size.size)) });
            } else if name == <crate::base::Texture as Op>::NAME {
                let t = me.typed::<crate::base::Texture>(op)?;
                let name = Symbol::defined(&t.sym_name)?.to_owned();
                let tex = Texture { samples: t.samples.unwrap_or(1), ..Texture::plain(t.width, t.height, t.format) };
                let block = me.memory(&Memory { size: Some(tex.size()), align: (tex.samples > 1).then_some(0x1_0000), ..Memory::named(name.clone(), MemType::Coherent) })?;
                if let Some(bytes) = t.data {
                    me.emit(block, &Data::bytes(format!("{name}_data"), bytes));
                }
                me.textures.push((name, tex, None));
            }
        }
        for op in &ops {
            if me.rw.body[*op].name == Kernel::NAME {
                me.rw.move_before(*op, code, None);
            }
        }

        me.cmds.context();
        let passes: Vec<OpId> = ops.iter().copied().filter(|o| me.rw.body[*o].name == <crate::base::RenderPass as Op>::NAME).collect();
        for op in passes {
            let rp = me.typed::<crate::base::RenderPass>(op)?;
            if rp.load == Some(LoadOp::Clear) {
                return Err(Error("render pass: clear left for the backend (run detour-expand-clear)".into()));
            }
            let target = Symbol::name(&rp.color)?.to_owned();
            let t = me.textures.iter().find(|(n, _, _)| *n == target).map(|(_, t, _)| *t).ok_or_else(|| Error(format!("render pass: no texture @{target}")))?;
            let shared = me.cmds.pass_state(&target, &t);
            let stats = rp.statistics.as_ref().map(Symbol::name).transpose()?.map(str::to_owned);
            if let Some(q) = &stats {
                me.cmds.statistics(q, 0);
            }
            let inside = me.rw.body.top_block(op);
            for d in me.rw.body.block_ops(inside) {
                if me.rw.body[d].name != <crate::base::Draw as Op>::NAME {
                    return Err(Error(format!("render pass: {} is not a draw", me.rw.body[d].name)));
                }
                let draw = me.typed::<crate::base::Draw>(d)?;
                me.draw(&draw, &shared)?;
            }
            me.cmds.pc_flush();
            if let Some(q) = &stats {
                me.cmds.statistics(q, 48);
            }
        }
        let cmds = core::mem::take(&mut me.cmds);
        for (block, ops) in [(batch, cmds.batch), (code, cmds.code), (surfaces, cmds.surfaces)] {
            for data in ops {
                me.put(block, data);
            }
        }
        for op in ops {
            let name = &*me.rw.body[op].name;
            if name.starts_with("detour.") || (name == "func.func" && Stage::of(me.rw.body, op).is_none()) {
                me.rw.erase_op(op);
            }
        }
        Ok(())
    }
}

struct BufferSize {
    size: u32,
    readback: bool,
}

impl TryFrom<&crate::base::Buffer<'_>> for BufferSize {
    type Error = Error;

    fn try_from(b: &crate::base::Buffer<'_>) -> Result<BufferSize, Error> {
        let data = b.data.as_ref().map(|d| Const::try_from(d).map(|c| c.ty.element_count().unwrap_or(1) as u32 * c.ty.element_type().bit_width().unwrap_or(32) / 8)).transpose().map_err(|()| Error("buffer data: not a dense constant".into()))?;
        let size = b.size.or(data).ok_or_else(|| Error("buffer: no size and no data".into()))?;
        Ok(BufferSize { size, readback: b.readback })
    }
}
