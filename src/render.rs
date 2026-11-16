use std::time::Duration;

use mlirformat::{Attribute, Error};
use mlirpass::*;

use crate::{Bo, Caching, Engine, Fence, Gpu, Host, Native, Queue, Vm, Xe, Xe3Program};

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
/// One page of the pool's size field
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
/// The drawing rectangle covers every pixel the 16-bit coordinates reach
const DRAWING_MAX: u32 = 0xffff;
/// URB: units of 64 bytes the VS's entries take, starting after the push constants (8 KB each stage, 4 units of 8 KB)
const URB_SPACE: u32 = 5888;
const URB_START: u32 = 4;
const PUSH_STAGES: u32 = 4;
const PUSH_KB: u32 = 8;
const PUSH_ALLOC_SUB_OPCODE: u32 = 18;
/// Threads: the VS's and the PS's per pixel shader dispatcher
const VS_THREADS: u32 = 545;
const PS_THREADS: u32 = 127;
/// GRF blocks of 32 registers a thread takes (3: 128 registers)
const REGISTERS_PER_THREAD: u32 = 3;
/// The clip and SF provoking vertices GL ES takes (the last vertex)
const TRIANGLE_PROVOKING: u32 = 2;
const LINE_PROVOKING: u32 = 1;
const POINT_WIDTH_MAX: f32 = 255.875;
const POINT_WIDTH_MIN: f32 = 0.125;
/// Guardband: the clip test's window around the viewport (in NDC)
const GUARDBAND: f32 = 512.0;
/// Every SBE attribute's active components: all four
const COMPONENTS_XYZW: u32 = 3;
const SBE_ATTRIBUTES: usize = 32;
const SBE_SWIZZLES: u32 = 16;
/// The SF's point width comes from the state, the line distance from the true distance
const POINT_WIDTH_FROM_STATE: u32 = 1;
const AA_LINE_DISTANCE_TRUE: u32 = 1;
const POINT_RASTERIZATION_UPPER_RIGHT: u32 = 1;
const LINE_AA_REGION_1_PIXEL: u32 = 1;
/// The WM's barycentric mode: perspective pixel
const BARYCENTRIC_PERSPECTIVE_PIXEL: u32 = 1;
/// Surface state: no swizzle (SCS_RED, SCS_GREEN, SCS_BLUE, SCS_ALPHA)
const SCS: [u32; 4] = [4, 5, 6, 7];
const HALIGN_128B: u32 = 3;
const VALIGN_4: u32 = 1;
const MIP_TAIL_NONE: u32 = 15;
/// PS's SIMD width field: SIMD16
const PS_SIMD16: u32 = 0;
/// How long a draw may take before the GPU is called hung
const TIMEOUT: Duration = Duration::from_secs(5);

/// Surface formats (genxml's SURFACE_FORMAT)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceFormat {
    R32G32B32A32Float,
    R32G32B32Float,
    R32G32Float,
    R32Float,
    R32Uint,
    R8G8B8A8Unorm,
    Raw,
}

impl From<SurfaceFormat> for u32 {
    fn from(f: SurfaceFormat) -> u32 {
        match f {
            SurfaceFormat::R32G32B32A32Float => 0x000,
            SurfaceFormat::R32G32B32Float => 0x040,
            SurfaceFormat::R32G32Float => 0x085,
            SurfaceFormat::R32Float => 0x0d8,
            SurfaceFormat::R32Uint => 0x0d7,
            SurfaceFormat::R8G8B8A8Unorm => 0x0c7,
            SurfaceFormat::Raw => 0x1ff,
        }
    }
}

impl SurfaceFormat {
    pub const fn bytes(self) -> u32 {
        match self {
            SurfaceFormat::R32G32B32A32Float => 16,
            SurfaceFormat::R32G32B32Float => 12,
            SurfaceFormat::R32G32Float => 8,
            SurfaceFormat::R32Float | SurfaceFormat::R32Uint | SurfaceFormat::R8G8B8A8Unorm | SurfaceFormat::Raw => 4,
        }
    }

    /// A vertex attribute of `n` 32-bit floats
    pub fn floats(n: u32) -> Result<SurfaceFormat, Error> {
        match n {
            1 => Ok(SurfaceFormat::R32Float),
            2 => Ok(SurfaceFormat::R32G32Float),
            3 => Ok(SurfaceFormat::R32G32B32Float),
            4 => Ok(SurfaceFormat::R32G32B32A32Float),
            n => Err(Error(format!("a vertex attribute of {n} floats"))),
        }
    }

    const fn components(self) -> u32 {
        match self {
            SurfaceFormat::R32G32B32A32Float | SurfaceFormat::R8G8B8A8Unorm => 4,
            SurfaceFormat::R32G32B32Float => 3,
            SurfaceFormat::R32G32Float => 2,
            SurfaceFormat::R32Float | SurfaceFormat::R32Uint | SurfaceFormat::Raw => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primitive {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
}

impl From<Primitive> for u32 {
    fn from(t: Primitive) -> u32 {
        match t {
            Primitive::Points => XE3__3DPRIM_POINTLIST,
            Primitive::Lines => XE3__3DPRIM_LINELIST,
            Primitive::LineStrip => XE3__3DPRIM_LINESTRIP,
            Primitive::Triangles => XE3__3DPRIM_TRILIST,
            Primitive::TriangleStrip => XE3__3DPRIM_TRISTRIP,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Cull {
    #[default]
    None,
    Front,
    Back,
}

impl From<Cull> for u32 {
    fn from(c: Cull) -> u32 {
        match c {
            Cull::None => XE3_CULLMODE_NONE,
            Cull::Front => XE3_CULLMODE_FRONT,
            Cull::Back => XE3_CULLMODE_BACK,
        }
    }
}

/// A vertex shader input, in location order: which buffer, its format, where in a vertex
#[derive(Clone, Copy, Debug)]
pub struct VertexElement {
    pub buffer: u32,
    pub format: SurfaceFormat,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexBuffer<'d> {
    pub bytes: &'d [u8],
    pub stride: u32,
}

/// A linear color target: its pixels before the draw, and after it
pub struct Target<'d> {
    pub width: u32,
    pub height: u32,
    pub format: SurfaceFormat,
    pub pixels: &'d mut [u8],
}

impl Target<'_> {
    pub const fn pitch(&self) -> u32 {
        self.width * self.format.bytes()
    }
}

/// One draw: everything the commands are made of
pub struct Draw<'d, 't> {
    pub program: &'d Xe3Program<'t>,
    pub buffers: Vec<VertexBuffer<'d>>,
    pub elements: Vec<VertexElement>,
    /// uniform blocks by binding
    pub uniforms: Vec<(u32, &'d [u8])>,
    pub topology: Primitive,
    pub first: u32,
    pub count: u32,
    pub instances: u32,
    /// x, y, width, height in pixels (y from the bottom row of the target's memory)
    pub viewport: [f32; 4],
    pub depth_range: [f32; 2],
    /// x, y, width, height; None: the whole target
    pub scissor: Option<[u32; 4]>,
    pub cull: Cull,
    pub front_ccw: bool,
}

/// Names of what the image places
struct Names;

impl Names {
    const CODE: &str = "code";
    const SURFACES: &str = "surfaces";
    const BATCH: &str = "batch";
    const TARGET: &str = "target";
    const FENCE: &str = "fence";
    const NULL: &str = "null_surface";
    const BT_NULL: &str = "bt_null";
    const BT: &str = "bt";
    const RT: &str = "rt_surface";
    const VS: &str = "vs";
    const FS: &str = "fs";
    const CS: &str = "cs";
}

/// Builds a draw's image in a copy of the program's body
struct Build<'b, 't> {
    body: &'b mut Body<'t>,
    code: BlockId,
    surfaces: BlockId,
    batch: BlockId,
}

impl<'t> Build<'_, 't> {
    fn container(body: &mut Body<'t>, block: BlockId, data: OpData<'t>) -> BlockId {
        let op = body.create(data, &[]);
        let r = body.add_region(op);
        let inner = body.add_block(r, &[]);
        body.insert(op, block, None);
        inner
    }

    fn memory(body: &mut Body<'t>, m: &ImageMemory<'_>) -> Result<BlockId, Error> {
        let top = body.top_block(body.root);
        Ok(Build::container(body, top, OpData::head(m)?))
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

    /// A state placed under `name`, aligned to `align` bytes
    fn state(&mut self, surfaces: bool, name: &str, align: u32, op: &impl TypedOp) -> OpId {
        let o = self.put(if surfaces { self.surfaces } else { self.code }, op);
        self.body[o].set("sym_name", Attribute::str(name));
        self.body[o].set("align", A::from(align).into());
        o
    }

    fn words(&mut self, block: BlockId, name: &str, bytes: &[u8]) -> Result<(), Error> {
        let values = bytes.chunks(4).map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            A::from(u32::from_le_bytes(w)).into()
        });
        self.put(block, &ImageWords::new(A::str(name).into(), values.collect(), None, None));
        Ok(())
    }

    fn pc_flush(&mut self) {
        self.cmd(&xe3!(CmdPipeControl { depth_cache_flush_enable: true, render_target_cache_flush_enable: true, hdc_pipeline_flush_enable: true, command_streamer_stall_enable: true }));
    }

    /// The context Mesa's iris sets up on Xe2+: base addresses 0 (every pointer is an address), the fence, the modes
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
        self.cmd(&xe3!(CmdStateSystemMemFenceAddress { system_memory_fence_address: A::sym(Names::FENCE) }));
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

    fn surface(&mut self, name: &str, base: A, t: &Target<'_>) {
        self.state(true, name, 64, &xe3!(StateRenderSurfaceState {
            surface_type: XE3_SURFTYPE_2D,
            surface_format: u32::from(t.format),
            surface_horizontal_alignment: HALIGN_128B,
            surface_vertical_alignment: VALIGN_4,
            width: t.width - 1,
            height: t.height - 1,
            surface_pitch: t.pitch() - 1,
            mip_tail_start_lod: MIP_TAIL_NONE,
            mocs: MOCS,
            l1_cache_control: XE3_L1CC_WB,
            shader_channel_select_red: SCS[0],
            shader_channel_select_green: SCS[1],
            shader_channel_select_blue: SCS[2],
            shader_channel_select_alpha: SCS[3],
            surface_base_address: base,
        }));
    }

    /// A buffer surface of `size` bytes: its size split over width, height and depth
    fn buffer_surface(&mut self, name: &str, base: A, size: u32) {
        let n = size.max(1).next_multiple_of(4) - 1;
        self.state(true, name, 64, &xe3!(StateRenderSurfaceState {
            surface_type: XE3_SURFTYPE_BUFFER,
            surface_format: u32::from(SurfaceFormat::Raw),
            surface_horizontal_alignment: HALIGN_128B,
            surface_vertical_alignment: VALIGN_4,
            width: n & 0x7f,
            height: (n >> 7) & 0x3fff,
            depth: (n >> 21) & 0x7ff,
            surface_pitch: 0u32,
            mocs: MOCS,
            l1_cache_control: XE3_L1CC_WB,
            surface_base_address: base,
        }));
    }

    fn null_surface(&mut self) {
        self.state(true, Names::NULL, 64, &xe3!(StateRenderSurfaceState { surface_type: XE3_SURFTYPE_NULL, surface_format: u32::from(SurfaceFormat::R32Uint), mocs: MOCS }));
        self.state(true, Names::BT_NULL, 64, &xe3!(StateBindingTableState { surface_state_pointer: A::sym(Names::NULL) }));
    }

    /// The binding table: entry i points at surface `entries[i]` (the null surface where None)
    fn binding_table(&mut self, entries: &[Option<String>]) {
        for (i, e) in entries.iter().enumerate() {
            let at = A::sym(e.as_deref().unwrap_or(Names::NULL));
            let s = xe3!(StateBindingTableState { surface_state_pointer: at });
            match i {
                0 => {
                    self.state(true, Names::BT, 64, &s);
                }
                _ => {
                    let o = self.put(self.surfaces, &s);
                    self.body[o].set("align", A::from(4u32).into());
                }
            }
        }
    }

    fn draw(&mut self, d: &Draw<'_, 't>, t: &Target<'_>) -> Result<(), Error> {
        let program = d.program;
        let vs = program.kernel(ShaderStage::Vertex).ok_or_else(|| Error("no vertex kernel".into()))?;
        let fs = program.kernel(ShaderStage::Fragment).ok_or_else(|| Error("no fragment kernel".into()))?;
        let (vh, fh) = (program.head(vs)?, program.head(fs)?);
        let inputs = vh.inputs.unwrap_or(0);
        let varyings = vh.varyings.unwrap_or(0);
        if inputs as usize != d.elements.len() {
            return Err(Error(format!("the vertex shader reads {inputs} inputs, the draw gives {}", d.elements.len())));
        }
        if fh.targets.unwrap_or(1) != 1 || fh.bindings.unwrap_or(0) != 0 {
            return Err(Error("one color target and no textures, for now".into()));
        }

        let [x, y, w, h] = d.viewport;
        let [near, far] = d.depth_range;
        self.state(false, "cc_viewport", 32, &xe3!(StateCcViewport { minimum_depth: near.min(far), maximum_depth: near.max(far) }));
        let (tw, th) = (t.width as f32, t.height as f32);
        self.state(false, "sf_clip_viewport", 64, &xe3!(StateSfClipViewport {
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
        }));
        let [sx, sy, sw, sh] = d.scissor.unwrap_or([0, 0, t.width, t.height]);
        let (x0, y0, x1, y1) = if sw == 0 || sh == 0 { (1, 1, 0, 0) } else { (sx, sy, sx + sw - 1, sy + sh - 1) };
        self.state(false, "scissor", 64, &xe3!(StateScissorRect { scissor_rectangle_x_min: x0, scissor_rectangle_y_min: y0, scissor_rectangle_x_max: x1, scissor_rectangle_y_max: y1 }));
        self.state(false, "color_calc", 64, &xe3!(StateColorCalcState { alpha_test_format: 1u32 }));
        self.state(false, "blend", 64, &xe3!(StateBlendState { color_dither_enable: true, alpha_test_function: 1u32 }));
        let entry = self.put(self.code, &xe3!(StateBlendStateEntry {
            post_blend_color_clamp_enable: true,
            pre_blend_color_clamp_enable: true,
            color_clamp_range: XE3_COLORCLAMP_RTFORMAT,
            simple_float_blend_enable: true,
        }));
        self.body[entry].set("align", A::from(4u32).into());

        self.surface(Names::RT, A::sym(Names::TARGET), t);
        let mut entries = vec![Some(Names::RT.to_owned())];
        for &(binding, bytes) in &d.uniforms {
            let name = format!("ubo{binding}");
            self.buffer_surface(&format!("{name}_surface"), A::sym(name.clone()), bytes.len() as u32);
            let at = usize::from(XE3_BUFFER_BTI) + binding as usize;
            entries.resize(entries.len().max(at + 1), None);
            entries[at] = Some(format!("{name}_surface"));
        }
        let table = entries.len() as u32;
        self.binding_table(&entries);
        let bt = || A::within(Names::SURFACES, Names::BT);
        let null = || A::within(Names::SURFACES, Names::BT_NULL);
        let vs_buffers = vh.buffers.unwrap_or(0) > 0;

        self.cmd(&xe3!(Cmd3dstateViewportStatePointersCc { cc_viewport_pointer: A::sym("cc_viewport") }));
        self.cmd(&xe3!(Cmd3dstateViewportStatePointersSfClip { sf_clip_viewport_pointer: A::sym("sf_clip_viewport") }));
        self.cmd(&xe3!(Cmd3dstateScissorStatePointers { scissor_rect_pointer: A::sym("scissor") }));
        let units = inputs.max(2 + varyings).div_ceil(4).max(1);
        let entries_urb = URB_SPACE / units / 8 * 8;
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
        self.cmd(&xe3!(Cmd3dstateBlendStatePointers { blend_state_pointer_valid: true, blend_state_pointer: A::sym("blend") }));
        self.cmd(&xe3!(Cmd3dstateCcStatePointers { color_calc_state_pointer_valid: true, color_calc_state_pointer: A::sym("color_calc") }));
        self.cmd(&xe3!(Cmd3dstateConstantAll { shader_update_enable: 0x1fu32, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePoolAlloc { binding_table_pool_base_address: A::sym(Names::SURFACES), binding_table_pool_buffer_size: POOL / POOL_PAGE, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersVs { pointer_to_vs_binding_table: if vs_buffers { bt() } else { null() } }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersHs { pointer_to_hs_binding_table: null() }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersDs { pointer_to_ds_binding_table: null() }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersGs { pointer_to_gs_binding_table: null() }));
        self.cmd(&xe3!(Cmd3dstateBindingTablePointersPs { pointer_to_ps_binding_table: bt() }));
        self.cmd(&xe3!(CmdPipeControl { stall_at_pixel_scoreboard: true, render_target_cache_flush_enable: true }));
        self.cmd(&xe3!(Cmd3dstateMultisample { number_of_multisamples: 0u32 }));
        self.cmd(&xe3!(Cmd3dstateSampleMask { sample_mask: 1u32 }));
        self.cmd(&xe3!(Cmd3dstateVs {
            kernel_start_pointer: A::sym(Names::VS),
            registers_per_thread: REGISTERS_PER_THREAD,
            vertex_urb_entry_read_length: inputs.div_ceil(2).max(1),
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
        self.cmd(&xe3!(Cmd3dstatePs {
            kernel_0_enable: true,
            kernel_0_simd_width: PS_SIMD16,
            kernel_start_pointer_0: A::sym(Names::FS),
            registers_per_thread: REGISTERS_PER_THREAD,
            binding_table_entry_count: table,
            sampler_count: 0u32,
            maximum_number_of_threads_per_psd: PS_THREADS,
            dispatch_grf_start_register_for_constant_setup_data_0: u32::from(if fragment_inputs > 0 { Interpolate::setup(imask) } else { 1 + u8::from(imask) }),
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
            front_winding: u32::from(d.front_ccw),
            cull_mode: u32::from(d.cull),
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
            point_sprite_texture_coordinate_origin: 1u32,
            attribute_swizzle_enable: true,
            force_vertex_urb_entry_read_offset: true,
            force_vertex_urb_entry_read_length: true,
            attribute_active_component_format: vec![Fill::from(COMPONENTS_XYZW); SBE_ATTRIBUTES],
        }));
        let swiz = (0..SBE_SWIZZLES).map(|i| Fields(if i < varyings { StateSfOutputAttributeDetailFields { source_attribute: Some(i.into()), ..Default::default() } } else { Default::default() })).collect::<Vec<_>>();
        self.cmd(&xe3!(Cmd3dstateSbeSwiz { attribute: swiz }));
        self.cmd(&xe3!(Cmd3dstatePsBlend { has_writeable_rt: true }));
        self.cmd(&Cmd3dstateWmDepthStencil::default());
        self.cmd(&Cmd3dstateDepthBounds::default());
        self.cmd(&xe3!(Cmd3dstateDepthBuffer { surface_type: XE3_SURFTYPE_NULL, surface_format: XE3_D32_FLOAT, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateStencilBuffer { surface_type: XE3_SURFTYPE_NULL, mocs: MOCS }));
        self.cmd(&xe3!(Cmd3dstateHierDepthBuffer { mocs: MOCS }));
        self.cmd(&xe3!(CmdPipeControl { command_streamer_stall_enable: true }));

        self.cmd(&xe3!(Cmd3dstateVfTopology { primitive_topology_type: u32::from(d.topology) }));
        let vbs = d.buffers.iter().enumerate().map(|(i, b)| {
            let state = StateVertexBufferStateFields {
                vertex_buffer_index: Some((i as u32).into()),
                buffer_pitch: Some(b.stride.into()),
                address_modify_enable: Some(true.into()),
                mocs: Some(MOCS.into()),
                l3_bypass_disable: Some(true.into()),
                buffer_starting_address: Some(A::sym(format!("vb{i}")).into()),
                buffer_size: Some((b.bytes.len() as u32).into()),
                ..Default::default()
            };
            Fields(Cmd3dstateVertexBuffersElement { vertex_buffer_state: Some(Fields(state)) })
        });
        self.cmd(&xe3!(Cmd3dstateVertexBuffers { elements: vbs.collect::<Vec<_>>() }));
        let elements = d.elements.iter().map(|e| {
            let n = e.format.components();
            let control = |c: u32| if c < n { XE3_VFCOMP_STORE_SRC } else if c < 3 { XE3_VFCOMP_STORE_0 } else { XE3_VFCOMP_STORE_1_FP };
            let state = StateVertexElementStateFields {
                vertex_buffer_index: Some(e.buffer.into()),
                source_element_format: Some(u32::from(e.format).into()),
                source_element_offset: Some(e.offset.into()),
                valid: Some(true.into()),
                component_0_control: Some(control(0).into()),
                component_1_control: Some(control(1).into()),
                component_2_control: Some(control(2).into()),
                component_3_control: Some(control(3).into()),
                ..Default::default()
            };
            Fields(Cmd3dstateVertexElementsElement { element: Some(Fields(state)) })
        });
        self.cmd(&xe3!(Cmd3dstateVertexElements { elements: elements.collect::<Vec<_>>() }));
        for i in 0..d.elements.len() as u32 {
            self.cmd(&xe3!(Cmd3dstateVfInstancing { vertex_element_index: i, instancing_enable: false, instance_data_step_rate: 0u32 }));
        }
        self.cmd(&Cmd3dstateVfSgvs::default());
        self.cmd(&Cmd3dstateVfSgvs2::default());
        self.cmd(&xe3!(Cmd3dstateVfStatistics { statistics_enable: true }));
        self.cmd(&xe3!(Cmd3dstateVf { geometry_distribution_enable: true }));
        self.cmd(&xe3!(Cmd3dstateVfg {
            list_1_batch_size_scale: 2u32,
            list_2_batch_size_scale: 1u32,
            list_3_batch_size_scale: 2u32,
            strip_batch_size_scale: 3u32,
            patch_batch_size_scale: 1u32,
            patch_batch_size_multiplier: 31u32,
        }));
        self.cmd(&Cmd3dstatePrimitiveReplication::default());
        self.cmd(&xe3!(Cmd3dprimitiveExtended {
            vertex_access_type: XE3_SEQUENTIAL,
            vertex_count_per_instance: d.count,
            instance_count: d.instances,
            start_vertex_location: d.first,
            base_vertex_location: 0u32,
        }));
        Ok(())
    }
}

/// The GPU detour submits to: an Xe device
pub struct Device {
    pub xe: Xe,
}

impl Device {
    pub fn open() -> Result<Device, Error> {
        Xe::open_first().map(|xe| Device { xe }).map_err(|e| Error(format!("opening an Xe device: {e}")))
    }

    /// One draw into `target`: the program's kernels, the state and commands placed in one image, run, read back
    pub fn draw(&self, d: &Draw<'_, '_>, target: &mut Target<'_>) -> Result<(), Error> {
        let mut body = d.program.body.clone();
        let image = Device::image(&mut body, d, target)?;
        let pixels = self.run(&image, Names::TARGET)?;
        target.pixels.copy_from_slice(&pixels[..target.pixels.len()]);
        Ok(())
    }

    fn image<'t>(body: &mut Body<'t>, d: &Draw<'_, 't>, target: &Target<'_>) -> Result<mlirpass::Image, Error> {
        let root = body.root;
        let code = Build::memory(body, &ImageMemory::named(Names::CODE, MemType::Streaming))?;
        let batch_memory = Build::memory(body, &ImageMemory::named(Names::BATCH, MemType::Streaming))?;
        let batch = Build::container(body, batch_memory, OpData::head(&ImageStream::named(Names::BATCH))?);
        let surfaces = Build::memory(body, &ImageMemory { size: Some(u64::from(POOL)), align: Some(u64::from(POOL)), ..ImageMemory::named(Names::SURFACES, MemType::Streaming) })?;
        for &k in &d.program.kernels {
            let stage = body.head::<Xe3Kernel>(k)?.stage;
            let name = match stage {
                Some(ShaderStage::Vertex) => Names::VS,
                Some(ShaderStage::Fragment) => Names::FS,
                _ => Names::CS,
            };
            body[k].set("sym_name", Attribute::str(name));
            body.detach(k);
            body.insert(k, code, None);
        }
        let shaders: Vec<OpId> = body.ops(body.top_block(root)).filter(|&o| &*body[o].name == "gpu.module").collect();
        for s in shaders {
            body.detach(s);
        }
        let mut image = Build { body, code, surfaces, batch };
        image.put(code, &ImageData { align: Some(4096), ..ImageData::zeros(Names::FENCE, 64) });
        image.null_surface();
        let target_memory = Build::memory(image.body, &ImageMemory::named(Names::TARGET, MemType::Coherent))?;
        image.words(target_memory, &format!("{}_data", Names::TARGET), target.pixels)?;
        for (i, b) in d.buffers.iter().enumerate() {
            let m = Build::memory(image.body, &ImageMemory::named(format!("vb{i}"), MemType::Streaming))?;
            image.words(m, &format!("vb{i}_data"), b.bytes)?;
        }
        for &(binding, bytes) in &d.uniforms {
            let m = Build::memory(image.body, &ImageMemory::named(format!("ubo{binding}"), MemType::Streaming))?;
            image.words(m, &format!("ubo{binding}_data"), bytes)?;
        }
        image.context();
        image.draw(d, target)?;
        image.pc_flush();
        Xe3Image::end(body, root);
        ImageLayout(&Xe3Image).run(body, root)?;
        XE3_IMAGE.emit(body, root)
    }

    /// The image's segments in their own buffer objects at their addresses, its stream run on the render engine
    fn run(&self, image: &mlirpass::Image, readback: &str) -> Result<Vec<u8>, Error> {
        let io = |e: std::io::Error| Error(e.to_string());
        let xe = &self.xe;
        let vm = xe.vm().map_err(io)?;
        let queue = xe.queue(&vm, Engine::Render).map_err(io)?;
        let mut bos = Vec::new();
        for s in &image.segments {
            let mem = crate::MemType::from(s.mem);
            let caching = if mem == crate::MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
            let mut bo = xe.bo((s.bytes.len() as u64, caching).into()).map_err(io)?;
            bo.bytes().map_err(io)?[..s.bytes.len()].copy_from_slice(&s.bytes);
            let done = xe.fence(false).map_err(io)?;
            vm.bind(&bo, 0, s.addr, bo.size(), mem, &[&done]).map_err(io)?;
            if !done.wait(TIMEOUT).map_err(io)? {
                return Err(Error(format!("binding {} timed out", s.name)));
            }
            bos.push((s.name.as_str(), bo));
        }
        Native::fence();
        let done = xe.fence(false).map_err(io)?;
        queue.submit(&image.stream, &[], &[&done]).map_err(io)?;
        if !done.wait(TIMEOUT).map_err(io)? {
            return Err(Error("the draw did not finish: the GPU hung".into()));
        }
        if let Some(fault) = queue.fault(&done).map_err(io)? {
            return Err(Error(format!("GPU page fault at {fault:#x}")));
        }
        let (_, bo) = bos.iter_mut().find(|(n, _)| *n == readback).ok_or_else(|| Error(format!("no segment {readback}")))?;
        Ok(bo.bytes().map_err(io)?.to_vec())
    }
}
