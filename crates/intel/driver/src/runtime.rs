//! ⭐ The runtime: what stays between frames, and recording a frame without compiling anything.
//!
//! ```text
//!   Context      one VM, one render queue; a code heap (kernels, dynamic state, the fence) and the binding table pool,
//!                bound once; a ring of batch buffers; the context commands, filled once
//!   Pipeline     compiled once (the MLIR pipeline): its kernels in the code heap, all of a draw's commands as one
//!                template whose holes are the vertex buffers, the counts and the target's state
//!   Target       a render target in its own buffer object (exportable as a dma-buf), its state in the heaps
//!   Buffer       vertices (or what a compute kernel reads and writes), in their own buffer object
//!   Kernel       a compute shader compiled once, in the code heap
//!   Recorder     a frame: templates copied and filled (no MLIR, no attribute read), submitted to a free ring slot
//! ```
//!
//! anv does the same: a pipeline's state packed once at creation, merged with what changes at draw time.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::time::Duration;

use crate::base::{Attrs, Bo, ClearPipeline, DepthState, CompareFunction, StencilOperation, StencilState, Heads, Stage, Symbol, Caching, Engine, FRONTEND, Fence, Gpu, Host, ImageTarget, IndexFormat, MemType, Queue, Template, TextureFormat, Vm};
use crate::xe3_codegen::{A, KERNELS, Cmds, Codegen, DepthTarget, StencilTarget, DrawArgs, Heap, IndexArgs, Isl, Pipeline as Layout, Texture, TextureKind, Xe3};
use crate::xe3_dialect::XE3;
use mlirpass::OpData;

use crate::xe3::{Canvas, Error};

/// Where buffer objects start in the GPU's address space (below 4 GB: state pointers are 32 bits).
const BASE: u64 = 0x10_0000;
/// Buffer objects are this aligned.
const PAGE: u64 = 0x1_0000;
const CODE_SIZE: u64 = 1 << 20;
/// The binding table pool (render surface states too).
const POOL_SIZE: u64 = crate::xe3_codegen::POOL as u64;
const RING: usize = 4;
const BATCH_SIZE: u64 = 1 << 20;
/// The draw template's names for the render pass state (`Cmds::pass_state` makes them per target).
const SHARED: [&str; 5] = ["cc", "sf", "ccs", "rss", "sc"];
/// Each ring slot's binding tables (written at submit, in the pool).
const TABLES: u64 = 0x2_0000;
/// a compute engine frame's binding table area (its dispatches' tables only)
const COMPUTE_TABLES: u64 = 0x4000;

/// A buffer object bound at a fixed address, with a bump allocator in it.
struct Region<'d, G: Gpu<Cmd = u64> + 'd> {
    bo: G::Bo<'d>,
    addr: u64,
    used: u64,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Region<'d, G> {
    fn alloc(&mut self, size: u64, align: u64) -> Result<u64, Error> {
        let at = self.used.next_multiple_of(align.max(1));
        if at + size > self.bo.size() {
            return Err(Error::msg(format!("heap at {:#x} full ({} bytes)", self.addr, self.bo.size())));
        }
        self.used = at + size;
        Ok(self.addr + at)
    }

    fn write(&mut self, addr: u64, bytes: &[u8]) -> Result<(), Error> {
        let at = (addr - self.addr) as usize;
        self.bo.bytes()?[at..at + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
}

/// One slot of the batch ring.
struct Slot<'d, G: Gpu<Cmd = u64> + 'd> {
    region: Region<'d, G>,
    fence: G::Fence<'d>,
    busy: bool,
}

/// ⭐ A render target: its own buffer object, its state (viewports, blend, surface, binding table) in the heaps.
pub struct Target<'d, G: Gpu<Cmd = u64> + 'd> {
    bo: G::Bo<'d>,
    pub addr: u64,
    pub texture: Texture,
    /// the pass state's addresses, in `SHARED` order (the last one the target's surface state)
    state: [u64; 5],
}

/// ⭐ A depth32float depth attachment (Tile4, GPU-only), maybe with a stencil8 (depth32float-stencil8: its own Tile4
/// surface after the depth's): what a pipeline with a depth (stencil) test draws with, next to its target.
pub struct DepthBuffer<'d, G: Gpu<Cmd = u64> + 'd> {
    _bo: G::Bo<'d>,
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pitch: u32,
    /// the stencil8's address and row pitch
    stencil: Option<(u64, u32)>,
    /// its depth as a texture to sample (`texture_depth_2d`)
    surface: u64,
}

/// ⭐ What a draw samples: a texture ([`Image2d`]), or a depth attachment's depth ([`DepthBuffer`], after
/// [`Recorder::depth_barrier`]).
pub trait Sampled {
    /// Its surface state's address.
    fn surface(&self) -> u64;
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Sampled for Image2d<'d, G> {
    fn surface(&self) -> u64 {
        self.surface
    }
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Sampled for DepthBuffer<'d, G> {
    fn surface(&self) -> u64 {
        self.surface
    }
}

/// The pass's depth attachment as the draws fill it in.
#[derive(Clone, Copy)]
struct Attached {
    addr: u64,
    pitch: u32,
    width: u32,
    height: u32,
    stencil: Option<(u64, u32)>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Target<'d, G> {
    pub fn bytes(&mut self) -> Result<&mut [u8], Error> {
        Ok(self.bo.bytes()?)
    }

    /// As a dma-buf (linear, `width * 4` bytes a row), for the compositor.
    pub fn export(&self) -> Result<OwnedFd, Error> {
        Ok(self.bo.export()?)
    }
}

/// A dma-buf imported as a canvas ([`Context::import_canvas`]): bound while this lives.
pub struct ImportedCanvas<'d, G: Gpu<Cmd = u64> + 'd> {
    _bo: G::Bo<'d>,
    pub canvas: Canvas,
}

/// A texture a fragment shader samples: its own buffer object (linear texels), its surface state in the pool.
pub struct Image2d<'d, G: Gpu<Cmd = u64> + 'd> {
    bo: G::Bo<'d>,
    pub addr: u64,
    pub texture: Texture,
    surface: u64,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Image2d<'d, G> {
    pub fn bytes(&mut self) -> Result<&mut [u8], Error> {
        Ok(self.bo.bytes()?)
    }
}

/// Vertices (or anything a draw reads) in their own buffer object.
pub struct Buffer<'d, G: Gpu<Cmd = u64> + 'd> {
    bo: G::Bo<'d>,
    pub addr: u64,
    pub size: u64,
    /// its buffer surface state (for `Recorder::bind`: a shader's `detour.load` / `store {binding}`)
    surface: u64,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Buffer<'d, G> {
    pub fn bytes(&mut self) -> Result<&mut [u8], Error> {
        Ok(self.bo.bytes()?)
    }
}

/// ⭐ A compiled pipeline: its kernels' addresses, its sampler states, the template of one draw.
pub struct Pipeline {
    pub layout: Layout,
    draw: Template,
    /// where each hole of `draw` gets its value
    fills: Vec<Fill>,
    /// the indexed draw (`3DSTATE_INDEX_BUFFER`, RANDOM access) and its holes
    indexed: Template,
    indexed_fills: Vec<Fill>,
    /// the indirect draws (sequential, indexed: their arguments loaded from memory) and their holes
    indirect: [(Template, Vec<Fill>); 2],
}

/// Where a hole of a pipeline's draw gets its value, worked out when the pipeline is made.
#[derive(Debug)]
enum Fill {
    /// the target's pass state (`SHARED` order)
    State(usize),
    Vertices,
    Instances,
    /// the draw's binding table (an offset in the pool)
    Table,
    /// vertex buffer i: its address, or (`true`) its size
    Buffer(usize, bool),
    /// the first vertex (the first index when indexed)
    First,
    /// the index buffer: its address, or (`true`) its size
    Index(bool),
    IndexFormat,
    BaseVertex,
    /// the depth attachment: its address, a row's bytes − 1, width − 1, height − 1; its stencil's address, row bytes − 1
    Depth(DepthField),
    /// the stencil reference value
    Reference,
    /// whether the draw tests the predicate (inside a conditional rendering)
    Predicate,
    /// dword k of an indirect draw's arguments: its address
    Indirect(u64),
    Value(u64),
}

#[derive(Debug, Clone, Copy)]
enum DepthField {
    Addr,
    Pitch,
    Width,
    Height,
    Stencil,
    StencilPitch,
}

/// ⭐ A compiled compute shader: its dispatches (COMPUTE_WALKER and the flush after it) packed once when it is made,
/// what a dispatch knows (the grid, the parameters, the binding table, the indirect arguments) left as holes.
pub struct Kernel {
    direct: (Template, Vec<Walk>),
    /// with a binding table (storage textures)
    bound: (Template, Vec<Walk>),
    /// the grid loaded from memory
    indirect: (Template, Vec<Walk>),
}

/// Where a hole of a kernel's dispatch gets its value, worked out when the kernel is made.
#[derive(Debug, Clone, Copy)]
enum Walk {
    /// workgroups along x, y
    Groups(usize),
    /// inline parameter k (0 past the ones given)
    Param(usize),
    /// the binding table (an offset in the pool)
    Table,
    /// dword k of the indirect arguments: its address
    Args(u64),
}

/// What one dispatch fills `Walk` holes with.
struct Grid<'p> {
    groups: [u32; 2],
    params: &'p [u32],
    table: u64,
    args: u64,
}

impl Walk {
    /// The holes' names in the templates (`Kernel`), only read when the kernel is made.
    const GROUPS: [&str; 2] = ["groups_x", "groups_y"];
    const PARAM: &str = "param";
    const TABLE: &str = "table";
    const ARGS: &str = "args";

    fn value(self, g: &Grid<'_>) -> u64 {
        match self {
            Walk::Groups(k) => u64::from(g.groups[k]),
            Walk::Param(k) => g.params.get(k).copied().map_or(0, u64::from),
            Walk::Table => g.table,
            Walk::Args(k) => g.args + 4 * k,
        }
    }

    fn params() -> Vec<A> {
        (0..crate::xe3_codegen::COMPUTE_PARAMS).map(|k| A::sym(format!("{}{k}", Self::PARAM))).collect()
    }
}

impl TryFrom<&str> for Walk {
    type Error = Error;

    fn try_from(name: &str) -> Result<Self, Error> {
        let index = |prefix: &str| name.strip_prefix(prefix).and_then(|k| k.parse().ok());
        if let Some(k) = Self::GROUPS.iter().position(|g| *g == name) {
            Ok(Walk::Groups(k))
        } else if name == Self::TABLE {
            Ok(Walk::Table)
        } else if let Some(k) = index(Self::PARAM) {
            Ok(Walk::Param(k))
        } else if let Some(k) = name.strip_prefix(Self::ARGS).and_then(|k| k.parse().ok()) {
            Ok(Walk::Args(k))
        } else {
            Err(Error::msg(format!("kernel: nothing fills @{name}")))
        }
    }
}

impl Kernel {
    fn emit(&(ref t, ref walk): &(Template, Vec<Walk>), dw: &mut Vec<u32>, g: &Grid<'_>) {
        t.emit_indexed(dw, |i| walk[i].value(g));
    }
}

/// ⭐ What stays between frames.
pub struct Context<'d, G: Gpu<Cmd = u64> + 'd, H: Host> {
    gpu: &'d G,
    vm: G::Vm<'d>,
    queue: G::Queue<'d>,
    next: u64,
    code: Region<'d, G>,
    pool: Region<'d, G>,
    /// state symbols ⇒ addresses (context-wide: the fence, the null surface and its table)
    names: BTreeMap<String, u64>,
    /// the null surface state (unbound binding table entries)
    null: u64,
    /// what starts every frame: the context commands, filled
    context: Vec<u32>,
    /// what ends every frame: a flush, the batch's end (an even number of dwords)
    tail: Vec<u32>,
    ring: Vec<Slot<'d, G>>,
    /// each ring slot's binding table area in the pool
    tables: Vec<u64>,
    turn: usize,
    /// the viewports and scissors made, by their values' bits (made once: a frame's are usually the last frame's)
    viewports: BTreeMap<[u32; 8], Viewport>,
    scissors: BTreeMap<[u32; 4], Scissor>,
    constants: BTreeMap<[u32; 4], BlendConstant>,
    /// a RAW buffer surface state with holes (`base`, `w`, `h`, `d`): a binding of part of a buffer, made per draw
    range: Template,
    /// the compute engine's queue (CCS), made by [`open_compute`](Self::open_compute)
    compute: Option<SideQueue<'d, G>>,
    /// the copy engine's queue (BCS), made by [`open_copy`](Self::open_copy)
    copy: Option<SideQueue<'d, G>>,
    host: PhantomData<H>,
}

/// The compute or copy engine's queue: its own ring of batch slots and binding table areas, the commands that start
/// and end its frames.
struct SideQueue<'d, G: Gpu<Cmd = u64> + 'd> {
    queue: G::Queue<'d>,
    ring: Vec<Slot<'d, G>>,
    tables: Vec<u64>,
    context: Vec<u32>,
    tail: Vec<u32>,
    turn: usize,
}

/// Which engine a frame runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// the render engine (RCS): draws and dispatches
    Render,
    /// the compute engine (CCS): dispatches only, beside the render engine's work (async compute)
    Compute,
    /// the copy engine (BCS): blits only ([`Recorder::blit`])
    Copy,
}

/// ⭐ A linear 2D image a blit reads or writes ([`Recorder::blit`]): its address, a row's bytes, its size in texels,
/// a texel's bytes (1, 2, 4, 8 or 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Linear {
    pub addr: u64,
    pub pitch: u32,
    pub width: u32,
    pub height: u32,
    pub bpp: u32,
}

impl<G: Gpu<Cmd = u64>> From<&Image2d<'_, G>> for Linear {
    fn from(i: &Image2d<'_, G>) -> Self {
        Linear { addr: i.addr, pitch: i.texture.layout().0, width: i.texture.width, height: i.texture.height, bpp: i.texture.format.bytes() }
    }
}

/// ⭐ A submitted frame, to wait for or to make another submission wait for ([`Context::submit_waiting`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Done {
    pub lane: Lane,
    pub slot: usize,
}

/// ⭐ A viewport ([`Context::viewport`]): its CC and SF / clip viewport states, for [`Recorder::viewport`].
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    cc: u64,
    sf: u64,
}

/// What a [`Context::clear_any_pipeline`] clears besides the color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearDepth {
    None,
    Depth,
    DepthStencil,
}

/// ⭐ A blend constant ([`Context::blend_constant`]), for [`Recorder::blend_constant`].
#[derive(Clone, Copy, Debug)]
pub struct BlendConstant(u64);

/// ⭐ A scissor rectangle ([`Context::scissor`]), for [`Recorder::scissor`].
#[derive(Clone, Copy, Debug)]
pub struct Scissor(u64);

impl<'d, G: Gpu<Cmd = u64> + 'd, H: Host> Context<'d, G, H> {
    /// Binds `size` bytes of `bo` at `addr` and waits for the bind.
    fn bind(gpu: &'d G, vm: &G::Vm<'d>, bo: &G::Bo<'d>, addr: u64, size: u64, mem: MemType) -> Result<(), Error> {
        let done = gpu.fence(false)?;
        vm.bind(bo, 0, addr, size, mem, &[&done])?;
        if !done.wait(Duration::from_secs(5))? {
            return Err(Error::Timeout);
        }
        Ok(())
    }

    /// A buffer object of `size` bytes bound at the next free address.
    fn region(gpu: &'d G, vm: &G::Vm<'d>, next: &mut u64, size: u64, caching: Caching, mem: MemType) -> Result<Region<'d, G>, Error> {
        let bo = gpu.bo((size, caching).into())?;
        let addr = next.next_multiple_of(PAGE);
        Self::bind(gpu, vm, &bo, addr, bo.size(), mem)?;
        *next = addr + bo.size();
        Ok(Region { bo, addr, used: 0 })
    }

    pub fn new(gpu: &'d G) -> Result<Self, Error> {
        let vm = gpu.vm()?;
        let queue = gpu.queue(&vm, Engine::Render)?;
        let mut next = BASE;
        let code = Self::region(gpu, &vm, &mut next, CODE_SIZE, Caching::WriteCombined, MemType::Streaming)?;
        let pool = Self::region(gpu, &vm, &mut next, POOL_SIZE, Caching::WriteCombined, MemType::Streaming)?;
        let mut ring = Vec::new();
        for _ in 0..RING {
            let region = Self::region(gpu, &vm, &mut next, BATCH_SIZE, Caching::WriteCombined, MemType::Streaming)?;
            ring.push(Slot { region, fence: gpu.fence(false)?, busy: false });
        }
        let mut me = Context { gpu, vm, queue, next, code, pool, names: BTreeMap::new(), null: 0, context: Vec::new(), tail: Vec::new(), ring, tables: Vec::new(), turn: 0, viewports: BTreeMap::new(), scissors: BTreeMap::new(), constants: BTreeMap::new(), range: Template::default(), compute: None, copy: None, host: PhantomData };
        let mut range = Cmds::default();
        range.buffer_surface_of("range", A::sym("base"), ["w", "h", "d"].map(A::sym));
        me.range = Self::template(&range.surfaces)?;
        let mut flush = Cmds::default();
        flush.pc_flush();
        Self::template(&flush.batch)?.emit(&mut me.tail, |_| 0);
        me.tail.extend(XE3.pack(<crate::xe3_dialect::CmdMiBatchBufferEnd as mlirformat::Op>::NAME, &[])?);
        for _ in 0..RING {
            let t = me.pool.alloc(TABLES, 64)?;
            me.tables.push(t);
        }
        let fence = me.code.alloc(64, 4096)?;
        me.names.insert("fence".into(), fence);
        me.names.insert("surfaces".into(), me.pool.addr);
        let mut cmds = Cmds::default();
        let null = cmds.null_surface();
        cmds.context();
        let context = Self::template(&cmds.batch)?;
        let placed = me.place(&cmds)?;
        me.null = placed[&null];
        me.names.extend(placed);
        if let Some(h) = context.holes.iter().find(|h| me.resolve(&BTreeMap::new(), &h.name).is_none()) {
            return Err(Error::msg(format!("context: nothing fills @{}", h.name)));
        }
        // every name it uses is placed by now and stays put: the same dwords every frame
        context.emit(&mut me.context, |n| me.names.get(n).copied().unwrap_or(0));
        H::fence();
        Ok(me)
    }

    /// The ops as one template.
    fn template(ops: &[OpData<'_>]) -> Result<Template, Error> {
        let mut t = Template::default();
        for op in ops {
            t.extend(&XE3.template(&op.name, &op.all_attrs())?);
        }
        Ok(t)
    }

    /// What a symbol is: an address, or (`pool::x`, `surfaces::x`) an offset in the binding table pool.
    fn resolve(&self, local: &BTreeMap<String, u64>, name: &str) -> Option<u64> {
        let get = |n: &str| local.get(n).or_else(|| self.names.get(n)).copied();
        match name.split_once("::") {
            Some((_, x)) => get(x).map(|a| a - self.pool.addr),
            None => get(name),
        }
    }

    /// Where a heap's state goes.
    fn heap(&mut self, heap: Heap) -> &mut Region<'d, G> {
        match heap {
            Heap::Code => &mut self.code,
            Heap::Surfaces => &mut self.pool,
        }
    }

    /// Places the state ops of `cmds` in the heaps (in order, each filled from what is placed before it); their
    /// names and addresses.
    fn place(&mut self, cmds: &Cmds<'_>) -> Result<BTreeMap<String, u64>, Error> {
        let mut local = BTreeMap::new();
        for (heap, ops) in [(Heap::Code, &cmds.code), (Heap::Surfaces, &cmds.surfaces)] {
            for op in ops {
                let attrs = op.all_attrs();
                let a = Attrs(&attrs);
                let t = XE3.template(&op.name, &attrs)?;
                let align = a.int("align")?.unwrap_or(64) as u64;
                let addr = self.heap(heap).alloc(t.dw.len() as u64 * 4, align)?;
                if let Ok(name) = a.str("sym_name") {
                    local.insert(name.to_owned(), addr);
                }
                if let Some(h) = t.holes.iter().find(|h| self.resolve(&local, &h.name).is_none()) {
                    return Err(Error::msg(format!("{}: @{} is not placed", op.name, h.name)));
                }
                let mut dw = Vec::new();
                t.emit(&mut dw, |n| self.resolve(&local, n).unwrap_or(0));
                self.heap(heap).write(addr, &le_bytes(&dw))?;
            }
        }
        Ok(local)
    }

    /// ⭐ Compiles pipeline `@name` of `src` (an MLIR module with the `detour.pipeline` and its shaders): its kernels
    /// into the code heap, its draw into a template.
    pub fn pipeline(&mut self, codegen: &Codegen<'_>, src: &str, name: &str) -> Result<Pipeline, Error> {
        self.pipeline_body(codegen, codegen.0.parse(src)?, name)
    }

    /// [`pipeline`](Self::pipeline) of a module already built as IR (the clear pipelines: no text).
    pub fn pipeline_body(&mut self, codegen: &Codegen<'_>, mut body: mlirpass::Body<'_>, name: &str) -> Result<Pipeline, Error> {
        codegen.0.run(&mut body, &format!("builtin.module({FRONTEND})"))?;
        let top = body.top_block(body.root);
        let named = |o: mlirpass::OpId| body[o].name == <crate::base::Pipeline as mlirformat::Op>::NAME && body[o].get(Symbol::ATTR).is_some_and(|s| Symbol::defined(s).is_ok_and(|s| s == name));
        let op = body.ops(top).find(|o| named(*o));
        let op = op.ok_or_else(|| Error::msg(format!("no detour.pipeline @{name}")))?;
        let layout = Layout::of(&body, op)?;
        codegen.0.run(&mut body, &format!("builtin.module({KERNELS})"))?;
        let mut kernels = BTreeMap::new();
        let mut all = Vec::new();
        body.walk(body.root, &mut |o| all.push(o));
        all.retain(|k| body[*k].name == <crate::base::Kernel as mlirformat::Op>::NAME);
        let layout = layout.with_kernels(&body, &all)?;
        for k in all {
            let kname = body.head::<crate::base::Kernel>(k)?.name()?.to_owned();
            if kname != layout.vertex && kname != layout.fragment {
                continue;
            }
            let bytes = Xe3.contents(&body, k)?;
            let addr = self.code.alloc(bytes.len() as u64, 64)?;
            self.code.write(addr, &bytes)?;
            kernels.insert(kname, addr);
        }
        let samplers = if layout.bindings > 0 {
            let mut cmds = Cmds::default();
            let sym = cmds.samplers(layout.bindings, &layout.sampler);
            self.place(&cmds)?[&sym]
        } else {
            0
        };
        let mut cmds = Cmds::default();
        let sym = cmds.blend(&layout.colors, layout.alpha_to_coverage);
        let blend = self.place(&cmds)?[&sym];
        // the same draw four times: sequential, and indexed (its index buffer, format and base vertex holes too), each
        // direct and indirect (its arguments' dwords holes)
        let args = |indexed: bool, indirect: bool| DrawArgs {
            blend: A::sym("blend"),
            bt: A::within("surfaces", "bt"),
            samplers: (layout.bindings > 0).then(|| A::sym("samplers")),
            buffers: (0..layout.strides.len()).map(|i| (A::sym(format!("vb{i}")), A::sym(format!("vb{i}_size")))).collect(),
            vertices: A::sym("vertices"),
            instances: A::sym("instances"),
            first: A::sym("first"),
            index: indexed.then(|| IndexArgs { format: A::sym("index_format"), addr: A::sym("ib"), size: A::sym("ib_size"), base_vertex: A::sym("base_vertex") }),
            depth: layout.depth.map(|d| DepthTarget {
                addr: A::sym("depth"),
                pitch: A::sym("depth_pitch"),
                width: A::sym("depth_width"),
                height: A::sym("depth_height"),
                stencil: d.stencil.map(|_| StencilTarget { addr: A::sym("stencil"), pitch: A::sym("stencil_pitch"), reference: A::sym("stencil_reference") }),
            }),
            indirect: indirect.then(|| (0..4 + u64::from(indexed)).map(|k| A::sym(format!("indirect{k}"))).collect()),
            predicate: A::sym("predicate"),
        };
        let template = |indexed: bool, indirect: bool| -> Result<Template, Error> {
            let mut cmds = Cmds::default();
            cmds.draw(&layout, args(indexed, indirect), &SHARED.map(String::from))?;
            Self::template(&cmds.batch)
        };
        let (draw, indexed) = (template(false, false)?, template(true, false)?);
        let (indirect, indexed_indirect) = (template(false, true)?, template(true, true)?);
        let fill = |n: &str| -> Result<Fill, Error> {
            if let Some(i) = SHARED.iter().position(|s| *s == n.trim_start_matches("surfaces::")) {
                return Ok(Fill::State(i));
            }
            Ok(match n {
                "vertices" => Fill::Vertices,
                "instances" => Fill::Instances,
                "first" => Fill::First,
                "ib" => Fill::Index(false),
                "ib_size" => Fill::Index(true),
                "index_format" => Fill::IndexFormat,
                "base_vertex" => Fill::BaseVertex,
                "depth" => Fill::Depth(DepthField::Addr),
                "depth_pitch" => Fill::Depth(DepthField::Pitch),
                "depth_width" => Fill::Depth(DepthField::Width),
                "depth_height" => Fill::Depth(DepthField::Height),
                "stencil" => Fill::Depth(DepthField::Stencil),
                "stencil_pitch" => Fill::Depth(DepthField::StencilPitch),
                "stencil_reference" => Fill::Reference,
                "predicate" => Fill::Predicate,
                "surfaces::bt" => Fill::Table,
                n if let Some(k) = n.strip_prefix("indirect").and_then(|k| k.parse().ok()) => Fill::Indirect(k),
                "samplers" => Fill::Value(samplers),
                "blend" => Fill::Value(blend),
                _ => {
                    if let Some(&addr) = kernels.get(n) {
                        return Ok(Fill::Value(addr));
                    }
                    if let Some(i) = n.strip_prefix("vb") {
                        let (i, size) = i.strip_suffix("_size").map_or((i, false), |i| (i, true));
                        if let Ok(i) = i.parse() {
                            return Ok(Fill::Buffer(i, size));
                        }
                    }
                    // context-wide: placed in `new`, never moved
                    Fill::Value(self.resolve(&BTreeMap::new(), n).ok_or_else(|| Error::msg(format!("pipeline @{name}: nothing fills @{n}")))?)
                }
            })
        };
        let fills = draw.holes.iter().map(|h| fill(&h.name)).collect::<Result<_, _>>()?;
        let indexed_fills = indexed.holes.iter().map(|h| fill(&h.name)).collect::<Result<_, _>>()?;
        let indirect_fills = indirect.holes.iter().map(|h| fill(&h.name)).collect::<Result<_, _>>()?;
        let indexed_indirect_fills = indexed_indirect.holes.iter().map(|h| fill(&h.name)).collect::<Result<_, _>>()?;
        H::fence();
        Ok(Pipeline { layout, draw, fills, indexed, indexed_fills, indirect: [(indirect, indirect_fills), (indexed_indirect, indexed_indirect_fills)] })
    }

    /// ⭐ Compiles compute shader `@name` of `src` (a `func.func` with `detour.stage = "compute"`) into the code heap.
    pub fn kernel(&mut self, codegen: &Codegen<'_>, src: &str, name: &str) -> Result<Kernel, Error> {
        self.kernel_body(codegen, codegen.0.parse(src)?, name)
    }

    /// [`kernel`](Self::kernel) of a module already built as IR (`Frost` builds its shaders' ops, no text).
    pub fn kernel_body(&mut self, codegen: &Codegen<'_>, mut body: mlirpass::Body<'_>, name: &str) -> Result<Kernel, Error> {
        codegen.0.run(&mut body, &format!("builtin.module({FRONTEND}, {KERNELS})"))?;
        let mut all = Vec::new();
        body.walk(body.root, &mut |o| all.push(o));
        let compute = |k: mlirpass::OpId| body[k].name == <crate::base::Kernel as mlirformat::Op>::NAME && body.head::<crate::base::Kernel>(k).is_ok_and(|k| k.stage == Some(Stage::Compute) && k.name().is_ok_and(|n| n == name));
        let k = all.into_iter().find(|k| compute(*k));
        let k = k.ok_or_else(|| Error::msg(format!("no compute shader @{name}")))?;
        let head = body.head::<crate::base::Kernel>(k)?;
        let (threads, shared) = (head.threads.unwrap_or(1), head.shared.unwrap_or(0));
        let bytes = Xe3.contents(&body, k)?;
        let addr = self.code.alloc(bytes.len() as u64, 64)?;
        self.code.write(addr, &bytes)?;
        H::fence();
        let at = crate::xe3_codegen::KernelAt { addr: A::from(addr), threads, shared };
        let groups = Walk::GROUPS.map(A::sym);
        let made = |f: &dyn Fn(&mut Cmds<'_>)| -> Result<(Template, Vec<Walk>), Error> {
            let mut cmds = Cmds::default();
            f(&mut cmds);
            let t = Self::template(&cmds.batch)?;
            let walk = t.holes.iter().map(|h| Walk::try_from(h.name.as_str())).collect::<Result<_, _>>()?;
            Ok((t, walk))
        };
        Ok(Kernel {
            direct: made(&|c| c.compute(&at, groups.clone(), Walk::params()))?,
            bound: made(&|c| c.compute_bound(&at, groups.clone(), Walk::params(), A::sym(Walk::TABLE), A::from(self.pool.addr)))?,
            indirect: made(&|c| c.compute_indirect(&at, [0, 1, 2].map(|k| A::sym(format!("{}{k}", Walk::ARGS))), Walk::params()))?,
        })
    }

    /// The pipeline that clears to `rgba`: a triangle over the whole target, the color a constant of its fragment
    /// shader (what `detour-expand-clear` makes, compiled once per color).
    pub fn clear_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, None, None, Some(rgba))
    }

    /// ⭐ The clear pipeline for every color (of `samples` 1 or 4, maybe clearing a depth / depth-stencil attachment too:
    /// `depth`, as [`clear_depth_pipeline`](Self::clear_depth_pipeline) / [`clear_depth_stencil_pipeline`]
    /// (Self::clear_depth_stencil_pipeline) do): the color a [`Recorder::clear`]'s, compiled once.
    pub fn clear_any_pipeline(&mut self, codegen: &Codegen<'_>, samples: u32, depth: ClearDepth) -> Result<Pipeline, Error> {
        let stencil = StencilState { compare: CompareFunction::Always, fail: StencilOperation::Replace, depth_fail: StencilOperation::Replace, pass: StencilOperation::Replace, read_mask: 0xff, write_mask: 0xff, back: None };
        let state = |stencil| DepthState { compare: CompareFunction::Always, write: true, stencil, bias: 0, bias_slope_scale: 0.0, bias_clamp: 0.0, bounds: None };
        let depth = match depth {
            ClearDepth::None => None,
            ClearDepth::Depth => Some(state(None)),
            ClearDepth::DepthStencil => Some(state(Some(stencil))),
        };
        self.clear_pipeline_of(codegen, (samples > 1).then_some(samples), depth, None)
    }

    /// The clear pipeline of a multisampled target of `samples` ([`msaa_target_samples`](Self::msaa_target_samples)).
    pub fn msaa_clear_pipeline_samples(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4], samples: u32) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, Some(samples), None, Some(rgba))
    }

    /// The clear pipeline of a 4x multisampled target ([`msaa_target`](Self::msaa_target)).
    pub fn msaa_clear_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, Some(4), None, Some(rgba))
    }

    /// A clear pipeline ([`ClearPipeline`], built as IR) for an rgba8unorm target of `samples`, maybe with a depth test.
    /// (`rgba` none: the blend constant's, [`Recorder::clear`].)
    fn clear_pipeline_of(&mut self, codegen: &Codegen<'_>, samples: Option<u32>, depth: Option<DepthState>, rgba: Option<[f32; 4]>) -> Result<Pipeline, Error> {
        let mut body = codegen.0.module();
        let root = body.root;
        ClearPipeline { name: "clear", format: TextureFormat::Rgba8Unorm, samples, depth, color: rgba.map(|c| c.map(f64::from)) }.emit(&mut body, root);
        self.pipeline_body(codegen, body, "clear")
    }

    /// A buffer of `bytes` (CPU-written, GPU-read).
    pub fn buffer(&mut self, bytes: &[u8]) -> Result<Buffer<'d, G>, Error> {
        let mut r = Self::region(self.gpu, &self.vm, &mut self.next, bytes.len() as u64, Caching::WriteCombined, MemType::Streaming)?;
        r.write(r.addr, bytes)?;
        let mut cmds = Cmds::default();
        let size = u32::try_from(bytes.len()).map_err(|_| Error::msg("buffer: more than 4 GiB"))?;
        let name = cmds.buffer_surface("buffer", A::sym("data"), size);
        self.names.insert("data".into(), r.addr);
        let placed = self.place(&cmds);
        self.names.remove("data");
        let surface = placed?[&name];
        H::fence();
        Ok(Buffer { bo: r.bo, addr: r.addr, size: bytes.len() as u64, surface })
    }

    /// A buffer of f32s (vertices, instances).
    pub fn floats(&mut self, values: &[f32]) -> Result<Buffer<'d, G>, Error> {
        self.buffer(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
    }

    /// The whole-target triangle the clear pipeline draws.
    pub fn clear_vertices(&mut self) -> Result<Buffer<'d, G>, Error> {
        self.floats(&ClearPipeline::CORNERS)
    }

    /// ⭐ A `width` × `height` depth32float depth attachment (Tile4: rows of 128 bytes, 32 rows at a time).
    pub fn depth_buffer(&mut self, width: u32, height: u32) -> Result<DepthBuffer<'d, G>, Error> {
        let (pitch, rows) = DepthTarget::layout(width, height);
        // a depth surface starts on a 64 KB boundary like the rest (Tile4 needs 4 KB)
        let r = Self::region(self.gpu, &self.vm, &mut self.next, u64::from(pitch) * u64::from(rows), Caching::WriteBack, MemType::Coherent)?;
        let surface = self.depth_surface(r.addr, width, height)?;
        Ok(DepthBuffer { _bo: r.bo, addr: r.addr, width, height, pitch, stencil: None, surface })
    }

    /// ⭐ A `width` × `height` depth32float-stencil8 attachment: the depth, then (64 KB on) its stencil8, both Tile4.
    pub fn depth_stencil_buffer(&mut self, width: u32, height: u32) -> Result<DepthBuffer<'d, G>, Error> {
        let (pitch, rows) = DepthTarget::layout(width, height);
        let (spitch, srows) = DepthTarget::stencil_layout(width, height);
        let depth = (u64::from(pitch) * u64::from(rows)).next_multiple_of(PAGE);
        let r = Self::region(self.gpu, &self.vm, &mut self.next, depth + u64::from(spitch) * u64::from(srows), Caching::WriteBack, MemType::Coherent)?;
        let surface = self.depth_surface(r.addr, width, height)?;
        Ok(DepthBuffer { _bo: r.bo, addr: r.addr, width, height, pitch, stencil: Some((r.addr + depth, spitch)), surface })
    }

    /// The surface state sampling a depth attachment at `addr`.
    fn depth_surface(&mut self, addr: u64, width: u32, height: u32) -> Result<u64, Error> {
        let mut cmds = Cmds::default();
        let name = cmds.depth_surface("depth_texture", A::sym("depth"), width, height);
        self.names.insert("depth".into(), addr);
        let placed = self.place(&cmds);
        self.names.remove("depth");
        H::fence();
        Ok(placed?[&name])
    }

    /// The clear pipeline for a pass with a depth-stencil attachment: color, depth (`clear_depth_vertices`'s z) and
    /// stencil (the draw's reference, `Recorder::stencil_reference`) all written.
    pub fn clear_depth_stencil_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        let stencil = StencilState { compare: CompareFunction::Always, fail: StencilOperation::Replace, depth_fail: StencilOperation::Replace, pass: StencilOperation::Replace, read_mask: 0xff, write_mask: 0xff, back: None };
        self.clear_pipeline_of(codegen, None, Some(DepthState { compare: CompareFunction::Always, write: true, stencil: Some(stencil), bias: 0, bias_slope_scale: 0.0, bias_clamp: 0.0, bounds: None }), Some(rgba))
    }

    /// The clear pipeline for a pass with a depth attachment: the target's color and depth (`clear_depth_vertices`'s z)
    /// both written, whatever was there.
    pub fn clear_depth_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, None, Some(DepthState { compare: CompareFunction::Always, write: true, stencil: None, bias: 0, bias_slope_scale: 0.0, bias_clamp: 0.0, bounds: None }), Some(rgba))
    }

    /// The whole-target triangle at depth `z` (WebGPU's `depthClearValue`).
    pub fn clear_depth_vertices(&mut self, z: f32) -> Result<Buffer<'d, G>, Error> {
        self.floats(&[-1.0, -1.0, z, 1.0, 3.0, -1.0, z, 1.0, -1.0, 3.0, z, 1.0])
    }

    /// ⭐ A render target of `width` × `height` `format` (linear; `mem` the GPU caching: Coherent to read it back on
    /// the CPU, Scanout for the compositor).
    pub fn target(&mut self, width: u32, height: u32, format: TextureFormat, mem: MemType) -> Result<Target<'d, G>, Error> {
        if format.compressed() {
            return Err(Error::msg(format!("target: {format:?} is compressed (sampled only)")));
        }
        let texture = Texture::plain(width, height, format);
        let caching = if mem == MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
        let r = Self::region(self.gpu, &self.vm, &mut self.next, u64::from(width * height * format.bytes()), caching, mem)?;
        let state = self.pass_state(r.addr, &texture)?;
        Ok(Target { bo: r.bo, addr: r.addr, texture, state })
    }

    /// ⭐ A 4x multisampled render target of `width` × `height` `format` (32-bit texels; Tile64, GPU-only): drawn by
    /// pipelines with `samples = 4`, resolved into a plain target with [`Recorder::resolve`].
    pub fn msaa_target(&mut self, width: u32, height: u32, format: TextureFormat) -> Result<Target<'d, G>, Error> {
        self.msaa_target_samples(width, height, format, 4)
    }

    /// ⭐ A multisampled render target of `samples` (2, 4 or 8), as [`msaa_target`](Self::msaa_target).
    pub fn msaa_target_samples(&mut self, width: u32, height: u32, format: TextureFormat, samples: u32) -> Result<Target<'d, G>, Error> {
        if format.bytes() != 4 || ![2, 4, 8].contains(&samples) {
            return Err(Error::msg(format!("msaa_target: {samples} samples of {}-byte texels (2, 4, 8 of 32-bit ones)", format.bytes())));
        }
        let texture = Texture { samples, ..Texture::plain(width, height, format) };
        // Tile64 starts on a 64 KB boundary: every region does (PAGE)
        let r = Self::region(self.gpu, &self.vm, &mut self.next, texture.size(), Caching::WriteCombined, MemType::Streaming)?;
        let state = self.pass_state(r.addr, &texture)?;
        Ok(Target { bo: r.bo, addr: r.addr, texture, state })
    }

    /// The pipeline that resolves a 4x target into a plain one: each pixel the mean of its samples
    /// (`detour.load_sample`), drawn with [`resolve_vertices`](Self::resolve_vertices).
    pub fn resolve_pipeline(&mut self, codegen: &Codegen<'_>) -> Result<Pipeline, Error> {
        self.pipeline(codegen, include_str!("resolve.mlir"), "resolve")
    }

    /// The resolve of a target of `samples` (2, 4 or 8): each pixel the mean of its samples, summed in order then times
    /// 1 / samples (exact for the powers of two).
    pub fn resolve_pipeline_samples(&mut self, codegen: &Codegen<'_>, samples: u32) -> Result<Pipeline, Error> {
        let src = include_str!("resolve.mlir");
        let head = &src[..src.find("func.func @resolve_fs").ok_or_else(|| Error::msg("resolve.mlir: no @resolve_fs"))?];
        let mut fs = String::from("func.func @resolve_fs(%px: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = \"fragment\"} {\n  %f = math.floor %px : vector<2xf32>\n  %xy = arith.fptosi %f : vector<2xf32> to vector<2xi32>\n");
        for k in 0..samples {
            fs += &format!("  %s{k} = arith.constant {k} : i32\n  %c{k} = \"detour.load_sample\"(%xy, %s{k}) {{binding = 0 : i32}} : (vector<2xi32>, i32) -> vector<4xf32>\n");
        }
        fs += "  %t0 = arith.addf %c0, %c1 : vector<4xf32>\n";
        for k in 2..samples {
            fs += &format!("  %t{} = arith.addf %t{}, %c{k} : vector<4xf32>\n", k - 1, k - 2);
        }
        fs += &format!("  %q = arith.constant dense<{:?}> : vector<4xf32>\n  %mean = arith.mulf %t{}, %q : vector<4xf32>\n  return %mean : vector<4xf32>\n}}\n", 1.0 / samples as f32, samples - 2);
        self.pipeline(codegen, &format!("{head}{fs}"), "resolve")
    }

    /// The whole-target triangle of a `width` × `height` resolve: clip-space corners, each with its pixel coordinates.
    pub fn resolve_vertices(&mut self, width: u32, height: u32) -> Result<Buffer<'d, G>, Error> {
        let (w, h) = (width as f32, height as f32);
        self.floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 2.0 * w, 0.0, -1.0, 3.0, 0.0, 1.0, 0.0, 2.0 * h])
    }

    /// ⭐ A viewport (`setViewport`) of a `target`-sized target: `[x, y, w, h]` (pixels, fractional allowed), depth from
    /// `min` to `max`; made once for the same values.
    pub fn viewport(&mut self, rect: [f32; 4], [min, max]: [f32; 2], target: [u32; 2]) -> Result<Viewport, Error> {
        let key = [rect[0].to_bits(), rect[1].to_bits(), rect[2].to_bits(), rect[3].to_bits(), min.to_bits(), max.to_bits(), target[0], target[1]];
        if let Some(v) = self.viewports.get(&key) {
            return Ok(*v);
        }
        let mut cmds = Cmds::default();
        let [cc, sf] = cmds.viewport(rect, [min, max], target);
        let placed = self.place(&cmds)?;
        H::fence();
        let v = Viewport { cc: placed[&cc], sf: placed[&sf] };
        self.viewports.insert(key, v);
        Ok(v)
    }

    /// ⭐ A scissor rectangle (`setScissorRect`): `[x, y, w, h]` pixels (empty: nothing drawn); made once for the same
    /// values.
    pub fn scissor(&mut self, rect: [u32; 4]) -> Result<Scissor, Error> {
        if let Some(s) = self.scissors.get(&rect) {
            return Ok(*s);
        }
        let mut cmds = Cmds::default();
        let name = cmds.scissor(rect);
        let placed = self.place(&cmds)?;
        H::fence();
        let s = Scissor(placed[&name]);
        self.scissors.insert(rect, s);
        Ok(s)
    }

    /// ⭐ A blend constant (`setBlendConstant`): what the `constant` blend factors read; made once for the same values.
    pub fn blend_constant(&mut self, rgba: [f32; 4]) -> Result<BlendConstant, Error> {
        let key = rgba.map(f32::to_bits);
        if let Some(c) = self.constants.get(&key) {
            return Ok(*c);
        }
        let mut cmds = Cmds::default();
        let name = cmds.color_calc(rgba);
        let placed = self.place(&cmds)?;
        H::fence();
        let c = BlendConstant(placed[&name]);
        self.constants.insert(key, c);
        Ok(c)
    }

    /// The pass state of a target at `addr` placed in the heaps: its addresses, in `SHARED` order.
    fn pass_state(&mut self, addr: u64, texture: &Texture) -> Result<[u64; 5], Error> {
        let mut cmds = Cmds::default();
        let shared = cmds.pass_state("target", texture);
        let mut local = BTreeMap::from([("target".to_owned(), addr)]);
        self.names.insert("target".into(), addr);
        local.extend(self.place(&cmds)?);
        self.names.remove("target");
        H::fence();
        Ok(shared.map(|n| local[&n]))
    }

    /// ⭐ A render target in a dma-buf someone else made (a compositor's canvas): linear, `width * 4` bytes a row.
    pub fn import_target(&mut self, fd: BorrowedFd<'_>, width: u32, height: u32, format: TextureFormat) -> Result<Target<'d, G>, Error> {
        let texture = Texture::plain(width, height, format);
        let bo = self.gpu.import(fd)?;
        let need = u64::from(width * height * format.bytes());
        if bo.size() < need {
            return Err(Error::msg(format!("import_target: a {} byte dma-buf for {width}x{height}", bo.size())));
        }
        let addr = self.next.next_multiple_of(PAGE);
        Self::bind(self.gpu, &self.vm, &bo, addr, bo.size().next_multiple_of(self.gpu.info().min_alignment), MemType::Streaming)?;
        self.next = addr + bo.size().next_multiple_of(PAGE);
        let state = self.pass_state(addr, &texture)?;
        Ok(Target { bo, addr, texture, state })
    }

    /// ⭐ A canvas in a dma-buf someone else made (a compositor's): RGBA8 premultiplied, linear, its pixels from byte
    /// `offset`, rows `stride` bytes apart (a multiple of 4), `size` (w, h) pixels.
    pub fn import_canvas(&mut self, fd: BorrowedFd<'_>, offset: u32, stride: u32, size: [u32; 2]) -> Result<ImportedCanvas<'d, G>, Error> {
        if !stride.is_multiple_of(4) || stride / 4 < size[0] {
            return Err(Error::msg(format!("import_canvas: rows of {stride} bytes for {} pixels", size[0])));
        }
        let bo = self.gpu.import(fd)?;
        let need = u64::from(offset) + u64::from(stride) * u64::from(size[1]);
        if bo.size() < need {
            return Err(Error::msg(format!("import_canvas: a {} byte dma-buf, {need} needed", bo.size())));
        }
        // coherent: xe binds someone else's dma-buf (an sg buffer object) only with a coherent PAT
        let addr = self.next.next_multiple_of(PAGE);
        Self::bind(self.gpu, &self.vm, &bo, addr, bo.size().next_multiple_of(self.gpu.info().min_alignment), MemType::Coherent)?;
        self.next = addr + bo.size().next_multiple_of(PAGE);
        Ok(ImportedCanvas { _bo: bo, canvas: Canvas { addr: addr + u64::from(offset), size, stride: stride / 4 } })
    }

    /// ⭐ A texture of `width` × `height` `format` with `texels` (linear, tightly packed: `width` texels a row; a
    /// compressed format's blocks, `width / 4` (rounded up) a row of blocks for each 4 rows), to sample.
    pub fn texture(&mut self, width: u32, height: u32, format: TextureFormat, texels: &[u8]) -> Result<Image2d<'d, G>, Error> {
        let texture = Texture::plain(width, height, format);
        let size = u64::from(format.image_bytes(width, height));
        if texels.len() as u64 != size {
            return Err(Error::msg(format!("texture: {} bytes of texels for {size}", texels.len())));
        }
        let mut r = Self::region(self.gpu, &self.vm, &mut self.next, size, Caching::WriteCombined, MemType::Streaming)?;
        r.write(r.addr, texels)?;
        let mut cmds = Cmds::default();
        let name = cmds.surface("texture", A::sym("image"), &texture);
        self.names.insert("image".into(), r.addr);
        let placed = self.place(&cmds);
        self.names.remove("image");
        let surface = placed?[&name];
        H::fence();
        Ok(Image2d { bo: r.bo, addr: r.addr, texture, surface })
    }

    /// ⭐ A texture with mip levels: `levels[l]` is level l's texels (`max(width >> l, 1)` × `max(height >> l, 1)`,
    /// tightly packed), placed as isl lays out a linear mipmapped 2D surface (`Texture::layout`).
    pub fn texture_levels(&mut self, width: u32, height: u32, format: TextureFormat, levels: &[&[u8]]) -> Result<Image2d<'d, G>, Error> {
        if format.compressed() {
            return Err(Error::msg(format!("{format:?}: compressed textures are one level, one layer for now")));
        }
        let texture = Texture { levels: levels.len() as u32, ..Texture::plain(width, height, format) };
        let (pitch, rows, offsets) = texture.layout();
        let bpp = format.bytes();
        let mut bytes = vec![0u8; (pitch * rows) as usize];
        for (l, (texels, (x, y))) in levels.iter().zip(offsets).enumerate() {
            let (w, h) = ((width >> l).max(1), (height >> l).max(1));
            if texels.len() as u32 != w * h * bpp {
                return Err(Error::msg(format!("texture level {l}: {} bytes for {w}x{h}", texels.len())));
            }
            for row in 0..h {
                let at = ((y + row) * pitch + x * bpp) as usize;
                bytes[at..at + (w * bpp) as usize].copy_from_slice(&texels[(row * w * bpp) as usize..][..(w * bpp) as usize]);
            }
        }
        let mut r = Self::region(self.gpu, &self.vm, &mut self.next, bytes.len() as u64, Caching::WriteCombined, MemType::Streaming)?;
        r.write(r.addr, &bytes)?;
        let mut cmds = Cmds::default();
        let name = cmds.surface("texture", A::sym("image"), &texture);
        self.names.insert("image".into(), r.addr);
        let placed = self.place(&cmds);
        self.names.remove("image");
        let surface = placed?[&name];
        H::fence();
        Ok(Image2d { bo: r.bo, addr: r.addr, texture, surface })
    }

    /// ⭐ A 2D array texture (`texture_2d_array`): `layers[k]` is layer k's texels (`width` × `height`, tightly packed),
    /// one level each, the layers `Texture::array_pitch` rows apart; sampled with (u, v, layer).
    pub fn texture_array(&mut self, width: u32, height: u32, format: TextureFormat, layers: &[&[u8]]) -> Result<Image2d<'d, G>, Error> {
        self.layered(Texture { layers: layers.len() as u32, ..Texture::plain(width, height, format) }, layers)
    }

    /// ⭐ A cube texture (`texture_cube`): `faces` the texels of +x, −x, +y, −y, +z, −z (`size` × `size` each, tightly
    /// packed), one level; sampled with a direction (u, v, w).
    pub fn texture_cube(&mut self, size: u32, format: TextureFormat, faces: [&[u8]; 6]) -> Result<Image2d<'d, G>, Error> {
        self.layered(Texture { layers: 6, kind: TextureKind::Cube, ..Texture::plain(size, size, format) }, &faces)
    }

    /// ⭐ A 3D texture (`texture_3d`): `slices[k]` is depth slice k's texels (`width` × `height`, tightly packed), one
    /// level; sampled at (u, v, w) normalized, filtered in w too.
    pub fn texture_3d(&mut self, width: u32, height: u32, format: TextureFormat, slices: &[&[u8]]) -> Result<Image2d<'d, G>, Error> {
        self.layered(Texture { layers: slices.len() as u32, kind: TextureKind::Volume, ..Texture::plain(width, height, format) }, slices)
    }

    /// ⭐ A 3D texture with mip levels: `levels[l][k]` is level l's depth slice k (`max(width >> l, 1)` ×
    /// `max(height >> l, 1)` texels, tightly packed; `max(depth >> l, 1)` slices, `depth` level 0's). Laid out as Gfx9+
    /// lays out 3D (isl's GFX4_2D: the depth slices `Texture::array_pitch` rows apart, each with its levels, level l in
    /// the first `max(depth >> l, 1)` of them).
    pub fn texture_3d_levels(&mut self, width: u32, height: u32, format: TextureFormat, levels: &[&[&[u8]]]) -> Result<Image2d<'d, G>, Error> {
        let depth = levels.first().map_or(0, |l| l.len());
        if depth == 0 || levels.iter().enumerate().any(|(l, slices)| slices.len() != (depth >> l).max(1)) {
            return Err(Error::msg(format!("texture_3d_levels: {:?} slices a level for depth {depth}", levels.iter().map(|l| l.len()).collect::<Vec<_>>())));
        }
        let bpp = format.bytes();
        let empty: Vec<Vec<u8>> = (0..levels.len()).map(|l| vec![0; (((width >> l).max(1)) * ((height >> l).max(1)) * bpp) as usize]).collect();
        let per_slice: Vec<Vec<&[u8]>> = (0..depth).map(|k| levels.iter().enumerate().map(|(l, slices)| slices.get(k).copied().unwrap_or(&empty[l])).collect()).collect();
        let texture = Texture { layers: depth as u32, levels: levels.len() as u32, kind: TextureKind::Volume, ..Texture::plain(width, height, format) };
        self.layers_of(texture, &per_slice)
    }

    /// `texture`'s layers `layers` (one level each) placed `Texture::array_pitch` rows apart, and its surface state.
    fn layered(&mut self, texture: Texture, layers: &[&[u8]]) -> Result<Image2d<'d, G>, Error> {
        let levels: Vec<Vec<&[u8]>> = layers.iter().map(|l| vec![*l]).collect();
        self.layers_of(texture, &levels)
    }

    /// ⭐ A layered texture with mip levels: a 2D array (`cube` false) or cube(s) (`cube`: 6 layers each, +x −x +y −y
    /// +z −z), `layers[k][l]` layer k's level l (`max(width >> l, 1)` × `max(height >> l, 1)`, tightly packed; every
    /// layer as many levels), each layer laid out as isl's mipmapped 2D image, `Texture::array_pitch` rows apart.
    pub fn texture_layers(&mut self, width: u32, height: u32, format: TextureFormat, cube: bool, layers: &[&[&[u8]]]) -> Result<Image2d<'d, G>, Error> {
        let levels = layers.first().map_or(1, |l| l.len() as u32);
        if layers.iter().any(|l| l.len() as u32 != levels) || (cube && !layers.len().is_multiple_of(6)) {
            return Err(Error::msg(format!("texture_layers: {} layers of {:?} levels{}", layers.len(), layers.iter().map(|l| l.len()).collect::<Vec<_>>(), if cube { " (cubes: 6 a cube)" } else { "" })));
        }
        let kind = if cube { TextureKind::Cube } else { TextureKind::Flat };
        let texture = Texture { layers: layers.len() as u32, levels, kind, ..Texture::plain(width, height, format) };
        let all: Vec<Vec<&[u8]>> = layers.iter().map(|l| l.to_vec()).collect();
        self.layers_of(texture, &all)
    }

    /// `texture`'s layers (`levels[k][l]`: layer k's level l) placed: each layer's levels where `Texture::layout` puts
    /// them, the layers `Texture::array_pitch` rows apart; and its surface state.
    fn layers_of(&mut self, texture: Texture, levels: &[Vec<&[u8]>]) -> Result<Image2d<'d, G>, Error> {
        let (width, height, format) = (texture.width, texture.height, texture.format);
        if format.compressed() {
            return Err(Error::msg(format!("{format:?}: compressed textures are one level, one layer for now")));
        }
        let (pitch, _, offsets) = texture.layout();
        let bpp = format.bytes();
        let layer = texture.array_pitch() * pitch;
        let mut bytes = vec![0u8; texture.size() as usize];
        for (k, ls) in levels.iter().enumerate() {
            for (l, (texels, (x, y))) in ls.iter().zip(&offsets).enumerate() {
                let (w, h) = ((width >> l).max(1), (height >> l).max(1));
                if texels.len() as u32 != w * h * bpp {
                    return Err(Error::msg(format!("texture layer {k} level {l}: {} bytes for {w}x{h}", texels.len())));
                }
                for row in 0..h {
                    let at = (k as u32 * layer + (y + row) * pitch + x * bpp) as usize;
                    bytes[at..at + (w * bpp) as usize].copy_from_slice(&texels[(row * w * bpp) as usize..][..(w * bpp) as usize]);
                }
            }
        }
        let mut r = Self::region(self.gpu, &self.vm, &mut self.next, bytes.len() as u64, Caching::WriteCombined, MemType::Streaming)?;
        r.write(r.addr, &bytes)?;
        let mut cmds = Cmds::default();
        let name = cmds.surface("texture", A::sym("image"), &texture);
        self.names.insert("image".into(), r.addr);
        let placed = self.place(&cmds);
        self.names.remove("image");
        let surface = placed?[&name];
        H::fence();
        Ok(Image2d { bo: r.bo, addr: r.addr, texture, surface })
    }

    /// The ring slot the next frame takes (`wait` it before writing what that frame reads).
    pub fn slot(&self) -> usize {
        self.turn
    }

    /// Ring slots.
    pub fn ring(&self) -> usize {
        RING
    }

    /// ⭐ Starts a frame (for the ring slot `submit` takes next).
    pub fn record(&self) -> Recorder<'_, 'd, G, H> {
        self.recorder(Lane::Render, &self.context, self.turn, self.tables[self.turn], TABLES)
    }

    /// ⭐ The compute engine's queue (CCS), for [`record_compute`](Self::record_compute): made once (later calls do
    /// nothing).
    pub fn open_compute(&mut self) -> Result<(), Error> {
        if self.compute.is_none() {
            let mut cmds = Cmds::default();
            cmds.compute_context();
            let side = self.side(Engine::Compute, &cmds, self.tail.clone())?;
            self.compute = Some(side);
        }
        Ok(())
    }

    /// ⭐ The copy engine's queue (BCS), for [`record_copy`](Self::record_copy): made once.
    pub fn open_copy(&mut self) -> Result<(), Error> {
        if self.copy.is_none() {
            let mut tail = Cmds::default();
            tail.copy_tail();
            let mut dw = Vec::new();
            Self::template(&tail.batch)?.emit(&mut dw, |_| 0);
            let side = self.side(Engine::Copy, &Cmds::default(), dw)?;
            self.copy = Some(side);
        }
        Ok(())
    }

    fn side(&mut self, engine: Engine, context: &Cmds<'_>, tail: Vec<u32>) -> Result<SideQueue<'d, G>, Error> {
        let queue = self.gpu.queue(&self.vm, engine)?;
        let mut ring = Vec::new();
        let mut tables = Vec::new();
        for _ in 0..RING {
            let region = Self::region(self.gpu, &self.vm, &mut self.next, BATCH_SIZE, Caching::WriteCombined, MemType::Streaming)?;
            ring.push(Slot { region, fence: self.gpu.fence(false)?, busy: false });
            tables.push(self.pool.alloc(COMPUTE_TABLES, 64)?);
        }
        let mut dw = Vec::new();
        Self::template(&context.batch)?.emit(&mut dw, |n| self.names.get(n).copied().unwrap_or(0));
        Ok(SideQueue { queue, ring, tables, context: dw, tail, turn: 0 })
    }

    /// ⭐ A frame for the copy engine (blits: [`Recorder::blit`]; ordered against the others' with
    /// [`submit_waiting`](Self::submit_waiting)). After [`open_copy`](Self::open_copy).
    pub fn record_copy(&self) -> Result<Recorder<'_, 'd, G, H>, Error> {
        let c = self.copy.as_ref().ok_or_else(|| Error::msg("record_copy: open_copy first"))?;
        Ok(self.recorder(Lane::Copy, &c.context, c.turn, c.tables[c.turn], COMPUTE_TABLES))
    }

    /// The side queue of `lane` (not the render engine's).
    fn side_of(&self, lane: Lane) -> Option<&SideQueue<'d, G>> {
        match lane {
            Lane::Render => None,
            Lane::Compute => self.compute.as_ref(),
            Lane::Copy => self.copy.as_ref(),
        }
    }

    fn side_mut(&mut self, lane: Lane) -> Option<&mut SideQueue<'d, G>> {
        match lane {
            Lane::Render => None,
            Lane::Compute => self.compute.as_mut(),
            Lane::Copy => self.copy.as_mut(),
        }
    }

    /// ⭐ A frame for the compute engine (async compute: [`submit_waiting`](Self::submit_waiting) orders it against
    /// the render engine's frames): dispatches, barriers and copies by compute, no draws. After
    /// [`open_compute`](Self::open_compute).
    pub fn record_compute(&self) -> Result<Recorder<'_, 'd, G, H>, Error> {
        let c = self.compute.as_ref().ok_or_else(|| Error::msg("record_compute: open_compute first"))?;
        Ok(self.recorder(Lane::Compute, &c.context, c.turn, c.tables[c.turn], COMPUTE_TABLES))
    }

    fn recorder<'s>(&'s self, lane: Lane, context: &[u32], slot: usize, tables: u64, size: u64) -> Recorder<'s, 'd, G, H> {
        let mut dw = Vec::with_capacity(4096);
        dw.extend_from_slice(context);
        Recorder { cx: self, frame: Recorded { dw, tables: Vec::new(), slot, lane }, state: [0; 5], colors: Vec::new(), samples: 1, depth: None, reference: 0, bound: Vec::new(), push: None, predicated: false, next_table: tables, tables_end: tables + size }
    }

    /// ⭐ Submits `frame` to its engine's queue once the submissions `after` are done (from either queue: the GPU
    /// waits, not the CPU); its [`Done`].
    pub fn submit_waiting(&mut self, frame: Recorded, after: &[Done]) -> Result<Done, Error> {
        let lane = frame.lane;
        let Recorded { mut dw, tables, slot, .. } = frame;
        dw.extend_from_slice(self.side_of(lane).map_or(&self.tail, |s| &s.tail));
        if dw.len() % 2 == 1 {
            dw.push(0);
        }
        if dw.len() as u64 * 4 > BATCH_SIZE {
            return Err(Error::msg(format!("a frame of {} dwords is larger than a batch buffer", dw.len())));
        }
        let turn = self.side_of(lane).map_or(self.turn, |s| s.turn);
        if slot != turn {
            return Err(Error::msg("frames are submitted in the order they are recorded"));
        }
        self.wait_done(Done { lane, slot })?;
        for (addr, entries) in tables {
            self.pool.write(addr, &le_bytes(&entries))?;
        }
        let own = match lane {
            Lane::Render => &mut self.ring[slot],
            side => &mut self.side_mut(side).ok_or_else(|| Error::msg("that engine's queue is not open"))?.ring[slot],
        };
        let addr = own.region.addr;
        own.region.write(addr, &le_bytes(&dw))?;
        H::fence();
        let ring_of = |l: Lane| -> Option<&Vec<Slot<'d, G>>> { if l == Lane::Render { Some(&self.ring) } else { self.side_of(l).map(|s| &s.ring) } };
        let waits: Vec<&G::Fence<'d>> = after.iter().filter_map(|d| ring_of(d.lane)?.get(d.slot).filter(|s| s.busy).map(|s| &s.fence)).collect();
        let (queue, own) = match lane {
            Lane::Render => (&self.queue, &self.ring[slot]),
            side => {
                let c = self.side_of(side).ok_or_else(|| Error::msg("that engine's queue is not open"))?;
                (&c.queue, &c.ring[slot])
            }
        };
        queue.submit(&addr, &waits, &[&own.fence])?;
        match lane {
            Lane::Render => {
                self.ring[slot].busy = true;
                self.turn = (self.turn + 1) % RING;
            }
            side => {
                if let Some(c) = self.side_mut(side) {
                    c.ring[slot].busy = true;
                    c.turn = (c.turn + 1) % RING;
                }
            }
        }
        Ok(Done { lane, slot })
    }

    /// ⭐ Waits for a submission ([`Done`]; no-op when it is done), and checks its queue for a fault.
    pub fn wait_done(&mut self, done: Done) -> Result<(), Error> {
        match done.lane {
            Lane::Render => self.wait(done.slot),
            side => {
                let Some(c) = self.side_mut(side) else { return Ok(()) };
                let slot = &mut c.ring[done.slot];
                if slot.busy {
                    if !slot.fence.wait(Duration::from_secs(5))? {
                        return Err(Error::Timeout);
                    }
                    slot.busy = false;
                    if let Some(reason) = c.queue.fault(&slot.fence)? {
                        return Err(Error::Fault(reason));
                    }
                }
                Ok(())
            }
        }
    }

    /// Submits a recorded frame to its ring slot (waiting for that slot's last frame first); the slot, to wait for or
    /// export.
    pub fn submit(&mut self, frame: Recorded) -> Result<usize, Error> {
        if frame.lane != Lane::Render {
            return Err(Error::msg("submit: a compute engine frame (submit_waiting it)"));
        }
        let Recorded { mut dw, tables, slot, .. } = frame;
        if slot != self.turn {
            return Err(Error::msg("frames are submitted in the order they are recorded"));
        }
        dw.extend_from_slice(&self.tail);
        if dw.len() % 2 == 1 {
            dw.push(0);
        }
        let n = self.turn;
        self.turn = (self.turn + 1) % RING;
        self.wait(n)?;
        for (addr, entries) in tables {
            self.pool.write(addr, &le_bytes(&entries))?;
        }
        let slot = &mut self.ring[n];
        if dw.len() as u64 * 4 > BATCH_SIZE {
            return Err(Error::msg(format!("a frame of {} dwords is larger than a batch buffer", dw.len())));
        }
        let addr = slot.region.addr;
        slot.region.write(addr, &le_bytes(&dw))?;
        H::fence();
        self.queue.submit(&addr, &[], &[&slot.fence])?;
        slot.busy = true;
        Ok(n)
    }

    /// Waits for the frame in ring slot `n` (no-op when it is done), and checks the queue for a fault.
    pub fn wait(&mut self, n: usize) -> Result<(), Error> {
        let slot = &mut self.ring[n];
        if slot.busy {
            if !slot.fence.wait(Duration::from_secs(5))? {
                return Err(Error::Timeout);
            }
            slot.busy = false;
            if let Some(reason) = self.queue.fault(&slot.fence)? {
                return Err(Error::Fault(reason));
            }
        }
        Ok(())
    }

    /// The fence of the frame in ring slot `n` as a sync_file (the compositor's acquire fence).
    pub fn export_fence(&self, n: usize) -> Result<OwnedFd, Error> {
        Ok(self.ring[n].fence.export()?)
    }
}

/// Dwords as the GPU reads them (little-endian).
fn le_bytes(dw: &[u32]) -> std::borrow::Cow<'_, [u8]> {
    if cfg!(target_endian = "little") {
        // SAFETY: u32 has no padding and every byte pattern is a valid u8; the length is the dwords' in bytes.
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(dw.as_ptr().cast(), dw.len() * 4) })
    } else {
        std::borrow::Cow::Owned(dw.iter().flat_map(|d| d.to_le_bytes()).collect())
    }
}

/// A recorded frame: its dwords, the binding tables to write (address, entries), its ring slot.
pub struct Recorded {
    pub dw: Vec<u32>,
    tables: Vec<(u64, Vec<u32>)>,
    slot: usize,
    lane: Lane,
}

/// ⭐ A frame being recorded: templates copied and filled.
/// One draw's counts: vertices (or indices), instances, the first vertex (or index), and the index buffer (address,
/// bytes), its format and the base vertex when indexed.
#[derive(Clone, Copy)]
struct Draw {
    count: u32,
    instances: u32,
    first: u32,
    index: Option<((u64, u64), IndexFormat, i32)>,
    /// an indirect draw's arguments: their address
    indirect: Option<u64>,
}

pub struct Recorder<'c, 'd, G: Gpu<Cmd = u64> + 'd, H: Host> {
    cx: &'c Context<'d, G, H>,
    frame: Recorded,
    state: [u64; 5],
    /// the color targets' surface states (binding table entries 0 ..): the first target's alone, or `targets`'
    colors: Vec<u64>,
    /// the target's samples
    samples: u32,
    /// the pass's depth attachment: address, pitch, width, height
    depth: Option<Attached>,
    /// the stencil reference value the draws from here on test against (`setStencilReference`)
    reference: u8,
    /// the buffers bound (their surface states), binding k at entry `BUFFER_BTI + k`
    bound: Vec<u64>,
    /// the push constants set (their surface state, their bytes), at entry `PUSH_BTI`
    push: Option<(u64, u32)>,
    /// inside a conditional rendering: the draws test the predicate
    predicated: bool,
    /// the end of this frame's binding table area
    tables_end: u64,
    next_table: u64,
}

impl<'d, G: Gpu<Cmd = u64> + 'd, H: Host> Recorder<'_, 'd, G, H> {
    /// Draws into `target` from here on (without a depth attachment).
    pub fn target(&mut self, target: &Target<'d, G>) {
        self.targets(&[target]);
    }

    /// ⭐ Draws into `targets` from here on (MRT: a pipeline's result k into target k; one size, one sample count;
    /// without a depth attachment).
    pub fn targets(&mut self, targets: &[&Target<'d, G>]) {
        let Some(first) = targets.first() else { return };
        self.state = first.state;
        self.colors = targets.iter().map(|t| t.state[3]).collect();
        self.samples = first.texture.samples;
        self.depth = None;
    }

    /// ⭐ Resolves 4x target `msaa` into the target (`target`, the same size) with `pipeline`
    /// ([`Context::resolve_pipeline`]) over `vertices` ([`Context::resolve_vertices`]): WebGPU's `resolveTarget`.
    pub fn resolve(&mut self, pipeline: &Pipeline, vertices: &Buffer<'d, G>, msaa: &Target<'d, G>) -> Result<(), Error> {
        if msaa.texture.samples < 2 || pipeline.layout.bindings != 1 {
            return Err(Error::msg("resolve: a multisampled target, a pipeline that loads one texture (resolve_pipeline_samples)"));
        }
        // the target's surface state is what the sampler reads it through
        self.emit(pipeline, &[(vertices.addr, vertices.size)], &[msaa.state[3]], Draw { count: 3, instances: 1, first: 0, index: None, indirect: None })
    }

    /// ⭐ The depth drawn so far readable by the draws after it that sample a [`DepthBuffer`] (a render pass's depth
    /// attachment then sampled: shadow maps).
    pub fn depth_barrier(&mut self) -> Result<(), Error> {
        let mut cmds = Cmds::default();
        cmds.depth_barrier();
        Context::<G, H>::template(&cmds.batch)?.emit(&mut self.frame.dw, |_| 0);
        Ok(())
    }

    /// ⭐ The draws from here on through `viewport` (`setViewport`; a target's draws start through the whole target).
    pub fn viewport(&mut self, viewport: &Viewport) {
        self.state[0] = viewport.cc;
        self.state[1] = viewport.sf;
    }

    /// ⭐ The target cleared to `color` (a render pass's `loadOp: "clear"`): `pipeline` ([`Context::clear_any_pipeline`])
    /// over `corners` ([`Context::clear_vertices`] / `clear_depth_vertices`) with `color` its blend constant; the
    /// draws after it blend with the constant they had.
    pub fn clear(&mut self, pipeline: &Pipeline, corners: &Buffer<'d, G>, color: &BlendConstant) -> Result<(), Error> {
        let kept = self.state[2];
        self.state[2] = color.0;
        let drawn = self.emit(pipeline, &[(corners.addr, corners.size)], &[], Draw { count: 3, instances: 1, first: 0, index: None, indirect: None });
        self.state[2] = kept;
        drawn
    }

    /// ⭐ The draws from here on blending with `constant` (`setBlendConstant`; a target's draws start with 0).
    pub fn blend_constant(&mut self, constant: &BlendConstant) {
        self.state[2] = constant.0;
    }

    /// ⭐ The draws from here on inside `scissor` (`setScissorRect`; a target's draws start inside the whole target).
    pub fn scissor(&mut self, scissor: &Scissor) {
        self.state[4] = scissor.0;
    }

    /// The buffers the draws from here on bind (`setBindGroup`): binding k is `buffers[k]`.
    pub fn bind(&mut self, buffers: &[&Buffer<'d, G>]) {
        self.bound = buffers.iter().map(|b| b.surface).collect();
    }

    /// ⭐ The buffers the draws from here on bind, each part of one (`GPUBufferBinding`'s offset and size; dynamic
    /// offsets): binding k is `size` bytes of `buffer` from `offset` (4-aligned). Their surface states go with the
    /// frame's binding tables.
    pub fn bind_ranges(&mut self, ranges: &[(&Buffer<'d, G>, u64, u64)]) -> Result<(), Error> {
        let mut bound = Vec::with_capacity(ranges.len());
        for &(buffer, offset, size) in ranges {
            if !offset.is_multiple_of(4) || offset + size > buffer.size || size > u64::from(u32::MAX) {
                return Err(Error::msg(format!("bind_ranges: {size} bytes from {offset} of a {}-byte buffer", buffer.size)));
            }
            let at = self.next_table;
            self.next_table = (at + 64).next_multiple_of(64);
            if self.next_table > self.tables_end {
                return Err(Error::msg("a frame's binding tables are full"));
            }
            let [w, h, d] = Cmds::buffer_size(size as u32);
            let mut dw = Vec::new();
            self.cx.range.emit(&mut dw, |n| match n {
                "base" => buffer.addr + offset,
                "w" => w.into(),
                "h" => h.into(),
                _ => d.into(),
            });
            self.frame.tables.push((at, dw));
            bound.push(at);
        }
        self.bound = bound;
        Ok(())
    }

    /// ⭐ The push constants of the draws from here on (`setImmediates`; Vulkan `vkCmdPushConstants`): `data` (a multiple
    /// of 4 bytes, at most 256), copied into the frame now, read by `detour.push_constant {offset}`.
    pub fn push_constants(&mut self, data: &[u8]) -> Result<(), Error> {
        if data.is_empty() || !data.len().is_multiple_of(4) || data.len() > Self::PUSH_BYTES {
            return Err(Error::msg(format!("push_constants: {} bytes (a multiple of 4, 4 ..= {})", data.len(), Self::PUSH_BYTES)));
        }
        let words: Vec<u32> = data.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let at = self.next_table;
        let surface = (at + data.len() as u64).next_multiple_of(64);
        self.next_table = (surface + 64).next_multiple_of(64);
        if self.next_table > self.tables_end {
            return Err(Error::msg("a frame's binding tables are full"));
        }
        let [w, h, d] = Cmds::buffer_size(data.len() as u32);
        let mut dw = Vec::new();
        self.cx.range.emit(&mut dw, |n| match n {
            "base" => at,
            "w" => w.into(),
            "h" => h.into(),
            _ => d.into(),
        });
        self.frame.tables.push((at, words));
        self.frame.tables.push((surface, dw));
        self.push = Some((surface, data.len() as u32));
        Ok(())
    }

    const PUSH_BYTES: usize = 256;

    /// ⭐ Conditional rendering (Vulkan `vkCmdBeginConditionalRenderingEXT`): the draws from here on, until
    /// [`end_conditional`](Self::end_conditional), are skipped unless the 32-bit value at byte `offset` (4-aligned) of
    /// `buffer` is not 0 (`inverted`: unless it is 0), read when the GPU gets here (written earlier in the frame counts).
    pub fn begin_conditional(&mut self, buffer: &Buffer<'d, G>, offset: u64, inverted: bool) -> Result<(), Error> {
        if !offset.is_multiple_of(4) || offset + 4 > buffer.size {
            return Err(Error::msg(format!("begin_conditional: the value at {offset} of a {}-byte buffer", buffer.size)));
        }
        let at = buffer.addr + offset;
        self.put(|c| c.condition(A::from(at), inverted))?;
        self.predicated = true;
        Ok(())
    }

    /// ⭐ The draws from here on unconditional again (`vkCmdEndConditionalRenderingEXT`).
    pub fn end_conditional(&mut self) {
        self.predicated = false;
    }

    /// Draws with `depth` as the depth attachment from here on (after `target`; the same size).
    pub fn depth(&mut self, depth: &DepthBuffer<'d, G>) {
        self.depth = Some(Attached { addr: depth.addr, pitch: depth.pitch, width: depth.width, height: depth.height, stencil: depth.stencil });
    }

    /// The stencil reference value of the draws from here on (`setStencilReference`; 0 at first).
    pub fn stencil_reference(&mut self, reference: u8) {
        self.reference = reference;
    }

    /// One draw of `pipeline`: `vertices` vertices (× `instances`) from `buffers`, sampling `textures` (binding i is
    /// `textures[i]`).
    pub fn draw(&mut self, pipeline: &Pipeline, buffers: &[&Buffer<'d, G>], textures: &[&dyn Sampled], vertices: u32, instances: u32) -> Result<(), Error> {
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        self.draw_at(pipeline, &at, textures, vertices, instances)
    }

    /// `draw` with the vertex buffers as (GPU address, bytes): parts of buffers, e.g. a run of instances.
    pub fn draw_at(&mut self, pipeline: &Pipeline, buffers: &[(u64, u64)], textures: &[&dyn Sampled], vertices: u32, instances: u32) -> Result<(), Error> {
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        self.emit(pipeline, buffers, &surfaces, Draw { count: vertices, instances, first: 0, index: None, indirect: None })
    }

    /// ⭐ An indexed draw (`drawIndexed`): `indices` indices from `first_index` of `index` (its format), each plus
    /// `base_vertex`, picking vertices of `buffers` (× `instances`).
    #[allow(clippy::too_many_arguments, reason = "drawIndexed's parameters")]
    pub fn draw_indexed(
        &mut self,
        pipeline: &Pipeline,
        buffers: &[&Buffer<'d, G>],
        index: (&Buffer<'d, G>, IndexFormat),
        textures: &[&dyn Sampled],
        indices: u32,
        instances: u32,
        first_index: u32,
        base_vertex: i32,
    ) -> Result<(), Error> {
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        let index = Some(((index.0.addr, index.0.size), index.1, base_vertex));
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        self.emit(pipeline, &at, &surfaces, Draw { count: indices, instances, first: first_index, index, indirect: None })
    }

    /// ⭐ An indirect draw (`drawIndirect`): its arguments at byte `offset` of `args`, as WebGPU lays them (vertex
    /// count, instance count, first vertex, first instance: u32s), read by the GPU when it draws.
    pub fn draw_indirect(&mut self, pipeline: &Pipeline, buffers: &[&Buffer<'d, G>], textures: &[&dyn Sampled], (args, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        if !offset.is_multiple_of(4) || offset + 16 > args.size {
            return Err(Error::msg(format!("draw_indirect: arguments at {offset} of a {}-byte buffer", args.size)));
        }
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        self.emit(pipeline, &at, &surfaces, Draw { count: 0, instances: 0, first: 0, index: None, indirect: Some(args.addr + offset) })
    }

    /// ⭐ An indexed indirect draw (`drawIndexedIndirect`): its arguments at byte `offset` of `args` (index count,
    /// instance count, first index, base vertex (i32), first instance), indices from `index`.
    pub fn draw_indexed_indirect(&mut self, pipeline: &Pipeline, buffers: &[&Buffer<'d, G>], index: (&Buffer<'d, G>, IndexFormat), textures: &[&dyn Sampled], (args, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        if !offset.is_multiple_of(4) || offset + 20 > args.size {
            return Err(Error::msg(format!("draw_indexed_indirect: arguments at {offset} of a {}-byte buffer", args.size)));
        }
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        let index = Some(((index.0.addr, index.0.size), index.1, 0));
        self.emit(pipeline, &at, &surfaces, Draw { count: 0, instances: 0, first: 0, index, indirect: Some(args.addr + offset) })
    }

    /// ⭐ A multi-draw indirect (`multiDrawIndirect`, Vulkan `vkCmdDrawIndirect(Count)`): `max_count` indirect draws, their
    /// arguments `stride` bytes apart from byte `offset` of `args` (as [`draw_indirect`](Self::draw_indirect)'s;
    /// `indexed`: [`draw_indexed_indirect`](Self::draw_indexed_indirect)'s, from that index buffer); with `count` (a
    /// buffer and a byte offset) only the first min(that u32, `max_count`), read when the GPU gets here (a predicate;
    /// not inside a conditional rendering).
    #[allow(clippy::too_many_arguments, reason = "multiDrawIndirect's parameters")]
    pub fn multi_draw_indirect(
        &mut self,
        pipeline: &Pipeline,
        buffers: &[&Buffer<'d, G>],
        indexed: Option<(&Buffer<'d, G>, IndexFormat)>,
        textures: &[&dyn Sampled],
        (args, offset, stride): (&Buffer<'d, G>, u64, u64),
        max_count: u32,
        count: Option<(&Buffer<'d, G>, u64)>,
    ) -> Result<(), Error> {
        let size = if indexed.is_some() { 20 } else { 16 };
        if !offset.is_multiple_of(4) || !stride.is_multiple_of(4) || stride < size || offset + stride * u64::from(max_count.saturating_sub(1)) + size > args.size {
            return Err(Error::msg(format!("multi_draw_indirect: {max_count} arguments {stride} bytes apart from {offset} of a {}-byte buffer", args.size)));
        }
        let count = match count {
            Some((c, at)) if !at.is_multiple_of(4) || at + 4 > c.size => return Err(Error::msg(format!("multi_draw_indirect: the count at {at} of a {}-byte buffer", c.size))),
            Some(_) if self.predicated => return Err(Error::msg("multi_draw_indirect: a draw count inside a conditional rendering (one predicate)")),
            Some((c, at)) => Some(c.addr + at),
            None => None,
        };
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        let index = indexed.map(|(b, f)| ((b.addr, b.size), f, 0));
        let was = self.predicated;
        for i in 0..max_count {
            if let Some(count) = count {
                self.put(|c| c.draw_count_predicate(A::from(count), i))?;
                self.predicated = true;
            }
            let drawn = self.emit(pipeline, &at, &surfaces, Draw { count: 0, instances: 0, first: 0, index, indirect: Some(args.addr + offset + stride * u64::from(i)) });
            if drawn.is_err() {
                self.predicated = was;
                return drawn;
            }
        }
        self.predicated = was;
        Ok(())
    }

    /// A draw sampling the surface states `textures`.
    fn emit(&mut self, pipeline: &Pipeline, buffers: &[(u64, u64)], textures: &[u64], d: Draw) -> Result<(), Error> {
        if self.frame.lane != Lane::Render {
            return Err(Error::msg("draw: in a compute engine frame (it has no 3D pipeline)"));
        }
        if pipeline.layout.samples != self.samples {
            return Err(Error::msg(format!("draw: a {}-sample pipeline into a {}-sample target", pipeline.layout.samples, self.samples)));
        }
        if self.colors.len() as u32 != pipeline.layout.targets {
            return Err(Error::msg(format!("draw: a pipeline of {} targets into {}", pipeline.layout.targets, self.colors.len())));
        }
        if textures.len() as u32 != pipeline.layout.bindings {
            return Err(Error::msg(format!("draw: {} textures for a pipeline that samples {}", textures.len(), pipeline.layout.bindings)));
        }
        let buffers_used = pipeline.layout.buffers as usize;
        if self.bound.len() < buffers_used {
            return Err(Error::msg(format!("draw: {} buffers bound for a pipeline that uses {buffers_used}", self.bound.len())));
        }
        // its binding table: the targets, then the textures (surface state addresses; their base is 0), then with buffers
        // the null surface up to BUFFER_BTI and the bound buffers
        let bt = self.next_table;
        let mut entries: Vec<u32> = self.colors.iter().chain(textures).map(|a| *a as u32).collect();
        let push = match (pipeline.layout.push, self.push) {
            (0, _) => None,
            (want, Some((surface, size))) if size >= want => Some(surface),
            (want, set) => return Err(Error::msg(format!("draw: a pipeline reading {want} bytes of push constants, {} set", set.map_or(0, |p| p.1)))),
        };
        if pipeline.layout.reads_memory() {
            let null = self.cx.null as u32;
            entries.resize(usize::from(crate::xe3_codegen::BUFFER_BTI), null);
            if let Some(push) = push {
                entries[usize::from(crate::xe3_codegen::PUSH_BTI)] = push as u32;
            }
            entries.extend(self.bound[..buffers_used].iter().map(|a| *a as u32));
        }
        self.next_table = (bt + entries.len() as u64 * 4).next_multiple_of(64);
        if self.next_table > self.tables_end {
            return Err(Error::msg("a frame's binding tables are full"));
        }
        self.frame.tables.push((bt, entries));
        let bt = bt - self.cx.pool.addr;
        let state = self.state;
        if pipeline.layout.depth.is_some() != self.depth.is_some() {
            return Err(Error::msg("draw: a pipeline's depth test and the pass's depth attachment go together"));
        }
        let (depth, reference, predicated) = (self.depth, self.reference, self.predicated);
        if pipeline.layout.depth.is_some_and(|d| d.stencil.is_some()) && depth.is_some_and(|d| d.stencil.is_none()) {
            return Err(Error::msg("draw: a pipeline's stencil test needs a depth-stencil attachment"));
        }
        let (template, fills) = match (d.index, d.indirect) {
            (Some(_), None) => (&pipeline.indexed, &pipeline.indexed_fills),
            (None, None) => (&pipeline.draw, &pipeline.fills),
            (None, Some(_)) => (&pipeline.indirect[0].0, &pipeline.indirect[0].1),
            (Some(_), Some(_)) => (&pipeline.indirect[1].0, &pipeline.indirect[1].1),
        };
        let value = |i: usize| -> u64 {
            match &fills[i] {
                Fill::State(i) => state[*i],
                Fill::Vertices => d.count.into(),
                Fill::Instances => d.instances.into(),
                Fill::First => d.first.into(),
                Fill::Table => bt,
                Fill::Buffer(i, size) => buffers.get(*i).map_or(0, |b| if *size { b.1 } else { b.0 }),
                Fill::Index(size) => d.index.map_or(0, |((addr, bytes), ..)| if *size { bytes } else { addr }),
                Fill::IndexFormat => d.index.map_or(0, |(_, f, _)| f.isl().into()),
                // a 32-bit two's complement field
                Fill::BaseVertex => d.index.map_or(0, |(.., b)| u64::from(b as u32)),
                Fill::Reference => reference.into(),
                Fill::Predicate => predicated.into(),
                Fill::Indirect(k) => d.indirect.map_or(0, |a| a + 4 * k),
                Fill::Depth(f) => depth.map_or(0, |a| match f {
                    DepthField::Addr => a.addr,
                    DepthField::Pitch => u64::from(a.pitch - 1),
                    DepthField::Width => u64::from(a.width - 1),
                    DepthField::Height => u64::from(a.height - 1),
                    DepthField::Stencil => a.stencil.map_or(0, |s| s.0),
                    DepthField::StencilPitch => a.stencil.map_or(0, |s| u64::from(s.1 - 1)),
                }),
                Fill::Value(v) => *v,
            }
        };
        template.emit_indexed(&mut self.frame.dw, value);
        Ok(())
    }

    /// ⭐ Runs `kernel` over a grid of `groups` (x, y) workgroups of 16 invocations (`detour.workgroup_id {dim}`,
    /// `detour.lane`), its arguments `params` (up to 8 dwords: buffer addresses, sizes ...). What it writes is flushed
    /// before the next command.
    pub fn dispatch(&mut self, kernel: &Kernel, groups: [u32; 2], params: &[u32]) -> Result<(), Error> {
        self.computes()?;
        if params.len() > crate::xe3_codegen::COMPUTE_PARAMS {
            return Err(Error::msg(format!("dispatch: {} parameters, at most {}", params.len(), crate::xe3_codegen::COMPUTE_PARAMS)));
        }
        Kernel::emit(&kernel.direct, &mut self.frame.dw, &Grid { groups, params, table: 0, args: 0 });
        Ok(())
    }

    /// ⭐ [`dispatch`](Self::dispatch) with storage textures: binding k (`detour.texture_store` / `texture_load
    /// {binding = k}`) is `textures[k]`, read and written as its format (typed messages through its surface state).
    /// Sample them after a [`texture_barrier`](Self::texture_barrier).
    pub fn dispatch_with(&mut self, kernel: &Kernel, groups: [u32; 2], params: &[u32], textures: &[&dyn Sampled]) -> Result<(), Error> {
        self.computes()?;
        if params.len() > crate::xe3_codegen::COMPUTE_PARAMS {
            return Err(Error::msg(format!("dispatch: {} parameters, at most {}", params.len(), crate::xe3_codegen::COMPUTE_PARAMS)));
        }
        let bt = self.next_table;
        let entries: Vec<u32> = textures.iter().map(|t| t.surface() as u32).collect();
        self.next_table = (bt + entries.len().max(1) as u64 * 4).next_multiple_of(64);
        if self.next_table > self.tables_end {
            return Err(Error::msg("a frame's binding tables are full"));
        }
        self.frame.tables.push((bt, entries));
        Kernel::emit(&kernel.bound, &mut self.frame.dw, &Grid { groups, params, table: bt - self.cx.pool.addr, args: 0 });
        Ok(())
    }

    /// ⭐ Sets an event (`vkCmdSetEvent`): the u32 at byte `offset` of `buffer` becomes `value` once the work recorded
    /// before is done and flushed. Another queue's frame can [`wait_event`](Self::wait_event) for it, no syncobj between
    /// them (both are submitted; the waiter polls).
    pub fn set_event(&mut self, buffer: &Buffer<'d, G>, offset: u64, value: u32) -> Result<(), Error> {
        let at = Self::event_at(buffer, offset)?;
        let copy = self.frame.lane == Lane::Copy;
        self.put(|c| c.set_event(A::from(at), value, copy))
    }

    /// ⭐ Waits for an event (`vkCmdWaitEvents`): the GPU polls until the u32 at byte `offset` of `buffer` is `value`,
    /// then what is recorded after runs (seeing what was written before it was set).
    pub fn wait_event(&mut self, buffer: &Buffer<'d, G>, offset: u64, value: u32) -> Result<(), Error> {
        let at = Self::event_at(buffer, offset)?;
        let copy = self.frame.lane == Lane::Copy;
        self.put(|c| c.wait_event(A::from(at), value, copy))
    }

    fn event_at(buffer: &Buffer<'d, G>, offset: u64) -> Result<u64, Error> {
        match offset.is_multiple_of(8) && offset + 8 <= buffer.size {
            true => Ok(buffer.addr + offset),
            false => Err(Error::msg(format!("event: byte {offset} of a {}-byte buffer (8-aligned, 8 bytes)", buffer.size))),
        }
    }

    /// Refuses a dispatch in a copy engine frame (it has no EUs).
    fn computes(&self) -> Result<(), Error> {
        match self.frame.lane {
            Lane::Copy => Err(Error::msg("dispatch: in a copy engine frame (blits only)")),
            _ => Ok(()),
        }
    }

    /// ⭐ A rectangle copy on the copy engine (XY_BLOCK_COPY_BLT; a copy engine frame only, [`Context::record_copy`]):
    /// `size` texels from `from` of `src` to `to` of `dst`, both linear 2D images of the same texel size.
    pub fn blit(&mut self, src: Linear, from: [u32; 2], dst: Linear, to: [u32; 2], size: [u32; 2]) -> Result<(), Error> {
        if self.frame.lane != Lane::Copy {
            return Err(Error::msg("blit: only in a copy engine frame (record_copy)"));
        }
        let inside = |l: &Linear, [x, y]: [u32; 2]| x + size[0] <= l.width && y + size[1] <= l.height;
        if src.bpp != dst.bpp || !inside(&src, from) || !inside(&dst, to) {
            return Err(Error::msg(format!("blit: {size:?} texels from {from:?} of {src:?} to {to:?} of {dst:?}")));
        }
        let mut cmds = Cmds::default();
        cmds.block_copy(src.bpp, (A::from(src.addr), src.pitch, [src.width, src.height]), from, (A::from(dst.addr), dst.pitch, [dst.width, dst.height]), to, size)?;
        Context::<G, H>::template(&cmds.batch)?.emit(&mut self.frame.dw, |_| 0);
        Ok(())
    }

    /// ⭐ Storage texture writes before here visible to the samples after (data port flushed, texture cache invalidated).
    pub fn texture_barrier(&mut self) -> Result<(), Error> {
        self.put(|c| c.texture_barrier())
    }

    /// ⭐ An indirect dispatch (`dispatchWorkgroupsIndirect`): the grid's x, y, z workgroup counts the u32s at byte
    /// `offset` of `args`, read when the GPU runs it (z workgroups more rows of the grid: `detour.workgroup_id {dim = 2}`).
    pub fn dispatch_indirect(&mut self, kernel: &Kernel, (args, offset): (&Buffer<'d, G>, u64), params: &[u32]) -> Result<(), Error> {
        self.computes()?;
        if params.len() > crate::xe3_codegen::COMPUTE_PARAMS {
            return Err(Error::msg(format!("dispatch: {} parameters, at most {}", params.len(), crate::xe3_codegen::COMPUTE_PARAMS)));
        }
        if !offset.is_multiple_of(4) || offset + 12 > args.size {
            return Err(Error::msg(format!("dispatch_indirect: arguments at {offset} of a {}-byte buffer", args.size)));
        }
        Kernel::emit(&kernel.indirect, &mut self.frame.dw, &Grid { groups: [0; 2], params, table: 0, args: args.addr + offset });
        Ok(())
    }

    /// Commands made alone (no holes) appended to the frame.
    fn put(&mut self, f: impl FnOnce(&mut Cmds<'_>)) -> Result<(), Error> {
        let mut cmds = Cmds::default();
        f(&mut cmds);
        Context::<G, H>::template(&cmds.batch)?.emit(&mut self.frame.dw, |_| 0);
        Ok(())
    }

    /// The byte `offset` of `buffer` checked to hold `bytes`, 8-byte aligned: its GPU address.
    fn slot(buffer: &Buffer<'d, G>, offset: u64, bytes: u64) -> Result<u64, Error> {
        if !offset.is_multiple_of(8) || offset + bytes > buffer.size {
            return Err(Error::msg(format!("a query result at {offset} of a {}-byte buffer", buffer.size)));
        }
        Ok(buffer.addr + offset)
    }

    /// ⭐ A timestamp query (`writeTimestamp`): the GPU's timestamp (u64, ticks of the GT's reference clock,
    /// `Gt::reference_clock` Hz) written at byte `offset` of `buffer` once the work recorded before it is done.
    pub fn timestamp(&mut self, (buffer, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        let at = Self::slot(buffer, offset, 8)?;
        self.put(|c| c.timestamp(A::from(at)))
    }

    /// ⭐ An occlusion query's begin or end (`beginOcclusionQuery` / `endOcclusionQuery`): the samples that passed
    /// the depth and stencil tests so far (u64) at byte `offset` of `buffer`; the query's result is end − begin.
    pub fn depth_count(&mut self, (buffer, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        let at = Self::slot(buffer, offset, 8)?;
        self.put(|c| c.depth_count(A::from(at)))
    }

    /// ⭐ Pipeline statistics (a `pipeline-statistics` query's begin or end): each of `registers`
    /// (`crate::xe3_codegen::IA_VERTICES_COUNT` ...) a u64 from byte `offset` of `buffer` on, once the work before it is
    /// done; a result is end − begin.
    pub fn statistics(&mut self, registers: &[u32], (buffer, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        let at = Self::slot(buffer, offset, 8 * registers.len() as u64)?;
        self.put(|c| {
            for (k, reg) in registers.iter().enumerate() {
                let to = at + 8 * k as u64;
                c.statistic(*reg, [A::from(to), A::from(to + 4)]);
            }
        })
    }

    pub fn finish(self) -> Recorded {
        self.frame
    }
}
