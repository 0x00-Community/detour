use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::time::Duration;

use crate::base::{Attrs, Bo, ClearPipeline, DepthState, CompareFunction, StencilOperation, StencilState, Heads, Stage, Symbol, Caching, Engine, FRONTEND, Fence, Gpu, Host, ImageTarget, IndexFormat, MemType, Queue, Template, TextureFormat, Vm};
use crate::xe3_codegen::{A, KERNELS, Cmds, Codegen, DepthTarget, StencilTarget, DrawArgs, Heap, IndexArgs, Isl, Pipeline as Layout, Texture, TextureKind, Xe3};
use crate::xe3_dialect::XE3;
use mlirpass::OpData;

use crate::xe3::{Canvas, Error};

const BASE: u64 = 0x10_0000;

const PAGE: u64 = 0x1_0000;
const CODE_SIZE: u64 = 1 << 20;

const POOL_SIZE: u64 = crate::xe3_codegen::POOL as u64;
const RING: usize = 4;
const BATCH_SIZE: u64 = 1 << 20;

const SHARED: [&str; 5] = ["cc", "sf", "ccs", "rss", "sc"];

const TABLES: u64 = 0x2_0000;

const COMPUTE_TABLES: u64 = 0x4000;

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

struct Slot<'d, G: Gpu<Cmd = u64> + 'd> {
    region: Region<'d, G>,
    fence: G::Fence<'d>,
    busy: bool,
}

pub struct Target<'d, G: Gpu<Cmd = u64> + 'd> {
    bo: G::Bo<'d>,
    pub addr: u64,
    pub texture: Texture,
    state: [u64; 5],
}

pub struct DepthBuffer<'d, G: Gpu<Cmd = u64> + 'd> {
    _bo: G::Bo<'d>,
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pitch: u32,
    stencil: Option<(u64, u32)>,
    surface: u64,
}

pub trait Sampled {
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

    pub fn export(&self) -> Result<OwnedFd, Error> {
        Ok(self.bo.export()?)
    }
}

pub struct ImportedCanvas<'d, G: Gpu<Cmd = u64> + 'd> {
    _bo: G::Bo<'d>,
    pub canvas: Canvas,
}

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

pub struct Buffer<'d, G: Gpu<Cmd = u64> + 'd> {
    bo: G::Bo<'d>,
    pub addr: u64,
    pub size: u64,
    surface: u64,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Buffer<'d, G> {
    pub fn bytes(&mut self) -> Result<&mut [u8], Error> {
        Ok(self.bo.bytes()?)
    }
}

pub struct Pipeline {
    pub layout: Layout,
    draw: Template,
    fills: Vec<Fill>,
    indexed: Template,
    indexed_fills: Vec<Fill>,
    indirect: [(Template, Vec<Fill>); 2],
}

#[derive(Debug)]
enum Fill {
    State(usize),
    Vertices,
    Instances,
    Table,
    Buffer(usize, bool),
    First,
    Index(bool),
    IndexFormat,
    BaseVertex,
    Depth(DepthField),
    Reference,
    Predicate,
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

pub struct Kernel {
    direct: (Template, Vec<Walk>),
    bound: (Template, Vec<Walk>),
    indirect: (Template, Vec<Walk>),
}

#[derive(Debug, Clone, Copy)]
enum Walk {
    Groups(usize),
    Param(usize),
    Table,
    Args(u64),
}

struct Grid<'p> {
    groups: [u32; 2],
    params: &'p [u32],
    table: u64,
    args: u64,
}

impl Walk {
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

pub struct Context<'d, G: Gpu<Cmd = u64> + 'd, H: Host> {
    gpu: &'d G,
    vm: G::Vm<'d>,
    queue: G::Queue<'d>,
    next: u64,
    code: Region<'d, G>,
    pool: Region<'d, G>,
    names: BTreeMap<String, u64>,
    null: u64,
    context: Vec<u32>,
    tail: Vec<u32>,
    ring: Vec<Slot<'d, G>>,
    tables: Vec<u64>,
    turn: usize,
    viewports: BTreeMap<[u32; 8], Viewport>,
    scissors: BTreeMap<[u32; 4], Scissor>,
    constants: BTreeMap<[u32; 4], BlendConstant>,
    range: Template,
    compute: Option<SideQueue<'d, G>>,
    copy: Option<SideQueue<'d, G>>,
    host: PhantomData<H>,
}

struct SideQueue<'d, G: Gpu<Cmd = u64> + 'd> {
    queue: G::Queue<'d>,
    ring: Vec<Slot<'d, G>>,
    tables: Vec<u64>,
    context: Vec<u32>,
    tail: Vec<u32>,
    turn: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Render,
    Compute,
    Copy,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Done {
    pub lane: Lane,
    pub slot: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    cc: u64,
    sf: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearDepth {
    None,
    Depth,
    DepthStencil,
}

#[derive(Clone, Copy, Debug)]
pub struct BlendConstant(u64);

#[derive(Clone, Copy, Debug)]
pub struct Scissor(u64);

impl<'d, G: Gpu<Cmd = u64> + 'd, H: Host> Context<'d, G, H> {
    fn bind(gpu: &'d G, vm: &G::Vm<'d>, bo: &G::Bo<'d>, addr: u64, size: u64, mem: MemType) -> Result<(), Error> {
        let done = gpu.fence(false)?;
        vm.bind(bo, 0, addr, size, mem, &[&done])?;
        if !done.wait(Duration::from_secs(5))? {
            return Err(Error::Timeout);
        }
        Ok(())
    }

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
        context.emit(&mut me.context, |n| me.names.get(n).copied().unwrap_or(0));
        H::fence();
        Ok(me)
    }

    fn template(ops: &[OpData<'_>]) -> Result<Template, Error> {
        let mut t = Template::default();
        for op in ops {
            t.extend(&XE3.template(&op.name, &op.all_attrs())?);
        }
        Ok(t)
    }

    fn resolve(&self, local: &BTreeMap<String, u64>, name: &str) -> Option<u64> {
        let get = |n: &str| local.get(n).or_else(|| self.names.get(n)).copied();
        match name.split_once("::") {
            Some((_, x)) => get(x).map(|a| a - self.pool.addr),
            None => get(name),
        }
    }

    fn heap(&mut self, heap: Heap) -> &mut Region<'d, G> {
        match heap {
            Heap::Code => &mut self.code,
            Heap::Surfaces => &mut self.pool,
        }
    }

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

    pub fn pipeline(&mut self, codegen: &Codegen<'_>, src: &str, name: &str) -> Result<Pipeline, Error> {
        self.pipeline_body(codegen, codegen.0.parse(src)?, name)
    }

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

    pub fn kernel(&mut self, codegen: &Codegen<'_>, src: &str, name: &str) -> Result<Kernel, Error> {
        self.kernel_body(codegen, codegen.0.parse(src)?, name)
    }

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

    pub fn clear_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, None, None, Some(rgba))
    }

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

    pub fn msaa_clear_pipeline_samples(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4], samples: u32) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, Some(samples), None, Some(rgba))
    }

    pub fn msaa_clear_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, Some(4), None, Some(rgba))
    }

    fn clear_pipeline_of(&mut self, codegen: &Codegen<'_>, samples: Option<u32>, depth: Option<DepthState>, rgba: Option<[f32; 4]>) -> Result<Pipeline, Error> {
        let mut body = codegen.0.module();
        let root = body.root;
        ClearPipeline { name: "clear", format: TextureFormat::Rgba8Unorm, samples, depth, color: rgba.map(|c| c.map(f64::from)) }.emit(&mut body, root);
        self.pipeline_body(codegen, body, "clear")
    }

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

    pub fn floats(&mut self, values: &[f32]) -> Result<Buffer<'d, G>, Error> {
        self.buffer(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
    }

    pub fn clear_vertices(&mut self) -> Result<Buffer<'d, G>, Error> {
        self.floats(&ClearPipeline::CORNERS)
    }

    pub fn depth_buffer(&mut self, width: u32, height: u32) -> Result<DepthBuffer<'d, G>, Error> {
        let (pitch, rows) = DepthTarget::layout(width, height);
        let r = Self::region(self.gpu, &self.vm, &mut self.next, u64::from(pitch) * u64::from(rows), Caching::WriteBack, MemType::Coherent)?;
        let surface = self.depth_surface(r.addr, width, height)?;
        Ok(DepthBuffer { _bo: r.bo, addr: r.addr, width, height, pitch, stencil: None, surface })
    }

    pub fn depth_stencil_buffer(&mut self, width: u32, height: u32) -> Result<DepthBuffer<'d, G>, Error> {
        let (pitch, rows) = DepthTarget::layout(width, height);
        let (spitch, srows) = DepthTarget::stencil_layout(width, height);
        let depth = (u64::from(pitch) * u64::from(rows)).next_multiple_of(PAGE);
        let r = Self::region(self.gpu, &self.vm, &mut self.next, depth + u64::from(spitch) * u64::from(srows), Caching::WriteBack, MemType::Coherent)?;
        let surface = self.depth_surface(r.addr, width, height)?;
        Ok(DepthBuffer { _bo: r.bo, addr: r.addr, width, height, pitch, stencil: Some((r.addr + depth, spitch)), surface })
    }

    fn depth_surface(&mut self, addr: u64, width: u32, height: u32) -> Result<u64, Error> {
        let mut cmds = Cmds::default();
        let name = cmds.depth_surface("depth_texture", A::sym("depth"), width, height);
        self.names.insert("depth".into(), addr);
        let placed = self.place(&cmds);
        self.names.remove("depth");
        H::fence();
        Ok(placed?[&name])
    }

    pub fn clear_depth_stencil_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        let stencil = StencilState { compare: CompareFunction::Always, fail: StencilOperation::Replace, depth_fail: StencilOperation::Replace, pass: StencilOperation::Replace, read_mask: 0xff, write_mask: 0xff, back: None };
        self.clear_pipeline_of(codegen, None, Some(DepthState { compare: CompareFunction::Always, write: true, stencil: Some(stencil), bias: 0, bias_slope_scale: 0.0, bias_clamp: 0.0, bounds: None }), Some(rgba))
    }

    pub fn clear_depth_pipeline(&mut self, codegen: &Codegen<'_>, rgba: [f32; 4]) -> Result<Pipeline, Error> {
        self.clear_pipeline_of(codegen, None, Some(DepthState { compare: CompareFunction::Always, write: true, stencil: None, bias: 0, bias_slope_scale: 0.0, bias_clamp: 0.0, bounds: None }), Some(rgba))
    }

    pub fn clear_depth_vertices(&mut self, z: f32) -> Result<Buffer<'d, G>, Error> {
        self.floats(&[-1.0, -1.0, z, 1.0, 3.0, -1.0, z, 1.0, -1.0, 3.0, z, 1.0])
    }

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

    pub fn msaa_target(&mut self, width: u32, height: u32, format: TextureFormat) -> Result<Target<'d, G>, Error> {
        self.msaa_target_samples(width, height, format, 4)
    }

    pub fn msaa_target_samples(&mut self, width: u32, height: u32, format: TextureFormat, samples: u32) -> Result<Target<'d, G>, Error> {
        if format.bytes() != 4 || ![2, 4, 8].contains(&samples) {
            return Err(Error::msg(format!("msaa_target: {samples} samples of {}-byte texels (2, 4, 8 of 32-bit ones)", format.bytes())));
        }
        let texture = Texture { samples, ..Texture::plain(width, height, format) };
        let r = Self::region(self.gpu, &self.vm, &mut self.next, texture.size(), Caching::WriteCombined, MemType::Streaming)?;
        let state = self.pass_state(r.addr, &texture)?;
        Ok(Target { bo: r.bo, addr: r.addr, texture, state })
    }

    pub fn resolve_pipeline(&mut self, codegen: &Codegen<'_>) -> Result<Pipeline, Error> {
        self.pipeline(codegen, include_str!("resolve.mlir"), "resolve")
    }

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

    pub fn resolve_vertices(&mut self, width: u32, height: u32) -> Result<Buffer<'d, G>, Error> {
        let (w, h) = (width as f32, height as f32);
        self.floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1.0, 2.0 * w, 0.0, -1.0, 3.0, 0.0, 1.0, 0.0, 2.0 * h])
    }

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

    pub fn import_canvas(&mut self, fd: BorrowedFd<'_>, offset: u32, stride: u32, size: [u32; 2]) -> Result<ImportedCanvas<'d, G>, Error> {
        if !stride.is_multiple_of(4) || stride / 4 < size[0] {
            return Err(Error::msg(format!("import_canvas: rows of {stride} bytes for {} pixels", size[0])));
        }
        let bo = self.gpu.import(fd)?;
        let need = u64::from(offset) + u64::from(stride) * u64::from(size[1]);
        if bo.size() < need {
            return Err(Error::msg(format!("import_canvas: a {} byte dma-buf, {need} needed", bo.size())));
        }
        let addr = self.next.next_multiple_of(PAGE);
        Self::bind(self.gpu, &self.vm, &bo, addr, bo.size().next_multiple_of(self.gpu.info().min_alignment), MemType::Coherent)?;
        self.next = addr + bo.size().next_multiple_of(PAGE);
        Ok(ImportedCanvas { _bo: bo, canvas: Canvas { addr: addr + u64::from(offset), size, stride: stride / 4 } })
    }

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

    pub fn texture_array(&mut self, width: u32, height: u32, format: TextureFormat, layers: &[&[u8]]) -> Result<Image2d<'d, G>, Error> {
        self.layered(Texture { layers: layers.len() as u32, ..Texture::plain(width, height, format) }, layers)
    }

    pub fn texture_cube(&mut self, size: u32, format: TextureFormat, faces: [&[u8]; 6]) -> Result<Image2d<'d, G>, Error> {
        self.layered(Texture { layers: 6, kind: TextureKind::Cube, ..Texture::plain(size, size, format) }, &faces)
    }

    pub fn texture_3d(&mut self, width: u32, height: u32, format: TextureFormat, slices: &[&[u8]]) -> Result<Image2d<'d, G>, Error> {
        self.layered(Texture { layers: slices.len() as u32, kind: TextureKind::Volume, ..Texture::plain(width, height, format) }, slices)
    }

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

    fn layered(&mut self, texture: Texture, layers: &[&[u8]]) -> Result<Image2d<'d, G>, Error> {
        let levels: Vec<Vec<&[u8]>> = layers.iter().map(|l| vec![*l]).collect();
        self.layers_of(texture, &levels)
    }

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

    pub fn slot(&self) -> usize {
        self.turn
    }

    pub fn ring(&self) -> usize {
        RING
    }

    pub fn record(&self) -> Recorder<'_, 'd, G, H> {
        self.recorder(Lane::Render, &self.context, self.turn, self.tables[self.turn], TABLES)
    }

    pub fn open_compute(&mut self) -> Result<(), Error> {
        if self.compute.is_none() {
            let mut cmds = Cmds::default();
            cmds.compute_context();
            let side = self.side(Engine::Compute, &cmds, self.tail.clone())?;
            self.compute = Some(side);
        }
        Ok(())
    }

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

    pub fn record_copy(&self) -> Result<Recorder<'_, 'd, G, H>, Error> {
        let c = self.copy.as_ref().ok_or_else(|| Error::msg("record_copy: open_copy first"))?;
        Ok(self.recorder(Lane::Copy, &c.context, c.turn, c.tables[c.turn], COMPUTE_TABLES))
    }

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

    pub fn record_compute(&self) -> Result<Recorder<'_, 'd, G, H>, Error> {
        let c = self.compute.as_ref().ok_or_else(|| Error::msg("record_compute: open_compute first"))?;
        Ok(self.recorder(Lane::Compute, &c.context, c.turn, c.tables[c.turn], COMPUTE_TABLES))
    }

    fn recorder<'s>(&'s self, lane: Lane, context: &[u32], slot: usize, tables: u64, size: u64) -> Recorder<'s, 'd, G, H> {
        let mut dw = Vec::with_capacity(4096);
        dw.extend_from_slice(context);
        Recorder { cx: self, frame: Recorded { dw, tables: Vec::new(), slot, lane }, state: [0; 5], colors: Vec::new(), samples: 1, depth: None, reference: 0, bound: Vec::new(), push: None, predicated: false, next_table: tables, tables_end: tables + size }
    }

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

    pub fn export_fence(&self, n: usize) -> Result<OwnedFd, Error> {
        Ok(self.ring[n].fence.export()?)
    }
}

fn le_bytes(dw: &[u32]) -> std::borrow::Cow<'_, [u8]> {
    if cfg!(target_endian = "little") {
        std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(dw.as_ptr().cast(), dw.len() * 4) })
    } else {
        std::borrow::Cow::Owned(dw.iter().flat_map(|d| d.to_le_bytes()).collect())
    }
}

pub struct Recorded {
    pub dw: Vec<u32>,
    tables: Vec<(u64, Vec<u32>)>,
    slot: usize,
    lane: Lane,
}

#[derive(Clone, Copy)]
struct Draw {
    count: u32,
    instances: u32,
    first: u32,
    index: Option<((u64, u64), IndexFormat, i32)>,
    indirect: Option<u64>,
}

pub struct Recorder<'c, 'd, G: Gpu<Cmd = u64> + 'd, H: Host> {
    cx: &'c Context<'d, G, H>,
    frame: Recorded,
    state: [u64; 5],
    colors: Vec<u64>,
    samples: u32,
    depth: Option<Attached>,
    reference: u8,
    bound: Vec<u64>,
    push: Option<(u64, u32)>,
    predicated: bool,
    tables_end: u64,
    next_table: u64,
}

impl<'d, G: Gpu<Cmd = u64> + 'd, H: Host> Recorder<'_, 'd, G, H> {
    pub fn target(&mut self, target: &Target<'d, G>) {
        self.targets(&[target]);
    }

    pub fn targets(&mut self, targets: &[&Target<'d, G>]) {
        let Some(first) = targets.first() else { return };
        self.state = first.state;
        self.colors = targets.iter().map(|t| t.state[3]).collect();
        self.samples = first.texture.samples;
        self.depth = None;
    }

    pub fn resolve(&mut self, pipeline: &Pipeline, vertices: &Buffer<'d, G>, msaa: &Target<'d, G>) -> Result<(), Error> {
        if msaa.texture.samples < 2 || pipeline.layout.bindings != 1 {
            return Err(Error::msg("resolve: a multisampled target, a pipeline that loads one texture (resolve_pipeline_samples)"));
        }
        self.emit(pipeline, &[(vertices.addr, vertices.size)], &[msaa.state[3]], Draw { count: 3, instances: 1, first: 0, index: None, indirect: None })
    }

    pub fn depth_barrier(&mut self) -> Result<(), Error> {
        let mut cmds = Cmds::default();
        cmds.depth_barrier();
        Context::<G, H>::template(&cmds.batch)?.emit(&mut self.frame.dw, |_| 0);
        Ok(())
    }

    pub fn viewport(&mut self, viewport: &Viewport) {
        self.state[0] = viewport.cc;
        self.state[1] = viewport.sf;
    }

    pub fn clear(&mut self, pipeline: &Pipeline, corners: &Buffer<'d, G>, color: &BlendConstant) -> Result<(), Error> {
        let kept = self.state[2];
        self.state[2] = color.0;
        let drawn = self.emit(pipeline, &[(corners.addr, corners.size)], &[], Draw { count: 3, instances: 1, first: 0, index: None, indirect: None });
        self.state[2] = kept;
        drawn
    }

    pub fn blend_constant(&mut self, constant: &BlendConstant) {
        self.state[2] = constant.0;
    }

    pub fn scissor(&mut self, scissor: &Scissor) {
        self.state[4] = scissor.0;
    }

    pub fn bind(&mut self, buffers: &[&Buffer<'d, G>]) {
        self.bound = buffers.iter().map(|b| b.surface).collect();
    }

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

    pub fn begin_conditional(&mut self, buffer: &Buffer<'d, G>, offset: u64, inverted: bool) -> Result<(), Error> {
        if !offset.is_multiple_of(4) || offset + 4 > buffer.size {
            return Err(Error::msg(format!("begin_conditional: the value at {offset} of a {}-byte buffer", buffer.size)));
        }
        let at = buffer.addr + offset;
        self.put(|c| c.condition(A::from(at), inverted))?;
        self.predicated = true;
        Ok(())
    }

    pub fn end_conditional(&mut self) {
        self.predicated = false;
    }

    pub fn depth(&mut self, depth: &DepthBuffer<'d, G>) {
        self.depth = Some(Attached { addr: depth.addr, pitch: depth.pitch, width: depth.width, height: depth.height, stencil: depth.stencil });
    }

    pub fn stencil_reference(&mut self, reference: u8) {
        self.reference = reference;
    }

    pub fn draw(&mut self, pipeline: &Pipeline, buffers: &[&Buffer<'d, G>], textures: &[&dyn Sampled], vertices: u32, instances: u32) -> Result<(), Error> {
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        self.draw_at(pipeline, &at, textures, vertices, instances)
    }

    pub fn draw_at(&mut self, pipeline: &Pipeline, buffers: &[(u64, u64)], textures: &[&dyn Sampled], vertices: u32, instances: u32) -> Result<(), Error> {
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        self.emit(pipeline, buffers, &surfaces, Draw { count: vertices, instances, first: 0, index: None, indirect: None })
    }

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

    pub fn draw_indirect(&mut self, pipeline: &Pipeline, buffers: &[&Buffer<'d, G>], textures: &[&dyn Sampled], (args, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        if !offset.is_multiple_of(4) || offset + 16 > args.size {
            return Err(Error::msg(format!("draw_indirect: arguments at {offset} of a {}-byte buffer", args.size)));
        }
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        self.emit(pipeline, &at, &surfaces, Draw { count: 0, instances: 0, first: 0, index: None, indirect: Some(args.addr + offset) })
    }

    pub fn draw_indexed_indirect(&mut self, pipeline: &Pipeline, buffers: &[&Buffer<'d, G>], index: (&Buffer<'d, G>, IndexFormat), textures: &[&dyn Sampled], (args, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        if !offset.is_multiple_of(4) || offset + 20 > args.size {
            return Err(Error::msg(format!("draw_indexed_indirect: arguments at {offset} of a {}-byte buffer", args.size)));
        }
        let at: Vec<(u64, u64)> = buffers.iter().map(|b| (b.addr, b.size)).collect();
        let surfaces: Vec<u64> = textures.iter().map(|t| t.surface()).collect();
        let index = Some(((index.0.addr, index.0.size), index.1, 0));
        self.emit(pipeline, &at, &surfaces, Draw { count: 0, instances: 0, first: 0, index, indirect: Some(args.addr + offset) })
    }

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

    pub fn dispatch(&mut self, kernel: &Kernel, groups: [u32; 2], params: &[u32]) -> Result<(), Error> {
        self.computes()?;
        if params.len() > crate::xe3_codegen::COMPUTE_PARAMS {
            return Err(Error::msg(format!("dispatch: {} parameters, at most {}", params.len(), crate::xe3_codegen::COMPUTE_PARAMS)));
        }
        Kernel::emit(&kernel.direct, &mut self.frame.dw, &Grid { groups, params, table: 0, args: 0 });
        Ok(())
    }

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

    pub fn set_event(&mut self, buffer: &Buffer<'d, G>, offset: u64, value: u32) -> Result<(), Error> {
        let at = Self::event_at(buffer, offset)?;
        let copy = self.frame.lane == Lane::Copy;
        self.put(|c| c.set_event(A::from(at), value, copy))
    }

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

    fn computes(&self) -> Result<(), Error> {
        match self.frame.lane {
            Lane::Copy => Err(Error::msg("dispatch: in a copy engine frame (blits only)")),
            _ => Ok(()),
        }
    }

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

    pub fn texture_barrier(&mut self) -> Result<(), Error> {
        self.put(|c| c.texture_barrier())
    }

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

    fn put(&mut self, f: impl FnOnce(&mut Cmds<'_>)) -> Result<(), Error> {
        let mut cmds = Cmds::default();
        f(&mut cmds);
        Context::<G, H>::template(&cmds.batch)?.emit(&mut self.frame.dw, |_| 0);
        Ok(())
    }

    fn slot(buffer: &Buffer<'d, G>, offset: u64, bytes: u64) -> Result<u64, Error> {
        if !offset.is_multiple_of(8) || offset + bytes > buffer.size {
            return Err(Error::msg(format!("a query result at {offset} of a {}-byte buffer", buffer.size)));
        }
        Ok(buffer.addr + offset)
    }

    pub fn timestamp(&mut self, (buffer, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        let at = Self::slot(buffer, offset, 8)?;
        self.put(|c| c.timestamp(A::from(at)))
    }

    pub fn depth_count(&mut self, (buffer, offset): (&Buffer<'d, G>, u64)) -> Result<(), Error> {
        let at = Self::slot(buffer, offset, 8)?;
        self.put(|c| c.depth_count(A::from(at)))
    }

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
