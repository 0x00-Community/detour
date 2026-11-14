//! detour MLIR ⇒ AGX (Apple M1, AGX2's USC), on mlirpass's AGX target (and Apple M3 / M4 on its Apple9 target:
//! [`apple9`]): detour's compute shaders (after detour-core's
//! frontend) become upstream `gpu.func` kernels (detour-core's bridge), which mlirpass lowers, allocates and encodes.
//!
//! ```text
//!   func.func {detour.stage = "compute"}
//!      │  agx-check                what mlirpass's AGX target lowers, else refused here (this crate)
//!      │  convert-detour-to-gpu     ⇒ gpu.module @shaders { gpu.func kernel } (detour-core)
//!      │  convert-gpu-to-agx       ⇒ agx.kernel of virtual agx.usc (mlirpass)
//!      │  agx-regalloc, agx-wait   GPR halves, waits for the loads (mlirpass)
//!      │  agx-end, agx-layout      Stream Terminate, addresses (mlirpass)
//!      ▼
//!   AGX_IMAGE: the memories' bytes and the CDM stream ⇒ asahi
//! ```
//!
//! A dispatch is [`Dispatch`] (mlirpass's `AgxDispatch`): the pipeline (USC words), the parameter buffer, the CDM
//! stream. Not run on an M1: the instructions decode with Mesa's own model as the IR says; the whole is unchecked
//! against hardware. What detour's own AGX backend made before it moved to mlirpass is the golden test
//! (tests/ints.rs).

// mlirpass-macros' #[pass] expands to `alloc::…`
extern crate alloc;

/// Apple M3 / M4 (mlirpass's Apple9 target).
pub mod apple9;

use mlirformat::{Error, Op};
use mlirpass::{A, AgxDialect, Body, Emit, Heads, Image, ImageData, ImageDialect, ImageMemory, ImageStream, MemType, OpId, Quiet, Rewriter, Symbol, pass};

use crate::base::{Compiler, FRONTEND, GpuPasses, Lane, Load, Stage, Store, WorkgroupId};

// ⚠ mlirpass-macros' #[pass] expands to `crate::agx_codegen::…`: the names it uses, at the root.
#[allow(unused_imports, reason = "for mlirpass' macro expansions")]
pub(crate) use mlirpass::{OptionValue, Options, PassDecl, Run};

pub use mlirpass::{AGX_IMAGE, AgxCode, AgxDispatch as Dispatch, AgxEmit, AgxImage as Agx, AgxKernel};

/// ⭐ The AGX backend's shader passes, after [`FRONTEND`].
pub const BACKEND: &str = "agx-check, convert-detour-to-gpu, convert-gpu-to-agx, agx-regalloc, agx-wait";

/// ⭐ The image's passes, once the dispatch and the memories are in the module.
pub const END: &str = "agx-end, agx-layout";

crate::pass_set! {
    /// This crate's pass ([`BACKEND`]'s first; the bridge is detour-core's, the rest mlirpass's).
    AgxPasses: AgxCheck
}

/// `agx-check`: what mlirpass's AGX target lowers, checked before detour-core's bridge spells it upstream (which it
/// does for every target): compute shaders of one thread a workgroup (the target's are 16 invocations, whatever
/// `known_block_size` says) and no shared memory; of detour's ops only the workgroup id, the lane, and loads / stores of
/// 4 bytes a component at a flat global address. Everything else is refused here, as detour's own AGX backend did,
/// rather than left to the target (which refuses most of it, but would ignore a block size).
#[pass("agx-check", on = "builtin.module")]
pub fn agx_check<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let top = body.top_block(op);
    let shaders: Vec<OpId> = body.ops(top).filter(|f| body[*f].name == "func.func" && Stage::of(body, *f).is_some()).collect();
    for f in shaders {
        if Stage::of(body, f) != Some(Stage::Compute) {
            return Err(Error(format!("AGX: only compute shaders, not {:?}", Stage::of(body, f))));
        }
        for a in ["detour.workgroup", "detour.shared"] {
            if body[f].get(a).map(crate::base::Attrs::as_int).transpose()?.is_some_and(|n| n > i64::from(a == "detour.workgroup")) {
                return Err(Error(format!("AGX: {a} (one thread a workgroup, no shared memory)")));
            }
        }
        let mut ops = Vec::new();
        body.walk(f, &mut |o| ops.push(o));
        for o in ops {
            let flat = |size: Option<u32>, binding: Option<u32>, shared: Option<bool>| match (size.unwrap_or(4), binding, shared.unwrap_or(false)) {
                (4, None, false) => Ok(()),
                _ => Err(Error(format!("AGX: a load / store of size {size:?}, binding {binding:?}, shared {shared:?}"))),
            };
            match &*body[o].name {
                n if n == WorkgroupId::NAME || n == Lane::NAME => {}
                n if n == Load::NAME => {
                    let l = body.head::<Load>(o)?;
                    flat(l.size, l.binding, l.shared)?;
                }
                n if n == Store::NAME => {
                    let s = body.head::<Store>(o)?;
                    flat(s.size, s.binding, s.shared)?;
                }
                n if n.starts_with("detour.") => return Err(Error(format!("AGX: {n} is not lowered"))),
                _ => {}
            }
        }
    }
    Ok(())
}

/// ⭐ The compiler with the AGX backend: mlirpass's `agx` and `image` dialects loaded (its image has no
/// `image.kernel`: a kernel is an `agx.kernel`), the check and detour-core's bridge registered beside mlirpass's own
/// passes.
pub struct Codegen<'i>(pub Compiler<'i>);

impl Default for Codegen<'_> {
    fn default() -> Self {
        let mut c = Compiler::default();
        c.opt.load(AgxDialect);
        c.opt.load(ImageDialect);
        c.passes.with(AgxPasses::add).with(GpuPasses::add);
        Codegen(c)
    }
}

impl Codegen<'_> {
    pub fn lower<'a>(&'a self, src: &'a str, pipeline: &str) -> Result<Body<'a>, Error> {
        let mut body = self.0.parse(src)?;
        self.0.run(&mut body, pipeline)?;
        Ok(body)
    }

    /// The shader pipeline: the frontend, then the shaders.
    pub fn pipeline() -> String {
        format!("builtin.module({FRONTEND}, {BACKEND})")
    }
}

/// ⭐ A shader built into an image with its dispatch ([`Codegen::image`]).
pub struct Built {
    /// The module at the end, printed.
    pub module: String,
    pub image: Image,
    /// The kernel's instructions, and where they are.
    pub kernel: Vec<u8>,
    pub kernel_addr: u64,
}

impl Built {
    /// The image as text: `stream ADDR END`, then each memory `segment NAME ADDR HEX` (code, src, dst; the
    /// format of tests/ints/*.expected, where zero tails are cut to `+N zeros`).
    pub fn dump(&self) -> String {
        let mut s = format!("stream {:#x} {:#x}\n", self.image.stream, self.image.stream_end);
        for name in ["code", "src", "dst"] {
            if let Some(g) = self.image.segment(name) {
                s += &format!("segment {name} {:#x} {}\n", g.addr, g.bytes.iter().map(|b| format!("{b:02x}")).collect::<String>());
            }
        }
        s
    }
}

impl Codegen<'_> {
    /// ⭐ The one compute shader of `src` dispatched over `groups` workgroups, its parameters the addresses of two
    /// 64 KiB buffers @src and @dst: @code (the CDM stream, the pipeline, the parameters, the kernel: all in the USC
    /// window) and the buffers, laid out and emitted.
    pub fn image(&self, src: &str, groups: u32) -> Result<Built, Error> {
        let mut body = self.lower(src, &Codegen::pipeline())?;
        let mut kernels = Vec::new();
        body.walk(body.root, &mut |o| {
            if body[o].name == AgxKernel::NAME {
                kernels.push(o);
            }
        });
        let [k] = kernels[..] else { return Err(Error(format!("{} kernels, one wanted", kernels.len()))) };
        let head = body.head::<AgxKernel>(k)?;
        let gprs = head.gprs.ok_or_else(|| Error("a kernel without gprs".into()))?;
        let d = Dispatch { kernel: Symbol::defined(&head.sym_name)?.to_owned(), gprs, groups, params: vec![A::sym("src"), A::sym("dst")] };

        let top = body.top_block(body.root);
        let (code, block) = body.container(&ImageMemory::named("code", MemType::Streaming))?;
        let (cdm, cdm_block) = body.container(&ImageStream { cdm: true, ..ImageStream::named("cdm") })?;
        let mut quiet = Quiet;
        let mut rw = Rewriter::new(&mut body, &mut quiet);
        rw.set_end(top);
        rw.insert(code);
        rw.set_end(block);
        rw.insert(cdm);
        rw.set_end(cdm_block);
        for o in d.stream() {
            rw.create(o, &[]);
        }
        rw.set_end(block);
        for o in d.pipeline("params").into_iter().chain([d.params()?]) {
            rw.create(o, &[]);
        }
        let shaders = rw.body.parent_op(k);
        rw.move_before(k, block, None);
        // the gpu.module the kernel came out of is empty now
        if let Some(m) = shaders.filter(|m| rw.body[*m].name == "gpu.module" && rw.body.ops(rw.body.top_block(*m)).next().is_none()) {
            rw.erase_op(m);
            let root = rw.body.root;
            rw.modify(root, |b| {
                b[root].remove("gpu.container_module");
            });
        }
        for buffer in ["src", "dst"] {
            let (m, b) = rw.body.container(&ImageMemory::named(buffer, MemType::Coherent))?;
            rw.set_end(top);
            rw.insert(m);
            rw.set_end(b);
            rw.emit(&ImageData::zeros(buffer, 0x1_0000));
        }
        self.0.run(&mut body, &format!("builtin.module({END})"))?;
        let image = AGX_IMAGE.emit(&body, body.root)?;
        let kernel = AgxCode::of(&body, k)?.bytes;
        let at = match body[k].get("offset") {
            Some(a) => mlirpass::Attrs::as_int(a)? as u64,
            None => return Err(Error("the kernel was not placed".into())),
        };
        let code = image.segment("code").ok_or_else(|| Error("no @code".into()))?;
        if code.bytes.get(at as usize..at as usize + kernel.len()) != Some(&kernel[..]) {
            return Err(Error("@code does not hold the kernel where it was placed".into()));
        }
        let kernel_addr = code.addr + at;
        Ok(Built { module: self.0.show(&body).to_string(), image, kernel, kernel_addr })
    }
}
