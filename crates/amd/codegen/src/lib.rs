//! detour MLIR ⇒ RDNA2 (GFX10.3), on mlirpass's RDNA2 target: detour's compute shaders (after detour-core's frontend)
//! checked for what RDNA2 takes, spelled as upstream `gpu.func` kernels (detour-core's bridge), then lowered,
//! allocated and encoded by mlirpass.
//!
//! ```text
//!   func.func {detour.stage = "compute"}
//!      │  rdna2-precheck           what RDNA2 takes (precheck.rs): nothing else is dropped or lowered as something else
//!      │  convert-detour-to-gpu     ⇒ gpu.module @shaders { gpu.func kernel } (detour-core)
//!      │  convert-gpu-to-rdna2     ⇒ rdna2.kernel of virtual rdna2.isa (mlirpass)
//!      │  rdna2-regalloc, rdna2-waitcnt   SGPRs, VGPRs; waits for the loads (mlirpass)
//!      │  rdna2-pm4, rdna2-pad, rdna2-layout   the stream's packets, the kernel's padding, addresses (mlirpass)
//!      ▼
//!   RDNA2_IMAGE: the memories' bytes and the IB's address ⇒ amdgpu
//! ```
//!
//! A dispatch is [`Dispatch`] (mlirpass's `Rdna2Dispatch`): its ops go in an `image.stream`. Not run on an AMD GPU
//! yet: the instructions encode as llvm-mc does, the packets as Mesa's macros; the whole is unchecked against
//! hardware. What detour's own RDNA2 backend made before it moved to mlirpass is the golden test (tests/kernels.rs).

// mlirpass-macros' #[pass] expands to `alloc::…`
extern crate alloc;

mod precheck;

pub use precheck::*;

use mlirformat::{Error, Op};
use mlirpass::{A, Body, Emit, Heads, Image, ImageData, ImageDialect, ImageMemory, ImageStream, MemType, Quiet, Rdna2Dialect, Rewriter, Symbol};

use crate::base::{Compiler, FRONTEND, GpuPasses};

// ⚠ mlirpass-macros' #[pass] expands to `crate::rdna2_codegen::…`: the names it uses, at the root.
#[allow(unused_imports, reason = "for mlirpass' macro expansions")]
pub(crate) use mlirpass::{OptionValue, Options, PassDecl, Run};

pub use mlirpass::{RDNA2_IMAGE, Rdna2Code, Rdna2Dispatch as Dispatch, Rdna2Emit, Rdna2Image as Rdna2, Rdna2Kernel};

/// ⭐ The RDNA2 backend's shader passes, after [`FRONTEND`] (no canonicalize between: the bridge's ops as they are).
pub const BACKEND: &str = "rdna2-precheck, convert-detour-to-gpu, convert-gpu-to-rdna2, rdna2-regalloc, rdna2-waitcnt";

/// ⭐ The image's passes, once the dispatch and the memories are in the module.
pub const END: &str = "rdna2-pm4, rdna2-pad, rdna2-layout";

crate::pass_set! {
    /// This crate's pass ([`BACKEND`]'s first; the bridge is detour-core's, the rest mlirpass's).
    Rdna2Passes: Rdna2Precheck
}

/// ⭐ The compiler with the RDNA2 backend: mlirpass's `rdna2` and `image` dialects loaded (its image has no
/// `image.kernel`: a kernel is an `rdna2.kernel`), the precheck and the bridge registered beside mlirpass's passes.
pub struct Codegen<'i>(pub Compiler<'i>);

impl Default for Codegen<'_> {
    fn default() -> Self {
        let mut c = Compiler::default();
        c.opt.load(Rdna2Dialect);
        c.opt.load(ImageDialect);
        c.passes.with(Rdna2Passes::add).with(GpuPasses::add);
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

    /// ⭐ The first compute shader of `src` dispatched over `groups` workgroups (x), its parameters the addresses of
    /// two 64 KiB buffers @src and @dst: @code (the IB, then the kernel) and the buffers, laid out and emitted.
    pub fn image(&self, src: &str, groups: u32) -> Result<Built, Error> {
        let mut body = self.lower(src, &Codegen::pipeline())?;
        let mut kernels = Vec::new();
        body.walk(body.root, &mut |o| {
            if body[o].name == Rdna2Kernel::NAME {
                kernels.push(o);
            }
        });
        let &k = kernels.first().ok_or_else(|| Error("no kernel".into()))?;
        let head = body.head::<Rdna2Kernel>(k)?;
        let vgprs = head.vgprs.ok_or_else(|| Error("a kernel without vgprs".into()))?;
        let d = Dispatch { kernel: Symbol::defined(&head.sym_name)?.to_owned(), vgprs, groups: [groups, 1, 1], params: vec![A::sym("src"), A::sym("dst")], shader_engines: 1 };

        let top = body.top_block(body.root);
        let (code, block) = body.container(&ImageMemory::named("code", MemType::Streaming))?;
        let (ib, ib_block) = body.container(&ImageStream::named("ib"))?;
        let mut quiet = Quiet;
        let mut rw = Rewriter::new(&mut body, &mut quiet);
        rw.set_end(top);
        rw.insert(code);
        rw.set_end(block);
        rw.insert(ib);
        rw.set_end(ib_block);
        for o in d.ops()? {
            rw.create(o, &[]);
        }
        let shaders = rw.body.parent_op(k);
        rw.move_before(k, block, None);
        // the gpu.module the kernel came out of, if empty now
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
        let image = RDNA2_IMAGE.emit(&body, body.root)?;
        let kernel = Rdna2Code::of(&body, k)?.bytes;
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

/// ⭐ A shader built into an image with its dispatch ([`Codegen::image`]).
pub struct Built {
    /// The module at the end, printed.
    pub module: String,
    pub image: Image,
    /// The kernel's instructions (without its padding), and where they are.
    pub kernel: Vec<u8>,
    pub kernel_addr: u64,
}

impl Built {
    /// The image as text: `stream ADDR END`, then each memory `segment NAME ADDR HEX` (code, src, dst).
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
