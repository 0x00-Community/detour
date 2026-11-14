//! detour MLIR ⇒ Apple9 (Apple M3 / M4), on mlirpass's Apple9 target: detour's compute shaders (after detour-core's
//! frontend) become upstream `gpu.func` kernels (detour-core's bridge), which mlirpass lowers, allocates and encodes.
//!
//! ```text
//!   func.func {detour.stage = "compute", detour.workgroup = 1 | 2}
//!      │  apple9-check             compute shaders of 16 or 32 invocations a workgroup (this crate)
//!      │  convert-detour-to-gpu     ⇒ gpu.module @shaders { gpu.func kernel {known_block_size = 16 | 32} } (detour-core)
//!      │  convert-gpu-to-apple9    ⇒ apple9.kernel (workgroups of 16: two a hardware workgroup of 32) (mlirpass)
//!      │  apple9-regalloc, apple9-control, apple9-wait
//!      ▼
//!   Apple9Code: the machine code, its parameter words, GPRs, threadgroup memory, workgroup
//! ```
//!
//! How it runs is mlirpass's Apple9 ABI (`Apple9Code`): binding 0 the memory (byte addresses: detour's flat ones),
//! binding 1 the parameter words (then the grid's size in workgroups). There is no kernel driver for an M4 here:
//! mlirpass's scripts/apple9_metal.swift runs the code inside a Metal pipeline (tests/apple9.rs, on an M4).

use mlirformat::{Error, Op};
use mlirpass::{Apple9Dialect, Body, Emit, OpId, pass};

use crate::base::{Compiler, FRONTEND, GpuPasses, Stage};

pub use mlirpass::{Apple9Code, Apple9Emit, Apple9Kernel};

/// ⭐ The Apple9 backend's shader passes, after [`FRONTEND`].
pub const APPLE9_BACKEND: &str = "apple9-check, convert-detour-to-gpu, convert-gpu-to-apple9, apple9-regalloc, apple9-control, apple9-wait";

crate::pass_set! {
    /// This module's pass ([`APPLE9_BACKEND`]'s first).
    Apple9Passes: Apple9Check
}

/// `apple9-check`: compute shaders only, of one or two detour threads a workgroup (16 or 32 invocations: the target's
/// workgroups), one thread's block size spelled (16 x 1 x 1). What else the target cannot lower it refuses itself.
#[pass("apple9-check", on = "builtin.module")]
pub fn apple9_check<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let top = body.top_block(op);
    let shaders: Vec<OpId> = body.ops(top).filter(|f| body[*f].name == "func.func" && Stage::of(body, *f).is_some()).collect();
    for f in shaders {
        if Stage::of(body, f) != Some(Stage::Compute) {
            return Err(Error(format!("Apple9: only compute shaders, not {:?}", Stage::of(body, f))));
        }
        let threads = body[f].get("detour.workgroup").map(crate::base::Attrs::as_int).transpose()?.unwrap_or(1);
        if !(1..=2).contains(&threads) {
            return Err(Error(format!("Apple9: detour.workgroup {threads} (1 or 2: 16 or 32 invocations)")));
        }
        // one thread: the bridge leaves the block size unsaid, which the target would take as its own 32
        if threads == 1 {
            body[f].set("known_block_size", mlirformat::Attribute::dense_i32(&[16, 1, 1]));
        }
    }
    Ok(())
}

/// ⭐ The compiler with the Apple9 backend: mlirpass's `apple9` dialect loaded, the check and detour-core's bridge
/// registered beside mlirpass's own passes.
pub struct Apple9<'i>(pub Compiler<'i>);

impl Default for Apple9<'_> {
    fn default() -> Self {
        let mut c = Compiler::default();
        c.opt.load(Apple9Dialect);
        c.passes.with(Apple9Passes::add).with(GpuPasses::add);
        Apple9(c)
    }
}

impl Apple9<'_> {
    /// The shader pipeline: the frontend, then the shaders.
    pub fn pipeline() -> String {
        format!("builtin.module({FRONTEND}, {APPLE9_BACKEND})")
    }

    /// ⭐ Every compute shader of `src`, as machine code and what running it needs.
    pub fn kernels(&self, src: &str) -> Result<Vec<Apple9Code>, Error> {
        let mut body = self.0.parse(src)?;
        self.0.run(&mut body, &Self::pipeline())?;
        let root = body.root;
        let codes = Apple9Emit.emit(&body, root)?;
        if codes.is_empty() {
            return Err(Error(format!("no {}", Apple9Kernel::NAME)));
        }
        Ok(codes)
    }
}
