use mlirformat::{Error, Op};
use mlirpass::{Apple9Dialect, Body, Emit, OpId, pass};

use crate::base::{Compiler, FRONTEND, GpuPasses, Stage};

pub use mlirpass::{Apple9Code, Apple9Emit, Apple9Kernel};

pub const APPLE9_BACKEND: &str = "apple9-check, convert-detour-to-gpu, convert-gpu-to-apple9, apple9-regalloc, apple9-control, apple9-wait";

crate::pass_set! {
    Apple9Passes: Apple9Check
}

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
        if threads == 1 {
            body[f].set("known_block_size", mlirformat::Attribute::dense_i32(&[16, 1, 1]));
        }
    }
    Ok(())
}

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
    pub fn pipeline() -> String {
        format!("builtin.module({FRONTEND}, {APPLE9_BACKEND})")
    }

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
