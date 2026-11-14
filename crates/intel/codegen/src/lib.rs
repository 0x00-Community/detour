//! detour MLIR ⇒ Xe3: the lowering of an detour module (after the frontend of `detour-compiler`) to what the GPU runs.
//!
//! ```text
//!   detour.* + func.func {detour.stage}
//!      │  convert-detour-to-xe3     shaders ⇒ image.kernel of virtual xe3.eu (shader.rs); resources, render passes ⇒
//!      │                           image.memory, xe3.state, image.stream of xe3.cmd (render.rs)
//!      │  xe3-regalloc             registers (regalloc.rs; loops and branches stay structured)
//!      │  xe3-control              loops and branches ⇒ break / while, if / else / endif (control.rs)
//!      │  xe3-swsb                 software scoreboard (swsb.rs)
//!      │  xe3-branches             each branch's JIP / UIP (control.rs)
//!      │  xe3-pad, xe3-end         zeros after each kernel; each batch ended (layout.rs)
//!      │  xe3-layout               GPU addresses; symbols ⇒ numbers (layout.rs)
//!      ▼
//!   XE3_IMAGE: the memories' bytes and the batch's address ⇒ the driver
//! ```

extern crate alloc;

mod control;
mod layout;
mod regalloc;
mod render;
mod shader;
mod swsb;

pub use control::*;
pub use layout::*;
pub use regalloc::*;
pub use render::*;
pub use shader::*;
pub use swsb::*;

use mlirformat::{Error, Op};
use mlirpass::{Body, ConversionConfig, ConversionPatterns, ConversionTarget, Converter, Emit, Mode, pass};

pub use crate::base::{A, WithAttrs};
use crate::base::{Atomic, Barrier, Compiler, FRONTEND, Image, Lane, Load, LoadSample, Sample, SinkConstants, Stage, CoverageMask, PushConstant, SampleIndex, Store, TextureLoad, TextureStore, Unpack4xu8, WorkgroupId};
use crate::xe3_dialect::Xe3Dialect;

// ⚠ mlirpass-macros' #[pass] / #[patterns] expand to `crate::xe3_codegen::…`: the names they use, at the root.
#[allow(unused_imports, reason = "for mlirpass' macro expansions")]
pub(crate) use mlirpass::{ConversionRewriter, Lowering, OpId, OptLowering, OptionValue, Options, PassDecl, Run, ValueId};

/// `xe3!(CmdPipeControl, command_streamer_stall_enable = true, ...)`: genxml item `CmdPipeControl`'s typed op with
/// these fields set (each value into the field's type: a number, a bool or a place into a `Fill`, an f32 into a
/// `Real`, ...), the rest absent; a misspelt field does not compile.
#[macro_export]
macro_rules! xe3 {
    ($ty:ident $(, $field:ident = $value:expr)* $(,)?) => {
        $crate::xe3_codegen::xe3_op(&$crate::xe3_value!($ty $(, $field = $value)*))
    };
}

/// `xe3_value!(CmdPipeControl, ...)`: as [`xe3!`], the typed value itself (to set more fields, or a nested struct /
/// a group element such as `StateVertexBufferStateFields`).
#[macro_export]
macro_rules! xe3_value {
    ($ty:ident $(, $field:ident = $value:expr)* $(,)?) => {{
        // the rest absent (an item whose every field is given has no rest)
        #[allow(clippy::needless_update)]
        let value = crate::xe3_dialect::$ty { $($field: Some(::core::convert::Into::into($value)),)* ..::core::default::Default::default() };
        value
    }};
}

/// A genxml item's typed op, detached.
pub fn xe3_op<'a, O>(typed: &O) -> mlirpass::OpData<'a>
where
    for<'o> mlirformat::Operation<'static>: TryFrom<&'o O, Error = mlirformat::Error>,
{
    mlirpass::OpData::of(typed).expect("a genxml op serializes").0
}

/// MOCS index for everything detour reads and writes: L3 and L4 write-back (Xe2+, BSpec 71582; Mesa isl `mocs.internal`).
pub const MOCS: u32 = 1 << 1;
/// The binding table entry of buffer binding 0 (`detour.load` / `detour.store {binding}`): after the target (0) and room
/// for 31 textures, so vertex and fragment shaders agree on it without knowing each other's textures.
pub const BUFFER_BTI: u8 = 32;

/// The binding table entry of the push constants (a buffer surface over them, the entry before the buffers').
pub const PUSH_BTI: u8 = BUFFER_BTI - 1;

/// ⭐ The Xe3 backend's passes, after [`FRONTEND`].
pub const BACKEND: &str = "convert-detour-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad, xe3-end, xe3-layout";

/// The passes that make a module's shaders kernels ready to place (the runtime's: [`BACKEND`] up to `xe3-pad`).
pub const KERNELS: &str = "convert-detour-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad";

crate::pass_set! {
    /// The passes of [`BACKEND`].
    Xe3Passes: ConvertDetourToXe3, Xe3Regalloc, Xe3Control, Xe3Swsb, Xe3Branches, Xe3Pad, Xe3End, Xe3Layout
}

/// `convert-detour-to-xe3`.
#[pass("convert-detour-to-xe3", on = "builtin.module")]
pub fn convert_detour_to_xe3<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let top = body.top_block(op);
    let shaders: Vec<OpId> = body.ops(top).filter(|f| body[*f].name == "func.func" && Stage::of(body, *f).is_some()).collect();
    body.sink_constants(&shaders);
    ShaderToXe3::count_bindings(body, &shaders)?;
    let tc = ShaderTypes::converter();
    let mut patterns = ConversionPatterns::default();
    let i = patterns.converter(&tc);
    ShaderToXe3::populate(&mut patterns, i);
    let mut target = ConversionTarget::default();
    target.legal_dialect("xe3").legal_dialect("image").illegal_dialect("arith").illegal_dialect("vector").illegal_op("func.func").illegal_op("func.return").illegal_dialect("math");
    for o in [Sample::NAME, LoadSample::NAME, Load::NAME, Store::NAME, Atomic::NAME, Barrier::NAME, TextureStore::NAME, TextureLoad::NAME, PushConstant::NAME, SampleIndex::NAME, CoverageMask::NAME, WorkgroupId::NAME, Lane::NAME, Unpack4xu8::NAME] {
        target.illegal_op(o);
    }
    // structured control flow stays, of registers
    for o in ["scf.for", "scf.if", "scf.while", "scf.condition", "scf.yield"] {
        target.dynamic_op(o, |b, op| Some(ShaderToXe3::structured_legal(b, op)));
    }
    Converter::new(&target, &patterns, ConversionConfig::default()).apply(body, &shaders, Mode::Full)?;
    ShaderToXe3::place_arguments(body, op)?;
    Render::lower(body, op)
}

/// ⭐ The compiler with the Xe3 backend: the `xe3` dialect loaded, its passes registered.
pub struct Codegen<'i>(pub Compiler<'i>);

impl Default for Codegen<'_> {
    fn default() -> Self {
        let mut c = Compiler::default();
        c.opt.load(Xe3Dialect);
        c.passes.with(Xe3Passes::add);
        Codegen(c)
    }
}

impl Codegen<'_> {
    /// The whole pipeline: the frontend, then the backend.
    pub fn pipeline() -> String {
        format!("builtin.module({FRONTEND}, {BACKEND})")
    }

    /// MLIR text through `pipeline` (e.g. to look at what a stage made).
    pub fn lower<'a>(&'a self, src: &'a str, pipeline: &str) -> Result<Body<'a>, Error> {
        let mut body = self.0.parse(src)?;
        self.0.run(&mut body, pipeline)?;
        Ok(body)
    }

    /// ⭐ MLIR text (detour, or xe3 already) ⇒ the image.
    pub fn compile(&self, src: &str, pipeline: &str) -> Result<Image, Error> {
        let body = self.lower(src, pipeline)?;
        XE3_IMAGE.emit(&body, body.root)
    }
}
