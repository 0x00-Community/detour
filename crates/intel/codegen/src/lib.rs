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

#[allow(unused_imports, reason = "for mlirpass' macro expansions")]
pub(crate) use mlirpass::{ConversionRewriter, Lowering, OpId, OptLowering, OptionValue, Options, PassDecl, Run, ValueId};

#[macro_export]
macro_rules! xe3 {
    ($ty:ident $(, $field:ident = $value:expr)* $(,)?) => {
        $crate::xe3_codegen::xe3_op(&$crate::xe3_value!($ty $(, $field = $value)*))
    };
}

#[macro_export]
macro_rules! xe3_value {
    ($ty:ident $(, $field:ident = $value:expr)* $(,)?) => {{
        #[allow(clippy::needless_update)]
        let value = crate::xe3_dialect::$ty { $($field: Some(::core::convert::Into::into($value)),)* ..::core::default::Default::default() };
        value
    }};
}

pub fn xe3_op<'a, O>(typed: &O) -> mlirpass::OpData<'a>
where
    for<'o> mlirformat::Operation<'static>: TryFrom<&'o O, Error = mlirformat::Error>,
{
    mlirpass::OpData::of(typed).expect("a genxml op serializes").0
}

pub const MOCS: u32 = 1 << 1;

pub const BUFFER_BTI: u8 = 32;

pub const PUSH_BTI: u8 = BUFFER_BTI - 1;

pub const BACKEND: &str = "convert-detour-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad, xe3-end, xe3-layout";

pub const KERNELS: &str = "convert-detour-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad";

crate::pass_set! {
    Xe3Passes: ConvertDetourToXe3, Xe3Regalloc, Xe3Control, Xe3Swsb, Xe3Branches, Xe3Pad, Xe3End, Xe3Layout
}

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
    for o in ["scf.for", "scf.if", "scf.while", "scf.condition", "scf.yield"] {
        target.dynamic_op(o, |b, op| Some(ShaderToXe3::structured_legal(b, op)));
    }
    Converter::new(&target, &patterns, ConversionConfig::default()).apply(body, &shaders, Mode::Full)?;
    ShaderToXe3::place_arguments(body, op)?;
    Render::lower(body, op)
}

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
    pub fn pipeline() -> String {
        format!("builtin.module({FRONTEND}, {BACKEND})")
    }

    pub fn lower<'a>(&'a self, src: &'a str, pipeline: &str) -> Result<Body<'a>, Error> {
        let mut body = self.0.parse(src)?;
        self.0.run(&mut body, pipeline)?;
        Ok(body)
    }

    pub fn compile(&self, src: &str, pipeline: &str) -> Result<Image, Error> {
        let body = self.lower(src, pipeline)?;
        XE3_IMAGE.emit(&body, body.root)
    }
}
