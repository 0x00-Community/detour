//! AMD RDNA2 (gfx1036): `gpu.func` kernels lowered to GFX10 instructions, registers, waits, PM4 packets and the image laid out.

mod build;
mod convert;
mod dispatch;
mod draw;
mod emit;
mod lower;
mod pass_layout;
mod pass_pad;
mod pass_pm4;
mod pass_regalloc;
mod pass_waitcnt;
mod pm4;
mod shader;

pub use build::*;
pub use convert::*;
pub use dispatch::*;
pub use draw::*;
pub use emit::*;
pub use lower::*;
pub use pass_layout::*;
pub use pass_pad::*;
pub use pass_pm4::*;
pub use pass_regalloc::*;
pub use pass_waitcnt::*;
pub use pm4::*;
pub use shader::*;

use pass_regalloc::Alloc;

/// AMD RDNA2 (gfx1036) machine code
pub struct Rdna2Target;

impl crate::Target for Rdna2Target {
    fn name(&self) -> &'static str {
        "rdna2"
    }

    fn pipeline(&self) -> &'static str {
        "builtin.module(gpu-inline-calls, canonicalize, cse, convert-gpu-to-rdna2, rdna2-regalloc, rdna2-waitcnt)"
    }
}

impl crate::GpuTarget for Rdna2Target {
    fn kernel(&self) -> &'static str {
        <crate::Rdna2Kernel as mlirformat::Op>::NAME
    }

    fn code(&self, body: &crate::Body<'_>, kernel: crate::OpId) -> Result<crate::KernelCode, mlirformat::Error> {
        let k = Rdna2Code::of(&body.operation(kernel))?;
        Ok(crate::KernelCode { name: k.name, align: k.align, pad: k.pad, bytes: k.bytes, info: alloc::vec![("params", u64::from(k.params)), ("sgprs", u64::from(k.sgprs)), ("vgprs", u64::from(k.vgprs))] })
    }

    fn image_pipeline(&self) -> &'static str {
        "builtin.module(rdna2-pm4, rdna2-pad, rdna2-layout)"
    }
}
