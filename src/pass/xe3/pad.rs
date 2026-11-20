use mlirformat::Error;

use crate::{Body, Heads, OpId, PatternSet, Rewrites, Updates, Xe3Kernel, pass};

/// The GRF a thread's end-of-thread send reads from
pub const XE3_EOT_GRF: u8 = 112;

/// Bytes after each kernel (the EU prefetches past its last instruction)
pub const XE3_KERNEL_PAD: u64 = 128;

/// Each kernel padded with `XE3_KERNEL_PAD` zero bytes
#[pass("xe3-pad", on = "builtin.module")]
pub fn xe3_pad<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let mut patterns = PatternSet::default();
    patterns.add(PadKernel);
    Rewrites::apply(body, op, &patterns);
    Ok(())
}

crate::rewrites! {
    PadKernel((<Xe3Kernel as mlirformat::Op>::NAME), 1) |rw, op| {
        if rw.body.head::<Xe3Kernel>(op).map_or(true, |k| k.pad.is_some()) {
            return false;
        }
        rw.update::<Xe3Kernel>(op, |k| k.pad = Some(XE3_KERNEL_PAD)).is_ok()
    }
}
