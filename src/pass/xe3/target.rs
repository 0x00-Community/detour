use alloc::vec::Vec;

use mlirformat::{Error, Op};

use crate::{Body, Emit, Heads, ImageEmit, ImageStream, Kernels, OpId, Pack, PatternSet, Rewrites, Updates};
use crate::{CmdMiBatchBufferEnd, CmdMiNoop, XE3, Xe3Kernel};

pub use crate::{Xe3Code, Xe3Image};

/// The GRF a thread's end-of-thread send reads from
pub const XE3_EOT_GRF: u8 = 112;

/// Bytes after each kernel (the EU prefetches past its last instruction)
pub const XE3_KERNEL_PAD: u64 = 128;

pub struct Xe3Emit;

impl Emit for Xe3Emit {
    type Output = Vec<Xe3Code>;

    fn emit(&self, body: &Body<'_>, op: OpId) -> Result<Vec<Xe3Code>, Error> {
        Kernels::all(body, op, Xe3Kernel::NAME).into_iter().map(|k| Xe3Code::of(&body.operation(k))).collect()
    }
}

/// What an Xe3 image needs before it is laid out: kernels padded, batches ended
pub struct Xe3Batch;

impl Xe3Batch {
    pub fn pad(body: &mut Body<'_>, op: OpId) {
        let mut patterns = PatternSet::default();
        patterns.add(PadKernel);
        Rewrites::apply(body, op, &patterns);
    }

    pub fn end(body: &mut Body<'_>, op: OpId) {
        let mut patterns = PatternSet::default();
        patterns.add(EndBatch);
        Rewrites::apply(body, op, &patterns);
    }
}

crate::rewrites! {
    PadKernel((Xe3Kernel::NAME), 1) |rw, op| {
        if rw.body.head::<Xe3Kernel>(op).map_or(true, |k| k.pad.is_some()) {
            return false;
        }
        rw.update::<Xe3Kernel>(op, |k| k.pad = Some(XE3_KERNEL_PAD)).is_ok()
    }

    EndBatch((ImageStream::NAME), 1) |rw, op| {
        let block = rw.body.top_block(op);
        let ops = rw.body.block_ops(block);
        if ops.iter().rev().find(|o| rw.body[**o].name != CmdMiNoop::NAME).is_some_and(|o| rw.body[*o].name == CmdMiBatchBufferEnd::NAME) {
            return false;
        }
        let dw: usize = 1 + ops.iter().map(|c| Pack(XE3).template(&rw.body[*c].name, &rw.body[*c].all_attrs()).map_or(0, |t| t.dw.len())).sum::<usize>();
        rw.set_end(block);
        rw.emit(&CmdMiBatchBufferEnd::default());
        if !dw.is_multiple_of(2) {
            rw.emit(&CmdMiNoop::default());
        }
        true
    }
}

pub const XE3_IMAGE: ImageEmit<'_, Xe3Image> = ImageEmit(&Xe3Image);
