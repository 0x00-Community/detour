use mlirformat::{Error, Op};
use mlirpass::{Body, OpId, PatternSet, pass};

use crate::base::{Heads, ImageEmit, ImageTarget, Kernel, Layout, Rewrites, Stream, Updates};
use crate::xe3_dialect::{CmdMiBatchBufferEnd, CmdMiNoop, Inst, XE3};

const KERNEL_PAD: u64 = 128;

pub struct Xe3;

impl ImageTarget for Xe3 {
    const BASE: u64 = 0x10_0000;
    const PAGE: u64 = 0x1_0000;

    fn stream(&self, body: &Body<'_>, stream: OpId) -> Result<Vec<u8>, Error> {
        let mut dw = Vec::new();
        for c in body.ops(body.top_block(stream)) {
            dw.extend(XE3.pack(&body[c].name, &body[c].all_attrs())?);
        }
        Ok(Xe3::le(&dw))
    }

    fn kernel(&self, body: &Body<'_>, kernel: OpId) -> Result<Vec<u8>, Error> {
        let mut bytes = Vec::new();
        for i in body.ops(body.top_block(kernel)) {
            bytes.extend(Inst::try_from((&*body[i].name, &body[i].all_attrs()[..]))?.encode());
        }
        let pad = body.head::<Kernel>(kernel)?.pad.unwrap_or(0);
        bytes.resize(bytes.len() + pad as usize, 0);
        Ok(bytes)
    }

    fn item(&self, body: &Body<'_>, op: OpId) -> Result<Vec<u8>, Error> {
        match &*body[op].name {
            name if name.starts_with("xe3.state.") => Ok(Xe3::le(&XE3.pack(name, &body[op].all_attrs())?)),
            name => Err(Error(format!("{name} cannot be placed in memory"))),
        }
    }
}

impl Xe3 {
    fn le(dw: &[u32]) -> Vec<u8> {
        dw.iter().flat_map(|d| d.to_le_bytes()).collect()
    }
}

crate::rewrites! {
    PadKernel((Kernel::NAME), 1) |rw, op| {
        if rw.body.head::<Kernel>(op).map_or(true, |k| k.pad.is_some()) {
            return false;
        }
        rw.update::<Kernel>(op, |k| k.pad = Some(KERNEL_PAD)).is_ok()
    }

    EndBatch((Stream::NAME), 1) |rw, op| {
        let block = rw.body.top_block(op);
        let ops = rw.body.block_ops(block);
        if ops.iter().rev().find(|o| rw.body[**o].name != CmdMiNoop::NAME).is_some_and(|o| rw.body[*o].name == CmdMiBatchBufferEnd::NAME) {
            return false;
        }
        let dw: usize = 1 + ops.iter().map(|c| XE3.template(&rw.body[*c].name, &rw.body[*c].all_attrs()).map_or(0, |t| t.dw.len())).sum::<usize>();
        rw.set_end(block);
        rw.emit(&CmdMiBatchBufferEnd::default());
        if !dw.is_multiple_of(2) {
            rw.emit(&CmdMiNoop::default());
        }
        true
    }
}

#[pass("xe3-pad", on = "builtin.module")]
pub fn xe3_pad<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let mut patterns = PatternSet::default();
    patterns.add(PadKernel);
    Rewrites::apply(body, op, &patterns);
    Ok(())
}

#[pass("xe3-end", on = "builtin.module")]
pub fn xe3_end<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let mut patterns = PatternSet::default();
    patterns.add(EndBatch);
    Rewrites::apply(body, op, &patterns);
    Ok(())
}

#[pass("xe3-layout", on = "builtin.module")]
pub fn xe3_layout<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    Layout(&Xe3).run(body, op)
}

pub const XE3_IMAGE: ImageEmit<'_, Xe3> = ImageEmit(&Xe3);
