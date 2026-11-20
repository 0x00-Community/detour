//! Intel Xe3: SPIR-V shaders lowered to EU instructions, registers, structured control flow, SWSB, and a draw's whole image (`Xe3Draw`).

mod control;
mod convert;
mod draw;
mod pad;
mod regalloc;
mod swsb;
mod to_xe3;

pub use control::*;
pub use convert::*;
pub use draw::*;
pub use pad::*;
pub use regalloc::*;
pub use swsb::*;
pub use to_xe3::*;

use alloc::vec::Vec;

use mlirformat::Op;

use crate::{Body, OpId, Xe3Kernel};

impl Xe3Kernel<'_> {
    /// Every `xe3.kernel` under `op`
    pub fn all(body: &Body<'_>, op: OpId) -> Vec<OpId> {
        let mut out = Vec::new();
        body.walk(op, &mut |o| {
            if body[o].name == Self::NAME {
                out.push(o);
            }
        });
        out
    }
}
