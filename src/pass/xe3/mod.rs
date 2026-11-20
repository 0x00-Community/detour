//! Intel Xe3: SPIR-V and GPU kernels lowered to EU instructions, registers, structured control flow, SWSB, and a draw's whole image (`Xe3Draw`).

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
