//! Intel Xe3: SPIR-V and GPU kernels lowered to EU instructions, registers, structured control flow, SWSB, batches ended, the image laid out, and a draw's whole image (`Xe3Draw`).

mod control;
mod convert;
mod draw;
mod end;
mod layout;
mod pad;
mod regalloc;
mod swsb;
mod target;
mod to_xe3;

pub use control::*;
pub use convert::*;
pub use draw::*;
pub use end::*;
pub use layout::*;
pub use pad::*;
pub use regalloc::*;
pub use swsb::*;
pub use target::*;
pub use to_xe3::*;
