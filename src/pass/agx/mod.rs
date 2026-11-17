//! Apple AGX (G13): `gpu.func` kernels lowered to USC instructions, registers, waits, CDM streams ended and the image laid out.

mod build;
mod convert;
mod dispatch;
mod end;
mod layout;
mod regalloc;
mod stream;
mod target;
mod to_agx;
mod wait;

pub use build::*;
pub use convert::*;
pub use dispatch::*;
pub use end::*;
pub use layout::*;
pub use regalloc::*;
pub use stream::*;
pub use target::*;
pub use to_agx::*;
pub use wait::*;

use regalloc::Alloc;
