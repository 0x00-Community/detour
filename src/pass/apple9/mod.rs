//! Apple9 (M3 / M4): `gpu.func` kernels lowered to its instructions, control flow on the execution-mask stack, registers, waits.

mod control;
mod convert;
mod regalloc;
mod target;
mod to_apple9;
mod wait;

pub use control::*;
pub use convert::*;
pub use regalloc::*;
pub use target::*;
pub use to_apple9::*;
pub use wait::*;
