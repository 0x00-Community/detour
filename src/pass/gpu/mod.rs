//! What the GPU lowerings share: register allocation, a draw described without
//! the GPU, the lane-by-lane lowering helpers (SPIR-V's matrix arithmetic among them) and `gpu-local-arrays`.

mod draw;
mod lanes;
mod local_arrays;
mod regalloc;

pub use draw::*;
pub use lanes::*;
pub use local_arrays::*;
pub use regalloc::*;
