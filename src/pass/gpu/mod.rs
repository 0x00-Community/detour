//! What the GPU lowerings share: kernels and stages, register allocation, a draw described without
//! the GPU, the lane-by-lane lowering helpers, `gpu-inline-calls` and `gpu-local-arrays`.

mod draw;
mod inline_calls;
mod lanes;
mod local_arrays;
mod regalloc;
mod shader;

pub use draw::*;
pub use inline_calls::*;
pub use lanes::*;
pub use local_arrays::*;
pub use regalloc::*;
pub use shader::*;
