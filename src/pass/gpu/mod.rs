//! What Xe3's lowering builds on: kernels and stages, register allocation, a draw described
//! without the GPU, the lane-by-lane lowering helpers, laying out and writing an image, and
//! `gpu-inline-calls`.

mod draw;
mod image;
mod inline_calls;
mod lanes;
mod module_to_binary;
mod regalloc;
mod shader;

pub use draw::*;
pub use image::*;
pub use inline_calls::*;
pub use lanes::*;
pub use module_to_binary::*;
pub use regalloc::*;
pub use shader::*;
