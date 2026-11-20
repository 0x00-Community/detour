//! Vulkan on detour: a device's objects by handle, command buffers recorded and run on the GPU at vkQueueSubmit;
//! pipelines are the applications' SPIR-V compiled by the toolchain. The calls take Vulkan's create infos as Rust
//! structures (what a loader-facing entry point will hand them on)
mod command;
mod device;
mod handle;
mod types;

pub use command::*;
pub use device::*;
pub use handle::*;
pub use types::*;
