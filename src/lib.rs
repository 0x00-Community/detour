//! detour: a user-mode driver for Intel Xe3 and AMD RDNA2 integrated graphics over mlirpass: the kernel interface, shaders to
//! machine code, and draws (`Device::draw`). Its GPU dialects are mlirformat plugins (`dialect`), its lowerings mlirpass plugins
//! (`pass`), and the machine code it writes (`binary`). The kernel
//! interface, the devices and the Vulkan layer over them (`vk`) are Linux's; the compiler builds anywhere.

extern crate alloc;

mod binary;
mod dialect;
mod kmd;
mod pass;
#[cfg(target_os = "linux")]
mod render;
mod shader;
mod tiling;
#[cfg(target_os = "linux")]
mod vk;

pub use binary::*;
pub use dialect::*;
pub use kmd::*;
pub use pass::*;
#[cfg(target_os = "linux")]
pub use render::*;
pub use shader::*;
pub use tiling::*;
#[cfg(target_os = "linux")]
pub use vk::*;

// The passes came from mlirpass and name its items as `crate::…`, as its #[pass] / #[patterns] /
// rewrites! expansions do: mlirpass's names, at the root (detour's own items take precedence)
#[allow(unused_imports, reason = "for mlirpass's names and macro expansions")]
use mlirpass::*;
