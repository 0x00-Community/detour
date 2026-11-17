//! detour: a GPU user-mode driver over mlirpass: the kernel interface, shaders to machine code, OpenGL (gl) and Vulkan (vk).
//! Its GPU dialects are mlirformat plugins (`dialect`), its lowerings mlirpass plugins (`pass`), and their machine code
//! and images what it writes (`binary`).

extern crate alloc;

mod binary;
mod dialect;
mod gl;
mod kmd;
mod pass;
mod render;
mod shader;
mod vk;

pub use binary::*;
pub use dialect::*;
pub use gl::*;
pub use kmd::*;
pub use pass::*;
pub use render::*;
pub use shader::*;
// gl's names win over the GPU dialects' ops of the same name (`agx.texture`: AgxTexture)
pub use gl::{Names, Texture};

// The passes came from mlirpass and name its items as `crate::…`, as its #[pass] / #[patterns] /
// rewrites! expansions do: mlirpass's names, at the root (detour's own items take precedence)
#[allow(unused_imports, reason = "for mlirpass's names and macro expansions")]
use mlirpass::*;
