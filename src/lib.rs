//! detour: a user-mode driver for Intel Xe3 and AMD RDNA2 integrated graphics over mlirpass: the kernel interface, shaders to
//! machine code and OpenGL (gl). Its GPU dialects are mlirformat plugins (`dialect`), its lowerings mlirpass plugins
//! (`pass`), and their machine code and images what it writes (`binary`).

extern crate alloc;

mod binary;
mod dialect;
mod gl;
mod kmd;
mod pass;
mod render;
mod shader;
mod tiling;

pub use binary::*;
pub use dialect::*;
pub use gl::*;
pub use kmd::*;
pub use pass::*;
pub use render::*;
pub use shader::*;
pub use tiling::*;
// gl's names win over mlirpass's of the same name, glob-imported below
pub use gl::{Names, Texture};

// The passes came from mlirpass and name its items as `crate::…`, as its #[pass] / #[patterns] /
// rewrites! expansions do: mlirpass's names, at the root (detour's own items take precedence)
#[allow(unused_imports, reason = "for mlirpass's names and macro expansions")]
use mlirpass::*;
