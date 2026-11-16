//! detour: a GPU user-mode driver over mlirpass: the kernel interface, shaders to machine code, OpenGL (gl) and Vulkan (vk)

extern crate alloc;

mod gl;
mod kmd;
mod passes;
mod render;
mod shader;
mod vk;
mod xe;

pub use gl::*;
pub use kmd::*;
pub use passes::*;
pub use render::*;
pub use shader::*;
pub use xe::*;

// mlirpass's #[pass] / #[patterns] expand to `crate::…`: the names they use, at the root
#[allow(unused_imports, reason = "for mlirpass's macro expansions")]
pub(crate) use mlirpass::{Body, ConversionPatterns, ConversionRewriter, Exec, Lowering, OpId, OptLowering, OptionValue, Options, PassDecl, Pattern, PatternSet, Plain, Rewriter, Run, ValueId};
