//! AMD RDNA2 (gfx1036): SPIR-V vertex and fragment shaders compiled to GFX10 machine code (`Rdna2Graphics`), the
//! graphics ABI they keep (`shader`) and a draw's PM4 state (`draw`).

mod draw;
mod gfx;
mod gfx_mir;
mod shader;

pub use draw::*;
pub use gfx::Rdna2Graphics;
pub use shader::*;
