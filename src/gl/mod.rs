//! OpenGL ES 3.0 on detour: the objects, the bindings and the state; shaders through mlirpass's GLSL
mod buffer;
mod context;
mod error;
mod names;
mod program;
mod state;
mod vertex;

pub use buffer::*;
pub use context::*;
pub use error::*;
pub use names::*;
pub use program::*;
pub use state::*;
pub use vertex::*;
