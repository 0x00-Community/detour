//! What detour writes outside MLIR: Xe3 machine code (`xe3`), and records packed by their layout
//! tables (`pack`).

mod pack;
mod xe3;

pub use pack::{Hole, Pack, Template};
pub use xe3::{Xe3Code, Xe3Isa};
