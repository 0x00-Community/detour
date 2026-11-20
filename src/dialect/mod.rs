//! The GPU dialects detour adds to mlirformat: Xe3's instruction set and command stream (`xe3`, loaded
//! with `opt.load(Xe3Dialect)`), RDNA2's instruction table (`rdna2`), what they share (`isa`) and the
//! layout tables records are packed by (`layout`).

mod isa;
mod layout;
mod rdna2;
mod xe3;

pub use layout::{Field as PackField, Fields, Fill, Group as PackGroup, Item as PackItem, Layouts, Real, Ty as PackTy, attribute_of};
pub use isa::{OpRows, RegType, ShaderStage};
pub use rdna2::*;
pub use xe3::*;
