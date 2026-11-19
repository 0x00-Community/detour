//! The GPU dialects detour adds to mlirformat: Xe3's instruction set and command
//! stream (`xe3`), its register types and op tables (`isa`), and the image a GPU runs (`image`).
//! Load them into an `MlirOpt` with `opt.load(Xe3Dialect)` and the like.

mod image;
mod isa;
mod xe3;

pub use image::{
    Containers as ImageContainers, Data as ImageData, Field as PackField, Fields, Fill, Group as PackGroup, ImageDialect, Item as PackItem, Layouts, MemType as ImageMemType, Memory as ImageMemory, Placed as ImagePlaced, Real, Stream as ImageStream, Ty as PackTy,
    Words as ImageWords, attribute_of,
};
pub use isa::{IsaInt, OpRows, RegType, ShaderStage};
pub use xe3::*;
