//! What detour reads and writes outside MLIR: each GPU's machine code (`rdna2`, `xe3`), records
//! packed by their layout tables, and images written as their memories' bytes; formats are plugins
//! (`Read`, `Write`) loaded into `Binaries`.

mod format;
mod image;
mod rdna2;
mod xe3;

pub use format::{Binaries, Pool, Read, Write};
pub use image::{Hole, Image, ImageBytes, ImageWriter, Pack, Segment, Template};
pub use rdna2::{Rdna2Code, Rdna2Image, Rdna2Isa, Rdna2Ib, Rdna2Words, Rdna2Writer};
pub use xe3::{Xe3Code, Xe3Image, Xe3Isa, Xe3Writer};
