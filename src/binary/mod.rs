//! What detour reads and writes outside MLIR: Xe3's machine code (`xe3`), records packed by
//! their layout tables, and images written as their memories' bytes; formats are plugins
//! (`Read`, `Write`) loaded into `Binaries`.

mod format;
mod image;
mod xe3;

pub use format::{Binaries, Pool, Read, Write};
pub use image::{Hole, Image, ImageBytes, ImageWriter, Pack, Segment, Template};
pub use xe3::{Xe3Code, Xe3Image, Xe3Isa, Xe3Writer};
