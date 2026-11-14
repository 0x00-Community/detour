#![warn(clippy::print_stdout, clippy::print_stderr)]

#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]

#![forbid(unsafe_code)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "We cast u16s to u8 in various places where we know for sure that it's < 256"
)]

extern crate alloc;
extern crate core;

#[cfg(all())]
use png as _;
#[cfg(all())]
extern crate std;

#[cfg(all(not(all()), not(all())))]
compile_error!("vello_cpu must have at least one of the u8 or f32 pipelines enabled");

mod render;

pub mod detour;
mod coarse;
mod dispatch;
mod filter;
mod record;
#[cfg(all())]
mod text;
#[cfg(all(all(), all(), debug_assertions, not(doc)))]
mod text_debug;
mod util;

#[doc(hidden)]
pub mod fine;
#[doc(hidden)]
pub mod region;
#[doc(hidden)]
pub mod span;

pub use render::{
    PixelFormat, RasterizerSettings, RenderContext, RenderSettings, Resources, TargetInit,
};

#[cfg(all())]
pub use glifo::Glyph;
#[cfg(all())]
pub use text::{CpuGlyphRunBackend, GlyphRunBuilder};
pub use vello_common::reexports::cpu::*;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum RenderMode {
    #[default]
    OptimizeSpeed,
    OptimizeQuality,
}
