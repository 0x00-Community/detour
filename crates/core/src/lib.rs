//! detour without a vendor. A vendor's crates (`crates/intel/*`, `crates/apple/*`) build on these:
//!
//! - kmd: the kernel driver and the host CPU as traits ([`Gpu`], [`Host`]); a driver crate implements them, nothing
//!   above it names a driver's structs or ioctls.
//! - compiler: the `detour` dialect, the shader [`Stage`]s, the [`Compiler`] context and the [`FRONTEND`] passes; a
//!   backend loads its dialect and registers its lowering on the same `Compiler`.
//! - image: what a GPU runs, for every vendor: the `image` dialect (memories, command streams, kernels, data), the
//!   layout that gives them addresses ([`Layout`]) and the bytes ([`Image`]); a vendor says how its ops become bytes
//!   ([`ImageTarget`]).
//! - pack: table-driven packing of commands / states / registers ([`Layouts`], [`Template`]).
//! - regalloc: the linear scan every backend allocates its kernels' registers with ([`LinearScan`]; a vendor's
//!   [`RegTarget`] says what its ops and values are).
//! - raster: the CPU half of 2D rasterization, vello_common's strips as quad instances and atlases ([`Strips`]).
//! - scene: the 2D scene a window paints into ([`PaintScene`], recorded as a [`Scene`]) and CSS filters ([`Filter`]);
//!   no GPU in it, it builds everywhere acde runs.

extern crate alloc;

#[cfg(feature = "compiler")]
mod compiler;
mod compose;
mod filter;
#[cfg(feature = "compiler")]
mod image;
#[cfg(feature = "compiler")]
mod kmd;
#[cfg(feature = "compiler")]
mod pack;
#[cfg(feature = "compiler")]
mod raster;
#[cfg(feature = "compiler")]
mod regalloc;
mod scene;

#[cfg(feature = "compiler")]
pub use compiler::*;
pub use compose::*;
pub use filter::*;
#[cfg(feature = "compiler")]
pub use image::*;
#[cfg(feature = "compiler")]
pub use kmd::*;
#[cfg(feature = "compiler")]
pub use pack::*;
#[cfg(feature = "compiler")]
pub use raster::*;
#[cfg(feature = "compiler")]
pub use regalloc::*;
pub use scene::*;

// ⚠ mlirpass-macros' #[pass] / #[patterns] expand to `crate::base::…`: the names they use, at the root.
#[cfg(feature = "compiler")]
#[allow(unused_imports, reason = "for mlirpass' macro expansions")]
pub(crate) use mlirpass::{ConversionPatterns, ConversionRewriter, Lowering, OpId, OptLowering, OptionValue, Options, PassDecl, Run, ValueId};
