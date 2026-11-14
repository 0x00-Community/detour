//! detour as its users see it, aggregated at the repository root (a git dependency on detour gets this crate): the 2D
//! scene ([`PaintScene`], [`Scene`], [`Filter`]) and a page's parts as the compositor draws them ([`Part`], [`Pass`])
//! always, the drivers of the GPU models whose features are on (`xe3`,
//! `agx-g13`, `rdna2`), the Vulkan fallback (`fallback-vulkan`) and the `<canvas>` 3D contexts (`webgl`, `webgl2`,
//! `webgpu`), these four on by default. Building only the models a machine has keeps the rest of detour (their dialects,
//! codegen, kernel interfaces) out of the build.
//!
//! Who does a job is detour's call, not the caller's: [`Backend::pick`] answers for a [`Capability`]. A driver that is
//! built in and has the capability does it; otherwise the Vulkan fallback; with neither (the fallback turned off), it
//! panics — no silent degradation.

extern crate alloc;

#[cfg(feature = "compiler")]
#[allow(unused_imports)]
pub(crate) use mlirpass::{ConversionPatterns, ConversionRewriter, Lowering, OpId, OptLowering, OptionValue, Options, PassDecl, Run, ValueId};

#[path = "../crates/core/src/lib.rs"]
pub mod base;
#[cfg(feature = "fallback-vulkan")]
#[path = "../crates/fallback/src/lib.rs"]
pub mod fallback;
#[cfg(feature = "xe3")]
#[path = "../crates/intel/dialect/src/lib.rs"]
pub mod xe3_dialect;
#[cfg(feature = "xe3")]
#[path = "../crates/intel/codegen/src/lib.rs"]
pub mod xe3_codegen;
#[cfg(feature = "xe3")]
#[path = "../crates/intel/xe/src/lib.rs"]
pub mod xe;
#[cfg(feature = "xe3")]
#[path = "../crates/intel/driver/src/lib.rs"]
pub mod xe3;
#[cfg(feature = "xe3")]
#[path = "../crates/vello_cpu/src/lib.rs"]
pub mod vello_cpu;
#[cfg(feature = "agx-g13")]
#[path = "../crates/apple/codegen/src/lib.rs"]
pub mod agx_codegen;
#[cfg(feature = "agx-g13")]
#[path = "../crates/apple/asahi/src/lib.rs"]
pub mod asahi;
#[cfg(feature = "rdna2")]
#[path = "../crates/amd/codegen/src/lib.rs"]
pub mod rdna2_codegen;
#[cfg(feature = "rdna2")]
#[path = "../crates/amd/amdgpu/src/lib.rs"]
pub mod amdgpu;

pub use crate::base::{
    Backdrop, BoxShadowCommand, Content, Effect, Part, Pass, Px, Shadow, ClipCommand, ColorMatrix, DropShadow, FillCommand, Filter, FilterEffect, GaussianBlur, Glyph, GlyphRunCommand, LayerCommand,
    NormalizedCoord, Paint, PaintRef, PaintScene, RenderCommand, Scene, StrokeCommand,
};
#[cfg(unix)]
pub use crate::base::{Dmabuf, Submitted};
/// Opening a driver's device (`Xe::open_first`) for [`PageRaster`], and the host its `Context` runs on.
#[cfg(feature = "xe3")]
pub use crate::base::{Gpu, Native};

#[cfg(all(feature = "xe3", feature = "fallback-vulkan", target_os = "linux"))]
mod shared_images;
#[cfg(all(feature = "xe3", feature = "fallback-vulkan", target_os = "linux"))]
pub use shared_images::SharedImages;
#[cfg(all(feature = "xe3", target_os = "linux"))]
mod cpu_raster;
#[cfg(all(feature = "xe3", target_os = "linux"))]
pub use cpu_raster::CpuRaster;
#[cfg(all(feature = "xe3", feature = "fallback-vulkan", target_os = "linux"))]
mod page_raster;
#[cfg(all(feature = "xe3", feature = "fallback-vulkan", target_os = "linux"))]
pub use page_raster::PageRaster;

/// Something a user asks detour to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// Draw pages (2D scenes, their layers, filters and shadows) into dma-buf canvases.
    Pages,
    /// WebGPU for pages (`navigator.gpu`, `<canvas>` `webgpu` contexts); feature `webgpu`.
    WebGpu,
    /// `<canvas>` `webgl` contexts (WebGL 1); feature `webgl`.
    WebGl,
    /// `<canvas>` `webgl2` contexts; feature `webgl2`.
    WebGl2,
}

impl Capability {
    /// Whether this build of detour has the capability at all (its feature is on).
    pub const fn built(self) -> bool {
        match self {
            Self::Pages => true,
            Self::WebGpu => cfg!(feature = "webgpu"),
            Self::WebGl => cfg!(feature = "webgl"),
            Self::WebGl2 => cfg!(feature = "webgl2"),
        }
    }
}

/// Who does a [`Capability`] here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// detour's driver for Intel Xe3.
    #[cfg(feature = "xe3")]
    Xe3,
    /// The Vulkan fallback ([`fallback`]).
    #[cfg(feature = "fallback-vulkan")]
    Vulkan,
}

impl Backend {
    /// The built-in drivers that have `capability` today (none for WebGPU yet, M6). Pages are drawn by [`PageRaster`]
    /// (with `xe3`): each pass by Xe3 when it can, the rest by the fallback; so for pages this answers the fallback,
    /// which is always there underneath.
    const fn native(capability: Capability) -> Option<Self> {
        match capability {
            Capability::Pages | Capability::WebGpu | Capability::WebGl | Capability::WebGl2 => None,
        }
    }

    /// ⭐ Who does `capability` here. Panics when the capability's feature is off, or when no driver built in has it and
    /// `fallback-vulkan` is off.
    pub fn pick(capability: Capability) -> Self {
        assert!(capability.built(), "detour: built without {capability:?} (its feature is off)");
        if let Some(native) = Self::native(capability) {
            return native;
        }
        #[cfg(feature = "fallback-vulkan")]
        return Self::Vulkan;
        #[cfg(not(feature = "fallback-vulkan"))]
        panic!("detour: no driver built in does {capability:?}, and fallback-vulkan is off")
    }
}
