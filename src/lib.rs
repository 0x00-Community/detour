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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    Pages,
    WebGpu,
    WebGl,
    WebGl2,
}

impl Capability {
    pub const fn built(self) -> bool {
        match self {
            Self::Pages => true,
            Self::WebGpu => cfg!(feature = "webgpu"),
            Self::WebGl => cfg!(feature = "webgl"),
            Self::WebGl2 => cfg!(feature = "webgl2"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    #[cfg(feature = "xe3")]
    Xe3,
    #[cfg(feature = "fallback-vulkan")]
    Vulkan,
}

impl Backend {
    const fn native(capability: Capability) -> Option<Self> {
        match capability {
            Capability::Pages | Capability::WebGpu | Capability::WebGl | Capability::WebGl2 => None,
        }
    }

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
