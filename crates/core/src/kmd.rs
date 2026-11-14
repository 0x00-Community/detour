//! The kernel driver and the host CPU as traits. A GPU driver (`detour-intel-xe`, `detour-apple-asahi`) implements [`Gpu`];
//! a CPU architecture implements [`Host`]. Nothing above this names a driver's structs or ioctls. The DRM part every
//! Linux driver shares (render node, GEM handles, syncobjs) is here too ([`Node`], [`Syncobj`]).

#[cfg(target_os = "linux")]
mod drm;
mod gpu;
mod host;

#[cfg(target_os = "linux")]
pub use drm::*;
pub use gpu::*;
pub use host::*;
