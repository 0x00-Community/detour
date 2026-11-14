mod blur_plan;
mod compose;
mod offscreen;
#[cfg(target_os = "linux")]
mod device;
#[cfg(target_os = "linux")]
mod raster;
mod request;
mod vello_scene;

pub use crate::base::{Backdrop, Content, Effect, Part, Pass, Px, Shadow};
#[cfg(target_os = "linux")]
pub use crate::base::{Dmabuf, Submitted};
pub use blur_plan::Blur;
pub use compose::{Composer, Op};
pub use offscreen::{GpuImage, ImageSet, Offscreen, Overrides, Shared};
pub use request::DeviceRequest;
#[cfg(not(target_os = "linux"))]
pub use offscreen::Exported;
#[cfg(target_os = "linux")]
pub use device::{Device, Exported, Fence};
#[cfg(target_os = "linux")]
pub use raster::Raster;
pub use vello_scene::Vello;
pub use {vello, wgpu};

fn texture(device: &wgpu::Device, (width, height): (u32, u32), usage: wgpu::TextureUsages, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
    })
}
