use std::os::fd::{AsFd, FromRawFd, OwnedFd};
use std::time::Instant;

use crate::base::{Content, Dmabuf, Native, Pass, Scene, Submitted};
use crate::fallback::{ImageSet, Overrides, Raster as Vulkan, Shared};
use crate::xe3::Raster as Xe3;
use crate::xe::Xe;

use crate::{CpuRaster, SharedImages};

enum Painters<'d> {
    Vulkan(Box<Vulkan>),
    Xe3(Box<Xe3Pages<'d>>),
    Cpu(Box<CpuRaster>),
}

pub struct PageRaster<'d> {
    painters: Painters<'d>,
    pub refused: Option<String>,
}

impl<'d> PageRaster<'d> {
    pub fn new(vulkan: Vulkan, xe: Option<&'d Xe>, max: [u32; 2], images: std::sync::Arc<ImageSet>) -> Self {
        let painters = match xe.and_then(|xe| Xe3::new(xe, max).ok()) {
            Some(xe) => Painters::Xe3(Box::new(Xe3Pages { xe, shared: SharedImages::new(images), vulkan })),
            None => Painters::Vulkan(Box::new(vulkan)),
        };
        Self { painters, refused: None }
    }

    pub fn cpu() -> Self {
        Self { painters: Painters::Cpu(Box::default()), refused: None }
    }

    pub fn xe3(&self) -> bool {
        matches!(self.painters, Painters::Xe3(_))
    }

    pub fn on_cpu(&self) -> bool {
        matches!(self.painters, Painters::Cpu(_))
    }

    pub fn canvases(&mut self, plane: usize, dmabufs: &[Dmabuf<'_>], size: (u32, u32)) -> Vec<Result<(), String>> {
        match &mut self.painters {
            Painters::Vulkan(vulkan) => vulkan.canvases(plane, dmabufs, size),
            Painters::Xe3(x) => {
                x.xe.canvases(plane, dmabufs, size);
                x.vulkan.canvases(plane, dmabufs, size)
            }
            Painters::Cpu(cpu) => cpu.canvases(plane, dmabufs, size),
        }
    }

    pub fn release(&mut self, plane: usize) {
        match &mut self.painters {
            Painters::Vulkan(vulkan) => vulkan.release(plane),
            Painters::Xe3(x) => {
                x.xe.release(plane);
                x.vulkan.release(plane);
            }
            Painters::Cpu(cpu) => cpu.release(plane),
        }
    }

    pub fn idle(&mut self) {
        match &mut self.painters {
            Painters::Vulkan(vulkan) => vulkan.idle(),
            Painters::Xe3(x) => x.vulkan.idle(),
            Painters::Cpu(cpu) => cpu.idle(),
        }
    }

    pub fn icon(&mut self, dmabuf: Dmabuf<'_>, size: (u32, u32), scene: Scene, images: &mut Overrides) -> Result<(), String> {
        match &mut self.painters {
            Painters::Vulkan(vulkan) => vulkan.icon(dmabuf, size, scene, images),
            Painters::Xe3(x) => x.vulkan.icon(dmabuf, size, scene, images),
            Painters::Cpu(cpu) => cpu.icon(dmabuf, size, scene),
        }
    }

    pub fn draw<C: Content>(&mut self, passes: Vec<Pass<C>>, images: &mut Overrides, lock: Option<&Shared>) -> Result<Submitted, String> {
        match &mut self.painters {
            Painters::Vulkan(vulkan) => vulkan.draw(passes, images, lock),
            Painters::Xe3(x) => x.draw(&mut self.refused, passes, images, lock),
            Painters::Cpu(cpu) => cpu.draw(passes),
        }
    }

    fn merge(a: &OwnedFd, b: &OwnedFd) -> Result<OwnedFd, String> {
        let mut data = SyncMerge { name: *b"detour pages\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", fd2: std::os::fd::AsRawFd::as_raw_fd(b), fence: -1, flags: 0, pad: 0 };
        unsafe { rustix::ioctl::ioctl(a.as_fd(), rustix::ioctl::Updater::<{ rustix::ioctl::opcode::read_write::<SyncMerge>(b'>', 3) }, SyncMerge>::new(&mut data)) }.map_err(|e| format!("merging sync_files: {e}"))?;
        Ok(unsafe { OwnedFd::from_raw_fd(data.fence) })
    }
}

struct Xe3Pages<'d> {
    xe: Xe3<'d, Xe, Native>,
    shared: SharedImages<'d, Xe>,
    vulkan: Vulkan,
}

impl<'d> Xe3Pages<'d> {
    fn draw<C: Content>(&mut self, refused: &mut Option<String>, passes: Vec<Pass<C>>, images: &mut Overrides, lock: Option<&Shared>) -> Result<Submitted, String> {
        let Self { xe, shared, vulkan } = self;
        let started = Instant::now();
        let mut rest = Vec::new();
        let mut waited = std::time::Duration::ZERO;
        if !shared.is_empty() {
            let _ = vulkan.device.device.poll(crate::fallback::wgpu::PollType::wait_indefinitely());
        }
        let _ = shared.refresh(&mut xe.cx);
        let mut last = None;
        for pass in passes {
            match xe.submit(&pass, shared) {
                Ok((n, w)) => {
                    last = Some(n);
                    waited += w;
                }
                Err(e) => {
                    *refused = Some(e.to_string());
                    rest.push(pass);
                }
            }
        }
        let xe_fence = last.map(|n| xe.cx.export_fence(n)).transpose().map_err(|e| e.to_string())?;
        let vulkan = if rest.is_empty() { None } else { Some(vulkan.draw(rest, images, lock)?) };
        let fence = match (xe_fence, vulkan.as_ref().map(|v| v.fence.as_ref())) {
            (Some(a), Some(Some(b))) => Some(PageRaster::merge(&a, b)?),
            (Some(a), Some(None)) | (Some(a), None) => Some(a),
            (None, Some(Some(b))) => Some(b.try_clone().map_err(|e| e.to_string())?),
            (None, _) => None,
        };
        waited += vulkan.as_ref().map_or_else(Default::default, |v| v.waited);
        Ok(Submitted { fence, waited, recorded: started.elapsed().saturating_sub(waited) })
    }
}

#[repr(C)]
struct SyncMerge {
    name: [u8; 32],
    fd2: i32,
    fence: i32,
    flags: u32,
    pad: u32,
}
