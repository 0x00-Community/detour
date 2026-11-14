//! ⭐ The fallback's GPU images (2D canvases, their shadows: drawn by wgpu) as Xe3's pages read them: each one made
//! exportable ([`Shared::exporting`](crate::fallback::Shared::exporting)) imported into Xe3's VM once, by its dma-buf.

use std::collections::HashMap;
use std::os::fd::AsFd;
use std::sync::{Arc, Weak};

use crate::base::{Gpu, Host};
use crate::fallback::{GpuImage, ImageSet};
use crate::xe3::{Context, Error, GpuTexels, ImportedCanvas, XeImages};

/// ⭐ The images of an [`ImageSet`] imported on Xe3: [`XeImages`] for `Pages` / `Raster`.
pub struct SharedImages<'d, G: Gpu<Cmd = u64> + 'd> {
    set: Arc<ImageSet>,
    imported: HashMap<u64, (Weak<GpuImage>, ImportedCanvas<'d, G>, GpuTexels)>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> SharedImages<'d, G> {
    pub fn new(set: Arc<ImageSet>) -> Self {
        Self { set, imported: HashMap::new() }
    }

    /// Whether no image of the set is alive.
    pub fn is_empty(&self) -> bool {
        self.set.live().is_empty()
    }

    /// Imports the images made since (those exported and at most 65535 × 65535), forgets those gone. Before a frame;
    /// the fallback's work on them is the caller's to have waited for. An image that fails to import is left out (a
    /// page painting with it is the fallback's then).
    pub fn refresh<H: Host>(&mut self, cx: &mut Context<'d, G, H>) -> Vec<Error> {
        self.imported.retain(|_, (image, ..)| image.strong_count() > 0);
        let mut failed = Vec::new();
        for image in self.set.live() {
            let (w, h) = (image.texture.width(), image.texture.height());
            let Some(exported) = image.exported().filter(|_| !self.imported.contains_key(&image.id()) && w <= u32::from(u16::MAX) && h <= u32::from(u16::MAX)) else { continue };
            match cx.import_canvas(exported.fd.as_fd(), exported.offset, exported.stride, [w, h]) {
                Ok(imported) => {
                    let texels = GpuTexels { addr: imported.canvas.addr, stride: imported.canvas.stride, size: (w as u16, h as u16), straight: true };
                    self.imported.insert(image.id(), (Arc::downgrade(&image), imported, texels));
                }
                Err(e) => failed.push(e),
            }
        }
        failed
    }
}

impl<G: Gpu<Cmd = u64>> XeImages for SharedImages<'_, G> {
    fn find(&self, id: u64) -> Option<GpuTexels> {
        self.imported.get(&id).map(|(.., t)| *t)
    }
}
