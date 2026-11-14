// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::sync::Arc;
use alloc::vec::Vec;
use vello_common::pixmap::Pixmap;

#[derive(Debug, Default)]
pub(crate) struct FilterContext {
    /// The rendered pixmaps for each filter layer.
    layers: Vec<Option<Arc<Pixmap>>>,
    scratch: ScratchBuffer,
    /// detour: the filter layers are drawn elsewhere (on the GPU): each one an image by id
    /// (`detour::FILTER_IMAGE_BASE` + its layer id) rather than a pixmap
    elsewhere: bool,
}

impl FilterContext {
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            scratch: ScratchBuffer::new(),
            elsewhere: false,
        }
    }

    /// detour: every filter layer drawn elsewhere, an image by id.
    pub(crate) fn elsewhere() -> Self {
        Self { elsewhere: true, ..Self::default() }
    }

    /// detour: what filter layer `id` is painted with: its pixmap, or its image by id when drawn elsewhere.
    pub(crate) fn filter_source(&self, id: usize) -> Option<vello_common::paint::ImageSource> {
        use vello_common::paint::{ImageId, ImageSource};
        if self.elsewhere {
            return Some(ImageSource::OpaqueId { id: ImageId::new(crate::vello_cpu::detour::FILTER_IMAGE_BASE + id as u32), may_have_transparency: true });
        }
        self.filter_layer(id).map(ImageSource::Pixmap)
    }

    pub(crate) fn scratch(&mut self) -> &mut ScratchBuffer {
        &mut self.scratch
    }

    pub(crate) fn set_layer(&mut self, id: usize, pixmap: Pixmap) {
        if id >= self.layers.len() {
            self.layers.resize_with(id + 1, || None);
        }
        self.layers[id] = Some(Arc::new(pixmap));
    }

    pub(crate) fn filter_layer(&self, id: usize) -> Option<Arc<Pixmap>> {
        self.layers.get(id).and_then(Option::as_ref).cloned()
    }
}

#[derive(Debug, Default)]
pub(crate) struct ScratchBuffer {
    scratch_buffer: Option<Pixmap>,
}

impl ScratchBuffer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get_scratch_buffer(&mut self, width: u16, height: u16) -> &mut Pixmap {
        match &mut self.scratch_buffer {
            None => {
                self.scratch_buffer = Some(Pixmap::new(width, height));
            }
            Some(buf) if buf.width() < width || buf.height() < height => {
                buf.resize(width, height);
            }
            Some(_) => {}
        }

        self.scratch_buffer.as_mut().unwrap()
    }
}
