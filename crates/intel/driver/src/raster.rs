//! ⭐ The strip compositor on Xe3: draws what `detour-raster` made (quad instances over an R8 alpha atlas and an RGBA8
//! paint atlas) through the runtime. One pipeline (`raster/strips.mlir`) compiled once; per ring slot an instance
//! buffer and the atlases, written when the slot is free (the GPU may still read the other slots' frames).

use crate::base::{ATLAS_WIDTH, Gpu, Host, Instance, LUT_WIDTH, Strips, TextureFormat};

use crate::xe3::{Buffer, Codegen, Context, Error, Image2d, Pipeline, Recorder};

const STRIPS: &str = include_str!("raster/strips.mlir");

/// ⭐ Draws `Strips`.
pub struct Compositor<'d, G: Gpu<Cmd = u64> + 'd> {
    pipeline: Pipeline,
    /// the same without blending: the instance's color replaces the target's (`Strips::clear_rect`)
    replace: Pipeline,
    /// per ring slot: `Strips::segments`
    segments: Vec<Vec<(u32, bool)>>,
    quad: Buffer<'d, G>,
    /// per ring slot: instances, alpha atlas, paint atlas
    slots: Vec<(Buffer<'d, G>, Image2d<'d, G>, Image2d<'d, G>)>,
    capacity: usize,
    rows: u32,
    masks: u32,
    lut_rows: u32,
    /// per ring slot: the `Strips::mask_serial` its masks are up to
    mask_serials: Vec<u64>,
    /// instances uploaded per slot
    counts: Vec<u32>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Compositor<'d, G> {
    /// For up to `capacity` instances a frame, an alpha atlas of `rows` rows and a paint atlas of `lut_rows` rows.
    pub fn new<H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>, capacity: usize, rows: u32, lut_rows: u32) -> Result<Self, Error> {
        Self::with_masks(cx, codegen, capacity, rows, 0, lut_rows)
    }

    /// `new` with `masks` rows of masks after the alpha rows (`Strips::with_masks`).
    pub fn with_masks<H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>, capacity: usize, rows: u32, masks: u32, lut_rows: u32) -> Result<Self, Error> {
        let pipeline = cx.pipeline(codegen, STRIPS, "strips")?;
        let replace = cx.pipeline(codegen, &STRIPS.replace("blend = \"premultiplied\", ", ""), "strips")?;
        let quad = cx.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
        let mut slots = Vec::new();
        for _ in 0..cx.ring() {
            let instances = cx.buffer(&vec![0; capacity * Instance::SIZE])?;
            let atlas = cx.texture(ATLAS_WIDTH, rows + masks, TextureFormat::R8Unorm, &vec![0; (ATLAS_WIDTH * (rows + masks)) as usize])?;
            let mut texels = vec![0; (LUT_WIDTH * lut_rows * 4) as usize];
            texels[..(LUT_WIDTH * 4) as usize].fill(255);
            let lut = cx.texture(LUT_WIDTH, lut_rows, TextureFormat::Rgba8Unorm, &texels)?;
            slots.push((instances, atlas, lut));
        }
        Ok(Compositor { pipeline, replace, segments: vec![Vec::new(); slots.len()], quad, counts: vec![0; slots.len()], mask_serials: vec![0; slots.len()], slots, capacity, rows, masks, lut_rows })
    }

    /// The atlas rows `Strips::new` must be made with: alpha, paint.
    pub fn rows(&self) -> (u32, u32) {
        (self.rows, self.lut_rows)
    }

    /// The mask rows `Strips::with_masks` must be made with.
    pub fn masks(&self) -> u32 {
        self.masks
    }

    /// A new `Strips` (after a panic) restarts its mask serial: upload every mask row again.
    pub fn reset_masks(&mut self) {
        self.mask_serials.fill(0);
    }

    /// Writes `strips` into ring slot `slot` (free: `Context::wait` it first).
    pub fn upload(&mut self, slot: usize, strips: &Strips) -> Result<(), Error> {
        let n = strips.instances.len();
        if n > self.capacity {
            return Err(Error::msg(format!("{n} strip instances, room for {}", self.capacity)));
        }
        let (instances, atlas, lut) = &mut self.slots[slot];
        for (to, i) in instances.bytes()?.as_chunks_mut::<{ Instance::SIZE }>().0.iter_mut().zip(&strips.instances) {
            i.write(to);
        }
        let (a, l) = strips.used();
        let to = atlas.bytes()?;
        to[..a].copy_from_slice(&strips.atlas[..a]);
        for r in strips.mask_changes(self.mask_serials[slot]) {
            to[r.clone()].copy_from_slice(&strips.atlas[r]);
        }
        self.mask_serials[slot] = strips.mask_serial();
        // row 0 (white) is the same every frame: written when the slot was made
        let white = (LUT_WIDTH * 4) as usize;
        lut.bytes()?[white..l].copy_from_slice(&strips.lut[white..l]);
        self.counts[slot] = n as u32;
        self.segments[slot] = strips.segments.clone();
        Ok(())
    }

    /// Draws what `upload` put in `slot` into the recorder's target.
    pub fn draw<H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, slot: usize) -> Result<(), Error> {
        self.draw_range(r, slot, 0..self.counts[slot])
    }

    /// Draws instances `range` of what `upload` put in `slot` into the recorder's target: several targets' strips
    /// uploaded together (`Strips::set_region` per target), each drawn into its own target, all in one submission.
    pub fn draw_range<H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, slot: usize, range: std::ops::Range<u32>) -> Result<(), Error> {
        let (instances, atlas, lut) = &self.slots[slot];
        let end = range.end.min(self.counts[slot]);
        let segments = &self.segments[slot];
        for (i, &(first, clear)) in segments.iter().enumerate() {
            let next = segments.get(i + 1).map_or(self.counts[slot], |s| s.0);
            let (a, b) = (first.max(range.start), next.min(end));
            if a >= b {
                continue;
            }
            let at = instances.addr + u64::from(a) * Instance::SIZE as u64;
            let size = u64::from(b - a) * Instance::SIZE as u64;
            let pipeline = if clear { &self.replace } else { &self.pipeline };
            r.draw_at(pipeline, &[(self.quad.addr, self.quad.size), (at, size)], &[atlas, lut], 4, b - a)?;
        }
        Ok(())
    }
}
