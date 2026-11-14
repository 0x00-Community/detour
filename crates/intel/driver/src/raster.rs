use crate::base::{ATLAS_WIDTH, Gpu, Host, Instance, LUT_WIDTH, Strips, TextureFormat};

use crate::xe3::{Buffer, Codegen, Context, Error, Image2d, Pipeline, Recorder};

const STRIPS: &str = include_str!("raster/strips.mlir");

pub struct Compositor<'d, G: Gpu<Cmd = u64> + 'd> {
    pipeline: Pipeline,
    replace: Pipeline,
    segments: Vec<Vec<(u32, bool)>>,
    quad: Buffer<'d, G>,
    slots: Vec<(Buffer<'d, G>, Image2d<'d, G>, Image2d<'d, G>)>,
    capacity: usize,
    rows: u32,
    masks: u32,
    lut_rows: u32,
    mask_serials: Vec<u64>,
    counts: Vec<u32>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Compositor<'d, G> {
    pub fn new<H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>, capacity: usize, rows: u32, lut_rows: u32) -> Result<Self, Error> {
        Self::with_masks(cx, codegen, capacity, rows, 0, lut_rows)
    }

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

    pub fn rows(&self) -> (u32, u32) {
        (self.rows, self.lut_rows)
    }

    pub fn masks(&self) -> u32 {
        self.masks
    }

    pub fn reset_masks(&mut self) {
        self.mask_serials.fill(0);
    }

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
        let white = (LUT_WIDTH * 4) as usize;
        lut.bytes()?[white..l].copy_from_slice(&strips.lut[white..l]);
        self.counts[slot] = n as u32;
        self.segments[slot] = strips.segments.clone();
        Ok(())
    }

    pub fn draw<H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, slot: usize) -> Result<(), Error> {
        self.draw_range(r, slot, 0..self.counts[slot])
    }

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
