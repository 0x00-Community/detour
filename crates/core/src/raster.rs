//! 2D rasterization for the GPU, vendor-independent: the CPU half. vello_common (the same 0.3 acde's vello_cpu uses)
//! turns paths into sparse strips: per 4-row band, runs of 4x4 tiles with a coverage alpha per pixel ("alpha" runs)
//! and runs fully inside the path ("fill" runs). Clips are vello_common's too: a clipped path's strips are already
//! intersected with the clip (`ClipContext`), nothing here knows of them. Each run becomes one quad instance; the
//! alphas go into an R8 atlas, the paints into an RGBA8 one, both sampled by the fragment shader; a backend draws the
//! instances (`paint * color * coverage`, premultiplied, blended over the target in order).
//!
//! ```text
//!   instance   rect   x, y, w, h in clip space (the run's pixels)
//!              uv     u0, v0, u1, v1 in the alpha atlas: an alpha run's columns, or for a fill run the solid column
//!              color  premultiplied r, g, b, a: a solid paint's color, else 1
//!              paint  u0, du/dx, du/dy, v in the paint atlas: u is affine in the pixel position (a gradient's t
//!                     scaled by 4095 / 4096), interpolated to the pixel center; a solid paint's u is 0 on the white row
//!   alphas     ATLAS_WIDTH texels wide, 4 rows a band; column 0 of band 0 is 255 (the solid column); an alpha
//!              run's pixel column x is atlas column c + x, its 4 alphas the 4 rows (vello's alphas are column-major,
//!              4 a column); a run that would cross the atlas' right edge is split
//!   paints     LUT_WIDTH texels a row; row 0 white; a gradient's row its LUT (vello_common's f32 LUT, premultiplied,
//!              resampled to LUT_WIDTH entries and rounded to unorm8)
//! ```
//!
//! A gradient is sampled the way vello_cpu does (`fine/common/gradient`): t at the pixel center through the encoded
//! transform, then entry `trunc(t * (n - 1))` of its LUT. Here entry `floor(u * 4096)` = `trunc(t * 4095)` of the
//! resampled row; the pad extend is the sampler's clamp to the edge. Only linear gradients with pad extend for now.
//!
//! Like vello_gpu (`visit_strip_fill_segments`, one instance a segment), but the instance is plain floats: the shader
//! does no unpacking, and needs neither uniforms nor integer arithmetic.

use std::collections::HashMap;

use vello_common::color::cache_key::CacheKey;
use vello_common::color::{PremulColor, Srgb};
use vello_common::encode::{EncodedGradient, EncodedKind, GradientCacheKey};
use vello_common::fearless_simd::{Level, dispatch};
use vello_common::geometry::RectU16;
use vello_common::kurbo::Point;
use vello_common::peniko::Extend;
use vello_common::strip::{StripAlphaFillSegment, StripFillSegment, visit_strip_fill_segments};
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;

pub use vello_common;

/// Texels across the alpha atlas.
pub const ATLAS_WIDTH: u32 = 4096;
/// Texels across the paint atlas: entries of a gradient's LUT.
pub const LUT_WIDTH: u32 = 4096;
/// Rows of a band (a strip's height).
pub const BAND: u32 = Tile::HEIGHT as u32;

/// ⭐ One quad: a run of a strip.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Instance {
    pub rect: [f32; 4],
    pub uv: [f32; 4],
    pub color: [f32; 4],
    pub paint: [f32; 4],
}

impl Instance {
    /// Bytes an instance takes in the vertex buffer.
    pub const SIZE: usize = 64;

    /// Writes the instance's bytes (its floats in order, little endian) into `to` in one go, nothing staged (for
    /// write-combined memory).
    #[cfg(target_endian = "little")]
    pub fn write(&self, to: &mut [u8; Self::SIZE]) {
        // SAFETY: `Instance` is `repr(C)`, 16 f32 without padding (SIZE bytes); every bit pattern is a valid u8; on
        // a little-endian host they are its floats' little-endian bytes in order.
        *to = unsafe { std::mem::transmute::<Instance, [u8; Self::SIZE]>(*self) };
    }

    /// Writes the instance's bytes (its floats in order, little endian) into `to` in one go, nothing staged (for
    /// write-combined memory).
    #[cfg(target_endian = "big")]
    pub fn write(&self, to: &mut [u8; Self::SIZE]) {
        let mut v = [0u8; Self::SIZE];
        for (d, f) in v.chunks_exact_mut(4).zip(self.rect.iter().chain(&self.uv).chain(&self.color).chain(&self.paint)) {
            d.copy_from_slice(&f.to_le_bytes());
        }
        *to = v;
    }
}

/// What fills a path.
#[derive(Clone, Copy, Debug)]
pub enum StripPaint<'p> {
    Solid(PremulColor<Srgb>),
    /// vello_common's encoding of a gradient (`EncodeExt::encode_into`)
    Gradient(&'p EncodedGradient),
}

/// Where a run's paint comes from: its color, and the paint atlas coordinates at pixel (x, y) (edges) and per pixel.
#[derive(Clone, Copy)]
struct Source {
    color: [f32; 4],
    /// u = u0 + dux * x + duy * y, in pixels
    u0: f32,
    dux: f32,
    duy: f32,
    v: f32,
}

impl Source {
    fn at(&self, x: u32, y: u32, w: u32) -> [f32; 4] {
        [self.u0 + self.dux * x as f32 + self.duy * y as f32, self.dux * w as f32, self.duy * BAND as f32, self.v]
    }
}

/// A gradient's paint atlas row: its texels, the frame it was last put in the paint atlas, its row there.
type Ramp = (Box<[u8]>, u64, u32);

/// ⭐ A frame's worth: the instances in painting order and the atlases they sample.
pub struct Strips {
    width: u32,
    height: u32,
    /// where the strips' (0, 0) is in the target, and the target's size (`set_region`)
    origin: (u32, u32),
    target: (u32, u32),
    /// runs of `instances` in order: (first instance, whether it clears: its color replaces the target's)
    pub segments: Vec<(u32, bool)>,
    pub instances: Vec<Instance>,
    /// `ATLAS_WIDTH` × (`rows` + `masks`) R8 texels: the frame's alphas, then the masks
    pub atlas: Vec<u8>,
    rows: u32,
    /// rows of masks after the frame's (`mask`): coverage kept across frames, e.g. glyphs
    masks: u32,
    /// per band of the mask rows: the `mask_serial` it last changed at
    mask_bands: Vec<u64>,
    mask_serial: u64,
    /// gradient rows made before, by color ramp: the row's texels, the frame (`clear`s) it was last put in the paint
    /// atlas, and its row there
    ramps: HashMap<CacheKey<GradientCacheKey>, Ramp>,
    frame: u64,
    /// `LUT_WIDTH` × `lut_rows` RGBA8 texels
    pub lut: Vec<u8>,
    lut_rows: u32,
    lut_next: u32,
    /// the next free alpha atlas column, and its band
    column: u32,
    band: u32,
}

impl Strips {
    /// For a `width` × `height` target, with an alpha atlas of `rows` rows (a multiple of `BAND`) and a paint atlas of
    /// `lut_rows` rows.
    pub fn new(width: u32, height: u32, rows: u32, lut_rows: u32) -> Strips {
        Strips::with_masks(width, height, rows, 0, lut_rows)
    }

    /// `new` with `masks` rows of masks (a multiple of `BAND`) after the alpha rows.
    pub fn with_masks(width: u32, height: u32, rows: u32, masks: u32, lut_rows: u32) -> Strips {
        let mut s = Strips {
            width,
            height,
            origin: (0, 0),
            target: (width, height),
            segments: Vec::new(),
            instances: Vec::new(),
            atlas: vec![0; (ATLAS_WIDTH * (rows + masks)) as usize],
            rows,
            masks,
            mask_bands: vec![0; (masks / BAND) as usize],
            mask_serial: 0,
            ramps: HashMap::new(),
            frame: 0,
            lut: vec![0; (LUT_WIDTH * lut_rows * 4) as usize],
            lut_rows,
            lut_next: 1,
            column: 0,
            band: 0,
        };
        s.clear();
        s
    }

    /// Strips pushed from now on are `width` × `height` pixels (what the strip generator was made for) placed at
    /// `origin` in a `target`-sized target: a region of a bigger target, drawn on its own.
    pub fn set_region(&mut self, origin: (u32, u32), (width, height): (u32, u32), target: (u32, u32)) {
        self.origin = origin;
        self.width = width;
        self.height = height;
        self.target = target;
    }

    /// Instances from here on clear (`true`: their color replaces the target's) or blend over.
    fn mode(&mut self, clear: bool) {
        if self.segments.last().is_none_or(|s| s.1 != clear) {
            self.segments.push((self.instances.len() as u32, clear));
        }
    }

    /// The target's pixels (x, y, w, h) in clip space.
    fn clip_rect(&self, [x, y, w, h]: [f32; 4]) -> [f32; 4] {
        let (fw, fh) = (self.target.0 as f32, self.target.1 as f32);
        [x / fw * 2.0 - 1.0, y / fh * 2.0 - 1.0, w / fw * 2.0, h / fh * 2.0]
    }

    /// The solid column of the alpha atlas, the same texel all across: u0 = u1 at its center.
    fn solid_uv(&self) -> [f32; 4] {
        let u = 0.5 / ATLAS_WIDTH as f32;
        let v = 0.5 * BAND as f32 / self.atlas_rows() as f32;
        [u, v, u, v]
    }

    /// Clear the target's pixels in (x, y, w, h) to transparent, here in painting order.
    pub fn clear_rect(&mut self, [x, y, w, h]: [u32; 4]) {
        self.mode(true);
        self.instances.push(Instance {
            rect: self.clip_rect([x as f32, y as f32, w as f32, h as f32]),
            uv: self.solid_uv(),
            color: [0.0; 4],
            paint: [0.0, 0.0, 0.0, self.row_v(0)],
        });
    }

    /// Bytes of `atlas` and `lut` the instances so far use (from the start): what a backend has to upload.
    pub fn used(&self) -> (usize, usize) {
        ((((self.band + 1) * BAND * ATLAS_WIDTH).min(self.rows * ATLAS_WIDTH) as usize), (self.lut_next * LUT_WIDTH * 4) as usize)
    }

    /// Starts a new frame.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.segments.clear();
        let (atlas, _) = self.used();
        self.atlas[..atlas].fill(0);
        for row in 0..BAND {
            self.atlas[(row * ATLAS_WIDTH) as usize] = 255;
        }
        self.lut[..(LUT_WIDTH * 4) as usize].fill(255);
        self.lut_next = 1;
        self.column = 1;
        self.band = 0;
        self.frame += 1;
    }

    /// Rows of the alpha atlas texture: the frame's, then the masks.
    fn atlas_rows(&self) -> u32 {
        self.rows + self.masks
    }

    /// The bands the alpha atlas holds.
    fn bands(&self) -> u32 {
        self.rows / BAND
    }

    /// The center of paint atlas row `row`, as v.
    fn row_v(&self, row: u32) -> f32 {
        (row as f32 + 0.5) / self.lut_rows as f32
    }

    /// A run of `w` pixels at (`x`, `y`) sampling the alpha atlas over `uv`.
    fn quad(&mut self, x: u32, y: u32, w: u32, uv: [f32; 4], src: &Source) {
        let (tx, ty) = (x + self.origin.0, y + self.origin.1);
        self.mode(false);
        self.instances.push(Instance {
            rect: self.clip_rect([tx as f32, ty as f32, w as f32, BAND as f32]),
            uv,
            color: src.color,
            paint: src.at(x, y, w),
        });
    }

    fn fill(&mut self, s: StripFillSegment, src: &Source) {
        let (x, w) = (u32::from(s.tile_x0) * u32::from(Tile::WIDTH), u32::from(s.tile_x1 - s.tile_x0) * u32::from(Tile::WIDTH));
        self.quad(x, u32::from(s.tile_y) * BAND, w, self.solid_uv(), src);
    }

    fn alpha(&mut self, s: StripAlphaFillSegment, alphas: &[u8], src: &Source) -> Result<(), String> {
        let x0 = u32::from(s.tile_x0) * u32::from(Tile::WIDTH);
        let w = u32::from(s.tile_x1 - s.tile_x0) * u32::from(Tile::WIDTH);
        let y = u32::from(s.tile_y) * BAND;
        let col = (s.alpha_idx / BAND) as usize;
        let mut done = 0;
        while done < w {
            if self.column == ATLAS_WIDTH {
                self.column = 0;
                self.band += 1;
            }
            if self.band >= self.bands() {
                return Err(format!("alpha atlas full ({} bands)", self.bands()));
            }
            let n = (w - done).min(ATLAS_WIDTH - self.column);
            for i in 0..n {
                let from = &alphas[(col + (done + i) as usize) * BAND as usize..][..BAND as usize];
                for (row, a) in from.iter().enumerate() {
                    self.atlas[((self.band * BAND + row as u32) * ATLAS_WIDTH + self.column + i) as usize] = *a;
                }
            }
            let (aw, ah) = (ATLAS_WIDTH as f32, self.atlas_rows() as f32);
            let v0 = (self.band * BAND) as f32 / ah;
            let uv = [self.column as f32 / aw, v0, (self.column + n) as f32 / aw, v0 + BAND as f32 / ah];
            self.quad(x0 + done, y, n, uv, src);
            self.column += n;
            done += n;
        }
        Ok(())
    }

    /// A gradient's paint atlas row and how u follows the pixel position.
    fn gradient(&mut self, g: &EncodedGradient) -> Result<Source, String> {
        if !matches!(g.kind, EncodedKind::Linear(_)) || g.extend != Extend::Pad {
            return Err(format!("gradient {:?} with {:?} extend: only linear with pad for now", g.kind, g.extend));
        }
        let row = self.ramp(g)?;
        // t = (transform * p).x: affine in the pixel position p
        let k = (LUT_WIDTH - 1) as f32 / LUT_WIDTH as f32;
        let t = |p: Point| (g.transform * p).x as f32;
        let t0 = t(Point::ZERO);
        Ok(Source { color: [1.0; 4], u0: t0 * k, dux: (t(Point::new(1.0, 0.0)) - t0) * k, duy: (t(Point::new(0.0, 1.0)) - t0) * k, v: self.row_v(row) })
    }

    /// The paint atlas row of `g`'s color ramp: the one it got earlier in this frame, else a new one (its texels
    /// from an earlier frame if the ramp was seen, else made).
    fn ramp(&mut self, g: &EncodedGradient) -> Result<u32, String> {
        let frame = self.frame;
        if let Some((_, f, row)) = self.ramps.get(&g.cache_key)
            && *f == frame
        {
            return Ok(*row);
        }
        if self.lut_next == self.lut_rows {
            return Err(format!("paint atlas full ({} rows)", self.lut_rows));
        }
        let row = self.lut_next;
        self.lut_next += 1;
        let to = &mut self.lut[(row * LUT_WIDTH * 4) as usize..][..(LUT_WIDTH * 4) as usize];
        if let Some((texels, f, r)) = self.ramps.get_mut(&g.cache_key) {
            to.copy_from_slice(texels);
            (*f, *r) = (frame, row);
            return Ok(row);
        }
        // the LUT vello_cpu samples: at the SIMD level it detects too (its mul_add fuses or not with the level)
        let lut = dispatch!(Level::new(), simd => g.f32_lut(simd)).lut();
        let n = lut.len();
        for (j, texel) in to.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            // texel j is what t in [j / 4095, (j + 1) / 4095) samples: entry trunc(t * (n - 1)) at its start
            let e = &lut[(j * (n - 1) / (LUT_WIDTH as usize - 1)).min(n - 1)];
            for (t, v) in texel.iter_mut().zip(e) {
                *t = (v * 255.0).round() as u8;
            }
        }
        if self.ramps.len() >= 256 {
            self.ramps.retain(|_, (_, f, _)| *f == frame);
        }
        self.ramps.insert(g.cache_key.clone(), (to.into(), frame, row));
        Ok(row)
    }

    /// Writes the coverage of `storage`'s strips into the mask rows, over what is there (the larger coverage wins):
    /// strips generated for an `ATLAS_WIDTH` × `masks` target, pixel (x, y) going to mask (x, y). A mask is made once
    /// and drawn (`push_mask`) in any later frame.
    pub fn mask(&mut self, storage: &StripStorage) {
        let bounds = RectU16::new(0, 0, (ATLAS_WIDTH / u32::from(Tile::WIDTH)) as u16, (self.masks / BAND) as u16);
        self.mask_serial += 1;
        let base = (self.rows * ATLAS_WIDTH) as usize;
        visit_strip_fill_segments(
            &storage.strips,
            bounds,
            self,
            |s, seg| {
                let (x0, y) = (u32::from(seg.tile_x0) * u32::from(Tile::WIDTH), u32::from(seg.tile_y) * BAND);
                let w = u32::from(seg.tile_x1 - seg.tile_x0) * u32::from(Tile::WIDTH);
                let col = (seg.alpha_idx / BAND) as usize;
                for i in 0..w as usize {
                    for (row, a) in storage.alphas[(col + i) * BAND as usize..][..BAND as usize].iter().enumerate() {
                        let t = &mut s.atlas[base + ((y + row as u32) * ATLAS_WIDTH + x0) as usize + i];
                        *t = (*t).max(*a);
                    }
                }
                s.mask_bands[seg.tile_y as usize] = s.mask_serial;
            },
            |s, seg| {
                let (x0, y) = (u32::from(seg.tile_x0) * u32::from(Tile::WIDTH), u32::from(seg.tile_y) * BAND);
                let w = u32::from(seg.tile_x1 - seg.tile_x0) * u32::from(Tile::WIDTH);
                for row in 0..BAND {
                    let at = base + ((y + row) * ATLAS_WIDTH + x0) as usize;
                    s.atlas[at..at + w as usize].fill(255);
                }
                s.mask_bands[seg.tile_y as usize] = s.mask_serial;
            },
        );
    }

    /// Clears the mask pixels in (x, y, w, h) (a mask's place being reused).
    pub fn clear_mask(&mut self, [x, y, w, h]: [u32; 4]) {
        self.mask_serial += 1;
        let (x1, y1) = ((x + w).min(ATLAS_WIDTH), (y + h).min(self.masks));
        if x1 <= x || y1 <= y {
            return;
        }
        for row in y..y1 {
            let at = ((self.rows + row) * ATLAS_WIDTH + x) as usize;
            self.atlas[at..at + (x1 - x) as usize].fill(0);
        }
        for band in y / BAND..y1.div_ceil(BAND) {
            self.mask_bands[band as usize] = self.mask_serial;
        }
    }

    /// The `mask_serial` now: masks changed after it are in `mask_changes`.
    pub fn mask_serial(&self) -> u64 {
        self.mask_serial
    }

    /// The byte ranges of `atlas` (mask rows) changed after `serial` (an earlier `mask_serial`).
    pub fn mask_changes(&self, serial: u64) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        let band = (BAND * ATLAS_WIDTH) as usize;
        let base = (self.rows * ATLAS_WIDTH) as usize;
        let mut i = 0;
        std::iter::from_fn(move || {
            while i < self.mask_bands.len() && self.mask_bands[i] <= serial {
                i += 1;
            }
            let start = i;
            while i < self.mask_bands.len() && self.mask_bands[i] > serial {
                i += 1;
            }
            (i > start).then(|| base + start * band..base + i * band)
        })
    }

    /// Mask pixels (`mx`, `my`, `w`, `h`) drawn with their top left at (`x`, `y`) (the strips' pixels) in `color`
    /// (premultiplied), cut to `clip` (x0, y0, x1, y1) and the strips' size.
    pub fn push_mask(&mut self, [x, y]: [i32; 2], [mx, my, w, h]: [u32; 4], color: PremulColor<Srgb>, clip: [i32; 4]) {
        let (x0, y0) = (x.max(clip[0]).max(0), y.max(clip[1]).max(0));
        let x1 = (x + w as i32).min(clip[2]).min(self.width as i32);
        let y1 = (y + h as i32).min(clip[3]).min(self.height as i32);
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let (u, v) = ((mx as i32 + x0 - x) as f32, ((self.rows + my) as i32 + y0 - y) as f32);
        let (cw, ch) = ((x1 - x0) as f32, (y1 - y0) as f32);
        let (aw, ah) = (ATLAS_WIDTH as f32, self.atlas_rows() as f32);
        let (tx, ty) = ((x0 as u32 + self.origin.0) as f32, (y0 as u32 + self.origin.1) as f32);
        self.mode(false);
        self.instances.push(Instance {
            rect: self.clip_rect([tx, ty, cw, ch]),
            uv: [u / aw, v / ah, (u + cw) / aw, (v + ch) / ah],
            color: color.components,
            paint: [0.0, 0.0, 0.0, self.row_v(0)],
        });
    }

    /// ⭐ A filled path's strips (`StripGenerator::generate_filled_path`, clipped there if need be) in `paint`.
    pub fn push(&mut self, storage: &StripStorage, paint: StripPaint<'_>) -> Result<(), String> {
        self.push_alpha(storage, paint, 1.0)
    }

    /// `push` with the paint's opacity multiplied by `alpha` (a group opacity applied to each of its draws).
    pub fn push_alpha(&mut self, storage: &StripStorage, paint: StripPaint<'_>, alpha: f32) -> Result<(), String> {
        let mut src = match paint {
            StripPaint::Solid(c) => Source { color: c.components, u0: 0.0, dux: 0.0, duy: 0.0, v: self.row_v(0) },
            StripPaint::Gradient(g) => self.gradient(g)?,
        };
        if alpha != 1.0 {
            src.color = src.color.map(|c| c * alpha);
        }
        let bounds = RectU16::new(0, 0, self.width.div_ceil(u32::from(Tile::WIDTH)) as u16, self.height.div_ceil(BAND) as u16);
        let mut err = Ok(());
        let mut me = (self, &mut err);
        visit_strip_fill_segments(
            &storage.strips,
            bounds,
            &mut me,
            |(s, e), seg| {
                if e.is_ok() {
                    **e = s.alpha(seg, &storage.alphas, &src);
                }
            },
            |(s, _), seg| s.fill(seg, &src),
        );
        err
    }
}
