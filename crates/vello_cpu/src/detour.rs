//! detour: the coarse rasterization of a scene as plain data, for detour's fine rasterization on the GPU (its roadmap
//! R3 / R4). Made by [`RenderContext::coarse`](crate::vello_cpu::RenderContext::coarse): what `vello_cpu`'s single-threaded
//! dispatcher buckets before its fine rasterizer runs, row by row, and what that fine rasterizer reads. Filter layers
//! are rendered and filtered on the CPU first, as `vello_cpu` does: they are image paints among `filter_paints`.
//! [`RenderContext::coarse_layered`](crate::vello_cpu::RenderContext::coarse_layered) leaves them to the caller instead ([`Layered`]):
//! each one's own coarse rasterization, painted with as an image by id.
//!
//! A row is a strip row (`Tile::HEIGHT` pixels). Its fine rasterization, as `fine::rasterize_region`: when
//! `root_is_blend_target` and the row has commands, the background isolated (a buffer pushed over the target's
//! initial contents); the depth commands front to back (from the last), each filling the depth buckets of its range
//! that no later opaque draw filled yet (a bucket is [`DEPTH_BUCKET_WIDTH`] pixels); the rest initialized as the
//! target says; the render commands back to front, a paint fill skipped where the depth buffer holds a later draw.

#![allow(missing_docs, reason = "detour: plain data, each type documented")]

use alloc::vec::Vec;

use crate::vello_cpu::coarse::CommandBucketer;
use crate::vello_cpu::coarse::RenderCmd;
use crate::vello_cpu::peniko::BlendMode;
use crate::vello_cpu::span::TileAlignedSpan;
use vello_common::encode::EncodedPaint;
use vello_common::mask::Mask;

pub use crate::vello_cpu::coarse::PaintFillAttrs;
/// The synthetic bold of a glyph run (glifo's, which `vello_cpu` does not re-export).
#[cfg(all())]
pub use glifo::FontEmbolden;

/// The image ids filter layers are painted with when they are drawn elsewhere ([`Layered`]): this + the layer's id.
pub const FILTER_IMAGE_BASE: u32 = 1 << 30;

/// ⭐ A filter layer drawn elsewhere: its content's coarse rasterization (`width × height` pixels, its pixmap's
/// bbox: whole tiles), the filter that runs over it then ([`filter`]), and the image id the scenes that composite it
/// paint with (`FILTER_IMAGE_BASE` + its id).
#[derive(Debug)]
pub struct FilterLayer<'s> {
    pub image: u32,
    pub coarse: Coarse<'s>,
    pub filter: crate::vello_cpu::filter_effects::Filter,
    pub transform: crate::vello_cpu::kurbo::Affine,
}

/// ⭐ A scene's coarse rasterization with its filter layers drawn elsewhere
/// ([`RenderContext::coarse_layered`](crate::vello_cpu::RenderContext::coarse_layered)): each filter layer's, in the order they
/// are drawn (a layer before those painting with it), then the root's.
#[derive(Debug)]
pub struct Layered<'s> {
    pub filters: Vec<FilterLayer<'s>>,
    pub root: Coarse<'s>,
}

/// ⭐ `filter` over `pixmap` (a filter layer's drawn content) as `vello_cpu`'s u8 kernel runs it.
pub fn filter(filter: &crate::vello_cpu::filter_effects::Filter, pixmap: &mut vello_common::pixmap::Pixmap, transform: crate::vello_cpu::kurbo::Affine) {
    crate::vello_cpu::filter::filter_lowp(filter, pixmap, &mut crate::vello_cpu::filter::context::ScratchBuffer::new(), transform);
}

/// A depth bucket's width in pixels.
pub const DEPTH_BUCKET_WIDTH: u16 = crate::vello_cpu::coarse::depth::DEPTH_BUCKET_WIDTH;

/// A horizontal run of pixels in a row, tile aligned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub x: u16,
    pub width: u16,
}

impl From<TileAlignedSpan> for Span {
    fn from(s: TileAlignedSpan) -> Self {
        Self { x: s.pixel_x(), width: s.pixel_width() }
    }
}

/// A render command of a row (`coarse::cmd::RenderCmd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmd {
    /// `span` filled with paint `attrs` (an index into [`Coarse::paints`]), its coverage the alpha masks from
    /// `alpha` (an index into [`Coarse::alphas`]: `Tile::HEIGHT` bytes a pixel column, as `vello_cpu` lays them) or full.
    PaintFill { span: Span, alpha: Option<u32>, attrs: u32 },
    /// A layer buffer pushed (over `span`, or the row).
    PushBuf(Option<Span>),
    /// The top layer buffer popped.
    PopBuf,
    /// The top layer buffer composited into the one below over `span` with layer `attrs` (an index into
    /// [`Coarse::layers`]), its coverage from `alpha` or full.
    LayerFill { span: Span, alpha: Option<u32>, attrs: u32 },
}

/// An opaque fill of whole depth buckets `buckets` (front to back, writing the depth buffer) with paint `attrs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthCmd {
    pub buckets: core::ops::Range<u16>,
    pub attrs: u32,
}

/// One strip row's commands.
#[derive(Debug, Clone, Default)]
pub struct Row {
    pub cmds: Vec<Cmd>,
    pub depth: Vec<DepthCmd>,
}

/// How a layer is composited (`coarse::cmd::LayerFillAttrs`).
#[derive(Debug, Clone)]
pub struct LayerAttrs {
    pub blend_mode: BlendMode,
    pub opacity: f32,
    pub mask: Option<Mask>,
    pub draw_id: u32,
}

/// ⭐ A scene's coarse rasterization (its paints and alpha masks borrowed from its `RenderContext`).
#[derive(Debug)]
pub struct Coarse<'s> {
    pub width: u16,
    pub height: u16,
    pub rows: Vec<Row>,
    pub paints: Vec<PaintFillAttrs>,
    pub layers: Vec<LayerAttrs>,
    /// The alpha masks the commands' `alpha` index (thread 0's: single threaded).
    pub alphas: &'s [u8],
    /// The scene's encoded paints (a paint's index into them), then `filter_paints` (the filtered layers as images).
    pub encoded_paints: &'s [EncodedPaint],
    pub filter_paints: Vec<EncodedPaint>,
    pub root_is_blend_target: bool,
}

impl<'s> Coarse<'s> {
    /// What `b` bucketed (its filter paints taken).
    pub(crate) fn of(b: &mut CommandBucketer, alphas: &'s [u8], encoded_paints: &'s [EncodedPaint], (width, height): (u16, u16), root_is_blend_target: bool) -> Self {
        let rows = b
            .rows()
            .iter()
            .map(|r| Row {
                cmds: r
                    .render_cmds
                    .iter()
                    .map(|c| match *c {
                        RenderCmd::PaintFill(p) => Cmd::PaintFill { span: p.span.into(), alpha: p.alpha_idx(), attrs: p.attrs_idx },
                        RenderCmd::PushBuf(s) => Cmd::PushBuf(s.map(Span::from)),
                        RenderCmd::PopBuf => Cmd::PopBuf,
                        RenderCmd::LayerFill(l) => Cmd::LayerFill { span: l.span.into(), alpha: l.alpha_idx(), attrs: l.attrs_idx },
                    })
                    .collect(),
                depth: r
                    .depth_cmds
                    .iter()
                    .map(|d| {
                        let range = d.bucket_range();
                        DepthCmd { buckets: range.start..range.end, attrs: d.attrs_idx }
                    })
                    .collect(),
            })
            .collect();
        let layers = b.layer_fill_attrs.iter().map(|l| LayerAttrs { blend_mode: l.blend_mode, opacity: l.opacity, mask: l.mask.clone(), draw_id: l.draw_id }).collect();
        Self {
            width,
            height,
            rows,
            paints: b.paint_fill_attrs.clone(),
            layers,
            alphas,
            encoded_paints,
            filter_paints: core::mem::take(&mut b.filter_paints),
            root_is_blend_target,
        }
    }
}

/// detour: a pixel center's offset (the paint is sampled at `x + 0.5`).
pub const PIXEL_CENTER: f64 = 0.5;

/// detour: where a gradient's positions come from, in f32: the first pixel center's image `p0` and the advances; pixel
/// (x, y)'s position `fma(x, x_advance, fma(y, y_advance, p0))` (both coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Positions {
    pub p0: (f32, f32),
    pub x_advance: (f32, f32),
    pub y_advance: (f32, f32),
}

impl Positions {
    /// A paint's positions: `transform` of the first pixel center, its advances.
    pub fn new(transform: crate::vello_cpu::kurbo::Affine, x_advance: crate::vello_cpu::kurbo::Vec2, y_advance: crate::vello_cpu::kurbo::Vec2) -> Self {
        let p0 = transform * crate::vello_cpu::kurbo::Point::new(PIXEL_CENTER, PIXEL_CENTER);
        Self { p0: (p0.x as f32, p0.y as f32), x_advance: (x_advance.x as f32, x_advance.y as f32), y_advance: (y_advance.x as f32, y_advance.y as f32) }
    }

    pub fn of(g: &vello_common::encode::EncodedGradient) -> Self {
        Self::new(g.transform, g.x_advance, g.y_advance)
    }

    /// An image's (in its pixels).
    pub fn of_image(i: &vello_common::encode::EncodedImage) -> Self {
        Self::new(i.transform, i.x_advance, i.y_advance)
    }

    #[inline(always)]
    pub fn at(&self, x: f32, y: f32) -> (f32, f32) {
        (x.mul_add(self.x_advance.0, y.mul_add(self.y_advance.0, self.p0.0)), x.mul_add(self.x_advance.1, y.mul_add(self.y_advance.1, self.p0.1)))
    }
}


/// detour: `x^e` (x ≥ 0) as detour's fine rasterization on the GPU computes it too, where `vello_cpu` uses libm's
/// `powf` (box shadows): `exp2(e log2(x))`, 0 for x ≤ 0 or a subnormal x. `log2` from the exponent bits and the
/// mantissa m in [√½, √2) by `2/ln 2 · atanh((m - 1) / (m + 1))` (odd series to t⁹); `exp2` from ⌊y⌋ in the
/// exponent bits and 2^f by its Taylor series to f⁸; every step a fused multiply-add or an exact operation.
pub fn pow(x: f32, e: f32) -> f32 {
    let bits = x.to_bits();
    let biased = (bits >> 23) & 0xff;
    if x <= 0.0 || biased == 0 {
        return 0.0;
    }
    let mut exponent = biased as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > core::f32::consts::SQRT_2 {
        m *= 0.5;
        exponent += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let s = t * t;
    let p = (1.0_f32 / 9.0).mul_add(s, 1.0 / 7.0).mul_add(s, 1.0 / 5.0).mul_add(s, 1.0 / 3.0).mul_add(s, 1.0);
    let log2 = (t * p).mul_add(2.0 / core::f32::consts::LN_2, exponent as f32);
    exp2(e * log2)
}

/// detour: `2^y` as [`pow`] computes it (0 below 2⁻¹²⁶, infinity above 2¹²⁷).
pub fn exp2(y: f32) -> f32 {
    let yi = y.floor();
    if yi < -126.0 {
        return 0.0;
    }
    if yi > 127.0 {
        return f32::INFINITY;
    }
    let f = y - yi;
    // ln 2^k / k!, k = 8 ..= 0, by Horner
    // (as bits: the f32 nearest each, what the GPU uses too)
    const C: [f32; 9] = [f32::from_bits(0x35b16011), f32::from_bits(0x377fe5fe), f32::from_bits(0x39218489), f32::from_bits(0x3aaec3ff), f32::from_bits(0x3c1d955b), f32::from_bits(0x3d635847), f32::from_bits(0x3e75fdf0), f32::from_bits(0x3f317218), f32::from_bits(0x3f800000)];
    let mut p = C[0];
    for c in &C[1..] {
        p = p.mul_add(f, *c);
    }
    f32::from_bits((p.to_bits() as i32 + ((yi as i32) << 23)) as u32)
}

/// ⭐ detour: a scene recorded for the GPU's coarse rasterization ([`RenderContext::new_recording`](crate::vello_cpu::RenderContext::new_recording)):
/// its draws' geometry, not stripped, in order, with their paint attributes (each its own, empty or not; draw ids in
/// vello_cpu's order: a fill's, a layer's at its pop) and the layers pushed and popped.
#[derive(Debug, Clone, Default)]
pub struct Recording {
    pub width: u16,
    pub height: u16,
    pub items: Vec<Item>,
    /// The first thing drawn the recording cannot take (vello_cpu's own rasterization then).
    pub refused: Option<&'static str>,
}

/// What a draw covers: a path filled with a fill rule under a transform (a stroke's expanded), or a fast path's
/// rectangle (`fill_rect_fast`: in pixels, vello_common's `rect::render` strips it).
#[derive(Debug, Clone)]
pub enum Geometry {
    Path { path: crate::vello_cpu::kurbo::BezPath, fill_rule: crate::vello_cpu::peniko::Fill, transform: crate::vello_cpu::kurbo::Affine },
    Rect(crate::vello_cpu::kurbo::Rect),
}

/// One thing a recording does.
#[derive(Debug, Clone)]
pub enum Item {
    Fill { geometry: Geometry, attrs: PaintFillAttrs },
    /// a layer pushed, clipped to a path (non-zero or even-odd, its transform)
    Push { clip: Option<Geometry> },
    /// the last layer pushed, popped: how it is composited
    Pop(LayerAttrs),
}
