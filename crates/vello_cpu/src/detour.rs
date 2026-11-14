#![allow(missing_docs, reason = "detour: plain data, each type documented")]

use alloc::vec::Vec;

use crate::vello_cpu::coarse::CommandBucketer;
use crate::vello_cpu::coarse::RenderCmd;
use crate::vello_cpu::peniko::BlendMode;
use crate::vello_cpu::span::TileAlignedSpan;
use vello_common::encode::EncodedPaint;
use vello_common::mask::Mask;

pub use crate::vello_cpu::coarse::PaintFillAttrs;

#[cfg(all())]
pub use glifo::FontEmbolden;

pub const FILTER_IMAGE_BASE: u32 = 1 << 30;

#[derive(Debug)]
pub struct FilterLayer<'s> {
    pub image: u32,
    pub coarse: Coarse<'s>,
    pub filter: crate::vello_cpu::filter_effects::Filter,
    pub transform: crate::vello_cpu::kurbo::Affine,
}

#[derive(Debug)]
pub struct Layered<'s> {
    pub filters: Vec<FilterLayer<'s>>,
    pub root: Coarse<'s>,
}

pub fn filter(filter: &crate::vello_cpu::filter_effects::Filter, pixmap: &mut vello_common::pixmap::Pixmap, transform: crate::vello_cpu::kurbo::Affine) {
    crate::vello_cpu::filter::filter_lowp(filter, pixmap, &mut crate::vello_cpu::filter::context::ScratchBuffer::new(), transform);
}

pub const DEPTH_BUCKET_WIDTH: u16 = crate::vello_cpu::coarse::depth::DEPTH_BUCKET_WIDTH;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmd {
    PaintFill { span: Span, alpha: Option<u32>, attrs: u32 },
    PushBuf(Option<Span>),
    PopBuf,
    LayerFill { span: Span, alpha: Option<u32>, attrs: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthCmd {
    pub buckets: core::ops::Range<u16>,
    pub attrs: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Row {
    pub cmds: Vec<Cmd>,
    pub depth: Vec<DepthCmd>,
}

#[derive(Debug, Clone)]
pub struct LayerAttrs {
    pub blend_mode: BlendMode,
    pub opacity: f32,
    pub mask: Option<Mask>,
    pub draw_id: u32,
}

#[derive(Debug)]
pub struct Coarse<'s> {
    pub width: u16,
    pub height: u16,
    pub rows: Vec<Row>,
    pub paints: Vec<PaintFillAttrs>,
    pub layers: Vec<LayerAttrs>,
    pub alphas: &'s [u8],
    pub encoded_paints: &'s [EncodedPaint],
    pub filter_paints: Vec<EncodedPaint>,
    pub root_is_blend_target: bool,
}

impl<'s> Coarse<'s> {
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

pub const PIXEL_CENTER: f64 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Positions {
    pub p0: (f32, f32),
    pub x_advance: (f32, f32),
    pub y_advance: (f32, f32),
}

impl Positions {
    pub fn new(transform: crate::vello_cpu::kurbo::Affine, x_advance: crate::vello_cpu::kurbo::Vec2, y_advance: crate::vello_cpu::kurbo::Vec2) -> Self {
        let p0 = transform * crate::vello_cpu::kurbo::Point::new(PIXEL_CENTER, PIXEL_CENTER);
        Self { p0: (p0.x as f32, p0.y as f32), x_advance: (x_advance.x as f32, x_advance.y as f32), y_advance: (y_advance.x as f32, y_advance.y as f32) }
    }

    pub fn of(g: &vello_common::encode::EncodedGradient) -> Self {
        Self::new(g.transform, g.x_advance, g.y_advance)
    }

    pub fn of_image(i: &vello_common::encode::EncodedImage) -> Self {
        Self::new(i.transform, i.x_advance, i.y_advance)
    }

    #[inline(always)]
    pub fn at(&self, x: f32, y: f32) -> (f32, f32) {
        (x.mul_add(self.x_advance.0, y.mul_add(self.y_advance.0, self.p0.0)), x.mul_add(self.x_advance.1, y.mul_add(self.y_advance.1, self.p0.1)))
    }
}

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

pub fn exp2(y: f32) -> f32 {
    let yi = y.floor();
    if yi < -126.0 {
        return 0.0;
    }
    if yi > 127.0 {
        return f32::INFINITY;
    }
    let f = y - yi;
    const C: [f32; 9] = [f32::from_bits(0x35b16011), f32::from_bits(0x377fe5fe), f32::from_bits(0x39218489), f32::from_bits(0x3aaec3ff), f32::from_bits(0x3c1d955b), f32::from_bits(0x3d635847), f32::from_bits(0x3e75fdf0), f32::from_bits(0x3f317218), f32::from_bits(0x3f800000)];
    let mut p = C[0];
    for c in &C[1..] {
        p = p.mul_add(f, *c);
    }
    f32::from_bits((p.to_bits() as i32 + ((yi as i32) << 23)) as u32)
}

#[derive(Debug, Clone, Default)]
pub struct Recording {
    pub width: u16,
    pub height: u16,
    pub items: Vec<Item>,
    pub refused: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub enum Geometry {
    Path { path: crate::vello_cpu::kurbo::BezPath, fill_rule: crate::vello_cpu::peniko::Fill, transform: crate::vello_cpu::kurbo::Affine },
    Rect(crate::vello_cpu::kurbo::Rect),
}

#[derive(Debug, Clone)]
pub enum Item {
    Fill { geometry: Geometry, attrs: PaintFillAttrs },
    Push { clip: Option<Geometry> },
    Pop(LayerAttrs),
}
