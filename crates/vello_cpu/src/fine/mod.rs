mod common;
mod highp;
mod lowp;

use crate::vello_cpu::coarse::depth::DepthBuffer;
use crate::vello_cpu::coarse::{CommandBucketer, RenderCmd, RowState};
use crate::vello_cpu::filter::context::ScratchBuffer;
use crate::vello_cpu::fine::common::gradient::GradientPainter;
pub(crate) use crate::vello_cpu::fine::common::gradient::calculate_t_vals;
pub(crate) use crate::vello_cpu::fine::common::gradient::linear::SimdLinearKind;
pub(crate) use crate::vello_cpu::fine::common::gradient::radial::SimdRadialKind;
pub(crate) use crate::vello_cpu::fine::common::gradient::sweep::SimdSweepKind;
use crate::vello_cpu::fine::common::image::{FilteredImagePainter, NNImagePainter, PlainNNImagePainter};
use crate::vello_cpu::fine::common::rounded_blurred_rect::BlurredRoundedRectFiller;
use crate::vello_cpu::peniko::{BlendMode, ImageQuality};
use crate::vello_cpu::region::Region;
use crate::vello_cpu::util::EncodedImageExt;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::iter;
use vello_common::TargetInit;
use vello_common::color::AlphaColor;
use vello_common::encode::{
    EncodedBlurredRoundedRectangle, EncodedGradient, EncodedImage, EncodedKind, EncodedPaint,
};
use vello_common::fearless_simd::{
    Bytes, Simd, SimdBase, SimdFloat, SimdInt, SimdInto, f32x4, f32x8, f32x16, u8x16, u8x32, u32x4,
    u32x8,
};
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, ImageSource, Paint, PremulColor, Tint};
use vello_common::pixmap::Pixmap;
use vello_common::simd::Splat4thExt;
use vello_common::tile::Tile;
use vello_common::util::{VecPool, f32_to_u8};

#[doc(hidden)]
pub use crate::vello_cpu::coarse::PaintFillAttrs;
#[doc(hidden)]
pub use crate::vello_cpu::span::Span;
#[doc(hidden)]
pub use crate::vello_cpu::span::TileAlignedSpan;
pub use highp::F32Kernel;
pub use lowp::U8Kernel;

const PIXEL_CENTER_OFFSET: f64 = 0.5;

pub(crate) const COLOR_COMPONENTS: usize = 4;

pub(crate) const TILE_HEIGHT_COMPONENTS: usize = Tile::HEIGHT as usize * COLOR_COMPONENTS;

pub trait Numeric: Copy + Default + Clone + Debug + PartialEq + Send + Sync + 'static {
    const ZERO: Self;

    const ONE: Self;
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Numeric for u8 {
    const ZERO: Self = 0;
    const ONE: Self = 255;
}

pub trait NumericVec<S: Simd>: Copy + Clone + Send + Sync {
    fn from_f32(simd: S, val: f32x16<S>) -> Self;

    fn from_u8(simd: S, val: u8x16<S>) -> Self;
}

impl<S: Simd> NumericVec<S> for f32x16<S> {
    #[inline(always)]
    fn from_f32(_: S, val: Self) -> Self {
        val
    }

    #[inline(always)]
    fn from_u8(simd: S, val: u8x16<S>) -> Self {
        let converted = u8_to_f32(val);
        converted * Self::splat(simd, 1.0 / 255.0)
    }
}

impl<S: Simd> NumericVec<S> for u8x16<S> {
    #[inline(always)]
    fn from_f32(simd: S, val: f32x16<S>) -> Self {
        let v1 = f32x16::splat(simd, 255.0);
        let v2 = f32x16::splat(simd, 0.5);
        let mulled = val.mul_add(v1, v2);

        f32_to_u8(mulled)
    }

    #[inline(always)]
    fn from_u8(_: S, val: Self) -> Self {
        val
    }
}

#[inline(always)]
pub(crate) fn u8_to_f32<S: Simd>(val: u8x16<S>) -> f32x16<S> {
    let simd = val.simd;
    let zeroes = u8x16::splat(simd, 0);

    #[cfg(target_endian = "little")]
    let (p1, p2, p3, p4) = {
        let lo = simd.zip_low_u8x16(val, zeroes);
        let hi = simd.zip_high_u8x16(val, zeroes);

        (
            simd.zip_low_u8x16(lo, zeroes),
            simd.zip_high_u8x16(lo, zeroes),
            simd.zip_low_u8x16(hi, zeroes),
            simd.zip_high_u8x16(hi, zeroes),
        )
    };

    #[cfg(target_endian = "big")]
    let (p1, p2, p3, p4) = {
        let lo = simd.zip_low_u8x16(zeroes, val);
        let hi = simd.zip_high_u8x16(zeroes, val);

        (
            simd.zip_low_u8x16(zeroes, lo),
            simd.zip_high_u8x16(zeroes, lo),
            simd.zip_low_u8x16(zeroes, hi),
            simd.zip_high_u8x16(zeroes, hi),
        )
    };

    let p1 = p1.bitcast::<u32x4<S>>().to_float::<f32x4<S>>();
    let p2 = p2.bitcast::<u32x4<S>>().to_float::<f32x4<S>>();
    let p3 = p3.bitcast::<u32x4<S>>().to_float::<f32x4<S>>();
    let p4 = p4.bitcast::<u32x4<S>>().to_float::<f32x4<S>>();

    simd.combine_f32x8(simd.combine_f32x4(p1, p2), simd.combine_f32x4(p3, p4))
}

pub trait CompositeType<N: Numeric, S: Simd>: Copy + Clone + Send + Sync {
    const LENGTH: usize;

    fn from_slice(simd: S, slice: &[N]) -> Self;

    fn from_color(simd: S, color: [N; 4]) -> Self;
}

impl<S: Simd> CompositeType<f32, S> for f32x16<S> {
    const LENGTH: usize = 16;

    #[inline(always)]
    fn from_slice(simd: S, slice: &[f32]) -> Self {
        <Self as SimdBase<_>>::from_slice(simd, slice)
    }

    #[inline(always)]
    fn from_color(simd: S, color: [f32; 4]) -> Self {
        Self::block_splat(f32x4::from_slice(simd, &color[..]))
    }
}

impl<S: Simd> CompositeType<u8, S> for u8x32<S> {
    const LENGTH: usize = 32;

    #[inline(always)]
    fn from_slice(simd: S, slice: &[u8]) -> Self {
        <Self as SimdBase<_>>::from_slice(simd, slice)
    }

    #[inline(always)]
    fn from_color(simd: S, color: [u8; 4]) -> Self {
        u32x8::block_splat(u32x4::splat(simd, u32::from_ne_bytes(color))).to_bytes()
    }
}

pub trait FineKernel<S: Simd>: Send + Sync + 'static {
    type Numeric: Numeric;

    type Composite: CompositeType<Self::Numeric, S>;

    type NumericVec: NumericVec<S>;

    fn extract_color(color: PremulColor) -> [Self::Numeric; 4];

    fn pack(simd: S, scratch: &[Self::Numeric], width: usize, region: &mut Region<'_>);

    fn unpack(simd: S, region: &mut Region<'_>, width: usize, scratch: &mut [Self::Numeric]);

    #[expect(
        private_interfaces,
        reason = "`FineKernel` is public but this specific method is not needed."
    )]
    fn filter_layer(
        pixmap: &mut Pixmap,
        filter: &Filter,
        filter_scratch: &mut ScratchBuffer,
        transform: Affine,
    );

    fn copy_solid(simd: S, target: &mut [Self::Numeric], color: [Self::Numeric; 4]);
    fn gradient_painter<'a>(
        simd: S,
        gradient: &'a EncodedGradient,
        t_vals: &'a [f32],
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || GradientPainter::new(simd, gradient, t_vals),
        )
    }

    fn gradient_painter_with_undefined<'a>(
        simd: S,
        gradient: &'a EncodedGradient,
        t_vals: &'a [f32],
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || GradientPainter::new(simd, gradient, t_vals),
        )
    }
    fn plain_nn_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || PlainNNImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }

    fn nn_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || NNImagePainter::new(simd, image, pixmap, start_x, start_y),
        )
    }

    fn medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::<S, 1>::new(simd, image, pixmap, start_x, start_y),
        )
    }

    fn plain_medium_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::<S, 1>::new(simd, image, pixmap, start_x, start_y),
        )
    }

    fn high_quality_image_painter<'a>(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter + 'a {
        simd.vectorize(
            #[inline(always)]
            || FilteredImagePainter::<S, 2>::new(simd, image, pixmap, start_x, start_y),
        )
    }

    fn blurred_rounded_rectangle_painter(
        simd: S,
        rect: &EncodedBlurredRoundedRectangle,
        start_x: f64,
        start_y: f64,
    ) -> impl Painter {
        simd.vectorize(
            #[inline(always)]
            || BlurredRoundedRectFiller::new(simd, rect, start_x, start_y),
        )
    }
    fn apply_mask(simd: S, dest: &mut [Self::Numeric], src: impl Iterator<Item = Self::NumericVec>);

    fn apply_painter<'a>(simd: S, dest: &mut [Self::Numeric], painter: impl Painter + 'a);

    fn apply_tint(simd: S, dest: &mut [Self::Numeric], tint: &Tint);

    fn alpha_composite_solid(
        simd: S,
        target: &mut [Self::Numeric],
        src: [Self::Numeric; 4],
        alphas: Option<&[u8]>,
    );

    fn alpha_composite_buffer(
        simd: S,
        dest: &mut [Self::Numeric],
        src: &[Self::Numeric],
        alphas: Option<&[u8]>,
    );

    fn blend(
        simd: S,
        dest: &mut [Self::Numeric],
        start_x: u16,
        start_y: u16,
        src: impl Iterator<Item = Self::Composite>,
        blend_mode: BlendMode,
        alphas: Option<&[u8]>,
        mask: Option<&Mask>,
    );

    fn fill_solid(simd: S, dest: &mut [Self::Numeric], color: PremulColor, alphas: Option<&[u8]>) {
        let color = Self::extract_color(color);

        if color[3] == Self::Numeric::ONE && alphas.is_none() {
            Self::copy_solid(simd, dest, color);
        } else {
            Self::alpha_composite_solid(simd, dest, color, alphas);
        }
    }
}

pub(crate) fn rasterize_region<S: Simd, T: FineKernel<S>>(
    fine: &mut Fine<S, T>,
    depth: &mut DepthBuffer,
    region: &mut Region<'_>,
    bucketer: &CommandBucketer,
    resources: FineResources<'_>,
    mut target_init: TargetInit<PremulColor>,
    root_is_blend_target: bool,
) {
    let scene_y = region.row_idx as u16 * Tile::HEIGHT;
    let row = &bucketer.rows()[region.row_idx];
    let span = Span::new(0, region.width()).tile_aligned();

    fine.set_row_y(scene_y);
    depth.clear();

    let has_cmds = !row.depth_cmds.is_empty() || !row.render_cmds.is_empty();
    let isolate_bg = root_is_blend_target
        && has_cmds
        && match target_init {
            TargetInit::SrcOver => true,
            TargetInit::Clear(color) => !color.is_transparent(),
        };
    if isolate_bg {
        fine.init_uncovered_range(span, region, target_init, depth);
        fine.push_buf(None);

        target_init = TargetInit::Clear(PremulColor::from_alpha_color(AlphaColor::TRANSPARENT));
    }

    for &cmd in row.depth_cmds.iter().rev() {
        let attrs = &bucketer.paint_fill_attrs[cmd.attrs_idx as usize];

        depth.for_each_unset_run_and_write(cmd.bucket_range(), attrs.draw_id, |bucket_range| {
            let span = bucket_range.span();
            fine.paint_fill(span, attrs, resources, None);
        });
    }

    fine.init_uncovered_range(span, region, target_init, depth);

    for cmd in &row.render_cmds {
        fine.run_cmd(*cmd, bucketer, row, scene_y, resources, depth);
    }

    if isolate_bg {
        fine.layer_fill(scene_y, span, BlendMode::default(), 1.0, None, None);
        fine.pop_buf();
    }

    fine.pack(region);
}

#[derive(Debug)]
#[doc(hidden)]
pub struct Fine<S: Simd, T: FineKernel<S>> {
    simd: S,
    buffer_span: TileAlignedSpan,
    blend_buffers: Vec<Vec<T::Numeric>>,
    buffer_pool: VecPool<T::Numeric>,
    paint_buf: Vec<T::Numeric>,
    f32_buf: Vec<f32>,
    row_y: u16,
    origin: (u16, u16),
}

impl<S: Simd, T: FineKernel<S>> Fine<S, T> {
    #[doc(hidden)]
    pub fn new(simd: S, buffer_width: u16) -> Self {
        let scratch_len = usize::from(buffer_width) * TILE_HEIGHT_COMPONENTS;
        Self {
            simd,
            buffer_span: TileAlignedSpan::try_from(Span::new(0, buffer_width)).unwrap(),
            blend_buffers: vec![vec![T::Numeric::ZERO; scratch_len]],
            buffer_pool: VecPool::new(false),
            paint_buf: Vec::new(),
            f32_buf: Vec::new(),
            row_y: 0,
            origin: (0, 0),
        }
    }

    fn set_row_y(&mut self, row_y: u16) {
        self.row_y = row_y;
    }

    fn set_paint_offset(&mut self, paint_offset: (u16, u16)) {
        self.origin = paint_offset;
    }

    fn scratch_range(span: TileAlignedSpan) -> core::ops::Range<usize> {
        let start = usize::from(span.pixel_x()) * TILE_HEIGHT_COMPONENTS;
        let len = usize::from(span.pixel_width()) * TILE_HEIGHT_COMPONENTS;
        start..start + len
    }

    fn init_uncovered_range(
        &mut self,
        scratch_span: TileAlignedSpan,
        region: &mut Region<'_>,
        target_init: TargetInit<PremulColor>,
        depth: &DepthBuffer,
    ) {
        match target_init {
            TargetInit::SrcOver => {
                depth.for_each_unset_run(
                    scratch_span,
                    #[inline]
                    |span| {
                        if let Some(span) = span
                            .as_span()
                            .intersect(Span::new(0, region.width()))
                        {
                            let mut region = region.sub_span(span.pixel_x(), span.pixel_width());
                            self.unpack(span.pixel_x(), &mut region);
                        }
                    },
                );
            }
            TargetInit::Clear(color) if color.is_transparent() => {
                depth.for_each_unset_run(
                    scratch_span,
                    #[inline]
                    |span| {
                        let range = Self::scratch_range(span);
                        self.blend_buffers.last_mut().unwrap()[range].fill(T::Numeric::ZERO);
                    },
                );
            }
            TargetInit::Clear(color) => {
                let color = T::extract_color(color);
                depth.for_each_unset_run(
                    scratch_span,
                    #[inline]
                    |span| {
                        let range = Self::scratch_range(span);
                        T::copy_solid(
                            self.simd,
                            &mut self.blend_buffers.last_mut().unwrap()[range],
                            color,
                        );
                    },
                );
            }
        }
    }

    fn push_buf(&mut self, span: Option<TileAlignedSpan>) {
        let mut buf = self.buffer_pool.take();
        buf.resize(self.blend_buffers[0].len(), T::Numeric::ZERO);

        if let Some(span) = span.and_then(|span| span.intersect(self.buffer_span)) {
            buf[Self::scratch_range(span)].fill(T::Numeric::ZERO);
        }
        self.blend_buffers.push(buf);
    }

    fn pop_buf(&mut self) {
        let popped = self.blend_buffers.pop().unwrap();
        self.buffer_pool.submit(popped);
    }

    #[doc(hidden)]
    pub fn pack(&self, region: &mut Region<'_>) {
        let width = usize::from(region.width());
        let scratch = self.blend_buffers.last().unwrap();

        T::pack(self.simd, scratch, width, region);
    }

    #[doc(hidden)]
    pub fn unpack(&mut self, scratch_x_start: u16, region: &mut Region<'_>) {
        let scratch_x = usize::from(scratch_x_start);
        let width = usize::from(region.width());
        let scratch = self.blend_buffers.last_mut().unwrap();

        T::unpack(
            self.simd,
            region,
            width,
            &mut scratch[scratch_x * TILE_HEIGHT_COMPONENTS..],
        );
    }

    fn run_cmd(
        &mut self,
        cmd: RenderCmd,
        bucketer: &CommandBucketer,
        row: &RowState,
        row_y: u16,
        resources: FineResources<'_>,
        depth: &DepthBuffer,
    ) {
        match cmd {
            RenderCmd::PaintFill(cmd) => {
                let attrs = &bucketer.paint_fill_attrs[cmd.attrs_idx as usize];
                let alpha_buffer = resources.alpha_buffers[attrs.thread_idx as usize];

                let Some(span) = cmd.span.intersect(self.buffer_span) else {
                    return;
                };

                let paint_fill = |fine: &mut Self, span: TileAlignedSpan| {
                    let alphas = cmd.alpha_idx().map(|alpha_idx| {
                        let alpha_offset = alpha_idx as usize
                            + usize::from(span.pixel_x() - cmd.span.pixel_x())
                                * Tile::HEIGHT as usize;
                        &alpha_buffer[alpha_offset..]
                    });

                    fine.paint_fill(span, attrs, resources, alphas);
                };

                if !row.can_skip_depth(span, attrs.draw_id) {
                    depth.for_each_visible_run(span, attrs.draw_id, |span| {
                        paint_fill(self, span);
                    });
                } else {
                    paint_fill(self, span);
                }
            }
            RenderCmd::PushBuf(span) => self.push_buf(span),
            RenderCmd::PopBuf => self.pop_buf(),
            RenderCmd::LayerFill(cmd) => {
                let Some(span) = cmd.span.intersect(self.buffer_span) else {
                    return;
                };

                let attrs = &bucketer.layer_fill_attrs[cmd.attrs_idx as usize];
                let alpha_buffer = resources.alpha_buffers[attrs.thread_idx as usize];

                let layer_fill = |fine: &mut Self, span: TileAlignedSpan| {
                    let alphas = cmd.alpha_idx().map(|alpha_idx| {
                        let alpha_offset = alpha_idx as usize
                            + usize::from(span.pixel_x() - cmd.span.pixel_x())
                                * Tile::HEIGHT as usize;
                        &alpha_buffer[alpha_offset..]
                    });

                    fine.layer_fill(
                        row_y,
                        span,
                        attrs.blend_mode,
                        attrs.opacity,
                        attrs.mask.as_ref(),
                        alphas,
                    );
                };

                if !row.can_skip_depth(span, attrs.draw_id) {
                    depth.for_each_visible_run(span, attrs.draw_id, |span| {
                        layer_fill(self, span);
                    });
                } else {
                    layer_fill(self, span);
                }
            }
        }
    }

    fn opacity(&mut self, span: TileAlignedSpan, opacity: f32) {
        let target = self.blend_buffers.last_mut().unwrap();
        let target = &mut target[Self::scratch_range(span)];

        T::apply_mask(
            self.simd,
            target,
            iter::repeat(T::NumericVec::from_f32(
                self.simd,
                f32x16::splat(self.simd, opacity),
            )),
        );
    }

    fn mask(&mut self, row_y: u16, span: TileAlignedSpan, mask: &Mask) {
        let x = span.pixel_x();
        let width = span.pixel_width();
        let target = self.blend_buffers.last_mut().unwrap();
        let target = &mut target[Self::scratch_range(span)];
        let y = u32::from(row_y) + u32x4::from_slice(self.simd, &[0, 1, 2, 3]);
        let iter = (x..x.saturating_add(width)).map(|x| {
            let x_in_range = x < mask.width();

            macro_rules! sample {
                ($idx:expr) => {
                    if x_in_range && (y[$idx] as u16) < mask.height() {
                        mask.sample(x, y[$idx] as u16)
                    } else {
                        0
                    }
                };
            }

            let s1 = sample!(0);
            let s2 = sample!(1);
            let s3 = sample!(2);
            let s4 = sample!(3);

            let samples = u8x16::from_slice(
                self.simd,
                &[
                    s1, s1, s1, s1, s2, s2, s2, s2, s3, s3, s3, s3, s4, s4, s4, s4,
                ],
            );
            T::NumericVec::from_u8(self.simd, samples)
        });

        T::apply_mask(self.simd, target, iter);
    }

    fn layer_fill(
        &mut self,
        row_y: u16,
        span: TileAlignedSpan,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<&Mask>,
        alphas: Option<&[u8]>,
    ) {
        if opacity != 1.0 {
            self.opacity(span, opacity);
        }
        if let Some(mask) = mask {
            self.mask(row_y, span, mask);
        }

        let x = span.pixel_x();
        let (source, rest) = self.blend_buffers.split_last_mut().unwrap();
        let target = rest.last_mut().unwrap();
        let range = Self::scratch_range(span);
        let source = &mut source[range.clone()];
        let target = &mut target[range];

        if blend_mode == BlendMode::default() {
            T::alpha_composite_buffer(self.simd, target, source, alphas);
        } else {
            T::blend(
                self.simd,
                target,
                x,
                row_y,
                source
                    .chunks_exact(T::Composite::LENGTH)
                    .map(|s| T::Composite::from_slice(self.simd, s)),
                blend_mode,
                alphas,
                None,
            );
        }
    }

    #[doc(hidden)]
    pub fn paint_fill(
        &mut self,
        span: TileAlignedSpan,
        attrs: &PaintFillAttrs,
        resources: FineResources<'_>,
        alphas: Option<&[u8]>,
    ) {
        self.set_paint_offset(attrs.origin);
        match &attrs.paint {
            Paint::Solid(color) => {
                self.solid_fill(span, *color, attrs, alphas);
            }
            Paint::Indexed(index) => {
                self.indexed_fill(span, index.index(), attrs, resources, alphas);
            }
        }
    }

    fn solid_fill(
        &mut self,
        span: TileAlignedSpan,
        color: PremulColor,
        attrs: &PaintFillAttrs,
        alphas: Option<&[u8]>,
    ) {
        if attrs.blend_mode == BlendMode::default() && attrs.mask.is_none() {
            let scratch = self.blend_buffers.last_mut().unwrap();
            T::fill_solid(
                self.simd,
                &mut scratch[Self::scratch_range(span)],
                color,
                alphas,
            );
            return;
        }

        if span.pixel_width() == 0 {
            return;
        }

        let x = span.pixel_x();
        let color = T::extract_color(color);
        let simd = self.simd;
        let color = T::Composite::from_color(simd, color);
        let scratch = self.blend_buffers.last_mut().unwrap();
        T::blend(
            simd,
            &mut scratch[Self::scratch_range(span)],
            x,
            self.row_y,
            iter::repeat(color),
            attrs.blend_mode,
            alphas,
            attrs.mask.as_ref(),
        );
    }

    fn indexed_fill(
        &mut self,
        span: TileAlignedSpan,
        paint_index: usize,
        attrs: &PaintFillAttrs,
        resources: FineResources<'_>,
        alphas: Option<&[u8]>,
    ) {
        let x = span.pixel_x();
        let y = self.row_y;
        let sample_x = x.saturating_add(self.origin.0);
        let sample_y = y.saturating_add(self.origin.1);
        let width = span.pixel_width();
        let len = usize::from(width) * TILE_HEIGHT_COMPONENTS;
        if self.paint_buf.len() < len {
            self.paint_buf.resize(len, T::Numeric::ZERO);
        }

        let t_len = usize::from(width) * Tile::HEIGHT as usize;
        if self.f32_buf.len() < t_len {
            self.f32_buf.resize(t_len, 0.0);
        }

        let simd = self.simd;
        let width = usize::from(width);
        let start = usize::from(x) * TILE_HEIGHT_COMPONENTS;
        let dest = &mut self.blend_buffers.last_mut().unwrap()[start..start + len];
        let color_buf = &mut self.paint_buf[..len];
        let encoded_paint = resources
            .encoded_paints
            .get(paint_index)
            .unwrap_or_else(|| {
                &resources.filter_paints[paint_index - resources.encoded_paints.len()]
            });

        let sampler_x = f64::from(sample_x) + PIXEL_CENTER_OFFSET;
        let sampler_y = f64::from(sample_y) + PIXEL_CENTER_OFFSET;
        let default_blend = attrs.blend_mode == BlendMode::default();

        macro_rules! fill_complex_paint {
            ($may_have_transparency:expr, $filler:expr) => {
                fill_complex_paint!($may_have_transparency, $filler, None::<&Tint>)
            };
            ($may_have_transparency:expr, $filler:expr, $tint:expr) => {
                if $may_have_transparency
                    || alphas.is_some()
                    || !default_blend
                    || attrs.mask.is_some()
                {
                    T::apply_painter(simd, color_buf, $filler);
                    if let Some(tint) = $tint {
                        T::apply_tint(simd, color_buf, tint);
                    }

                    if default_blend && attrs.mask.is_none() {
                        T::alpha_composite_buffer(simd, dest, color_buf, alphas);
                    } else {
                        T::blend(
                            simd,
                            dest,
                            x,
                            y,
                            color_buf
                                .chunks_exact(T::Composite::LENGTH)
                                .map(|s| T::Composite::from_slice(simd, s)),
                            attrs.blend_mode,
                            alphas,
                            attrs.mask.as_ref(),
                        );
                    }
                } else {
                    T::apply_painter(simd, dest, $filler);
                    if let Some(tint) = $tint {
                        T::apply_tint(simd, dest, tint);
                    }
                }
            };
        }

        match encoded_paint {
            EncodedPaint::BlurredRoundedRect(rect) => {
                fill_complex_paint!(
                    true,
                    T::blurred_rounded_rectangle_painter(simd, rect, sampler_x, sampler_y)
                );
            }
            EncodedPaint::Gradient(gradient) => {
                let t_vals = &mut self.f32_buf[..width * Tile::HEIGHT as usize];

                match &gradient.kind {
                    EncodedKind::Linear(kind) => {
                        calculate_t_vals(
                            simd,
                            SimdLinearKind::new(simd, *kind),
                            t_vals,
                            gradient,
                            sampler_x,
                            sampler_y,
                        );
                        fill_complex_paint!(
                            gradient.may_have_transparency,
                            T::gradient_painter(simd, gradient, t_vals)
                        );
                    }
                    EncodedKind::Sweep(kind) => {
                        calculate_t_vals(
                            simd,
                            SimdSweepKind::new(simd, kind),
                            t_vals,
                            gradient,
                            sampler_x,
                            sampler_y,
                        );
                        fill_complex_paint!(
                            gradient.may_have_transparency,
                            T::gradient_painter(simd, gradient, t_vals)
                        );
                    }
                    EncodedKind::Radial(kind) => {
                        calculate_t_vals(
                            simd,
                            SimdRadialKind::new(simd, kind),
                            t_vals,
                            gradient,
                            sampler_x,
                            sampler_y,
                        );

                        if kind.has_undefined() {
                            fill_complex_paint!(
                                gradient.may_have_transparency,
                                T::gradient_painter_with_undefined(simd, gradient, t_vals)
                            );
                        } else {
                            fill_complex_paint!(
                                gradient.may_have_transparency,
                                T::gradient_painter(simd, gradient, t_vals)
                            );
                        }
                    }
                }
            }
            EncodedPaint::Image(image) => {
                let pixmap = match &image.source {
                    ImageSource::Pixmap(pixmap) => pixmap.clone(),
                    ImageSource::OpaqueId { id, .. } => resources
                        .image_resolver
                        .resolve(*id)
                        .unwrap_or_else(|| panic!("Image {:?} not found in registry", id)),
                    ImageSource::ExternalTexture { .. } => {
                        unimplemented!("External textures are not supported by `vello_cpu`")
                    }
                };
                let tint = image.tint.as_ref();

                match (image.has_skew(), image.nearest_neighbor()) {
                    (false, false) => {
                        if image.sampler.quality == ImageQuality::Medium {
                            fill_complex_paint!(
                                image.may_have_transparency,
                                T::plain_medium_quality_image_painter(
                                    simd, image, &pixmap, sampler_x, sampler_y
                                ),
                                tint
                            );
                        } else {
                            fill_complex_paint!(
                                image.may_have_transparency,
                                T::high_quality_image_painter(
                                    simd, image, &pixmap, sampler_x, sampler_y
                                ),
                                tint
                            );
                        }
                    }
                    (true, false) => {
                        if image.sampler.quality == ImageQuality::Medium {
                            fill_complex_paint!(
                                image.may_have_transparency,
                                T::medium_quality_image_painter(
                                    simd, image, &pixmap, sampler_x, sampler_y
                                ),
                                tint
                            );
                        } else {
                            fill_complex_paint!(
                                image.may_have_transparency,
                                T::high_quality_image_painter(
                                    simd, image, &pixmap, sampler_x, sampler_y
                                ),
                                tint
                            );
                        }
                    }
                    (false, true) => {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::plain_nn_image_painter(simd, image, &pixmap, sampler_x, sampler_y),
                            tint
                        );
                    }
                    (true, true) => {
                        fill_complex_paint!(
                            image.may_have_transparency,
                            T::nn_image_painter(simd, image, &pixmap, sampler_x, sampler_y),
                            tint
                        );
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct FineResources<'a> {
    pub alpha_buffers: &'a [&'a [u8]],
    pub encoded_paints: &'a [EncodedPaint],
    pub filter_paints: &'a [EncodedPaint],
    pub image_resolver: &'a dyn ImageResolver,
}

impl Debug for FineResources<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FineResources")
            .field("alpha_buffers", &self.alpha_buffers.len())
            .field("encoded_paints", &self.encoded_paints.len())
            .field("filter_paints", &self.filter_paints.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct FineRenderParams {
    pub(crate) scene_size: (u16, u16),
    pub(crate) target_offset: (u16, u16),
}

pub trait Painter: Sized {
    fn paint_u8(self, buf: &mut [u8]);

    fn paint_f32(self, buf: &mut [f32]);
}

pub trait PosExt<S: Simd> {
    fn splat_pos(simd: S, pos: f32, x_advance: f32, y_advance: f32) -> Self;
}

impl<S: Simd> PosExt<S> for f32x4<S> {
    #[inline(always)]
    fn splat_pos(simd: S, pos: f32, _: f32, y_advance: f32) -> Self {
        let columns: [f32; Tile::HEIGHT as usize] = [0.0, 1.0, 2.0, 3.0];
        let column_mask: Self = columns.simd_into(simd);

        column_mask.mul_add(Self::splat(simd, y_advance), Self::splat(simd, pos))
    }
}

impl<S: Simd> PosExt<S> for f32x8<S> {
    #[inline(always)]
    fn splat_pos(simd: S, pos: f32, x_advance: f32, y_advance: f32) -> Self {
        simd.combine_f32x4(
            f32x4::splat_pos(simd, pos, x_advance, y_advance),
            f32x4::splat_pos(simd, pos + x_advance, x_advance, y_advance),
        )
    }
}

pub(crate) struct ShaderResultF32<S: Simd> {
    pub(crate) r: f32x8<S>,
    pub(crate) g: f32x8<S>,
    pub(crate) b: f32x8<S>,
    pub(crate) a: f32x8<S>,
}

impl<S: Simd> ShaderResultF32<S> {
    #[inline(always)]
    pub(crate) fn get(&self) -> [[f32x4<S>; 4]; 2] {
        let (r_1, r_2) = self.r.simd.split_f32x8(self.r);
        let (g_1, g_2) = self.g.simd.split_f32x8(self.g);
        let (b_1, b_2) = self.b.simd.split_f32x8(self.b);
        let (a_1, a_2) = self.a.simd.split_f32x8(self.a);

        [[r_1, g_1, b_1, a_1], [r_2, g_2, b_2, a_2]]
    }
}

mod macros {
    macro_rules! f32x16_painter {
        ($($type_path:tt)+) => {
            impl<S: Simd> crate::vello_cpu::fine::Painter for $($type_path)+ {
                fn paint_u8(mut self, buf: &mut [u8]) {
                    use vello_common::fearless_simd::*;
                    use crate::vello_cpu::fine::NumericVec;

                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            let converted = u8x16::<S>::from_f32(next.simd, next);
                            converted.store_slice(chunk);
                        }
                    })
                }

                fn paint_f32(mut self, buf: &mut [f32]) {
                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            next.store_slice(chunk);
                        }
                    })
                }
            }
        };
    }

    macro_rules! u8x16_painter {
        ($($type_path:tt)+) => {
            impl<S: Simd> crate::vello_cpu::fine::Painter for $($type_path)+ {
                fn paint_u8(mut self, buf: &mut [u8]) {
                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            next.store_slice(chunk);
                        }
                    })
                }

                fn paint_f32(mut self, buf: &mut [f32]) {
                    use vello_common::fearless_simd::*;
                    use crate::vello_cpu::fine::NumericVec;

                    self.simd.vectorize(#[inline(always)] || {
                        for chunk in buf.chunks_exact_mut(16) {
                            let next = self.next().unwrap();
                            let converted = f32x16::<S>::from_u8(next.simd, next);
                            converted.store_slice(chunk);
                        }
                    })
                }
            }
        };
    }

    pub(crate) use f32x16_painter;
    pub(crate) use u8x16_painter;
}
