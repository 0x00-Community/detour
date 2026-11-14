use crate::vello_cpu::RenderMode;
use crate::vello_cpu::dispatch::Dispatcher;
#[cfg(any())]
use crate::vello_cpu::dispatch::multi_threaded::MultiThreadedDispatcher;
#[cfg(all())]
use crate::vello_cpu::text::{GlyphAtlasResources, GlyphRunBuilder};
#[cfg(all())]
use glifo::GlyphPrepCache;

use crate::vello_cpu::dispatch::single_threaded::SingleThreadedDispatcher;
use crate::vello_cpu::kurbo::{PathEl, Point};
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;
use vello_common::blurred_rounded_rect::BlurredRoundedRectangle;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::encode::{EncodeExt, EncodedPaint, invert_paint_transform};
use vello_common::fearless_simd::Level;
use vello_common::filter::FilterData;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{ImageId, ImageResolver, Paint, PaintType, Tint};
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Fill};
use vello_common::pixmap::{Pixmap, PixmapMut};
use vello_common::render_state::RenderState;
use vello_common::transforms::{RootTransforms, Transforms};
use vello_common::util::into_fast_path_rect;

#[cfg(all())]
pub(crate) const DEFAULT_GLYPH_ATLAS_SIZE: u16 = 4096;

pub(crate) const ATLAS_IMAGE_ID_BASE: u32 = u32::MAX / 2;

#[derive(Debug, Default)]
pub struct Resources {
    pub(crate) image_registry: ImageRegistry,
    #[cfg(all())]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(all())]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn before_render(&mut self, render_mode: RenderMode) {
        #[cfg(all())]
        self.prepare_glyph_cache(render_mode);

        #[cfg(not(all()))]
        let _ = render_mode;
    }

    pub(crate) fn after_render(&mut self) {
        #[cfg(all())]
        self.maintain_glyph_cache();
    }
}

pub type TargetInit = vello_common::TargetInit<AlphaColor<Srgb>>;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum PixelFormat {
    #[default]
    Rgba8,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RasterizerSettings {
    pub render_mode: RenderMode,
    pub target_init: TargetInit,
    pub pixel_format: PixelFormat,
    pub offset: (u16, u16),
}

impl Default for RasterizerSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::OptimizeSpeed,
            target_init: TargetInit::Clear(AlphaColor::TRANSPARENT),
            pixel_format: PixelFormat::Rgba8,
            offset: (0, 0),
        }
    }
}

#[derive(Debug)]
pub struct RenderContext {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) state: RenderState,
    root_transforms: RootTransforms,
    pub(crate) mask: Option<Mask>,
    pub(crate) temp_path: BezPath,
    pub(crate) aliasing_threshold: Option<u8>,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    pub(crate) filter: Option<Filter>,
    #[cfg_attr(
        not(all()),
        allow(dead_code, reason = "used when the `text` feature is enabled")
    )]
    pub(crate) render_settings: RenderSettings,
    dispatcher: Box<dyn Dispatcher>,
}

#[derive(Copy, Clone, Debug)]
pub struct RenderSettings {
    pub level: Level,
    pub num_threads: u16,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            level: Level::try_detect().unwrap_or(Level::baseline()),
            #[cfg(any())]
            num_threads: (std::thread::available_parallelism()
                .unwrap()
                .get()
                .saturating_sub(1) as u16)
                .min(8),
            #[cfg(not(any()))]
            num_threads: 0,
        }
    }
}

impl RenderContext {
    pub fn new(width: u16, height: u16) -> Self {
        Self::new_with(width, height, RenderSettings::default())
    }

    pub fn new_with(width: u16, height: u16, settings: RenderSettings) -> Self {
        #[cfg(any())]
        let dispatcher: Box<dyn Dispatcher> = if settings.num_threads == 0 {
            Box::new(SingleThreadedDispatcher::new(width, height, settings.level))
        } else {
            Box::new(MultiThreadedDispatcher::new(
                width,
                height,
                settings.num_threads,
                settings.level,
            ))
        };

        #[cfg(not(any()))]
        let dispatcher: Box<dyn Dispatcher> =
            { Box::new(SingleThreadedDispatcher::new(width, height, settings.level)) };

        let encoded_paints = vec![];
        let temp_path = BezPath::new();
        let aliasing_threshold = None;

        Self {
            width,
            height,
            dispatcher,
            state: RenderState::default(),
            root_transforms: RootTransforms::default(),
            aliasing_threshold,
            render_settings: settings,
            mask: None,
            temp_path,
            encoded_paints,
            filter: None,
        }
    }

    fn transforms(&self) -> &Transforms {
        &self.state.transforms
    }

    fn transforms_mut(&mut self) -> &mut Transforms {
        &mut self.state.transforms
    }

    fn encode_current_paint(&mut self) -> Option<Paint> {
        match self.state.paint.clone() {
            PaintType::Solid(s) => Some(s.into()),
            PaintType::Gradient(g) => {
                let transform = self
                    .root_transforms
                    .effective_paint_transform(self.transforms());
                g.encode_into(&mut self.encoded_paints, transform, None)
            }
            PaintType::Image(i) => {
                let transform = self
                    .root_transforms
                    .effective_paint_transform(self.transforms());
                i.encode_into(&mut self.encoded_paints, transform, self.state.tint)
            }
        }
    }

    fn paint_transform_has_area(&self) -> bool {
        let transform = self
            .root_transforms
            .effective_paint_transform(self.transforms());
        invert_paint_transform(transform).is_some()
    }

    fn paint_has_area(&self) -> bool {
        matches!(self.state.paint, PaintType::Solid(_)) || self.paint_transform_has_area()
    }

    pub fn fill_path(&mut self, path: &BezPath) {
        if !self.paint_has_area() {
            return;
        }

        self.with_optional_filter(|ctx| {
            let Some(paint) = ctx.encode_current_paint() else {
                return;
            };
            let transform = ctx
                .root_transforms
                .effective_path_transform(ctx.transforms());
            ctx.dispatcher.fill_path(
                path,
                ctx.state.fill_rule,
                transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
            );
        });
    }

    pub fn stroke_path(&mut self, path: &BezPath) {
        if !self.paint_has_area() {
            return;
        }

        self.with_optional_filter(|ctx| {
            let Some(paint) = ctx.encode_current_paint() else {
                return;
            };
            let transform = ctx
                .root_transforms
                .effective_path_transform(ctx.transforms());
            ctx.dispatcher.stroke_path(
                path,
                &ctx.state.stroke,
                transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
            );
        });
    }

    pub fn fill_rect(&mut self, rect: &Rect) {
        if !self.paint_has_area() {
            return;
        }

        self.with_optional_filter(|ctx| {
            let Some(paint) = ctx.encode_current_paint() else {
                return;
            };
            let transform = ctx
                .root_transforms
                .effective_path_transform(ctx.transforms());

            if let Some(transformed_rect) =
                into_fast_path_rect(*rect, &transform, ctx.aliasing_threshold)
            {
                ctx.dispatcher.fill_rect_fast(
                    &transformed_rect,
                    paint,
                    ctx.state.blend_mode,
                    ctx.mask.clone(),
                );
            } else {
                ctx.rect_to_temp_path(rect);
                ctx.dispatcher.fill_path(
                    &ctx.temp_path,
                    ctx.state.fill_rule,
                    transform,
                    paint,
                    ctx.state.blend_mode,
                    ctx.aliasing_threshold,
                    ctx.mask.clone(),
                );
            }
        });
    }

    pub fn stroke_rect(&mut self, rect: &Rect) {
        if !self.paint_has_area() {
            return;
        }

        self.with_optional_filter(|ctx| {
            ctx.rect_to_temp_path(rect);
            let Some(paint) = ctx.encode_current_paint() else {
                return;
            };
            let transform = ctx
                .root_transforms
                .effective_path_transform(ctx.transforms());
            ctx.dispatcher.stroke_path(
                &ctx.temp_path,
                &ctx.state.stroke,
                transform,
                paint,
                ctx.state.blend_mode,
                ctx.aliasing_threshold,
                ctx.mask.clone(),
            );
        });
    }

    fn rect_to_temp_path(&mut self, rect: &Rect) {
        self.temp_path.truncate(0);
        self.temp_path
            .push(PathEl::MoveTo(Point::new(rect.x0, rect.y0)));
        self.temp_path
            .push(PathEl::LineTo(Point::new(rect.x1, rect.y0)));
        self.temp_path
            .push(PathEl::LineTo(Point::new(rect.x1, rect.y1)));
        self.temp_path
            .push(PathEl::LineTo(Point::new(rect.x0, rect.y1)));
        self.temp_path.push(PathEl::ClosePath);
    }

    pub fn fill_blurred_rounded_rect(
        &mut self,
        rect: &Rect,
        radius: f32,
        std_dev: f32,
        invert: bool,
    ) {
        let rect = rect.abs();
        let color = match self.state.paint {
            PaintType::Solid(s) => s,
            _ => BLACK,
        };

        let blurred_rect = BlurredRoundedRectangle {
            rect,
            color,
            radius,
            std_dev,
            invert,
        };

        let kernel_size = 2.5 * std_dev;
        let inflated_rect = rect.inflate(f64::from(kernel_size), f64::from(kernel_size));
        let transform = self
            .root_transforms
            .effective_path_transform(self.transforms());
        let paint_transform = self
            .root_transforms
            .effective_paint_transform(self.transforms());

        let Some(paint) = blurred_rect.encode_into(&mut self.encoded_paints, paint_transform, None)
        else {
            return;
        };

        self.rect_to_temp_path(&inflated_rect);
        self.dispatcher.fill_path(
            &self.temp_path,
            Fill::NonZero,
            transform,
            paint,
            self.state.blend_mode,
            self.aliasing_threshold,
            self.mask.clone(),
        );
    }

    #[cfg(all())]
    pub fn glyph_run<'a>(
        &'a mut self,
        resources: &'a mut Resources,
        font: &crate::vello_cpu::peniko::FontData,
    ) -> GlyphRunBuilder<'a> {
        glifo::GlyphRunBuilder::new(
            font.clone(),
            self.transforms().scene_transform(),
            *self.transforms().paint_transform(),
            crate::vello_cpu::text::CpuGlyphRunBackend {
                ctx: self,
                resources,
                atlas_cache_enabled: false,
            },
        )
    }

    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<f32>,
        mask: Option<Mask>,
        filter: Option<Filter>,
    ) {
        let mask = mask.and_then(|m| {
            if m.width() != self.width || m.height() != self.height {
                None
            } else {
                Some(m)
            }
        });

        let blend_mode = blend_mode.unwrap_or_default();
        let opacity = opacity.unwrap_or(1.0).clamp(0.0, 1.0);
        let layer_transform = self
            .root_transforms
            .effective_path_transform(self.transforms());
        let filter_data = filter.map(|filter| FilterData::new(filter, layer_transform));

        let relative_transform = filter_data.as_ref().map_or(Affine::IDENTITY, |data| {
            let (shift_x, shift_y) = data.source_shift();
            Affine::translate((f64::from(shift_x), f64::from(shift_y)))
        });
        self.root_transforms.push_root(relative_transform);

        self.dispatcher.push_layer(
            clip_path,
            self.state.fill_rule,
            layer_transform,
            blend_mode,
            opacity,
            self.aliasing_threshold,
            mask,
            filter_data,
        );
    }

    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None, None);
    }

    pub fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(None, Some(blend_mode), None, None, None);
    }

    pub fn push_opacity_layer(&mut self, opacity: f32) {
        self.push_layer(None, None, Some(opacity), None, None);
    }

    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask), None);
    }

    pub fn push_filter_layer(&mut self, filter: Filter) {
        self.push_layer(None, None, None, None, Some(filter));
    }

    pub fn set_aliasing_threshold(&mut self, aliasing_threshold: Option<u8>) {
        self.aliasing_threshold = aliasing_threshold;
    }

    pub fn pop_layer(&mut self) {
        self.dispatcher.pop_layer();
        self.root_transforms.pop_root();
    }

    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.state.stroke = stroke;
    }

    pub fn stroke(&self) -> &Stroke {
        &self.state.stroke
    }

    #[cfg(all())]
    pub(crate) fn stroke_mut(&mut self) -> &mut Stroke {
        &mut self.state.stroke
    }

    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.state.paint = paint.into();
    }

    pub fn paint(&self) -> &PaintType {
        &self.state.paint
    }

    pub fn set_tint(&mut self, tint: Option<Tint>) {
        self.state.tint = tint;
    }

    pub fn reset_tint(&mut self) {
        self.state.tint = None;
    }

    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.state.blend_mode = blend_mode;
    }

    pub fn blend_mode(&self) -> BlendMode {
        self.state.blend_mode
    }

    pub fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.transforms_mut().set_paint_transform(paint_transform);
    }

    pub fn paint_transform(&self) -> &Affine {
        self.transforms().paint_transform()
    }

    pub fn reset_paint_transform(&mut self) {
        self.transforms_mut().reset_paint_transform();
    }

    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.state.fill_rule = fill_rule;
    }

    pub fn set_mask(&mut self, mask: Mask) {
        self.mask = Some(mask);
    }

    pub fn reset_mask(&mut self) {
        self.mask = None;
    }

    pub fn fill_rule(&self) -> &Fill {
        &self.state.fill_rule
    }

    pub fn set_transform(&mut self, transform: Affine) {
        self.transforms_mut().set_transform(transform);
    }

    pub fn transform(&self) -> &Affine {
        self.transforms().transform()
    }

    pub fn reset_transform(&mut self) {
        self.transforms_mut().reset_transform();
    }

    pub fn set_filter_effect(&mut self, filter: Filter) {
        self.filter = Some(filter);
    }

    pub fn reset_filter_effect(&mut self) {
        self.filter = None;
    }

    pub fn reset_and_resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;

        self.reset();
    }

    pub fn reset(&mut self) {
        self.dispatcher.reset(self.width, self.height);
        self.encoded_paints.clear();
        self.mask = None;
        self.root_transforms.reset();
        self.state.reset();
        self.aliasing_threshold = None;
    }

    pub fn push_clip_path(&mut self, path: &BezPath) {
        let transform = self.transforms().clip_path_transform();
        self.dispatcher.push_clip_path(
            path,
            self.state.fill_rule,
            transform,
            self.aliasing_threshold,
        );
    }

    pub fn push_clip_rect(&mut self, rect: &Rect) {
        let transform = self.transforms().clip_path_transform();

        if let Some(rect) = into_fast_path_rect(*rect, &transform, self.aliasing_threshold) {
            self.dispatcher.push_clip_rect(&rect);
        } else {
            self.rect_to_temp_path(rect);
            self.dispatcher.push_clip_path(
                &self.temp_path,
                self.state.fill_rule,
                transform,
                self.aliasing_threshold,
            );
        }
    }

    pub fn pop_clip(&mut self) {
        self.dispatcher.pop_clip();
    }

    pub fn flush(&mut self) {
        self.dispatcher.flush();
    }

    pub fn render<'a>(&self, target: impl Into<PixmapMut<'a>>, resources: &mut Resources) {
        self.render_with(target, resources, RasterizerSettings::default());
    }

    pub fn render_with<'a>(
        &self,
        target: impl Into<PixmapMut<'a>>,
        resources: &mut Resources,
        settings: RasterizerSettings,
    ) {
        assert!(
            !self.dispatcher.has_layers(),
            "some layers haven't been popped yet"
        );

        resources.before_render(settings.render_mode);
        let mut target = target.into();
        let target_fully_covered = settings.offset == (0, 0)
            && self.width >= target.width()
            && self.height >= target.height();
        if let TargetInit::Clear(color) = settings.target_init
            && !target_fully_covered
        {
            let color = color.premultiply().to_rgba8().to_u32();
            let data = target.data_mut();
            if color == 0 {
                data.fill(0);
            } else {
                let color = color.to_ne_bytes();
                for pixel in data.chunks_exact_mut(4) {
                    pixel.copy_from_slice(&color);
                }
            }
        }

        match settings.target_init {
            TargetInit::Clear(color) => {
                target.set_may_have_transparency(color.components[3] != 1.0);
            }
            TargetInit::SrcOver => {}
        }

        self.dispatcher.rasterize(
            target,
            self.width,
            self.height,
            settings,
            &self.encoded_paints,
            &resources.image_registry,
        );
        resources.after_render();
    }

    pub fn coarse(&self, resources: &mut Resources) -> Option<crate::vello_cpu::detour::Coarse<'_>> {
        assert!(!self.dispatcher.has_layers(), "some layers haven't been popped yet");
        resources.before_render(RenderMode::OptimizeSpeed);
        let coarse = self.dispatcher.coarse((self.width, self.height), &self.encoded_paints, &resources.image_registry);
        resources.after_render();
        coarse
    }

    pub fn new_recording(width: u16, height: u16) -> Self {
        let mut ctx = Self::new(width, height);
        ctx.dispatcher = Box::new(crate::vello_cpu::dispatch::record::RecordingDispatcher::new(width, height));
        ctx
    }

    pub fn recording(&self) -> Option<(&crate::vello_cpu::detour::Recording, &[EncodedPaint])> {
        assert!(!self.dispatcher.has_layers(), "some layers haven't been popped yet");
        self.dispatcher.recording().map(|r| (r, self.encoded_paints.as_slice()))
    }

    pub fn coarse_layered(&self, resources: &mut Resources) -> Option<crate::vello_cpu::detour::Layered<'_>> {
        assert!(!self.dispatcher.has_layers(), "some layers haven't been popped yet");
        resources.before_render(RenderMode::OptimizeSpeed);
        let layered = self.dispatcher.coarse_layered((self.width, self.height), &self.encoded_paints);
        resources.after_render();
        layered
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    pub fn render_settings(&self) -> &RenderSettings {
        &self.render_settings
    }

    fn with_optional_filter<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self),
    {
        if let Some(filter) = self.filter.clone() {
            self.push_filter_layer(filter);
            f(self);
            self.pop_layer();
        } else {
            f(self);
        }
    }

    pub fn take_current_state(&mut self) -> RenderState {
        core::mem::take(&mut self.state)
    }

    pub fn save_current_state(&mut self) -> RenderState {
        self.state.clone()
    }

    pub fn restore_state(&mut self, state: RenderState) {
        self.state = state;
    }

    pub fn is_multi_threaded(&self) -> bool {
        self.dispatcher.is_multi_threaded()
    }
}

impl Resources {
    pub fn register_image(&mut self, pixmap: Arc<Pixmap>) -> ImageId {
        self.image_registry.register(pixmap)
    }

    pub fn destroy_image(&mut self, id: ImageId) -> bool {
        self.image_registry.destroy(id)
    }

    pub fn resolve_image(&self, id: ImageId) -> Option<Arc<Pixmap>> {
        self.image_registry.resolve(id)
    }

    pub fn clear_images(&mut self) {
        self.image_registry.clear();
    }
}

#[derive(Debug, Default)]
pub(crate) struct ImageRegistry {
    images: HashMap<u32, Arc<Pixmap>>,
    next_id: u32,
}

impl ImageRegistry {
    fn register(&mut self, pixmap: Arc<Pixmap>) -> ImageId {
        let id = self.next_id;
        assert!(
            id < ATLAS_IMAGE_ID_BASE,
            "image registry exhausted non-atlas image IDs"
        );

        self.next_id += 1;
        self.images.insert(id, pixmap);
        ImageId::new(id)
    }

    #[cfg(all())]
    pub(crate) fn register_atlas_page(&mut self, page_index: u32, pixmap: Arc<Pixmap>) {
        self.images.insert(
            ImageId::new(ATLAS_IMAGE_ID_BASE + page_index).as_u32(),
            pixmap,
        );
    }

    pub(crate) fn destroy(&mut self, id: ImageId) -> bool {
        self.images.remove(&id.as_u32()).is_some()
    }

    #[cfg(all())]
    pub(crate) fn destroy_atlas_page(&mut self, page_index: u32) -> bool {
        self.destroy(ImageId::new(ATLAS_IMAGE_ID_BASE + page_index))
    }

    fn resolve(&self, id: ImageId) -> Option<Arc<Pixmap>> {
        self.images.get(&id.as_u32()).cloned()
    }

    fn clear(&mut self) {
        self.images.clear();
        self.next_id = 0;
    }
}

impl ImageResolver for ImageRegistry {
    fn resolve(&self, id: ImageId) -> Option<Arc<Pixmap>> {
        self.images.get(&id.as_u32()).cloned()
    }
}

#[cfg(any())]
mod tests {
    #[cfg(all())]
    use crate::vello_cpu::peniko::{Blob, FontData};
    use crate::vello_cpu::{RasterizerSettings, RenderContext, Resources, TargetInit};
    #[cfg(all())]
    use alloc::sync::Arc;
    use alloc::vec;
    #[cfg(all())]
    use glifo::Glyph;
    use vello_common::color::PremulRgba8;
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::kurbo::{Rect, Shape};
    use vello_common::peniko::ImageAlphaType;
    use vello_common::pixmap::{PixelMetadata, Pixmap, PixmapMut};
    use vello_common::tile::Tile;

    const GRAY: PremulRgba8 = PremulRgba8 {
        r: 9,
        g: 10,
        b: 11,
        a: 255,
    };

    fn red_pixel() -> PremulRgba8 {
        RED.premultiply().to_rgba8()
    }

    fn blue_pixel() -> PremulRgba8 {
        BLUE.premultiply().to_rgba8()
    }

    fn transparent_pixel() -> PremulRgba8 {
        PremulRgba8::from_u32(0)
    }

    fn solid_pixmap(width: u16, height: u16, color: PremulRgba8) -> Pixmap {
        Pixmap::from_parts(
            bytemuck::cast_vec(vec![color; usize::from(width) * usize::from(height)]),
            width,
            height,
            PixelMetadata::new(ImageAlphaType::AlphaPremultiplied, color.a != 255),
        )
    }

    fn red_rect_context(width: u16, height: u16, rect: Rect) -> RenderContext {
        let mut ctx = RenderContext::new(width, height);
        ctx.set_paint(RED);
        ctx.fill_rect(&rect);
        ctx.flush();
        ctx
    }

    #[test]
    fn clip_overflow() {
        let mut ctx = RenderContext::new(100, 100);

        for _ in 0..(usize::from(u16::MAX) + 1).div_ceil(usize::from(Tile::HEIGHT * Tile::WIDTH)) {
            ctx.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
        }

        ctx.push_clip_layer(&Rect::new(20.0, 20.0, 180.0, 180.0).to_path(0.1));
        ctx.pop_layer();
        ctx.flush();
    }

    #[test]
    fn render_with_offset_clears_pixels_outside_scene() {
        let ctx = red_rect_context(2, 2, Rect::new(0.0, 0.0, 2.0, 2.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(4, 3, GRAY);

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                offset: (1, 1),
                ..Default::default()
            },
        );

        for y in 0..3 {
            for x in 0..4 {
                let expected = if (1..=2).contains(&x) && (1..=2).contains(&y) {
                    red_pixel()
                } else {
                    transparent_pixel()
                };

                assert_eq!(pixmap.sample(x, y), expected, "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn render_clips_scene_to_target_bounds() {
        let ctx = red_rect_context(3, 3, Rect::new(0.0, 0.0, 3.0, 3.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(4, 4, GRAY);

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                offset: (2, 1),
                ..Default::default()
            },
        );

        for y in 0..4 {
            for x in 0..4 {
                let expected = if (2..=3).contains(&x) && (1..=3).contains(&y) {
                    red_pixel()
                } else {
                    transparent_pixel()
                };
                assert_eq!(pixmap.sample(x, y), expected, "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn render_into_padded_pixmap() {
        let ctx = red_rect_context(2, 2, Rect::new(0.0, 0.0, 2.0, 2.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(4, 2, GRAY);

        ctx.render(&mut pixmap, &mut resources);

        for y in 0..2 {
            for x in 0..4 {
                let expected = if x < 2 {
                    red_pixel()
                } else {
                    transparent_pixel()
                };
                assert_eq!(pixmap.sample(x, y), expected, "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn reset_and_resize_updates_scene_size() {
        let mut ctx = RenderContext::new(8, 4);
        let mut resources = Resources::new();
        let mut pixmap = Pixmap::new(4, 8);

        ctx.reset_and_resize(4, 8);
        assert_eq!(ctx.width(), 4);
        assert_eq!(ctx.height(), 8);

        ctx.set_paint(BLUE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 4.0, 8.0));
        ctx.flush();
        ctx.render(&mut pixmap, &mut resources);

        for y in 0..8 {
            for x in 0..4 {
                assert_eq!(pixmap.sample(x, y), blue_pixel(), "pixel at ({x}, {y})");
            }
        }
    }

    #[test]
    fn reset_restores_default_aliasing_threshold() {
        let mut ctx = RenderContext::new(4, 4);
        ctx.set_aliasing_threshold(Some(128));

        ctx.reset();

        assert_eq!(ctx.aliasing_threshold, None);
    }

    #[test]
    fn render_into_raw_buffer() {
        let ctx = red_rect_context(2, 1, Rect::new(0.0, 0.0, 2.0, 1.0));
        let mut resources = Resources::new();
        let mut buffer = vec![0; 3 * 2 * 4];
        for pixel in buffer.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[GRAY.r, GRAY.g, GRAY.b, GRAY.a]);
        }

        {
            let pixmap = PixmapMut::new(3, 2, &mut buffer).unwrap();
            ctx.render_with(
                pixmap,
                &mut resources,
                RasterizerSettings {
                    offset: (1, 1),
                    ..Default::default()
                },
            );
        }

        let expected = [
            transparent_pixel(),
            transparent_pixel(),
            transparent_pixel(),
            transparent_pixel(),
            red_pixel(),
            red_pixel(),
        ];
        for (pixel, expected) in buffer.chunks_exact(4).zip(expected) {
            assert_eq!(pixel, [expected.r, expected.g, expected.b, expected.a]);
        }
    }

    #[test]
    fn pixmap_mut_validates_buffer_length() {
        let mut short_buffer = vec![0; 3 * 2 * 4 - 1];
        assert!(PixmapMut::new(3, 2, &mut short_buffer).is_none());

        let mut exact_buffer = vec![0; 3 * 2 * 4];
        assert!(PixmapMut::new(3, 2, &mut exact_buffer).is_some());
    }

    #[test]
    fn render_preserves_target_under_opaque_draw() {
        let ctx = red_rect_context(2, 1, Rect::new(0.0, 0.0, 1.0, 1.0));
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(2, 1, blue_pixel());

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                target_init: TargetInit::SrcOver,
                ..Default::default()
            },
        );

        assert_eq!(pixmap.sample(0, 0), red_pixel());
        assert_eq!(pixmap.sample(1, 0), blue_pixel());
    }

    #[test]
    fn render_preserves_target_under_translucent_draw() {
        let mut ctx = RenderContext::new(1, 1);
        ctx.set_paint(RED.with_alpha(0.5));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
        ctx.flush();

        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(1, 1, blue_pixel());

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                target_init: TargetInit::SrcOver,
                ..Default::default()
            },
        );

        assert_eq!(
            pixmap.sample(0, 0),
            PremulRgba8 {
                r: 128,
                g: 0,
                b: 127,
                a: 255,
            }
        );
    }

    #[test]
    fn opaque_clear_updates_target_transparency_hint() {
        let mut ctx = RenderContext::new(1, 1);
        ctx.flush();
        let mut resources = Resources::new();
        let mut pixmap = Pixmap::new(1, 1);

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                target_init: TargetInit::Clear(BLUE),
                ..Default::default()
            },
        );

        assert!(!pixmap.may_have_transparency());
        assert_eq!(pixmap.sample(0, 0), blue_pixel());
    }

    #[test]
    fn translucent_clear_updates_target_transparency_hint() {
        let mut ctx = RenderContext::new(1, 1);
        ctx.flush();
        let mut resources = Resources::new();
        let mut pixmap = solid_pixmap(1, 1, blue_pixel());
        let clear_color = RED.with_alpha(0.5);

        ctx.render_with(
            &mut pixmap,
            &mut resources,
            RasterizerSettings {
                target_init: TargetInit::Clear(clear_color),
                ..Default::default()
            },
        );

        assert!(pixmap.may_have_transparency());
        assert_eq!(pixmap.sample(0, 0), clear_color.premultiply().to_rgba8());
    }

    #[cfg(any())]
    #[test]
    fn multithreaded_crash_after_reset() {
        use crate::vello_cpu::{Level, RasterizerSettings, RenderMode, RenderSettings};

        let mut pixmap = Pixmap::new(200, 200);
        let settings = RenderSettings {
            level: Level::try_detect().unwrap_or(Level::baseline()),
            num_threads: 1,
        };
        let rasterizer_settings = RasterizerSettings {
            render_mode: RenderMode::OptimizeQuality,
            ..Default::default()
        };

        let mut resources = Resources::new();
        let mut ctx = RenderContext::new_with(200, 200, settings);
        ctx.reset();
        ctx.fill_path(&Rect::new(0.0, 0.0, 100.0, 100.0).to_path(0.1));
        ctx.flush();
        ctx.render_with(&mut pixmap, &mut resources, rasterizer_settings);
        ctx.flush();
        ctx.render_with(&mut pixmap, &mut resources, rasterizer_settings);
    }

    #[cfg(any())]
    #[test]
    fn multithreaded_render_empty_frame_after_reset() {
        use crate::vello_cpu::RenderSettings;

        let mut ctx = RenderContext::new_with(
            100,
            100,
            RenderSettings {
                num_threads: 4,
                ..Default::default()
            },
        );
        let mut resources = Resources::new();
        let mut pixmap = Pixmap::new(100, 100);

        ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
        ctx.flush();
        ctx.render(&mut pixmap, &mut resources);

        ctx.reset();
        ctx.flush();
        ctx.render(&mut pixmap, &mut resources);
    }

    #[cfg(any())]
    #[test]
    fn multithreaded_push_clip_path_before_draw() {
        use crate::vello_cpu::RenderSettings;

        let mut ctx = RenderContext::new_with(
            100,
            100,
            RenderSettings {
                num_threads: 1,
                ..Default::default()
            },
        );
        let clip = Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1);

        ctx.push_clip_path(&clip);
        ctx.flush();
        ctx.pop_clip();
        ctx.flush();
    }

    #[cfg(any())]
    #[test]
    fn multithreaded_reset_with_pending_tasks() {
        use crate::vello_cpu::RenderSettings;

        let mut ctx = RenderContext::new_with(
            100,
            100,
            RenderSettings {
                num_threads: 4,
                ..Default::default()
            },
        );

        for _ in 0..300 {
            ctx.fill_rect(&Rect::new(0.0, 0.0, 100., 100.0));
        }

        ctx.reset();
    }

    #[cfg(any())]
    #[test]
    fn multithreaded_drop_with_pending_tasks() {
        use crate::vello_cpu::RenderSettings;

        for _ in 0..10 {
            let mut ctx = RenderContext::new_with(
                100,
                100,
                RenderSettings {
                    num_threads: 4,
                    ..Default::default()
                },
            );

            for _ in 0..300 {
                ctx.fill_rect(&Rect::new(0.0, 0.0, 100., 100.0));
            }

            drop(ctx);
        }
    }

    #[cfg(all())]
    #[test]
    fn glyph_atlas_resources_are_lazy() {
        const ROBOTO_FONT: &[u8] = include_bytes!("../../assets/roboto/Roboto-Regular.ttf");

        let font = FontData::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);
        let glyphs = [Glyph {
            id: 1,
            x: 0.0,
            y: 0.0,
        }];

        let mut resources = Resources::new();
        let mut ctx = RenderContext::new(100, 100);

        ctx.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        ctx.fill_path(&Rect::new(10.0, 10.0, 20.0, 20.0).to_path(0.1));
        ctx.glyph_run(&mut resources, &font)
            .fill_glyphs(glyphs.into_iter())
            .unwrap();

        assert!(resources.glyph_resources.is_none());

        ctx.glyph_run(&mut resources, &font)
            .atlas_cache(true)
            .fill_glyphs(glyphs.into_iter())
            .unwrap();

        assert!(resources.glyph_resources.is_some());
    }
}
