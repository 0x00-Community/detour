use crate::vello_cpu::RasterizerSettings;
use crate::vello_cpu::coarse::CommandBucketer;
use crate::vello_cpu::coarse::depth::DepthBuffer;
use crate::vello_cpu::dispatch::Dispatcher;
use crate::vello_cpu::filter::context::FilterContext;
use crate::vello_cpu::fine::{Fine, FineKernel, FineRenderParams, FineResources, rasterize_region};
use crate::vello_cpu::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::vello_cpu::peniko::{BlendMode, Fill};
use crate::vello_cpu::record::RecordedFill;
use crate::vello_cpu::region::Regions;
use core::cell::RefCell;
use vello_common::TargetInit;
use vello_common::color::AlphaColor;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd};
use vello_common::filter::FilterData;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint, PremulColor};
use vello_common::pixmap::{Pixmap, PixmapMut};
use vello_common::record::{
    CommandRecorder, LayerClip, LayerProps, Node, PoppedLayer, RecordedLayerKind,
};
use vello_common::strip_generator::{GenerationMode, StripStorage};
use vello_common::util::strip_bbox;
use vello_common::viewport::ViewportState;

#[derive(Debug)]
pub(crate) struct SingleThreadedDispatcher {
    bucketer: RefCell<CommandBucketer>,
    viewport: ViewportState,
    recorder: CommandRecorder<RecordedFill>,
    strip_storage: StripStorage,
    level: Level,
}

impl SingleThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            bucketer: RefCell::new(CommandBucketer::from_wh(width, height)),
            viewport: ViewportState::new(width, height, level),
            recorder: CommandRecorder::new(width, height),
            strip_storage: StripStorage::new(GenerationMode::Append),
            level,
        }
    }

    #[cfg(all())]
    fn rasterize_f32(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::vello_cpu::fine::F32Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    #[cfg(all())]
    fn rasterize_u8(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::vello_cpu::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let filters = self.rasterize_filter_layers::<S, F>(simd, encoded_paints, image_resolver);
        let target_init = settings.target_init.map(PremulColor::from_alpha_color);
        let params = FineRenderParams {
            scene_size: (scene_width, scene_height),
            target_offset: settings.offset,
        };

        self.bucket_and_rasterize::<S, F>(
            simd,
            &self.recorder.nodes,
            RectU16::new(0, 0, scene_width, scene_height),
            &filters,
            target,
            params,
            target_init,
            self.recorder.root_is_blend_target,
            encoded_paints,
            image_resolver,
        );
    }

    fn bucket_and_rasterize<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        cmds: &[Node],
        viewport: RectU16,
        filter_ctx: &FilterContext,
        mut target: PixmapMut<'_>,
        params: FineRenderParams,
        target_init: TargetInit<PremulColor>,
        root_is_blend_target: bool,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let mut bucketer = self.bucketer.borrow_mut();
        bucketer.reset(viewport);
        bucketer.bucket_commands(
            cmds,
            &self.recorder.draws,
            &self.recorder.layers,
            &self.strip_storage.strips,
            encoded_paints,
            filter_ctx,
        );

        let alpha_buffers = &[self.strip_storage.alphas.as_slice()];
        let resources = FineResources {
            alpha_buffers,
            encoded_paints,
            filter_paints: &bucketer.filter_paints,
            image_resolver,
        };
        let mut regions = Regions::new(
            &mut target,
            params.scene_size,
            params.target_offset,
            bucketer.rows().len(),
        );
        Self::rasterize_target::<S, F>(
            simd,
            &bucketer,
            resources,
            &mut regions,
            target_init,
            root_is_blend_target,
        );
    }

    fn rasterize_target<S: Simd, F: FineKernel<S>>(
        simd: S,
        bucketer: &CommandBucketer,
        resources: FineResources<'_>,
        regions: &mut Regions<'_>,
        target_init: TargetInit<PremulColor>,
        root_is_blend_target: bool,
    ) {
        let mut fine = Fine::<S, F>::new(simd, bucketer.width());
        let mut depth = DepthBuffer::new(bucketer.width());
        regions.update(|region| {
            rasterize_region::<S, F>(
                &mut fine,
                &mut depth,
                region,
                bucketer,
                resources,
                target_init,
                root_is_blend_target,
            );
        });
    }

    fn record_fill(
        &mut self,
        strip_start: usize,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        let strip_end = self.strip_storage.strips.len();
        let strip_range = strip_start..strip_end;
        let strips = &self.strip_storage.strips[strip_range.start..strip_range.end];
        let draw = RecordedFill::new(0, strip_range, paint, blend_mode, mask);

        self.recorder.push_draw(draw, strips);
    }

    fn rasterize_filter_layers<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) -> FilterContext {
        let mut filter_ctx = FilterContext::new(self.recorder.layers.len());
        for id in self.recorder.filter_layers.iter().rev().copied() {
            let RecordedLayerKind::Filter {
                filter_data: filter_plan,
                placement,
            } = &self.recorder.layers[id as usize].kind
            else {
                unreachable!("filter_layers only contains filter layers");
            };
            let pixmap_bbox = placement.pixmap_bbox();
            if pixmap_bbox.is_empty() {
                continue;
            }

            let width = pixmap_bbox.width();
            let height = pixmap_bbox.height();
            let mut pixmap = Pixmap::new(width, height);
            let params = FineRenderParams {
                scene_size: (width, height),
                target_offset: (0, 0),
            };

            self.bucket_and_rasterize::<S, F>(
                simd,
                &self.recorder.layers[id as usize].nodes,
                pixmap_bbox,
                &filter_ctx,
                (&mut pixmap).into(),
                params,
                TargetInit::Clear(PremulColor::from_alpha_color(AlphaColor::TRANSPARENT)),
                false,
                encoded_paints,
                image_resolver,
            );

            F::filter_layer(
                &mut pixmap,
                &filter_plan.filter,
                filter_ctx.scratch(),
                filter_plan.transform,
            );

            filter_ctx.set_layer(id as usize, pixmap);
        }

        filter_ctx
    }
}

impl Dispatcher for SingleThreadedDispatcher {
    fn has_layers(&self) -> bool {
        self.recorder.has_layers()
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let strip_start = self.strip_storage.strips.len();
        let strip_storage = &mut self.strip_storage;
        self.viewport
            .with_generator_and_clip(|strip_generator, clip_path| {
                strip_generator.generate_filled_path(
                    path,
                    fill_rule,
                    transform,
                    aliasing_threshold,
                    strip_storage,
                    clip_path.map(|clip| clip.path),
                );
            });
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let strip_start = self.strip_storage.strips.len();
        let strip_storage = &mut self.strip_storage;
        self.viewport
            .with_generator_and_clip(|strip_generator, clip_path| {
                strip_generator.generate_stroked_path(
                    path,
                    stroke,
                    transform,
                    aliasing_threshold,
                    strip_storage,
                    clip_path.map(|clip| clip.path),
                );
            });
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        let strip_start = self.strip_storage.strips.len();
        let strip_storage = &mut self.strip_storage;
        self.viewport
            .with_generator_and_clip(|strip_generator, clip_path| {
                strip_generator.generate_filled_rect_fast(rect, strip_storage, clip_path);
            });
        self.record_fill(strip_start, paint, blend_mode, mask);
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter_data: Option<FilterData>,
    ) {
        if let Some(filter_data) = &filter_data {
            self.viewport.push_root_viewport(filter_data);
        }

        let clip_path = clip_path.map(|clip_path| {
            let strip_start = self.strip_storage.strips.len();
            let strip_storage = &mut self.strip_storage;
            self.viewport
                .with_generator_and_clip(|strip_generator, existing_clip| {
                    strip_generator.generate_filled_path(
                        clip_path,
                        fill_rule,
                        clip_transform,
                        aliasing_threshold,
                        strip_storage,
                        existing_clip.map(|clip| clip.path),
                    );

                    let strip_range = strip_start..strip_storage.strips.len();
                    LayerClip {
                        bbox: strip_bbox(&strip_storage.strips[strip_range.clone()])
                            .unwrap_or(RectU16::ZERO),
                        strip_range,
                        thread_idx: 0,
                    }
                })
        });

        self.recorder.push_layer(
            LayerProps {
                blend_mode,
                opacity,
                mask,
                clip_path,
            },
            filter_data,
        );
    }

    fn pop_layer(&mut self) {
        match self.recorder.pop_layer() {
            PoppedLayer::Regular => {}
            PoppedLayer::Filter => {
                self.viewport.pop_root_viewport();
            }
        }
    }

    fn reset(&mut self, width: u16, height: u16) {
        self.recorder.reset(width, height);
        self.strip_storage.clear();
        self.viewport.reset(width, height);
    }

    fn flush(&mut self) {
    }

    fn rasterize(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        #[cfg(all(all(), not(all())))]
        {
            self.rasterize_u8(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }

        #[cfg(all(all(), not(all())))]
        {
            self.rasterize_f32(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }

        #[cfg(all(all(), all()))]
        match settings.render_mode {
            crate::vello_cpu::RenderMode::OptimizeSpeed => {
                self.rasterize_u8(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
            crate::vello_cpu::RenderMode::OptimizeQuality => {
                self.rasterize_f32(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
        }

        #[cfg(all(not(all()), not(all())))]
        {
            let _ = (
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }
    }

    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.viewport
            .push_clip_path(path, fill_rule, transform, aliasing_threshold);
    }

    fn push_clip_rect(&mut self, rect: &Rect) {
        self.viewport.push_clip_rect(rect);
    }

    fn pop_clip(&mut self) {
        self.viewport.pop_clip();
    }

    fn coarse<'s>(&'s self, (width, height): (u16, u16), encoded_paints: &'s [EncodedPaint], image_resolver: &dyn ImageResolver) -> Option<crate::vello_cpu::detour::Coarse<'s>> {
        use crate::vello_cpu::fine::U8Kernel;
        use vello_common::fearless_simd::dispatch;
        Some(dispatch!(self.level, simd => {
            let filters = self.rasterize_filter_layers::<_, U8Kernel>(simd, encoded_paints, image_resolver);
            let mut bucketer = self.bucketer.borrow_mut();
            bucketer.reset(RectU16::new(0, 0, width, height));
            bucketer.bucket_commands(&self.recorder.nodes, &self.recorder.draws, &self.recorder.layers, &self.strip_storage.strips, encoded_paints, &filters);
            crate::vello_cpu::detour::Coarse::of(&mut bucketer, &self.strip_storage.alphas, encoded_paints, (width, height), self.recorder.root_is_blend_target)
        }))
    }

    fn coarse_layered<'s>(&'s self, (width, height): (u16, u16), encoded_paints: &'s [EncodedPaint]) -> Option<crate::vello_cpu::detour::Layered<'s>> {
        let filter_ctx = FilterContext::elsewhere();
        let mut bucketer = self.bucketer.borrow_mut();
        let mut filters = alloc::vec::Vec::new();
        for id in self.recorder.filter_layers.iter().rev().copied() {
            let RecordedLayerKind::Filter { filter_data: plan, placement } = &self.recorder.layers[id as usize].kind else {
                unreachable!("filter_layers only contains filter layers");
            };
            let bbox = placement.pixmap_bbox();
            if bbox.is_empty() {
                continue;
            }
            bucketer.reset(bbox);
            bucketer.bucket_commands(&self.recorder.layers[id as usize].nodes, &self.recorder.draws, &self.recorder.layers, &self.strip_storage.strips, encoded_paints, &filter_ctx);
            let coarse = crate::vello_cpu::detour::Coarse::of(&mut bucketer, &self.strip_storage.alphas, encoded_paints, (bbox.width(), bbox.height()), false);
            filters.push(crate::vello_cpu::detour::FilterLayer { image: crate::vello_cpu::detour::FILTER_IMAGE_BASE + id, coarse, filter: plan.filter.clone(), transform: plan.transform });
        }
        bucketer.reset(RectU16::new(0, 0, width, height));
        bucketer.bucket_commands(&self.recorder.nodes, &self.recorder.draws, &self.recorder.layers, &self.strip_storage.strips, encoded_paints, &filter_ctx);
        let root = crate::vello_cpu::detour::Coarse::of(&mut bucketer, &self.strip_storage.alphas, encoded_paints, (width, height), self.recorder.root_is_blend_target);
        Some(crate::vello_cpu::detour::Layered { filters, root })
    }

    fn is_multi_threaded(&self) -> bool {
        false
    }
}

#[allow(
    dead_code,
    reason = "useful debug utility, can be enabled by uncommenting the call site"
)]
#[cfg(all(debug_assertions, all(), all()))]
fn save_filtered_layer_debug(pixmap: &Pixmap, layer_id: usize) {
    use std::path::PathBuf;

    let diffs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_tests/diffs");
    let _ = std::fs::create_dir_all(&diffs_path);
    let filename = diffs_path.join(alloc::format!("filtered_layer_{layer_id}.png"));

    if let Ok(png_data) = pixmap.clone().into_png() {
        let _ = std::fs::write(&filename, &png_data);
    }
}

#[cfg(any())]
mod tests {
    use super::*;
    use crate::vello_cpu::kurbo::Shape;
    use vello_common::color::palette::css::BLUE;
    use vello_common::paint::PremulColor;

    #[test]
    fn buffers_cleared_on_reset() {
        let mut dispatcher = SingleThreadedDispatcher::new(100, 100, Level::new());

        dispatcher.fill_path(
            &Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
            BlendMode::default(),
            None,
            None,
        );

        assert!(!dispatcher.strip_storage.strips.is_empty());
        assert!(!dispatcher.recorder.nodes.is_empty());

        dispatcher.reset(100, 100);

        assert!(dispatcher.strip_storage.strips.is_empty());
        assert!(dispatcher.strip_storage.alphas.is_empty());
        assert!(dispatcher.recorder.nodes.is_empty());
        assert!(dispatcher.recorder.layers.is_empty());
        assert!(!dispatcher.viewport.has_root_viewports());
    }
}
