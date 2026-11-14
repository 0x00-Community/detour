use super::cmd::{DepthFill, LayerFill, LayerFillAttrs, PaintFill, PaintFillAttrs, RenderCmd};
use super::depth::{DepthSegment, DepthState};
use crate::vello_cpu::coarse::depth;
use crate::vello_cpu::filter::context::FilterContext;
use crate::vello_cpu::kurbo::{Affine, Vec2};
use crate::vello_cpu::peniko::{BlendMode, Extend, ImageQuality, ImageSampler};
use crate::vello_cpu::record::RecordedFill;
use crate::vello_cpu::span::{Span, TileAlignedSpan};

use alloc::vec;
use alloc::vec::Vec;
use vello_common::encode::{EncodedImage, EncodedPaint};
use vello_common::filter::FilterLayerPlacement;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageSource, IndexedPaint, Paint};
use vello_common::record::{LayerClip, LayerProps, Node, RecordedLayer, RecordedLayerKind};
use vello_common::strip::{Strip, visit_strip_fill_segments};
use vello_common::tile::Tile;
use vello_common::util::{Clear, RectExt, RetainVec, VecPool};

#[derive(Debug, Default)]
pub(crate) struct RowState {
    pub(crate) render_cmds: Vec<RenderCmd>,
    pub(crate) depth_cmds: Vec<DepthFill>,
    depth: DepthState,
    pub(super) layer_depth: usize,
    layer_stack: Vec<RowLayerState>,
}

#[derive(Debug)]
struct RowLayerState {
    push_cmd_idx: usize,
    span: Option<TileAlignedSpan>,
}

impl RowState {
    pub(super) fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.render_cmds.clear();
        self.depth_cmds.clear();
        self.depth.reset();
        self.layer_depth = 0;
        self.layer_stack.clear();
    }

    #[inline]
    pub(super) fn push_cmd(&mut self, cmd: RenderCmd) {
        match cmd {
            RenderCmd::PaintFill(cmd) => self.include_current_span(cmd.span),
            RenderCmd::LayerFill(cmd) => self.include_current_span(cmd.span),
            RenderCmd::PushBuf(_) | RenderCmd::PopBuf => {}
        }

        self.render_cmds.push(cmd);
    }

    #[inline]
    pub(super) fn push_buf(&mut self) {
        let push_cmd_idx = self.render_cmds.len();
        self.render_cmds.push(RenderCmd::PushBuf(None));
        self.layer_stack.push(RowLayerState {
            push_cmd_idx,
            span: None,
        });
        self.layer_depth += 1;
    }

    #[inline]
    pub(super) fn pop_buf(&mut self) {
        let layer = self.layer_stack.pop().unwrap();
        match &mut self.render_cmds[layer.push_cmd_idx] {
            RenderCmd::PushBuf(push_span) => *push_span = layer.span,
            _ => unreachable!("layer stack must point to a PushBuf command"),
        }

        self.render_cmds.push(RenderCmd::PopBuf);

        self.layer_depth -= 1;
    }

    #[inline]
    pub(super) fn push_depth_fill(&mut self, cmd: DepthFill, draw_id: u32) {
        let span = cmd.span();
        self.depth.include_span(span, draw_id);
        self.depth_cmds.push(cmd);
    }

    #[inline]
    pub(crate) fn can_skip_depth(&self, span: TileAlignedSpan, draw_id: u32) -> bool {
        self.depth.can_skip(span, draw_id)
    }

    #[inline]
    fn include_current_span(&mut self, span: TileAlignedSpan) {
        if let Some(layer) = self.layer_stack.last_mut() {
            match &mut layer.span {
                Some(layer_span) => layer_span.extend(span),
                None => layer.span = Some(span),
            }
        }
    }
}

impl Clear for RowState {
    fn clear(&mut self) {
        Self::clear(self);
    }
}

fn debug_assert_tile_aligned(point: (u16, u16), description: &str) {
    debug_assert_eq!(
        point.0 % Tile::WIDTH,
        0,
        "{description} must be tile-width aligned",
    );
    debug_assert_eq!(
        point.1 % Tile::HEIGHT,
        0,
        "{description} must be tile-height aligned",
    );
}

#[derive(Debug)]
pub(crate) struct CommandBucketer {
    viewport: RectU16,
    pub(super) clip_bboxes: Vec<RectU16>,
    pub(super) rows: RetainVec<RowState>,
    pub(crate) paint_fill_attrs: Vec<PaintFillAttrs>,
    pub(crate) layer_fill_attrs: Vec<LayerFillAttrs>,
    pub(crate) filter_paints: Vec<EncodedPaint>,
    pub(super) active_layers: Vec<ActiveLayer>,
    occupied_rows_pool: VecPool<usize>,
    occupied_rows_bool_scratch: Vec<bool>,
    pub(super) next_draw_id: u32,
}

impl CommandBucketer {
    pub(crate) fn from_wh(width: u16, height: u16) -> Self {
        Self::new(RectU16::new(0, 0, width, height))
    }

    pub(crate) fn new(mut viewport: RectU16) -> Self {
        viewport = viewport.snap_to_tile_coordinates();
        let clip_bbox = RectU16::new(0, 0, viewport.width(), viewport.height());
        let num_rows = usize::from(clip_bbox.height() / Tile::HEIGHT);

        Self {
            viewport,
            clip_bboxes: vec![clip_bbox],
            rows: RetainVec::with_len(num_rows, RowState::new),
            paint_fill_attrs: Vec::new(),
            layer_fill_attrs: Vec::new(),
            filter_paints: Vec::new(),
            active_layers: Vec::new(),
            occupied_rows_pool: VecPool::default(),
            occupied_rows_bool_scratch: vec![false; num_rows],
            next_draw_id: 1,
        }
    }

    fn bbox_span(bbox: RectU16) -> TileAlignedSpan {
        if bbox.is_empty() {
            TileAlignedSpan::try_from(Span::new(bbox.x0, 0)).unwrap()
        } else {
            TileAlignedSpan::try_from(Span::new(bbox.x0, bbox.x1 - bbox.x0)).unwrap()
        }
    }

    pub(crate) fn rows(&self) -> &[RowState] {
        self.rows.as_slice()
    }

    pub(crate) fn width(&self) -> u16 {
        self.clip_bboxes[0].width()
    }

    fn viewport_origin(&self) -> (u16, u16) {
        (self.viewport.x0, self.viewport.y0)
    }

    pub(crate) fn reset(&mut self, mut viewport: RectU16) {
        viewport = viewport.snap_to_tile_coordinates();
        let clip_bbox = RectU16::new(0, 0, viewport.width(), viewport.height());
        let num_rows = usize::from(viewport.height() / Tile::HEIGHT);

        self.rows.clear();
        self.rows.resize_with(num_rows, RowState::new);
        self.paint_fill_attrs.clear();
        self.layer_fill_attrs.clear();
        self.filter_paints.clear();
        for layer in self.active_layers.drain(..) {
            self.occupied_rows_pool.submit(layer.occupied_rows);
        }
        self.occupied_rows_bool_scratch.clear();
        self.occupied_rows_bool_scratch.resize(num_rows, false);
        self.next_draw_id = 1;
        self.viewport = viewport;
        self.clip_bboxes.truncate(1);
        self.clip_bboxes[0] = clip_bbox;
    }

    fn next_draw_id(&mut self) -> u32 {
        let draw_id = self.next_draw_id;
        self.next_draw_id += 1;

        draw_id
    }

    #[inline(always)]
    pub(super) fn ensure_row_layers(&mut self, row_idx: usize) {
        let layer_depth = self.rows[row_idx].layer_depth;
        if layer_depth == self.active_layers.len() {
            return;
        }

        for layer_idx in layer_depth..self.active_layers.len() {
            self.rows[row_idx].push_buf();
            self.active_layers[layer_idx].occupied_rows.push(row_idx);
        }
    }

    pub(crate) fn bucket_commands(
        &mut self,
        nodes: &[Node],
        draws: &[RecordedFill],
        layers: &[RecordedLayer],
        strips: &[Strip],
        encoded_paints: &[EncodedPaint],
        filter_ctx: &FilterContext,
    ) {
        debug_assert_tile_aligned(self.viewport_origin(), "viewport origin");

        for node in nodes {
            for RecordedFill {
                thread_idx,
                strip_range,
                paint,
                blend_mode,
                mask,
            } in node.draws_in(draws)
            {
                let draw_id = self.next_draw_id();
                let attrs = PaintFillAttrs {
                    paint: paint.clone(),
                    blend_mode: *blend_mode,
                    mask: mask.clone(),
                    draw_id,
                    thread_idx: *thread_idx,
                    origin: self.viewport_origin(),
                };
                self.generate_fill(&strips[strip_range.clone()], &attrs, encoded_paints);
            }

            if let Some(id) = node.layer {
                let layer = &layers[id as usize];
                let props = &layer.props;

                match &layer.kind {
                    RecordedLayerKind::Regular => {
                        self.push_layer(props);
                        self.bucket_commands(
                            &layer.nodes,
                            draws,
                            layers,
                            strips,
                            encoded_paints,
                            filter_ctx,
                        );
                        self.pop_layer(strips);
                    }
                    RecordedLayerKind::Filter { placement, .. } => {
                        let needs_layer = props.blend_mode != BlendMode::default()
                            || props.opacity != 1.0
                            || props.mask.is_some()
                            || props.clip_path.is_some();

                        if needs_layer {
                            self.push_layer(props);
                        }

                        if let Some(source) = filter_ctx.filter_source(id as usize) {
                            self.generate_filter_layer_fill(
                                source,
                                *placement,
                                encoded_paints.len(),
                            );
                        }

                        if needs_layer {
                            self.pop_layer(strips);
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn push_layer(&mut self, props: &LayerProps) {
        let parent_bbox = *self.clip_bboxes.last().unwrap();
        let bbox = props
            .clip_path
            .as_ref()
            .map(|clip| {
                let clip_bbox = clip.bbox.relative_to_origin(self.viewport_origin());
                clip_bbox.intersect(parent_bbox)
            })
            .unwrap_or(parent_bbox);

        if props.clip_path.is_some() {
            self.clip_bboxes.push(bbox);
        }

        self.active_layers.push(ActiveLayer {
            mask: props.mask.clone(),
            blend_mode: props.blend_mode,
            opacity: props.opacity,
            clip: props.clip_path.clone(),
            span: Self::bbox_span(bbox),
            occupied_rows: self.occupied_rows_pool.take(),
        });

        if props.blend_mode.is_destructive() && !bbox.is_empty() {
            let row_start = usize::from(bbox.y0 / Tile::HEIGHT);
            let row_end = usize::from(bbox.y1.div_ceil(Tile::HEIGHT)).min(self.rows.len());
            for row_idx in row_start..row_end {
                self.ensure_row_layers(row_idx);
            }
        }
    }

    pub(crate) fn pop_layer(&mut self, strips: &[Strip]) {
        let mut layer = self.active_layers.pop().unwrap();
        let opacity = layer.opacity;
        let blend_mode = layer.blend_mode;

        if let Some(clip) = layer.clip {
            let attrs_idx = self.layer_fill_attrs.len() as u32;
            let draw_id = self.next_draw_id();

            self.layer_fill_attrs.push(LayerFillAttrs {
                blend_mode,
                opacity,
                mask: layer.mask.clone(),
                draw_id,
                thread_idx: clip.thread_idx,
            });
            self.clip_bboxes.pop();

            let clip_strips = &strips[clip.strip_range];

            for &row_idx in &layer.occupied_rows {
                self.occupied_rows_bool_scratch[row_idx] = true;
            }

            self.generate(
                clip_strips,
                |bucketer, fill| {
                    if bucketer.occupied_rows_bool_scratch[fill.row_idx] {
                        bucketer.rows[fill.row_idx].push_cmd(RenderCmd::LayerFill(LayerFill::new(
                            fill.span, None, attrs_idx,
                        )));
                    }
                },
                |bucketer, fill| {
                    if bucketer.occupied_rows_bool_scratch[fill.row_idx] {
                        bucketer.rows[fill.row_idx].push_cmd(RenderCmd::LayerFill(LayerFill::new(
                            fill.span,
                            Some(fill.alpha_idx),
                            attrs_idx,
                        )));
                    }
                },
            );

            for row_idx in layer.occupied_rows.drain(..) {
                self.rows[row_idx].pop_buf();
                self.occupied_rows_bool_scratch[row_idx] = false;
            }

            self.occupied_rows_pool.submit(layer.occupied_rows);
        } else {
            let attrs_idx = self.layer_fill_attrs.len() as u32;
            let draw_id = self.next_draw_id();

            self.layer_fill_attrs.push(LayerFillAttrs {
                blend_mode,
                opacity,
                mask: layer.mask.clone(),
                draw_id,
                thread_idx: 0,
            });

            for row_idx in layer.occupied_rows.drain(..) {
                let row = &mut self.rows[row_idx];

                row.push_cmd(RenderCmd::LayerFill(LayerFill::new(
                    layer.span, None, attrs_idx,
                )));
                row.pop_buf();
            }

            self.occupied_rows_pool.submit(layer.occupied_rows);
        }
    }

    pub(crate) fn generate_filter_layer_fill(
        &mut self,
        source: ImageSource,
        placement: FilterLayerPlacement,
        static_paint_count: usize,
    ) {
        let origin = self.viewport_origin();
        let dest_bbox = placement.dest_bbox();
        let src_sample_shift = placement.src_origin();

        let src_offset = (
            i32::from(src_sample_shift.0) - i32::from(dest_bbox.x0),
            i32::from(src_sample_shift.1) - i32::from(dest_bbox.y0),
        );

        let dest_bbox = dest_bbox.relative_to_origin(origin);
        let clip_bbox = *self.clip_bboxes.last().unwrap();
        let clipped_dest_bbox = dest_bbox.intersect(clip_bbox);
        if clipped_dest_bbox.is_empty() {
            return;
        }

        let draw_id = self.next_draw_id();
        let span = Self::bbox_span(clipped_dest_bbox);
        let paint_idx = static_paint_count + self.filter_paints.len();
        self.filter_paints.push(EncodedPaint::Image(EncodedImage {
            source,
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
            may_have_transparency: true,
            transform: Affine::translate((f64::from(src_offset.0), f64::from(src_offset.1))),
            x_advance: Vec2::new(1.0, 0.0),
            y_advance: Vec2::new(0.0, 1.0),
            tint: None,
        }));
        let attrs_idx = self.paint_fill_attrs.len() as u32;
        self.paint_fill_attrs.push(PaintFillAttrs {
            paint: Paint::Indexed(IndexedPaint::new(paint_idx)),
            blend_mode: BlendMode::default(),
            mask: None,
            draw_id,
            thread_idx: 0,
            origin,
        });
        let row_start = usize::from(clipped_dest_bbox.y0 / Tile::HEIGHT);
        let row_end = usize::from(clipped_dest_bbox.y1.div_ceil(Tile::HEIGHT));
        for row_idx in row_start..row_end {
            self.push_fill(GeneratedFill { row_idx, span }, attrs_idx, None);
        }
    }

    pub(crate) fn generate_fill(
        &mut self,
        strip_buf: &[Strip],
        attrs: &PaintFillAttrs,
        encoded_paints: &[EncodedPaint],
    ) {
        if strip_buf.is_empty() {
            return;
        }

        debug_assert_ne!(attrs.draw_id, 0, "fill draw IDs should start at 1");

        let attrs_idx = self.paint_fill_attrs.len() as u32;
        self.paint_fill_attrs.push(attrs.clone());

        let draw_id =
            (self.active_layers.is_empty()
                && attrs.blend_mode == BlendMode::default()
                && attrs.mask.is_none()
                && !attrs.paint.may_have_transparency(encoded_paints))
                .then_some(attrs.draw_id);

        self.generate(
            strip_buf,
            |bucketer, fill| {
                bucketer.push_fill(fill, attrs_idx, draw_id);
            },
            |bucketer, fill| {
                let row_idx = fill.row_idx;
                bucketer.ensure_row_layers(row_idx);
                bucketer.rows[row_idx].push_cmd(RenderCmd::PaintFill(PaintFill::new(
                    fill.span,
                    Some(fill.alpha_idx),
                    attrs_idx,
                )));
            },
        );
    }

    pub(crate) fn generate<F, A>(
        &mut self,
        strip_buf: &[Strip],
        mut fill_cmd: F,
        mut alpha_fill_cmd: A,
    ) where
        F: FnMut(&mut Self, GeneratedFill),
        A: FnMut(&mut Self, GeneratedAlphaFill),
    {
        if strip_buf.is_empty() {
            return;
        }

        let clip_bbox = *self.clip_bboxes.last().unwrap();

        if clip_bbox.is_empty() {
            return;
        }

        let clip_x0 = clip_bbox.x0;
        let clip_x1 = clip_bbox.x1;

        debug_assert_tile_aligned((clip_x0, clip_bbox.y0), "clip start");
        debug_assert_tile_aligned((clip_x1, clip_bbox.y1), "clip end");

        let origin = self.viewport_origin();
        debug_assert_tile_aligned(origin, "viewport origin");

        let origin_tile_x = origin.0 / Tile::WIDTH;
        let origin_tile_y = origin.1 / Tile::HEIGHT;
        let clip_scene_y0 = origin.1.saturating_add(clip_bbox.y0);
        let clip_scene_y1 = origin.1.saturating_add(clip_bbox.y1);
        let clip_scene_x0 = origin.0.saturating_add(clip_x0);
        let clip_scene_x1 = origin.0.saturating_add(clip_x1);

        let tile_bounds = RectU16::new(
            clip_scene_x0 / Tile::WIDTH,
            clip_scene_y0 / Tile::HEIGHT,
            clip_scene_x1 / Tile::WIDTH,
            clip_scene_y1 / Tile::HEIGHT,
        );

        visit_strip_fill_segments(
            strip_buf,
            tile_bounds,
            self,
            |bucketer, segment| {
                let row_idx = usize::from(segment.tile_y - origin_tile_y);
                alpha_fill_cmd(
                    bucketer,
                    GeneratedAlphaFill {
                        row_idx,
                        span: TileAlignedSpan::from_tiles(
                            segment.tile_x0 - origin_tile_x,
                            segment.tile_x1 - segment.tile_x0,
                        ),
                        alpha_idx: segment.alpha_idx,
                    },
                );
            },
            |bucketer, segment| {
                let row_idx = usize::from(segment.tile_y - origin_tile_y);
                fill_cmd(
                    bucketer,
                    GeneratedFill {
                        row_idx,
                        span: TileAlignedSpan::from_tiles(
                            segment.tile_x0 - origin_tile_x,
                            segment.tile_x1 - segment.tile_x0,
                        ),
                    },
                );
            },
        );
    }

    fn push_fill(&mut self, fill: GeneratedFill, attrs_idx: u32, draw_id: Option<u32>) {
        self.ensure_row_layers(fill.row_idx);
        let row = &mut self.rows[fill.row_idx];
        let draw_id = draw_id.filter(|_| row.layer_depth == 0);

        let Some(draw_id) = draw_id else {
            row.push_cmd(RenderCmd::PaintFill(PaintFill::new(
                fill.span, None, attrs_idx,
            )));

            return;
        };

        depth::split_opaque_span(fill.span, |segment| match segment {
            DepthSegment::Regular(span) => {
                row.push_cmd(RenderCmd::PaintFill(PaintFill::new(span, None, attrs_idx)));
            }
            DepthSegment::Opaque(bucket_range) => {
                row.push_depth_fill(DepthFill::new(bucket_range, attrs_idx), draw_id);
            }
        });
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ActiveLayer {
    pub(crate) mask: Option<Mask>,
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
    pub(crate) clip: Option<LayerClip>,
    pub(crate) span: TileAlignedSpan,
    pub(crate) occupied_rows: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedFill {
    pub(crate) row_idx: usize,
    pub(crate) span: TileAlignedSpan,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneratedAlphaFill {
    pub(crate) row_idx: usize,
    pub(crate) span: TileAlignedSpan,
    pub(crate) alpha_idx: u32,
}

#[cfg(any())]
mod tests {
    use super::LayerClip;
    use crate::vello_cpu::coarse::CommandBucketer;
    use crate::vello_cpu::coarse::cmd::{PaintFillAttrs, RenderCmd};
    use crate::vello_cpu::coarse::depth::{BucketRange, DEPTH_BUCKET_WIDTH};
    use vello_common::color::palette::css::{BLUE, RED};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::geometry::RectU16;
    use vello_common::paint::{Paint, PremulColor};
    use vello_common::peniko::{BlendMode, Compose, Mix};
    use vello_common::record::LayerProps;
    use vello_common::strip::Strip;
    use vello_common::tile::Tile;

    fn color(alpha: AlphaColor<Srgb>) -> PremulColor {
        PremulColor::from_alpha_color(alpha)
    }

    fn fill_attrs(paint: Paint) -> PaintFillAttrs {
        PaintFillAttrs {
            paint,
            blend_mode: BlendMode::default(),
            mask: None,
            draw_id: 1,
            thread_idx: 0,
            origin: (0, 0),
        }
    }

    fn layer_props() -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip_path: None,
        }
    }

    fn clipped_layer_props(bbox: RectU16) -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip_path: Some(LayerClip {
                strip_range: 0..0,
                thread_idx: 0,
                bbox,
            }),
        }
    }

    fn destructive_clipped_layer_props(bbox: RectU16) -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::new(Mix::Normal, Compose::Clear),
            ..clipped_layer_props(bbox)
        }
    }

    fn clipped_layer_props_with_strips(
        bbox: RectU16,
        strip_range: core::ops::Range<usize>,
    ) -> LayerProps {
        LayerProps {
            blend_mode: BlendMode::default(),
            opacity: 1.0,
            mask: None,
            clip_path: Some(LayerClip {
                strip_range,
                thread_idx: 0,
                bbox,
            }),
        }
    }

    fn count_layer_fills(cmds: &[RenderCmd]) -> usize {
        cmds.iter()
            .filter(|cmd| matches!(cmd, RenderCmd::LayerFill(_)))
            .count()
    }

    #[test]
    fn opaque_fill_inside_layer_does_not_use_depth_write() {
        let mut bucketer = CommandBucketer::from_wh(DEPTH_BUCKET_WIDTH, 4);
        let strips = [
            Strip::new(0, 0, 0, false),
            Strip::new(DEPTH_BUCKET_WIDTH, 0, 0, true),
        ];

        bucketer.push_layer(&layer_props());
        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];
        assert_eq!(row.depth_cmds.len(), 0);
        assert_eq!(row.render_cmds.len(), 2);
        assert!(matches!(row.render_cmds[0], RenderCmd::PushBuf(_)));
        assert!(
            matches!(row.render_cmds[1], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH)
        );
    }

    #[test]
    fn alpha_fill_is_clipped_to_active_layer_bbox() {
        let mut bucketer = CommandBucketer::from_wh(8, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(12, 0, 48, false)];

        bucketer.push_layer(&clipped_layer_props(RectU16::new(4, 0, 8, 4)));
        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];
        assert_eq!(row.render_cmds.len(), 2);
        assert!(matches!(row.render_cmds[0], RenderCmd::PushBuf(_)));
        assert!(matches!(
            row.render_cmds[1],
            RenderCmd::PaintFill(cmd)
                if cmd.span.pixel_x() == 4
                    && cmd.span.pixel_width() == 4
                    && cmd.alpha_idx() == Some(u32::from(4 * Tile::HEIGHT))
        ));
    }

    #[test]
    fn disjoint_nested_clip_bounds_do_not_emit_commands() {
        let mut bucketer = CommandBucketer::from_wh(16, 4);
        let strips = [Strip::new(0, 0, 0, false), Strip::new(16, 0, 0, true)];

        bucketer.push_layer(&clipped_layer_props(RectU16::new(0, 0, 4, 4)));
        bucketer.push_layer(&clipped_layer_props(RectU16::new(8, 0, 12, 4)));
        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        assert!(bucketer.rows().iter().all(|row| row.render_cmds.is_empty()));
    }

    #[test]
    fn empty_destructive_clip_does_not_push_rows() {
        let mut bucketer = CommandBucketer::from_wh(16, 4);

        bucketer.push_layer(&clipped_layer_props(RectU16::new(0, 0, 4, 4)));
        bucketer.push_layer(&destructive_clipped_layer_props(RectU16::new(8, 0, 12, 4)));

        assert!(bucketer.rows().iter().all(|row| row.render_cmds.is_empty()));
    }

    #[test]
    fn opaque_fill_uses_depth_write_when_possible() {
        let end = DEPTH_BUCKET_WIDTH * 2 + 4;
        let mut bucketer = CommandBucketer::from_wh(end, 4);
        let strips = [Strip::new(4, 0, 0, false), Strip::new(end, 0, 0, true)];

        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];
        assert_eq!(row.depth_cmds.len(), 1);
        assert_eq!(row.depth_cmds[0].bucket_range(), BucketRange::new(1, 2));
        assert_eq!(row.render_cmds.len(), 2);
        assert!(
            matches!(row.render_cmds[0], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 4 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH - 4)
        );
        assert!(
            matches!(row.render_cmds[1], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == DEPTH_BUCKET_WIDTH * 2 && cmd.span.pixel_width() == 4)
        );
    }

    #[test]
    fn non_opaque_fill_uses_regular_commands() {
        let mut bucketer = CommandBucketer::from_wh(DEPTH_BUCKET_WIDTH, 4);
        let strips = [
            Strip::new(0, 0, 0, false),
            Strip::new(DEPTH_BUCKET_WIDTH, 0, 0, true),
        ];

        bucketer.generate_fill(
            &strips,
            &fill_attrs(Paint::Solid(color(BLUE.with_alpha(0.5)))),
            &[],
        );

        let row = &bucketer.rows()[0];
        assert_eq!(row.depth_cmds.len(), 0);
        assert_eq!(row.render_cmds.len(), 1);
        assert!(
            matches!(row.render_cmds[0], RenderCmd::PaintFill(cmd) if cmd.span.pixel_x() == 0 && cmd.span.pixel_width() == DEPTH_BUCKET_WIDTH)
        );
    }

    #[test]
    fn clips_fills_correctly_inside_nonzero_origin_viewport() {
        let mut bucketer = CommandBucketer::new(RectU16::new(32, 32, 96, 96));
        bucketer.push_layer(&clipped_layer_props(RectU16::new(40, 32, 72, 96)));

        let strips = [
            Strip::new(40, 32, 0, false),
            Strip::new(72, 32, 32 * u32::from(Tile::HEIGHT), false),
        ];
        bucketer.generate_fill(&strips, &fill_attrs(Paint::Solid(color(RED))), &[]);

        let row = &bucketer.rows()[0];

        assert!(matches!(row.render_cmds[0], RenderCmd::PushBuf(_)));
        assert!(matches!(
            row.render_cmds[1],
            RenderCmd::PaintFill(cmd)
                if cmd.span.pixel_x() == 8
                    && cmd.span.pixel_width() == 32
                    && cmd.alpha_idx() == Some(0)
        ));
    }

    #[test]
    fn culls_clip_strips_above_viewport_origin() {
        let mut bucketer = CommandBucketer::new(RectU16::new(0, 32, 64, 96));

        let alpha = u32::from(Tile::HEIGHT);
        let strips = [
            Strip::new(0, 32, 0, false),
            Strip::new(16, 32, 16 * alpha, false),
            Strip::new(0, 0, 16 * alpha, false),
            Strip::new(16, 0, 32 * alpha, false),
            Strip::new(0, 32, 32 * alpha, false),
            Strip::new(16, 32, 48 * alpha, false),
        ];

        bucketer.push_layer(&clipped_layer_props_with_strips(
            RectU16::new(0, 0, 16, 40),
            2..6,
        ));
        bucketer.generate_fill(&strips[0..2], &fill_attrs(Paint::Solid(color(RED))), &[]);
        bucketer.pop_layer(&strips);

        let row = &bucketer.rows()[0];
        assert_eq!(
            count_layer_fills(&row.render_cmds),
            1,
            "row 0 must be composited exactly once, got {:?}",
            row.render_cmds
        );
    }
}
