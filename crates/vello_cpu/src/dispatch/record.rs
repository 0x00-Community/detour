use crate::vello_cpu::RasterizerSettings;
use crate::vello_cpu::detour::{Geometry, Item, LayerAttrs, Recording};
use crate::vello_cpu::coarse::PaintFillAttrs;
use crate::vello_cpu::dispatch::Dispatcher;
use crate::vello_cpu::kurbo::{Affine, BezPath, Rect, Stroke, StrokeCtx};
use crate::vello_cpu::peniko::{BlendMode, Fill};
use alloc::vec::Vec;
use vello_common::encode::EncodedPaint;
use vello_common::filter::FilterData;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::PixmapMut;

#[derive(Debug)]
pub(crate) struct RecordingDispatcher {
    recording: Recording,
    open: Vec<LayerAttrs>,
    next_draw_id: u32,
    stroke_ctx: StrokeCtx,
}

impl RecordingDispatcher {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        Self { recording: Recording { width, height, ..Recording::default() }, open: Vec::new(), next_draw_id: 1, stroke_ctx: StrokeCtx::default() }
    }

    fn draw_id(&mut self) -> u32 {
        let id = self.next_draw_id;
        self.next_draw_id += 1;
        id
    }

    fn refuse(&mut self, what: &'static str) {
        self.recording.refused.get_or_insert(what);
    }

    fn fill(&mut self, geometry: Geometry, paint: Paint, blend_mode: BlendMode, mask: Option<Mask>) {
        let draw_id = self.draw_id();
        let attrs = PaintFillAttrs { paint, blend_mode, mask, draw_id, thread_idx: 0, origin: (0, 0) };
        self.recording.items.push(Item::Fill { geometry, attrs });
    }
}

impl Dispatcher for RecordingDispatcher {
    fn has_layers(&self) -> bool {
        !self.open.is_empty()
    }

    fn fill_path(&mut self, path: &BezPath, fill_rule: Fill, transform: Affine, paint: Paint, blend_mode: BlendMode, aliasing_threshold: Option<u8>, mask: Option<Mask>) {
        if aliasing_threshold.is_some() {
            self.refuse("a fill without anti-aliasing");
        }
        self.fill(Geometry::Path { path: path.clone(), fill_rule, transform }, paint, blend_mode, mask);
    }

    fn stroke_path(&mut self, path: &BezPath, stroke: &Stroke, transform: Affine, paint: Paint, blend_mode: BlendMode, aliasing_threshold: Option<u8>, mask: Option<Mask>) {
        if aliasing_threshold.is_some() {
            self.refuse("a stroke without anti-aliasing");
        }
        let c = transform.as_coeffs();
        let tolerance = 0.25 / c[0].abs().max(c[3].abs()).max(1.);
        vello_common::flatten::expand_stroke(path.iter(), stroke, tolerance, &mut self.stroke_ctx);
        let expanded = self.stroke_ctx.output().clone();
        self.fill(Geometry::Path { path: expanded, fill_rule: Fill::NonZero, transform }, paint, blend_mode, mask);
    }

    fn fill_rect_fast(&mut self, rect: &Rect, paint: Paint, blend_mode: BlendMode, mask: Option<Mask>) {
        self.fill(Geometry::Rect(*rect), paint, blend_mode, mask);
    }

    fn push_clip_path(&mut self, _: &BezPath, _: Fill, _: Affine, _: Option<u8>) {
        self.refuse("a clip path (push_clip_path)");
    }

    fn push_clip_rect(&mut self, _: &Rect) {
        self.refuse("a clip rectangle (push_clip_rect)");
    }

    fn pop_clip(&mut self) {}

    fn push_layer(&mut self, clip_path: Option<&BezPath>, fill_rule: Fill, clip_transform: Affine, blend_mode: BlendMode, opacity: f32, aliasing_threshold: Option<u8>, mask: Option<Mask>, filter_data: Option<FilterData>) {
        if filter_data.is_some() {
            self.refuse("a filter layer");
        }
        if aliasing_threshold.is_some() && clip_path.is_some() {
            self.refuse("a clip without anti-aliasing");
        }
        if blend_mode.is_destructive() {
            self.refuse("a layer of a destructive blend mode");
        }
        let clip = clip_path.map(|p| Geometry::Path { path: p.clone(), fill_rule, transform: clip_transform });
        self.open.push(LayerAttrs { blend_mode, opacity, mask, draw_id: 0 });
        self.recording.items.push(Item::Push { clip });
    }

    fn pop_layer(&mut self) {
        let Some(mut layer) = self.open.pop() else { return };
        layer.draw_id = self.draw_id();
        self.recording.items.push(Item::Pop(layer));
    }

    fn reset(&mut self, width: u16, height: u16) {
        *self = Self::new(width, height);
    }

    fn flush(&mut self) {}

    fn rasterize(&self, _: PixmapMut<'_>, _: u16, _: u16, _: RasterizerSettings, _: &[EncodedPaint], _: &dyn ImageResolver) {}

    fn is_multi_threaded(&self) -> bool {
        false
    }

    fn recording(&self) -> Option<&Recording> {
        Some(&self.recording)
    }
}
