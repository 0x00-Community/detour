use crate::vello_cpu::coarse::depth::BucketRange;
use crate::vello_cpu::peniko::BlendMode;
use crate::vello_cpu::span::TileAlignedSpan;
use core::num::NonZeroU32;
use vello_common::mask::Mask;
use vello_common::paint::Paint;

#[derive(Debug, Clone, Copy)]
pub(crate) enum RenderCmd {
    PaintFill(PaintFill),
    PushBuf(Option<TileAlignedSpan>),
    PopBuf,
    LayerFill(LayerFill),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PaintFill {
    pub(crate) span: TileAlignedSpan,
    alpha_idx: Option<AlphaIdx>,
    pub(crate) attrs_idx: u32,
}

impl PaintFill {
    pub(crate) fn new(span: TileAlignedSpan, alpha_idx: Option<u32>, attrs_idx: u32) -> Self {
        Self {
            span,
            alpha_idx: alpha_idx.map(AlphaIdx::new),
            attrs_idx,
        }
    }

    pub(crate) fn alpha_idx(self) -> Option<u32> {
        self.alpha_idx.map(AlphaIdx::get)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DepthFill {
    bucket_range: BucketRange,
    pub(crate) attrs_idx: u32,
}

impl DepthFill {
    pub(crate) fn new(bucket_range: BucketRange, attrs_idx: u32) -> Self {
        Self {
            bucket_range,
            attrs_idx,
        }
    }

    pub(crate) fn bucket_range(self) -> BucketRange {
        self.bucket_range
    }

    pub(crate) fn span(self) -> TileAlignedSpan {
        self.bucket_range.span()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerFill {
    pub(crate) span: TileAlignedSpan,
    alpha_idx: Option<AlphaIdx>,
    pub(crate) attrs_idx: u32,
}

impl LayerFill {
    pub(crate) fn new(span: TileAlignedSpan, alpha_idx: Option<u32>, attrs_idx: u32) -> Self {
        Self {
            span,
            alpha_idx: alpha_idx.map(AlphaIdx::new),
            attrs_idx,
        }
    }

    pub(crate) fn alpha_idx(self) -> Option<u32> {
        self.alpha_idx.map(AlphaIdx::get)
    }
}

#[derive(Debug, Clone, Copy)]
struct AlphaIdx(NonZeroU32);

impl AlphaIdx {
    fn new(alpha_idx: u32) -> Self {
        Self(NonZeroU32::new(alpha_idx.checked_add(1).expect("alpha index overflow")).unwrap())
    }

    fn get(self) -> u32 {
        self.0.get() - 1
    }
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct PaintFillAttrs {
    pub paint: Paint,
    pub blend_mode: BlendMode,
    pub mask: Option<Mask>,
    pub draw_id: u32,
    pub thread_idx: u8,
    pub origin: (u16, u16),
}

#[derive(Debug, Clone)]
pub(crate) struct LayerFillAttrs {
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
    pub(crate) mask: Option<Mask>,
    pub(crate) draw_id: u32,
    pub(crate) thread_idx: u8,
}

#[cfg(any())]
mod tests {
    use super::{AlphaIdx, RenderCmd};
    use core::mem::{needs_drop, size_of};

    #[test]
    fn alpha_idx_size() {
        assert_eq!(size_of::<Option<AlphaIdx>>(), 4);
    }

    #[test]
    fn render_cmd_assertions() {
        assert_eq!(size_of::<RenderCmd>(), 16);
        assert!(!needs_drop::<RenderCmd>());
    }
}
