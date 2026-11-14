pub(crate) mod context;
mod drop_shadow;
mod flood;
mod gaussian_blur;
mod offset;
mod shift;

use context::ScratchBuffer;
use vello_common::filter::PreparedFilter;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
use vello_common::pixmap::Pixmap;

pub(crate) trait FilterEffect {
    fn execute_lowp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer);

    fn execute_highp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer);
}

pub(crate) fn filter_lowp(
    filter: &Filter,
    pixmap: &mut Pixmap,
    filter_scratch: &mut ScratchBuffer,
    transform: Affine,
) {
    let prepared_filter = PreparedFilter::new(filter, &transform);

    match prepared_filter {
        PreparedFilter::Flood(flood) => {
            flood.execute_lowp(pixmap, filter_scratch);
        }
        PreparedFilter::GaussianBlur(blur) => {
            blur.execute_lowp(pixmap, filter_scratch);
        }
        PreparedFilter::Offset(offset) => {
            offset.execute_lowp(pixmap, filter_scratch);
        }
        PreparedFilter::DropShadow(drop_shadow) => {
            drop_shadow.execute_lowp(pixmap, filter_scratch);
        }
    }
}

pub(crate) fn filter_highp(
    filter: &Filter,
    pixmap: &mut Pixmap,
    filter_scratch: &mut ScratchBuffer,
    transform: Affine,
) {
    let prepared_filter = PreparedFilter::new(filter, &transform);

    match prepared_filter {
        PreparedFilter::Flood(flood) => {
            flood.execute_highp(pixmap, filter_scratch);
        }
        PreparedFilter::GaussianBlur(blur) => {
            blur.execute_highp(pixmap, filter_scratch);
        }
        PreparedFilter::Offset(offset) => {
            offset.execute_highp(pixmap, filter_scratch);
        }
        PreparedFilter::DropShadow(drop_shadow) => {
            drop_shadow.execute_highp(pixmap, filter_scratch);
        }
    }
}
