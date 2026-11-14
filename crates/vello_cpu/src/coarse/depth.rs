use crate::vello_cpu::span::TileAlignedSpan;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::tile::Tile;

pub(crate) const DEPTH_BUCKET_WIDTH: u16 = 128;
const DEPTH_BUCKET_TILE_WIDTH: u16 = DEPTH_BUCKET_WIDTH / Tile::WIDTH;
const _: () = assert!(
    DEPTH_BUCKET_WIDTH.is_multiple_of(Tile::WIDTH),
    "depth bucket width must be a multiple of tile width"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BucketRange {
    pub(crate) start: u16,
    pub(crate) end: u16,
}

impl BucketRange {
    pub(crate) fn new(start: u16, end: u16) -> Self {
        Self { start, end }
    }

    pub(crate) fn span(self) -> TileAlignedSpan {
        TileAlignedSpan::from_tiles(
            self.start * DEPTH_BUCKET_TILE_WIDTH,
            (self.end - self.start) * DEPTH_BUCKET_TILE_WIDTH,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DepthSegment {
    Regular(TileAlignedSpan),
    Opaque(BucketRange),
}

pub(crate) fn split_opaque_span(span: TileAlignedSpan, mut segment: impl FnMut(DepthSegment)) {
    let x = span.tile_x();
    let end = span.tile_end();
    let aligned_x = x.next_multiple_of(DEPTH_BUCKET_TILE_WIDTH);
    let aligned_end = (end / DEPTH_BUCKET_TILE_WIDTH) * DEPTH_BUCKET_TILE_WIDTH;

    if aligned_x >= aligned_end {
        segment(DepthSegment::Regular(span));

        return;
    }

    if x < aligned_x {
        segment(DepthSegment::Regular(TileAlignedSpan::from_tiles(
            x,
            aligned_x - x,
        )));
    }

    if aligned_x < aligned_end {
        segment(DepthSegment::Opaque(BucketRange::new(
            aligned_x / DEPTH_BUCKET_TILE_WIDTH,
            aligned_end / DEPTH_BUCKET_TILE_WIDTH,
        )));
    }

    if aligned_end < end {
        segment(DepthSegment::Regular(TileAlignedSpan::from_tiles(
            aligned_end,
            end - aligned_end,
        )));
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DepthState {
    bounds: Option<TileAlignedSpan>,
    max_draw_id: u32,
}

impl DepthState {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn include_span(&mut self, span: TileAlignedSpan, draw_id: u32) {
        if let Some(bounds) = &mut self.bounds {
            bounds.extend(span);
        } else {
            self.bounds = Some(span);
        }

        self.max_draw_id = self.max_draw_id.max(draw_id);
    }

    pub(crate) fn can_skip(self, span: TileAlignedSpan, draw_id: u32) -> bool {
        if draw_id >= self.max_draw_id {
            return true;
        }

        let Some(opaque_bounds) = self.bounds else {
            return true;
        };

        let opaque_start = opaque_bounds.tile_x();
        let opaque_end = opaque_bounds.tile_end();
        let x = span.tile_x();
        let end = span.tile_end();

        x >= opaque_end || end <= opaque_start
    }
}

#[derive(Debug)]
pub(crate) struct DepthBuffer {
    data: Vec<u32>,
}

impl DepthBuffer {
    pub(crate) fn new(buffer_width: u16) -> Self {
        Self {
            data: vec![0; usize::from(buffer_width.div_ceil(DEPTH_BUCKET_WIDTH))],
        }
    }

    pub(crate) fn for_each_unset_run(
        &self,
        span: TileAlignedSpan,
        mut f: impl FnMut(TileAlignedSpan),
    ) {
        let (mut idx, depth_end) = self.range(span);
        while let Some((span, _)) = self.next_unset_run(&mut idx, depth_end, span) {
            f(span);
        }
    }

    pub(crate) fn for_each_unset_run_and_write(
        &mut self,
        bucket_range: BucketRange,
        draw_id: u32,
        mut f: impl FnMut(BucketRange),
    ) {
        let bounds = bucket_range.span();
        let mut idx = usize::from(bucket_range.start);
        let depth_end = usize::from(bucket_range.end);

        while let Some((_, depth_range)) = self.next_unset_run(&mut idx, depth_end, bounds) {
            let bucket_start =
                u16::try_from(depth_range.start).expect("depth bucket range start overflow");
            let bucket_end =
                u16::try_from(depth_range.end).expect("depth bucket range end overflow");
            f(BucketRange::new(bucket_start, bucket_end));
            self.mark(depth_range, draw_id);
        }
    }

    pub(crate) fn for_each_visible_run(
        &self,
        span: TileAlignedSpan,
        draw_id: u32,
        mut f: impl FnMut(TileAlignedSpan),
    ) {
        let (mut idx, depth_end) = self.range(span);

        while let Some(span) = self.next_visible_run(&mut idx, depth_end, draw_id, span) {
            f(span);
        }
    }

    fn range(&self, span: TileAlignedSpan) -> (usize, usize) {
        (
            usize::from(span.pixel_x() / DEPTH_BUCKET_WIDTH),
            usize::from(span.pixel_end().div_ceil(DEPTH_BUCKET_WIDTH)).min(self.data.len()),
        )
    }

    pub(crate) fn clear(&mut self) {
        self.data.fill(0);
    }

    fn mark(&mut self, range: Range<usize>, draw_id: u32) {
        self.data[range].fill(draw_id);
    }

    fn next_unset_run(
        &self,
        idx: &mut usize,
        end: usize,
        bounds: TileAlignedSpan,
    ) -> Option<(TileAlignedSpan, Range<usize>)> {
        while *idx < end && self.data[*idx] != 0 {
            *idx += 1;
        }

        let run_start = *idx;
        while *idx < end && self.data[*idx] == 0 {
            *idx += 1;
        }

        if run_start == *idx {
            return None;
        }

        Some((
            bucket_span(run_start, *idx).intersect(bounds)?,
            run_start..*idx,
        ))
    }

    fn next_visible_run(
        &self,
        idx: &mut usize,
        end: usize,
        draw_id: u32,
        bounds: TileAlignedSpan,
    ) -> Option<TileAlignedSpan> {
        while *idx < end && self.data[*idx] > draw_id {
            *idx += 1;
        }

        let run_start = *idx;
        while *idx < end && self.data[*idx] <= draw_id {
            *idx += 1;
        }

        if run_start == *idx {
            return None;
        }

        bucket_span(run_start, *idx).intersect(bounds)
    }
}

fn bucket_span(start: usize, end: usize) -> TileAlignedSpan {
    BucketRange::new(start as u16, end as u16).span()
}

#[cfg(any())]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::ops::Range;

    fn buffer(bucket_count: usize) -> DepthBuffer {
        DepthBuffer::new(bucket_count as u16 * DEPTH_BUCKET_WIDTH)
    }

    fn buckets(start: usize, end: usize) -> TileAlignedSpan {
        bucket_span(start, end)
    }

    fn bucket_range(span: TileAlignedSpan) -> (usize, usize) {
        (
            usize::from(span.pixel_x() / DEPTH_BUCKET_WIDTH),
            usize::from(span.pixel_end() / DEPTH_BUCKET_WIDTH),
        )
    }

    fn visible_runs(
        buffer: &DepthBuffer,
        span: TileAlignedSpan,
        draw_id: u32,
    ) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        buffer.for_each_visible_run(span, draw_id, |span| {
            runs.push(bucket_range(span));
        });
        runs
    }

    fn unset_runs(buffer: &DepthBuffer, span: TileAlignedSpan) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        buffer.for_each_unset_run(span, |span| {
            runs.push(bucket_range(span));
        });
        runs
    }

    fn write_buckets(buffer: &mut DepthBuffer, range: Range<usize>, draw_id: u32) {
        buffer.for_each_unset_run_and_write(
            BucketRange::new(range.start as u16, range.end as u16),
            draw_id,
            |_| {},
        );
    }

    fn assert_depth(buffer: &DepthBuffer, ranges: &[(Range<usize>, u32)]) {
        let mut expected = vec![0; buffer.data.len()];
        for (range, draw_id) in ranges {
            expected[range.clone()].fill(*draw_id);
        }

        assert_eq!(buffer.data, expected);
    }

    #[test]
    fn split_opaque_span_extracts_aligned_middle() {
        let mut segments = Vec::new();
        split_opaque_span(
            TileAlignedSpan::from_tiles(1, DEPTH_BUCKET_TILE_WIDTH * 3),
            |segment| {
                segments.push(segment);
            },
        );

        assert_eq!(
            segments,
            [
                DepthSegment::Regular(TileAlignedSpan::from_tiles(1, DEPTH_BUCKET_TILE_WIDTH - 1)),
                DepthSegment::Opaque(BucketRange::new(1, 3)),
                DepthSegment::Regular(TileAlignedSpan::from_tiles(DEPTH_BUCKET_TILE_WIDTH * 3, 1)),
            ]
        );
    }

    #[test]
    fn depth_state_skips_when_no_later_overlapping_opaque_draw_exists() {
        let mut state = DepthState::default();
        let opaque = buckets(1, 2);
        state.include_span(opaque, 7);

        assert!(state.can_skip(buckets(0, 1), 1));
        assert!(state.can_skip(opaque, 7));
        assert!(!state.can_skip(opaque, 6));

        state.reset();
        assert!(state.can_skip(opaque, 1));
    }

    #[test]
    fn visible_runs_skip_interleaved_later_draws() {
        let mut buffer = buffer(5);
        write_buckets(&mut buffer, 1..2, 10);
        write_buckets(&mut buffer, 3..4, 10);

        assert_eq!(
            visible_runs(&buffer, buckets(0, 5), 9),
            [(0, 1), (2, 3), (4, 5)]
        );
        assert_eq!(visible_runs(&buffer, buckets(0, 5), 10), [(0, 5)]);
    }

    #[test]
    fn unset_runs_and_writes_fill_interleaved_gaps() {
        let mut buffer = buffer(5);
        write_buckets(&mut buffer, 1..2, 10);
        write_buckets(&mut buffer, 3..4, 10);

        assert_eq!(unset_runs(&buffer, buckets(0, 5)), [(0, 1), (2, 3), (4, 5)]);

        let mut written_runs = Vec::new();
        buffer.for_each_unset_run_and_write(BucketRange::new(0, 5), 7, |range| {
            written_runs.push((usize::from(range.start), usize::from(range.end)));
        });
        assert_eq!(written_runs, [(0, 1), (2, 3), (4, 5)]);
        assert_depth(
            &buffer,
            [(0..1, 7), (1..2, 10), (2..3, 7), (3..4, 10), (4..5, 7)].as_slice(),
        );
    }

    #[test]
    fn visible_and_unset_runs_are_limited_to_the_requested_span() {
        let mut buffer = buffer(6);
        write_buckets(&mut buffer, 1..2, 10);
        write_buckets(&mut buffer, 4..5, 10);

        assert_eq!(visible_runs(&buffer, buckets(2, 5), 9), [(2, 4)]);
        assert_eq!(unset_runs(&buffer, buckets(2, 5)), [(2, 4)]);
    }

    #[test]
    fn unset_runs_clip_to_span_not_aligned_to_depth_buckets() {
        let buffer = buffer(3);
        let span = TileAlignedSpan::from_tiles(2, DEPTH_BUCKET_TILE_WIDTH + 3);
        let mut runs = Vec::new();

        buffer.for_each_unset_run(span, |span| {
            runs.push((span.pixel_x(), span.pixel_end()));
        });

        assert_eq!(
            runs,
            [(Tile::WIDTH * 2, DEPTH_BUCKET_WIDTH + Tile::WIDTH * 5)]
        );
    }

    #[test]
    fn visible_runs_only_skip_buckets_with_later_draw_ids() {
        let mut buffer = buffer(6);
        write_buckets(&mut buffer, 0..1, 4);
        write_buckets(&mut buffer, 1..2, 9);
        write_buckets(&mut buffer, 2..3, 6);
        write_buckets(&mut buffer, 3..4, 12);
        write_buckets(&mut buffer, 5..6, 2);

        assert_eq!(
            visible_runs(&buffer, buckets(0, 6), 6),
            [(0, 1), (2, 3), (4, 6)]
        );
    }

    #[test]
    fn clear_resets_all_buckets() {
        let mut buffer = buffer(3);
        write_buckets(&mut buffer, 0..3, 10);

        buffer.clear();

        assert_depth(&buffer, &[]);
    }
}
