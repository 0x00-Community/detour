pub(crate) const COST_THRESHOLD: f32 = 250.0;

use crate::vello_cpu::dispatch::multi_threaded::RenderTaskType;
use crate::vello_cpu::kurbo::{Affine, PathEl, PathSeg, Point, segments};

pub(crate) fn estimate_render_task_cost(task: &RenderTaskType, paths: &[PathEl]) -> f32 {
    const LAYER_COST: f32 = 10.0;

    match task {
        RenderTaskType::FillRect { rect, .. } => {
            const FAST_PATH_SPEEDUP: f32 = 5.0;

            let path_length = (2.0 * (rect.width().abs() + rect.height().abs())) as f32;
            path_length_adjusted_cost(4.0, path_length) / FAST_PATH_SPEEDUP
        }
        RenderTaskType::FillPath {
            path_range,
            transform,
            ..
        } => {
            let path = &paths[path_range.start as usize..path_range.end as usize];
            estimate_path_cost(segments(path.iter().copied()), *transform, false)
        }
        RenderTaskType::StrokePath {
            path_range,
            transform,
            ..
        } => {
            let path = &paths[path_range.start as usize..path_range.end as usize];
            estimate_path_cost(segments(path.iter().copied()), *transform, true)
        }
        RenderTaskType::PushLayer { clip_path, .. } => {
            LAYER_COST
                + clip_path
                    .as_ref()
                    .map(|(path_range, transform)| {
                        let path = &paths[path_range.start as usize..path_range.end as usize];
                        estimate_path_cost(segments(path.iter().copied()), *transform, false)
                    })
                    .unwrap_or(0.0)
        }
        RenderTaskType::PopLayer => LAYER_COST,
    }
}

pub(crate) fn estimate_path_cost(
    path: impl IntoIterator<Item = PathSeg>,
    transform: Affine,
    is_stroke: bool,
) -> f32 {
    let cost_data = PathCostData::new(path, transform);

    const CURVE_MULTIPLIER: f32 = 2.5;
    const STROKE_MULTIPLIER: f32 = 1.5;

    let mut cost = cost_data.num_line_segments as f32;
    cost += cost_data.num_curve_segments as f32 * CURVE_MULTIPLIER;
    cost = path_length_adjusted_cost(cost, cost_data.path_length as f32);

    cost *= if is_stroke { STROKE_MULTIPLIER } else { 1.0 };
    cost
}

fn path_length_adjusted_cost(cost: f32, path_length: f32) -> f32 {
    cost + cost * (path_length / 1024.0)
}

struct PathCostData {
    num_line_segments: u64,
    num_curve_segments: u64,
    path_length: f64,
}

impl PathCostData {
    fn new(path: impl IntoIterator<Item = PathSeg>, transform: Affine) -> Self {
        let mut num_line_segments = 0;
        let mut num_curve_segments = 0;
        let mut path_length = 0.0;

        let mut register_path_length = |mut p0: Point, mut p1: Point| {
            p0 = transform * p0;
            p1 = transform * p1;
            let dx = (p1.x - p0.x).abs();
            let dy = (p1.y - p0.y).abs();
            path_length += dx + dy;
        };

        for seg in path.into_iter() {
            match seg {
                PathSeg::Line(l) => {
                    num_line_segments += 1;

                    register_path_length(l.p0, l.p1);
                }
                PathSeg::Quad(q) => {
                    num_curve_segments += 1;

                    register_path_length(q.p0, q.p2);
                }
                PathSeg::Cubic(c) => {
                    num_curve_segments += 1;

                    register_path_length(c.p0, c.p3);
                }
            }
        }

        Self {
            num_line_segments,
            num_curve_segments,
            path_length,
        }
    }
}
