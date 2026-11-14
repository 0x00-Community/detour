//! vello_cpu's coarse rasterization as detour takes it (crates/vello_cpu, `RenderContext::coarse`): a scene of a
//! filled circle, a rectangle under a semi-transparent layer clipped to a rounded rectangle, and a linear gradient;
//! printed row by row (the CPU half of R3: what the GPU's fine rasterization will run). No GPU.

use detour::vello_cpu::detour::{Cmd, Coarse};
use detour::vello_cpu::color::{AlphaColor, Srgb};
use detour::vello_cpu::kurbo::{Circle, Point, Rect, RoundedRect, Shape};
use detour::vello_cpu::peniko::Gradient;
use detour::vello_cpu::{RenderContext, Resources};

fn main() {
    let mut ctx = RenderContext::new(256, 64);
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(255, 128, 0, 255));
    ctx.fill_path(&Circle::new((32.0, 32.0), 24.5).to_path(0.1));
    ctx.push_clip_layer(&RoundedRect::new(70.0, 8.0, 150.0, 56.0, 10.0).to_path(0.1));
    ctx.push_opacity_layer(0.5);
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(0, 64, 255, 255));
    ctx.fill_rect(&Rect::new(60.0, 0.0, 160.0, 64.0));
    ctx.pop_layer();
    ctx.pop_layer();
    let g = Gradient::new_linear(Point::new(170.0, 0.0), Point::new(250.0, 0.0)).with_stops([AlphaColor::<Srgb>::from_rgba8(255, 0, 0, 255), AlphaColor::<Srgb>::from_rgba8(0, 0, 255, 255)].as_slice());
    ctx.set_paint(g);
    ctx.fill_rect(&Rect::new(170.0, 4.0, 250.0, 60.0));

    let mut resources = Resources::new();
    let c: Coarse<'_> = ctx.coarse(&mut resources).expect("single threaded");
    println!(
        "{}x{}: {} rows, {} paint attrs, {} layer attrs, {} alpha bytes, {} encoded paints, {} filter paints, root a blend target: {}",
        c.width,
        c.height,
        c.rows.len(),
        c.paints.len(),
        c.layers.len(),
        c.alphas.len(),
        c.encoded_paints.len(),
        c.filter_paints.len(),
        c.root_is_blend_target
    );
    for (y, row) in c.rows.iter().enumerate().filter(|(y, _)| y % 4 == 0) {
        let cmds: Vec<String> = row
            .cmds
            .iter()
            .map(|cmd| match cmd {
                Cmd::PaintFill { span, alpha, attrs } => format!("fill[{}+{} a{:?} #{attrs}]", span.x, span.width, alpha),
                Cmd::PushBuf(s) => format!("push{:?}", s.map(|s| (s.x, s.width))),
                Cmd::PopBuf => "pop".into(),
                Cmd::LayerFill { span, alpha, attrs } => format!("layer[{}+{} a{:?} #{attrs}]", span.x, span.width, alpha),
            })
            .collect();
        let depth: Vec<String> = row.depth.iter().map(|d| format!("{:?} #{}", d.buckets, d.attrs)).collect();
        println!("row {y:2}: depth {depth:?}; {}", cmds.join(" "));
    }
    assert!(c.rows.iter().any(|r| r.cmds.iter().any(|c| matches!(c, Cmd::PushBuf(_)))), "the layer's commands");
    println!("ok");
}
