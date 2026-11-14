use vello_cpu::color::palette::css::YELLOW;
use vello_cpu::kurbo::Affine;
use vello_cpu::{
    Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
    color::palette::css::{BLUE, GREEN, RED},
    kurbo::{Circle, Rect, Shape},
};

fn main() {
    let settings = RenderSettings {
        level: Level::new(),
        num_threads: 0,
    };
    let rasterizer_settings = RasterizerSettings {
        render_mode: RenderMode::OptimizeSpeed,
        ..Default::default()
    };

    let mut ctx = RenderContext::new_with(100, 100, settings);
    let mut resources = Resources::new();

    ctx.set_paint(BLUE);
    ctx.fill_path(&Rect::new(25.0, 25.0, 75.0, 75.0).to_path(0.1));

    ctx.set_paint(RED.with_alpha(0.5));
    ctx.fill_rect(&Rect::new(50.0, 50.0, 85.0, 85.0));

    ctx.set_paint(GREEN);
    ctx.stroke_path(&Circle::new((50.0, 50.0), 30.0).to_path(0.1));

    ctx.flush();

    let mut pixmap_1 = Pixmap::new(100, 100);
    ctx.render_with(&mut pixmap_1, &mut resources, rasterizer_settings);

    let png_1 = pixmap_1.into_png().unwrap();
    std::fs::write("example_basic1.png", png_1).unwrap();

    ctx.reset();

    ctx.set_transform(Affine::scale(3.0));
    ctx.set_paint(YELLOW);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 20.0, 20.0));
    ctx.flush();

    let mut pixmap_2 = Pixmap::new(100, 100);
    ctx.render_with(&mut pixmap_2, &mut resources, rasterizer_settings);
    let png_2 = pixmap_2.into_png().unwrap();
    std::fs::write("example_basic2.png", png_2).unwrap();
}
