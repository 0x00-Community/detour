use vello_cpu::Pixmap;
use vello_cpu::RenderContext;
use vello_cpu::Resources;
use vello_cpu::color::palette::css::{BLUE, RED, WHITE};
use vello_cpu::kurbo::{Circle, Rect, Shape};

fn main() {
    let mut ctx = RenderContext::new(200, 200);

    let clip_1 = Circle::new((75.0, 75.0), 50.0).to_path(0.1);
    let clip_2 = Circle::new((125.0, 75.0), 50.0).to_path(0.1);

    {
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        ctx.push_clip_path(&clip_1.to_path(0.1));
        ctx.push_clip_path(&clip_2.to_path(0.1));
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));
        ctx.pop_clip();
        ctx.set_paint(BLUE.with_alpha(0.2));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        ctx.pop_clip();

        ctx.flush();

        save_pixmap(&ctx, "example_clipping1");
    }

    {
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        ctx.push_clip_layer(&clip_1.to_path(0.1));
        ctx.push_clip_layer(&clip_2.to_path(0.1));
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));
        ctx.pop_layer();
        ctx.set_paint(BLUE.with_alpha(0.2));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 200.0, 200.0));

        ctx.pop_layer();

        ctx.flush();
        save_pixmap(&ctx, "example_clipping2");
    }
}

fn save_pixmap(ctx: &RenderContext, filename: &str) {
    let mut resources = Resources::new();
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render(&mut pixmap, &mut resources);
    let png = pixmap.into_png().unwrap();
    std::fs::write(format!("{filename}.png"), png).unwrap();
}
