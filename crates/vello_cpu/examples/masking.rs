use vello_cpu::color::palette::css::{BLUE, RED, WHITE};
use vello_cpu::kurbo::Rect;
use vello_cpu::{Mask, Pixmap, RenderContext, Resources};

const SIZE: u16 = 200;

fn main() {
    let mask = {
        let mut mask_ctx = RenderContext::new(SIZE, SIZE);
        let mut mask_resources = Resources::new();
        let mut pixmap = Pixmap::new(SIZE, SIZE);

        mask_ctx.set_paint(RED);
        mask_ctx.fill_rect(&Rect::new(30.0, 30.0, 170.0, 170.0));
        mask_ctx.flush();
        mask_ctx.render(&mut pixmap, &mut mask_resources);

        Mask::new_luminance(&pixmap)
    };

    let mut ctx = RenderContext::new(SIZE, SIZE);

    {
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64));

        ctx.set_mask(mask.clone());
        ctx.set_paint(BLUE);
        ctx.fill_rect(&Rect::new(20.0, 20.0, 130.0, 130.0));
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(70.0, 70.0, 180.0, 180.0));
        ctx.reset_mask();

        ctx.flush();
        save_pixmap(&ctx, "example_masking1");
    }

    ctx.reset();
    {
        ctx.set_paint(WHITE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64));

        ctx.push_mask_layer(mask);
        ctx.set_paint(BLUE);
        ctx.fill_rect(&Rect::new(20.0, 20.0, 130.0, 130.0));
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(70.0, 70.0, 180.0, 180.0));
        ctx.pop_layer();

        ctx.flush();
        save_pixmap(&ctx, "example_masking2");
    }
}

fn save_pixmap(ctx: &RenderContext, filename: &str) {
    let mut resources = Resources::new();
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render(&mut pixmap, &mut resources);
    let png = pixmap.into_png().unwrap();
    std::fs::write(format!("{filename}.png"), png).unwrap();
}
