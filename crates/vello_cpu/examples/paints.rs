use std::sync::Arc;
use std::{io::Cursor, path::Path};
use vello_cpu::{
    Image, ImageSource, Pixmap, RenderContext, Resources,
    color::palette::css::{BLUE, CYAN, DEEP_PINK, MAGENTA, NAVY, ORANGE, PURPLE, WHITE, YELLOW},
    kurbo::{Affine, Point, Rect},
    peniko::{
        ColorStop, ColorStops, Extend, Gradient, ImageSampler, LinearGradientPosition,
        RadialGradientPosition, SweepGradientPosition,
    },
};

fn main() {
    let mut ctx = RenderContext::new(500, 500);
    let base_rect = Rect::new(25.0, 25.0, 225.0, 225.0);

    ctx.set_paint(linear_gradient(&base_rect));
    ctx.fill_rect(&base_rect);

    ctx.set_transform(Affine::translate((250.0, 0.0)));
    ctx.set_paint(radial_gradient(&base_rect));
    ctx.fill_rect(&base_rect);

    ctx.set_transform(Affine::translate((0.0, 250.0)));
    ctx.set_paint(sweep_gradient(&base_rect));
    ctx.fill_rect(&base_rect);

    ctx.set_transform(Affine::translate((250.0, 250.0)));
    ctx.set_paint(pattern());
    ctx.set_paint_transform(Affine::scale(15.0));
    ctx.fill_rect(&base_rect);

    ctx.flush();

    save_pixmap(&ctx, "example_paint");
}

fn linear_gradient(rect: &Rect) -> Gradient {
    Gradient {
        kind: LinearGradientPosition {
            start: Point::new(rect.x0, rect.y0),
            end: Point::new(rect.x1, rect.y1),
        }
        .into(),
        stops: color_stops([
            ColorStop::from((0.0, DEEP_PINK)),
            ColorStop::from((0.55, ORANGE)),
            ColorStop::from((1.0, YELLOW)),
        ]),
        extend: Extend::Pad,
        ..Default::default()
    }
}

fn radial_gradient(rect: &Rect) -> Gradient {
    let center = rect.center();
    #[allow(clippy::cast_possible_truncation, reason = "necessary for conversion")]
    let radius = (rect.width().min(rect.height()) * 0.5) as f32;
    Gradient {
        kind: RadialGradientPosition {
            start_center: center,
            start_radius: radius * 0.2,
            end_center: center,
            end_radius: radius,
        }
        .into(),
        stops: color_stops([
            ColorStop::from((0.0, CYAN)),
            ColorStop::from((0.6, BLUE)),
            ColorStop::from((1.0, NAVY)),
        ]),
        extend: Extend::Pad,
        ..Default::default()
    }
}

fn sweep_gradient(rect: &Rect) -> Gradient {
    let center = rect.center();
    Gradient {
        kind: SweepGradientPosition {
            center,
            start_angle: 0.0,
            end_angle: std::f32::consts::PI * 2.0,
        }
        .into(),
        stops: color_stops([
            ColorStop::from((0.0, MAGENTA)),
            ColorStop::from((0.5, PURPLE)),
            ColorStop::from((1.0, WHITE)),
        ]),
        extend: Extend::Pad,
        ..Default::default()
    }
}

fn pattern() -> Image {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../vello_tests/tests/assets/rgb_image_2x2.png");
    let file = std::fs::read(path).unwrap();
    let pixmap = Pixmap::from_png(Cursor::new(file)).unwrap();

    Image {
        image: ImageSource::Pixmap(Arc::new(pixmap)),
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            ..Default::default()
        },
    }
}

fn save_pixmap(ctx: &RenderContext, filename: &str) {
    let mut resources = Resources::new();
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render(&mut pixmap, &mut resources);
    let png = pixmap.into_png().unwrap();
    std::fs::write(format!("{filename}.png"), png).unwrap();
}

fn color_stops(stops: [ColorStop; 3]) -> ColorStops {
    ColorStops::from(stops.as_slice())
}
