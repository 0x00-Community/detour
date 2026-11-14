use vello_common::color::palette::css::TRANSPARENT;
#[cfg(not(all()))]
use vello_common::kurbo::common::FloatFuncs as _;
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;

pub(crate) fn offset_pixels(pixmap: &mut Pixmap, dx: f32, dy: f32) {
    let dx_pixels = dx.round() as i32;
    let dy_pixels = dy.round() as i32;

    if dx_pixels == 0 && dy_pixels == 0 {
        return;
    }

    let width = pixmap.width();
    let height = pixmap.height();
    let transparent = TRANSPARENT.premultiply().to_rgba8();

    match (dx_pixels >= 0, dy_pixels >= 0) {
        (true, true) => {
            for y in (0..height).rev() {
                for x in (0..width).rev() {
                    process_offset_pixel(
                        pixmap,
                        x,
                        y,
                        dx_pixels,
                        dy_pixels,
                        width,
                        height,
                        transparent,
                    );
                }
            }
        }
        (true, false) => {
            for y in 0..height {
                for x in (0..width).rev() {
                    process_offset_pixel(
                        pixmap,
                        x,
                        y,
                        dx_pixels,
                        dy_pixels,
                        width,
                        height,
                        transparent,
                    );
                }
            }
        }
        (false, true) => {
            for y in (0..height).rev() {
                for x in 0..width {
                    process_offset_pixel(
                        pixmap,
                        x,
                        y,
                        dx_pixels,
                        dy_pixels,
                        width,
                        height,
                        transparent,
                    );
                }
            }
        }
        (false, false) => {
            for y in 0..height {
                for x in 0..width {
                    process_offset_pixel(
                        pixmap,
                        x,
                        y,
                        dx_pixels,
                        dy_pixels,
                        width,
                        height,
                        transparent,
                    );
                }
            }
        }
    }
}

#[inline(always)]
fn process_offset_pixel(
    pixmap: &mut Pixmap,
    x: u16,
    y: u16,
    dx_pixels: i32,
    dy_pixels: i32,
    width: u16,
    height: u16,
    transparent: PremulRgba8,
) {
    let new_x = x as i32 + dx_pixels;
    let new_y = y as i32 + dy_pixels;

    if new_x >= 0 && new_x < width as i32 && new_y >= 0 && new_y < height as i32 {
        let pixel = pixmap.sample(x, y);
        pixmap.set_pixel(new_x as u16, new_y as u16, pixel);
    }

    let should_clear = (dx_pixels > 0 && x < dx_pixels as u16)
        || (dx_pixels < 0 && x >= (width as i32 + dx_pixels) as u16)
        || (dy_pixels > 0 && y < dy_pixels as u16)
        || (dy_pixels < 0 && y >= (height as i32 + dy_pixels) as u16);

    if should_clear {
        pixmap.set_pixel(x, y, transparent);
    }
}

#[cfg(any())]
mod tests {
    use super::*;

    #[test]
    fn test_offset_pixels_positive() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.set_pixel(
            1,
            1,
            PremulRgba8 {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
        );

        offset_pixels(&mut pixmap, 1.0, 1.0);

        let moved = pixmap.sample(2, 2);
        assert_eq!(moved.a, 255);

        let cleared = pixmap.sample(1, 1);
        assert_eq!(cleared.a, 0);
    }

    #[test]
    fn test_offset_pixels_negative() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.set_pixel(
            2,
            2,
            PremulRgba8 {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
        );

        offset_pixels(&mut pixmap, -1.0, -1.0);

        let moved = pixmap.sample(1, 1);
        assert_eq!(moved.a, 255);

        let cleared = pixmap.sample(2, 2);
        assert_eq!(cleared.a, 0);
    }

    #[test]
    fn test_offset_pixels_fractional() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.set_pixel(
            1,
            1,
            PremulRgba8 {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
        );

        offset_pixels(&mut pixmap, 0.6, -0.4);

        let moved = pixmap.sample(2, 1);
        assert_eq!(moved.a, 255);
    }

    #[test]
    fn test_offset_pixels_out_of_bounds() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.set_pixel(
            1,
            1,
            PremulRgba8 {
                r: 255,
                g: 255,
                b: 255,
                a: 255,
            },
        );

        offset_pixels(&mut pixmap, 10.0, 10.0);

        let cleared = pixmap.sample(1, 1);
        assert_eq!(cleared.a, 0);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(pixmap.sample(x, y).a, 0);
            }
        }
    }
}
