use super::FilterEffect;
use super::gaussian_blur::apply_blur;
use super::shift::offset_pixels;
use crate::vello_cpu::filter::context::ScratchBuffer;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::filter::drop_shadow::DropShadow;
use vello_common::filter_effects::EdgeMode;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(all()))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

impl FilterEffect for DropShadow {
    fn execute_lowp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        apply_drop_shadow(
            pixmap,
            self.dx,
            self.dy,
            self.std_deviation,
            self.n_decimations,
            &self.kernel[..usize::from(self.kernel_size)],
            self.color,
            self.edge_mode,
            self.composite_original,
            filter_scratch,
        );
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        Self::execute_lowp(self, pixmap, filter_scratch);
    }
}

fn apply_drop_shadow(
    pixmap: &mut Pixmap,
    dx: f32,
    dy: f32,
    std_deviation: f32,
    n_decimations: usize,
    kernel: &[f32],
    color: AlphaColor<Srgb>,
    edge_mode: EdgeMode,
    composite_original: bool,
    filter_scratch: &mut ScratchBuffer,
) {
    let mut shadow_pixmap = pixmap.clone();

    offset_pixels(&mut shadow_pixmap, dx, dy);

    if std_deviation > 0.0 {
        let scratch =
            filter_scratch.get_scratch_buffer(shadow_pixmap.width(), shadow_pixmap.height());
        apply_blur(
            &mut shadow_pixmap,
            scratch,
            n_decimations,
            kernel,
            edge_mode,
        );
    }

    write_colored_shadow(&shadow_pixmap, pixmap, color, composite_original);
}

fn write_colored_shadow(
    shadow: &Pixmap,
    dst: &mut Pixmap,
    color: AlphaColor<Srgb>,
    composite_original: bool,
) {
    let width = dst.width();
    let height = dst.height();

    let shadow_r = (color.components[0] * 255.0).round() as u8;
    let shadow_g = (color.components[1] * 255.0).round() as u8;
    let shadow_b = (color.components[2] * 255.0).round() as u8;

    for y in 0..height {
        for x in 0..width {
            let alpha = shadow.sample(x, y).a;

            let shadow_alpha = (u8_to_norm(alpha) * color.components[3]).min(1.0);
            let final_alpha = norm_to_u8(shadow_alpha);

            let alpha_u16 = u16::from(final_alpha);
            let premultiply = |channel: u8| ((u16::from(channel) * alpha_u16) / 255) as u8;

            let colored_shadow = PremulRgba8 {
                r: premultiply(shadow_r),
                g: premultiply(shadow_g),
                b: premultiply(shadow_b),
                a: final_alpha,
            };

            let result = if composite_original {
                compose_src_over(dst.sample(x, y), colored_shadow)
            } else {
                colored_shadow
            };

            dst.set_pixel(x, y, result);
        }
    }
}

fn compose_src_over(src: PremulRgba8, dst: PremulRgba8) -> PremulRgba8 {
    let src_a = u8_to_norm(src.a);

    PremulRgba8 {
        r: src_over_channel(src.r, dst.r, src_a),
        g: src_over_channel(src.g, dst.g, src_a),
        b: src_over_channel(src.b, dst.b, src_a),
        a: src_over_channel(src.a, dst.a, src_a),
    }
}

#[inline]
fn src_over_channel(src: u8, dst: u8, src_alpha: f32) -> u8 {
    let result = u8_to_norm(src) + u8_to_norm(dst) * (1.0 - src_alpha);
    norm_to_u8(result)
}

#[inline]
fn u8_to_norm(value: u8) -> f32 {
    value as f32 / 255.0
}

#[inline]
fn norm_to_u8(value: f32) -> u8 {
    (value * 255.0).round() as u8
}

#[cfg(any())]
mod tests {
    use super::*;
    use vello_common::color::Srgb;

    #[test]
    fn test_u8_to_norm() {
        assert_eq!(u8_to_norm(0), 0.0);
        assert!((u8_to_norm(255) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_to_u8() {
        assert_eq!(norm_to_u8(0.0), 0);
        assert_eq!(norm_to_u8(1.0), 255);
        assert_eq!(norm_to_u8(0.5), 128);
    }

    #[test]
    fn test_conversion_roundtrip() {
        for value in [0, 1, 50, 127, 128, 200, 254, 255] {
            let normalized = u8_to_norm(value);
            let back = norm_to_u8(normalized);
            assert_eq!(back, value);
        }
    }

    #[test]
    fn test_compose_src_over_opaque_source() {
        let src = PremulRgba8 {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        };
        let dst = PremulRgba8 {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        };

        let result = compose_src_over(src, dst);
        assert_eq!(result.r, 255);
        assert_eq!(result.g, 0);
        assert_eq!(result.b, 0);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn test_compose_src_over_transparent_source() {
        let src = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        let dst = PremulRgba8 {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        };

        let result = compose_src_over(src, dst);
        assert_eq!(result.r, 0);
        assert_eq!(result.g, 255);
        assert_eq!(result.b, 0);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn test_compose_src_over_semi_transparent() {
        let src = PremulRgba8 {
            r: 128,
            g: 0,
            b: 0,
            a: 128,
        };
        let dst = PremulRgba8 {
            r: 0,
            g: 128,
            b: 0,
            a: 128,
        };

        let result = compose_src_over(src, dst);
        assert_eq!(
            result,
            PremulRgba8 {
                r: 128,
                g: 64,
                b: 0,
                a: 192,
            }
        );
    }

    #[test]
    fn test_compose_shadow_color() {
        let mut shadow_pixmap = Pixmap::new(2, 2);
        let mut dst_pixmap = Pixmap::new(2, 2);

        shadow_pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 0,
                g: 0,
                b: 0,
                a: 255,
            },
        );

        let shadow_color = AlphaColor {
            components: [1.0, 0.0, 0.0, 1.0],
            cs: std::marker::PhantomData::<Srgb>,
        };

        write_colored_shadow(&shadow_pixmap, &mut dst_pixmap, shadow_color, true);

        let result = dst_pixmap.sample(0, 0);
        assert_eq!(result.r, 255);
        assert_eq!(result.g, 0);
        assert_eq!(result.b, 0);
        assert_eq!(result.a, 255);
    }
}
