use super::FilterEffect;
use crate::vello_cpu::filter::context::ScratchBuffer;
use vello_common::filter::gaussian_blur::{DecimationSizer, GaussianBlur};
use vello_common::filter_effects::EdgeMode;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(all()))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

impl FilterEffect for GaussianBlur {
    fn execute_lowp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        if self.std_deviation <= 0.0 {
            return;
        }

        let scratch = filter_scratch.get_scratch_buffer(pixmap.width(), pixmap.height());
        apply_blur(
            pixmap,
            scratch,
            self.n_decimations,
            &self.kernel[..usize::from(self.kernel_size)],
            self.edge_mode,
        );
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        Self::execute_lowp(self, pixmap, filter_scratch);
    }
}

pub(crate) fn apply_blur(
    pixmap: &mut Pixmap,
    scratch: &mut Pixmap,
    n_decimations: usize,
    kernel: &[f32],
    edge_mode: EdgeMode,
) {
    let radius = (kernel.len() / 2) as u8;
    let width = pixmap.width();
    let height = pixmap.height();

    if n_decimations == 0 {
        convolve(pixmap, scratch, width, height, kernel, radius, edge_mode);
        return;
    }

    let mut sizer = DecimationSizer::new(width, height);

    for _ in 0..n_decimations {
        let (w, h) = sizer.current();
        downscale(pixmap, w, h, edge_mode);
        sizer.downscale();
    }

    let (w, h) = sizer.current();
    convolve(pixmap, scratch, w, h, kernel, radius, edge_mode);

    for _ in 0..n_decimations {
        let (w, h) = sizer.current();
        upscale(pixmap, w, h, edge_mode);
        sizer.upscale();
    }

    debug_assert_eq!(
        sizer.current(),
        (width, height),
        "Final dimensions should match original"
    );
}

pub(crate) fn convolve(
    src: &mut Pixmap,
    scratch: &mut Pixmap,
    width: u16,
    height: u16,
    kernel: &[f32],
    radius: u8,
    edge_mode: EdgeMode,
) {
    convolve_x(src, scratch, width, height, kernel, radius, edge_mode);
    convolve_y(scratch, src, width, height, kernel, radius, edge_mode);
}

pub(crate) fn convolve_x(
    src: &Pixmap,
    dst: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    kernel: &[f32],
    radius: u8,
    edge_mode: EdgeMode,
) {
    for y in 0..src_height {
        for x in 0..src_width {
            let mut rgba = [0.0_f32; 4];

            for (j, &k) in kernel.iter().enumerate() {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "This cast never wraps because `kernel.len()` is never greater than `u8::MAX` due to the restriction on `MAX_KERNEL_SIZE`"
                )]
                let j = j as i32;
                let src_x = x as i32 + j - radius as i32;
                let p = sample_x(src, src_x, y, src_width, edge_mode);

                rgba[0] += p.r as f32 * k;
                rgba[1] += p.g as f32 * k;
                rgba[2] += p.b as f32 * k;
                rgba[3] += p.a as f32 * k;
            }

            dst.set_pixel(
                x,
                y,
                PremulRgba8 {
                    r: rgba[0].round() as u8,
                    g: rgba[1].round() as u8,
                    b: rgba[2].round() as u8,
                    a: rgba[3].round() as u8,
                },
            );
        }
    }
}

pub(crate) fn convolve_y(
    src: &Pixmap,
    dst: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    kernel: &[f32],
    radius: u8,
    edge_mode: EdgeMode,
) {
    for y in 0..src_height {
        for x in 0..src_width {
            let mut rgba = [0.0_f32; 4];

            for (j, &k) in kernel.iter().enumerate() {
                #[expect(
                    clippy::cast_possible_wrap,
                    reason = "This cast never wraps because `kernel.len()` is never greater than `u8::MAX` due to the restriction on `MAX_KERNEL_SIZE`"
                )]
                let j = j as i32;
                let src_y = y as i32 + j - radius as i32;
                let p = sample_y(src, x, src_y, src_height, edge_mode);

                rgba[0] += p.r as f32 * k;
                rgba[1] += p.g as f32 * k;
                rgba[2] += p.b as f32 * k;
                rgba[3] += p.a as f32 * k;
            }

            dst.set_pixel(
                x,
                y,
                PremulRgba8 {
                    r: rgba[0].round() as u8,
                    g: rgba[1].round() as u8,
                    b: rgba[2].round() as u8,
                    a: rgba[3].round() as u8,
                },
            );
        }
    }
}

pub(crate) fn downscale(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    let dst_width = src_width.div_ceil(2);
    let dst_height = src_height.div_ceil(2);
    downscale_x(src, src_width, src_height, dst_width, edge_mode);
    downscale_y(src, dst_width, src_height, dst_height, edge_mode);
    (dst_width, dst_height)
}

fn downscale_x(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    dst_width: u16,
    edge_mode: EdgeMode,
) {
    for y in 0..src_height {
        let mut p0 = sample_x(src, -1, y, src_width, edge_mode);
        let mut p1 = sample_x(src, 0, y, src_width, edge_mode);

        for x in 0..dst_width {
            let src_x = (x * 2) as i32;
            let p2 = sample_x(src, src_x + 1, y, src_width, edge_mode);
            let p3 = sample_x(src, src_x + 2, y, src_width, edge_mode);

            src.set_pixel(x, y, decimate_weighted(p0, p1, p2, p3));

            p0 = p2;
            p1 = p3;
        }
    }
}

fn downscale_y(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    dst_height: u16,
    edge_mode: EdgeMode,
) {
    for x in 0..src_width {
        let mut p0 = sample_y(src, x, -1, src_height, edge_mode);
        let mut p1 = sample_y(src, x, 0, src_height, edge_mode);

        for y in 0..dst_height {
            let src_y = (y * 2) as i32;
            let p2 = sample_y(src, x, src_y + 1, src_height, edge_mode);
            let p3 = sample_y(src, x, src_y + 2, src_height, edge_mode);

            src.set_pixel(x, y, decimate_weighted(p0, p1, p2, p3));

            p0 = p2;
            p1 = p3;
        }
    }
}

pub(crate) fn upscale(
    src: &mut Pixmap,
    src_width: u16,
    src_height: u16,
    edge_mode: EdgeMode,
) -> (u16, u16) {
    let dst_width = src_width * 2;
    let dst_height = src_height * 2;
    upscale_x(src, src_width, src_height, edge_mode);
    upscale_y(src, dst_width, src_height, edge_mode);
    (dst_width, dst_height)
}

fn upscale_x(src: &mut Pixmap, src_width: u16, src_height: u16, edge_mode: EdgeMode) {
    for y in 0..src_height {
        let mut p0 = sample_x(src, src_width as i32, y, src_width, edge_mode);
        let mut p1 = sample_x(src, src_width as i32 - 1, y, src_width, edge_mode);

        for x in (0..src_width).rev() {
            let src_x = x as i32;
            let p2 = sample_x(src, src_x - 1, y, src_width, edge_mode);

            let dst_x = x * 2;
            src.set_pixel(dst_x, y, interpolate_25_75(p2, p1));
            src.set_pixel(dst_x + 1, y, interpolate_75_25(p1, p0));

            p0 = p1;
            p1 = p2;
        }
    }
}

fn upscale_y(src: &mut Pixmap, src_width: u16, src_height: u16, edge_mode: EdgeMode) {
    for x in 0..src_width {
        let mut p0 = sample_y(src, x, src_height as i32, src_height, edge_mode);
        let mut p1 = sample_y(src, x, src_height as i32 - 1, src_height, edge_mode);

        for y in (0..src_height).rev() {
            let src_y = y as i32;
            let p2 = sample_y(src, x, src_y - 1, src_height, edge_mode);

            let dst_y = y * 2;
            src.set_pixel(x, dst_y, interpolate_25_75(p2, p1));
            src.set_pixel(x, dst_y + 1, interpolate_75_25(p1, p0));

            p0 = p1;
            p1 = p2;
        }
    }
}

const TRANSPARENT_BLACK: PremulRgba8 = PremulRgba8 {
    r: 0,
    g: 0,
    b: 0,
    a: 0,
};

#[inline(always)]
fn sample_x(src: &Pixmap, x: i32, y: u16, width: u16, edge_mode: EdgeMode) -> PremulRgba8 {
    sample(x, width, edge_mode, |src_x| src.sample(src_x, y))
}

#[inline(always)]
fn sample_y(src: &Pixmap, x: u16, y: i32, height: u16, edge_mode: EdgeMode) -> PremulRgba8 {
    sample(y, height, edge_mode, |src_y| src.sample(x, src_y))
}

#[inline(always)]
fn sample<F>(coord: i32, size: u16, edge_mode: EdgeMode, sample_fn: F) -> PremulRgba8
where
    F: FnOnce(u16) -> PremulRgba8,
{
    if edge_mode == EdgeMode::None && (coord < 0 || coord >= size as i32) {
        return TRANSPARENT_BLACK;
    }
    let extended_coord = extend(coord, size, edge_mode);
    sample_fn(extended_coord)
}

#[inline(always)]
fn extend(coord: i32, size: u16, edge_mode: EdgeMode) -> u16 {
    match edge_mode {
        EdgeMode::Duplicate => {
            coord.clamp(0, size as i32 - 1) as u16
        }
        EdgeMode::None => {
            coord as u16
        }
        EdgeMode::Wrap => {
            let mut c = coord % size as i32;
            if c < 0 {
                c += size as i32;
            }
            c as u16
        }
        EdgeMode::Mirror => {
            let period = size as i32 * 2;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= size as i32 {
                c = period - c - 1;
            }
            c as u16
        }
    }
}

#[inline(always)]
fn decimate_weighted(
    p0: PremulRgba8,
    p1: PremulRgba8,
    p2: PremulRgba8,
    p3: PremulRgba8,
) -> PremulRgba8 {
    let r = ((p0.r as u32 + p1.r as u32 * 3 + p2.r as u32 * 3 + p3.r as u32 + 4) >> 3) as u8;
    let g = ((p0.g as u32 + p1.g as u32 * 3 + p2.g as u32 * 3 + p3.g as u32 + 4) >> 3) as u8;
    let b = ((p0.b as u32 + p1.b as u32 * 3 + p2.b as u32 * 3 + p3.b as u32 + 4) >> 3) as u8;
    let a = ((p0.a as u32 + p1.a as u32 * 3 + p2.a as u32 * 3 + p3.a as u32 + 4) >> 3) as u8;
    PremulRgba8 { r, g, b, a }
}

#[inline(always)]
fn interpolate_25_75(p0: PremulRgba8, p1: PremulRgba8) -> PremulRgba8 {
    let r = ((p0.r as u32 + p1.r as u32 * 3 + 2) >> 2) as u8;
    let g = ((p0.g as u32 + p1.g as u32 * 3 + 2) >> 2) as u8;
    let b = ((p0.b as u32 + p1.b as u32 * 3 + 2) >> 2) as u8;
    let a = ((p0.a as u32 + p1.a as u32 * 3 + 2) >> 2) as u8;
    PremulRgba8 { r, g, b, a }
}

#[inline(always)]
fn interpolate_75_25(p0: PremulRgba8, p1: PremulRgba8) -> PremulRgba8 {
    let r = ((p0.r as u32 * 3 + p1.r as u32 + 2) >> 2) as u8;
    let g = ((p0.g as u32 * 3 + p1.g as u32 + 2) >> 2) as u8;
    let b = ((p0.b as u32 * 3 + p1.b as u32 + 2) >> 2) as u8;
    let a = ((p0.a as u32 * 3 + p1.a as u32 + 2) >> 2) as u8;
    PremulRgba8 { r, g, b, a }
}

#[cfg(any())]
mod tests {
    use super::*;
    use vello_common::filter::gaussian_blur::{
        MAX_KERNEL_SIZE, compute_gaussian_kernel, plan_decimated_blur,
    };

    #[test]
    fn test_extend_duplicate() {
        let size = 10;

        assert_eq!(extend(5, size, EdgeMode::Duplicate), 5);

        assert_eq!(extend(-1, size, EdgeMode::Duplicate), 0);
        assert_eq!(extend(-10, size, EdgeMode::Duplicate), 0);

        assert_eq!(extend(10, size, EdgeMode::Duplicate), 9);
        assert_eq!(extend(20, size, EdgeMode::Duplicate), 9);
    }

    #[test]
    fn test_extend_wrap() {
        let size = 10;

        assert_eq!(extend(5, size, EdgeMode::Wrap), 5);

        assert_eq!(extend(10, size, EdgeMode::Wrap), 0);
        assert_eq!(extend(11, size, EdgeMode::Wrap), 1);
        assert_eq!(extend(25, size, EdgeMode::Wrap), 5);

        assert_eq!(extend(-1, size, EdgeMode::Wrap), 9);
        assert_eq!(extend(-2, size, EdgeMode::Wrap), 8);
    }

    #[test]
    fn test_extend_mirror() {
        let size = 10;

        assert_eq!(extend(5, size, EdgeMode::Mirror), 5);

        assert_eq!(extend(10, size, EdgeMode::Mirror), 9);
        assert_eq!(extend(11, size, EdgeMode::Mirror), 8);

        assert_eq!(extend(19, size, EdgeMode::Mirror), 0);
        assert_eq!(extend(20, size, EdgeMode::Mirror), 0);

        assert_eq!(extend(-1, size, EdgeMode::Mirror), 0);
        assert_eq!(extend(-2, size, EdgeMode::Mirror), 1);
    }

    #[test]
    fn test_extend_none() {
        let size = 10;
        assert_eq!(extend(5, size, EdgeMode::None), 5);
        assert_eq!(extend(9, size, EdgeMode::None), 9);
        assert_eq!(extend(10, size, EdgeMode::None), 10);
        assert_eq!(extend(11, size, EdgeMode::None), 11);
    }

    #[test]
    fn test_decimate_weighted() {
        let p0 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        let p1 = PremulRgba8 {
            r: 8,
            g: 8,
            b: 8,
            a: 8,
        };
        let p2 = PremulRgba8 {
            r: 8,
            g: 8,
            b: 8,
            a: 8,
        };
        let p3 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };

        let result = decimate_weighted(p0, p1, p2, p3);
        assert_eq!(result.r, 6);
        assert_eq!(result.g, 6);
        assert_eq!(result.b, 6);
        assert_eq!(result.a, 6);
    }

    #[test]
    fn test_decimate_weighted_uniform() {
        let p = PremulRgba8 {
            r: 100,
            g: 150,
            b: 200,
            a: 255,
        };
        let result = decimate_weighted(p, p, p, p);
        assert_eq!(result.r, 100);
        assert_eq!(result.g, 150);
        assert_eq!(result.b, 200);
        assert_eq!(result.a, 255);
    }

    #[test]
    fn test_interpolate_25_75() {
        let p0 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };
        let p1 = PremulRgba8 {
            r: 100,
            g: 100,
            b: 100,
            a: 100,
        };

        let result = interpolate_25_75(p0, p1);
        assert_eq!(result.r, 75);
        assert_eq!(result.g, 75);
        assert_eq!(result.b, 75);
        assert_eq!(result.a, 75);
    }

    #[test]
    fn test_interpolate_75_25() {
        let p0 = PremulRgba8 {
            r: 100,
            g: 100,
            b: 100,
            a: 100,
        };
        let p1 = PremulRgba8 {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        };

        let result = interpolate_75_25(p0, p1);
        assert_eq!(result.r, 75);
        assert_eq!(result.g, 75);
        assert_eq!(result.b, 75);
        assert_eq!(result.a, 75);
    }

    #[test]
    fn test_interpolation_symmetry() {
        let p0 = PremulRgba8 {
            r: 50,
            g: 100,
            b: 150,
            a: 200,
        };
        let p1 = PremulRgba8 {
            r: 200,
            g: 150,
            b: 100,
            a: 50,
        };

        let r1 = interpolate_25_75(p0, p1);
        let r2 = interpolate_75_25(p1, p0);

        assert!((r1.r as i32 - r2.r as i32).abs() <= 0);
        assert!((r1.g as i32 - r2.g as i32).abs() <= 0);
        assert!((r1.b as i32 - r2.b as i32).abs() <= 0);
        assert!((r1.a as i32 - r2.a as i32).abs() <= 0);
    }

    #[test]
    fn test_small_image_sizes() {
        let mut pixmap = Pixmap::new(1, 1);
        let (n_decimations, kernel, kernel_size) = plan_decimated_blur(2.0);

        let result = std::panic::catch_unwind(move || {
            let mut scratch = Pixmap::new(1, 1);
            apply_blur(
                &mut pixmap,
                &mut scratch,
                n_decimations,
                &kernel[..usize::from(kernel_size)],
                EdgeMode::None,
            );
        });

        assert!(result.is_ok());
    }

    #[test]
    fn test_downscale_odd_dimensions() {
        let mut pixmap = Pixmap::new(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                pixmap.set_pixel(
                    x,
                    y,
                    PremulRgba8 {
                        r: 255,
                        g: 255,
                        b: 255,
                        a: 255,
                    },
                );
            }
        }

        let (new_width, new_height) = downscale(&mut pixmap, 5, 5, EdgeMode::Duplicate);
        assert_eq!(new_width, 3);
        assert_eq!(new_height, 3);
    }

    #[test]
    fn test_upscale_dimensions() {
        let mut pixmap = Pixmap::new(6, 6);
        let (new_width, new_height) = upscale(&mut pixmap, 3, 3, EdgeMode::Duplicate);
        assert_eq!(new_width, 6);
        assert_eq!(new_height, 6);
    }

    #[test]
    fn test_convolve_x_uniform() {
        let mut src = Pixmap::new(5, 3);
        let mut dst = Pixmap::new(5, 3);
        src.data_mut().fill(PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        });

        let (kernel, kernel_size) = compute_gaussian_kernel(1.0);
        convolve_x(
            &src,
            &mut dst,
            5,
            3,
            &kernel[..usize::from(kernel_size)],
            kernel_size / 2,
            EdgeMode::Duplicate,
        );

        for y in 0..3 {
            for x in 0..5 {
                let pixel = dst.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 255,
                    }
                );
            }
        }
    }

    #[test]
    fn test_convolve_y_uniform() {
        let mut src = Pixmap::new(3, 5);
        let mut dst = Pixmap::new(3, 5);
        src.data_mut().fill(PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        });

        let (kernel, kernel_size) = compute_gaussian_kernel(1.0);
        convolve_y(
            &src,
            &mut dst,
            3,
            5,
            &kernel[..usize::from(kernel_size)],
            kernel_size / 2,
            EdgeMode::Duplicate,
        );

        for y in 0..5 {
            for x in 0..3 {
                let pixel = dst.sample(x, y);
                assert_eq!(
                    pixel,
                    PremulRgba8 {
                        r: 128,
                        g: 128,
                        b: 128,
                        a: 255,
                    }
                );
            }
        }
    }

    #[test]
    fn test_convolve_identity_kernel() {
        let mut src = Pixmap::new(3, 3);
        let mut dst = Pixmap::new(3, 3);
        src.set_pixel(
            1,
            1,
            PremulRgba8 {
                r: 255,
                g: 100,
                b: 50,
                a: 200,
            },
        );

        let kernel = [1.0];
        convolve_x(&src, &mut dst, 3, 3, &kernel, 0, EdgeMode::None);

        assert_eq!(dst.sample(1, 1), src.sample(1, 1));
    }

    #[test]
    fn test_large_sigma_clamped_to_max() {
        let (kernel, kernel_size) = compute_gaussian_kernel(100.0);
        assert_eq!(usize::from(kernel_size), MAX_KERNEL_SIZE);

        let sum: f32 = kernel.iter().take(usize::from(kernel_size)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_decimation_reduces_kernel_size() {
        let (n_decimations, _kernel, kernel_size) = plan_decimated_blur(100.0);
        assert_eq!(kernel_size, 11);
        assert_eq!(n_decimations, 6, "Large sigma should trigger decimation");
    }

    #[test]
    fn test_sample_x_at_boundaries() {
        let mut pixmap = Pixmap::new(3, 1);
        pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 10,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            1,
            0,
            PremulRgba8 {
                r: 20,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            2,
            0,
            PremulRgba8 {
                r: 30,
                g: 0,
                b: 0,
                a: 255,
            },
        );

        let mut p = sample_x(&pixmap, -1, 0, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 10);

        p = sample_x(&pixmap, 3, 0, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 30);

        p = sample_x(&pixmap, -1, 0, 3, EdgeMode::None);
        assert_eq!(p.a, 0);
    }

    #[test]
    fn test_sample_y_at_boundaries() {
        let mut pixmap = Pixmap::new(1, 3);
        pixmap.set_pixel(
            0,
            0,
            PremulRgba8 {
                r: 10,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            0,
            1,
            PremulRgba8 {
                r: 20,
                g: 0,
                b: 0,
                a: 255,
            },
        );
        pixmap.set_pixel(
            0,
            2,
            PremulRgba8 {
                r: 30,
                g: 0,
                b: 0,
                a: 255,
            },
        );

        let mut p = sample_y(&pixmap, 0, -1, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 10);

        p = sample_y(&pixmap, 0, 3, 3, EdgeMode::Duplicate);
        assert_eq!(p.r, 30);

        p = sample_y(&pixmap, 0, -1, 3, EdgeMode::None);
        assert_eq!(p.a, 0);
    }

    #[test]
    fn test_kernel_normalization_precision() {
        for sigma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let (kernel, kernel_size) = compute_gaussian_kernel(sigma);
            let sum: f32 = kernel.iter().take(usize::from(kernel_size)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Kernel for σ={} not normalized: sum={}",
                sigma,
                sum
            );
        }
    }

    #[test]
    fn test_downscale_upscale_roundtrip() {
        let mut pixmap = Pixmap::new(8, 8);
        pixmap.data_mut().fill(PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        });

        let (w1, h1) = downscale(&mut pixmap, 8, 8, EdgeMode::Duplicate);
        assert_eq!(w1, 4);
        assert_eq!(h1, 4);

        let (w2, h2) = upscale(&mut pixmap, w1, h1, EdgeMode::Duplicate);
        assert_eq!(w2, 8);
        assert_eq!(h2, 8);
    }

    fn pixmap_from_red(width: u16, height: u16, values: &[&[u8]]) -> Pixmap {
        let mut pixmap = Pixmap::new(width, height);
        for (y, row) in values.iter().enumerate() {
            for (x, &r) in row.iter().enumerate() {
                pixmap.set_pixel(
                    x as u16,
                    y as u16,
                    PremulRgba8 {
                        r,
                        g: 0,
                        b: 0,
                        a: 255,
                    },
                );
            }
        }
        pixmap
    }

    fn assert_red_values<const W: usize, const H: usize>(pixmap: &Pixmap, expected: [[u8; W]; H]) {
        for (y, row) in expected.iter().enumerate() {
            for (x, &want) in row.iter().enumerate() {
                let got = pixmap.sample(x as u16, y as u16).r;

                assert_eq!(
                    got, want,
                    "red mismatch at ({x}, {y}): got {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn test_downscale_x_non_uniform() {
        let mut pixmap = pixmap_from_red(4, 2, &[&[0, 40, 80, 120], &[20, 60, 100, 140]]);

        let dst_width = 4_u16.div_ceil(2);
        downscale_x(&mut pixmap, 4, 2, dst_width, EdgeMode::Duplicate);

        assert_red_values(&pixmap, [[25, 95], [45, 115]]);
    }

    #[test]
    fn test_downscale_y_non_uniform() {
        let mut pixmap = pixmap_from_red(2, 4, &[&[25, 95], &[45, 115], &[35, 105], &[55, 125]]);

        let dst_height = 4_u16.div_ceil(2);
        downscale_y(&mut pixmap, 2, 4, dst_height, EdgeMode::Duplicate);

        assert_red_values(&pixmap, [[34, 104], [46, 116]]);
    }

    #[test]
    fn test_upscale_x_non_uniform() {
        let mut pixmap = pixmap_from_red(4, 2, &[&[34, 104], &[46, 116]]);

        upscale_x(&mut pixmap, 2, 2, EdgeMode::Duplicate);

        assert_red_values(&pixmap, [[34, 52, 87, 104], [46, 64, 99, 116]]);
    }

    #[test]
    fn test_upscale_y_non_uniform() {
        let mut pixmap = pixmap_from_red(4, 4, &[&[34, 52, 87, 104], &[46, 64, 99, 116]]);

        upscale_y(&mut pixmap, 4, 2, EdgeMode::Duplicate);

        #[rustfmt::skip]
        assert_red_values(&pixmap, [
            [ 34,  52,  87, 104],
            [ 37,  55,  90, 107],
            [ 43,  61,  96, 113],
            [ 46,  64,  99, 116],
        ]);
    }
}
