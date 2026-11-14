use crate::vello_cpu::fine::macros::{f32x16_painter, u8x16_painter};
use crate::vello_cpu::fine::{Splat4thExt, u8_to_f32};
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{
    Bytes, Select, Simd, SimdBase, SimdFloat, f32x4, f32x16, u8x16, u32x4,
};
use vello_common::pixmap::Pixmap;
use vello_common::simd::element_wise_splat;

#[derive(Debug)]
pub(crate) struct PlainNNImagePainter<'a, S: Simd> {
    nn: NNImagePainter<'a, S>,
    simd: S,
}

impl<'a, S: Simd> PlainNNImagePainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        Self { nn: NNImagePainter::new(simd, image, pixmap, start_x, start_y), simd }
    }
}

impl<S: Simd> Iterator for PlainNNImagePainter<'_, S> {
    type Item = u8x16<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.nn.next()
    }
}

u8x16_painter!(PlainNNImagePainter<'_, S>);

#[derive(Debug)]
pub(crate) struct NNImagePainter<'a, S: Simd> {
    data: ImagePainterData<'a, S>,
    positions: crate::vello_cpu::detour::Positions,
    x: f32,
    y: f32,
    simd: S,
}

impl<'a, S: Simd> NNImagePainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        let data = ImagePainterData::new(simd, image, pixmap, start_x, start_y);
        let positions = crate::vello_cpu::detour::Positions::of_image(image);
        let (x, y) = ((start_x - crate::vello_cpu::detour::PIXEL_CENTER) as f32, (start_y - crate::vello_cpu::detour::PIXEL_CENTER) as f32);

        Self { data, positions, x, y, simd }
    }
}

impl<S: Simd> Iterator for NNImagePainter<'_, S> {
    type Item = u8x16<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let (mut xs, mut ys) = ([0.0_f32; 4], [0.0_f32; 4]);
        for i in 0..4 {
            (xs[i], ys[i]) = self.positions.at(self.x, self.y + i as f32);
        }
        let x_positions = extend_mode(f32x4::from_slice(self.simd, &xs), self.data.image.sampler.x_extend, self.data.width, self.data.width_inv);
        let y_positions = extend_mode(f32x4::from_slice(self.simd, &ys), self.data.image.sampler.y_extend, self.data.height, self.data.height_inv);

        let samples = sample(self.simd, &self.data, x_positions, y_positions);

        self.x += 1.0;

        Some(samples)
    }
}

u8x16_painter!(NNImagePainter<'_, S>);

#[derive(Debug)]
pub(crate) struct FilteredImagePainter<'a, S: Simd, const QUALITY: u8> {
    data: ImagePainterData<'a, S>,
    positions: crate::vello_cpu::detour::Positions,
    x: f32,
    y: f32,
    simd: S,
}

impl<'a, S: Simd, const QUALITY: u8> FilteredImagePainter<'a, S, QUALITY> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        let data = ImagePainterData::new(simd, image, pixmap, start_x, start_y);
        let positions = crate::vello_cpu::detour::Positions::of_image(image);
        let (x, y) = ((start_x - crate::vello_cpu::detour::PIXEL_CENTER) as f32, (start_y - crate::vello_cpu::detour::PIXEL_CENTER) as f32);

        Self { data, positions, x, y, simd }
    }
}

impl<S: Simd, const QUALITY: u8> Iterator for FilteredImagePainter<'_, S, QUALITY> {
    type Item = f32x16<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let (mut xs, mut ys) = ([0.0_f32; 4], [0.0_f32; 4]);
        for i in 0..4 {
            (xs[i], ys[i]) = self.positions.at(self.x, self.y + i as f32);
        }
        let x_positions = f32x4::from_slice(self.simd, &xs);
        let y_positions = f32x4::from_slice(self.simd, &ys);

        let x_shifted = x_positions + 0.5;
        let y_shifted = y_positions + 0.5;
        let x_base = x_shifted.floor();
        let y_base = y_shifted.floor();
        let x_fract = x_shifted - x_base;
        let y_fract = y_shifted - y_base;

        let mut interpolated_color = f32x16::splat(self.simd, 0.0);

        let sample = |x_pos: f32x4<S>, y_pos: f32x4<S>| {
            u8_to_f32(sample(self.simd, &self.data, x_pos, y_pos))
        };

        macro_rules! extend_x {
            ($idx:expr,$offsets:expr) => {
                extend_mode(
                    x_base + $offsets[$idx],
                    self.data.image.sampler.x_extend,
                    self.data.width,
                    self.data.width_inv,
                )
            };
        }

        macro_rules! extend_y {
            ($idx:expr,$offsets:expr) => {
                extend_mode(
                    y_base + $offsets[$idx],
                    self.data.image.sampler.y_extend,
                    self.data.height,
                    self.data.height_inv,
                )
            };
        }

        match QUALITY {
            1 => {
                let cx = [1.0 - x_fract, x_fract];
                let cy = [1.0 - y_fract, y_fract];

                const OFFSETS: [f32; 2] = [-1.0, 0.0];

                let x_positions = [extend_x!(0, OFFSETS), extend_x!(1, OFFSETS)];

                let y_positions = [extend_y!(0, OFFSETS), extend_y!(1, OFFSETS)];

                for x_idx in 0..2 {
                    let x_positions = x_positions[x_idx];

                    for y_idx in 0..2 {
                        let y_positions = y_positions[y_idx];
                        let color_sample = sample(x_positions, y_positions);
                        let w = element_wise_splat(self.simd, cx[x_idx] * cy[y_idx]);

                        interpolated_color = w.mul_add(color_sample, interpolated_color);
                    }
                }

                interpolated_color *= f32x16::splat(self.simd, 1.0 / 255.0);
            }
            2 => {
                let cx = weights(self.simd, x_fract);
                let cy = weights(self.simd, y_fract);

                const OFFSETS: [f32; 4] = [-2.0, -1.0, 0.0, 1.0];

                let x_positions = [
                    extend_x!(0, OFFSETS),
                    extend_x!(1, OFFSETS),
                    extend_x!(2, OFFSETS),
                    extend_x!(3, OFFSETS),
                ];

                let y_positions = [
                    extend_y!(0, OFFSETS),
                    extend_y!(1, OFFSETS),
                    extend_y!(2, OFFSETS),
                    extend_y!(3, OFFSETS),
                ];

                for x_idx in 0..4 {
                    let x_positions = x_positions[x_idx];
                    for y_idx in 0..4 {
                        let y_positions = y_positions[y_idx];

                        let color_sample = sample(x_positions, y_positions);
                        let w = element_wise_splat(self.simd, cx[x_idx] * cy[y_idx]);

                        interpolated_color = w.mul_add(color_sample, interpolated_color);
                    }
                }

                interpolated_color *= f32x16::splat(self.simd, 1.0 / 255.0);

                let alphas = interpolated_color.splat_4th();

                interpolated_color = interpolated_color
                    .min(alphas)
                    .min(f32x16::splat(self.simd, 1.0))
                    .max(f32x16::splat(self.simd, 0.0));
            }
            _ => panic!(
                "Unknown value for `FilteredImagePainter`'s const-generic `QUALITY` parameter. Expected `1` for bilinear or `2` for bicubic, got: `{QUALITY}`."
            ),
        }

        self.x += 1.0;

        Some(interpolated_color)
    }
}

f32x16_painter!(FilteredImagePainter<'_, S, 1>);

f32x16_painter!(FilteredImagePainter<'_, S, 2>);

#[derive(Debug)]
pub(crate) struct ImagePainterData<'a, S: Simd> {
    pub(crate) image: &'a EncodedImage,
    pub(crate) pixmap: &'a Pixmap,
    pub(crate) height: f32x4<S>,
    pub(crate) height_inv: f32x4<S>,
    pub(crate) width: f32x4<S>,
    pub(crate) width_inv: f32x4<S>,
    pub(crate) width_u32: u32x4<S>,
}

impl<'a, S: Simd> ImagePainterData<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        _start_x: f64,
        _start_y: f64,
    ) -> Self {
        simd.vectorize(
            #[inline(always)]
            || {
                let width = pixmap.width() as f32;
                let height = pixmap.height() as f32;

                let width_inv = f32x4::splat(simd, 1.0 / width);
                let height_inv = f32x4::splat(simd, 1.0 / height);
                let width = f32x4::splat(simd, width);
                let width_u32 = u32x4::splat(simd, pixmap.width() as u32);
                let height = f32x4::splat(simd, height);

                Self {
                    pixmap,
                    image,
                    width,
                    height,
                    width_u32,
                    width_inv,
                    height_inv,
                }
            },
        )
    }
}

#[inline(always)]
pub(crate) fn sample<S: Simd>(
    simd: S,
    data: &ImagePainterData<'_, S>,
    x_positions: f32x4<S>,
    y_positions: f32x4<S>,
) -> u8x16<S> {
    let idx = x_positions.to_int::<u32x4<S>>() + y_positions.to_int::<u32x4<S>>() * data.width_u32;

    u32x4::from_slice(
        simd,
        &[
            data.pixmap.sample_idx(idx[0]).to_u32(),
            data.pixmap.sample_idx(idx[1]).to_u32(),
            data.pixmap.sample_idx(idx[2]).to_u32(),
            data.pixmap.sample_idx(idx[3]).to_u32(),
        ],
    )
    .to_bytes()
}

#[inline(always)]
pub(crate) fn extend_mode<S: Simd>(
    val: f32x4<S>,
    extend: crate::vello_cpu::peniko::Extend,
    max: f32x4<S>,
    inv_max: f32x4<S>,
) -> f32x4<S> {
    match extend {
        crate::vello_cpu::peniko::Extend::Pad => val.min(max - 1.0).max(0.0),
        crate::vello_cpu::peniko::Extend::Repeat => euclid_mod(val.floor(), max, inv_max),
        crate::vello_cpu::peniko::Extend::Reflect => {
            let period = max + max;
            let r = euclid_mod(val.floor(), period, inv_max * 0.5);
            r.simd_ge(max).select(period - 1.0 - r, r)
        }
    }
}

#[inline(always)]
fn euclid_mod<S: Simd>(t: f32x4<S>, m: f32x4<S>, inv_m: f32x4<S>) -> f32x4<S> {
    let q = (t * inv_m).round_ties_even();
    let r = q.mul_add(-m, t);
    r.simd_lt(0.0).select(r + m, r)
}

fn weights<S: Simd>(simd: S, fract: f32x4<S>) -> [f32x4<S>; 4] {
    simd.vectorize(
        #[inline(always)]
        || {
            let s = fract.simd;
            const MF: [[f32; 4]; 4] = mf_resampler();

            [
                single_weight(
                    fract,
                    f32x4::splat(s, MF[0][0]),
                    f32x4::splat(s, MF[0][1]),
                    f32x4::splat(s, MF[0][2]),
                    f32x4::splat(s, MF[0][3]),
                ),
                single_weight(
                    fract,
                    f32x4::splat(s, MF[1][0]),
                    f32x4::splat(s, MF[1][1]),
                    f32x4::splat(s, MF[1][2]),
                    f32x4::splat(s, MF[1][3]),
                ),
                single_weight(
                    fract,
                    f32x4::splat(s, MF[2][0]),
                    f32x4::splat(s, MF[2][1]),
                    f32x4::splat(s, MF[2][2]),
                    f32x4::splat(s, MF[2][3]),
                ),
                single_weight(
                    fract,
                    f32x4::splat(s, MF[3][0]),
                    f32x4::splat(s, MF[3][1]),
                    f32x4::splat(s, MF[3][2]),
                    f32x4::splat(s, MF[3][3]),
                ),
            ]
        },
    )
}

#[inline(always)]
fn single_weight<S: Simd>(
    t: f32x4<S>,
    a: f32x4<S>,
    b: f32x4<S>,
    c: f32x4<S>,
    d: f32x4<S>,
) -> f32x4<S> {
    t.mul_add(d, c).mul_add(t, b).mul_add(t, a)
}

const fn mf_resampler() -> [[f32; 4]; 4] {
    cubic_resampler(1.0 / 3.0, 1.0 / 3.0)
}

const fn cubic_resampler(b: f32, c: f32) -> [[f32; 4]; 4] {
    [
        [
            (1.0 / 6.0) * b,
            -(3.0 / 6.0) * b - c,
            (3.0 / 6.0) * b + 2.0 * c,
            -(1.0 / 6.0) * b - c,
        ],
        [
            1.0 - (2.0 / 6.0) * b,
            0.0,
            -3.0 + (12.0 / 6.0) * b + c,
            2.0 - (9.0 / 6.0) * b - c,
        ],
        [
            (1.0 / 6.0) * b,
            (3.0 / 6.0) * b + c,
            3.0 - (15.0 / 6.0) * b - 2.0 * c,
            -2.0 + (9.0 / 6.0) * b + c,
        ],
        [0.0, 0.0, -c, (1.0 / 6.0) * b + c],
    ]
}

#[cfg(any())]
mod tests {
    use super::*;
    use vello_common::fearless_simd::{Fallback, SimdMask};

    fn assert_extend(mode: crate::vello_cpu::peniko::Extend, max: f32, values: [f32; 4], expected: [u32; 4]) {
        let simd = Fallback::new();
        let inv_max = f32x4::splat(simd, 1.0 / max);
        let max = f32x4::splat(simd, max);
        let values = f32x4::from_slice(simd, &values);
        let extended = extend_mode(values, mode, max, inv_max).to_int::<u32x4<_>>();
        let expected = u32x4::from_slice(simd, &expected);

        assert!(
            extended.simd_eq(expected).all_true(),
            "expected {expected:?}, got {extended:?}"
        );
    }

    #[test]
    fn taps_at_boundaries() {
        assert_extend(
            crate::vello_cpu::peniko::Extend::Reflect,
            2.0,
            [(-1.0_f32).next_down(); 4],
            [1; 4],
        );
        assert_extend(
            crate::vello_cpu::peniko::Extend::Repeat,
            3.0,
            [15.0_f32.next_down(); 4],
            [2; 4],
        );
    }

    #[test]
    fn repeat() {
        assert_extend(
            crate::vello_cpu::peniko::Extend::Repeat,
            10.0,
            [-f32::EPSILON, -1.25, 9.75, 12.25],
            [9, 8, 9, 2],
        );
    }

    #[test]
    fn reflect() {
        assert_extend(
            crate::vello_cpu::peniko::Extend::Reflect,
            10.0,
            [-1.25, 0.25, 9.75, 12.25],
            [1, 0, 9, 7],
        );
    }

    #[test]
    fn large_image() {
        assert_extend(
            crate::vello_cpu::peniko::Extend::Repeat,
            4096.0,
            [-4097.25, 4096.25, 12_345.75, 1_000_000.75],
            [4094, 0, 57, 576],
        );
        assert_extend(
            crate::vello_cpu::peniko::Extend::Reflect,
            4096.0,
            [-4097.25, 4096.25, 12_345.75, 1_000_000.75],
            [4094, 4095, 4038, 576],
        );
    }

    #[test]
    fn large_coordinates_preserve_texel() {
        assert_extend(
            crate::vello_cpu::peniko::Extend::Repeat,
            2.0,
            [16_777_215.0, -16_777_215.0, 16_777_215.0, -16_777_215.0],
            [1; 4],
        );
        assert_extend(
            crate::vello_cpu::peniko::Extend::Reflect,
            3.0,
            [16_777_215.0, -16_777_215.0, 16_777_215.0, -16_777_215.0],
            [2; 4],
        );
    }
}
