// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::vello_cpu::fine::common::image::{ImagePainterData, extend_mode, sample};
use crate::vello_cpu::fine::macros::u8x16_painter;
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{f32x4, prelude::*, u8x16, u16x16};
use vello_common::pixmap::Pixmap;
use vello_common::simd::element_wise_splat;
use vello_common::util::{Div255Ext, f32_to_u8, narrow, widen};

/// A faster bilinear image renderer for the u8 pipeline.
#[derive(Debug)]
pub(crate) struct BilinearImagePainter<'a, S: Simd> {
    data: ImagePainterData<'a, S>,
    // detour: each pixel's position from `positions` (as the gradients and the nearest-neighbor painters), the
    // column's first pixel at (x, y); upstream adds the advance from the span's start
    positions: crate::vello_cpu::detour::Positions,
    x: f32,
    y: f32,
    simd: S,
}

impl<'a, S: Simd> BilinearImagePainter<'a, S> {
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

impl<S: Simd> Iterator for BilinearImagePainter<'_, S> {
    type Item = u8x16<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let (mut xs, mut ys) = ([0.0_f32; 4], [0.0_f32; 4]);
        for i in 0..4 {
            (xs[i], ys[i]) = self.positions.at(self.x, self.y + i as f32);
        }
        let x_positions = f32x4::from_slice(self.simd, &xs);
        let y_positions = f32x4::from_slice(self.simd, &ys);

        let extend_x = |x_pos: f32x4<S>| {
            extend_mode(
                x_pos,
                self.data.image.sampler.x_extend,
                self.data.width,
                self.data.width_inv,
            )
        };

        let extend_y = |y_pos: f32x4<S>| {
            extend_mode(
                y_pos,
                self.data.image.sampler.y_extend,
                self.data.height,
                self.data.height_inv,
            )
        };

        let x_shifted = x_positions + 0.5;
        let y_shifted = y_positions + 0.5;
        let x_base = x_shifted.floor();
        let y_base = y_shifted.floor();

        let fx = f32_to_u8(element_wise_splat(
            self.simd,
            (x_shifted - x_base).mul_add(255.0, 0.5),
        ));
        let fy = f32_to_u8(element_wise_splat(
            self.simd,
            (y_shifted - y_base).mul_add(255.0, 0.5),
        ));

        let fx = widen(fx);
        let fy = widen(fy);
        let fx_inv = u16x16::splat(self.simd, 255) - fx;
        let fy_inv = u16x16::splat(self.simd, 255) - fy;

        let x_pos1 = extend_x(x_base - 1.0);
        let x_pos2 = extend_x(x_base);
        let y_pos1 = extend_y(y_base - 1.0);
        let y_pos2 = extend_y(y_base);

        let p00 = widen(sample(self.simd, &self.data, x_pos1, y_pos1));
        let p10 = widen(sample(self.simd, &self.data, x_pos2, y_pos1));
        let p01 = widen(sample(self.simd, &self.data, x_pos1, y_pos2));
        let p11 = widen(sample(self.simd, &self.data, x_pos2, y_pos2));

        let ip1 = (p00 * fx_inv + p10 * fx).div_255();
        let ip2 = (p01 * fx_inv + p11 * fx).div_255();
        let res = narrow((ip1 * fy_inv + ip2 * fy).div_255());

        self.x += 1.0;

        Some(res)
    }
}

u8x16_painter!(BilinearImagePainter<'_, S>);

/// A faster bilinear image renderer for axis-aligned images (no skew) in the u8 pipeline.
///
/// detour: the general painter (`BilinearImagePainter`), each pixel positioned on its own; upstream precomputes the
/// y positions and weights once a span.
#[derive(Debug)]
pub(crate) struct PlainBilinearImagePainter<'a, S: Simd> {
    bilinear: BilinearImagePainter<'a, S>,
    simd: S,
}

impl<'a, S: Simd> PlainBilinearImagePainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        Self { bilinear: BilinearImagePainter::new(simd, image, pixmap, start_x, start_y), simd }
    }
}

impl<S: Simd> Iterator for PlainBilinearImagePainter<'_, S> {
    type Item = u8x16<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.bilinear.next()
    }
}

u8x16_painter!(PlainBilinearImagePainter<'_, S>);
