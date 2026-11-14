use crate::vello_cpu::filter::context::ScratchBuffer;
use crate::vello_cpu::filter::filter_highp;
use crate::vello_cpu::fine::FineKernel;
use crate::vello_cpu::fine::{COLOR_COMPONENTS, Painter, Splat4thExt};
use crate::vello_cpu::peniko::BlendMode;
use crate::vello_cpu::region::Region;
use vello_common::fearless_simd::*;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::Affine;
use vello_common::mask::Mask;
use vello_common::paint::{PremulColor, Tint, TintMode};
use vello_common::pixmap::Pixmap;
use vello_common::tile::Tile;

pub(crate) mod blend;
pub(crate) mod compose;

#[derive(Clone, Copy, Debug)]
pub struct F32Kernel;

impl<S: Simd> FineKernel<S> for F32Kernel {
    type Numeric = f32;
    type Composite = f32x16<S>;
    type NumericVec = f32x16<S>;

    #[inline(always)]
    fn extract_color(color: PremulColor) -> [Self::Numeric; 4] {
        color.as_premul_f32().components
    }

    #[expect(
        private_interfaces,
        reason = "`FineKernel` is public but this specific method is not needed."
    )]
    fn filter_layer(
        pixmap: &mut Pixmap,
        filter: &Filter,
        filter_scratch: &mut ScratchBuffer,
        transform: Affine,
    ) {
        filter_highp(filter, pixmap, filter_scratch, transform);
    }

    #[inline(never)]
    fn copy_solid(simd: S, dest: &mut [Self::Numeric], src: [Self::Numeric; 4]) {
        simd.vectorize(
            #[inline(always)]
            || {
                let color = f32x16::block_splat(src.simd_into(simd));

                for el in dest.chunks_exact_mut(16) {
                    color.store_slice(el);
                }
            },
        );
    }

    fn apply_mask(
        simd: S,
        dest: &mut [Self::Numeric],
        mut src: impl Iterator<Item = Self::NumericVec>,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for el in dest.chunks_exact_mut(16) {
                    let loaded = f32x16::from_slice(simd, el);
                    let mulled = loaded * src.next().unwrap();
                    mulled.store_slice(el);
                }
            },
        );
    }

    #[inline(always)]
    fn apply_painter<'a>(_: S, dest: &mut [Self::Numeric], painter: impl Painter + 'a) {
        painter.paint_f32(dest);
    }

    fn apply_tint(simd: S, dest: &mut [Self::Numeric], tint: &Tint) {
        let premul = tint.color.premultiply();
        let [r, g, b, a] = premul.components;

        simd.vectorize(
            #[inline(always)]
            || {
                let tint_v = f32x16::block_splat(f32x4::from_slice(simd, &[r, g, b, a]));

                match tint.mode {
                    TintMode::AlphaMask => {
                        for chunk in dest.chunks_exact_mut(16) {
                            let pixel = f32x16::from_slice(simd, chunk);
                            let alphas = pixel.splat_4th();
                            let tinted = tint_v * alphas;
                            tinted.store_slice(chunk);
                        }
                    }
                    TintMode::Multiply => {
                        for chunk in dest.chunks_exact_mut(16) {
                            let pixel = f32x16::from_slice(simd, chunk);
                            let tinted = pixel * tint_v;
                            tinted.store_slice(chunk);
                        }
                    }
                }
            },
        );
    }

    #[inline(always)]
    fn alpha_composite_solid(
        simd: S,
        dest: &mut [Self::Numeric],
        src: [Self::Numeric; 4],
        alphas: Option<&[u8]>,
    ) {
        if let Some(alphas) = alphas {
            alpha_fill::alpha_composite_solid(
                simd,
                dest,
                src,
                bytemuck::cast_slice::<u8, [u8; 4]>(alphas).iter().copied(),
            );
        } else {
            fill::alpha_composite_solid(simd, dest, src);
        }
    }

    fn alpha_composite_buffer(
        simd: S,
        dest: &mut [Self::Numeric],
        src: &[Self::Numeric],
        alphas: Option<&[u8]>,
    ) {
        if let Some(alphas) = alphas {
            alpha_fill::alpha_composite_arbitrary(
                simd,
                dest,
                src.chunks_exact(16).map(|el| f32x16::from_slice(simd, el)),
                bytemuck::cast_slice::<u8, [u8; 4]>(alphas).iter().copied(),
            );
        } else {
            fill::alpha_composite_arbitrary(
                simd,
                dest,
                src.chunks_exact(16).map(|el| f32x16::from_slice(simd, el)),
            );
        }
    }

    fn blend(
        simd: S,
        dest: &mut [Self::Numeric],
        mut start_x: u16,
        start_y: u16,
        src: impl Iterator<Item = Self::Composite>,
        blend_mode: BlendMode,
        alphas: Option<&[u8]>,
        mask: Option<&Mask>,
    ) {
        let alpha_iter = alphas.map(|a| bytemuck::cast_slice::<u8, [u8; 4]>(a).iter().copied());

        let mask_iter = mask.map(|m| {
            let width = m.width();
            let height = m.height();

            core::iter::from_fn(move || {
                let samples = if start_x < width && start_y + 3 < height {
                    [
                        m.sample(start_x, start_y),
                        m.sample(start_x, start_y + 1),
                        m.sample(start_x, start_y + 2),
                        m.sample(start_x, start_y + 3),
                    ]
                } else {
                    [
                        if start_x < width && start_y < height {
                            m.sample(start_x, start_y)
                        } else {
                            255
                        },
                        if start_x < width && start_y + 1 < height {
                            m.sample(start_x, start_y + 1)
                        } else {
                            255
                        },
                        if start_x < width && start_y + 2 < height {
                            m.sample(start_x, start_y + 2)
                        } else {
                            255
                        },
                        if start_x < width && start_y + 3 < height {
                            m.sample(start_x, start_y + 3)
                        } else {
                            255
                        },
                    ]
                };

                start_x += 1;

                Some(samples)
            })
        });

        match (alpha_iter, mask_iter) {
            (Some(alpha_iter), Some(mut mask_iter)) => {
                let iter = alpha_iter.map(|a1| {
                    let a2 = mask_iter.next().unwrap();
                    [
                        ((a1[0] as u16 * a2[0] as u16) / 255) as u8,
                        ((a1[1] as u16 * a2[1] as u16) / 255) as u8,
                        ((a1[2] as u16 * a2[2] as u16) / 255) as u8,
                        ((a1[3] as u16 * a2[3] as u16) / 255) as u8,
                    ]
                });
                alpha_fill::blend(simd, dest, src, iter, blend_mode);
            }
            (None, Some(mask_iter)) => alpha_fill::blend(simd, dest, src, mask_iter, blend_mode),
            (Some(alpha_iter), None) => alpha_fill::blend(simd, dest, src, alpha_iter, blend_mode),
            (None, None) => {
                fill::blend(simd, dest, src, blend_mode);
            }
        }
    }

    fn pack(_simd: S, scratch: &[Self::Numeric], width: usize, region: &mut Region<'_>) {
        for y in 0..region.height {
            let row = &mut region.row_mut(y)[..width * COLOR_COMPONENTS];
            for (dx, pixel) in row.chunks_exact_mut(COLOR_COMPONENTS).enumerate() {
                let idx = COLOR_COMPONENTS * (Tile::HEIGHT as usize * dx + usize::from(y));
                let src = &scratch[idx..idx + COLOR_COMPONENTS];
                pixel[0] = (src[0] * 255.0 + 0.5) as u8;
                pixel[1] = (src[1] * 255.0 + 0.5) as u8;
                pixel[2] = (src[2] * 255.0 + 0.5) as u8;
                pixel[3] = (src[3] * 255.0 + 0.5) as u8;
            }
        }
    }

    fn unpack(_simd: S, region: &mut Region<'_>, width: usize, scratch: &mut [Self::Numeric]) {
        for y in 0..region.height {
            let row = &region.row_mut(y)[..width * COLOR_COMPONENTS];
            for (dx, pixel) in row.chunks_exact(COLOR_COMPONENTS).enumerate() {
                let idx = COLOR_COMPONENTS * (Tile::HEIGHT as usize * dx + usize::from(y));
                scratch[idx] = pixel[0] as f32 / 255.0;
                scratch[idx + 1] = pixel[1] as f32 / 255.0;
                scratch[idx + 2] = pixel[2] as f32 / 255.0;
                scratch[idx + 3] = pixel[3] as f32 / 255.0;
            }
        }
    }
}

mod fill {
    use crate::vello_cpu::fine::Splat4thExt;
    use crate::vello_cpu::fine::highp::blend;
    use crate::vello_cpu::fine::highp::compose::ComposeExt;
    use crate::vello_cpu::peniko::BlendMode;

    use vello_common::fearless_simd::*;

    #[inline(always)]
    pub(super) fn alpha_composite_solid<S: Simd>(s: S, dest: &mut [f32], src: [f32; 4]) {
        s.vectorize(
            #[inline(always)]
            || {
                let one_minus_alpha = 1.0 - f32x16::block_splat(f32x4::splat(s, src[3]));
                let src_c = f32x16::block_splat(f32x4::simd_from(s, src));

                for next_dest in dest.chunks_exact_mut(16) {
                    alpha_composite_inner(s, next_dest, src_c, one_minus_alpha);
                }
            },
        );
    }

    #[inline(always)]
    pub(super) fn alpha_composite_arbitrary<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for (next_dest, next_src) in dest.chunks_exact_mut(16).zip(src) {
                    let one_minus_alpha = 1.0 - next_src.splat_4th();
                    alpha_composite_inner(simd, next_dest, next_src, one_minus_alpha);
                }
            },
        );
    }

    pub(super) fn blend<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
        blend_mode: BlendMode,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for (next_dest, next_src) in dest.chunks_exact_mut(16).zip(src) {
                    let bg_v = f32x16::from_slice(simd, next_dest);
                    let src_c = blend::mix(next_src, bg_v, blend_mode);
                    let res = blend_mode.compose(simd, src_c, bg_v, None);
                    res.store_slice(next_dest);
                }
            },
        );
    }

    #[inline(always)]
    fn alpha_composite_inner<S: Simd>(
        s: S,
        dest: &mut [f32],
        src: f32x16<S>,
        one_minus_alpha: f32x16<S>,
    ) {
        let mut bg_c = f32x16::from_slice(s, dest);
        bg_c = one_minus_alpha.mul_add(bg_c, src);
        bg_c.store_slice(dest);
    }
}

mod alpha_fill {
    use crate::vello_cpu::fine::Splat4thExt;
    use crate::vello_cpu::fine::highp::compose::ComposeExt;
    use crate::vello_cpu::fine::highp::{blend, extract_masks};
    use crate::vello_cpu::peniko::BlendMode;
    use vello_common::fearless_simd::*;

    #[inline(always)]
    pub(super) fn alpha_composite_solid<S: Simd>(
        s: S,
        dest: &mut [f32],
        src: [f32; 4],
        alphas: impl Iterator<Item = [u8; 4]>,
    ) {
        s.vectorize(
            #[inline(always)]
            || {
                let src_a = f32x16::splat(s, src[3]);
                let src_c = f32x16::block_splat(src.simd_into(s));
                let one = f32x16::splat(s, 1.0);

                for (next_dest, next_mask) in dest.chunks_exact_mut(16).zip(alphas) {
                    alpha_composite_inner(s, next_dest, &next_mask, src_c, src_a, one);
                }
            },
        );
    }

    pub(super) fn alpha_composite_arbitrary<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
        alphas: impl Iterator<Item = [u8; 4]>,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                let one = f32x16::splat(simd, 1.0);

                for ((next_dest, next_mask), next_src) in
                    dest.chunks_exact_mut(16).zip(alphas).zip(src)
                {
                    let src_a = next_src.splat_4th();
                    alpha_composite_inner(simd, next_dest, &next_mask, next_src, src_a, one);
                }
            },
        );
    }

    pub(super) fn blend<S: Simd, T: Iterator<Item = f32x16<S>>>(
        simd: S,
        dest: &mut [f32],
        src: T,
        alphas: impl Iterator<Item = [u8; 4]>,
        blend_mode: BlendMode,
    ) {
        simd.vectorize(
            #[inline(always)]
            || {
                for ((next_dest, next_mask), next_src) in
                    dest.chunks_exact_mut(16).zip(alphas).zip(src)
                {
                    let masks = extract_masks(simd, &next_mask);
                    let bg = f32x16::from_slice(simd, next_dest);
                    let src_c = blend::mix(next_src, bg, blend_mode);
                    let res = blend_mode.compose(simd, src_c, bg, Some(masks));
                    res.store_slice(next_dest);
                }
            },
        );
    }

    #[inline(always)]
    fn alpha_composite_inner<S: Simd>(
        s: S,
        dest: &mut [f32],
        masks: &[u8; 4],
        src_c: f32x16<S>,
        src_a: f32x16<S>,
        one: f32x16<S>,
    ) {
        let bg_c = f32x16::from_slice(s, dest);
        let mask_a = extract_masks(s, masks);
        let inv_src_a_mask_a = src_a.mul_add(-mask_a, one);

        let res = bg_c.mul_add(inv_src_a_mask_a, src_c * mask_a);
        res.store_slice(dest);
    }
}

#[inline(always)]
fn extract_masks<S: Simd>(simd: S, masks: &[u8; 4]) -> f32x16<S> {
    let mut base_mask = [
        masks[0] as f32,
        masks[1] as f32,
        masks[2] as f32,
        masks[3] as f32,
    ]
    .simd_into(simd);

    base_mask *= f32x4::splat(simd, 1.0 / 255.0);

    let res = f32x16::block_splat(base_mask);
    let zip_low = res.zip_low(res);

    zip_low.zip_low(zip_low)
}
