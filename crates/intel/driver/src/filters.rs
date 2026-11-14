use crate::base::vello_common::filter::drop_shadow::DropShadow;
use crate::base::vello_common::filter::gaussian_blur::GaussianBlur;
use crate::base::vello_common::filter_effects::EdgeMode;
use crate::base::{Gpu, Host};

use crate::xe3::{Codegen, Context, Error, Kernel, Recorder};

const BLUR: &str = include_str!("filters/blur.mlir");
const SHADOW: &str = include_str!("filters/shadow.mlir");

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub addr: u32,
    pub stride: u32,
}

pub struct Filters {
    blur_x: Kernel,
    blur_y: Kernel,
    down_x: Kernel,
    down_y: Kernel,
    up_x: Kernel,
    up_y: Kernel,
    offset: Kernel,
    shadow: Kernel,
}

impl Filters {
    pub fn new<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>) -> Result<Self, Error> {
        let mut k = |src: &str, name: &str| cx.kernel(codegen, src, name).map_err(|e| Error::msg(format!("@{name}: {e}")));
        Ok(Filters {
            blur_x: k(BLUR, "blur_x")?,
            blur_y: k(BLUR, "blur_y")?,
            down_x: k(BLUR, "down_x")?,
            down_y: k(BLUR, "down_y")?,
            up_x: k(BLUR, "up_x")?,
            up_y: k(BLUR, "up_y")?,
            offset: k(SHADOW, "offset")?,
            shadow: k(SHADOW, "shadow")?,
        })
    }

    pub fn weights(plan: &GaussianBlur) -> usize {
        (usize::from(plan.kernel_size) * 4).next_multiple_of(64)
    }

    pub fn write_weights(plan: &GaussianBlur, dst: &mut [u8]) {
        for (to, k) in dst.as_chunks_mut::<4>().0.iter_mut().zip(&plan.kernel[..usize::from(plan.kernel_size)]) {
            *to = k.to_le_bytes();
        }
    }

    #[allow(clippy::too_many_arguments, reason = "the images a blur goes through")]
    pub fn blur<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, src: Plane, dst: Plane, (w, h): (u32, u32), plan: &GaussianBlur, weights: u32, [a, b]: [Plane; 2]) -> Result<(), Error> {
        if plan.edge_mode != EdgeMode::None {
            return Err(Error::msg(format!("filters: a blur with edge mode {:?} (not yet)", plan.edge_mode)));
        }
        let wh = |w: u32, h: u32| w | h << 16;
        let groups = |w: u32, h: u32| [w.div_ceil(16), h];
        let klen = u32::from(plan.kernel_size);
        let pass = |r: &mut Recorder<'_, 'd, G, H>, k: &Kernel, (w, h): (u32, u32), s: Plane, d: Plane, rest: &[u32]| {
            let mut p = vec![s.addr, s.stride, d.addr, d.stride];
            p.extend_from_slice(rest);
            r.dispatch(k, groups(w, h), &p)
        };
        if plan.std_deviation <= 0.0 {
            return Err(Error::msg("filters: a blur of σ 0 (nothing to do: the caller copies)"));
        }
        if plan.n_decimations == 0 {
            pass(r, &self.blur_x, (w, h), src, a, &[wh(w, h), weights, klen])?;
            return pass(r, &self.blur_y, (w, h), a, dst, &[wh(w, h), weights, klen]);
        }
        let mut sizes = Vec::with_capacity(plan.n_decimations);
        let (mut cw, mut ch) = (w, h);
        let mut cur = src;
        for _ in 0..plan.n_decimations {
            sizes.push((cw, ch));
            let dw = cw.div_ceil(2);
            pass(r, &self.down_x, (dw, ch), cur, a, &[wh(cw, ch)])?;
            let dh = ch.div_ceil(2);
            pass(r, &self.down_y, (dw, dh), a, b, &[wh(dw, ch)])?;
            (cw, ch, cur) = (dw, dh, b);
        }
        pass(r, &self.blur_x, (cw, ch), b, a, &[wh(cw, ch), weights, klen])?;
        pass(r, &self.blur_y, (cw, ch), a, b, &[wh(cw, ch), weights, klen])?;
        for (i, (tw, th)) in sizes.into_iter().rev().enumerate() {
            let uw = (cw * 2).min(tw);
            pass(r, &self.up_x, (uw, ch), b, a, &[wh(cw, ch), uw])?;
            let uh = (ch * 2).min(th);
            let to = if i + 1 == plan.n_decimations { dst } else { b };
            pass(r, &self.up_y, (uw, uh), a, to, &[wh(uw, ch), uh])?;
            (cw, ch) = (uw, uh);
        }
        Ok(())
    }

    pub fn shadow_blur(d: &DropShadow) -> GaussianBlur {
        GaussianBlur { std_deviation: d.std_deviation, n_decimations: d.n_decimations, kernel: d.kernel, kernel_size: d.kernel_size, edge_mode: d.edge_mode }
    }

    #[allow(clippy::too_many_arguments, reason = "the images a drop shadow goes through")]
    pub fn drop_shadow<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, src: Plane, dst: Plane, (w, h): (u32, u32), d: &DropShadow, weights: u32, [s0, a, b]: [Plane; 3]) -> Result<(), Error> {
        if d.edge_mode != EdgeMode::None {
            return Err(Error::msg(format!("filters: a drop shadow with edge mode {:?} (not yet)", d.edge_mode)));
        }
        let wh = w | h << 16;
        let groups = [w.div_ceil(16), h];
        let (dx, dy) = (d.dx.round() as i32, d.dy.round() as i32);
        let mut shadow = src;
        if dx != 0 || dy != 0 {
            r.dispatch(&self.offset, groups, &[src.addr, src.stride, s0.addr, s0.stride, wh, dx as u32, dy as u32])?;
            shadow = s0;
        }
        if d.std_deviation > 0.0 {
            self.blur(r, shadow, s0, (w, h), &Self::shadow_blur(d), weights, [a, b])?;
            shadow = s0;
        }
        let [cr, cg, cb, ca] = d.color.components;
        let byte = |c: f32| u32::from((c * 255.0).round() as u8);
        let rgb = byte(cr) | byte(cg) << 8 | byte(cb) << 16;
        if src.stride != shadow.stride || src.stride != dst.stride {
            return Err(Error::msg("filters: a drop shadow's images of different strides"));
        }
        r.dispatch(&self.shadow, groups, &[src.addr, shadow.addr, dst.addr, src.stride, wh, rgb | u32::from(d.composite_original) << 24, ca.to_bits()])
    }
}
