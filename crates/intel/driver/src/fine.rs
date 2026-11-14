use crate::base::vello_common::encode::{EncodedKind, EncodedPaint, RadialKind};
use crate::base::vello_common::fearless_simd::{Level, dispatch};
use crate::base::vello_common::mask::Mask;
use crate::base::vello_common::paint::{ImageSource, Paint, TintMode};
use crate::vello_cpu::detour::Positions;
use crate::vello_cpu::peniko::{Extend, ImageQuality};
use crate::base::{Gpu, Host};
use crate::vello_cpu::detour::{Cmd, Coarse};
use crate::vello_cpu::peniko::{BlendMode, Compose, Mix};

use crate::xe3::{Codegen, Context, Error, Kernel, Recorder};

const FINE: &str = include_str!("fine/fine.mlir");

const ATTRS: usize = 20;

const BLEND: usize = 14;

const MASK: usize = 15;

struct Blend(u32);

impl Blend {
    const SRC_OVER: Blend = Blend(1 << 4 | 3 << 6 | 1 << 9);
}

impl TryFrom<BlendMode> for Blend {
    type Error = Error;

    fn try_from(b: BlendMode) -> Result<Blend, Error> {
        if b == BlendMode::default() {
            return Ok(Blend(0));
        }
        let mix = match b.mix {
            Mix::Normal => 0,
            Mix::Multiply => 1,
            Mix::Screen => 2,
            Mix::Overlay => 3,
            Mix::Darken => 4,
            Mix::Lighten => 5,
            Mix::HardLight => 6,
            Mix::Difference => 7,
            Mix::Exclusion => 8,
            Mix::ColorDodge => 9,
            Mix::ColorBurn => 10,
            Mix::SoftLight => 11,
            Mix::Hue => 12,
            Mix::Saturation => 13,
            Mix::Color => 14,
            Mix::Luminosity => 15,
        };
        let (fa, fb, sat) = match b.compose {
            Compose::Clear => (0, 0, 0),
            Compose::Copy => (1, 0, 0),
            Compose::SrcOver => (1, 3, 0),
            Compose::DestOver => (3, 1, 0),
            Compose::Dest => (0, 1, 0),
            Compose::Xor => (3, 3, 0),
            Compose::SrcIn => (2, 0, 0),
            Compose::DestIn => (0, 2, 0),
            Compose::SrcOut => (3, 0, 0),
            Compose::DestOut => (0, 3, 0),
            Compose::SrcAtop => (2, 3, 0),
            Compose::DestAtop => (3, 2, 0),
            Compose::Plus | Compose::PlusLighter => (1, 1, 1),
        };
        Ok(Blend(mix | fa << 4 | fb << 6 | sat << 8 | 1 << 9))
    }
}

const BUCKET: u32 = 128;

const TILE: u16 = 4;

pub struct Fine {
    kernel: Kernel,
}

struct Packed {
    rows: Vec<u32>,
    cmds: Vec<u32>,
    depth: Vec<u32>,
    attrs: Vec<u32>,
    alphas: Vec<u8>,
    depth_max: u32,
    fixups: Vec<(usize, Texels)>,
}

#[derive(Clone, Copy, Debug)]
enum Texels {
    At(u64),
    Filter(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct GpuTexels {
    pub addr: u64,
    pub stride: u32,
    pub size: (u16, u16),
    pub straight: bool,
}

impl Packed {
    fn mask(base: usize, data: &mut Vec<u32>, m: &Mask) -> u32 {
        let at = (base + data.len()) as u32 + 1;
        data.push(u32::from(m.width()) | u32::from(m.height()) << 16);
        let bytes: Vec<u8> = (0..m.height()).flat_map(|y| (0..m.width()).map(move |x| m.sample(x, y))).collect();
        data.extend(bytes.chunks(4).map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        }));
        at
    }
}

pub struct Paints<'a> {
    pub paints: &'a [crate::vello_cpu::detour::PaintFillAttrs],
    pub layers: &'a [crate::vello_cpu::detour::LayerAttrs],
    pub encoded: &'a [EncodedPaint],
    pub filter_paints: &'a [EncodedPaint],
}

impl Packed {
    fn attrs(p: &Paints<'_>, images: &[GpuTexels], filters: &[(u32, (u16, u16))]) -> Result<(Vec<u32>, Vec<(usize, Texels)>), Error> {
        let pp = p;
        let paint_of = |i: usize| pp.encoded.get(i).or_else(|| pp.filter_paints.get(i - pp.encoded.len()));
        let n = pp.paints.len() + pp.layers.len();
        let mut attrs = Vec::with_capacity(n * ATTRS);
        let mut luts: Vec<u32> = Vec::new();
        let mut fixups = Vec::new();
        let level = Level::try_detect().unwrap_or(Level::baseline());
        for p in pp.paints {
            if p.thread_idx != 0 {
                return Err(Error::msg("fine: alphas of another thread"));
            }
            let blend = match (Blend::try_from(p.blend_mode)?, &p.mask) {
                (Blend(0), Some(_)) => Blend::SRC_OVER,
                (b, _) => b,
            };
            let first = attrs.len();
            match &p.paint {
                Paint::Solid(color) => {
                    attrs.extend([u32::from_le_bytes(color.as_premul_rgba8().to_u8_array()), p.draw_id]);
                    attrs.resize(attrs.len() + ATTRS - 2, 0);
                }
                Paint::Indexed(i) if let Some(EncodedPaint::Image(img)) = paint_of(i.index()) => {
                    let (texels, (w, h), stride) = match &img.source {
                        ImageSource::Pixmap(pixmap) => {
                            let at = (n * ATTRS + luts.len()) as u32;
                            luts.extend(pixmap.data().iter().map(|p| u32::from_le_bytes([p.r, p.g, p.b, p.a])));
                            (at * 4, (pixmap.width(), pixmap.height()), u32::from(pixmap.width()))
                        }
                        ImageSource::OpaqueId { id, .. } if let Some(&(image, size)) = filters.iter().find(|(f, _)| *f == id.as_u32()) => {
                            fixups.push((attrs.len(), Texels::Filter(image)));
                            (0, size, u32::from(size.0))
                        }
                        ImageSource::OpaqueId { id, .. } => {
                            let g = images.get(id.as_u32() as usize).ok_or_else(|| Error::msg(format!("fine: no GPU image {}", id.as_u32())))?;
                            if g.stride >= 1 << 31 {
                                return Err(Error::msg("fine: a GPU image's rows too long"));
                            }
                            fixups.push((attrs.len(), Texels::At(g.addr)));
                            (0, g.size, g.stride | u32::from(g.straight) << 31)
                        }
                        ImageSource::ExternalTexture { .. } => return Err(Error::msg("fine: an external texture (not yet)")),
                    };
                    let kind = match img.sampler.quality {
                        ImageQuality::Low => 4,
                        ImageQuality::Medium => 5,
                        ImageQuality::High => 6,
                    };
                    let (tint_mode, tint) = img.tint.as_ref().map_or((0, 0), |t| {
                        let [r, g, b, a] = t.color.premultiply().components;
                        let u8_of = |v: f32| (v * 255.0 + 0.5) as u8;
                        let mode = match t.mode {
                            TintMode::AlphaMask => 1,
                            TintMode::Multiply => 2,
                        };
                        (mode, u32::from_le_bytes([u8_of(r), u8_of(g), u8_of(b), u8_of(a)]))
                    });
                    let pos = Positions::of_image(img);
                    let extend = |e: Extend| match e {
                        Extend::Pad => 0,
                        Extend::Repeat => 1,
                        Extend::Reflect => 2,
                    };
                    let (ox, oy) = (u32::from(p.origin.0), u32::from(p.origin.1));
                    if ox > 0xffff || oy > 0x3fff {
                        return Err(Error::msg("fine: a paint origin too far"));
                    }
                    attrs.extend([texels, p.draw_id, kind, u32::from(w) | u32::from(h) << 16, pos.p0.0.to_bits(), pos.x_advance.0.to_bits(), pos.y_advance.0.to_bits(), extend(img.sampler.x_extend) | ox << 2 | oy << 18]);
                    attrs.extend([pos.p0.1.to_bits(), pos.x_advance.1.to_bits(), pos.y_advance.1.to_bits(), extend(img.sampler.y_extend) | tint_mode << 2, (1.0 / f32::from(w)).to_bits(), (1.0 / f32::from(h)).to_bits(), 0, 0]);
                    attrs.extend([tint, stride, 0, 0]);
                }
                Paint::Indexed(i) if let Some(EncodedPaint::BlurredRoundedRect(r)) = paint_of(i.index()) => {
                    let at = (n * ATTRS + luts.len()) as u32;
                    let pos = Positions::new(r.transform, r.x_advance, r.y_advance);
                    let [cr, cg, cb, ca] = r.color.as_premul_f32().components;
                    let f = [r.exponent, r.recip_exponent, r.scale, r.std_dev_inv, r.min_edge, r.w, r.h, r.width, r.height, r.r1];
                    luts.extend(f.map(f32::to_bits));
                    luts.push(u32::from(r.invert));
                    luts.extend([cr, cg, cb, ca, pos.p0.0, pos.p0.1, pos.x_advance.0, pos.x_advance.1, pos.y_advance.0, pos.y_advance.1].map(f32::to_bits));
                    attrs.extend([at, p.draw_id, 7]);
                    attrs.resize(attrs.len() + ATTRS - 3, 0);
                }
                Paint::Indexed(i) => {
                    let Some(EncodedPaint::Gradient(g)) = paint_of(i.index()) else { return Err(Error::msg("fine: an unknown paint")) };
                    let (kind, radial) = match &g.kind {
                        EncodedKind::Linear(_) => (1, [0, 0, 0]),
                        EncodedKind::Radial(RadialKind::Radial { bias, scale }) => (2, [0, bias.to_bits(), scale.to_bits()]),
                        EncodedKind::Radial(RadialKind::Strip { scaled_r0_squared }) => (2, [2, scaled_r0_squared.to_bits(), 0]),
                        EncodedKind::Radial(RadialKind::Focal { focal_data: f, fp0, fp1 }) => {
                            let negate = 1.0 - f.f_focal_x < 0.0;
                            let formula = match (f.is_focal_on_circle(), f.is_well_behaved(), f.is_swapped() || negate) {
                                (true, ..) => 1,
                                (false, true, _) => 0,
                                (false, false, true) => 2,
                                (false, false, false) => 3,
                            };
                            let flags = u32::from(negate) | u32::from(!f.is_natively_focal()) << 1 | u32::from(f.is_swapped()) << 2 | u32::from(!f.is_well_behaved()) << 3 | formula << 4;
                            (2, [1 | flags << 4, fp0.to_bits(), fp1.to_bits()])
                        }
                        EncodedKind::Sweep(k) => (3, [0, k.start_angle.to_bits(), k.inv_angle_delta.to_bits()]),
                    };
                    let undefined = g.has_undefined;
                    let (lut, scale): (Vec<u32>, f32) = dispatch!(level, simd => {
                        if undefined {
                            use crate::base::vello_common::fearless_simd::{SimdBase, SimdFloat, f32x4};
                            let l = g.f32_lut(simd);
                            let bytes = |e: &[f32; 4]| {
                                let v = f32x4::from_slice(simd, e).mul_add(f32x4::splat(simd, 255.0), f32x4::splat(simd, 0.5));
                                let mut out = [0f32; 4];
                                v.store_slice(&mut out);
                                u32::from_le_bytes(out.map(|c| c as u32 as u8))
                            };
                            (l.lut().iter().map(bytes).collect(), l.scale_factor())
                        } else {
                            let l = g.u8_lut(simd);
                            (l.lut().iter().map(|e| u32::from_le_bytes(*e)).collect(), l.scale_factor())
                        }
                    });
                    let last = (lut.len() as u32 - 1) | (u32::from(undefined) << 31);
                    let at = (n * ATTRS + luts.len()) as u32;
                    luts.extend(lut);
                    let pos = Positions::of(g);
                    let extend = match g.extend {
                        Extend::Pad => 0,
                        Extend::Repeat => 1,
                        Extend::Reflect => 2,
                    };
                    let (ox, oy) = (u32::from(p.origin.0), u32::from(p.origin.1));
                    if ox > 0xffff || oy > 0x3fff {
                        return Err(Error::msg("fine: a paint origin too far"));
                    }
                    attrs.extend([at, p.draw_id, kind, scale.to_bits(), pos.p0.0.to_bits(), pos.x_advance.0.to_bits(), pos.y_advance.0.to_bits(), extend | ox << 2 | oy << 18]);
                    attrs.extend([pos.p0.1.to_bits(), pos.x_advance.1.to_bits(), pos.y_advance.1.to_bits(), radial[0], radial[1], radial[2], 0, 0]);
                    attrs.extend([0, last, 0, 0]);
                }
            }
            attrs[first + BLEND] = blend.0;
            attrs[first + MASK] = p.mask.as_ref().map_or(0, |m| Packed::mask(n * ATTRS, &mut luts, m));
        }
        for l in pp.layers {
            attrs.push(l.opacity.mul_add(255.0, 0.5) as u8 as u32);
            attrs.resize(attrs.len() + ATTRS - 1, 0);
            let at = attrs.len() - ATTRS;
            attrs[at + BLEND] = Blend::try_from(l.blend_mode)?.0;
            attrs[at + MASK] = l.mask.as_ref().map_or(0, |m| Packed::mask(n * ATTRS, &mut luts, m));
        }
        attrs.extend(luts);
        Ok((attrs, fixups))
    }
}

impl Packed {
    fn new(c: &Coarse<'_>, images: &[GpuTexels], filters: &[(u32, (u16, u16))]) -> Result<Packed, Error> {
        let (mut attrs, fixups) = Packed::attrs(&Paints { paints: &c.paints, layers: &c.layers, encoded: c.encoded_paints, filter_paints: &c.filter_paints }, images, filters)?;
        let layer0 = c.paints.len() as u32;
        let (mut rows, mut cmds, mut depth, mut depth_max) = (Vec::new(), Vec::new(), Vec::new(), 0u32);
        let buckets = u32::from(c.width).div_ceil(BUCKET);
        let span = |s: crate::vello_cpu::detour::Span| u32::from(s.x) | u32::from(s.width) << 16;
        let overlaps = |s: crate::vello_cpu::detour::Span, b: u32| u32::from(s.x) < (b + 1) * BUCKET && u32::from(s.x) + u32::from(s.width) > b * BUCKET;
        for row in &c.rows {
            let mut d = 0u32;
            for cmd in &row.cmds {
                match cmd {
                    Cmd::PushBuf(_) => {
                        d += 1;
                        depth_max = depth_max.max(d);
                    }
                    Cmd::PopBuf => d = d.saturating_sub(1),
                    _ => {}
                }
            }
            for b in 0..buckets {
                let (first, dfirst) = (cmds.len() / 4, depth.len() / 2);
                for cmd in &row.cmds {
                    match *cmd {
                        Cmd::PaintFill { span: s, alpha, attrs } if overlaps(s, b) => cmds.extend([0, span(s), alpha.unwrap_or(u32::MAX), attrs]),
                        Cmd::LayerFill { span: s, alpha, attrs } if overlaps(s, b) => cmds.extend([1, span(s), alpha.unwrap_or(u32::MAX), layer0 + attrs]),
                        Cmd::PushBuf(_) => cmds.extend([2, 0, u32::MAX, 0]),
                        Cmd::PopBuf if cmds.len() > first * 4 && cmds[cmds.len() - 4] == 2 => cmds.truncate(cmds.len() - 4),
                        Cmd::PopBuf => cmds.extend([3, 0, u32::MAX, 0]),
                        _ => {}
                    }
                }
                for d in row.depth.iter().filter(|d| u32::from(d.buckets.start) <= b && b < u32::from(d.buckets.end)) {
                    depth.extend([u32::from(d.buckets.start) | u32::from(d.buckets.end) << 16, d.attrs]);
                }
                rows.extend([first as u32, (cmds.len() / 4 - first) as u32, dfirst as u32, (depth.len() / 2 - dfirst) as u32]);
            }
        }
        for v in [&mut cmds, &mut depth, &mut attrs] {
            if v.is_empty() {
                v.push(0);
            }
        }
        let mut alphas = c.alphas.to_vec();
        alphas.resize(alphas.len().next_multiple_of(4).max(4), 0);
        Ok(Packed { rows, cmds, depth, attrs, alphas, depth_max, fixups })
    }
}

impl Fine {
    pub fn new<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>) -> Result<Fine, Error> {
        Ok(Fine { kernel: cx.kernel(codegen, FINE, "fine")? })
    }

    pub fn draw<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, cx: &mut Context<'d, G, H>, coarse: &Coarse<'_>) -> Result<Vec<u8>, Error> {
        let tiles = Tiles::new(coarse, &[], &[])?;
        let mut buffer = cx.buffer(&vec![0; tiles.bytes()])?;
        let addr = buffer.addr;
        let at = tiles.write(buffer.bytes()?, addr, &[])?;
        let mut r = cx.record();
        self.record(&mut r, &at)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let from = (at.out - buffer.addr) as usize;
        Ok(buffer.bytes()?[from..][..tiles.pixels()].to_vec())
    }

    pub fn record<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, t: &TilesAt) -> Result<(), Error> {
        let [rows, cmds, depth, attrs, alphas, stack] = t.arrays;
        let (w, h) = t.size;
        r.dispatch(&self.kernel, [u32::from(w / TILE), u32::from(h / TILE)], &[rows, cmds, depth, attrs, alphas, t.out as u32, stack, u32::from(w) | t.depth_max << 16])
    }
}

pub struct Tiles {
    packed: Packed,
    size: (u16, u16),
}

#[derive(Clone, Copy, Debug)]
pub struct TilesAt {
    arrays: [u32; 6],
    pub out: u64,
    size: (u16, u16),
    depth_max: u32,
}

impl TilesAt {
    pub fn with_gpu(self, rows: u64, cmds: u64, depth: u64, alphas: u64) -> TilesAt {
        let mut arrays = self.arrays;
        arrays[..3].copy_from_slice(&[rows as u32, cmds as u32, depth as u32]);
        arrays[4] = alphas as u32;
        TilesAt { arrays, ..self }
    }
}

impl Tiles {
    pub fn new(coarse: &Coarse<'_>, images: &[GpuTexels], filters: &[(u32, (u16, u16))]) -> Result<Self, Error> {
        let (w, h) = (coarse.width, coarse.height);
        if !w.is_multiple_of(TILE) || !h.is_multiple_of(TILE) {
            return Err(Error::msg(format!("fine: {w}x{h}, not whole tiles (not yet)")));
        }
        let packed = Packed::new(coarse, images, filters)?;
        if packed.depth_max > u32::from(u16::MAX) {
            return Err(Error::msg("fine: layers too deep"));
        }
        Ok(Tiles { packed, size: (w, h) })
    }

    pub fn recorded(rec: &crate::vello_cpu::detour::Recording, encoded: &[EncodedPaint], images: &[GpuTexels]) -> Result<Self, Error> {
        use crate::vello_cpu::detour::Item;
        let (w, h) = (rec.width, rec.height);
        if !w.is_multiple_of(TILE) || !h.is_multiple_of(TILE) {
            return Err(Error::msg(format!("fine: {w}x{h}, not whole tiles (not yet)")));
        }
        let (mut paints, mut layers, mut depth, mut depth_max) = (Vec::new(), Vec::new(), 0u32, 0u32);
        for item in &rec.items {
            match item {
                Item::Fill { attrs, .. } => paints.push(attrs.clone()),
                Item::Push { .. } => {
                    depth += 1;
                    depth_max = depth_max.max(depth);
                }
                Item::Pop(l) => {
                    depth -= 1;
                    layers.push(l.clone());
                }
            }
        }
        let (mut attrs, fixups) = Packed::attrs(&Paints { paints: &paints, layers: &layers, encoded, filter_paints: &[] }, images, &[])?;
        if attrs.is_empty() {
            attrs.push(0);
        }
        let packed = Packed { rows: vec![0], cmds: vec![0], depth: vec![0], attrs, alphas: vec![0; 4], depth_max, fixups };
        Ok(Tiles { packed, size: (w, h) })
    }

    pub fn pixels(&self) -> usize {
        usize::from(self.size.0) * usize::from(self.size.1) * 4
    }

    fn parts(&self) -> [usize; 7] {
        let p = &self.packed;
        let tiles = usize::from(self.size.0 / TILE) * usize::from(self.size.1 / TILE);
        [p.rows.len() * 4, p.cmds.len() * 4, p.depth.len() * 4, p.attrs.len() * 4, p.alphas.len(), (tiles * p.depth_max.max(1) as usize * 64).max(4), self.pixels()]
    }

    pub fn bytes(&self) -> usize {
        self.parts().iter().map(|n| n.next_multiple_of(64)).sum()
    }

    pub fn write(&self, dst: &mut [u8], addr: u64, filtered: &[(u32, u64)]) -> Result<TilesAt, Error> {
        if dst.len() < self.bytes() {
            return Err(Error::msg("fine: no room for the tiles"));
        }
        let p = &self.packed;
        let words = |to: &mut [u8], v: &[u32]| to.as_chunks_mut::<4>().0.iter_mut().zip(v).for_each(|(t, w)| *t = w.to_le_bytes());
        let mut at = [0usize; 7];
        let mut next = 0;
        for (a, n) in at.iter_mut().zip(self.parts()) {
            *a = next;
            next += n.next_multiple_of(64);
        }
        let sizes = self.parts();
        let span = |k: usize| at[k]..at[k] + sizes[k];
        words(&mut dst[span(0)], &p.rows);
        words(&mut dst[span(1)], &p.cmds);
        words(&mut dst[span(2)], &p.depth);
        words(&mut dst[span(3)], &p.attrs);
        let attrs = (addr + at[3] as u64) as u32;
        for &(k, texels) in &p.fixups {
            let image = match texels {
                Texels::At(a) => a,
                Texels::Filter(id) => filtered.iter().find(|(f, _)| *f == id).map(|(_, a)| *a).ok_or_else(|| Error::msg(format!("fine: filter layer image {id} not placed")))?,
            };
            dst[at[3] + k * 4..][..4].copy_from_slice(&(image as u32).wrapping_sub(attrs).to_le_bytes());
        }
        dst[span(4)].copy_from_slice(&p.alphas);
        dst[span(5)].fill(0);
        dst[span(6)].fill(0);
        let gpu = |k: usize| addr + at[k] as u64;
        let arrays = std::array::from_fn(|k| gpu(k) as u32);
        Ok(TilesAt { arrays, out: gpu(6), size: self.size, depth_max: p.depth_max })
    }
}
