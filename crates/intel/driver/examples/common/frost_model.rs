//! The definition of `detour_intel::Frost`'s effects (its module docs) as plain Rust, byte for byte: the reference of
//! the `frost` and `compose` examples. Images are RGBA8, premultiplied unless said otherwise.

#![allow(dead_code, reason = "each example uses part of it")]

use detour::base::{Backdrop, Effect, Shadow};
use detour::xe3::Canvas;

pub const INV: f32 = 1.0 / 255.0;

/// An RGBA8 image, rows packed.
#[derive(Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[u8; 4]>,
}

pub fn pack(v: f32) -> u8 {
    (v + 0.5).floor().clamp(0.0, 255.0) as u8
}

pub fn lerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    std::array::from_fn(|c| t.mul_add(b[c] - a[c], a[c]))
}

/// `s` over `d`.
pub fn over(s: [f32; 4], d: [f32; 4]) -> [u8; 4] {
    let k = (255.0 - s[3]) * INV;
    std::array::from_fn(|c| pack(s[c] + d[c] * k))
}

/// `backdrop-filter: blur(blur) saturate(saturate) brightness(brightness)` as an effect.
pub fn frosted(blur: f32, saturate: f32, brightness: f32) -> Effect {
    let mut effect = Effect { blur, matrix: None };
    effect.matrix(detour::base::ColorMatrix::saturate(saturate).0);
    effect.matrix(detour::base::ColorMatrix::brightness(brightness).0);
    effect
}

pub fn premultiply(c: [f32; 4]) -> [f32; 4] {
    let k = c[3] * INV;
    [c[0] * k, c[1] * k, c[2] * k, c[3]]
}

pub const DOWN: [(i32, i32, f32); 5] = [(0, 0, 4.0), (-1, -1, 1.0), (1, 1, 1.0), (1, -1, 1.0), (-1, 1, 1.0)];
pub const UP: [(i32, i32, f32); 8] = [(-2, 0, 1.0), (-1, 1, 2.0), (0, 2, 1.0), (1, 1, 2.0), (2, 0, 1.0), (1, -1, 2.0), (0, -2, 1.0), (-1, -1, 2.0)];

impl Image {
    /// The texel at (x, y) clamped to the image.
    pub fn texel(&self, x: i32, y: i32) -> [f32; 4] {
        let (x, y) = (x.clamp(0, self.w as i32 - 1), y.clamp(0, self.h as i32 - 1));
        self.px[y as usize * self.w + x as usize].map(f32::from)
    }

    /// Tap (dx, dy) around (x0, y0) with the fractions (fx, fy).
    pub fn tap(&self, (x0, fx): (i32, f32), (y0, fy): (i32, f32), dx: i32, dy: i32) -> [f32; 4] {
        let (x, y) = (x0 + dx, y0 + dy);
        lerp(lerp(self.texel(x, y), self.texel(x + 1, y), fx), lerp(self.texel(x, y + 1), self.texel(x + 1, y + 1), fx), fy)
    }

    pub fn split(p: f32) -> (i32, f32) {
        let fl = p.floor();
        (fl as i32, p - fl)
    }

    pub fn sample(&self, x: f32, y: f32) -> [f32; 4] {
        self.tap(Self::split(x), Self::split(y), 0, 0)
    }

    /// One Kawase pass into `w` x `h`: taps (dx, dy, weight) summed in order, then normalized.
    pub fn pass(&self, w: usize, h: usize, taps: &[(i32, i32, f32)], norm: impl Fn(f32) -> f32) -> Image {
        let (kx, ky) = (self.w as f32 / w as f32, self.h as f32 / h as f32);
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let cx = Self::split((x as f32 + 0.5) * kx - 0.5);
                let cy = Self::split((y as f32 + 0.5) * ky - 0.5);
                let mut sum: Option<[f32; 4]> = None;
                for &(dx, dy, k) in taps {
                    let t = self.tap(cx, cy, dx, dy);
                    sum = Some(match sum {
                        None if k == 1.0 => t,
                        None => t.map(|v| v * k),
                        Some(a) if k == 1.0 => std::array::from_fn(|c| a[c] + t[c]),
                        Some(a) => std::array::from_fn(|c| t[c].mul_add(k, a[c])),
                    });
                }
                px.push(sum.expect("taps").map(|v| pack(norm(v))));
            }
        }
        Image { w, h, px }
    }

    /// Blurred through `n` levels.
    pub fn blur(&self, n: usize) -> Image {
        let mut levels = vec![self.clone()];
        for i in 1..=n {
            levels.push(levels[i - 1].pass((self.w >> i).max(1), (self.h >> i).max(1), &DOWN, |v| v * 0.125));
        }
        for i in (0..n).rev() {
            levels[i] = levels[i + 1].pass(levels[i].w, levels[i].h, &UP, |v| v * (1.0 / 12.0));
        }
        levels.swap_remove(0)
    }

    pub fn crop(&self, [x0, y0, w, h]: [usize; 4]) -> Image {
        Image { w, h, px: (0..h).flat_map(|y| self.px[(y0 + y) * self.w + x0..][..w].to_vec()).collect() }
    }

    pub fn canvas(&self) -> Canvas {
        Canvas { addr: 0, size: [self.w as u32, self.h as u32], stride: self.w as u32 }
    }

    /// `f` over every pixel of `[x, y, w, h]`.
    pub fn blend(&mut self, [x0, y0, w, h]: [u32; 4], f: impl Fn(usize, usize) -> [f32; 4]) {
        for y in y0 as usize..(y0 + h) as usize {
            for x in x0 as usize..(x0 + w) as usize {
                let d = self.texel(x as i32, y as i32);
                self.px[y * self.w + x] = over(f(x, y), d);
            }
        }
    }

    pub fn backdrop(&mut self, under: Option<&Image>, b: &Backdrop, clips: &[[u32; 4]]) {
        let Some(region) = self.canvas().region(b.rect) else { return };
        let pixels: Vec<[u32; 4]> = clips.iter().filter_map(|c| clip_to(intersect(region, *c), [self.w as u32, self.h as u32])).collect();
        let [x0, y0, w, h] = region.map(|v| v as usize);
        let mut src = self.crop([x0, y0, w, h]);
        if let Some(u) = under {
            let u = u.crop([x0, y0, w, h]);
            src.px = src.px.iter().zip(&u.px).map(|(c, u)| over(c.map(f32::from), u.map(f32::from))).collect();
        }
        let blurred = src.blur(b.levels() as usize);
        let [rx, ry, rw, rh] = b.rect;
        let (irw, irh) = (1.0 / rw, 1.0 / rh);
        let (hx, hy) = (rw * 0.5, rh * 0.5);
        let rr = b.radius.min(hx.min(hy));
        let m = b.effect.matrix.unwrap_or(Effect::IDENTITY);
        let tone_at = |x: usize, y: usize| {
            let (lx, ly) = (x as f32 + 0.5 - rx, y as f32 + 0.5 - ry);
            let c = blurred.sample((lx * irw).clamp(0.0, 1.0) * w as f32 - 0.5, (ly * irh).clamp(0.0, 1.0) * h as f32 - 0.5);
            let c = Self::matrixed(&m, c);
            let qx = ((lx - hx).abs() - hx) + rr;
            let qy = ((ly - hy).abs() - hy) + rr;
            let (mx, my) = (qx.max(0.0), qy.max(0.0));
            let d = ((mx * mx + my * my).sqrt() + qx.max(qy).min(0.0)) - rr;
            let mask = (0.5 - d).clamp(0.0, 1.0);
            c.map(|v| (v * mask).clamp(0.0, 255.0))
        };
        for p in pixels {
            self.blend(p, tone_at);
        }
    }

    /// The layer's w x h premultiplied.
    pub fn flat(layer: &Image, premultiplied: bool, w: usize, h: usize) -> Image {
        let mut flat = layer.crop([0, 0, w, h]);
        if !premultiplied {
            flat.px = flat.px.iter().map(|c| premultiply(c.map(f32::from)).map(pack)).collect();
        }
        flat
    }

    pub fn filtered(&mut self, layer: &Image, premultiplied: bool, e: &Effect, [x, y, w, h]: [u32; 4], clip: [u32; 4]) {
        let Some(pixels) = clip_to(clip, [self.w as u32, self.h as u32]) else { return };
        if w < 2 || h < 2 {
            return;
        }
        let flat = Self::flat(layer, premultiplied, w as usize, h as usize);
        let src = if e.blur > 0.0 { flat.blur(e.levels() as usize) } else { flat };
        let m = e.matrix.unwrap_or(Effect::IDENTITY);
        self.blend(pixels, |cx, cy| Self::matrixed(&m, src.texel(cx as i32 - x as i32, cy as i32 - y as i32)));
    }

    /// A premultiplied texel (0 ..= 255) made straight, through the colour matrix `m`, clamped, premultiplied again.
    fn matrixed(m: &[f32; 20], p: [f32; 4]) -> [f32; 4] {
        let c = if p[3] == 0.0 {
            [0.0; 4]
        } else {
            let k = 255.0 / p[3];
            [p[0] * k, p[1] * k, p[2] * k, p[3]]
        };
        let row = |i: usize| ((((m[i * 5] * c[0] + m[i * 5 + 1] * c[1]) + m[i * 5 + 2] * c[2]) + m[i * 5 + 3] * c[3]) + m[i * 5 + 4] * 255.0).clamp(0.0, 255.0);
        premultiply([row(0), row(1), row(2), row(3)])
    }

    /// `layer` with its top left at `origin`, over the canvas inside `clip`.
    pub fn place(&mut self, layer: &Image, [ox, oy]: [u32; 2], clip: [u32; 4]) {
        let Some(pixels) = clip_to(clip, [self.w as u32, self.h as u32]) else { return };
        self.blend(pixels, |x, y| layer.texel(x as i32 - ox as i32, y as i32 - oy as i32));
    }

    /// `rect` transparent.
    pub fn clear(&mut self, rect: [u32; 4]) {
        let Some([x0, y0, w, h]) = clip_to(rect, [self.w as u32, self.h as u32]) else { return };
        for y in y0..y0 + h {
            self.px[y as usize * self.w..][x0 as usize..(x0 + w) as usize].fill([0; 4]);
        }
    }

    pub fn shadow(&mut self, layer: &Image, s: &Shadow, [x, y, w, h]: [u32; 4], clip: [u32; 4]) {
        if w < 2 || h < 2 {
            return;
        }
        let (ex, ey) = (x as f32 + s.dx, y as f32 + s.dy);
        let Some(shifted) = self.canvas().region([ex, ey, w as f32, h as f32]) else { return };
        let Some(pixels) = clip_to(intersect(shifted, clip), [self.w as u32, self.h as u32]) else { return };
        let blurred = layer.crop([0, 0, w as usize, h as usize]).blur(s.levels() as usize);
        self.blend(pixels, |cx, cy| {
            let (sx, sy) = (cx as f32 - ex, cy as f32 - ey);
            let inside = |p: f32, n: u32| (0.0..=n as f32).contains(&(p + 0.5));
            let a = if inside(sx, w) && inside(sy, h) { blurred.sample(sx, sy)[3] * s.color[3] } else { 0.0 };
            [s.color[0] * a, s.color[1] * a, s.color[2] * a, a].map(|v| v.clamp(0.0, 255.0))
        });
    }
}

pub fn intersect(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    let (x0, y0) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x1, y1) = ((a[0] + a[2]).min(b[0] + b[2]), (a[1] + a[3]).min(b[1] + b[3]));
    [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
}

/// `r` ∩ the canvas, if not empty.
pub fn clip_to(r: [u32; 4], [w, h]: [u32; 2]) -> Option<[u32; 4]> {
    let r = intersect(r, [0, 0, w, h]);
    (r[2] > 0 && r[3] > 0).then_some(r)
}

