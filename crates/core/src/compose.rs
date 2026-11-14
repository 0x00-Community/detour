//! ⭐ A page as acde's compositor draws it (moved from detour-fallback's raster.rs / compose.rs, the same on every
//! backend): a list of [`Part`]s, each a layer drawn from its content's [`Scene`], placed on the window's canvas through
//! its CSS filter ([`Effect`]), under its drop shadows ([`Shadow`]), over its frosted glass ([`Backdrop`]). Blurs are dual
//! Kawase (1 ..= 5 levels, [`Effect::levels`]).

#[cfg(unix)]
use std::os::fd::{BorrowedFd, OwnedFd};
use std::time::Duration;

use crate::base::Scene;

/// A rectangle of whole pixels (x, y may be negative; empty when w or h ≤ 0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Px {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Px {
    pub fn sized((w, h): (u32, u32)) -> Self {
        Self { x: 0, y: 0, w: w as i32, h: h as i32 }
    }

    /// The smallest whole-pixel rectangle covering `[x, y, w, h]`.
    pub fn covering([x, y, w, h]: [f32; 4]) -> Self {
        let (x0, y0) = (x.floor() as i32, y.floor() as i32);
        Self { x: x0, y: y0, w: (x + w).ceil() as i32 - x0, h: (y + h).ceil() as i32 - y0 }
    }

    pub fn grown(self, m: i32) -> Self {
        Self { x: self.x - m, y: self.y - m, w: self.w + 2 * m, h: self.h + 2 * m }
    }

    pub fn intersection(self, o: Self) -> Option<Self> {
        let (x0, y0) = (self.x.max(o.x), self.y.max(o.y));
        let (x1, y1) = ((self.x + self.w).min(o.x + o.w), (self.y + self.h).min(o.y + o.h));
        (x1 > x0 && y1 > y0).then_some(Self { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
    }

    pub fn union(self, o: Self) -> Self {
        let (x0, y0) = (self.x.min(o.x), self.y.min(o.y));
        let (x1, y1) = ((self.x + self.w).max(o.x + o.w), (self.y + self.h).max(o.y + o.h));
        Self { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
    }

    pub fn contains(self, o: Self) -> bool {
        o.x >= self.x && o.y >= self.y && o.x + o.w <= self.x + self.w && o.y + o.h <= self.y + self.h
    }

    pub fn area(self) -> i64 {
        i64::from(self.w.max(0)) * i64::from(self.h.max(0))
    }

    pub fn overlaps(self, o: Self) -> bool {
        self.intersection(o).is_some()
    }

    /// x, y, w, h clamped at 0, as the compositor's regions take them.
    pub fn xywh(self) -> [u32; 4] {
        [self.x, self.y, self.w, self.h].map(|v| v.max(0) as u32)
    }

    pub fn of([x, y, w, h]: [u32; 4]) -> Self {
        Self { x: x as i32, y: y as i32, w: w as i32, h: h as i32 }
    }
}

/// Frosted glass (`backdrop-filter`): what is under the rounded rectangle `rect` (`radius`), through `effect` (its blur,
/// then its colour matrix).
#[derive(Clone, Copy, Debug)]
pub struct Backdrop {
    pub rect: [f32; 4],
    pub radius: f32,
    pub effect: Effect,
}

impl Backdrop {
    pub fn levels(&self) -> u32 {
        self.effect.levels()
    }
}

/// A `drop-shadow()`: the layer's alpha offset by (dx, dy), blurred, in `color` (straight alpha, 0 ..= 1).
#[derive(Clone, Copy, Debug)]
pub struct Shadow {
    pub dx: f32,
    pub dy: f32,
    pub blur: f32,
    pub color: [f32; 4],
}

/// A layer's CSS filter folded into one blur (σ, none at 0) and one colour matrix (4 rows of 5, the fifth the offset)
/// on straight colour in 0 ..= 1.
#[derive(Clone, Copy, Debug, Default)]
pub struct Effect {
    pub blur: f32,
    pub matrix: Option<[f32; 20]>,
}

impl Effect {
    pub const IDENTITY: [f32; 20] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    pub fn is_none(&self) -> bool {
        self.blur <= 0.0 && self.matrix.is_none()
    }

    pub fn blur(&mut self, sigma: f32) {
        self.blur = self.blur.hypot(sigma);
    }

    pub fn matrix(&mut self, m: [f32; 20]) {
        let a = self.matrix.unwrap_or(Self::IDENTITY);
        self.matrix = Some(core::array::from_fn(|k| {
            let (i, j) = (k / 5, k % 5);
            (0..4).map(|t| m[i * 5 + t] * a[t * 5 + j]).sum::<f32>() + if j == 4 { m[i * 5 + 4] } else { 0.0 }
        }));
    }

    /// The matrix column-major (4 × 4) and its offsets.
    pub fn columns(&self) -> ([f32; 16], [f32; 4]) {
        let m = self.matrix.unwrap_or(Self::IDENTITY);
        (core::array::from_fn(|k| m[(k % 4) * 5 + k / 4]), [m[4], m[9], m[14], m[19]])
    }
}

/// `levels()`: the Kawase levels of a blur of σ `self.blur`, `clamp(ceil(log2(max(σ, 1) / 2)), 1, 5)`.
macro_rules! levels {
    ($($t:ty),*) => {$(
        impl $t {
            pub fn levels(&self) -> u32 {
                ((self.blur.max(1.0) / 2.0).log2().ceil() as u32).clamp(1, 5)
            }
        }
    )*};
}

levels!(Effect, Shadow);

impl Effect {
    /// How far `shadows` and this effect's blur reach beyond what they are drawn from, in pixels.
    pub fn reach(&self, shadows: &[Shadow]) -> i32 {
        let shadows = shadows.iter().map(|s| (s.dx.abs().max(s.dy.abs()) + 4.0 * s.blur).ceil() as i32).max().unwrap_or(0);
        shadows.max((3.0 * self.blur).ceil() as i32)
    }
}

/// What a part draws: the scene of its content clipped to a region of the canvas.
pub trait Content {
    fn scene_for(&self, region: Px) -> Scene;
}

/// One layer of a page and how it is composited.
pub struct Part<C> {
    pub content: C,
    pub backdrop: Option<Backdrop>,
    pub shadows: Vec<Shadow>,
    pub effect: Effect,
    /// where the part can draw, on the canvas; `None`: anywhere
    pub bounds: Option<Px>,
}

impl<C> Part<C> {
    pub fn plain(content: C, bounds: Option<Px>) -> Self {
        Self { content, backdrop: None, shadows: Vec::new(), effect: Effect::default(), bounds }
    }

    /// Where the part is drawn for each target region (with room for its shadows and blur) and where it is clipped
    /// to, within `full`.
    pub fn pieces(&self, targets: &[Px], full: Px) -> Vec<(Px, Px)> {
        let margin = self.effect.reach(&self.shadows);
        targets
            .iter()
            .filter_map(|t| {
                let clip = self.bounds.map_or(Some(*t), |b| b.intersection(*t))?;
                let drawn = self.bounds.map_or(Some(*t), |b| b.intersection(t.grown(margin)))?.intersection(full)?;
                Some((drawn, clip))
            })
            .collect()
    }
}

/// One window's frame: its parts drawn into canvas `next` of plane `plane`, only in `targets` (all of it when `None`).
pub struct Pass<C> {
    pub plane: usize,
    pub size: (u32, u32),
    pub parts: Vec<Part<C>>,
    pub next: usize,
    pub targets: Option<Vec<Px>>,
}

/// One plane of an RGBA8 dma-buf (`DRM_FORMAT_ABGR8888`): its fd, where the pixels start, a row's bytes, the modifier.
#[cfg(unix)]
#[derive(Clone, Copy)]
pub struct Dmabuf<'f> {
    pub fd: BorrowedFd<'f>,
    pub offset: u32,
    pub stride: u32,
    pub modifier: u64,
}

#[cfg(unix)]
impl Dmabuf<'_> {
    /// `DRM_FORMAT_MOD_LINEAR`.
    pub const LINEAR: u64 = 0;
}

/// A submitted frame: its sync_file (when the backend could export one; otherwise the frame was waited for), how long the
/// previous frame was waited for, how long recording took.
#[cfg(unix)]
pub struct Submitted {
    pub fence: Option<OwnedFd>,
    pub waited: Duration,
    pub recorded: Duration,
}
