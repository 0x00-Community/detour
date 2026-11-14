//! CSS `filter` / `backdrop-filter`: a list of effects applied in order (Filter Effects Module Level 1).
//!
//! Only the CSS filter functions: blur, drop-shadow, and the color matrices of opacity / invert / brightness / contrast /
//! hue-rotate / saturate / sepia / grayscale. The matrices are anyrender 0.14's (MIT OR Apache-2.0), which follow the
//! spec's equivalents. No SVG filter graphs: acde does not render `url()` filters.

use kurbo::{Affine, Rect};
use peniko::color::{AlphaColor, Srgb};

/// ⭐ Effects applied one after another, with the room they need around what they filter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Filter {
    effects: Vec<FilterEffect>,
    /// The union of the effects' expansions, in user space, centered on the origin.
    expansion: Rect,
}

impl FromIterator<FilterEffect> for Filter {
    fn from_iter<I: IntoIterator<Item = FilterEffect>>(effects: I) -> Self {
        let effects: Vec<FilterEffect> = effects.into_iter().collect();
        let expansion = effects.iter().fold(Rect::ZERO, |all, e| all.union(e.expansion()));
        Self { effects, expansion }
    }
}

impl Filter {
    pub fn effects(&self) -> &[FilterEffect] {
        &self.effects
    }

    /// How far past what it filters the filter reaches, in user space (x0 / y0 negative).
    pub fn expansion(&self) -> Rect {
        self.expansion
    }

    /// [`expansion`](Self::expansion) in device space under `transform`.
    pub fn bounds_expansion(&self, transform: &Affine) -> Rect {
        transform.transform_rect_bbox(self.expansion)
    }

    /// [`bounds_expansion`](Self::bounds_expansion) under the linear part of `transform` (no translation).
    pub fn linear_bounds_expansion(&self, transform: &Affine) -> Rect {
        let [a, b, c, d, _, _] = transform.as_coeffs();
        self.bounds_expansion(&Affine::new([a, b, c, d, 0.0, 0.0]))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FilterEffect {
    GaussianBlur(GaussianBlur),
    DropShadow(DropShadow),
    ColorMatrix(ColorMatrix),
}

/// `blur()`: a gaussian of standard deviation σ; it reaches about 3σ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBlur {
    pub std_deviation: f32,
}

/// `drop-shadow()`: the content's alpha, blurred, offset and colored, under the content.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropShadow {
    pub dx: f32,
    pub dy: f32,
    pub std_deviation: f32,
    pub color: AlphaColor<Srgb>,
}

/// A 4×5 color matrix: rows R, G, B, A; columns R, G, B, A, offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMatrix(pub [f32; 20]);

impl From<GaussianBlur> for FilterEffect {
    fn from(blur: GaussianBlur) -> Self {
        Self::GaussianBlur(blur)
    }
}

impl From<DropShadow> for FilterEffect {
    fn from(shadow: DropShadow) -> Self {
        Self::DropShadow(shadow)
    }
}

impl From<ColorMatrix> for FilterEffect {
    fn from(matrix: ColorMatrix) -> Self {
        Self::ColorMatrix(matrix)
    }
}

impl FilterEffect {
    /// How far past its input the effect reaches, in user space, centered on the origin.
    pub fn expansion(&self) -> Rect {
        match self {
            Self::GaussianBlur(b) => {
                let r = f64::from(b.std_deviation * 3.0);
                Rect::new(-r, -r, r, r)
            }
            Self::DropShadow(s) => {
                let (r, dx, dy) = (f64::from(s.std_deviation * 3.0), f64::from(s.dx), f64::from(s.dy));
                Rect::new(-(r + (-dx).max(0.0)), -(r + (-dy).max(0.0)), r + dx.max(0.0), r + dy.max(0.0))
            }
            Self::ColorMatrix(_) => Rect::ZERO,
        }
    }
}

const LUMA: [f32; 3] = [0.213, 0.715, 0.072];

impl ColorMatrix {
    pub const IDENTITY: Self = Self::diagonal(1.0, 1.0, [0.0; 3]);

    /// R, G, B scaled by `scale`, alpha by `alpha`, `offset` added to R, G, B.
    const fn diagonal(scale: f32, alpha: f32, [r, g, b]: [f32; 3]) -> Self {
        #[rustfmt::skip]
        let m = [
            scale, 0.0, 0.0, 0.0, r,
            0.0, scale, 0.0, 0.0, g,
            0.0, 0.0, scale, 0.0, b,
            0.0, 0.0, 0.0, alpha, 0.0,
        ];
        Self(m)
    }

    /// A 3×3 RGB matrix, alpha kept.
    #[rustfmt::skip]
    const fn rgb([r, g, b]: [[f32; 3]; 3]) -> Self {
        Self([
            r[0], r[1], r[2], 0.0, 0.0,
            g[0], g[1], g[2], 0.0, 0.0,
            b[0], b[1], b[2], 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
        ])
    }

    /// <https://drafts.fxtf.org/filter-effects/#opacityEquivalent>
    pub fn opacity(amount: f32) -> Self {
        Self::diagonal(1.0, amount, [0.0; 3])
    }

    /// <https://drafts.fxtf.org/filter-effects/#invertEquivalent>
    pub fn invert(amount: f32) -> Self {
        Self::diagonal(1.0 - 2.0 * amount, 1.0, [amount; 3])
    }

    /// <https://drafts.fxtf.org/filter-effects/#brightnessEquivalent>
    pub fn brightness(amount: f32) -> Self {
        Self::diagonal(amount, 1.0, [0.0; 3])
    }

    /// <https://drafts.fxtf.org/filter-effects/#contrastEquivalent>
    pub fn contrast(amount: f32) -> Self {
        Self::diagonal(amount, 1.0, [0.5 - 0.5 * amount; 3])
    }

    /// <https://drafts.fxtf.org/filter-effects/#feColorMatrixElement> (hueRotate)
    pub fn hue_rotate(radians: f32) -> Self {
        let (sin, cos) = radians.sin_cos();
        let [lr, lg, lb] = LUMA;
        Self::rgb([
            [lr + cos * (1.0 - lr) - sin * lr, lg - cos * lg - sin * lg, lb - cos * lb + sin * (1.0 - lb)],
            [lr - cos * lr + sin * 0.143, lg + cos * (1.0 - lg) + sin * 0.140, lb - cos * lb - sin * 0.283],
            [lr - cos * lr - sin * (1.0 - lr), lg - cos * lg + sin * lg, lb + cos * (1.0 - lb) + sin * lb],
        ])
    }

    /// <https://drafts.fxtf.org/filter-effects/#feColorMatrixElement> (saturate)
    pub fn saturate(amount: f32) -> Self {
        let [lr, lg, lb] = LUMA;
        Self::rgb([
            [lr + amount * (1.0 - lr), lg - amount * lg, lb - amount * lb],
            [lr - amount * lr, lg + amount * (1.0 - lg), lb - amount * lb],
            [lr - amount * lr, lg - amount * lg, lb + amount * (1.0 - lb)],
        ])
    }

    /// <https://www.w3.org/TR/filter-effects-1/#sepiaEquivalent>
    pub fn sepia(amount: f32) -> Self {
        let k = 1.0 - amount;
        Self::rgb([
            [0.393 + 0.607 * k, 0.769 - 0.769 * k, 0.189 - 0.189 * k],
            [0.349 - 0.349 * k, 0.686 + 0.314 * k, 0.168 - 0.168 * k],
            [0.272 - 0.272 * k, 0.534 - 0.534 * k, 0.131 + 0.869 * k],
        ])
    }

    /// <https://www.w3.org/TR/filter-effects-1/#grayscaleEquivalent>
    pub fn grayscale(amount: f32) -> Self {
        let k = 1.0 - amount;
        Self::rgb([
            [0.2126 + 0.7874 * k, 0.7152 - 0.7152 * k, 0.0722 - 0.0722 * k],
            [0.2126 - 0.2126 * k, 0.7152 + 0.2848 * k, 0.0722 - 0.0722 * k],
            [0.2126 - 0.2126 * k, 0.7152 - 0.7152 * k, 0.0722 + 0.9278 * k],
        ])
    }
}
