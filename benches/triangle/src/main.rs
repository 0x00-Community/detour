//! A triangle in a window, its color interpolated between the corners (red, green, blue). `cargo run -p triangle`
//! on Xe3.

use detour::base::{Gpu, Native};
use detour::xe3::{Buffer, Codegen, Context, Error, Pipeline, Recorder};
use detour::xe::Xe;
use triangle::{BACKGROUND, Scene, Shown};

/// x, y in clip space (y down, as Xe3 / Vulkan rasterize it), then RGBA.
#[rustfmt::skip]
const VERTICES: [f32; 18] = [
    0.0, -0.8,   1.0, 0.0, 0.0, 1.0,
    -0.8, 0.8,   0.0, 1.0, 0.0, 1.0,
    0.8, 0.8,    0.0, 0.0, 1.0, 1.0,
];

struct Triangle<'d> {
    pipeline: Pipeline,
    vertices: Buffer<'d, Xe>,
}

impl<'d> Scene<'d> for Triangle<'d> {
    const TITLE: &'static str = "detour triangle";

    fn new(cx: &mut Context<'d, Xe, Native>, codegen: &Codegen<'_>) -> Result<Self, Error> {
        Ok(Self { pipeline: cx.pipeline(codegen, include_str!("triangle.mlir"), "triangle")?, vertices: cx.floats(&VERTICES)? })
    }

    fn record(&mut self, r: &mut Recorder<'_, 'd, Xe, Native>, _: f32, _: (u32, u32)) -> Result<(), Error> {
        r.draw(&self.pipeline, &[&self.vertices], &[], 3, 1)
    }

    /// The corners are background; the centroid is the mean of the three colors (±1 for interpolation and rounding).
    fn check(&self, px: &[u8], (w, h): (u32, u32)) -> Result<(), String> {
        let at = |x: u32, y: u32| -> [u8; 4] { px[((y * w + x) * 4) as usize..][..4].try_into().unwrap_or_default() };
        for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
            if at(x, y) != BACKGROUND {
                return Err(format!("corner ({x}, {y}) is {:?}, not the background", at(x, y)));
            }
        }
        // the centroid in clip space is (0, 0.8 / 3)
        let (cx, cy) = (w / 2, ((1.0 + 0.8 / 3.0) / 2.0 * f64::from(h)) as u32);
        let mean = 255.0 / 3.0;
        let p = at(cx, cy);
        let near = p[..3].iter().all(|&c| (f64::from(c) - mean).abs() <= 2.0) && p[3] == 255;
        if near { Ok(()) } else { Err(format!("centroid ({cx}, {cy}) is {p:?}, not about {mean:.0} gray")) }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Shown::<Triangle>::run(&Xe::open_first()?)
}
