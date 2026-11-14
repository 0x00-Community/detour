use detour::base::{Gpu, Native};
use detour::xe3::{Buffer, Codegen, Context, Error, Pipeline, Recorder};
use detour::xe::Xe;
use triangle::{BACKGROUND, Scene, Shown};

const CORNERS: [([f32; 3], [f32; 4]); 3] = [
    ([0.0, 0.8, 0.0], [1.0, 0.0, 0.0, 1.0]),
    ([-0.8, -0.6, 0.0], [0.0, 1.0, 0.0, 1.0]),
    ([0.8, -0.6, 0.0], [0.0, 0.0, 1.0, 1.0]),
];

const SPIN: f32 = 0.25;
const TILT: f32 = 0.35;

struct Camera {
    focal: f32,
    near: f32,
    far: f32,
    distance: f32,
}

impl Camera {
    const SEEN: Self = Self { focal: 1.732_050_8, near: 0.1, far: 10.0, distance: 2.5 };

    fn clip(&self, [x, y, z]: [f32; 3], angle: f32, aspect: f32) -> [f32; 4] {
        let (s, c) = angle.sin_cos();
        let (x, z) = (c * x + s * z, -s * x + c * z);
        let (st, ct) = TILT.sin_cos();
        let (y, z) = (ct * y - st * z, st * y + ct * z);
        let z = z + self.distance;
        [self.focal / aspect * x, -self.focal * y, self.far * (z - self.near) / (self.far - self.near), z]
    }
}

struct Triangle3d<'d> {
    pipeline: Pipeline,
    vertices: Buffer<'d, Xe>,
}

impl Triangle3d<'_> {
    fn vertices(angle: f32, aspect: f32) -> [f32; 24] {
        let mut out = [0.0; 24];
        for (to, (p, color)) in out.as_chunks_mut::<8>().0.iter_mut().zip(CORNERS) {
            to[..4].copy_from_slice(&Camera::SEEN.clip(p, angle, aspect));
            to[4..].copy_from_slice(&color);
        }
        out
    }
}

impl<'d> Scene<'d> for Triangle3d<'d> {
    const TITLE: &'static str = "detour triangle3d";

    fn new(cx: &mut Context<'d, Xe, Native>, codegen: &Codegen<'_>) -> Result<Self, Error> {
        let pipeline = cx.pipeline(codegen, include_str!("triangle3d.mlir"), "triangle3d")?;
        Ok(Self { pipeline, vertices: cx.floats(&Self::vertices(0.0, 1.0))? })
    }

    fn record(&mut self, r: &mut Recorder<'_, 'd, Xe, Native>, seconds: f32, (w, h): (u32, u32)) -> Result<(), Error> {
        let angle = seconds * SPIN * std::f32::consts::TAU;
        let floats = Self::vertices(angle, w as f32 / h as f32);
        for (to, f) in self.vertices.bytes()?.as_chunks_mut::<4>().0.iter_mut().zip(floats) {
            to.copy_from_slice(&f.to_le_bytes());
        }
        r.draw(&self.pipeline, &[&self.vertices], &[], 3, 1)
    }

    fn animated(&self) -> bool {
        true
    }

    fn check(&self, px: &[u8], (w, h): (u32, u32)) -> Result<(), String> {
        let at = |x: u32, y: u32| -> [u8; 4] { px[((y * w + x) * 4) as usize..][..4].try_into().unwrap_or_default() };
        for (x, y) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
            if at(x, y) != BACKGROUND {
                return Err(format!("corner ({x}, {y}) is {:?}, not the background", at(x, y)));
            }
        }
        let clip: Vec<[f32; 4]> = CORNERS.iter().map(|(p, _)| Camera::SEEN.clip(*p, 0.0, w as f32 / h as f32)).collect();
        let (sx, sy) = clip.iter().fold((0.0, 0.0), |(x, y), c| (x + c[0] / c[3] / 3.0, y + c[1] / c[3] / 3.0));
        let (x, y) = (((sx + 1.0) / 2.0 * w as f32) as u32, ((sy + 1.0) / 2.0 * h as f32) as u32);
        let p = at(x, y);
        if p == BACKGROUND || p[..3].iter().any(|&c| c < 40) || p[3] != 255 {
            return Err(format!("projected centroid ({x}, {y}) is {p:?}, not a mix of the three colors"));
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Shown::<Triangle3d>::run(&Xe::open_first()?)
}
