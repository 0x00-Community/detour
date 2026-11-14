use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const PIPELINE: &str = r#"detour.pipeline @cubed {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], filter = "nearest",
  vertex_buffers = [{stride = 20, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x3", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %d: vector<3xf32>) -> (vector<4xf32>, vector<3xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %d : vector<4xf32>, vector<3xf32>
}
func.func @fs(%d: vector<3xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = "detour.sample"(%d) {binding = 0 : i64, cube = true} : (vector<3xf32>) -> vector<4xf32>
  return %c : vector<4xf32>
}
"#;

const W: u32 = 24;
const H: u32 = 4;

fn texel(face: u32, x: u32, y: u32) -> [u8; 4] {
    [(face * 40 + 10) as u8, (x * 60 + 5) as u8, (y * 60 + 7) as u8, 255]
}

fn direction(f: u32, u: f32, v: f32) -> [f32; 3] {
    let (s, t) = (2.0 * u - 1.0, 2.0 * v - 1.0);
    match f {
        0 => [1.0, -t, -s],
        1 => [-1.0, -t, s],
        2 => [s, 1.0, t],
        3 => [s, -1.0, -t],
        4 => [s, -t, 1.0],
        _ => [-s, -t, -1.0],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let p = cx.pipeline(&codegen, PIPELINE, "cubed")?;
    let faces: Vec<Vec<u8>> = (0..6).map(|f| (0..4).flat_map(|y| (0..4).flat_map(move |x| texel(f, x, y))).collect()).collect();
    let tex = cx.texture_cube(4, TextureFormat::Rgba8Unorm, [&faces[0], &faces[1], &faces[2], &faces[3], &faces[4], &faces[5]])?;
    let mut v = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let (f, i) = (x / 4, x % 4);
            let d = direction(f, (i as f32 + 0.5) / 4.0, (y as f32 + 0.5) / 4.0).map(|c| 3.0 * c);
            let clip = |px: u32, py: u32| [px as f32 / W as f32 * 2.0 - 1.0, py as f32 / H as f32 * 2.0 - 1.0];
            let (a, b) = (clip(x, y), clip(x + 1, y + 1));
            for (cx_, cy_) in [(a[0], a[1]), (b[0], a[1]), (a[0], b[1]), (a[0], b[1]), (b[0], a[1]), (b[0], b[1])] {
                v.extend([cx_, cy_, d[0], d[1], d[2]]);
            }
        }
    }
    let vb = cx.floats(&v)?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&target);
    r.draw(&p, &[&vb], &[&tex], 6 * W * H, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = target.bytes()?;
    let mut ok = true;
    for f in 0..6 {
        let mut bad = 0;
        for y in 0..H {
            for x in 4 * f..4 * f + 4 {
                bad += usize::from(px[((y * W + x) * 4) as usize..][..4] != texel(f, x % 4, y));
            }
        }
        println!("face {}: {bad} of 16 texels wrong", ["+x", "-x", "+y", "-y", "+z", "-z"][f as usize]);
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
