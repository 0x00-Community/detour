use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const PIPELINE: &str = r#"detour.pipeline @arrayed {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], filter = "nearest",
  vertex_buffers = [{stride = 24, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x4", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %d: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %d : vector<4xf32>, vector<4xf32>
}
func.func @fs(%d: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = "detour.sample"(%d) {binding = 0 : i64, cube = true} : (vector<4xf32>) -> vector<4xf32>
  return %c : vector<4xf32>
}
"#;

const W: u32 = 24;
const H: u32 = 20;
const LAYERS: [f32; 5] = [0.0, 1.0, 2.0, 1.4, 9.0];
const PICKED: [u32; 5] = [0, 1, 2, 1, 2];

fn texel(cube: u32, face: u32, x: u32, y: u32) -> [u8; 4] {
    [(face * 40 + 10) as u8, (x * 60 + 5) as u8, (y * 60 + 7) as u8, (255 - cube * 70) as u8]
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
    let p = cx.pipeline(&codegen, PIPELINE, "arrayed")?;
    let faces: Vec<Vec<u8>> = (0..18).map(|l| (0..4).flat_map(|y| (0..4).flat_map(move |x| texel(l / 6, l % 6, x, y))).collect()).collect();
    let levels: Vec<[&[u8]; 1]> = faces.iter().map(|f| [f.as_slice()]).collect();
    let layers: Vec<&[&[u8]]> = levels.iter().map(|l| l.as_slice()).collect();
    let tex = cx.texture_layers(4, 4, TextureFormat::Rgba8Unorm, true, &layers)?;
    let mut v = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let (f, i, j) = (x / 4, x % 4, y % 4);
            let d = direction(f, (i as f32 + 0.5) / 4.0, (j as f32 + 0.5) / 4.0).map(|c| 3.0 * c);
            let layer = LAYERS[(y / 4) as usize];
            let clip = |px: u32, py: u32| [px as f32 / W as f32 * 2.0 - 1.0, py as f32 / H as f32 * 2.0 - 1.0];
            let (a, b) = (clip(x, y), clip(x + 1, y + 1));
            for (cx_, cy_) in [(a[0], a[1]), (b[0], a[1]), (a[0], b[1]), (a[0], b[1]), (b[0], a[1]), (b[0], b[1])] {
                v.extend([cx_, cy_, d[0], d[1], d[2], layer]);
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
    for block in 0..5u32 {
        let mut bad = 0;
        for y in 4 * block..4 * block + 4 {
            for x in 0..W {
                bad += usize::from(px[((y * W + x) * 4) as usize..][..4] != texel(PICKED[block as usize], x / 4, x % 4, y % 4));
            }
        }
        println!("cube {} (picks {}): {bad} of 96 texels wrong", LAYERS[block as usize], PICKED[block as usize]);
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
