use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

fn pipeline(filter: &str) -> String {
    format!(
        r#"detour.pipeline @volume {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], filter = "{filter}",
  vertex_buffers = [{{stride = 20, attributes = [{{location = 0, format = "float32x2", offset = 0}}, {{location = 1, format = "float32x3", offset = 8}}]}}]}}
func.func @vs(%p: vector<2xf32>, %uvw: vector<3xf32>) -> (vector<4xf32>, vector<3xf32>) attributes {{detour.stage = "vertex"}} {{
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{{mask = array<i64: 0, 1, 2, 3>}}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uvw : vector<4xf32>, vector<3xf32>
}}
func.func @fs(%uvw: vector<3xf32>) -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  %c = "detour.sample"(%uvw) {{binding = 0 : i64}} : (vector<3xf32>) -> vector<4xf32>
  return %c : vector<4xf32>
}}
"#
    )
}

fn texel(x: u32, y: u32, z: u32) -> [u8; 4] {
    [(x * 60 + 10) as u8, (y * 60 + 20) as u8, (z * 60 + 30) as u8, 255]
}

fn strips(ws: &[f32]) -> Vec<f32> {
    let n = ws.len() as f32;
    let mut v = Vec::new();
    for (k, &w) in ws.iter().enumerate() {
        let (x0, x1) = (k as f32 / n * 2.0 - 1.0, (k as f32 + 1.0) / n * 2.0 - 1.0);
        for (x, y, u, vv) in [(x0, -1.0, 0.0, 0.0), (x1, -1.0, 1.0, 0.0), (x0, 1.0, 0.0, 1.0), (x0, 1.0, 0.0, 1.0), (x1, -1.0, 1.0, 0.0), (x1, 1.0, 1.0, 1.0)] {
            v.extend([x, y, u, vv, w]);
        }
    }
    v
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mut ok = true;
    let nearest = cx.pipeline(&codegen, &pipeline("nearest"), "volume")?;
    let slices: Vec<Vec<u8>> = (0..4).map(|z| (0..4).flat_map(|y| (0..4).flat_map(move |x| texel(x, y, z))).collect()).collect();
    let refs: Vec<&[u8]> = slices.iter().map(Vec::as_slice).collect();
    let volume = cx.texture_3d(4, 4, TextureFormat::Rgba8Unorm, &refs)?;
    let vb = cx.floats(&strips(&[0.125, 0.375, 0.625, 0.875]))?;
    let mut target = cx.target(16, 4, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&target);
    r.draw(&nearest, &[&vb], &[&volume], 24, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = target.bytes()?;
    for z in 0..4 {
        let mut bad = 0;
        for y in 0..4 {
            for x in 4 * z..4 * z + 4 {
                bad += usize::from(px[((y * 16 + x) * 4) as usize..][..4] != texel(x % 4, y, z));
            }
        }
        println!("slice {z} (w {}): {bad} of 16 texels wrong", (z as f32 + 0.5) / 4.0);
        ok &= bad == 0;
    }

    let linear = cx.pipeline(&codegen, &pipeline("linear"), "volume")?;
    let (black, white) = ([0u8, 0, 0, 255].repeat(4), [255u8, 255, 255, 255].repeat(4));
    let two = cx.texture_3d(2, 2, TextureFormat::Rgba8Unorm, &[&black, &white])?;
    let vb = cx.floats(&strips(&[0.5]))?;
    let mut target = cx.target(4, 4, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    r.target(&target);
    r.draw(&linear, &[&vb], &[&two], 6, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = target.bytes()?;
    let grey = (0..16).all(|i| (127..=128).contains(&px[i * 4]) && px[i * 4] == px[i * 4 + 1] && px[i * 4 + 3] == 255);
    println!("black and white slices, linear at w 0.5: {:?} {}", &px[..4], if grey { "grey" } else { "not grey" });
    ok &= grey;
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
