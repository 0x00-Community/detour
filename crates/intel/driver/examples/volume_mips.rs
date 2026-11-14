//! 3D textures with mip levels (`Context::texture_3d_levels`) on Xe3: a 16 x 16 x 8 volume of 5 levels (16 x 16 x 8,
//! 8 x 8 x 4, 4 x 4 x 2, 2 x 2 x 1, 1 x 1 x 1), every level's every slice one color of its own. Into a 16 x 16 target,
//! one draw per (level, slice): u and v run 0 ..= 2^level over it (repeat), so the derivatives pick that level, and w
//! is the slice's center at that level; the nearest level, nearest texel. Every pixel the color of that slice of that
//! level.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const SIZE: u32 = 16;
const DEPTH: usize = 8;
const LEVELS: usize = 5;

const PIPELINE: &str = r#"detour.pipeline @mips {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  sampler = {mag_filter = "nearest", min_filter = "nearest", mipmap_filter = "nearest", address_mode_u = "repeat", address_mode_v = "repeat"},
  vertex_buffers = [{stride = 20, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x3", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %uvw: vector<3xf32>) -> (vector<4xf32>, vector<3xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uvw : vector<4xf32>, vector<3xf32>
}
func.func @fs(%uvw: vector<3xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = "detour.sample"(%uvw) {binding = 0 : i64} : (vector<3xf32>) -> vector<4xf32>
  return %c : vector<4xf32>
}
"#;

fn color(level: usize, slice: usize) -> [u8; 4] {
    [(level * 50 + 10) as u8, (slice * 30 + 5) as u8, (level * 13 + slice * 7) as u8, 255]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let p = cx.pipeline(&codegen, PIPELINE, "mips")?;
    let data: Vec<Vec<Vec<u8>>> = (0..LEVELS)
        .map(|l| {
            let n = ((SIZE >> l).max(1) * (SIZE >> l).max(1)) as usize;
            (0..(DEPTH >> l).max(1)).map(|s| color(l, s).repeat(n)).collect()
        })
        .collect();
    let slices: Vec<Vec<&[u8]>> = data.iter().map(|l| l.iter().map(|s| s.as_slice()).collect()).collect();
    let levels: Vec<&[&[u8]]> = slices.iter().map(|l| l.as_slice()).collect();
    let tex = cx.texture_3d_levels(SIZE, SIZE, TextureFormat::Rgba8Unorm, &levels)?;
    let mut ok = true;
    for l in 0..LEVELS {
        let depth = (DEPTH >> l).max(1);
        for s in 0..depth {
            let (span, w) = ((1u32 << l) as f32, (s as f32 + 0.5) / depth as f32);
            let quad: Vec<f32> = [(-1.0, -1.0, 0.0, 0.0), (1.0, -1.0, span, 0.0), (-1.0, 1.0, 0.0, span), (-1.0, 1.0, 0.0, span), (1.0, -1.0, span, 0.0), (1.0, 1.0, span, span)]
                .iter()
                .flat_map(|&(x, y, u, v)| [x, y, u, v, w])
                .collect();
            let vb = cx.floats(&quad)?;
            let mut target = cx.target(SIZE, SIZE, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
            let mut r = cx.record();
            r.target(&target);
            r.draw(&p, &[&vb], &[&tex], 6, 1)?;
            let n = cx.submit(r.finish())?;
            cx.wait(n)?;
            let px = target.bytes()?;
            let bad = px[..(SIZE * SIZE * 4) as usize].chunks(4).filter(|c| *c != color(l, s)).count();
            if bad > 0 {
                println!("level {l} slice {s}: {bad} of {} pixels wrong, e.g. {:?} (want {:?})", SIZE * SIZE, &px[..4], color(l, s));
            }
            ok &= bad == 0;
        }
        println!("level {l}: {depth} slices sampled");
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
