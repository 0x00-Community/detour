use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 512;
const H: u32 = 256;
const T: u32 = 256;
const BLUE: [u8; 4] = [0, 0, 255, 255];

fn color(l: u32) -> [u8; 4] {
    [(30 * l + 20) as u8, (255 - 25 * l) as u8, (60 + 20 * l) as u8, 255]
}

fn square(x: u32, y: u32, n: u32) -> Vec<f32> {
    let clip = |px: u32, py: u32| [px as f32 / W as f32 * 2.0 - 1.0, py as f32 / H as f32 * 2.0 - 1.0];
    let (a, b) = (clip(x, y), clip(x + n, y + n));
    vec![a[0], a[1], 0.0, 0.0, b[0], a[1], 1.0, 0.0, a[0], b[1], 0.0, 1.0, b[0], b[1], 1.0, 1.0]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mipped = cx.pipeline(&codegen, include_str!("mipmap/mipped.mlir"), "mipped")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let levels: Vec<Vec<u8>> = (0..9).map(|l| color(l).repeat(((T >> l) * (T >> l)) as usize)).collect();
    let refs: Vec<&[u8]> = levels.iter().map(Vec::as_slice).collect();
    let image = cx.texture_levels(T, T, TextureFormat::Rgba8Unorm, &refs)?;
    let squares = [(0, 0, 256, 0), (256, 0, 128, 1), (384, 0, 64, 2), (448, 0, 16, 4)];
    let quads = squares.iter().map(|&(x, y, n, _)| cx.floats(&square(x, y, n))).collect::<Result<Vec<_>, _>>()?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    for q in &quads {
        r.draw(&mipped, &[q], &[&image], 4, 1)?;
    }
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let mut bad = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            let want = squares.iter().find(|&&(sx, sy, n, _)| (sx..sx + n).contains(&x) && (sy..sy + n).contains(&y)).map_or(BLUE, |&(.., l)| color(l));
            if p != want {
                bad.push((x, y, p, want));
            }
        }
    }
    println!("{} wrong of {}", bad.len(), W * H);
    assert!(bad.is_empty(), "wrong pixels (x, y, got, want) {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
