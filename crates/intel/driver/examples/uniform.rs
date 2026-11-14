use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;
const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

#[rustfmt::skip]
const QUAD: [f32; 12] = [-1.0, -1.0, 0.0, -1.0, -1.0, 1.0,   -1.0, 1.0, 0.0, -1.0, 0.0, 1.0];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let moved = cx.pipeline(&codegen, include_str!("uniform/moved.mlir"), "moved")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let vb = cx.floats(&QUAD)?;
    let translation = cx.floats(&[0.5, 0.0])?;
    let color = cx.floats(&[1.0, 0.5, 0.25, 1.0])?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.bind(&[&translation, &color]);
    r.draw(&moved, &[&vb], &[], 6, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let mut bad = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            let want = if (W / 4..W * 3 / 4).contains(&x) { ORANGE } else { BLUE };
            if p != want {
                bad.push((x, y, p));
            }
        }
    }
    println!("{} wrong of {}", bad.len(), W * H);
    assert!(bad.is_empty(), "wrong pixels {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
