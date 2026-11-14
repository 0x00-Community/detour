use detour::base::{Gpu, IndexFormat, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const W: u32 = 1920;
const H: u32 = 1080;

#[rustfmt::skip]
const VERTICES: [f32; 16] = [
    -1.0, -1.0,   0.0, -1.0,   -1.0, 1.0,   0.0, 1.0,
    0.5, 0.0,     1.0, 0.0,    0.5, 1.0,    1.0, 1.0,
];

const QUAD: [u32; 6] = [0, 1, 2, 2, 1, 3];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let tri = cx.pipeline(&codegen, include_str!("frames/triangle.mlir"), "tri")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let vb = cx.floats(&VERTICES)?;
    let short: Vec<u8> = QUAD.iter().flat_map(|i| (*i as u16).to_le_bytes()).collect();
    let ib16 = cx.buffer(&short)?;
    let long: Vec<u8> = [9, 9, 9].iter().chain(&QUAD).flat_map(|i: &u32| i.to_le_bytes()).collect();
    let ib32 = cx.buffer(&long)?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.draw_indexed(&tri, &[&vb], (&ib16, IndexFormat::Uint16), &[], 6, 1, 0, 0)?;
    r.draw_indexed(&tri, &[&vb], (&ib32, IndexFormat::Uint32), &[], 6, 1, 3, 4)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let (mut orange, mut bad) = (0usize, Vec::new());
    for y in 0..H {
        for x in 0..W {
            let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            let want = if x < W / 2 || (x >= W * 3 / 4 && y >= H / 2) { ORANGE } else { BLUE };
            orange += usize::from(p == ORANGE);
            if p != want {
                bad.push((x, y, p));
            }
        }
    }
    println!("{orange} orange pixels of {}, {} wrong", W * H, bad.len());
    assert!(bad.is_empty(), "wrong pixels {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
