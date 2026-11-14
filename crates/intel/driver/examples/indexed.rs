//! Indexed draws (`drawIndexed`): two quads of 4 vertices each from 6 indices, into a 1920x1080 target cleared blue,
//! checked pixel by pixel. The first uses u16 indices; the second u32 indices from index 3 on (the first three are
//! padding) with base vertex 4, so it picks the second quad's vertices — the format, the first index and the base
//! vertex all have to be right for the picture to be.

use detour::base::{Gpu, IndexFormat, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const W: u32 = 1920;
const H: u32 = 1080;

/// Two quads (x, y in clip space, y down): the left half, and the bottom of the right quarter.
#[rustfmt::skip]
const VERTICES: [f32; 16] = [
    -1.0, -1.0,   0.0, -1.0,   -1.0, 1.0,   0.0, 1.0,
    0.5, 0.0,     1.0, 0.0,    0.5, 1.0,    1.0, 1.0,
];
/// A quad's two triangles over its four corners.
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
            // pixel centers never sit on the quads' edges (W / 2, 3 W / 4, H / 2 are whole pixels)
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
