//! Viewports and scissor rectangles (`setViewport` / `setScissorRect`) on Xe3, pixel by pixel in a 64x64 target cleared
//! blue: a quad over all of clip space drawn through a viewport inside the target (and a quarter of clip space: the
//! viewport's quarter, its transform), one past its edges (clamped), a
//! scissor, a viewport and a scissor together (their intersection), an empty scissor (nothing), and back to the whole
//! target after each pass starts.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context, Scissor, Viewport};
use detour::xe::Xe;

const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const N: u32 = 64;

/// A case: its name, its viewport and scissor (none: the target's), the pixels it lights.
type Case = (&'static str, Option<Viewport>, Option<Scissor>, Vec<(u32, u32)>);

/// The pixels of [x0, x1) × [y0, y1).
fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<(u32, u32)> {
    (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let tri = cx.pipeline(&codegen, include_str!("frames/triangle.mlir"), "tri")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let quad = cx.floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])?;
    // the top left quarter of clip space (y down)
    let quarter = cx.floats(&[-1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0])?;
    let size = [N, N];
    let cases: Vec<Case> = vec![
        ("viewport (8, 4) 32 x 16", Some(cx.viewport([8.0, 4.0, 32.0, 16.0], [0.0, 1.0], size)?), None, rect(8, 4, 40, 20)),
        ("viewport (8, 4) 32 x 16, a quarter of clip space: its top left quarter", Some(cx.viewport([8.0, 4.0, 32.0, 16.0], [0.0, 1.0], size)?), None, rect(8, 4, 24, 12)),
        ("viewport (48, 40) 32 x 32, past the edges", Some(cx.viewport([48.0, 40.0, 32.0, 32.0], [0.0, 1.0], size)?), None, rect(48, 40, 64, 64)),
        ("scissor (10, 10) 5 x 7", None, Some(cx.scissor([10, 10, 5, 7])?), rect(10, 10, 15, 17)),
        ("viewport (0, 0) 32 x 32, scissor (20, 24) 30 x 30", Some(cx.viewport([0.0, 0.0, 32.0, 32.0], [0.0, 1.0], size)?), Some(cx.scissor([20, 24, 30, 30])?), rect(20, 24, 32, 32)),
        ("empty scissor", None, Some(cx.scissor([5, 5, 0, 9])?), Vec::new()),
        ("neither (a new pass: the whole target)", None, None, rect(0, 0, N, N)),
    ];
    let mut ok = true;
    for (name, viewport, scissor, want) in cases {
        let mut target = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        if let Some(v) = &viewport {
            r.viewport(v);
        }
        if let Some(s) = &scissor {
            r.scissor(s);
        }
        r.draw(&tri, &[if name.contains("quarter") { &quarter } else { &quad }], &[], 6, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let mut lit = Vec::new();
        for y in 0..N {
            for x in 0..N {
                let p: [u8; 4] = px[((y * N + x) * 4) as usize..][..4].try_into()?;
                match p {
                    ORANGE => lit.push((x, y)),
                    BLUE => {}
                    other => return Err(format!("{name}: pixel ({x}, {y}) is {other:?}").into()),
                }
            }
        }
        let good = lit == want;
        println!("{name}: {} pixels lit, {}", lit.len(), if good { "as expected" } else { "wrong" });
        ok &= good;
    }
    // the same viewport twice is one state
    let (a, b) = (cx.viewport([1.0, 2.0, 3.0, 4.0], [0.0, 1.0], size)?, cx.viewport([1.0, 2.0, 3.0, 4.0], [0.0, 1.0], size)?);
    println!("the same viewport made twice: {:?} {:?}", a, b);
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
