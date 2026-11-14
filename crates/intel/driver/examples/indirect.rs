//! Indirect draws (`drawIndirect`, `drawIndexedIndirect`): the arguments read from a buffer when the GPU draws, into a
//! 1920x1080 target cleared blue, checked pixel by pixel. The `indexed` example's two quads, now indexed indirect (the
//! second from index 3 with base vertex 4, in the arguments), a third quad drawn sequentially from vertex 4 (indirect,
//! first vertex in the arguments), a draw of 0 instances (nothing), and a quad whose vertex count a compute shader
//! writes into the arguments in the same frame, just before the draw reads them. Then indirect dispatches
//! (`dispatchWorkgroupsIndirect`): a 3 x 2 x 2 grid, and one whose counts a dispatch writes in the same frame.

use detour::base::{Gpu, IndexFormat, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const W: u32 = 1920;
const H: u32 = 1080;

/// Quads (x, y in clip space, y down), 4 vertices each: the left half, the bottom of the right quarter; then two
/// triangle-list quads of 6 vertices: the last eighth's top quarter, the fifth eighth's top quarter.
#[rustfmt::skip]
const VERTICES: [f32; 40] = [
    -1.0, -1.0,   0.0, -1.0,   -1.0, 1.0,   0.0, 1.0,
    0.5, 0.0,     1.0, 0.0,    0.5, 1.0,    1.0, 1.0,
    0.75, -1.0,   1.0, -1.0,   0.75, -0.5,  0.75, -0.5,  1.0, -1.0,  1.0, -0.5,
    0.0, -1.0,    0.25, -1.0,  0.0, -0.5,   0.0, -0.5,   0.25, -1.0, 0.25, -0.5,
];
const QUAD: [u32; 6] = [0, 1, 2, 2, 1, 3];

/// Each workgroup (x, y, z) of an `nx × ny × nz` grid stores its index + 1 at dword (z ny + y) nx + x of `out` (one
/// invocation does).
const MARK: &str = r#"
func.func @mark(%out: i32, %nx: i32, %ny: i32) attributes {detour.stage = "compute"} {
  %x = "detour.workgroup_id"() {dim = 0 : i32} : () -> i32
  %y = "detour.workgroup_id"() {dim = 1 : i32} : () -> i32
  %z = "detour.workgroup_id"() {dim = 2 : i32} : () -> i32
  %l = "detour.lane"() : () -> i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %zy = arith.muli %z, %ny : i32
  %r = arith.addi %zy, %y : i32
  %rx = arith.muli %r, %nx : i32
  %i = arith.addi %rx, %x : i32
  %v = arith.addi %i, %c1 : i32
  %b = arith.shli %i, %c2 : i32
  %at = arith.addi %out, %b : i32
  %first = arith.cmpi eq, %l, %c0 : i32
  scf.if %first {
    "detour.store"(%at, %v) : (i32, i32) -> ()
  }
  return
}
"#;

/// Stores `value` at byte address `at` (one invocation does).
const WRITE: &str = r#"
func.func @write(%at: i32, %value: i32) attributes {detour.stage = "compute"} {
  %l = "detour.lane"() : () -> i32
  %c0 = arith.constant 0 : i32
  %first = arith.cmpi eq, %l, %c0 : i32
  scf.if %first {
    "detour.store"(%at, %value) : (i32, i32) -> ()
  }
  return
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let tri = cx.pipeline(&codegen, include_str!("frames/triangle.mlir"), "tri")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let write = cx.kernel(&codegen, WRITE, "write")?;
    let corners = cx.clear_vertices()?;
    let vb = cx.floats(&VERTICES)?;
    let short: Vec<u8> = QUAD.iter().flat_map(|i| (*i as u16).to_le_bytes()).collect();
    let ib16 = cx.buffer(&short)?;
    let long: Vec<u8> = [9, 9, 9].iter().chain(&QUAD).flat_map(|i: &u32| i.to_le_bytes()).collect();
    let ib32 = cx.buffer(&long)?;
    // the arguments, u32s: indexed [count, instances, first index, base vertex, first instance] × 2 (from 0, 20), then
    // sequential [count, instances, first vertex, first instance] × 3 (from 40, 56, 72): the third quad, 0 instances
    // of vertices 0 ..= 5 (a triangle over blue if drawn),
    // and the fourth quad with a count of 0 the compute shader makes 6
    let words: [u32; 22] = [6, 1, 0, 0, 0, 6, 1, 3, 4, 0, 6, 1, 8, 0, 6, 0, 0, 0, 0, 1, 14, 0];
    let args = cx.buffer(&words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>())?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.dispatch(&write, [1, 1], &[(args.addr + 72) as u32, 6])?;
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.draw_indexed_indirect(&tri, &[&vb], (&ib16, IndexFormat::Uint16), &[], (&args, 0))?;
    r.draw_indexed_indirect(&tri, &[&vb], (&ib32, IndexFormat::Uint32), &[], (&args, 20))?;
    r.draw_indirect(&tri, &[&vb], &[], (&args, 40))?;
    r.draw_indirect(&tri, &[&vb], &[], (&args, 56))?;
    r.draw_indirect(&tri, &[&vb], &[], (&args, 72))?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let (mut orange, mut bad) = (0usize, Vec::new());
    for y in 0..H {
        for x in 0..W {
            let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            // the left half; the right quarter's bottom (indexed, base vertex 4); the last eighth's top quarter
            // (sequential from 8);
            // the fifth eighth's top quarter (the count the compute shader wrote)
            let right = x >= W * 3 / 4;
            let third = (W / 2..W * 5 / 8).contains(&x);
            let last = x >= W * 7 / 8;
            let want = if x < W / 2 || (right && y >= H / 2) || (last && y < H / 4) || (third && y < H / 4) { ORANGE } else { BLUE };
            orange += usize::from(p == ORANGE);
            if p != want {
                bad.push((x, y, p));
            }
        }
    }
    println!("{orange} orange pixels of {}, {} wrong", W * H, bad.len());
    assert!(bad.is_empty(), "wrong pixels {:?}", &bad[..bad.len().min(8)]);
    dispatches(&mut cx, &codegen, &write)?;
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}

/// Indirect dispatches: a 3 x 2 x 2 grid from arguments in a buffer, then a 4 x 1 x 3 one whose counts a dispatch
/// writes in the same frame; each workgroup's mark where it should be, nothing past the grid.
fn dispatches<'d>(cx: &mut Context<'d, Xe, Native>, codegen: &Codegen<'_>, write: &detour::xe3::Kernel) -> Result<(), Box<dyn std::error::Error>> {
    let mark = cx.kernel(codegen, MARK, "mark")?;
    let args = cx.buffer(&[3u32, 2, 2, 0, 0, 0].iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>())?;
    let mut a = cx.buffer(&vec![0; 64 * 4])?;
    let mut b = cx.buffer(&vec![0; 64 * 4])?;
    let mut r = cx.record();
    r.dispatch_indirect(&mark, (&args, 0), &[a.addr as u32, 3, 2])?;
    for (k, v) in [4u32, 1, 3].into_iter().enumerate() {
        r.dispatch(write, [1, 1], &[(args.addr + 12 + 4 * k as u64) as u32, v])?;
    }
    r.dispatch_indirect(&mark, (&args, 12), &[b.addr as u32, 4, 1])?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    for (name, buf, count) in [("3 x 2 x 2", &mut a, 12u32), ("4 x 1 x 3 (written in the frame)", &mut b, 12)] {
        let got: Vec<u32> = buf.bytes()?.as_chunks::<4>().0[..64].iter().map(|w| u32::from_le_bytes(*w)).collect();
        let want: Vec<u32> = (0..64).map(|i| if i < count { i + 1 } else { 0 }).collect();
        println!("indirect dispatch {name}: {}", if got == want { "every workgroup marked, none past" } else { "wrong" });
        assert_eq!(got, want);
    }
    Ok(())
}
