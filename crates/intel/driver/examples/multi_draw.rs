//! Multi-draw indirect (`multiDrawIndirect`; Vulkan `vkCmdDrawIndirectCount`: `Recorder::multi_draw_indirect`) on Xe3:
//! eight eighth-wide quads of one vertex buffer, draw k's arguments (32 bytes apart) picking quad k (first vertex, or
//! base vertex when indexed). Into 1920x1080 cleared blue, each case its own target, pixel by pixel: no count (all 8),
//! a count of 3 (the first three), 0 (none), 20 (all 8: at most `max_count`), and indexed with a count of 5.

use detour::base::{Gpu, IndexFormat, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;
const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

const SOLID: &str = r#"detour.pipeline @solid {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}
func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}
func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[1.0, 0.5019608, 0.2509804, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
"#;

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let solid = cx.pipeline(&codegen, SOLID, "solid")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let mut quads = Vec::new();
    for k in 0..8 {
        let (x0, x1) = (-1.0 + 0.25 * k as f32, -0.75 + 0.25 * k as f32);
        quads.extend([x0, -1.0, x1, -1.0, x0, 1.0, x0, 1.0, x1, -1.0, x1, 1.0]);
    }
    let vb = cx.floats(&quads)?;
    let index = cx.buffer(&words(&[0, 1, 2, 3, 4, 5]))?;
    // 32 bytes a draw: drawIndirect's (count, instances, first vertex, first instance), drawIndexedIndirect's (count,
    // instances, first index, base vertex, first instance)
    let plain = cx.buffer(&words(&(0..8).flat_map(|k| [6, 1, 6 * k, 0, 0, 0, 0, 0]).collect::<Vec<u32>>()))?;
    let indexed = cx.buffer(&words(&(0..8).flat_map(|k| [6, 1, 0, 6 * k, 0, 0, 0, 0]).collect::<Vec<u32>>()))?;
    let counts = cx.buffer(&words(&[3, 0, 20, 5]))?;
    let mut ok = true;
    for (name, count, use_index, drawn) in [
        ("no count", None, false, 8),
        ("count 3", Some(0u64), false, 3),
        ("count 0", Some(4), false, 0),
        ("count 20, max 8", Some(8), false, 8),
        ("indexed, count 5", Some(12), true, 5),
    ] {
        let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        let (args, ix) = if use_index { (&indexed, Some((&index, IndexFormat::Uint32))) } else { (&plain, None) };
        r.multi_draw_indirect(&solid, &[&vb], ix, &[], (args, 0, 32), 8, count.map(|at| (&counts, at)))?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let mut bad = 0;
        for y in 0..H {
            for x in 0..W {
                let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
                bad += usize::from(p != if x / (W / 8) < drawn { ORANGE } else { BLUE });
            }
        }
        println!("{name}: {drawn} of 8 quads wanted; {bad} of {} pixels wrong", W * H);
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
