use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe3_codegen::{CL_INVOCATION_COUNT, CL_PRIMITIVES_COUNT, IA_PRIMITIVES_COUNT, IA_VERTICES_COUNT, PS_INVOCATION_COUNT, VS_INVOCATION_COUNT};
use detour::xe::Xe;

const W: u32 = 1920;
const H: u32 = 1080;

#[rustfmt::skip]
const VERTICES: [f32; 36] = [
    -1.0, -1.0,  0.0, -1.0,  -1.0, 1.0,  -1.0, 1.0,  0.0, -1.0,  0.0, 1.0,
    0.5, 0.0,    1.0, 0.0,   0.5, 1.0,   0.5, 1.0,   1.0, 0.0,   1.0, 1.0,
    1.5, 0.0,    2.0, 0.0,   1.5, 1.0,   1.5, 1.0,   2.0, 0.0,   2.0, 1.0,
];

const STATS: [(u32, &str); 6] = [
    (IA_VERTICES_COUNT, "vertices"),
    (IA_PRIMITIVES_COUNT, "primitives"),
    (VS_INVOCATION_COUNT, "vertex shader invocations"),
    (CL_INVOCATION_COUNT, "clipper invocations"),
    (CL_PRIMITIVES_COUNT, "clipper primitives"),
    (PS_INVOCATION_COUNT, "fragment shader invocations"),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let tri = cx.pipeline(&codegen, include_str!("frames/triangle.mlir"), "tri")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let quads: Vec<_> = (0..3).map(|k| cx.floats(&VERTICES[k * 12..][..12])).collect::<Result<_, _>>()?;
    let mut results = cx.buffer(&vec![0; 4096])?;
    let target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.timestamp((&results, 256))?;
    r.statistics(&STATS.map(|(reg, _)| reg), (&results, 64))?;
    r.depth_count((&results, 0))?;
    for (k, q) in quads.iter().enumerate() {
        r.draw(&tri, &[q], &[], 6, 1)?;
        r.depth_count((&results, 8 * (k as u64 + 1)))?;
    }
    r.statistics(&STATS.map(|(reg, _)| reg), (&results, 128))?;
    r.timestamp((&results, 264))?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let words = results.bytes()?;
    let u64_at = |o: usize| u64::from_le_bytes(words[o..][..8].try_into().expect("8 bytes"));
    let passed: Vec<u64> = (0..3).map(|k| u64_at(8 * (k + 1)) - u64_at(8 * k)).collect();
    let want = [u64::from(W / 2 * H), u64::from(W / 4 * H / 2), 0];
    println!("occlusion: samples passed {passed:?}, the quads' pixels {want:?}");
    assert_eq!(passed, want);
    let stats: Vec<u64> = (0..STATS.len()).map(|k| u64_at(128 + 8 * k) - u64_at(64 + 8 * k)).collect();
    for ((_, name), v) in STATS.iter().zip(&stats) {
        println!("  {name}: {v}");
    }
    assert_eq!((stats[0], stats[1], stats[3]), (18, 6, 6));
    assert!(stats[2] >= 1 && stats[2] <= 18 && stats[5] >= want[0] + want[1], "statistics {stats:?}");
    let (t0, t1) = (u64_at(256), u64_at(264));
    let hz = xe.driver_info().gts.iter().find(|g| !g.media).map_or(0, |g| g.reference_clock);
    println!("timestamps {t0} .. {t1}: {} ticks, {:.1} µs at {hz} Hz", t1 - t0, (t1 - t0) as f64 * 1e6 / f64::from(hz.max(1)));
    assert!(t1 > t0 && hz > 0);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
