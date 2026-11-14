//! MSAA at 2x, 4x and 8x: a triangle into a multisampled target cleared blue, resolved (the mean of each pixel's
//! samples) into a plain one, checked pixel by pixel. A pixel is the triangle's color inside, the clear color outside,
//! and on an edge the mean of the two weighted by how many of its samples (Mesa's standard positions, Vulkan's) the
//! triangle covers, worked out here exactly: the vertices sit on the rasterizer's 1/256-pixel grid, the samples on a
//! 1/16 one, so the edge functions are integers. A pixel with a sample exactly on an edge is not checked (the
//! tie-breaking rule decides it). The colors are multiples of 8, so every mean is a whole unorm8 value.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const TRI: [u8; 4] = [248, 128, 64, 255];
const CLEAR: [u8; 4] = [0, 0, 248, 255];
const W: u32 = 1920;
const H: u32 = 1080;
/// The triangle's corners in pixels (x right, y down), on the 1/256 grid.
const CORNERS: [(f64, f64); 3] = [(100.25, 60.5), (1800.75, 300.125), (500.5, 1000.25)];
/// Mesa's intel_sample_positions_2x / _4x / _8x, in sixteenths of a pixel.
const SAMPLES_2X: [(i64, i64); 2] = [(12, 12), (4, 4)];
const SAMPLES_4X: [(i64, i64); 4] = [(6, 2), (14, 6), (2, 10), (10, 14)];
const SAMPLES_8X: [(i64, i64); 8] = [(9, 5), (7, 11), (13, 9), (5, 3), (3, 13), (1, 7), (11, 15), (15, 1)];

/// How many of pixel (x, y)'s samples (at `positions`) are inside the triangle; `None` when one sits on an edge.
fn covered(x: u32, y: u32, positions: &[(i64, i64)]) -> Option<u32> {
    // everything in 1/256 pixel
    let v = CORNERS.map(|(px, py)| ((px * 256.0) as i64, (py * 256.0) as i64));
    let mut n = 0;
    for &(sx, sy) in positions {
        let p = (i64::from(x) * 256 + sx * 16, i64::from(y) * 256 + sy * 16);
        let e: Vec<i64> = (0..3).map(|i| {
            let (a, b) = (v[i], v[(i + 1) % 3]);
            (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
        }).collect();
        if e.contains(&0) {
            return None;
        }
        n += u32::from(e.iter().all(|e| *e > 0) || e.iter().all(|e| *e < 0));
    }
    Some(n)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let corners = cx.clear_vertices()?;
    let clip: Vec<f32> = CORNERS.iter().flat_map(|(x, y)| [(x / f64::from(W) * 2.0 - 1.0) as f32, (y / f64::from(H) * 2.0 - 1.0) as f32]).collect();
    let vb = cx.floats(&clip)?;
    let rv = cx.resolve_vertices(W, H)?;
    let mut ok = true;
    for (n, positions) in [(2u32, &SAMPLES_2X[..]), (4, &SAMPLES_4X[..]), (8, &SAMPLES_8X[..])] {
        // the triangle's color TRI (248 / 255 red, a multiple of 8)
        let src = include_str!("msaa/triangle.mlir").replace("samples = 4 : i32", &format!("samples = {n} : i32")).replace("0.98823529", &format!("{:?}", 248.0f32 / 255.0));
        let tri = cx.pipeline(&codegen, &src, "tri")?;
        let clear = cx.msaa_clear_pipeline_samples(&codegen, CLEAR.map(|c| f32::from(c) / 255.0), n)?;
        let resolve = cx.resolve_pipeline_samples(&codegen, n)?;
        let msaa = cx.msaa_target_samples(W, H, TextureFormat::Rgba8Unorm, n)?;
        let mut out = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

        let mut r = cx.record();
        r.target(&msaa);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        r.draw(&tri, &[&vb], &[], 3, 1)?;
        r.target(&out);
        r.resolve(&resolve, &rv, &msaa)?;
        let done = cx.submit(r.finish())?;
        cx.wait(done)?;

        let px = out.bytes()?;
        let (mut counts, mut skipped, mut bad) = (vec![0usize; n as usize + 1], 0usize, Vec::new());
        for y in 0..H {
            for x in 0..W {
                let Some(k) = covered(x, y, positions) else {
                    skipped += 1;
                    continue;
                };
                counts[k as usize] += 1;
                let want: [u8; 4] = std::array::from_fn(|c| ((u32::from(TRI[c]) * k + u32::from(CLEAR[c]) * (n - k)) / n) as u8);
                let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
                if got != want {
                    bad.push((x, y, k, got, want));
                }
            }
        }
        println!("{n}x: pixels by covered samples 0..={n}: {counts:?}, {skipped} with a sample on an edge, {} wrong", bad.len());
        if !bad.is_empty() {
            println!("   first (x, y, samples, got, want): {:?}", &bad[..bad.len().min(6)]);
        }
        ok &= bad.is_empty();
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
