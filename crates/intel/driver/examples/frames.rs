use std::time::{Duration, Instant};

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Buffer, Codegen, Context, Pipeline, Recorded, Target};
use detour::xe::Xe;

const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const W: u32 = 1920;
const H: u32 = 1080;

struct Scene<'s, 'd> {
    target: &'s Target<'d, Xe>,
    clear: &'s Pipeline,
    corners: &'s Buffer<'d, Xe>,
    tri: &'s Pipeline,
    vb: &'s Buffer<'d, Xe>,
}

impl<'d> Scene<'_, 'd> {
    fn record(&self, cx: &Context<'d, Xe, Native>, times: &mut Vec<Duration>) -> Recorded {
        let t = Instant::now();
        let mut r = cx.record();
        r.target(self.target);
        r.draw(self.clear, &[self.corners], &[], 3, 1).expect("draw");
        r.draw(self.tri, &[self.vb], &[], 3, 1).expect("draw");
        let dw = r.finish();
        times.push(t.elapsed());
        dw
    }
}

fn percentile(v: &mut [Duration], p: f64) -> Duration {
    v.sort();
    v[((v.len() - 1) as f64 * p) as usize]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = std::env::args().nth(1).map_or(Ok(1000), |a| a.parse())?;
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let t = Instant::now();
    let tri = cx.pipeline(&codegen, include_str!("frames/triangle.mlir"), "tri")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    println!("2 pipelines compiled in {:?} (once)", t.elapsed());
    let corners = cx.clear_vertices()?;
    let vb = cx.floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0])?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut record = Vec::new();
    let scene = Scene { target: &target, clear: &clear, corners: &corners, tri: &tri, vb: &vb };

    let t = Instant::now();
    let mut last = 0;
    for _ in 0..n {
        let dw = scene.record(&cx, &mut record);
        last = cx.submit(dw)?;
    }
    cx.wait(last)?;
    let total = t.elapsed();
    println!("{n} frames back to back in {total:?}: {:.0} frames/s, {} dwords a frame", n as f64 / total.as_secs_f64(), scene.record(&cx, &mut Vec::new()).dw.len());
    println!("recording: p50 {:?} p99 {:?}", percentile(&mut record, 0.5), percentile(&mut record, 0.99));

    let mut latency = Vec::new();
    for _ in 0..n.min(200) {
        let dw = scene.record(&cx, &mut Vec::new());
        let t = Instant::now();
        let s = cx.submit(dw)?;
        cx.wait(s)?;
        latency.push(t.elapsed());
        last = s;
    }
    println!("submit to done: p50 {:?} p99 {:?}", percentile(&mut latency, 0.5), percentile(&mut latency, 0.99));

    let dmabuf = target.export()?;
    let sync_file = cx.export_fence(last)?;
    println!("exported: dma-buf fd {dmabuf:?}, sync_file fd {sync_file:?}");

    let px = target.bytes()?;
    let (mut orange, mut bad) = (0usize, Vec::new());
    for y in 0..H {
        for x in 0..W {
            let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into().unwrap();
            orange += usize::from(p == ORANGE);
            let s = (f64::from(x) + 0.5) / f64::from(W) + (f64::from(y) + 0.5) / f64::from(H);
            let edge = 1.0 / f64::from(W.min(H));
            let want = if s < 1.0 - edge { Some(ORANGE) } else if s > 1.0 + edge { Some(BLUE) } else { None };
            if want.is_some_and(|w| w != p) || (want.is_none() && p != ORANGE && p != BLUE) {
                bad.push((x, y, p));
            }
        }
    }
    println!("{orange} orange pixels of {}, {} wrong", W * H, bad.len());
    assert!(bad.is_empty(), "wrong pixels {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
