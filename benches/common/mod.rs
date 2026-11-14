//! What `against_vulkan` and `against_opengl` share: the three workloads, the harness that times them the same way
//! through every API (`Api`), the pixel / word checks, and detour's side. Each API draws the same thing into the same
//! 1920x1080 RGBA8 target and hands back the pixels with memory row 0 at clip y = -1 (detour's and Vulkan's top row,
//! GL's bottom row: the same rows in memory), so one check serves all.
//!
//! - `tri`: a clear and one triangle covering half the target (the runtime's `frames` example): per-frame CPU cost,
//!   frames back to back (up to four in flight, detour's ring), submit-to-done latency.
//! - `fill`: a clear and `LAYERS` full-target quads blended as premultiplied alpha (one instanced draw): fill-bound.
//! - `dispatch`: a 16-lane-workgroup kernel writing its index, 1 and `DISPATCHES` dispatches in one submission, each
//!   one's writes visible to the next (detour flushes after every dispatch; Vulkan / GL get the matching barrier).

use std::time::{Duration, Instant};

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe::Xe;
use detour::xe3::{Buffer, Codegen, Context, Error, Kernel, Pipeline, Target};

pub const W: u32 = 1920;
pub const H: u32 = 1080;
/// `tri`'s clear color and triangle color
pub const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
pub const ORANGE: [f32; 4] = [1.0, 0.5, 0.25, 1.0];
/// `tri`: the triangle (clip space), half the target
pub const TRIANGLE: [f32; 6] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0];
/// `fill`: the clear color, then `LAYERS` full-target quads of the premultiplied `LAYER` color
pub const FILL_CLEAR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];
pub const LAYERS: u32 = 32;
pub const LAYER: [f32; 4] = [0.25, 0.125, 0.0625, 0.25];
/// `fill`: the unit quad's corners (triangle strip) and one instance: (x, y, w, h) in clip space, premultiplied color
pub const QUAD: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
const FULL: [f32; 8] = [-1.0, -1.0, 2.0, 2.0, LAYER[0], LAYER[1], LAYER[2], LAYER[3]];
/// `fill`'s instance buffer: `FULL` once per layer, as little-endian bytes
pub fn instances() -> Vec<u8> {
    FULL.iter().cycle().take(FULL.len() * LAYERS as usize).flat_map(|f| f.to_le_bytes()).collect()
}
/// `dispatch`: grid sizes (invocations, 16 a workgroup) and dispatches in the long submission
pub const LANES: [u32; 3] = [16, 4096, 1 << 20];
pub const DISPATCHES: u32 = 200;
/// frames in flight at most (detour's batch ring)
pub const RING: usize = 4;
/// frames run before anything is timed (the GPU clocks up)
const WARMUP: usize = 100;
/// the most `fill` hardware may be off the step-by-step unorm8 model, per channel: implementations round each blend
/// step a little differently and the 0.75 contraction keeps the drift within a few steps
const FILL_SLACK: f64 = 2.0;

/// The same shaders for Vulkan (through naga's GLSL frontend to SPIR-V) and GL (compiled by the driver).
pub const TRI_VS: &str = "#version 450
layout(location = 0) in vec2 p;
void main() { gl_Position = vec4(p, 0.0, 1.0); }
";
pub const TRI_FS: &str = "#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0, 0.5, 0.25, 1.0); }
";
pub const FILL_VS: &str = "#version 450
layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 rect;
layout(location = 2) in vec4 color;
layout(location = 0) out vec4 c;
void main() { gl_Position = vec4(rect.xy + corner * rect.zw, 0.0, 1.0); c = color; }
";
pub const FILL_FS: &str = "#version 450
layout(location = 0) in vec4 c;
layout(location = 0) out vec4 o;
void main() { o = c; }
";
pub const COMPUTE: &str = "#version 450
layout(local_size_x = 16) in;
layout(std430, binding = 0) buffer Out { uint v[]; };
void main() { uint i = gl_GlobalInvocationID.x; v[i] = i; }
";

/// detour's kernel: the same as `COMPUTE`.
const KERNEL: &str = r#"
func.func @k(%out: i32) attributes {detour.stage = "compute"} {
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b = arith.muli %g, %c16 : i32
  %i = arith.addi %b, %l : i32
  %o = arith.muli %i, %c4 : i32
  %a = arith.addi %out, %o : i32
  "detour.store"(%a, %i) : (i32, i32) -> ()
  return
}
"#;

#[derive(Clone, Copy, PartialEq)]
pub enum Scene {
    Tri,
    Fill,
}

/// One API driving the GPU. `frame` records and submits a frame (waiting first only when `RING` frames are in
/// flight) and returns what `wait` waits for, and the time spent recording when the API records separately from
/// submitting (GL does not).
pub trait Api {
    type Done: Copy;
    fn name(&self) -> String;
    fn frame(&mut self, scene: Scene) -> (Self::Done, Option<Duration>);
    fn wait(&mut self, done: Self::Done);
    /// The target after the last frame (waited for), `W` × `H` RGBA8, row 0 at clip y = -1.
    fn pixels(&mut self) -> Vec<u8>;
    /// `count` dispatches of `lanes` invocations in one submission, waited for; the time spent recording them, as
    /// `frame`.
    fn dispatch(&mut self, lanes: u32, count: u32) -> Option<Duration>;
    /// The first `lanes` words the kernel wrote.
    fn words(&mut self, lanes: u32) -> Vec<u32>;
}

/// What one API measured; `Report::table` sets two side by side.
pub struct Report {
    name: String,
    /// recording p50, frame (record + submit) p50, frames/s back to back, submit to done p50 / p99
    tri: (Option<Duration>, Duration, f64, Duration, Duration),
    /// a `fill` frame, back to back
    fill: Duration,
    /// per grid: one dispatch, `DISPATCHES` in one submission, per dispatch, per dispatch recording
    dispatch: Vec<(Duration, Duration, Duration, Option<Duration>)>,
}

trait Percentile {
    fn at(&mut self, p: f64) -> Duration;
}

impl Percentile for Vec<Duration> {
    fn at(&mut self, p: f64) -> Duration {
        self.sort();
        self[((self.len() - 1) as f64 * p) as usize]
    }
}

impl Report {
    /// Runs the three workloads on `api` (`n` frames each timing loop) and checks what it drew and wrote.
    pub fn run<A: Api>(api: &mut A, n: usize) -> Report {
        let name = api.name();
        eprintln!("{name}: tri");
        let tri = Self::frames(api, Scene::Tri, n);
        Check::tri(&api.pixels(), &name);
        eprintln!("{name}: fill");
        let fill = Self::frames(api, Scene::Fill, n / 4).2;
        Check::fill(&api.pixels(), &name);
        let fill = Duration::from_secs_f64(1.0 / fill);
        eprintln!("{name}: dispatch");
        let dispatch = LANES
            .iter()
            .map(|&lanes| {
                let mut time = |count| {
                    let t = Instant::now();
                    let record = api.dispatch(lanes, count);
                    (t.elapsed(), record)
                };
                // the second of each: the first pays for whatever is set up lazily
                let (_, (one, _), _, (many, record)) = (time(1), time(1), time(DISPATCHES), time(DISPATCHES));
                Check::words(&api.words(lanes), lanes, &name);
                (one, many, many.saturating_sub(one) / (DISPATCHES - 1), record.map(|r| r / DISPATCHES))
            })
            .collect();
        Report { name, tri, fill, dispatch }
    }

    /// Warm up, then frames back to back (throughput) and one at a time (CPU cost and latency).
    fn frames<A: Api>(api: &mut A, scene: Scene, n: usize) -> (Option<Duration>, Duration, f64, Duration, Duration) {
        let mut last = None;
        for _ in 0..WARMUP {
            last = Some(api.frame(scene).0);
        }
        last.map(|d| api.wait(d));
        let t = Instant::now();
        for _ in 0..n {
            last = Some(api.frame(scene).0);
        }
        last.map(|d| api.wait(d));
        let fps = n as f64 / t.elapsed().as_secs_f64();
        let (mut record, mut cpu, mut latency) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n.min(500) {
            let t = Instant::now();
            let (d, r) = api.frame(scene);
            cpu.push(t.elapsed());
            api.wait(d);
            latency.push(t.elapsed());
            record.extend(r);
        }
        let record = (!record.is_empty()).then(|| record.at(0.5));
        (record, cpu.at(0.5), fps, latency.at(0.5), latency.at(0.99))
    }

    /// The reports side by side, one column each, as a Markdown table.
    pub fn table(reports: &[Report], n: usize) {
        let us = |d: Duration| format!("{:.1} µs", d.as_secs_f64() * 1e6);
        let ms = |d: Duration| format!("{:.3} ms", d.as_secs_f64() * 1e3);
        let mut rows: Vec<(String, Box<dyn Fn(&Report) -> String>)> = vec![
            ("tri: record p50".into(), Box::new(move |r| r.tri.0.map_or("—".into(), us))),
            ("tri: record + submit p50".into(), Box::new(move |r| us(r.tri.1))),
            (format!("tri: {n} frames back to back"), Box::new(|r| format!("{:.0} fps", r.tri.2))),
            ("tri: submit to done p50".into(), Box::new(move |r| ms(r.tri.3))),
            ("tri: submit to done p99".into(), Box::new(move |r| ms(r.tri.4))),
            (format!("fill: {LAYERS} blended full quads, a frame"), Box::new(move |r| ms(r.fill))),
        ];
        for (k, lanes) in LANES.iter().enumerate() {
            rows.push((format!("dispatch {lanes} lanes: one, submit to done"), Box::new(move |r| ms(r.dispatch[k].0))));
            rows.push((format!("dispatch {lanes} lanes: {DISPATCHES} in one submission"), Box::new(move |r| ms(r.dispatch[k].1))));
            rows.push((format!("dispatch {lanes} lanes: each"), Box::new(move |r| us(r.dispatch[k].2))));
            rows.push((format!("dispatch {lanes} lanes: each, recording it"), Box::new(move |r| r.dispatch[k].3.map_or("—".into(), us))));
        }
        let names: Vec<&str> = reports.iter().map(|r| r.name.as_str()).collect();
        println!("\n| {W}x{H} RGBA8 | {} |\n| --- |{}", names.join(" | "), " --- |".repeat(reports.len()));
        for (k, cell) in rows {
            println!("| {k} | {} |", reports.iter().map(|r| cell(r)).collect::<Vec<_>>().join(" | "));
        }
    }
}

/// The checks: every pixel / word against what the workload means.
struct Check;

impl Check {
    fn unorm(c: [f32; 4]) -> [u8; 4] {
        c.map(|v| (v * 255.0).round() as u8)
    }

    fn tri(px: &[u8], name: &str) {
        let (orange, blue) = (Self::unorm(ORANGE), Self::unorm(BLUE));
        let mut bad = Vec::new();
        for y in 0..H {
            for x in 0..W {
                let p: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into().unwrap();
                // inside is x / W + y / H < 1 at pixel centers; within a pixel of the edge the fill rule decides
                let s = (f64::from(x) + 0.5) / f64::from(W) + (f64::from(y) + 0.5) / f64::from(H);
                let edge = 1.0 / f64::from(W.min(H));
                let want = if s < 1.0 - edge { Some(orange) } else if s > 1.0 + edge { Some(blue) } else { None };
                if want.is_some_and(|w| w != p) || (want.is_none() && p != orange && p != blue) {
                    bad.push((x, y, p));
                }
            }
        }
        assert!(bad.is_empty(), "{name} tri: {} wrong pixels {:?}", bad.len(), &bad[..bad.len().min(8)]);
        eprintln!("{name} tri: {} pixels, 0 wrong", W * H);
    }

    fn fill(px: &[u8], name: &str) {
        // each step blends onto what the target holds, unorm8
        let mut want = FILL_CLEAR.map(f64::from);
        for _ in 0..LAYERS {
            want = std::array::from_fn(|i| {
                let v = f64::from(LAYER[i]) + want[i] * (1.0 - f64::from(LAYER[3]));
                (v * 255.0).round() / 255.0
            });
        }
        let first: [u8; 4] = px[..4].try_into().unwrap();
        let off = px.as_chunks::<4>().0.iter().filter(|p| **p != first).count();
        assert_eq!(off, 0, "{name} fill: {off} pixels differ from the first {first:?}");
        let worst = (0..4).map(|i| (f64::from(first[i]) - want[i] * 255.0).abs()).fold(0.0, f64::max);
        assert!(worst <= FILL_SLACK, "{name} fill: {first:?}, model {:?}", want.map(|v| v * 255.0));
        eprintln!("{name} fill: all {} pixels {first:?} (model {:?})", W * H, want.map(|v| (v * 255.0).round()));
    }

    fn words(got: &[u32], lanes: u32, name: &str) {
        let bad = (0..lanes).filter(|&i| got[i as usize] != i).count();
        assert_eq!(bad, 0, "{name} dispatch {lanes}: {bad} wrong words");
    }
}

/// detour on Xe3: its runtime, the pipelines and kernel compiled once.
pub struct Detour<'d> {
    cx: Context<'d, Xe, Native>,
    clear: [Pipeline; 2],
    tri: Pipeline,
    fill: Pipeline,
    kernel: Kernel,
    corners: Buffer<'d, Xe>,
    triangle: Buffer<'d, Xe>,
    quad: Buffer<'d, Xe>,
    full: Buffer<'d, Xe>,
    out: Buffer<'d, Xe>,
    target: Target<'d, Xe>,
    last: usize,
}

impl<'d> Detour<'d> {
    /// The first render node driven by xe.
    pub fn device() -> Result<Xe, Error> {
        Ok(Xe::open_first()?)
    }

    pub fn new(xe: &'d Xe) -> Result<Self, Error> {
        let t = Instant::now();
        let codegen = Codegen::default();
        let mut cx = Context::<Xe, Native>::new(xe)?;
        let clear = [cx.clear_pipeline(&codegen, BLUE)?, cx.clear_pipeline(&codegen, FILL_CLEAR)?];
        let tri = cx.pipeline(&codegen, include_str!("../../crates/intel/driver/examples/frames/triangle.mlir"), "tri")?;
        let fill = cx.pipeline(&codegen, include_str!("fill.mlir"), "fill")?;
        let kernel = cx.kernel(&codegen, KERNEL, "k")?;
        eprintln!("detour: 4 pipelines + 1 kernel (MLIR + Xe3 codegen) in {:?}", t.elapsed());
        Ok(Detour {
            corners: cx.clear_vertices()?,
            triangle: cx.floats(&TRIANGLE)?,
            quad: cx.floats(&QUAD)?,
            full: cx.buffer(&instances())?,
            out: cx.buffer(&vec![0; *LANES.iter().max().unwrap() as usize * 4])?,
            target: cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Streaming)?,
            cx,
            clear,
            tri,
            fill,
            kernel,
            last: 0,
        })
    }
}

impl Api for Detour<'_> {
    type Done = usize;

    fn name(&self) -> String {
        "detour".into()
    }

    fn frame(&mut self, scene: Scene) -> (usize, Option<Duration>) {
        let t = Instant::now();
        let mut r = self.cx.record();
        r.target(&self.target);
        match scene {
            Scene::Tri => {
                r.draw(&self.clear[0], &[&self.corners], &[], 3, 1).expect("draw");
                r.draw(&self.tri, &[&self.triangle], &[], 3, 1).expect("draw");
            }
            Scene::Fill => {
                r.draw(&self.clear[1], &[&self.corners], &[], 3, 1).expect("draw");
                r.draw(&self.fill, &[&self.quad, &self.full], &[], 4, LAYERS).expect("draw");
            }
        }
        let dw = r.finish();
        let record = t.elapsed();
        self.last = self.cx.submit(dw).expect("submit");
        (self.last, Some(record))
    }

    fn wait(&mut self, done: usize) {
        self.cx.wait(done).expect("wait");
    }

    fn pixels(&mut self) -> Vec<u8> {
        self.cx.wait(self.last).expect("wait");
        self.target.bytes().expect("bytes").to_vec()
    }

    fn dispatch(&mut self, lanes: u32, count: u32) -> Option<Duration> {
        let t = Instant::now();
        let mut r = self.cx.record();
        for _ in 0..count {
            r.dispatch(&self.kernel, [lanes / 16, 1], &[self.out.addr as u32]).expect("dispatch");
        }
        let dw = r.finish();
        let record = t.elapsed();
        let s = self.cx.submit(dw).expect("submit");
        self.cx.wait(s).expect("wait");
        Some(record)
    }

    fn words(&mut self, lanes: u32) -> Vec<u32> {
        let b = self.out.bytes().expect("bytes");
        b.as_chunks::<4>().0[..lanes as usize].iter().map(|w| u32::from_le_bytes(*w)).collect()
    }
}

/// `cargo bench ... -- [N]`: frames per timing loop (default 2000); cargo's own `--bench` flag is skipped.
pub fn frames_arg() -> usize {
    std::env::args().skip(1).find(|a| !a.starts_with('-')).map_or(2000, |a| a.parse().expect("N"))
}
