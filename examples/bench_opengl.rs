//! detour's GL ES on Xe3 against the official one (Mesa's libGLESv2, iris), the same work on both: each case under
//! examples/cases compiled and linked, then FRAMES frames of clear + its triangle into a SIZE² RGBA32F target, each
//! waited for. Per case: compile + link, the first frame, and the mean / fastest of the frames after it.
//!
//!   cc -O2 -o glbench examples/support/glbench.c -lEGL -lGLESv2
//!   cargo run --release --example bench_opengl -- GLBENCH [CASE...]      (FRAMES, default 100; SIZE, default 256)

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[path = "glcase/mod.rs"]
mod glcase;

use detour::{ClearMask, Device, DrawMode, Toolchain};
use glcase::{Case, VERTICES};

/// Frames a case draws when FRAMES is not set
const FRAMES: u32 = 100;
/// The target's side when SIZE is not set
const SIZE: u32 = 256;
/// Nanoseconds a millisecond
const NS_MS: f64 = 1e6;

/// One driver's times for a case
struct Times {
    compile: Duration,
    first: Duration,
    mean: Duration,
    min: Duration,
}

impl Times {
    /// The frames' times after the first, as mean and fastest
    fn of(compile: Duration, frames: &[Duration]) -> Times {
        let rest = &frames[1..];
        Times { compile, first: frames[0], mean: rest.iter().sum::<Duration>() / rest.len() as u32, min: rest.iter().copied().min().unwrap_or_default() }
    }

    /// The case on detour
    fn detour(case: &Case, tools: &Toolchain<'_>, device: &Device<'_>, size: u32, frames: u32) -> Result<Times, String> {
        let start = Instant::now();
        let mut gl = case.context(tools, size)?;
        let compile = start.elapsed();
        gl.viewport(0, 0, size as i32, size as i32);
        gl.clear_color([0.0; 4]);
        let mut times = Vec::new();
        for _ in 0..frames {
            let t = Instant::now();
            gl.clear(ClearMask { color: true, ..Default::default() });
            gl.draw_arrays(device, DrawMode::Triangles, 0, VERTICES).map_err(|e| e.0)?;
            gl.finish(device).map_err(|e| e.0)?;
            times.push(t.elapsed());
        }
        Ok(Times::of(compile, &times))
    }

    /// The case on Mesa, through glbench
    fn mesa(case: &Case, glbench: &Path, size: u32, frames: u32) -> Result<Times, String> {
        let f = |n: &str| case.dir.join(n);
        let o = Command::new(glbench).args([f("shader.vert"), f("shader.frag"), f("resources")]).args([size.to_string(), size.to_string(), frames.to_string()]).output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).trim().to_owned());
        }
        let ns: Vec<u64> = String::from_utf8_lossy(&o.stdout).split_whitespace().map(|v| v.parse().map_err(|e| format!("glbench: {e}"))).collect::<Result<_, _>>()?;
        let [compile, first, mean, min] = ns[..] else { return Err("glbench: four times expected".into()) };
        Ok(Times { compile: Duration::from_nanos(compile), first: Duration::from_nanos(first), mean: Duration::from_nanos(mean), min: Duration::from_nanos(min) })
    }

    fn ms(d: Duration) -> String {
        format!("{:.3}", d.as_nanos() as f64 / NS_MS)
    }

    /// detour over Mesa (how many times as long)
    fn ratio(ours: Duration, theirs: Duration) -> String {
        format!("{:.2}×", ours.as_secs_f64() / theirs.as_secs_f64().max(f64::MIN_POSITIVE))
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let glbench = PathBuf::from(args.next().expect("the glbench binary (examples/support/glbench.c)"));
    let num = |v: &str, d: u32| std::env::var(v).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let (frames, size) = (num("FRAMES", FRAMES).max(2), num("SIZE", SIZE));
    let only: Vec<String> = args.collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cases");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().filter_map(|e| Some(e.ok()?.path())).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let tools = Toolchain::new();
    let xe = <detour::Xe as detour::Gpu>::open_first().expect("an Xe3 GPU");
    let device = Device::new(&xe);
    println!("{frames} frames of {size}×{size} RGBA32F a case; times in ms (detour / mesa), ratio = detour ÷ mesa");
    println!("{:<18} {:>21} {:>8} {:>21} {:>8} {:>21} {:>8}", "case", "compile", "", "first frame", "", "frame (mean)", "");
    for dir in dirs.into_iter().filter(|d| only.is_empty() || only.iter().any(|o| d.ends_with(o))) {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let case = Case { dir };
        match (Times::detour(&case, &tools, &device, size, frames), Times::mesa(&case, &glbench, size, frames)) {
            (Ok(d), Ok(m)) => println!(
                "{name:<18} {:>10} / {:<8} {:>8} {:>10} / {:<8} {:>8} {:>10} / {:<8} {:>8}   (fastest {} / {})",
                Times::ms(d.compile), Times::ms(m.compile), Times::ratio(d.compile, m.compile),
                Times::ms(d.first), Times::ms(m.first), Times::ratio(d.first, m.first),
                Times::ms(d.mean), Times::ms(m.mean), Times::ratio(d.mean, m.mean),
                Times::ms(d.min), Times::ms(m.min),
            ),
            (d, m) => println!("{name:<18} detour: {}  mesa: {}", d.err().map_or("ok".into(), |e| e.lines().next().unwrap_or("").to_owned()), m.err().unwrap_or_else(|| "ok".into())),
        }
    }
}
