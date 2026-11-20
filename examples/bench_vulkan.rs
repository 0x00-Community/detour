//! detour's Vulkan on Xe3 against the official Vulkan driver (Mesa's ANV), the same work on both: each case under
//! examples/cases made SPIR-V for Vulkan once (detour's GLSL front end; loose uniforms in a block at set 0, binding
//! 0), then on each side the same calls (examples/vkcase on detour, examples/support/vkrun.c on ANV): shader modules +
//! pipeline, then FRAMES submissions of one command buffer (clear by the render pass's load op + the triangle into
//! a SIZE² RGBA32F image), each waited for on a fence.
//!
//!   cc -O2 -o vkrun examples/support/vkrun.c -lvulkan
//!   cargo run --release --example bench_vulkan -- VKRUN [CASE...]      (FRAMES, default 100; SIZE, default 256)

use std::path::{Path, PathBuf};
use std::time::Duration;

#[path = "vkcase/mod.rs"]
mod vkcase;

use detour::{Card, Device, Toolchain, VkDevice};
use vkcase::{VkCase, VkScene};

/// Frames a case draws when FRAMES is not set
const FRAMES: u32 = 100;
/// The target's side when SIZE is not set
const SIZE: u32 = 256;
/// Nanoseconds a millisecond
const NS_MS: f64 = 1e6;

struct Ratio;

impl Ratio {
    fn ms(d: Duration) -> String {
        format!("{:.3}", d.as_nanos() as f64 / NS_MS)
    }

    /// detour over Mesa (how many times as long)
    fn of(ours: Duration, theirs: Duration) -> String {
        format!("{:.2}×", ours.as_secs_f64() / theirs.as_secs_f64().max(f64::MIN_POSITIVE))
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let vkrun = PathBuf::from(args.next().expect("the vkrun binary (examples/support/vkrun.c)"));
    let num = |v: &str, d: u32| std::env::var(v).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let (frames, size) = (num("FRAMES", FRAMES).max(2), num("SIZE", SIZE));
    let only: Vec<String> = args.collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let scratch = root.join("target/bench_vulkan");
    std::fs::create_dir_all(&scratch).unwrap();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root.join("examples/cases")).unwrap().filter_map(|e| Some(e.ok()?.path())).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let card = Card::open_first().expect("a GPU detour drives");
    let tools = Toolchain::new(card.isa());
    let device = Device::new(&card);
    println!("{frames} frames of {size}×{size} RGBA32F a case; times in ms (detour / mesa), ratio = detour ÷ mesa");
    println!("{:<18} {:>21} {:>8} {:>21} {:>8} {:>21} {:>8}", "case", "compile", "", "first frame", "", "frame (mean)", "");
    for dir in dirs.into_iter().filter(|d| only.is_empty() || only.iter().any(|o| d.ends_with(o))) {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let case = match VkCase::of(&vkcase::Case { dir }, &tools) {
            Ok(c) => c,
            Err(e) => {
                println!("{name:<18} {e}");
                continue;
            }
        };
        let mut vk = VkDevice::new(&tools, &device);
        let ours = VkScene::new(&mut vk, &case, size).and_then(|s| s.run(&mut vk, frames));
        match (ours, case.mesa(&vkrun, &scratch, size, frames, None)) {
            (Ok(d), Ok(m)) => println!(
                "{name:<18} {:>10} / {:<8} {:>8} {:>10} / {:<8} {:>8} {:>10} / {:<8} {:>8}   (fastest {} / {})",
                Ratio::ms(d.compile), Ratio::ms(m.compile), Ratio::of(d.compile, m.compile),
                Ratio::ms(d.first), Ratio::ms(m.first), Ratio::of(d.first, m.first),
                Ratio::ms(d.mean), Ratio::ms(m.mean), Ratio::of(d.mean, m.mean),
                Ratio::ms(d.min), Ratio::ms(m.min),
            ),
            (d, m) => println!("{name:<18} detour: {}  mesa: {}", d.err().map_or("ok".into(), |e: String| e.lines().next().unwrap_or("").to_owned()), m.err().unwrap_or_else(|| "ok".into())),
        }
    }
}
