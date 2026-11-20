//! Each case under examples/cases drawn through detour's Vulkan on Xe3 and through the local Vulkan driver (Mesa's
//! ANV, examples/support/vkrun.c), the same SPIR-V and the same calls on both, compared byte for byte.
//!
//!   cc -O2 -o vkrun examples/support/vkrun.c -lvulkan
//!   cargo run --example compare_vulkan -- VKRUN [CASE...]      (DETOUR_FULL=1: the whole error)

use std::fs;
use std::path::{Path, PathBuf};

#[path = "vkcase/mod.rs"]
mod vkcase;

use detour::{Card, Device, Toolchain, VkDevice};
use vkcase::{VkCase, VkScene};

const SIZE: u32 = 64;
const PIXEL: usize = 16;
/// One frame is drawn on each side
const FRAMES: u32 = 1;

struct Case {
    dir: PathBuf,
}

impl Case {
    fn detour(&self, vk: &mut VkDevice<'_, '_>, case: &VkCase) -> Result<Vec<u8>, String> {
        let scene = VkScene::new(vk, case, SIZE)?;
        scene.run(vk, FRAMES)?;
        scene.pixels(vk)
    }

    fn mesa(&self, case: &VkCase, vkrun: &Path, scratch: &Path, out: &Path) -> Result<Vec<u8>, String> {
        case.mesa(vkrun, scratch, SIZE, FRAMES, Some(out))?;
        fs::read(out).map_err(|e| e.to_string())
    }

    /// "ok", or what differs
    fn run(&self, tools: &Toolchain<'_>, device: &Device<'_>, vkrun: &Path, scratch: &Path, out: &Path) -> Result<String, String> {
        let short = |e: String| if std::env::var_os("DETOUR_FULL").is_some() { e } else { e.lines().next().unwrap_or("").to_owned() };
        let case = VkCase::of(&vkcase::Case { dir: self.dir.clone() }, tools).map_err(|e| format!("spirv: {}", short(e)))?;
        let mut vk = VkDevice::new(tools, device);
        let ours = self.detour(&mut vk, &case).map_err(|e| format!("detour: {}", short(e)))?;
        let theirs = self.mesa(&case, vkrun, scratch, out).map_err(|e| format!("mesa: {e}"))?;
        let px = |b: &[u8], i: usize| b[i * PIXEL..(i + 1) * PIXEL].chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>();
        let covered = theirs.chunks(PIXEL).filter(|p| p.iter().any(|b| *b != 0)).count();
        let differing: Vec<usize> = (0..ours.len() / PIXEL).filter(|&i| ours[i * PIXEL..(i + 1) * PIXEL] != theirs[i * PIXEL..(i + 1) * PIXEL]).collect();
        match differing.first() {
            None => Ok(format!("ok ({covered} pixels)")),
            Some(&i) => Err(format!("{} of {covered} pixels differ; ({}, {}) detour {:?} mesa {:?}", differing.len(), i as u32 % SIZE, i as u32 / SIZE, px(&ours, i), px(&theirs, i))),
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let vkrun = PathBuf::from(args.next().expect("the vkrun binary (examples/support/vkrun.c)"));
    let only: Vec<String> = args.collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("target/compare_vulkan");
    fs::create_dir_all(&out).unwrap();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("examples/cases")).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let card = Card::open_first().unwrap();
    let tools = Toolchain::new(card.isa());
    let device = Device::new(&card);
    let mut failed = 0;
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        let r = Case { dir }.run(&tools, &device, &vkrun, &out, &out.join(format!("{name}.raw")));
        failed += usize::from(r.is_err());
        println!("{name}: {}", r.unwrap_or_else(|e| format!("FAIL {e}")));
    }
    std::process::exit(i32::from(failed > 0));
}
