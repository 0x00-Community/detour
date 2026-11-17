//! Each case under examples/cases (shader.vert, shader.frag, resources) drawn by detour's GL on Xe3 and by the local
//! Mesa (mlirpass's glrun), compared byte for byte. `resources` is glrun's: one line each,
//!   attribute NAME LOCATION N V...   three vertices of N floats
//!   uniform NAME LOCATION f|i V...   glUniform{1,2,3,4}{f,i}v (16 floats: glUniformMatrix4fv)
//!
//!   cargo run --example compare -- GLRUN [CASE...]      (DETOUR_FULL=1: the whole error, with the IR)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "glcase/mod.rs"]
mod glcase;

use detour::{ClearMask, Device, DrawMode, Toolchain};
use glcase::VERTICES;

const SIZE: u32 = 64;
const PIXEL: usize = 16;

/// A case's pixels from both, or why one could not draw
struct Case {
    dir: PathBuf,
}

impl Case {
    fn detour(&self, tools: &Toolchain<'_>, device: &Device<'_>) -> Result<Vec<u8>, String> {
        let mut gl = glcase::Case { dir: self.dir.clone() }.context(tools, SIZE)?;
        gl.viewport(0, 0, SIZE as i32, SIZE as i32);
        gl.clear_color([0.0; 4]);
        gl.clear(ClearMask { color: true, ..Default::default() });
        gl.draw_arrays(device, DrawMode::Triangles, 0, VERTICES).map_err(|e| e.0)?;
        gl.framebuffer.pixels(device).map_err(|e| e.0)
    }

    fn mesa(&self, glrun: &Path, out: &Path) -> Result<Vec<u8>, String> {
        let o = Command::new(glrun).arg("glsl").arg(self.dir.join("shader.vert")).arg(self.dir.join("shader.frag")).args([SIZE.to_string(), SIZE.to_string()]).arg(out).arg(self.dir.join("resources")).output().map_err(|e| e.to_string())?;
        if !o.status.success() {
            return Err(String::from_utf8_lossy(&o.stderr).trim().to_owned());
        }
        fs::read(out).map_err(|e| e.to_string())
    }

    /// "ok", or what differs
    fn run(&self, tools: &Toolchain<'_>, device: &Device<'_>, glrun: &Path, out: &Path) -> Result<String, String> {
        let ours = self.detour(tools, device).map_err(|e| if std::env::var_os("DETOUR_FULL").is_some() { format!("detour: {e}") } else { format!("detour: {}", e.lines().next().unwrap_or("")) })?;
        let theirs = self.mesa(glrun, out).map_err(|e| format!("mesa: {e}"))?;
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
    let glrun = PathBuf::from(args.next().expect("the glrun binary"));
    let only: Vec<String> = args.collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("target/compare");
    fs::create_dir_all(&out).unwrap();
    let mut dirs: Vec<PathBuf> = fs::read_dir(root.join("examples/cases")).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    let tools = Toolchain::new();
    let xe = <detour::Xe as detour::Gpu>::open_first().unwrap();
    let device = Device::new(&xe);
    let mut failed = 0;
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        if !only.is_empty() && !only.contains(&name) {
            continue;
        }
        let r = Case { dir }.run(&tools, &device, &glrun, &out.join(format!("{name}.raw")));
        failed += usize::from(r.is_err());
        println!("{name}: {}", r.unwrap_or_else(|e| format!("FAIL {e}")));
    }
    std::process::exit(i32::from(failed > 0));
}
