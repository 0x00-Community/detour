//! Surfaces exported as dma-bufs: a scanout surface (Tile4, and Linear) drawn into, exported (`Device::export`),
//! imported by a second device as another process would and read back right; then given to the display engine as a
//! framebuffer (DRM_IOCTL_MODE_ADDFB2 with the modifier, on the GPU's card node) and removed again, which KMS only
//! takes from a buffer it can scan out.
//!
//!   cargo run --release --example export

use std::fs::OpenOptions;
use std::os::fd::{AsFd, AsRawFd};

use detour::{ColorFormat, ColorTarget, Compiled, Device, Draw, Exported, Gpu, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat, Xe};
use mlirpass::{GlslReader, GlslStage};

const SHADERS: [&str; 2] = [
    "#version 300 es\nlayout(location = 0) in vec2 pos;\nlayout(location = 1) in vec4 color;\nout vec4 c;\nvoid main() { c = color; gl_Position = vec4(pos, 0.0, 1.0); }\n",
    "#version 300 es\nprecision highp float;\nin vec4 c;\nout vec4 o;\nvoid main() { o = c; }\n",
];
const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";
const W: u32 = 1920;
const H: u32 = 1080;
const COLOR: [u8; 4] = [10, 200, 90, 255];

/// struct drm_mode_fb_cmd2, struct drm_prime_handle
#[repr(C)]
#[derive(Default)]
struct FbCmd2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}
#[repr(C)]
struct PrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}
const DRM_IOCTL_MODE_ADDFB2: u64 = 0xc068_64b8;
const DRM_IOCTL_MODE_RMFB: u64 = 0xc004_64af;
const DRM_IOCTL_PRIME_FD_TO_HANDLE: u64 = 0xc00c_642e;
const DRM_MODE_FB_MODIFIERS: u32 = 2;

unsafe extern "C" {
    fn ioctl(fd: i32, request: u64, ...) -> i32;
}

fn errno() -> std::io::Error {
    std::io::Error::last_os_error()
}

/// The card node of the GPU whose render node `render` is
fn card(render: &std::path::Path) -> Option<std::path::PathBuf> {
    let name = render.file_name()?.to_str()?;
    let dir = std::fs::read_dir(format!("/sys/class/drm/{name}/device/drm")).ok()?;
    dir.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).find(|n| n.starts_with("card")).map(|n| format!("/dev/dri/{n}").into())
}

/// `e` given to KMS as a `W`×`H` framebuffer on `card`, then removed
fn framebuffer(card: &std::path::Path, e: &Exported) -> Result<(), String> {
    let node = OpenOptions::new().read(true).write(true).open(card).map_err(|err| format!("{}: {err}", card.display()))?;
    let mut prime = PrimeHandle { handle: 0, flags: 0, fd: e.fd.as_raw_fd() };
    if unsafe { ioctl(node.as_raw_fd(), DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut prime) } != 0 {
        return Err(format!("PRIME import: {}", errno()));
    }
    let mut fb = FbCmd2 { width: W, height: H, pixel_format: e.fourcc, flags: DRM_MODE_FB_MODIFIERS, ..Default::default() };
    (fb.handles[0], fb.pitches[0], fb.offsets[0], fb.modifier[0]) = (prime.handle, e.pitch, e.offset as u32, e.modifier);
    if unsafe { ioctl(node.as_raw_fd(), DRM_IOCTL_MODE_ADDFB2, &mut fb) } != 0 {
        return Err(format!("ADDFB2 (modifier {:#x}, pitch {}): {}", e.modifier, e.pitch, errno()));
    }
    let mut id = fb.fb_id;
    if unsafe { ioctl(node.as_raw_fd(), DRM_IOCTL_MODE_RMFB, &mut id) } != 0 {
        return Err(format!("RMFB: {}", errno()));
    }
    Ok(())
}

fn target(tiling: Tiling) -> ColorTarget<'static> {
    ColorTarget { width: W, height: H, format: ColorFormat::Rgba8Unorm, tiling, load: TargetLoad::Keep(&[]) }
}

/// A quad over the whole target, `COLOR`
fn fill(device: &Device<'_>, program: &Compiled, s: detour::Surface, tiling: Tiling) -> Result<(), String> {
    let mut v = Vec::new();
    for (x, y) in [(-1.0f32, -1.0f32), (1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        v.extend([x, y].iter().flat_map(|f| f.to_le_bytes()));
        v.extend(COLOR);
    }
    let d = Draw {
        buffers: vec![VertexBuffer { bytes: &v, stride: 12 }],
        elements: vec![
            VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 },
            VertexElement { buffer: 0, format: VertexFormat::Unorm8x4, offset: 8 },
        ],
        count: 6,
        viewport: [0.0, 0.0, W as f32, H as f32],
        target: target(tiling),
        ..Draw::default()
    };
    device.draw(program, &d, s).map_err(|e| e.0)
}

fn run(xe: &Xe, program: &Compiled, tiling: Tiling, card: Option<&std::path::Path>) -> Result<String, String> {
    let device = Device::new(xe);
    let s = device.scanout(&target(tiling)).map_err(|e| e.0)?;
    fill(&device, program, s, tiling)?;
    let e = device.export(s).map_err(|e| e.0)?;
    device.finish(s).map_err(|e| e.0)?;
    let other = Device::new(xe);
    let imported = other.import(&target(tiling), e.fd.as_fd(), e.offset, e.pitch).map_err(|e| e.0)?;
    let pixels = other.read(imported, (W * H * 4) as usize).map_err(|e| e.0)?;
    let wrong = pixels.as_chunks::<4>().0.iter().filter(|p| **p != COLOR).count();
    if wrong > 0 {
        return Err(format!("{wrong} pixels read through the import are not the color drawn"));
    }
    let kms = match card {
        Some(c) => framebuffer(c, &e).map(|()| format!(", taken by KMS on {} as a framebuffer", c.display()))?,
        None => ", no card node to try KMS on".into(),
    };
    Ok(format!("pitch {}, modifier {:#x}, fourcc {:#x}: read right through another device{kms}", e.pitch, e.modifier, e.fourcc))
}

fn main() {
    let tools = Toolchain::new();
    let mut body = tools.body();
    GlslReader::read(&mut body, SHADERS[0], GlslStage::Vertex).unwrap();
    GlslReader::read(&mut body, SHADERS[1], GlslStage::Fragment).unwrap();
    let root = body.root;
    tools.pipeline(TO_SPIRV).and_then(|p| p.run(&mut body, root)).unwrap();
    let program = tools.compile(body).unwrap_or_else(|e| panic!("{}", e.0));
    let xe = Xe::open_first().unwrap();
    let card = card(xe.path());
    let mut failed = 0;
    for tiling in [Tiling::Tile4, Tiling::Linear] {
        let r = run(&xe, &program, tiling, card.as_deref());
        failed += usize::from(r.is_err());
        match r {
            Ok(m) => println!("ok   {tiling:?} scanout surface exported: {m}"),
            Err(e) => println!("FAIL {tiling:?} scanout surface exported: {e}"),
        }
    }
    std::process::exit(i32::from(failed > 0));
}
