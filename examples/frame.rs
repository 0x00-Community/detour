//! A frame of draws of two programs taking turns, whose vertices, uniforms, counts, first vertex and scissor all
//! differ, drawn with `Device::draw`: into a surface of the device's (read back), into a dma-buf of the GPU's and into
//! a udmabuf imported (done when their fences say, read by the CPU through the exporter's memory). Each cell lands
//! where it should, the imported buffers' row padding stays as it was, two shapes are lowered, and a frame takes the
//! time printed.
//!
//!   cargo run --release --example frame [FRAMES]

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::time::{Duration, Instant};

use detour::{
    Bo, Caching, ColorFormat, ColorTarget, Compiled, CullMode, Device, Draw, DrawPrimitive, Fence, Gpu, Surface, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat,
    Xe,
};
use mlirpass::{GlslReader, GlslStage};

const VARYING: [&str; 2] = [
    "#version 300 es\nlayout(location = 0) in vec2 pos;\nlayout(location = 1) in vec4 color;\nout vec4 c;\nvoid main() { c = color; gl_Position = vec4(pos, 0.0, 1.0); }\n",
    "#version 300 es\nprecision highp float;\nin vec4 c;\nout vec4 o;\nvoid main() { o = c; }\n",
];
const UNIFORM: [&str; 2] = [
    "#version 300 es\nlayout(location = 0) in vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n",
    "#version 300 es\nprecision highp float;\nlayout(std140, binding = 0) uniform U { vec4 color; } u;\nout vec4 o;\nvoid main() { o = u.color; }\n",
];
const TO_SPIRV: &str = "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})";
/// Cells of a grid, each a draw
const GRID: u32 = 8;
const CELL: u32 = 32;
const SIZE: u32 = GRID * CELL;
const FLOATS: usize = 6;
/// The bytes after each row of an imported buffer (acdes pads its canvases' rows so), and what they hold
const PAD: u32 = 64;
/// The heavy frame: draws of 2 MiB of vertices each, as many as acdes's text-heavy desktop makes
const HEAVY: usize = 2 << 20;
const HEAVY_DRAWS: u32 = 12;
const PADDING: u8 = 0xab;
/// struct udmabuf_create, UDMABUF_CREATE, UDMABUF_FLAGS_CLOEXEC
#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}
const UDMABUF_CREATE: u64 = 0x4018_7542;
const UDMABUF_FLAGS_CLOEXEC: u32 = 1;

unsafe extern "C" {
    fn ioctl(fd: i32, request: u64, arg: *mut UdmabufCreate) -> i32;
}

fn program(tools: &Toolchain<'_>, [vert, frag]: [&str; 2]) -> Vec<u8> {
    let mut body = tools.body();
    GlslReader::read(&mut body, vert, GlslStage::Vertex).unwrap();
    GlslReader::read(&mut body, frag, GlslStage::Fragment).unwrap();
    let root = body.root;
    tools.pipeline(TO_SPIRV).and_then(|p| p.run(&mut body, root)).unwrap();
    tools.program(&tools.show(&body)).unwrap()
}

/// Cell `k`'s color
fn color(k: u32) -> [u8; 4] {
    [(k * 37 % 200 + 40) as u8, (k * 91 % 200 + 40) as u8, (k * 53 % 200 + 40) as u8, 255]
}

/// Whether cell `k` takes its color from a uniform (its vertices' color is wrong)
fn uniform(k: u32) -> bool {
    k % 5 == 2 || k % 7 == 3
}

/// Two triangles over pixels x0..x1, y0..y1, in one color
fn quad(out: &mut Vec<f32>, [x0, y0, x1, y1]: [u32; 4], rgba: [u8; 4]) {
    let n = |p: u32| p as f32 / SIZE as f32 * 2.0 - 1.0;
    let c = rgba.map(|v| v as f32 / 255.0);
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y0), (x1, y1), (x0, y1)] {
        out.extend([n(x), n(y)]);
        out.extend(c);
    }
}

/// Cell `k`'s draw: a quad over the whole target first (skipped by `first`) for every fourth, then the cell's quad
/// one to three times; the left half of the cell as the scissor for even ones
fn cell(k: u32, floats: &mut Vec<f32>) -> (u32, u32, Option<[u32; 4]>) {
    floats.clear();
    let (x, y) = (k % GRID * CELL, k / GRID * CELL);
    let skipped = k % 4 == 3;
    if skipped {
        quad(floats, [0, 0, SIZE, SIZE], [255, 0, 0, 255]);
    }
    for _ in 0..k % 3 + 1 {
        quad(floats, [x, y, x + CELL, y + CELL], if uniform(k) { [255, 0, 255, 255] } else { color(k) });
    }
    let first = if skipped { 6 } else { 0 };
    let count = (floats.len() / FLOATS) as u32 - first;
    (first, count, k.is_multiple_of(2).then_some([x, y, CELL / 2, CELL]))
}

fn target(tiling: Tiling, load: TargetLoad<'_>) -> ColorTarget<'_> {
    ColorTarget { width: SIZE, height: SIZE, format: ColorFormat::Rgba8Unorm, tiling, load }
}

/// One frame into `surface`: cleared first when `clear`, a draw a cell
fn frame(device: &Device<'_>, programs: &[Compiled; 2], surface: Surface, tiling: Tiling, clear: bool) {
    let zero = [0u8; 4];
    let mut floats = Vec::new();
    let mut bytes = Vec::new();
    for k in 0..GRID * GRID {
        let (first, count, scissor) = cell(k, &mut floats);
        bytes.clear();
        bytes.extend(floats.iter().flat_map(|f| f.to_le_bytes()));
        let load = if k == 0 && clear { TargetLoad::Fill(&zero, &[]) } else { TargetLoad::Keep(&[]) };
        let block: Vec<u8> = color(k).iter().flat_map(|c| (*c as f32 / 255.0).to_le_bytes()).collect();
        let pos = VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 };
        let (program, elements, uniforms) = match uniform(k) {
            true => (&programs[1], vec![pos], vec![(0, block.as_slice())]),
            false => (&programs[0], vec![pos, VertexElement { buffer: 0, format: VertexFormat::Float32x4, offset: 8 }], Vec::new()),
        };
        let d = Draw {
            buffers: vec![VertexBuffer { bytes: &bytes, stride: (FLOATS * 4) as u32 }],
            elements,
            uniforms,
            textures: Vec::new(),
            primitive: DrawPrimitive::Triangles,
            indices: None,
            first,
            count,
            instances: 1,
            viewport: [0.0, 0.0, SIZE as f32, SIZE as f32],
            depth_range: [0.0, 1.0],
            scissor,
            cull: CullMode::None,
            front_ccw: true,
            blend: None,
            color_mask: [true; 4],
            target: target(tiling, load),
        };
        device.draw(program, &d, surface).unwrap();
    }
}

/// Each cell where its draw put it, in rows `pitch` bytes apart: the whole cell, or its left half (the right one
/// clear); the bytes past each row's pixels `PADDING` when `padded`
fn check(memory: &[u8], pitch: u32, padded: bool) -> Result<(), String> {
    let at = |x: u32, y: u32| -> [u8; 4] { memory[(y * pitch + x * 4) as usize..][..4].try_into().unwrap() };
    let mut wrong = Vec::new();
    for k in 0..GRID * GRID {
        let (x, y) = (k % GRID * CELL, k / GRID * CELL);
        let left = at(x + CELL / 4, y + CELL / 2);
        let right = at(x + 3 * CELL / 4, y + CELL / 2);
        let want_right = if k.is_multiple_of(2) { [0; 4] } else { color(k) };
        if left != color(k) || right != want_right {
            wrong.push(format!("cell {k}: {left:?} {right:?}, not {:?} {want_right:?}", color(k)));
        }
    }
    let padding = |y: u32| &memory[(y * pitch + SIZE * 4) as usize..][..(pitch - SIZE * 4) as usize];
    if padded && (0..SIZE).any(|y| padding(y).iter().any(|b| *b != PADDING)) {
        wrong.push("the row padding written".into());
    }
    if wrong.is_empty() { Ok(()) } else { Err(format!("{} wrong: {}", wrong.len(), wrong.join("; "))) }
}

/// Waits for sync_file `f`
fn done(xe: &Xe, f: Option<OwnedFd>) {
    let s = xe.fence(false).unwrap();
    s.import(f.expect("a fence").as_fd()).unwrap();
    assert!(s.wait(Duration::from_secs(5)).unwrap(), "the frame did not finish");
}

/// A udmabuf of `size` bytes, and its memory mapped
fn udmabuf(size: usize) -> Option<(OwnedFd, &'static mut [u8])> {
    use rustix::fs::{MemfdFlags, SealFlags};
    use rustix::mm::{MapFlags, ProtFlags};
    let memfd = rustix::fs::memfd_create("frame", MemfdFlags::ALLOW_SEALING | MemfdFlags::CLOEXEC).ok()?;
    rustix::fs::ftruncate(&memfd, size as u64).ok()?;
    rustix::fs::fcntl_add_seals(&memfd, SealFlags::SHRINK).ok()?;
    let dev = std::fs::OpenOptions::new().read(true).write(true).open("/dev/udmabuf").ok()?;
    let mut c = UdmabufCreate { memfd: memfd.as_raw_fd() as u32, flags: UDMABUF_FLAGS_CLOEXEC, offset: 0, size: size as u64 };
    let fd = unsafe { ioctl(dev.as_raw_fd(), UDMABUF_CREATE, &mut c) };
    if fd < 0 {
        return None;
    }
    let map = unsafe { rustix::mm::mmap(std::ptr::null_mut(), size, ProtFlags::READ | ProtFlags::WRITE, MapFlags::SHARED, &memfd, 0) }.ok()?;
    Some((unsafe { OwnedFd::from_raw_fd(fd) }, unsafe { std::slice::from_raw_parts_mut(map.cast(), size) }))
}

/// Draws of `HEAVY` bytes of vertices each (a cell's quad over and over), `HEAVY_DRAWS` a frame, into `surface`:
/// the time recording a frame takes and the time its fence then takes, a frame
fn heavy(xe: &Xe, device: &Device<'_>, program: &Compiled, surface: Surface, frames: u32) -> (f64, f64) {
    let mut floats = Vec::new();
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    for k in 0..HEAVY_DRAWS {
        floats.clear();
        let (x, y) = (k % GRID * CELL, k / GRID * CELL);
        while floats.len() * 4 < HEAVY {
            quad(&mut floats, [x, y, x + CELL, y + CELL], color(k));
        }
        bytes.push(floats.iter().flat_map(|f| f.to_le_bytes()).collect());
    }
    let (mut record, mut fence) = (0.0, 0.0);
    for _ in 0..frames {
        let started = Instant::now();
        for b in &bytes {
            let d = Draw {
                buffers: vec![VertexBuffer { bytes: b, stride: (FLOATS * 4) as u32 }],
                elements: vec![VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 }, VertexElement { buffer: 0, format: VertexFormat::Float32x4, offset: 8 }],
                uniforms: Vec::new(),
                textures: Vec::new(),
                primitive: DrawPrimitive::Triangles,
                indices: None,
                first: 0,
                count: (b.len() / (FLOATS * 4)) as u32,
                instances: 1,
                viewport: [0.0, 0.0, SIZE as f32, SIZE as f32],
                depth_range: [0.0, 1.0],
                scissor: None,
                cull: CullMode::None,
                front_ccw: true,
                blend: None,
                color_mask: [true; 4],
                target: target(Tiling::Linear, TargetLoad::Keep(&[])),
            };
            device.draw(program, &d, surface).unwrap();
        }
        let recorded = Instant::now();
        done(xe, device.fence(surface).unwrap());
        record += (recorded - started).as_secs_f64();
        fence += recorded.elapsed().as_secs_f64();
    }
    (record * 1e3 / f64::from(frames), fence * 1e3 / f64::from(frames))
}

/// Draws into dma-buf `fd` imported, `frames` frames, checked through `memory` (the exporter's)
fn imported(xe: &Xe, device: &Device<'_>, programs: &[Compiled; 2], what: &str, fd: BorrowedFd<'_>, memory: &mut dyn FnMut() -> Vec<u8>, frames: u32) {
    let pitch = SIZE * 4 + PAD;
    let surface = device.import(&target(Tiling::Linear, TargetLoad::Keep(&[])), fd, 0, pitch).unwrap();
    frame(device, programs, surface, Tiling::Linear, false);
    done(xe, device.fence(surface).unwrap());
    check(&memory(), pitch, true).unwrap_or_else(|e| panic!("{what}: {e}"));
    let started = Instant::now();
    for _ in 0..frames {
        frame(device, programs, surface, Tiling::Linear, false);
        done(xe, device.fence(surface).unwrap());
    }
    let each = started.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
    check(&memory(), pitch, true).unwrap_or_else(|e| panic!("{what}: {e}"));
    println!("{what}: every cell right, the row padding kept; {each:.2} ms a frame (fenced, nothing read back)");
    let (record, fence) = heavy(xe, device, &programs[0], surface, frames);
    let memory = memory();
    for k in 0..HEAVY_DRAWS {
        let (x, y) = (k % GRID * CELL + CELL / 2, k / GRID * CELL + CELL / 2);
        let at = (y * pitch + x * 4) as usize;
        assert_eq!(memory[at..at + 4], color(k), "{what}: heavy cell {k}");
    }
    println!("{what}, {HEAVY_DRAWS} draws of {} MiB of vertices a frame: recorded in {record:.2} ms, then {fence:.2} ms till its fence", HEAVY >> 20);
    device.free(surface).unwrap();
}

fn main() {
    let frames: u32 = std::env::args().nth(1).map_or(50, |n| n.parse().unwrap());
    let tools = Toolchain::new();
    let bytes = [program(&tools, VARYING), program(&tools, UNIFORM)];
    let programs = [tools.load(&bytes[0]).unwrap(), tools.load(&bytes[1]).unwrap()];
    let xe = Xe::open_first().unwrap();
    let device = Device::new(&xe);

    let surface = device.surface(&target(Tiling::Tile4, TargetLoad::Keep(&[]))).unwrap();
    let started = Instant::now();
    frame(&device, &programs, surface, Tiling::Tile4, true);
    let pixels = device.read(surface, (SIZE * SIZE * 4) as usize).unwrap();
    let first = started.elapsed();
    check(&pixels, SIZE * 4, false).unwrap();
    let started = Instant::now();
    for _ in 0..frames {
        frame(&device, &programs, surface, Tiling::Tile4, true);
        device.read(surface, (SIZE * SIZE * 4) as usize).unwrap();
    }
    let each = started.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
    check(&device.read(surface, (SIZE * SIZE * 4) as usize).unwrap(), SIZE * 4, false).unwrap();
    println!(
        "{} draws a frame, 2 programs taking turns: {} shape(s) lowered; the first frame {:.1} ms, then {each:.2} ms a frame ({frames} frames, read back)",
        GRID * GRID,
        device.lowered(),
        first.as_secs_f64() * 1e3
    );
    device.free(surface).unwrap();
    assert!(device.read(surface, 4).is_err(), "a freed surface read");

    let pitch = SIZE * 4 + PAD;
    let size = (u64::from(pitch) * u64::from(SIZE)).next_multiple_of(4096);
    let fill = |m: &mut [u8]| {
        m.fill(0);
        (0..SIZE).for_each(|y| m[(y * pitch + SIZE * 4) as usize..][..PAD as usize].fill(PADDING));
    };
    let mut bo = xe.bo((size, Caching::WriteCombined).into()).unwrap();
    fill(bo.bytes().unwrap());
    let fd = bo.export().unwrap();
    imported(&xe, &device, &programs, "a GPU buffer imported", fd.as_fd(), &mut || bo.bytes().unwrap().to_vec(), frames);

    match udmabuf(size as usize) {
        Some((fd, memory)) => {
            fill(memory);
            imported(&xe, &device, &programs, "a udmabuf imported", fd.as_fd(), &mut || memory.to_vec(), frames);
        }
        None => println!("a udmabuf imported: none made (no /dev/udmabuf to use)"),
    }
    println!("{} shape(s) lowered in all", device.lowered());
}
