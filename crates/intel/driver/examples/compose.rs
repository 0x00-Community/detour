//! A page composited on Xe3 (`Pages::compose`, as the fallback's `Raster` draws a `Pass`): parts rasterized by `Fine`,
//! placed by `Frost` over their frosted glass, under their drop shadows, through their CSS filters, against vello_cpu
//! (`OptimizeSpeed`) drawing each piece and the plain-Rust definition of the effects (common/frost_model.rs), byte for
//! byte. A wallpaper, a frosted window with a shadow, a panel through a filter (grey, blurred) clipped by its bounds, a
//! card with two shadows; a whole frame, a frame redrawing two damaged regions over it, a whole frame again (the
//! tiles' buffer kept from the first), each one submission. Then a whole frame into someone else's dma-buf (a udmabuf
//! over a memfd, as a compositor's canvas: an offset, rows padded), through `Raster` (the
//! fallback's interface: canvases per plane, passes, a sync_file back), not waited for: the sync_file polled, the memfd read.

use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::FileExt;

use detour::base::{Backdrop, Content, Dmabuf, Effect, Gpu, Native, PaintScene, Part, Pass, Px, Scene, Shadow};
use detour::xe3::{Canvas, Codegen, Context, CpuScene, Pages, Raster};
use detour::xe::Xe;
use detour::vello_cpu::kurbo::{Affine, Circle, Point, Rect, RoundedRect, Stroke};
use detour::vello_cpu::peniko::{Color, Fill, Gradient};
use detour::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};

#[path = "common/frost_model.rs"]
mod model;
use model::Image;

const W: u32 = 320;
const H: u32 = 200;

/// A part's content: a scene in canvas coordinates (whatever region is asked for).
struct Shapes(fn(&mut Scene));

impl Content for Shapes {
    fn scene_for(&self, _: Px) -> Scene {
        let mut s = Scene::new();
        (self.0)(&mut s);
        s
    }
}

fn wallpaper(s: &mut Scene) {
    let g = Gradient::new_linear(Point::new(0.0, 0.0), Point::new(320.0, 200.0)).with_stops([Color::from_rgba8(30, 60, 160, 255), Color::from_rgba8(240, 120, 40, 255)].as_slice());
    s.fill(Fill::NonZero, Affine::IDENTITY, &g, None, &Rect::new(0.0, 0.0, 320.0, 200.0));
    for i in 0..8 {
        let c = Color::from_rgba8(255, 255 - 25 * i as u8, 30 * i as u8, 255);
        s.fill(Fill::NonZero, Affine::IDENTITY, c, None, &Circle::new((20.0 + 40.0 * f64::from(i), 100.0 + 30.0 * (f64::from(i) * 0.9).sin()), 14.0));
    }
}

fn window(s: &mut Scene) {
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(255, 255, 255, 60), None, &RoundedRect::new(40.0, 30.0, 180.0, 130.0, 12.0));
    s.stroke(&Stroke::new(2.0), Affine::IDENTITY, Color::from_rgba8(255, 255, 255, 200), None, &RoundedRect::new(41.0, 31.0, 179.0, 129.0, 11.0));
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(20, 20, 30, 230), None, &Rect::new(56.0, 50.0, 150.0, 58.0));
}

fn panel(s: &mut Scene) {
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(200, 30, 60, 255), None, &RoundedRect::new(200.0, 20.0, 300.0, 90.0, 8.0));
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(30, 200, 90, 200), None, &Circle::new((250.0, 55.0), 22.0));
}

fn card(s: &mut Scene) {
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(250, 250, 245, 255), None, &RoundedRect::new(200.5, 120.25, 290.0, 180.0, 6.0));
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(40, 90, 220, 255), None, &Rect::new(210.0, 130.0, 260.0, 140.0));
}

fn parts() -> Vec<Part<Shapes>> {
    let grey = [0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.2126, 0.7152, 0.0722, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut filter = Effect::default();
    filter.matrix(grey);
    filter.blur(2.0);
    vec![
        Part { content: Shapes(wallpaper), backdrop: None, shadows: Vec::new(), effect: Effect::default(), bounds: None },
        Part {
            content: Shapes(window),
            backdrop: Some(Backdrop { rect: [40.0, 30.0, 140.0, 100.0], radius: 12.0, effect: model::frosted(10.0, 1.6, 1.05) }),
            shadows: vec![Shadow { dx: 4.0, dy: 6.0, blur: 6.0, color: [0.0, 0.0, 0.0, 0.5] }],
            effect: Effect::default(),
            bounds: Some(Px { x: 20, y: 10, w: 190, h: 150 }),
        },
        Part { content: Shapes(panel), backdrop: None, shadows: Vec::new(), effect: filter, bounds: Some(Px { x: 190, y: 10, w: 120, h: 90 }) },
        Part {
            content: Shapes(card),
            backdrop: None,
            shadows: vec![Shadow { dx: 0.0, dy: 3.0, blur: 2.0, color: [0.1, 0.1, 0.3, 0.6] }, Shadow { dx: -5.5, dy: 8.25, blur: 9.0, color: [0.0, 0.0, 0.0, 0.3] }],
            effect: Effect::default(),
            bounds: Some(Px { x: 180, y: 100, w: 130, h: 96 }),
        },
    ]
}

/// The reference: each piece drawn by vello_cpu (as `Pages` replays it), composited by the effects' definition.
fn reference(canvas: &mut Image, parts: &[Part<Shapes>], targets: Option<&[Px]>) {
    let full = Px::sized((W, H));
    let targets = targets.map_or_else(|| vec![full], <[Px]>::to_vec);
    let mut resources = Resources::new();
    targets.iter().for_each(|t| canvas.clear(t.xywh()));
    for part in parts {
        if let Some(b) = &part.backdrop {
            canvas.backdrop(None, b, &targets.iter().map(|t| t.xywh()).collect::<Vec<_>>());
        }
        for (drawn, clip) in part.pieces(&targets, full) {
            let [ox, oy, w, h] = drawn.xywh();
            let (tw, th) = (w.next_multiple_of(4) as u16, h.next_multiple_of(4) as u16);
            let mut ctx = RenderContext::new(tw, th);
            CpuScene::new(&mut ctx, &mut resources).append_scene(part.content.scene_for(drawn), Affine::translate((-f64::from(ox), -f64::from(oy))));
            let mut pixmap = Pixmap::new(tw, th);
            ctx.render_with(&mut pixmap, &mut resources, RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
            let layer = Image { w: usize::from(tw), h: usize::from(th), px: pixmap.data_as_u8_slice().as_chunks::<4>().0.to_vec() };
            for s in &part.shadows {
                canvas.shadow(&layer, s, drawn.xywh(), clip.xywh());
            }
            match part.effect.is_none() {
                true => canvas.place(&layer, [ox, oy], clip.xywh()),
                false => canvas.filtered(&layer, true, &part.effect, drawn.xywh(), clip.xywh()),
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mut pages = Pages::new(&mut cx, &codegen, [W, H])?;
    let mut buffer = cx.buffer(&vec![0x5a; (W * H * 4) as usize])?;
    let canvas = Canvas { addr: buffer.addr, size: [W, H], stride: W };
    let mut want = Image { w: W as usize, h: H as usize, px: vec![[0x5a; 4]; (W * H) as usize] };
    let parts = parts();
    let damaged = [Px { x: 30, y: 20, w: 60, h: 50 }, Px { x: 230, y: 90, w: 70, h: 60 }];
    for (frame, targets) in [None, Some(&damaged[..]), None].into_iter().enumerate() {
        let t = std::time::Instant::now();
        pages.compose(&mut cx, &canvas, &parts, targets, &())?;
        let took = t.elapsed();
        reference(&mut want, &parts, targets);
        let got = buffer.bytes()?;
        let bad = want.px.iter().zip(got.as_chunks::<4>().0).filter(|(w, g)| w != g).count();
        let distinct = got.as_chunks::<4>().0.iter().collect::<std::collections::HashSet<_>>().len();
        println!("frame {frame}: {} pixels ({distinct} colours), {bad} differ; composed in {took:?} (one submission)", W * H);
        assert_eq!(bad, 0);
    }
    udmabuf(&xe, &want)?;
    println!("ok");
    Ok(())
}

/// `struct udmabuf_create` (linux/udmabuf.h) and `UDMABUF_CREATE`, which returns the dma-buf's fd.
#[repr(C)]
struct UdmabufCreate {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

// SAFETY: UDMABUF_CREATE reads the struct and returns a new fd (or fails).
unsafe impl rustix::ioctl::Ioctl for UdmabufCreate {
    type Output = OwnedFd;
    const IS_MUTATING: bool = false;

    fn opcode(&self) -> rustix::ioctl::Opcode {
        rustix::ioctl::opcode::write::<UdmabufCreate>(b'u', 0x42)
    }

    fn as_ptr(&mut self) -> *mut std::ffi::c_void {
        (self as *mut Self).cast()
    }

    unsafe fn output_from_ptr(out: rustix::ioctl::IoctlOutput, _: *mut std::ffi::c_void) -> rustix::io::Result<OwnedFd> {
        // SAFETY: the ioctl succeeded, `out` is the new fd, owned by nobody else
        Ok(unsafe { OwnedFd::from_raw_fd(out) })
    }
}

/// A whole frame into a udmabuf canvas (from byte 4096, rows of `W + 24` pixels) through `Raster` (as acde's compositor
/// drives it), its sync_file polled: as `want`.
fn udmabuf(xe: &Xe, want: &Image) -> Result<(), Box<dyn std::error::Error>> {
    let Ok(dev) = std::fs::File::open("/dev/udmabuf") else {
        println!("udmabuf: /dev/udmabuf not open to us, skipped");
        return Ok(());
    };
    let (offset, stride) = (4096u32, (W + 24) * 4);
    let len = (u64::from(offset) + u64::from(stride) * u64::from(H)).next_multiple_of(4096);
    let memfd = rustix::fs::memfd_create("detour canvas", rustix::fs::MemfdFlags::ALLOW_SEALING | rustix::fs::MemfdFlags::CLOEXEC)?;
    rustix::fs::ftruncate(&memfd, len)?;
    rustix::fs::fcntl_add_seals(&memfd, rustix::fs::SealFlags::SHRINK)?;
    let memfd = std::fs::File::from(memfd);
    memfd.write_all_at(&vec![0x33; len as usize], 0)?;
    // SAFETY: UDMABUF_CREATE on /dev/udmabuf with a sealed memfd of `len` bytes
    let dmabuf = unsafe { rustix::ioctl::ioctl(&dev, UdmabufCreate { memfd: memfd.as_raw_fd() as u32, flags: 1, offset: 0, size: len }) }?;
    let mut raster = Raster::<Xe, Native>::new(xe, [W, H])?;
    let imported = raster.canvases(0, &[Dmabuf { fd: dmabuf.as_fd(), offset, stride, modifier: Dmabuf::LINEAR }], (W, H));
    assert!(imported.iter().all(Result::is_ok), "{imported:?}");
    let submitted = raster.draw(vec![Pass { plane: 0, size: (W, H), parts: parts(), next: 0, targets: None }], &())?;
    let fence = submitted.fence.ok_or("no sync_file")?;
    let mut fds = [rustix::event::PollFd::new(&fence, rustix::event::PollFlags::IN)];
    let ready = rustix::event::poll(&mut fds, Some(&rustix::event::Timespec { tv_sec: 5, tv_nsec: 0 }))?;
    let mut bytes = vec![0; len as usize];
    memfd.read_exact_at(&mut bytes, 0)?;
    let rows = (0..H as usize).map(|y| &bytes[offset as usize + y * stride as usize..][..W as usize * 4]);
    let bad = rows.zip(want.px.chunks(W as usize)).map(|(got, want)| got.as_chunks::<4>().0.iter().zip(want).filter(|(g, w)| g != w).count()).sum::<usize>();
    // the padding and what is before the canvas untouched
    let padding = (0..H as usize).all(|y| bytes[offset as usize + y * stride as usize + W as usize * 4..][..24 * 4].iter().all(|b| *b == 0x33)) && bytes[..offset as usize].iter().all(|b| *b == 0x33);
    println!("udmabuf: sync_file signalled {}, {} pixels, {bad} differ, padding kept {padding}", ready == 1, W * H);
    assert!(ready == 1 && bad == 0 && padding);
    Ok(())
}
