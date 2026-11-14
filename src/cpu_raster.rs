use std::collections::HashMap;
use std::os::fd::{AsFd, OwnedFd};
use std::time::Instant;

use crate::base::{Backdrop, Content, Dmabuf, Effect, PaintScene, Pass, Px, Scene, Shadow, Submitted};
use crate::vello_cpu::kurbo::Affine;
use crate::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};
use crate::xe3::CpuScene;

struct Mapped {
    ptr: *mut u8,
    len: usize,
    offset: usize,
    stride: usize,
    size: (u32, u32),
    fd: OwnedFd,
}

#[repr(C)]
struct DmaBufSync {
    flags: u64,
}

impl DmaBufSync {
    const READ_WRITE: u64 = 3;
    const START: u64 = 0;
    const END: u64 = 4;
}

impl Mapped {
    fn new(d: Dmabuf<'_>, size: (u32, u32)) -> Result<Self, String> {
        if d.modifier != Dmabuf::LINEAR {
            return Err(format!("cpu pages: modifier {:#x}, linear only", d.modifier));
        }
        let fd = d.fd.try_clone_to_owned().map_err(|e| e.to_string())?;
        let len = rustix::fs::seek(&fd, rustix::fs::SeekFrom::End(0)).map_err(|e| format!("cpu pages: the dma-buf's size: {e}"))? as usize;
        let (offset, stride) = (d.offset as usize, d.stride as usize);
        if stride < size.0 as usize * 4 || offset + stride * size.1 as usize > len {
            return Err(format!("cpu pages: a {size:?} canvas does not fit a {len} byte dma-buf (offset {offset}, stride {stride})"));
        }
        let ptr = unsafe { rustix::mm::mmap(std::ptr::null_mut(), len, rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE, rustix::mm::MapFlags::SHARED, &fd, 0) }
            .map_err(|e| format!("cpu pages: mapping the dma-buf: {e}"))?;
        Ok(Self { ptr: ptr.cast(), len, offset, stride, size, fd })
    }

    fn sync(&self, when: u64) {
        let mut sync = DmaBufSync { flags: DmaBufSync::READ_WRITE | when };
        let _ = unsafe { rustix::ioctl::ioctl(self.fd.as_fd(), rustix::ioctl::Updater::<{ rustix::ioctl::opcode::write::<DmaBufSync>(b'b', 0) }, DmaBufSync>::new(&mut sync)) };
    }

    fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        let _ = unsafe { rustix::mm::munmap(self.ptr.cast(), self.len) };
    }
}

struct Layer {
    w: usize,
    h: usize,
    px: Vec<[u8; 4]>,
}

impl From<Pixmap> for Layer {
    fn from(p: Pixmap) -> Self {
        let (w, h) = (usize::from(p.width()), usize::from(p.height()));
        Self { w, h, px: p.data_as_u8_slice().as_chunks::<4>().0.to_vec() }
    }
}

impl Layer {
    fn new(w: usize, h: usize) -> Self {
        Self { w, h, px: vec![[0; 4]; w * h] }
    }

    fn at(&self, x: usize, y: usize) -> [u8; 4] {
        self.px[y * self.w + x]
    }

    fn blurred(mut self, sigma: f32) -> Self {
        if sigma <= 0.0 || sigma.is_nan() || self.w == 0 || self.h == 0 {
            return self;
        }
        for r in Self::boxes(sigma) {
            self.box_pass(r, true);
            self.box_pass(r, false);
        }
        self
    }

    fn boxes(sigma: f32) -> [usize; 3] {
        const N: f32 = 3.0;
        let ideal = (12.0 * sigma * sigma / N + 1.0).sqrt();
        let mut wl = ideal.floor() as i32;
        if wl % 2 == 0 {
            wl -= 1;
        }
        let wu = wl + 2;
        let wlf = wl as f32;
        let m = ((12.0 * sigma * sigma - N * wlf * wlf - 4.0 * N * wlf - 3.0 * N) / (-4.0 * wlf - 4.0)).round() as i32;
        core::array::from_fn(|i| (((if (i as i32) < m { wl } else { wu }) - 1) / 2).max(0) as usize)
    }

    fn box_pass(&mut self, r: usize, rows: bool) {
        if r == 0 {
            return;
        }
        let (lines, len) = if rows { (self.h, self.w) } else { (self.w, self.h) };
        let index = |line: usize, i: usize| if rows { line * self.w + i } else { i * self.w + line };
        let width = (2 * r + 1) as u32;
        let mut line_px = vec![[0u8; 4]; len];
        for line in 0..lines {
            for (i, p) in line_px.iter_mut().enumerate() {
                *p = self.px[index(line, i)];
            }
            let clamped = |i: isize| line_px[i.clamp(0, len as isize - 1) as usize];
            let mut sum = [0u32; 4];
            for i in -(r as isize)..=(r as isize) {
                let p = clamped(i);
                (0..4).for_each(|c| sum[c] += u32::from(p[c]));
            }
            for i in 0..len {
                let out = &mut self.px[index(line, i)];
                *out = core::array::from_fn(|c| ((sum[c] + width / 2) / width) as u8);
                let (add, sub) = (clamped(i as isize + r as isize + 1), clamped(i as isize - r as isize));
                (0..4).for_each(|c| sum[c] = sum[c] + u32::from(add[c]) - u32::from(sub[c]));
            }
        }
    }
}

fn through(p: [u8; 4], m: &[f32; 20]) -> [u8; 4] {
    let a = f32::from(p[3]) / 255.0;
    let c = match a > 0.0 {
        true => [f32::from(p[0]) / 255.0 / a, f32::from(p[1]) / 255.0 / a, f32::from(p[2]) / 255.0 / a, a],
        false => [0.0; 4],
    };
    let o: [f32; 4] = core::array::from_fn(|i| ((0..4).map(|j| m[i * 5 + j] * c[j]).sum::<f32>() + m[i * 5 + 4]).clamp(0.0, 1.0));
    [o[0] * o[3], o[1] * o[3], o[2] * o[3], o[3]].map(|v| (v * 255.0 + 0.5) as u8)
}

fn over(dst: &mut [u8], src: [u8; 4], k: f32) {
    let s = src.map(|v| f32::from(v) * k);
    let keep = 1.0 - s[3] / 255.0;
    for c in 0..4 {
        dst[c] = (s[c] + f32::from(dst[c]) * keep + 0.5).min(255.0) as u8;
    }
}

fn rounded_rect(p: [f32; 2], half: [f32; 2], r: f32) -> f32 {
    let q = [p[0].abs() - half[0] + r, p[1].abs() - half[1] + r];
    q[0].max(0.0).hypot(q[1].max(0.0)) + q[0].max(q[1]).min(0.0) - r
}

struct Target<'m> {
    m: &'m mut Mapped,
    w: i32,
    h: i32,
}

impl Target<'_> {
    fn each(&mut self, rect: Px, clip: Option<Px>, mut f: impl FnMut(i32, i32, &mut [u8])) {
        let Some(r) = rect.intersection(Px { x: 0, y: 0, w: self.w, h: self.h }).and_then(|r| clip.map_or(Some(r), |c| r.intersection(c))) else { return };
        let (offset, stride) = (self.m.offset, self.m.stride);
        let bytes = self.m.bytes();
        for y in r.y..r.y + r.h {
            let row = offset + y as usize * stride;
            for x in r.x..r.x + r.w {
                let at = row + x as usize * 4;
                f(x, y, &mut bytes[at..at + 4]);
            }
        }
    }

    fn copy(&mut self, r: Px) -> Layer {
        let mut out = Layer::new(r.w as usize, r.h as usize);
        self.each(r, None, |x, y, p| out.px[(y - r.y) as usize * r.w as usize + (x - r.x) as usize] = [p[0], p[1], p[2], p[3]]);
        out
    }

    fn clear(&mut self, r: Px) {
        self.each(r, None, |_, _, p| p.fill(0));
    }

    fn place(&mut self, layer: &Layer, at: [i32; 2], clip: Px) {
        let rect = Px { x: at[0], y: at[1], w: layer.w as i32, h: layer.h as i32 };
        self.each(rect, Some(clip), |x, y, p| over(p, layer.at((x - at[0]) as usize, (y - at[1]) as usize), 1.0));
    }

    fn backdrop(&mut self, b: &Backdrop) {
        let [x0, y0] = [b.rect[0].max(0.0).floor() as i32, b.rect[1].max(0.0).floor() as i32];
        let [x1, y1] = [((b.rect[0] + b.rect[2]).max(0.0).ceil() as i32).min(self.w), ((b.rect[1] + b.rect[3]).max(0.0).ceil() as i32).min(self.h)];
        if x1 <= x0 + 1 || y1 <= y0 + 1 {
            return;
        }
        let region = Px { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
        let blurred = self.copy(region).blurred(b.effect.blur);
        let m = b.effect.matrix.unwrap_or(Effect::IDENTITY);
        let half = [b.rect[2] * 0.5, b.rect[3] * 0.5];
        let radius = b.radius.min(half[0].min(half[1]));
        self.each(region, None, |x, y, p| {
            let centre = [x as f32 + 0.5 - b.rect[0] - half[0], y as f32 + 0.5 - b.rect[1] - half[1]];
            let mask = (0.5 - rounded_rect(centre, half, radius)).clamp(0.0, 1.0);
            if mask > 0.0 {
                over(p, through(blurred.at((x - x0) as usize, (y - y0) as usize), &m), mask);
            }
        });
    }

    fn shadow(&mut self, layer: &Layer, s: &Shadow, at: [i32; 2], clip: Px) {
        if layer.w < 2 || layer.h < 2 {
            return;
        }
        let blurred = Layer { w: layer.w, h: layer.h, px: layer.px.iter().map(|p| [0, 0, 0, p[3]]).collect() }.blurred(s.blur);
        let (dx, dy) = (s.dx.round() as i32, s.dy.round() as i32);
        let rect = Px { x: at[0] + dx, y: at[1] + dy, w: layer.w as i32, h: layer.h as i32 };
        let [r, g, b, a] = s.color;
        self.each(rect, Some(clip), |x, y, p| {
            let k = f32::from(blurred.at((x - rect.x) as usize, (y - rect.y) as usize)[3]) / 255.0 * a;
            over(p, [r * k, g * k, b * k, k].map(|v| (v * 255.0 + 0.5) as u8), 1.0);
        });
    }

    fn filtered(&mut self, layer: Layer, e: &Effect, at: [i32; 2], clip: Px) {
        if layer.w < 2 || layer.h < 2 {
            return;
        }
        let blurred = layer.blurred(e.blur);
        let m = e.matrix.unwrap_or(Effect::IDENTITY);
        let rect = Px { x: at[0], y: at[1], w: blurred.w as i32, h: blurred.h as i32 };
        self.each(rect, Some(clip), |x, y, p| over(p, through(blurred.at((x - at[0]) as usize, (y - at[1]) as usize), &m), 1.0));
    }
}

#[derive(Default)]
pub struct CpuRaster {
    canvases: HashMap<usize, Vec<Option<Mapped>>>,
    resources: Resources,
}

impl CpuRaster {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn canvases(&mut self, plane: usize, dmabufs: &[Dmabuf<'_>], size: (u32, u32)) -> Vec<Result<(), String>> {
        let mut results = Vec::new();
        let mapped = dmabufs
            .iter()
            .map(|d| {
                let m = Mapped::new(*d, size);
                results.push(m.as_ref().map(|_| ()).map_err(Clone::clone));
                m.ok()
            })
            .collect();
        self.canvases.insert(plane, mapped);
        results
    }

    pub fn release(&mut self, plane: usize) {
        self.canvases.remove(&plane);
    }

    pub fn idle(&mut self) {
        self.resources = Resources::new();
    }

    fn layer(&mut self, scene: Scene, place: Affine, (w, h): (u16, u16)) -> Layer {
        let mut ctx = RenderContext::new(w, h);
        CpuScene::new(&mut ctx, &mut self.resources).append_scene(scene, place);
        let mut pixmap = Pixmap::new(w, h);
        ctx.render_with(&mut pixmap, &mut self.resources, RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
        pixmap.into()
    }

    pub fn draw<C: Content>(&mut self, passes: Vec<Pass<C>>) -> Result<Submitted, String> {
        let started = Instant::now();
        for pass in passes {
            self.draw_pass(pass)?;
        }
        Ok(Submitted { fence: None, waited: Default::default(), recorded: started.elapsed() })
    }

    fn draw_pass<C: Content>(&mut self, Pass { plane, size, parts, next, targets }: Pass<C>) -> Result<(), String> {
        let full = Px::sized(size);
        let mut mapped = self.canvases.get_mut(&plane).and_then(|c| c.get_mut(next)?.take()).ok_or("the canvas is not mapped yet")?;
        if size.0 > mapped.size.0 || size.1 > mapped.size.1 {
            let canvas = mapped.size;
            self.put_back(plane, next, mapped);
            return Err(format!("a {size:?} pass into a {canvas:?} canvas"));
        }
        mapped.sync(DmaBufSync::START);
        let drawn = self.compose(&mut mapped, &parts, targets.unwrap_or_else(|| vec![full]), full);
        mapped.sync(DmaBufSync::END);
        self.put_back(plane, next, mapped);
        drawn
    }

    fn compose<C: Content>(&mut self, mapped: &mut Mapped, parts: &[crate::base::Part<C>], targets: Vec<Px>, full: Px) -> Result<(), String> {
        let mut canvas = Target { m: mapped, w: full.w, h: full.h };
        targets.iter().for_each(|t| canvas.clear(*t));
        for part in parts {
            if let Some(b) = part.backdrop.as_ref().filter(|b| targets.iter().any(|t| Px::covering(b.rect).overlaps(*t))) {
                canvas.backdrop(b);
            }
            for (drawn, clip) in part.pieces(&targets, full) {
                let [ox, oy, w, h] = drawn.xywh();
                let scene = part.content.scene_for(drawn);
                if scene.commands.is_empty() || w == 0 || h == 0 {
                    continue;
                }
                let dims = (u16::try_from(w).map_err(|_| "a part past 65535 pixels")?, u16::try_from(h).map_err(|_| "a part past 65535 pixels")?);
                let layer = self.layer(scene, Affine::translate((-f64::from(ox), -f64::from(oy))), dims);
                let at = [ox as i32, oy as i32];
                for s in &part.shadows {
                    canvas.shadow(&layer, s, at, clip);
                }
                match part.effect.is_none() {
                    true => canvas.place(&layer, at, clip),
                    false => canvas.filtered(layer, &part.effect, at, clip),
                }
            }
        }
        Ok(())
    }

    fn put_back(&mut self, plane: usize, next: usize, mapped: Mapped) {
        if let Some(slot) = self.canvases.get_mut(&plane).and_then(|c| c.get_mut(next)) {
            *slot = Some(mapped);
        }
    }

    pub fn icon(&mut self, dmabuf: Dmabuf<'_>, (width, height): (u32, u32), scene: Scene) -> Result<(), String> {
        let mut mapped = Mapped::new(dmabuf, (width, height))?;
        let dims = (u16::try_from(width).map_err(|_| "an icon past 65535 pixels")?, u16::try_from(height).map_err(|_| "an icon past 65535 pixels")?);
        let layer = self.layer(scene, Affine::IDENTITY, dims);
        mapped.sync(DmaBufSync::START);
        let full = Px::sized((width, height));
        let mut canvas = Target { m: &mut mapped, w: full.w, h: full.h };
        canvas.clear(full);
        canvas.place(&layer, [0, 0], full);
        mapped.sync(DmaBufSync::END);
        Ok(())
    }
}
