//! The device on an AMD RDNA2 card (amdgpu): surfaces in GTT, linear; a draw recorded into the command buffer
//! being built (`Rdna2Draw`'s registers, only those it changes), its vertices, indices and uniforms copied into a
//! ring; the command buffer copied into a ring of its own, uncached, and submitted as one IB.

use std::cell::{RefCell, RefMut};
use std::collections::VecDeque;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::rc::Rc;

use mlirformat::Error;

use super::{DRM_FORMAT_ABGR8888, DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_R8, Exported, Surface, io, wait};
use crate::{
    Amdgpu, AmdgpuBo, AmdgpuIb, AmdgpuQueue, AmdgpuVm, Bo, BoDesc, Caching, ColorFormat, ColorTarget, Compiled, Draw, Engine, Fence, Gpu, Host, IndexFormat, MemType, Native, Queue, Rdna2Draw, Rdna2Surface,
    Rdna2Packet, Rdna2Write, Rdna2Shader, Rdna2Shadow, Shaders, Syncobj, TargetLoad, Tiling, Vm,
};

type SurfaceMem<'a> = super::SurfaceMem<'a, AmdgpuBo<'a>>;

/// Where the programs' code goes: chunks `CODE_SPAN` apart from `CODE_ADDR`, each `CODE_CHUNK` bytes or what one
/// program takes; each shader on 256 bytes
const CODE_ADDR: u64 = 1 << 32;
const CODE_SPAN: u64 = 1 << 26;
const CODE_CHUNK: usize = 1 << 20;
const CODE_ALIGN: usize = 256;
/// s_code_end: what the code chunks hold where no shader is (the instruction prefetch runs past a shader's end)
const CODE_END: u32 = 0xbf9f_0000;
/// The ring the draws' bytes go in, the command buffers' (mapped uncached: the CP fetches an IB before the IB's
/// ACQUIRE_MEM invalidates GL2, which may still hold what a batch before read at that place), their least size,
/// where each piece starts
const RING_ADDR: u64 = 1 << 36;
const CMDS_ADDR: u64 = 1 << 37;
const RING_MIN: usize = 4 << 20;
const RING_ALIGN: usize = 256;
/// Batches submitted the ring keeps the fences of (the oldest waited for past them: done long ago)
const INFLIGHT: usize = 64;
/// Where surface slot i is mapped (`SURFACE_ADDR + i * SURFACE_SPAN`): a surface takes at most a span
const SURFACE_ADDR: u64 = 1 << 40;
const SURFACE_SPAN: u64 = 1 << 30;
/// A page: what amdgpu maps
const PAGE: u64 = 4096;
/// The dwords a draw adds to the command buffer at most, the preamble's and the end's
const DRAW_DWORDS: usize = 1024;
const PREAMBLE_DWORDS: usize = 512;
/// The dwords a command buffer takes before it is submitted
const BATCH_DWORDS: usize = 1 << 18;

/// The code of the programs drawn: chunks mapped once, pieces taken (never given back: a program stays)
struct Code<'a> {
    chunks: Vec<(AmdgpuBo<'a>, u64, usize)>,
    /// where the last chunk's untaken bytes start
    at: usize,
}

impl<'a> Code<'a> {
    /// `code` placed: its address (on `CODE_ALIGN`)
    fn put(&mut self, amd: &'a Amdgpu, vm: &AmdgpuVm<'a>, code: &[u8]) -> Result<u64, Error> {
        let len = code.len().next_multiple_of(CODE_ALIGN) + CODE_ALIGN;
        if self.chunks.last().is_none_or(|(_, _, size)| self.at + len > *size) {
            let size = len.max(CODE_CHUNK);
            let addr = CODE_ADDR + self.chunks.len() as u64 * CODE_SPAN;
            if size as u64 > CODE_SPAN || addr + CODE_SPAN > RING_ADDR {
                return Err(Error(format!("no room for {len} bytes of programs")));
            }
            let mut bo = amd.bo((size as u64, Caching::WriteCombined).into()).map_err(io)?;
            let end = CODE_END.to_le_bytes();
            bo.bytes().map_err(io)?.as_chunks_mut::<4>().0.iter_mut().for_each(|w| *w = end);
            vm.bind(&bo, 0, addr, size as u64, MemType::Streaming, &[]).map_err(io)?;
            self.chunks.push((bo, addr, size));
            self.at = 0;
        }
        let (bo, base, _) = self.chunks.last_mut().unwrap();
        bo.bytes().map_err(io)?[self.at..self.at + code.len()].copy_from_slice(code);
        let addr = *base + self.at as u64;
        self.at += len;
        Ok(addr)
    }
}

/// Bytes the GPU reads (vertices, indices, uniforms, command buffers), each batch's after the last's around one
/// buffer: a place is written once the batches that read what it held are done. Positions count bytes ever written
/// (a place is a position modulo the size)
struct Ring<'a> {
    addr: u64,
    mem: MemType,
    bo: Option<AmdgpuBo<'a>>,
    size: usize,
    head: u64,
    /// where the batch being recorded started
    start: u64,
    /// each batch submitted: where it started, its fence
    inflight: VecDeque<(u64, Rc<Syncobj<'a>>)>,
}

impl<'a> Ring<'a> {
    fn new(addr: u64, mem: MemType) -> Ring<'a> {
        Ring { addr, mem, bo: None, size: 0, head: 0, start: 0, inflight: VecDeque::new() }
    }

    /// Whether `len` more bytes of the batch being recorded fit (the batch never laps itself; a piece that would
    /// cross the end skips to the start)
    fn fits(&self, len: usize) -> bool {
        (self.head - self.start) as usize + 2 * len <= self.size
    }

    /// `bytes` placed: their address
    fn put(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.put_with(bytes.len(), |to| to.copy_from_slice(bytes))
    }

    /// `len` bytes placed, `fill` writing them: their address
    fn put_with(&mut self, len: usize, fill: impl FnOnce(&mut [u8])) -> Result<u64, Error> {
        let size = self.size as u64;
        let mut at = self.head.next_multiple_of(RING_ALIGN as u64);
        if at % size + len as u64 > size {
            at = at.next_multiple_of(size);
        }
        let end = at + len as u64;
        // the batches that read what was here
        let mut last = None;
        while self.inflight.front().is_some_and(|(start, _)| *start + size < end) {
            last = self.inflight.pop_front();
        }
        if let Some((_, f)) = last {
            wait(&f, "a draw")?;
        }
        let place = (at % size) as usize;
        let bo = self.bo.as_mut().ok_or_else(|| Error("a ring with no memory".into()))?;
        fill(&mut bo.bytes().map_err(io)?[place..place + len]);
        self.head = end;
        Ok(self.addr + place as u64)
    }

    /// The batch recorded went to the GPU, `done` once it is
    fn submitted(&mut self, done: Rc<Syncobj<'a>>) -> Result<(), Error> {
        self.inflight.push_back((self.start, done));
        self.start = self.head;
        if self.inflight.len() > INFLIGHT
            && let Some((_, f)) = self.inflight.pop_front()
        {
            wait(&f, "a draw")?;
        }
        Ok(())
    }

    /// Room for a batch of `len` bytes (nothing running): the ring from its start, larger when it was too small
    fn restart(&mut self, amd: &'a Amdgpu, vm: &AmdgpuVm<'a>, len: usize) -> Result<(), Error> {
        self.inflight.clear();
        (self.head, self.start) = (0, 0);
        if 2 * len <= self.size {
            return Ok(());
        }
        let size = (4 * len).next_power_of_two().max(RING_MIN);
        if let Some(old) = self.bo.take() {
            vm.unbind(self.addr, old.size(), &[]).map_err(io)?;
        }
        let bo = amd.bo((size as u64, Caching::WriteCombined).into()).map_err(io)?;
        vm.bind(&bo, 0, self.addr, size as u64, self.mem, &[]).map_err(io)?;
        (self.bo, self.size) = (Some(bo), size);
        Ok(())
    }
}

/// The device's address space and what is in it
struct State<'a> {
    amd: &'a Amdgpu,
    vm: AmdgpuVm<'a>,
    queue: AmdgpuQueue<'a>,
    code: Code<'a>,
    /// each program's code: its id, the vertex and the fragment shader's addresses
    programs: Vec<(u64, u64, u64)>,
    ring: Ring<'a>,
    cmds: Ring<'a>,
    surfaces: Vec<Option<SurfaceMem<'a>>>,
    /// the command buffer being recorded, the registers it has set, its index type
    ib: Vec<u32>,
    shadow: Rdna2Shadow,
    index_type: Option<IndexFormat>,
    /// what the next submission waits for, the last submission, the surfaces the batch draws into and samples
    waits: Vec<Syncobj<'a>>,
    last: Option<Rc<Syncobj<'a>>>,
    written: Vec<usize>,
    sampled: Vec<usize>,
    /// a draw's registers and user data, kept for their memory
    regs: Vec<Rdna2Write>,
    user: Vec<u32>,
}

impl<'a> State<'a> {
    fn new(amd: &'a Amdgpu) -> Result<State<'a>, Error> {
        let vm = amd.vm().map_err(io)?;
        let queue = amd.queue(&vm, Engine::Render).map_err(io)?;
        let (mut ring, mut cmds) = (Ring::new(RING_ADDR, MemType::Streaming), Ring::new(CMDS_ADDR, MemType::Scanout));
        ring.restart(amd, &vm, RING_MIN / 4)?;
        cmds.restart(amd, &vm, RING_MIN / 4)?;
        Ok(State {
            amd,
            vm,
            queue,
            code: Code { chunks: Vec::new(), at: 0 },
            programs: Vec::new(),
            ring,
            cmds,
            surfaces: Vec::new(),
            ib: Vec::new(),
            shadow: Rdna2Shadow::default(),
            index_type: None,
            waits: Vec::new(),
            last: None,
            written: Vec::new(),
            sampled: Vec::new(),
            regs: Vec::new(),
            user: Vec::new(),
        })
    }

    fn target(&mut self, s: Surface) -> Result<&mut SurfaceMem<'a>, Error> {
        self.surfaces.get_mut(s.0).and_then(Option::as_mut).ok_or_else(|| Error(format!("no surface {s:?}")))
    }

    /// `bo` a surface for targets like `t`, its pixels from byte `offset`, rows `pitch` bytes apart
    fn adopt(&mut self, t: &ColorTarget<'_>, bo: AmdgpuBo<'a>, offset: u64, pitch: u32) -> Result<Surface, Error> {
        if t.tiling != Tiling::Linear {
            return Err(Error(format!("a {:?} surface: RDNA2 draws linear ones", t.tiling)));
        }
        if t.width == 0 || t.height == 0 {
            return Err(Error(format!("a {}x{} surface", t.width, t.height)));
        }
        if !offset.is_multiple_of(PAGE) {
            return Err(Error(format!("a surface at byte {offset} of its memory: not on {PAGE}")));
        }
        if !pitch.is_multiple_of(Rdna2Draw::PITCH_ALIGN) {
            return Err(Error(format!("rows of {pitch} bytes: RDNA2 draws rows on {} bytes", Rdna2Draw::PITCH_ALIGN)));
        }
        let need = u64::from(pitch) * u64::from(t.height);
        if pitch < t.row() || offset + need > bo.size() {
            return Err(Error(format!("a {}x{} surface, rows of {pitch} bytes from {offset}, in {} bytes", t.width, t.height, bo.size())));
        }
        let range = need.next_multiple_of(PAGE).min(bo.size() - offset);
        if range > SURFACE_SPAN {
            return Err(Error(format!("a surface of {range} bytes: more than {SURFACE_SPAN}")));
        }
        let taken: Vec<usize> = self.surfaces.iter().flatten().map(|t| t.slot).collect();
        let slot = (0..).find(|i| !taken.contains(i)).unwrap_or(0);
        let addr = SURFACE_ADDR + slot as u64 * SURFACE_SPAN;
        self.vm.bind(&bo, offset, addr, range, MemType::Streaming, &[]).map_err(|e| Error(format!("a surface's memory not mapped: {e}")))?;
        let target = SurfaceMem { bo, slot, addr, offset: offset as usize, range, width: t.width, height: t.height, format: t.format, tiling: t.tiling, pitch, fence: None };
        match self.surfaces.iter().position(Option::is_none) {
            Some(i) => {
                self.surfaces[i] = Some(target);
                Ok(Surface(i))
            }
            None => {
                self.surfaces.push(Some(target));
                Ok(Surface(self.surfaces.len() - 1))
            }
        }
    }

    /// The batch recorded ended and submitted, after what it waits for
    fn submit(&mut self) -> Result<(), Error> {
        if self.ib.is_empty() {
            return Ok(());
        }
        Rdna2Packet::pad(&mut self.ib);
        let ib = &self.ib;
        let addr = self.cmds.put_with(4 * ib.len(), |to| to.as_chunks_mut::<4>().0.iter_mut().zip(ib).for_each(|(to, w)| *to = w.to_le_bytes()))?;
        Native::fence();
        let done = Rc::new(self.amd.fence(false).map_err(io)?);
        let waits: Vec<&Syncobj<'a>> = self.waits.iter().collect();
        self.queue.submit(&AmdgpuIb { addr, dwords: self.ib.len() as u32 }, &waits, &[&done]).map_err(io)?;
        self.waits.clear();
        self.ib.clear();
        self.index_type = None;
        for s in self.written.drain(..).chain(self.sampled.drain(..)) {
            if let Some(Some(t)) = self.surfaces.get_mut(s) {
                t.fence = Some(done.clone());
            }
        }
        self.last = Some(done.clone());
        self.ring.submitted(done.clone())?;
        self.cmds.submitted(done)
    }

    /// Everything recorded submitted and done
    fn settle(&mut self) -> Result<(), Error> {
        self.submit()?;
        if let Some(f) = self.last.take() {
            wait(&f, "a draw")?;
        }
        Ok(())
    }

    /// The draws into surface `s`, and those sampling it, submitted and done
    fn idle(&mut self, s: Surface) -> Result<(), Error> {
        if self.written.contains(&s.0) || self.sampled.contains(&s.0) {
            self.submit()?;
        }
        if let Some(f) = &self.target(s)?.fence {
            wait(f, "a draw")?;
        }
        Ok(())
    }

    /// What `s` holds before a draw, as `load` says: nothing to do when it keeps what it has; otherwise its draws so
    /// far done, then its memory written
    fn load(&mut self, s: Surface, load: &TargetLoad<'_>) -> Result<(), Error> {
        if let TargetLoad::Keep([]) = load {
            return Ok(());
        }
        self.idle(s)?;
        self.target(s)?.load(load)
    }

    /// Program `p`'s code placed (once): its vertex and fragment shader's addresses
    fn program(&mut self, id: u64, vs: &Rdna2Shader, fs: &Rdna2Shader) -> Result<(u64, u64), Error> {
        if let Some(&(_, v, f)) = self.programs.iter().find(|(i, _, _)| *i == id) {
            return Ok((v, f));
        }
        Rdna2Draw::check(vs, fs)?;
        let v = self.code.put(self.amd, &self.vm, &vs.code)?;
        let f = self.code.put(self.amd, &self.vm, &fs.code)?;
        self.programs.push((id, v, f));
        Ok((v, f))
    }

    /// Room in the rings for the batch with a draw of `data` more bytes (the batch submitted first when they do not
    /// fit; a ring larger when one draw does not)
    fn room(&mut self, data: usize) -> Result<(), Error> {
        let data = data + 4 * RING_ALIGN;
        let cmds = |ib: usize| 4 * (ib + DRAW_DWORDS + PREAMBLE_DWORDS) + RING_ALIGN;
        if self.ib.len() + DRAW_DWORDS > BATCH_DWORDS || !self.ring.fits(data) || !self.cmds.fits(cmds(self.ib.len())) {
            self.submit()?;
        }
        if !self.ring.fits(data) || !self.cmds.fits(cmds(self.ib.len())) {
            self.settle()?;
            self.ring.restart(self.amd, &self.vm, data)?;
            self.cmds.restart(self.amd, &self.vm, cmds(0))?;
        }
        Ok(())
    }

    /// Draw `d` of program `vs`, `fs` (at `code`) into `s` recorded: its bytes in the ring, the registers it changes,
    /// its packets
    fn record(&mut self, vs: &Rdna2Shader, fs: &Rdna2Shader, code: (u64, u64), d: &Draw<'_>, s: Surface, pixels: [u32; 4]) -> Result<(), Error> {
        let mut vertices = Vec::with_capacity(d.buffers.len());
        for b in &d.buffers {
            vertices.push(if b.bytes.is_empty() { RING_ADDR } else { self.ring.put(b.bytes)? });
        }
        let mut uniforms = Vec::with_capacity(d.uniforms.len());
        for (binding, bytes) in &d.uniforms {
            uniforms.push((*binding, if bytes.is_empty() { RING_ADDR } else { self.ring.put(bytes)? }));
        }
        let indices = match d.indices {
            Some(ib) => {
                let size = match ib.format {
                    IndexFormat::U16 => 2,
                    IndexFormat::U32 => 4,
                };
                Some((self.ring.put(ib.bytes)?, ib.format, (ib.bytes.len() / size) as u32))
            }
            None => None,
        };
        let ubo = |binding: u32| uniforms.iter().find(|(b, _)| *b == binding).map(|(_, a)| *a).ok_or_else(|| Error(format!("a shader reads uniform block {binding}, the draw gives none")));

        let mut regs = std::mem::take(&mut self.regs);
        let mut user = std::mem::take(&mut self.user);
        regs.clear();
        let t = self.target(s)?;
        let target = Rdna2Surface { addr: t.addr, width: t.width, height: t.height, pitch: t.pitch / t.format.bytes(), format: t.format };
        Rdna2Draw::program(&mut regs, vs, fs, code.0, code.1);
        Rdna2Draw::target(&mut regs, &target, d.color_mask);
        Rdna2Draw::blend(&mut regs, d);
        Rdna2Draw::raster(&mut regs, d);
        Rdna2Draw::viewport(&mut regs, d, pixels);

        // the vertex shader's user SGPRs: a V# per input, the uniform blocks' addresses, the first vertex
        user.clear();
        user.resize(vs.user_sgprs() as usize, 0);
        for (k, e) in d.elements.iter().enumerate() {
            let b = d.buffers.get(e.buffer as usize).ok_or_else(|| Error(format!("vertex input {k} reads buffer {}, the draw gives {}", e.buffer, d.buffers.len())))?;
            let bytes = (b.bytes.len() as u64).saturating_sub(u64::from(e.offset));
            let v = Rdna2Draw::vertex(vertices[e.buffer as usize] + u64::from(e.offset), b.stride, bytes, e.format);
            let at = vs.input_sgpr(k as u32) as usize;
            user[at..at + 4].copy_from_slice(&v);
        }
        for &binding in &vs.ubos {
            let (a, at) = (ubo(binding)?, vs.ubo_sgpr(binding).unwrap_or(0) as usize);
            user[at..at + 2].copy_from_slice(&[a as u32, (a >> 32) as u32]);
        }
        user[vs.first_vertex_sgpr() as usize] = if d.indices.is_some() { 0 } else { d.first };
        Rdna2Draw::user_data(&mut regs, true, &user);

        // the fragment shader's: a T# and S# per texture, the uniform blocks' addresses
        user.clear();
        user.resize(fs.user_sgprs() as usize, 0);
        for &binding in &fs.textures {
            let x = d.textures.iter().find(|x| x.binding == binding).ok_or_else(|| Error(format!("the fragment shader samples binding {binding}, the draw gives no texture")))?;
            let t = self.target(x.surface)?;
            let image = Rdna2Surface { addr: t.addr, width: t.width, height: t.height, pitch: t.pitch / t.format.bytes(), format: t.format };
            let at = fs.texture_sgpr(binding).unwrap_or(0) as usize;
            user[at..at + 8].copy_from_slice(&Rdna2Draw::image(&image));
            user[at + 8..at + 12].copy_from_slice(&Rdna2Draw::sampler(x.filter, x.wrap));
        }
        for &binding in &fs.ubos {
            let (a, at) = (ubo(binding)?, fs.ubo_sgpr(binding).unwrap_or(0) as usize);
            user[at..at + 2].copy_from_slice(&[a as u32, (a >> 32) as u32]);
        }
        Rdna2Draw::user_data(&mut regs, false, &user);

        if self.ib.is_empty() {
            Rdna2Draw::preamble(&mut self.ib, &mut self.shadow, RING_ADDR);
        }
        self.shadow.set(&mut self.ib, &regs);
        if let Some((_, format, _)) = indices
            && self.index_type != Some(format)
        {
            Rdna2Draw::index_type(&mut self.ib, format);
            self.index_type = Some(format);
        }
        Rdna2Draw::packets(&mut self.ib, d.count, d.first, d.instances, indices);
        (self.regs, self.user) = (regs, user);

        if !self.written.contains(&s.0) {
            self.written.push(s.0);
        }
        for x in &d.textures {
            if !self.sampled.contains(&x.surface.0) {
                self.sampled.push(x.surface.0);
            }
        }
        Ok(())
    }
}

/// The device on an amdgpu card (`Device`'s): its state set up on first use
pub struct AmdgpuDevice<'a> {
    pub amd: &'a Amdgpu,
    state: RefCell<Option<State<'a>>>,
}

impl<'a> AmdgpuDevice<'a> {
    pub fn new(amd: &'a Amdgpu) -> AmdgpuDevice<'a> {
        AmdgpuDevice { amd, state: RefCell::new(None) }
    }

    fn state(&self) -> Result<RefMut<'_, State<'a>>, Error> {
        let mut st = self.state.borrow_mut();
        if st.is_none() {
            *st = Some(State::new(self.amd)?);
        }
        Ok(RefMut::map(st, |s| s.as_mut().unwrap()))
    }

    /// A new surface's memory for `t`: rows on 256 bytes
    fn new_surface(&self, t: &ColorTarget<'_>, caching: Caching, scanout: bool) -> Result<Surface, Error> {
        let mut st = self.state()?;
        let pitch = t.row().next_multiple_of(Rdna2Draw::PITCH_ALIGN);
        let size = (u64::from(pitch) * u64::from(t.height)).next_multiple_of(PAGE).max(PAGE);
        let bo = self.amd.bo(BoDesc { size, caching, scanout }).map_err(io)?;
        let s = st.adopt(t, bo, 0, pitch)?;
        st.load(s, &t.load)?;
        Ok(s)
    }

    pub fn surface(&self, t: &ColorTarget<'_>) -> Result<Surface, Error> {
        self.new_surface(t, Caching::WriteCombined, false)
    }

    pub fn scanout(&self, t: &ColorTarget<'_>) -> Result<Surface, Error> {
        self.new_surface(t, Caching::WriteCombined, true)
    }

    pub fn export(&self, s: Surface) -> Result<Exported, Error> {
        let mut st = self.state()?;
        let t = st.target(s)?;
        let fourcc = match t.format {
            ColorFormat::Rgba8Unorm => DRM_FORMAT_ABGR8888,
            ColorFormat::R8Unorm => DRM_FORMAT_R8,
            ColorFormat::Rgba32Float => return Err(Error("an Rgba32Float surface: no DRM format to export it as".into())),
        };
        let fd = t.bo.export().map_err(io)?;
        Ok(Exported { fd, offset: t.offset as u64, pitch: t.pitch, modifier: DRM_FORMAT_MOD_LINEAR, fourcc })
    }

    pub fn import(&self, t: &ColorTarget<'_>, fd: BorrowedFd<'_>, offset: u64, stride: u32) -> Result<Surface, Error> {
        let mut st = self.state()?;
        let bo = self.amd.import(fd).map_err(|e| Error(format!("a dma-buf not imported: {e}")))?;
        st.adopt(t, bo, offset, stride)
    }

    pub fn free(&self, s: Surface) -> Result<(), Error> {
        let mut st = self.state()?;
        st.idle(s)?;
        let t = st.surfaces.get_mut(s.0).and_then(Option::take).ok_or_else(|| Error(format!("no surface {s:?}")))?;
        st.vm.unbind(t.addr, t.range, &[]).map_err(io)
    }

    pub fn write(&self, s: Surface, rect: [u32; 4], pixels: &[u8]) -> Result<(), Error> {
        let mut st = self.state()?;
        st.idle(s)?;
        st.target(s)?.write_rect(rect, pixels)
    }

    pub fn lowered(&self) -> usize {
        self.state.borrow().as_ref().map_or(0, |s| s.programs.len())
    }

    pub fn read(&self, s: Surface, len: usize) -> Result<Vec<u8>, Error> {
        let mut st = self.state()?;
        st.idle(s)?;
        let mut pixels = st.target(s)?.pixels()?;
        if len > pixels.len() {
            return Err(Error(format!("{len} bytes of a surface of {}", pixels.len())));
        }
        pixels.truncate(len);
        Ok(pixels)
    }

    pub fn raw(&self, s: Surface) -> Result<Vec<u8>, Error> {
        let mut st = self.state()?;
        st.idle(s)?;
        Ok(st.target(s)?.memory()?.to_vec())
    }

    pub fn flush(&self, s: Surface) -> Result<(), Error> {
        let mut st = self.state()?;
        st.target(s)?;
        st.submit()
    }

    pub fn finish(&self, s: Surface) -> Result<(), Error> {
        self.state()?.idle(s)
    }

    pub fn fence(&self, s: Surface) -> Result<Option<OwnedFd>, Error> {
        let mut st = self.state()?;
        if st.written.contains(&s.0) {
            st.submit()?;
        }
        st.target(s)?.fence.as_ref().map(|f| f.export().map_err(io)).transpose()
    }

    pub fn after(&self, fd: BorrowedFd<'_>) -> Result<(), Error> {
        let mut st = self.state()?;
        let f = self.amd.fence(false).map_err(io)?;
        f.import(fd).map_err(io)?;
        st.waits.push(f);
        Ok(())
    }

    pub fn draw(&self, program: &Compiled, d: &Draw<'_>, s: Surface) -> Result<(), Error> {
        let Shaders::Rdna2 { vs, fs } = &program.shaders else {
            return Err(Error(format!("a program compiled for {:?}, drawn on RDNA2", program.isa())));
        };
        let mut st = self.state()?;
        let t = st.target(s)?;
        if !t.holds(&d.target) {
            return Err(Error(format!(
                "a draw into a {}x{} {:?} {:?} target, surface {s:?} is {}x{} {:?} {:?}",
                d.target.width, d.target.height, d.target.format, d.target.tiling, t.width, t.height, t.format, t.tiling
            )));
        }
        if d.elements.len() != vs.inputs as usize {
            return Err(Error(format!("the vertex shader reads {} inputs, the draw gives {}", vs.inputs, d.elements.len())));
        }
        let mut data: usize = d.buffers.iter().map(|b| b.bytes.len()).sum::<usize>() + d.uniforms.iter().map(|(_, u)| u.len()).sum::<usize>();
        if let Some(ib) = d.indices {
            let size = match ib.format {
                IndexFormat::U16 => 2,
                IndexFormat::U32 => 4,
            };
            if !ib.bytes.len().is_multiple_of(size) || (d.first as usize + d.count as usize) * size > ib.bytes.len() {
                return Err(Error(format!("indices {}..{} of {} bytes of {:?} indices", d.first, d.first + d.count, ib.bytes.len(), ib.format)));
            }
            data += ib.bytes.len();
        }
        for x in &d.textures {
            if x.surface == s {
                return Err(Error(format!("a draw into surface {s:?} sampling it")));
            }
            st.target(x.surface)?;
        }
        let code = st.program(program.id, vs, fs)?;
        st.load(s, &d.target.load)?;
        let Some(pixels) = Rdna2Draw::pixels(d) else { return Ok(()) };
        if d.count == 0 || d.instances == 0 {
            return Ok(());
        }
        // what the batch drew sampled, or what it sampled drawn: the batch so far first (its end flushes the caches)
        if d.textures.iter().any(|x| st.written.contains(&x.surface.0)) || st.sampled.contains(&s.0) {
            st.submit()?;
        }
        data += RING_ALIGN * (d.buffers.len() + d.uniforms.len() + 1);
        st.room(data)?;
        st.record(vs, fs, code, d, s, pixels)
    }
}

impl Drop for AmdgpuDevice<'_> {
    fn drop(&mut self) {
        if let Some(st) = self.state.get_mut() {
            let _ = st.settle();
        }
    }
}
