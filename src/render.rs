use std::cell::{RefCell, RefMut};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::rc::Rc;
use std::time::Duration;

use mlirformat::Error;

use crate::{
    Bo, Caching, ColorFormat, ColorTarget, Compiled, Draw, Engine, Fence, Gpu, Host, MemType, Native, Queue, Syncobj, TargetLoad, Template, Tiling, Vm, Xe, Xe3Draw, Xe3DrawImage, XeBo, XeQueue,
    XeVm,
};

/// How long a submission may take before the GPU is called hung
const TIMEOUT: Duration = Duration::from_secs(5);
/// Draw shapes kept resident
const RESIDENT: usize = 64;
/// Where shape slot i's image is laid out (`SLOT_BASE + i * SLOT_SPAN`), all below 4 GiB: its states are pointed at
/// with 32 bits
const SLOT_BASE: u64 = 1 << 24;
const SLOT_SPAN: u64 = 1 << 24;
/// What the device writes between draws below 4 GiB (scissor rectangles, viewports), and its size
const DYNAMIC_ADDR: u64 = 0xf000_0000;
const DYNAMIC_SIZE: usize = 1 << 20;
/// Where the first-level batches go, and their ring's size
const CALLS_ADDR: u64 = 1 << 32;
const CALLS_SIZE: u64 = 1 << 20;
/// Mapped past a ring's end: the command streamer prefetches beyond the batch end it reads (a GPU fault at batch +
/// 0x1000 otherwise), the VF beyond a buffer's
const GUARD: u64 = 1 << 16;
/// MI_BATCH_BUFFER_START of a second-level batch in the PPGTT (its address in the next two dwords), MI_BATCH_BUFFER_END
const MI_BATCH_BUFFER_START_2ND: u32 = 49 << 23 | 1 << 22 | 1 << 8 | 1;
const MI_BATCH_BUFFER_END: u32 = 10 << 23;
const CALL_BYTES: usize = 12;
/// A first-level batch's start: the CS reads it from a 64-byte boundary
const BATCH_ALIGN: usize = 64;
/// The uniforms' ring, vertex buffer i's (`VERTEX_ADDR + i * VERTEX_SPAN`), a ring's least size and where each
/// draw's bytes start in it
const UNIFORM_ADDR: u64 = 1 << 35;
const VERTEX_ADDR: u64 = 1 << 36;
const VERTEX_SPAN: u64 = 1 << 32;
const RING_MIN: usize = 1 << 20;
const RING_ALIGN: usize = 64;
/// Where surface slot i is mapped (`SURFACE_ADDR + i * SURFACE_SPAN`): a surface takes at most a span
const SURFACE_ADDR: u64 = 1 << 40;
const SURFACE_SPAN: u64 = 1 << 30;
/// States the device writes: on 64 bytes
const STATE_ALIGN: usize = 64;

fn io(e: std::io::Error) -> Error {
    Error(e.to_string())
}

fn wait(f: &Syncobj<'_>, what: &str) -> Result<(), Error> {
    if f.wait(TIMEOUT).map_err(io)? { Ok(()) } else { Err(Error(format!("{what} did not finish: the GPU hung"))) }
}

/// `bo` mapped at `addr` in `vm` from its byte `offset`, the mapping in place before returning
fn bind<'x>(xe: &'x Xe, vm: &XeVm<'x>, bo: &XeBo<'x>, offset: u64, addr: u64, range: u64, mem: MemType) -> Result<(), Error> {
    let done = xe.fence(false).map_err(io)?;
    vm.bind(bo, offset, addr, range, mem, &[&done]).map_err(io)?;
    wait(&done, &format!("binding {addr:#x}"))
}

/// `words` written into `bo` from byte `at`
fn put(bo: &mut XeBo<'_>, at: usize, words: &[u32]) -> Result<(), Error> {
    let memory = bo.bytes().map_err(io)?;
    let room = memory.get_mut(at..at + 4 * words.len()).ok_or_else(|| Error(format!("{} dwords past a memory's end at {at}", words.len())))?;
    for (d, w) in room.chunks_exact_mut(4).zip(words) {
        d.copy_from_slice(&w.to_le_bytes());
    }
    Ok(())
}

/// A color target the device keeps: its pixels stay on the GPU between draws
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Surface(usize);

/// A command a draw writes: a template and, for each of its holes, which of the values given fills it
struct Command {
    template: Template,
    slots: Vec<usize>,
}

impl Command {
    /// `t`, its holes named as `names` says (each name's value's index)
    fn new(t: &Template, names: &[String]) -> Result<Command, Error> {
        let slot = |h: &str| names.iter().position(|n| n == h).ok_or_else(|| Error(format!("a hole @{h} the device does not fill")));
        Ok(Command { template: t.clone(), slots: t.holes.iter().map(|h| slot(&h.name)).collect::<Result<_, _>>()? })
    }

    fn bytes(&self) -> usize {
        4 * self.template.dw.len()
    }

    /// Its dwords, the holes filled from `values`, after `out`'s
    fn emit(&self, out: &mut Vec<u32>, values: &[u64]) {
        self.template.emit_indexed(out, |i| values[self.slots[i]]);
    }
}

fn names(n: &[&str]) -> Vec<String> {
    n.iter().map(|s| s.to_string()).collect()
}

/// Bytes the GPU reads, written into a ring from its start once what read them before finished, each draw's after
/// the last's (the same bytes again: where they are)
struct Ring<'x> {
    bo: Option<XeBo<'x>>,
    addr: u64,
    /// the bytes the ring holds (its memory past them mapped too: `GUARD`)
    size: usize,
    at: usize,
    /// where the last bytes written start, and them
    last: Option<(usize, Vec<u8>)>,
}

impl<'x> Ring<'x> {
    fn new(addr: u64) -> Ring<'x> {
        Ring { bo: None, addr, size: 0, at: 0, last: None }
    }

    fn held(&self, bytes: &[u8]) -> Option<usize> {
        self.last.as_ref().filter(|(_, b)| b == bytes).map(|(at, _)| *at)
    }

    /// Where the ring is after `lens` more bytes from `at` (each on `RING_ALIGN`)
    fn after(at: usize, lens: impl Iterator<Item = usize>) -> usize {
        lens.fold(at, |at, n| if n == 0 { at } else { at.next_multiple_of(RING_ALIGN) + n })
    }

    /// Whether `bytes` are in place already, or fit after the last
    fn fits(&self, bytes: &[u8]) -> bool {
        self.held(bytes).is_some() || Ring::after(self.at, [bytes.len()].into_iter()) <= self.size
    }

    /// The ring from its start, with room for `len` bytes (nothing reading it)
    fn restart(&mut self, xe: &'x Xe, vm: &XeVm<'x>, len: usize) -> Result<(), Error> {
        (self.at, self.last) = (0, None);
        if len <= self.size {
            return Ok(());
        }
        let size = len.next_power_of_two().max(RING_MIN);
        let bytes = size as u64 + GUARD;
        if let Some(old) = self.bo.take() {
            vm.unbind(self.addr, old.size(), &[]).map_err(io)?;
        }
        let bo = xe.bo((bytes, Caching::WriteCombined).into()).map_err(io)?;
        bind(xe, vm, &bo, 0, self.addr, bytes, MemType::Streaming)?;
        (self.bo, self.size) = (Some(bo), size);
        Ok(())
    }

    /// `bytes` in the ring (they fit): their address
    fn put(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        if bytes.is_empty() {
            return Ok(self.addr);
        }
        if let Some(at) = self.held(bytes) {
            return Ok(self.addr + at as u64);
        }
        let at = self.at.next_multiple_of(RING_ALIGN);
        let bo = self.bo.as_mut().ok_or_else(|| Error("a ring with no memory".into()))?;
        bo.bytes().map_err(io)?[at..at + bytes.len()].copy_from_slice(bytes);
        self.at = at + bytes.len();
        match &mut self.last {
            Some((held, b)) => {
                *held = at;
                b.clear();
                b.extend_from_slice(bytes);
            }
            None => self.last = Some((at, bytes.to_vec())),
        }
        Ok(self.addr + at as u64)
    }
}

/// What a draw writes, from its image (`Xe3DrawCommands`)
struct Commands {
    primitive: Command,
    vertex_buffers: Command,
    scissor_rect: Command,
    scissor_pointer: Command,
    target_surface: Command,
    uniform_surface: Command,
    binding_table: Command,
    table_vs: Command,
    table_ps: Command,
    viewport_cc: Command,
    viewport_sf_clip: Command,
}

impl Commands {
    /// The bytes a draw's commands take at most
    fn bytes(&self) -> usize {
        [&self.vertex_buffers, &self.scissor_pointer, &self.table_vs, &self.table_ps, &self.viewport_cc, &self.viewport_sf_clip, &self.primitive].iter().map(|c| c.bytes()).sum()
    }
}

/// One draw shape resident: its image's memories bound at its slot, its context set up (`INIT`)
struct Shape<'x> {
    key: u64,
    slot: usize,
    /// when it was last drawn (the device's clock)
    used: u64,
    /// its memories: each one's buffer object, address and mapped bytes
    bos: Vec<(XeBo<'x>, u64, u64)>,
    /// its stream's second-level batches: the draw's state, the flush
    state: u64,
    flush: u64,
    commands: Commands,
    /// the binding table pool: its buffer object (in `bos`), address, where the device writes in it, up to where
    pool: usize,
    pool_addr: u64,
    pool_free: usize,
    pool_at: usize,
    pool_end: usize,
    null: u64,
    /// the uniform blocks' bindings, in the draw's order
    bindings: Vec<u32>,
}

impl<'x> Shape<'x> {
    fn new(xe: &'x Xe, vm: &XeVm<'x>, queue: &XeQueue<'x>, key: u64, slot: usize, parts: Xe3DrawImage, d: &Draw<'_>) -> Result<Shape<'x>, Error> {
        let end = SLOT_BASE + (slot as u64 + 1) * SLOT_SPAN;
        let c = &parts.commands;
        let vb: Vec<String> = (0..d.buffers.len()).flat_map(|i| [Xe3Draw::vertex_buffer(i), format!("{}_size", Xe3Draw::vertex_buffer(i))]).collect();
        let bindings: Vec<u32> = d.uniforms.iter().map(|(b, _)| *b).collect();
        let table: Vec<String> = ["rt", "null"].into_iter().map(String::from).chain(bindings.iter().map(|b| Xe3Draw::uniform_buffer(*b))).collect();
        let commands = Commands {
            primitive: Command::new(&c.primitive, &names(&["count", "first", "instances"]))?,
            vertex_buffers: Command::new(&c.vertex_buffers, &vb)?,
            scissor_rect: Command::new(&c.scissor_rect, &names(&["x0", "y0", "x1", "y1"]))?,
            scissor_pointer: Command::new(&c.scissor_pointer, &names(&["scissor"]))?,
            target_surface: Command::new(&c.target_surface, &names(&["target", "pitch", "width", "height"]))?,
            uniform_surface: Command::new(&c.uniform_surface, &names(&["ubo", "ubo_w", "ubo_h", "ubo_d"]))?,
            binding_table: Command::new(&c.binding_table, &table)?,
            table_vs: Command::new(&c.table_pointers_vs, &names(&["bt"]))?,
            table_ps: Command::new(&c.table_pointers_ps, &names(&["bt"]))?,
            viewport_cc: Command::new(&c.viewport_cc, &names(&["cc"]))?,
            viewport_sf_clip: Command::new(&c.viewport_sf_clip, &names(&["sf_clip"]))?,
        };
        let image = &parts.image;
        let mut bos = Vec::new();
        let mut pool = None;
        for s in &image.segments {
            let size = (s.bytes.len() as u64).next_multiple_of(xe.info().min_alignment.max(4096));
            if s.addr + size > end {
                return Err(Error(format!("a draw's image of {:#x} bytes does not fit its slot", s.addr + size - (end - SLOT_SPAN))));
            }
            let mem = MemType::from(s.mem);
            let caching = if mem == MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
            let mut bo = xe.bo((size, caching).into()).map_err(io)?;
            bo.bytes().map_err(io)?[..s.bytes.len()].copy_from_slice(&s.bytes);
            bind(xe, vm, &bo, 0, s.addr, size, mem)?;
            if s.addr == parts.pool {
                pool = Some((bos.len(), s.bytes.len()));
            }
            bos.push((bo, s.addr, size));
        }
        let (pool, pool_end) = pool.ok_or_else(|| Error("no binding table pool in the image".into()))?;
        let init = image.stream_named(Xe3Draw::INIT).ok_or_else(|| Error(format!("no stream {} in the image", Xe3Draw::INIT)))?;
        Native::fence();
        let done = xe.fence(false).map_err(io)?;
        queue.submit(&init, &[], &[&done]).map_err(io)?;
        wait(&done, "setting the context up")?;
        let pool_free = parts.pool_free as usize;
        Ok(Shape {
            key,
            slot,
            used: 0,
            bos,
            state: parts.state,
            flush: parts.flush,
            commands,
            pool,
            pool_addr: parts.pool,
            pool_free,
            pool_at: pool_free,
            pool_end,
            null: parts.null,
            bindings,
        })
    }

    /// The bytes a draw's binding table and its surface states take in the pool at most
    fn table_bytes(&self) -> usize {
        STATE_ALIGN * (1 + self.bindings.len()) + self.commands.binding_table.bytes().next_multiple_of(STATE_ALIGN)
    }

    /// Its memories unmapped (nothing of it running)
    fn release(self, vm: &XeVm<'x>) -> Result<(), Error> {
        for (_, addr, size) in &self.bos {
            vm.unbind(*addr, *size, &[]).map_err(io)?;
        }
        Ok(())
    }
}

/// What a draw's key holds: everything of it its image depends on (a hash: one per draw, no allocation)
fn key(program: u64, d: &Draw<'_>) -> u64 {
    let mut h = DefaultHasher::new();
    program.hash(&mut h);
    for b in &d.buffers {
        b.stride.hash(&mut h);
    }
    for e in &d.elements {
        (e.buffer, discriminant(&e.format), e.offset).hash(&mut h);
    }
    for (binding, _) in &d.uniforms {
        binding.hash(&mut h);
    }
    discriminant(&d.primitive).hash(&mut h);
    (discriminant(&d.cull), d.front_ccw, d.color_mask, d.blend.is_some()).hash(&mut h);
    if let Some(b) = &d.blend {
        (b.src.map(|f| f as u8), b.dst.map(|f| f as u8), b.equation.map(|e| e as u8), b.color.map(f32::to_bits)).hash(&mut h);
    }
    (discriminant(&d.target.format), d.target.tiling).hash(&mut h);
    h.finish()
}

/// A surface's memory and what last wrote it: the device's own, or a dma-buf imported
struct Target<'x> {
    bo: XeBo<'x>,
    slot: usize,
    addr: u64,
    /// where its pixels start in `bo`, the bytes mapped from there
    offset: usize,
    range: u64,
    width: u32,
    height: u32,
    format: ColorFormat,
    tiling: Tiling,
    /// a row's bytes in memory
    pitch: u32,
    fence: Option<Rc<Syncobj<'x>>>,
}

impl<'x> Target<'x> {
    fn row(&self) -> u32 {
        self.width * self.format.bytes()
    }

    fn len(&self) -> usize {
        self.pitch as usize * self.tiling.rows(self.height) as usize
    }

    /// Whether a draw describing its target as `t` draws into it
    fn holds(&self, t: &ColorTarget<'_>) -> bool {
        (t.width, t.height, t.format, t.tiling) == (self.width, self.height, self.format, self.tiling)
    }

    /// Its memory as laid out (nothing writing it)
    fn memory(&mut self) -> Result<&mut [u8], Error> {
        let (at, n) = (self.offset, self.len());
        self.bo.bytes().map_err(io)?.get_mut(at..at + n).ok_or_else(|| Error("a surface past its memory's end".into()))
    }

    /// Whether the CPU reads and writes it as its tiling lays rows out
    fn natural(&self) -> Result<(), Error> {
        match self.tiling == Tiling::Linear || self.pitch == self.tiling.pitch(self.row()) {
            true => Ok(()),
            false => Err(Error(format!("a {:?} surface with rows {} bytes apart: not read or written by the CPU", self.tiling, self.pitch))),
        }
    }

    /// Its pixels in rows (nothing writing it)
    fn pixels(&mut self) -> Result<Vec<u8>, Error> {
        self.natural()?;
        let (row, height, pitch, tiling) = (self.row() as usize, self.height, self.pitch as usize, self.tiling);
        let mut out = vec![0; row * height as usize];
        let memory = self.memory()?;
        match tiling {
            Tiling::Linear => out.chunks_exact_mut(row).zip(memory.chunks(pitch)).for_each(|(o, m)| o.copy_from_slice(&m[..row])),
            _ => tiling.untile(memory, row as u32, height, &mut out),
        }
        Ok(out)
    }

    /// `pixels` (rows) written into it (nothing using it)
    fn write(&mut self, pixels: &[u8]) -> Result<(), Error> {
        self.natural()?;
        let (row, height, pitch, tiling) = (self.row() as usize, self.height, self.pitch as usize, self.tiling);
        let pixels = pixels.get(..row * height as usize).ok_or_else(|| Error("fewer pixels than the target holds".into()))?;
        let memory = self.memory()?;
        match tiling {
            Tiling::Linear => memory.chunks_mut(pitch).zip(pixels.chunks_exact(row)).for_each(|(m, p)| m[..row].copy_from_slice(p)),
            _ => tiling.tile(pixels, row as u32, height, memory),
        }
        Ok(())
    }
}

/// What the batch being recorded has set since its last shape's state: its vertex buffers, binding table (the
/// surface and the uniforms' addresses), viewport and scissor
struct Current {
    shape: u64,
    flush: u64,
    vertices: Vec<u64>,
    table: Option<(usize, Vec<u64>)>,
    viewport: Option<u64>,
    scissor: Option<[u32; 4]>,
}

/// The device's address space and what is in it
struct State<'x> {
    xe: &'x Xe,
    vm: XeVm<'x>,
    queue: XeQueue<'x>,
    calls: XeBo<'x>,
    /// where the next dword goes in `calls`, and where the batch being recorded starts
    at: usize,
    start: Option<usize>,
    dynamic: XeBo<'x>,
    dynamic_at: usize,
    vertices: Vec<Ring<'x>>,
    uniforms: Ring<'x>,
    /// each binding's uniform bytes last written into the ring, and where
    held: Vec<(u32, u64, Vec<u8>)>,
    shapes: Vec<Shape<'x>>,
    clock: u64,
    surfaces: Vec<Option<Target<'x>>>,
    /// what the next submission waits for, the last submission, the surfaces the batch being recorded draws into
    waits: Vec<Syncobj<'x>>,
    last: Option<Rc<Syncobj<'x>>>,
    written: Vec<usize>,
    current: Option<Current>,
    /// the viewports written into the dynamic memory since it started over: each one's key, CC and SF_CLIP state
    viewports: Vec<(u64, u64, u64)>,
    words: Vec<u32>,
    lowered: usize,
}

impl<'x> State<'x> {
    fn new(xe: &'x Xe) -> Result<State<'x>, Error> {
        let vm = xe.vm().map_err(io)?;
        let queue = xe.queue(&vm, Engine::Render).map_err(io)?;
        let calls = xe.bo((CALLS_SIZE + GUARD, Caching::WriteCombined).into()).map_err(io)?;
        bind(xe, &vm, &calls, 0, CALLS_ADDR, CALLS_SIZE + GUARD, MemType::Streaming)?;
        let dynamic = xe.bo((DYNAMIC_SIZE as u64, Caching::WriteCombined).into()).map_err(io)?;
        bind(xe, &vm, &dynamic, 0, DYNAMIC_ADDR, DYNAMIC_SIZE as u64, MemType::Streaming)?;
        Ok(State {
            xe,
            vm,
            queue,
            calls,
            at: 0,
            start: None,
            dynamic,
            dynamic_at: 0,
            vertices: Vec::new(),
            uniforms: Ring::new(UNIFORM_ADDR),
            held: Vec::new(),
            shapes: Vec::new(),
            clock: 0,
            surfaces: Vec::new(),
            waits: Vec::new(),
            last: None,
            written: Vec::new(),
            current: None,
            viewports: Vec::new(),
            words: Vec::new(),
            lowered: 0,
        })
    }

    fn target(&mut self, s: Surface) -> Result<&mut Target<'x>, Error> {
        self.surfaces.get_mut(s.0).and_then(Option::as_mut).ok_or_else(|| Error(format!("no surface {s:?}")))
    }

    /// A surface slot no surface takes
    fn surface_slot(&self) -> usize {
        let taken: Vec<usize> = self.surfaces.iter().flatten().map(|t| t.slot).collect();
        (0..).find(|i| !taken.contains(i)).unwrap_or(0)
    }

    /// `bo` a surface for targets like `t`, its pixels from byte `offset`, rows `pitch` bytes apart, mapped as the
    /// first of `mems` the GPU takes
    fn adopt(&mut self, t: &ColorTarget<'_>, bo: XeBo<'x>, offset: u64, pitch: u32, mems: &[MemType]) -> Result<Surface, Error> {
        let align = self.xe.info().min_alignment.max(4096);
        if !offset.is_multiple_of(align) {
            return Err(Error(format!("a surface at byte {offset} of its memory: not on {align}")));
        }
        let rows = t.tiling.rows(t.height) as u64;
        let need = u64::from(pitch) * rows;
        if pitch < t.row() || offset + need > bo.size() {
            return Err(Error(format!("a {}x{} surface, rows of {pitch} bytes from {offset}, in {} bytes", t.width, t.height, bo.size())));
        }
        if t.tiling == Tiling::Tile4 && !(pitch.is_multiple_of(Tiling::TILE4_WIDTH)) || !pitch.is_multiple_of(4) {
            return Err(Error(format!("rows of {pitch} bytes for {:?}", t.tiling)));
        }
        let range = need.next_multiple_of(align).min(bo.size() - offset);
        if range > SURFACE_SPAN {
            return Err(Error(format!("a surface of {range} bytes: more than {SURFACE_SPAN}")));
        }
        let slot = self.surface_slot();
        let addr = SURFACE_ADDR + slot as u64 * SURFACE_SPAN;
        let mut tried = Vec::new();
        for mem in mems {
            match bind(self.xe, &self.vm, &bo, offset, addr, range, *mem) {
                Ok(()) => break,
                Err(e) => tried.push(format!("as {mem:?}: {}", e.0)),
            }
        }
        if tried.len() == mems.len() {
            return Err(Error(format!("a surface's memory not mapped: {}", tried.join("; "))));
        }
        let target = Target { bo, slot, addr, offset: offset as usize, range, width: t.width, height: t.height, format: t.format, tiling: t.tiling, pitch, fence: None };
        self.surfaces.push(Some(target));
        Ok(Surface(self.surfaces.len() - 1))
    }

    /// The batch recorded ended (the flush, MI_BATCH_BUFFER_END) and submitted, after what it waits for
    fn submit(&mut self) -> Result<(), Error> {
        let Some(start) = self.start.take() else { return Ok(()) };
        let flush = self.current.take().map_or(0, |c| c.flush);
        self.words.extend([MI_BATCH_BUFFER_START_2ND, flush as u32, (flush >> 32) as u32, MI_BATCH_BUFFER_END]);
        self.write()?;
        self.at = self.at.next_multiple_of(BATCH_ALIGN);
        Native::fence();
        let done = Rc::new(self.xe.fence(false).map_err(io)?);
        let waits: Vec<&Syncobj<'x>> = self.waits.iter().collect();
        self.queue.submit(&(CALLS_ADDR + start as u64), &waits, &[&done]).map_err(io)?;
        self.waits.clear();
        for s in self.written.drain(..) {
            if let Some(Some(t)) = self.surfaces.get_mut(s) {
                t.fence = Some(done.clone());
            }
        }
        self.last = Some(done);
        Ok(())
    }

    /// Everything recorded submitted and done
    fn settle(&mut self) -> Result<(), Error> {
        self.submit()?;
        if let Some(f) = self.last.take() {
            wait(&f, "a draw")?;
        }
        Ok(())
    }

    /// The draws into surface `s` submitted and done
    fn idle(&mut self, s: Surface) -> Result<(), Error> {
        if self.written.contains(&s.0) {
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
        let clears = match load {
            TargetLoad::Keep([]) => return Ok(()),
            TargetLoad::Keep(c) | TargetLoad::Fill(_, c) | TargetLoad::Pixels(_, c) => c,
        };
        self.idle(s)?;
        let t = self.target(s)?;
        match load {
            TargetLoad::Fill(texel, _) => {
                let memory = t.memory()?;
                match texel.iter().all(|b| *b == texel[0]) {
                    true => memory.fill(texel[0]),
                    false => memory.chunks_exact_mut(texel.len()).for_each(|p| p.copy_from_slice(texel)),
                }
            }
            TargetLoad::Pixels(p, _) => t.write(p)?,
            TargetLoad::Keep(_) => {}
        }
        let (pitch, tiling) = (t.pitch, t.tiling);
        let memory = t.memory()?;
        clears.iter().for_each(|c| c.apply_pitched(memory, pitch, tiling));
        Ok(())
    }

    /// The resident shape of `program` and `d`: lowered and made resident when it is not (the least recently drawn
    /// one put away when they are `RESIDENT`)
    fn shape(&mut self, program: &Compiled<'_>, d: &Draw<'_>) -> Result<usize, Error> {
        let k = key(program.id, d);
        self.clock += 1;
        if let Some(i) = self.shapes.iter().position(|s| s.key == k) {
            self.shapes[i].used = self.clock;
            return Ok(i);
        }
        if self.shapes.len() == RESIDENT {
            self.settle()?;
            let oldest = (0..self.shapes.len()).min_by_key(|&i| self.shapes[i].used).unwrap_or(0);
            self.shapes.swap_remove(oldest).release(&self.vm)?;
        }
        let slot = (0..RESIDENT).find(|i| !self.shapes.iter().any(|s| s.slot == *i)).unwrap_or(0);
        let mut body = program.body.clone();
        let image = Xe3Draw::image(&mut body, d, SLOT_BASE + slot as u64 * SLOT_SPAN)?;
        self.lowered += 1;
        let mut shape = Shape::new(self.xe, &self.vm, &self.queue, k, slot, image, d)?;
        shape.used = self.clock;
        self.shapes.push(shape);
        Ok(self.shapes.len() - 1)
    }

    /// Whether draw `d` of shape `i` fits in what the batch being recorded has room for
    fn fits(&self, i: usize, d: &Draw<'_>) -> bool {
        let sh = &self.shapes[i];
        let calls = self.at + 2 * CALL_BYTES + sh.commands.bytes() + 4 <= CALLS_SIZE as usize;
        let dynamic = self.dynamic_at.next_multiple_of(STATE_ALIGN) + 3 * STATE_ALIGN <= DYNAMIC_SIZE;
        let pool = sh.pool_at.next_multiple_of(STATE_ALIGN) + sh.table_bytes() <= sh.pool_end;
        let vertices = d.buffers.iter().enumerate().all(|(i, b)| self.vertices.get(i).is_some_and(|r| r.fits(b.bytes)) || b.bytes.is_empty());
        let fresh = d.uniforms.iter().filter(|(b, bytes)| !self.held.iter().any(|(h, _, held)| h == b && held == bytes)).map(|(_, bytes)| bytes.len());
        let uniforms = Ring::after(self.uniforms.at, fresh) <= self.uniforms.size;
        calls && dynamic && pool && vertices && uniforms
    }

    /// Every ring from its start, with room for draw `d` (nothing running)
    fn restart(&mut self, d: &Draw<'_>) -> Result<(), Error> {
        (self.at, self.dynamic_at) = (0, 0);
        self.held.clear();
        self.viewports.clear();
        for s in &mut self.shapes {
            s.pool_at = s.pool_free;
        }
        while self.vertices.len() < d.buffers.len() {
            self.vertices.push(Ring::new(VERTEX_ADDR + self.vertices.len() as u64 * VERTEX_SPAN));
        }
        for (i, r) in self.vertices.iter_mut().enumerate() {
            r.restart(self.xe, &self.vm, d.buffers.get(i).map_or(0, |b| b.bytes.len()))?;
        }
        let uniforms = Ring::after(0, d.uniforms.iter().map(|(_, b)| b.len()));
        self.uniforms.restart(self.xe, &self.vm, uniforms)
    }

    /// `words` written at the batch's end
    fn write(&mut self) -> Result<(), Error> {
        put(&mut self.calls, self.at, &self.words)?;
        self.at += 4 * self.words.len();
        self.words.clear();
        Ok(())
    }

    /// `words` written into the dynamic memory on `align`: their address
    fn dynamic(&mut self, words: &[u32]) -> Result<u64, Error> {
        let at = self.dynamic_at.next_multiple_of(STATE_ALIGN);
        put(&mut self.dynamic, at, words)?;
        self.dynamic_at = at + 4 * words.len();
        Ok(DYNAMIC_ADDR + at as u64)
    }

    /// Uniform block `binding`'s bytes in the ring: their address
    fn uniform(&mut self, binding: u32, bytes: &[u8]) -> Result<u64, Error> {
        if let Some((_, addr, _)) = self.held.iter().find(|(b, _, held)| *b == binding && held == bytes) {
            return Ok(*addr);
        }
        let addr = self.uniforms.put(bytes)?;
        self.held.retain(|(b, _, _)| *b != binding);
        self.held.push((binding, addr, bytes.to_vec()));
        Ok(addr)
    }

    /// Draw `d` of shape `i` into `s` recorded (it fits): its shape's state when the batch has another's, then its
    /// viewport, binding table, vertex buffers and scissor where the batch does not have them, and its primitive
    fn record(&mut self, i: usize, d: &Draw<'_>, s: Surface) -> Result<(), Error> {
        if self.start.is_none() {
            self.start = Some(self.at);
        }
        let (key, state, flush) = (self.shapes[i].key, self.shapes[i].state, self.shapes[i].flush);
        if self.current.as_ref().is_none_or(|c| c.shape != key) {
            self.words.extend([MI_BATCH_BUFFER_START_2ND, state as u32, (state >> 32) as u32]);
            self.write()?;
            self.current = Some(Current { shape: key, flush, vertices: Vec::new(), table: None, viewport: None, scissor: None });
        }
        let mut h = DefaultHasher::new();
        (d.viewport.map(f32::to_bits), d.depth_range.map(f32::to_bits), d.target.width, d.target.height).hash(&mut h);
        let viewport = h.finish();
        if self.current.as_ref().is_some_and(|c| c.viewport != Some(viewport)) {
            let (cc, sf_clip) = match self.viewports.iter().find(|v| v.0 == viewport) {
                Some(&(_, cc, sf_clip)) => (cc, sf_clip),
                None => {
                    let [cc, sf_clip] = Xe3Draw::viewports(d)?;
                    let (cc, sf_clip) = (self.dynamic(&cc)?, self.dynamic(&sf_clip)?);
                    self.viewports.push((viewport, cc, sf_clip));
                    (cc, sf_clip)
                }
            };
            let c = &self.shapes[i].commands;
            c.viewport_cc.emit(&mut self.words, &[cc]);
            c.viewport_sf_clip.emit(&mut self.words, &[sf_clip]);
            self.write()?;
            self.current.as_mut().unwrap().viewport = Some(viewport);
        }
        let mut uniforms = Vec::with_capacity(d.uniforms.len());
        for (binding, bytes) in &d.uniforms {
            uniforms.push(self.uniform(*binding, bytes)?);
        }
        if self.current.as_ref().is_some_and(|c| c.table.as_ref().is_none_or(|(t, u)| *t != s.0 || *u != uniforms)) {
            let lens: Vec<usize> = d.uniforms.iter().map(|(_, b)| b.len()).collect();
            self.table(i, s, &uniforms, &lens)?;
            self.current.as_mut().unwrap().table = Some((s.0, uniforms));
        }
        let mut values = Vec::with_capacity(2 * d.buffers.len());
        for (r, b) in self.vertices.iter_mut().zip(&d.buffers) {
            values.extend([r.put(b.bytes)?, b.bytes.len() as u64]);
        }
        if !values.is_empty() && self.current.as_ref().is_some_and(|c| c.vertices != values) {
            self.shapes[i].commands.vertex_buffers.emit(&mut self.words, &values);
            self.write()?;
            self.current.as_mut().unwrap().vertices = values;
        }
        let rect = Xe3Draw::scissor(d);
        if self.current.as_ref().is_some_and(|c| c.scissor != Some(rect)) {
            let mut words = Vec::new();
            self.shapes[i].commands.scissor_rect.emit(&mut words, &rect.map(u64::from));
            let at = self.dynamic(&words)?;
            self.shapes[i].commands.scissor_pointer.emit(&mut self.words, &[at]);
            self.write()?;
            self.current.as_mut().unwrap().scissor = Some(rect);
        }
        self.shapes[i].commands.primitive.emit(&mut self.words, &[d.count, d.first, d.instances].map(u64::from));
        self.write()?;
        if !self.written.contains(&s.0) {
            self.written.push(s.0);
        }
        Ok(())
    }

    /// A binding table of `s` and `uniforms` (the blocks' addresses, `lens` their bytes) written into shape `i`'s
    /// pool, and the pointers to it recorded
    fn table(&mut self, i: usize, s: Surface, uniforms: &[u64], lens: &[usize]) -> Result<(), Error> {
        let t = self.target(s)?;
        let rt = [t.addr, u64::from(t.pitch) - 1, u64::from(t.width) - 1, u64::from(t.height) - 1];
        let sh = &mut self.shapes[i];
        let (pool, pool_addr) = (sh.pool, sh.pool_addr);
        let mut at = sh.pool_at.next_multiple_of(STATE_ALIGN);
        let mut words = Vec::new();
        sh.commands.target_surface.emit(&mut words, &rt);
        put(&mut sh.bos[pool].0, at, &words)?;
        let mut table = vec![pool_addr + at as u64, sh.null];
        at += STATE_ALIGN;
        for (addr, len) in uniforms.iter().zip(lens) {
            words.clear();
            let [w, h, dd] = Xe3Draw::buffer_size(*len as u32);
            sh.commands.uniform_surface.emit(&mut words, &[*addr, u64::from(w), u64::from(h), u64::from(dd)]);
            put(&mut sh.bos[pool].0, at, &words)?;
            table.push(pool_addr + at as u64);
            at += STATE_ALIGN;
        }
        words.clear();
        sh.commands.binding_table.emit(&mut words, &table);
        put(&mut sh.bos[pool].0, at, &words)?;
        let bt = at as u64;
        sh.pool_at = at + 4 * words.len();
        sh.commands.table_vs.emit(&mut self.words, &[bt]);
        sh.commands.table_ps.emit(&mut self.words, &[bt]);
        self.write()
    }
}

/// The GPU detour submits to, and what it keeps there in one address space: its surfaces (its own or dma-bufs
/// imported), each program and draw shape's image (its context set up once), rings of the draws' vertices and
/// uniforms. A draw records a call of its shape's state when the batch has another shape's, then the commands its
/// target, uniforms, vertices, viewport, scissor and counts need; nothing waits for the GPU. The batch goes to the
/// GPU, one submission, when a surface it draws into is read, loaded with other pixels, finished, flushed or
/// fenced, or when a ring is full
pub struct Device<'x> {
    pub xe: &'x Xe,
    state: RefCell<Option<State<'x>>>,
}

impl<'x> Device<'x> {
    pub fn new(xe: &'x Xe) -> Device<'x> {
        Device { xe, state: RefCell::new(None) }
    }

    /// Its state, set up on first use
    fn state(&self) -> Result<RefMut<'_, State<'x>>, Error> {
        let mut st = self.state.borrow_mut();
        if st.is_none() {
            *st = Some(State::new(self.xe)?);
        }
        Ok(RefMut::map(st, |s| s.as_mut().unwrap()))
    }

    /// A new surface for target `t` (its size, format and layout), zero
    pub fn surface(&self, t: &ColorTarget<'_>) -> Result<Surface, Error> {
        let mut st = self.state()?;
        let size = (t.size() as u64).next_multiple_of(self.xe.info().min_alignment.max(4096));
        let bo = self.xe.bo((size, Caching::WriteCombined).into()).map_err(io)?;
        st.adopt(t, bo, 0, t.pitch(), &[MemType::Streaming])
    }

    /// A surface drawn straight into dma-buf `fd` (a GBM buffer, a udmabuf...): targets like `t` (its size, format:
    /// `Rgba8Unorm` is DRM's ABGR8888; tiling: `Linear`, or `Tile4` for I915_FORMAT_MOD_4_TILED), from byte `offset`
    /// (on a page), rows `stride` bytes apart. What it held stays until a draw writes it; `fence` says when the
    /// draws into it are done
    pub fn import(&self, t: &ColorTarget<'_>, fd: BorrowedFd<'_>, offset: u64, stride: u32) -> Result<Surface, Error> {
        let mut st = self.state()?;
        let bo = self.xe.import(fd).map_err(|e| Error(format!("a dma-buf not imported: {e}")))?;
        st.adopt(t, bo, offset, stride, &[MemType::Streaming, MemType::Coherent])
    }

    /// Surface `s` gone, once the draws into it are done; using it again is an error
    pub fn free(&self, s: Surface) -> Result<(), Error> {
        let mut st = self.state()?;
        st.idle(s)?;
        let t = st.surfaces.get_mut(s.0).and_then(Option::take).ok_or_else(|| Error(format!("no surface {s:?}")))?;
        for c in st.current.iter_mut() {
            c.table = c.table.take().filter(|(t, _)| *t != s.0);
        }
        st.vm.unbind(t.addr, t.range, &[]).map_err(io)
    }

    /// How many draw shapes were lowered into images so far (each costs a compile; a draw of a resident one does not)
    pub fn lowered(&self) -> usize {
        self.state.borrow().as_ref().map_or(0, |s| s.lowered)
    }

    /// The first `len` bytes of surface `s`'s pixels (rows), once the draws into it are done
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

    /// Surface `s`'s memory as it is laid out, once the draws into it are done
    pub fn raw(&self, s: Surface) -> Result<Vec<u8>, Error> {
        let mut st = self.state()?;
        st.idle(s)?;
        Ok(st.target(s)?.memory()?.to_vec())
    }

    /// The draws recorded (into `s` and every other surface) submitted, not waited for
    pub fn flush(&self, s: Surface) -> Result<(), Error> {
        let mut st = self.state()?;
        st.target(s)?;
        st.submit()
    }

    /// The draws into surface `s` submitted and done (glFinish)
    pub fn finish(&self, s: Surface) -> Result<(), Error> {
        self.state()?.idle(s)
    }

    /// The draws recorded submitted (when some draw into `s`): a sync_file signalled once the draws into `s` are
    /// done, None when nothing ever drew into it
    pub fn fence(&self, s: Surface) -> Result<Option<OwnedFd>, Error> {
        let mut st = self.state()?;
        if st.written.contains(&s.0) {
            st.submit()?;
        }
        st.target(s)?.fence.as_ref().map(|f| f.export().map_err(io)).transpose()
    }

    /// The next submission waits for sync_file `fd` (a compositor done reading a surface, say)
    pub fn after(&self, fd: BorrowedFd<'_>) -> Result<(), Error> {
        let mut st = self.state()?;
        let f = self.xe.fence(false).map_err(io)?;
        f.import(fd).map_err(io)?;
        st.waits.push(f);
        Ok(())
    }

    /// One draw into surface `s` (`d.target` its size, format and tiling), recorded: the first of its program and
    /// shape lowers it with the program's kernels into an image and makes it resident; the surface is loaded as
    /// `d.target.load` says. The draws go to the GPU together (`Device`)
    pub fn draw(&self, program: &Compiled<'_>, d: &Draw<'_>, s: Surface) -> Result<(), Error> {
        let mut st = self.state()?;
        let t = st.target(s)?;
        if !t.holds(&d.target) {
            return Err(Error(format!(
                "a draw into a {}x{} {:?} {:?} target, surface {s:?} is {}x{} {:?} {:?}",
                d.target.width, d.target.height, d.target.format, d.target.tiling, t.width, t.height, t.format, t.tiling
            )));
        }
        let i = st.shape(program, d)?;
        st.load(s, &d.target.load)?;
        if !st.fits(i, d) {
            st.settle()?;
            st.restart(d)?;
        }
        st.record(i, d, s)
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        if let Some(st) = self.state.get_mut() {
            let _ = st.settle();
        }
    }
}
