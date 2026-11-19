use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::rc::{Rc, Weak};
use std::time::Duration;

use mlirformat::Error;

use crate::{
    Bo, Caching, ColorTarget, Compiled, Draw, Engine, Fence, Gpu, Host, ImageBytes, MemType, Native, Queue, Syncobj, TargetClear, TargetLoad, Tiling, Vm, Xe,
    Template, Xe3Draw, Xe3DrawImage, Xe3Image, XeBo, XeQueue, XeVm,
};

/// How long a submission may take before the GPU is called hung
const TIMEOUT: Duration = Duration::from_secs(5);
/// Draw shapes kept resident, each with its own address space and queue
const RESIDENT: usize = 64;
/// Where a shape's first-level batches go in its address space (above its image), and their ring's size
const CALLS_ADDR: u64 = 1 << 32;
const CALLS_SIZE: u64 = 1 << 20;
/// Mapped past the ring's end, and past each memory's end up to its page: the command streamer prefetches beyond
/// the batch end it reads (a GPU fault at batch + 0x1000 otherwise)
const CALLS_GUARD: u64 = Xe3Image::PAGE;
/// MI_BATCH_BUFFER_START of a second-level batch in the PPGTT (its address in the next two dwords), MI_BATCH_BUFFER_END
const MI_BATCH_BUFFER_START_2ND: u32 = 49 << 23 | 1 << 22 | 1 << 8 | 1;
const MI_BATCH_BUFFER_END: u32 = 10 << 23;
const CALL_BYTES: usize = 12;
/// A first-level batch's start: the CS reads it from a 64-byte boundary
const BATCH_ALIGN: usize = 64;
/// Where vertex buffer i's ring is mapped (`VERTEX_ADDR + i * VERTEX_SPAN`, above the calls), its least size, and
/// where each draw's vertices start in it
const VERTEX_ADDR: u64 = 1 << 36;
const VERTEX_SPAN: u64 = 1 << 32;
const VERTEX_RING: usize = 1 << 20;
const VERTEX_ALIGN: usize = 64;

fn io(e: std::io::Error) -> Error {
    Error(e.to_string())
}

fn wait(f: &Syncobj<'_>, what: &str) -> Result<(), Error> {
    if f.wait(TIMEOUT).map_err(io)? { Ok(()) } else { Err(Error(format!("{what} did not finish: the GPU hung"))) }
}

/// A color target the device keeps: its pixels stay on the GPU between draws
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Surface(usize);

/// Draws recorded into a target and not submitted yet: calls of one shape's stream, in its first-level ring
struct Pending<'x> {
    shape: Rc<RefCell<Shape<'x>>>,
    start: usize,
}

/// A surface's memory and the draws into it: recorded, and submitted together when it is read or loaded, finished,
/// or when a draw of another shape comes
struct Target<'x> {
    bo: RefCell<XeBo<'x>>,
    /// a row's bytes of pixels, its rows, how they are laid out in `bo`
    row: u32,
    height: u32,
    tiling: Tiling,
    /// the last submission writing it
    fence: RefCell<Option<Rc<Syncobj<'x>>>>,
    pending: RefCell<Option<Pending<'x>>>,
}

impl<'x> Target<'x> {
    /// The recorded draws submitted
    fn flush(&self) -> Result<(), Error> {
        let Some(p) = self.pending.borrow_mut().take() else {
            return Ok(());
        };
        let mut s = p.shape.borrow_mut();
        s.recording = None;
        let previous = self.fence.borrow().clone();
        let done = s.submit(p.start, previous.as_deref())?;
        *self.fence.borrow_mut() = Some(done);
        Ok(())
    }

    /// The draws into it submitted and done
    fn idle(&self) -> Result<(), Error> {
        self.flush()?;
        if let Some(f) = self.fence.borrow_mut().take() {
            wait(&f, "a draw")?;
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.tiling.size(self.row, self.height)
    }

    /// What it holds before a draw, as `load` says (`width` pixels a row): nothing to do when it keeps what the
    /// last draw left; otherwise the draws so far done, then the pixels written as laid out
    fn load(&self, load: &TargetLoad<'_>, width: u32) -> Result<(), Error> {
        let (base, clears): (Option<&[u8]>, &[TargetClear]) = match load {
            TargetLoad::Keep([]) => return Ok(()),
            TargetLoad::Keep(c) => (None, c),
            TargetLoad::Fill(texel, c) => {
                self.idle()?;
                let n = self.size();
                let mut bo = self.bo.borrow_mut();
                let memory = &mut bo.bytes().map_err(io)?[..n];
                match texel.iter().all(|b| *b == texel[0]) {
                    true => memory.fill(texel[0]),
                    false => memory.chunks_exact_mut(texel.len()).for_each(|p| p.copy_from_slice(texel)),
                }
                (None, c)
            }
            TargetLoad::Pixels(p, c) => (Some(*p), c),
        };
        self.idle()?;
        let mut bo = self.bo.borrow_mut();
        let memory = bo.bytes().map_err(io)?;
        if let Some(p) = base {
            let len = (self.row * self.height) as usize;
            self.tiling.tile(p.get(..len).ok_or_else(|| Error("fewer pixels than the target holds".into()))?, self.row, self.height, memory);
        }
        clears.iter().for_each(|c| c.apply_laid_out(memory, width, self.tiling));
        Ok(())
    }
}

impl Drop for Target<'_> {
    fn drop(&mut self) {
        let _ = self.idle();
    }
}

/// A command a draw writes into the first-level batch: a template and, for each of its holes, which of the values
/// given fills it
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

/// Where a vertex buffer's bytes go: a ring of them, written from its start after the shape's earlier runs finished,
/// each draw's bytes after the last's unless they are the same
struct Vertices<'x> {
    bo: Option<XeBo<'x>>,
    addr: u64,
    /// the bytes the ring holds (its memory past them mapped too: the VF may read ahead)
    size: usize,
    at: usize,
    /// where the last bytes written start, and them
    last: Option<(usize, Vec<u8>)>,
}

impl<'x> Vertices<'x> {
    /// Whether `bytes` are in place already, or fit after the last
    fn fits(&self, bytes: &[u8]) -> bool {
        self.held(bytes).is_some() || self.at.next_multiple_of(VERTEX_ALIGN) + bytes.len() <= self.size
    }

    fn held(&self, bytes: &[u8]) -> Option<usize> {
        self.last.as_ref().filter(|(_, b)| b == bytes).map(|(at, _)| *at)
    }

    /// Room for `len` bytes from the ring's start (nothing of its shape running)
    fn restart(&mut self, xe: &'x Xe, vm: &XeVm<'x>, len: usize) -> Result<(), Error> {
        (self.at, self.last) = (0, None);
        if len <= self.size {
            return Ok(());
        }
        let size = len.next_power_of_two().max(VERTEX_RING);
        let bytes = (size as u64 + Xe3Image::PAGE).next_multiple_of(Xe3Image::PAGE);
        if let Some(old) = self.bo.take() {
            vm.unbind(self.addr, old.size(), &[]).map_err(io)?;
        }
        let bo = xe.bo((bytes, Caching::WriteCombined).into()).map_err(io)?;
        bind(xe, vm, &bo, self.addr, bytes, MemType::Streaming)?;
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
        let at = self.at.next_multiple_of(VERTEX_ALIGN);
        let bo = self.bo.as_mut().ok_or_else(|| Error("a vertex ring with no memory".into()))?;
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

/// One draw shape resident: its image's memories bound in its own address space, its context set up (`INIT`), its
/// `BATCH` stream called from a ring of first-level batches run on its own queue, each draw's primitive, vertex
/// buffers and scissor written into it between the calls
struct Shape<'x> {
    key: u64,
    xe: &'x Xe,
    vm: XeVm<'x>,
    queue: XeQueue<'x>,
    bos: Vec<XeBo<'x>>,
    /// the draw's uniform memories (`Xe3Draw::data`): its buffer object, the bytes in it
    data: Vec<(usize, Vec<u8>)>,
    /// its stream's second-level batches: the draw's state, the flush
    state: u64,
    flush: u64,
    primitive: Command,
    vertex_buffers: Command,
    scissor_rect: Command,
    scissor_pointer: Command,
    /// where the vertex buffers' holes' values go in the values given, and the scissor's whole-target rectangle
    vertices: Vec<Vertices<'x>>,
    full: [u32; 4],
    /// `Xe3Draw::DYNAMIC`: its buffer object, its address, where the next scissor goes in it
    dynamic: (usize, u64),
    scissors: usize,
    /// what the run being recorded has set: the vertex buffers' addresses and sizes, the scissor
    run_vertices: Vec<u64>,
    run_scissor: [u32; 4],
    target: (u64, u64),
    /// the surface bound at `target`
    bound: Option<Surface>,
    calls: XeBo<'x>,
    /// where the next call goes in `calls`
    at: usize,
    words: Vec<u32>,
    /// the last submission
    last: Option<Rc<Syncobj<'x>>>,
    /// the target its unsubmitted calls draw into
    recording: Option<Weak<Target<'x>>>,
}

impl<'x> Shape<'x> {
    fn new(xe: &'x Xe, key: u64, parts: &Xe3DrawImage, d: &Draw<'_>) -> Result<Shape<'x>, Error> {
        let image = &parts.image;
        let vm = xe.vm().map_err(io)?;
        let queue = xe.queue(&vm, Engine::Render).map_err(io)?;
        let calls = xe.bo((CALLS_SIZE + CALLS_GUARD, Caching::WriteCombined).into()).map_err(io)?;
        bind(xe, &vm, &calls, CALLS_ADDR, CALLS_SIZE + CALLS_GUARD, MemType::Streaming)?;
        let c = &parts.commands;
        let names = |n: &[&str]| n.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let vb_names: Vec<String> = (0..d.buffers.len()).flat_map(|i| [Xe3Draw::vertex_buffer(i), format!("{}_size", Xe3Draw::vertex_buffer(i))]).collect();
        let vertices = (0..d.buffers.len())
            .map(|i| Vertices { bo: None, addr: VERTEX_ADDR + i as u64 * VERTEX_SPAN, size: 0, at: 0, last: None })
            .collect();
        let mut shape = Shape {
            key,
            xe,
            vm,
            queue,
            bos: Vec::new(),
            data: Vec::new(),
            state: parts.state,
            flush: parts.flush,
            primitive: Command::new(&c.primitive, &names(&["count", "first", "instances"]))?,
            vertex_buffers: Command::new(&c.vertex_buffers, &vb_names)?,
            scissor_rect: Command::new(&c.scissor_rect, &names(&["x0", "y0", "x1", "y1"]))?,
            scissor_pointer: Command::new(&c.scissor_pointer, &names(&["scissor"]))?,
            vertices,
            full: [0, 0, d.target.width.max(1) - 1, d.target.height.max(1) - 1],
            dynamic: (usize::MAX, 0),
            scissors: 0,
            run_vertices: Vec::new(),
            run_scissor: [0; 4],
            target: (0, 0),
            bound: None,
            calls,
            at: 0,
            words: Vec::new(),
            last: None,
            recording: None,
        };
        let names: Vec<String> = Xe3Draw::data(d).into_iter().map(|(n, _)| n).collect();
        let mut data = vec![None; names.len()];
        for s in &image.segments {
            if s.name == Xe3Draw::TARGET {
                shape.target = (s.addr, (s.bytes.len() as u64).next_multiple_of(xe.info().min_alignment));
                continue;
            }
            let mem = MemType::from(s.mem);
            let caching = if mem == MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
            let mut bo = xe.bo(((s.bytes.len() as u64).next_multiple_of(Xe3Image::PAGE), caching).into()).map_err(io)?;
            bo.bytes().map_err(io)?[..s.bytes.len()].copy_from_slice(&s.bytes);
            bind(xe, &shape.vm, &bo, s.addr, bo.size(), mem)?;
            if let Some(i) = names.iter().position(|n| *n == s.name) {
                data[i] = Some((shape.bos.len(), Vec::new()));
            }
            if s.name == Xe3Draw::DYNAMIC {
                shape.dynamic = (shape.bos.len(), s.addr);
            }
            shape.bos.push(bo);
        }
        if shape.dynamic.0 == usize::MAX {
            return Err(Error(format!("no memory {} in the image", Xe3Draw::DYNAMIC)));
        }
        shape.data = data.into_iter().zip(&names).map(|(d, n)| d.ok_or_else(|| Error(format!("no memory {n} in the image")))).collect::<Result<_, _>>()?;
        let init = image.stream_named(Xe3Draw::INIT).ok_or_else(|| Error(format!("no stream {} in the image", Xe3Draw::INIT)))?;
        Native::fence();
        let done = xe.fence(false).map_err(io)?;
        shape.queue.submit(&init, &[], &[&done]).map_err(io)?;
        wait(&done, "setting the context up")?;
        Ok(shape)
    }

    /// The last submission finished
    fn settle(&mut self) -> Result<(), Error> {
        if let Some(f) = self.last.take() {
            wait(&f, "a draw")?;
        }
        Ok(())
    }

    /// Whether this draw's uniform bytes differ from the ones in place
    fn stale(&self, d: &Draw<'_>) -> bool {
        self.data.iter().zip(Xe3Draw::data(d)).any(|((_, held), (_, b))| held != b)
    }

    /// This draw's uniform bytes in place (nothing of this shape running)
    fn refresh(&mut self, d: &Draw<'_>) -> Result<(), Error> {
        for ((bo, held), (name, f)) in self.data.iter_mut().zip(Xe3Draw::data(d)) {
            if held != f {
                let room = self.bos[*bo].bytes().map_err(io)?;
                let len = room.len();
                room.get_mut(..f.len()).ok_or_else(|| Error(format!("{name}: {} bytes do not fit {len}", f.len())))?.copy_from_slice(f);
                held.clear();
                held.extend_from_slice(f);
            }
        }
        Ok(())
    }

    /// `target` bound where the image draws (nothing of this shape running)
    fn bind_target(&mut self, s: Surface, target: &Target<'x>) -> Result<(), Error> {
        if self.bound == Some(s) {
            return Ok(());
        }
        self.unbind_target()?;
        let (addr, range) = self.target;
        if target.bo.borrow().size() < range {
            return Err(Error(format!("a surface of {} bytes for a draw writing {range}", target.bo.borrow().size())));
        }
        bind(self.xe, &self.vm, &target.bo.borrow(), addr, range, MemType::Streaming)?;
        self.bound = Some(s);
        Ok(())
    }

    /// No surface bound (nothing of this shape running)
    fn unbind_target(&mut self) -> Result<(), Error> {
        if self.bound.take().is_some() {
            let (addr, range) = self.target;
            self.vm.unbind(addr, range, &[]).map_err(io)?;
        }
        Ok(())
    }

    /// Whether a draw's commands fit in the batch being recorded, its vertices in their rings and its scissor in
    /// `DYNAMIC`, with the state's call first when the run starts with it, the flush's call and the batch's end
    fn fits(&self, d: &Draw<'_>, starting: bool) -> bool {
        let commands = self.vertex_buffers.bytes() + self.scissor_pointer.bytes() + self.primitive.bytes();
        let calls = self.at + usize::from(starting) * CALL_BYTES + commands + CALL_BYTES + 4 <= CALLS_SIZE as usize;
        let scissors = self.scissors + Xe3Draw::SCISSOR_ALIGN as usize <= Xe3Draw::DYNAMIC_SIZE as usize;
        calls && scissors && self.vertices.iter().zip(&d.buffers).all(|(v, b)| v.fits(b.bytes))
    }

    /// Every ring from its start, each vertex ring with room for this draw's (nothing of this shape running)
    fn restart(&mut self, d: &Draw<'_>) -> Result<(), Error> {
        (self.at, self.scissors) = (0, 0);
        for (v, b) in self.vertices.iter_mut().zip(&d.buffers) {
            v.restart(self.xe, &self.vm, b.bytes.len())?;
        }
        Ok(())
    }

    /// `words` written at the batch's end
    fn write(&mut self) -> Result<(), Error> {
        let at = self.at;
        let calls = self.calls.bytes().map_err(io)?;
        for (i, w) in self.words.iter().enumerate() {
            calls[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        self.at += 4 * self.words.len();
        self.words.clear();
        Ok(())
    }

    /// One call of a second-level batch at `addr` recorded
    fn call(&mut self, addr: u64) -> Result<(), Error> {
        self.words.extend([MI_BATCH_BUFFER_START_2ND, addr as u32, (addr >> 32) as u32]);
        self.write()
    }

    /// A run started: the state's call, and nothing of the vertex buffers and scissor set in it yet but the image's
    fn start(&mut self) -> Result<(), Error> {
        self.run_vertices.clear();
        self.run_scissor = self.full;
        self.call(self.state)
    }

    /// One draw recorded (it fits): its vertices and scissor where the run does not have them, its primitive
    fn record(&mut self, d: &Draw<'_>) -> Result<(), Error> {
        let mut values = Vec::with_capacity(2 * d.buffers.len());
        for (v, b) in self.vertices.iter_mut().zip(&d.buffers) {
            values.extend([v.put(b.bytes)?, b.bytes.len() as u64]);
        }
        if !values.is_empty() && values != self.run_vertices {
            self.vertex_buffers.emit(&mut self.words, &values);
            self.write()?;
            self.run_vertices = values;
        }
        let rect = Xe3Draw::scissor(d);
        if rect != self.run_scissor {
            let (bo, addr) = self.dynamic;
            let at = self.scissors;
            self.scissor_rect.emit(&mut self.words, &rect.map(u64::from));
            let memory = self.bos[bo].bytes().map_err(io)?;
            for (i, w) in self.words.drain(..).enumerate() {
                memory[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
            self.scissors += Xe3Draw::SCISSOR_ALIGN as usize;
            self.scissor_pointer.emit(&mut self.words, &[addr + at as u64]);
            self.write()?;
            self.run_scissor = rect;
        }
        self.primitive.emit(&mut self.words, &[d.count, d.first, d.instances].map(u64::from));
        self.write()
    }

    /// The calls from `start` ended (the flush's call, MI_BATCH_BUFFER_END) and submitted after `after`
    fn submit(&mut self, start: usize, after: Option<&Syncobj<'x>>) -> Result<Rc<Syncobj<'x>>, Error> {
        self.call(self.flush)?;
        self.words.push(MI_BATCH_BUFFER_END);
        self.write()?;
        self.at = self.at.next_multiple_of(BATCH_ALIGN);
        Native::fence();
        let done = Rc::new(self.xe.fence(false).map_err(io)?);
        let wait: Vec<&Syncobj<'x>> = after.into_iter().collect();
        self.queue.submit(&(CALLS_ADDR + start as u64), &wait, &[&done]).map_err(io)?;
        self.last = Some(done.clone());
        Ok(done)
    }
}

/// `bo` mapped at `addr` in `vm`, the mapping in place before returning
fn bind<'x>(xe: &'x Xe, vm: &XeVm<'x>, bo: &XeBo<'x>, addr: u64, range: u64, mem: MemType) -> Result<(), Error> {
    let done = xe.fence(false).map_err(io)?;
    vm.bind(bo, 0, addr, range, mem, &[&done]).map_err(io)?;
    wait(&done, &format!("binding {addr:#x}"))
}

/// What an image depends on in a draw: everything but its vertices, uniforms' bytes, counts, scissor and what the
/// target holds (a hash: one per draw, no allocation)
fn key(program: u64, d: &Draw<'_>) -> u64 {
    let mut h = DefaultHasher::new();
    program.hash(&mut h);
    for b in &d.buffers {
        b.stride.hash(&mut h);
    }
    for e in &d.elements {
        (e.buffer, discriminant(&e.format), e.offset).hash(&mut h);
    }
    for (binding, bytes) in &d.uniforms {
        (binding, bytes.len()).hash(&mut h);
    }
    discriminant(&d.primitive).hash(&mut h);
    d.viewport.map(f32::to_bits).hash(&mut h);
    d.depth_range.map(f32::to_bits).hash(&mut h);
    (discriminant(&d.cull), d.front_ccw, d.color_mask, d.blend.is_some()).hash(&mut h);
    if let Some(b) = &d.blend {
        (b.src.map(|f| f as u8), b.dst.map(|f| f as u8), b.equation.map(|e| e as u8), b.color.map(f32::to_bits)).hash(&mut h);
    }
    (d.target.width, d.target.height, discriminant(&d.target.format), d.target.tiling).hash(&mut h);
    h.finish()
}

/// The GPU detour submits to, and what it keeps: its surfaces, and each program and draw shape's image, resident in
/// an address space of its own (its context set up once). A draw of a shape writes its uniforms when they changed
/// (after its shape's earlier draws), its vertices into a ring, and records its vertex buffers, scissor and primitive;
/// the draws into a surface go to the GPU together, when it is read or loaded with other pixels, finished, or a draw
/// of another shape or with other uniforms comes
pub struct Device<'x> {
    pub xe: &'x Xe,
    shapes: RefCell<Vec<Rc<RefCell<Shape<'x>>>>>,
    /// by `Surface`: None once freed (its number not taken again)
    surfaces: RefCell<Vec<Option<Rc<Target<'x>>>>>,
    lowered: Cell<usize>,
}

impl<'x> Device<'x> {
    pub fn new(xe: &'x Xe) -> Device<'x> {
        Device { xe, shapes: RefCell::new(Vec::new()), surfaces: RefCell::new(Vec::new()), lowered: Cell::new(0) }
    }

    fn get(&self, s: Surface) -> Result<Rc<Target<'x>>, Error> {
        self.surfaces.borrow().get(s.0).cloned().flatten().ok_or_else(|| Error(format!("no surface {s:?}")))
    }

    /// A new surface for target `t` (its size, format and layout), zero
    pub fn surface(&self, t: &ColorTarget<'_>) -> Result<Surface, Error> {
        let bo = self.xe.bo((t.size() as u64, Caching::WriteCombined).into()).map_err(io)?;
        let target = Target { bo: RefCell::new(bo), row: t.row(), height: t.height, tiling: t.tiling, fence: RefCell::new(None), pending: RefCell::new(None) };
        let mut surfaces = self.surfaces.borrow_mut();
        surfaces.push(Some(Rc::new(target)));
        Ok(Surface(surfaces.len() - 1))
    }

    /// Surface `s` gone, its memory freed once the draws into it are done; using it again is an error
    pub fn free(&self, s: Surface) -> Result<(), Error> {
        let target = self.surfaces.borrow_mut().get_mut(s.0).and_then(Option::take).ok_or_else(|| Error(format!("no surface {s:?}")))?;
        target.idle()?;
        for shape in self.shapes.borrow().iter() {
            let mut sh = shape.borrow_mut();
            if sh.bound == Some(s) {
                sh.settle()?;
                sh.unbind_target()?;
            }
        }
        Ok(())
    }

    /// How many draw shapes were lowered into images so far (each costs a compile; a draw of a resident one does not)
    pub fn lowered(&self) -> usize {
        self.lowered.get()
    }
    /// The first `len` bytes of surface `s`'s pixels (linear rows), once the draws into it are done
    pub fn read(&self, s: Surface, len: usize) -> Result<Vec<u8>, Error> {
        let t = self.get(s)?;
        t.idle()?;
        let all = (t.row * t.height) as usize;
        let mut pixels = vec![0; all];
        t.tiling.untile(t.bo.borrow_mut().bytes().map_err(io)?, t.row, t.height, &mut pixels);
        pixels.truncate(len.min(all));
        if len > all {
            return Err(Error(format!("{len} bytes of a surface of {all}")));
        }
        Ok(pixels)
    }

    /// Surface `s`'s memory as it is laid out, once the draws into it are done
    pub fn raw(&self, s: Surface) -> Result<Vec<u8>, Error> {
        let t = self.get(s)?;
        t.idle()?;
        let n = t.size();
        Ok(t.bo.borrow_mut().bytes().map_err(io)?[..n].to_vec())
    }

    /// The draws into surface `s` submitted, not waited for
    pub fn flush(&self, s: Surface) -> Result<(), Error> {
        self.get(s)?.flush()
    }

    /// The draws into surface `s` submitted and done (glFinish)
    pub fn finish(&self, s: Surface) -> Result<(), Error> {
        self.get(s)?.idle()
    }

    fn shape(&self, program: &Compiled<'_>, d: &Draw<'_>) -> Result<Rc<RefCell<Shape<'x>>>, Error> {
        let k = key(program.id, d);
        let mut shapes = self.shapes.borrow_mut();
        if let Some(s) = shapes.iter().find(|s| s.borrow().key == k) {
            return Ok(s.clone());
        }
        let mut body = program.body.clone();
        let image = Xe3Draw::image(&mut body, d)?;
        self.lowered.set(self.lowered.get() + 1);
        if shapes.len() == RESIDENT {
            shapes.remove(0);
        }
        let s = Rc::new(RefCell::new(Shape::new(self.xe, k, &image, d)?));
        shapes.push(s.clone());
        Ok(s)
    }

    /// One draw into surface `s`, recorded: the first of its program and shape lowers it with the program's kernels
    /// into an image, makes it resident and sets its context up; the surface is loaded as `d.target.load` says; a
    /// draw of the same shape writes the uniform bytes that changed (after its shape's earlier draws), puts its
    /// vertices in their rings, and adds its vertex buffers and scissor where they changed, and its primitive. The
    /// draws go to the GPU together (`Device`)
    pub fn draw(&self, program: &Compiled<'_>, d: &Draw<'_>, s: Surface) -> Result<(), Error> {
        let target = self.get(s)?;
        let shape = self.shape(program, d)?;
        target.load(&d.target.load, d.target.width)?;
        let other = shape.borrow().recording.as_ref().and_then(Weak::upgrade).filter(|t| !Rc::ptr_eq(t, &target));
        if let Some(t) = other {
            t.flush()?;
        }
        let same = target.pending.borrow().as_ref().is_some_and(|p| Rc::ptr_eq(&p.shape, &shape));
        let mut sh = shape.borrow_mut();
        if !same || !sh.fits(d, false) || sh.stale(d) || sh.bound != Some(s) {
            drop(sh);
            target.flush()?;
            sh = shape.borrow_mut();
            if sh.stale(d) || sh.bound != Some(s) {
                sh.settle()?;
                sh.refresh(d)?;
                sh.bind_target(s, &target)?;
            }
            if !sh.fits(d, true) {
                sh.settle()?;
                sh.restart(d)?;
            }
            *target.pending.borrow_mut() = Some(Pending { shape: shape.clone(), start: sh.at });
            sh.recording = Some(Rc::downgrade(&target));
            sh.start()?;
        }
        sh.record(d)
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        for t in self.surfaces.borrow().iter().flatten() {
            let _ = t.idle();
        }
    }
}
