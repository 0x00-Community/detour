use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::rc::{Rc, Weak};
use std::time::Duration;

use mlirformat::Error;

use crate::{
    Bo, Caching, ColorTarget, Compiled, Draw, Engine, Fence, Gpu, Host, ImageBytes, MemType, Native, Queue, Syncobj, TargetClear, TargetLoad, Tiling, Vm, Xe,
    Xe3Draw, Xe3DrawImage, Xe3Image, XeBo, XeQueue, XeVm,
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

/// One draw shape resident: its image's memories bound in its own address space, its context set up (`INIT`), its
/// `BATCH` stream called from a ring of first-level batches run on its own queue
struct Shape<'x> {
    key: u64,
    xe: &'x Xe,
    vm: XeVm<'x>,
    queue: XeQueue<'x>,
    bos: Vec<XeBo<'x>>,
    /// the draw's data memories (`Xe3Draw::data`): its buffer object, the bytes in it
    data: Vec<(usize, Vec<u8>)>,
    /// its stream's second-level batches: the draw's state, the primitive, the flush
    state: u64,
    primitive: u64,
    flush: u64,
    target: (u64, u64),
    /// the surface bound at `target`
    bound: Option<Surface>,
    calls: XeBo<'x>,
    /// where the next call goes in `calls`
    at: usize,
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
        let mut shape = Shape {
            key,
            xe,
            vm,
            queue,
            bos: Vec::new(),
            data: Vec::new(),
            state: parts.state,
            primitive: parts.primitive,
            flush: parts.flush,
            target: (0, 0),
            bound: None,
            calls,
            at: 0,
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
            shape.bos.push(bo);
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

    /// A draw's vertex and uniform bytes, in `Xe3Draw::data`'s order, without its names
    fn fresh<'d>(d: &Draw<'d>) -> impl Iterator<Item = &'d [u8]> {
        d.buffers.iter().map(|b| b.bytes).chain(d.uniforms.iter().map(|&(_, u)| u))
    }

    /// Whether this draw's vertex and uniform bytes differ from the ones in place
    fn stale(&self, d: &Draw<'_>) -> bool {
        self.data.iter().zip(Shape::fresh(d)).any(|((_, held), b)| held != b)
    }

    /// This draw's vertex and uniform bytes in place (nothing of this shape running)
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
        let (addr, range) = self.target;
        if self.bound.take().is_some() {
            self.vm.unbind(addr, range, &[]).map_err(io)?;
        }
        if target.bo.borrow().size() < range {
            return Err(Error(format!("a surface of {} bytes for a draw writing {range}", target.bo.borrow().size())));
        }
        bind(self.xe, &self.vm, &target.bo.borrow(), addr, range, MemType::Streaming)?;
        self.bound = Some(s);
        Ok(())
    }

    /// Whether one more call fits before the flush's call and the batch's end
    fn fits(&self) -> bool {
        self.at + 2 * CALL_BYTES + 4 <= CALLS_SIZE as usize
    }

    /// Room for a new run (the state's call, a call, the flush's, the end): the ring started over (its earlier
    /// batches finished) when there was not
    fn wrap(&mut self) -> Result<(), Error> {
        if self.at + 3 * CALL_BYTES + 4 > CALLS_SIZE as usize {
            self.settle()?;
            self.at = 0;
        }
        Ok(())
    }

    /// One call of a second-level batch at `addr` recorded
    fn call(&mut self, addr: u64) -> Result<(), Error> {
        let words = [MI_BATCH_BUFFER_START_2ND, addr as u32, (addr >> 32) as u32];
        let at = self.at;
        let calls = self.calls.bytes().map_err(io)?;
        for (i, w) in words.iter().enumerate() {
            calls[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        self.at += CALL_BYTES;
        Ok(())
    }

    /// The calls from `start` ended (the flush's call, MI_BATCH_BUFFER_END) and submitted after `after`
    fn submit(&mut self, start: usize, after: Option<&Syncobj<'x>>) -> Result<Rc<Syncobj<'x>>, Error> {
        self.call(self.flush)?;
        let at = self.at;
        self.calls.bytes().map_err(io)?[at..at + 4].copy_from_slice(&MI_BATCH_BUFFER_END.to_le_bytes());
        self.at = (at + 4).next_multiple_of(BATCH_ALIGN);
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

/// What an image depends on in a draw: everything but its bytes and what the target holds (a hash: one per draw,
/// no allocation)
fn key(program: u64, d: &Draw<'_>) -> u64 {
    let mut h = DefaultHasher::new();
    program.hash(&mut h);
    for b in &d.buffers {
        (b.bytes.len(), b.stride).hash(&mut h);
    }
    for e in &d.elements {
        (e.buffer, discriminant(&e.format), e.offset).hash(&mut h);
    }
    for (binding, bytes) in &d.uniforms {
        (binding, bytes.len()).hash(&mut h);
    }
    (discriminant(&d.primitive), d.first, d.count, d.instances).hash(&mut h);
    d.viewport.map(f32::to_bits).hash(&mut h);
    d.depth_range.map(f32::to_bits).hash(&mut h);
    (d.scissor, discriminant(&d.cull), d.front_ccw, d.color_mask, d.blend.is_some()).hash(&mut h);
    if let Some(b) = &d.blend {
        (b.src.map(|f| f as u8), b.dst.map(|f| f as u8), b.equation.map(|e| e as u8), b.color.map(f32::to_bits)).hash(&mut h);
    }
    (d.target.width, d.target.height, discriminant(&d.target.format), d.target.tiling).hash(&mut h);
    h.finish()
}

/// The GPU detour submits to, and what it keeps: its surfaces, and each program and draw shape's image, resident in
/// an address space of its own (its context set up once). A draw of a shape writes the data that changed and records
/// a call of its primitive; the draws into a surface go to the GPU together, when it is read or loaded with other
/// pixels, finished, or a draw of another shape comes
pub struct Device<'x> {
    pub xe: &'x Xe,
    shapes: RefCell<Vec<Rc<RefCell<Shape<'x>>>>>,
    surfaces: RefCell<Vec<Rc<Target<'x>>>>,
}

impl<'x> Device<'x> {
    pub fn new(xe: &'x Xe) -> Device<'x> {
        Device { xe, shapes: RefCell::new(Vec::new()), surfaces: RefCell::new(Vec::new()) }
    }

    fn get(&self, s: Surface) -> Result<Rc<Target<'x>>, Error> {
        self.surfaces.borrow().get(s.0).cloned().ok_or_else(|| Error(format!("no surface {s:?}")))
    }

    /// A new surface for target `t` (its size, format and layout), zero
    pub fn surface(&self, t: &ColorTarget<'_>) -> Result<Surface, Error> {
        let bo = self.xe.bo((t.size() as u64, Caching::WriteCombined).into()).map_err(io)?;
        let target = Target { bo: RefCell::new(bo), row: t.row(), height: t.height, tiling: t.tiling, fence: RefCell::new(None), pending: RefCell::new(None) };
        let mut surfaces = self.surfaces.borrow_mut();
        surfaces.push(Rc::new(target));
        Ok(Surface(surfaces.len() - 1))
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
        if shapes.len() == RESIDENT {
            shapes.remove(0);
        }
        let s = Rc::new(RefCell::new(Shape::new(self.xe, k, &image, d)?));
        shapes.push(s.clone());
        Ok(s)
    }

    /// One draw into surface `s`, recorded: the first of its program and shape lowers it with the program's kernels
    /// into an image, makes it resident and sets its context up; the surface is loaded as `d.target.load` says; a
    /// draw of the same shape writes the vertex and uniform bytes that changed (after its shape's earlier draws) and
    /// adds a call of its primitive. The draws go to the GPU together (`Device`)
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
        if !same || !sh.fits() || sh.stale(d) || sh.bound != Some(s) {
            drop(sh);
            target.flush()?;
            sh = shape.borrow_mut();
            if sh.stale(d) || sh.bound != Some(s) {
                sh.settle()?;
                sh.refresh(d)?;
                sh.bind_target(s, &target)?;
            }
            sh.wrap()?;
            *target.pending.borrow_mut() = Some(Pending { shape: shape.clone(), start: sh.at });
            sh.recording = Some(Rc::downgrade(&target));
            let state = sh.state;
            sh.call(state)?;
        }
        let primitive = sh.primitive;
        sh.call(primitive)
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        for t in self.surfaces.borrow().iter() {
            let _ = t.idle();
        }
    }
}
