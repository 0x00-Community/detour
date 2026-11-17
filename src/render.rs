use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Duration;

use mlirformat::Error;

use crate::{Bo, Caching, Compiled, Draw, Engine, Fence, Gpu, Host, Image, MemType, Native, Queue, TargetClear, TargetFill, TargetLoad, Vm, Xe, Xe3Draw, XeBo, XeQueue, XeVm};

/// How long a submission may take before the GPU is called hung
const TIMEOUT: Duration = Duration::from_secs(5);

/// A color target the device keeps: its pixels stay on the GPU between draws
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Surface(usize);

/// The GPU detour submits to, and what it keeps: its surfaces, and each program and draw shape's image, resident in
/// an address space of its own (its context set up once), so that a draw of the same shape only writes its data
/// and submits
pub struct Device<'x> {
    pub xe: &'x Xe,
    resident: RefCell<BTreeMap<(u64, String), Resident<'x>>>,
    surfaces: RefCell<Vec<XeBo<'x>>>,
}

/// An image on the GPU: its memories in buffer objects bound at their addresses (the target: the surface drawn
/// into), the queue that runs its streams
struct Resident<'x> {
    vm: XeVm<'x>,
    queue: XeQueue<'x>,
    bos: Vec<(String, XeBo<'x>)>,
    batch: u64,
    /// where the target is, how many bytes, and which surface is bound there
    target: (u64, u64, Option<Surface>),
}

/// What an image depends on in a draw: everything but its bytes
struct Shape;

impl Shape {
    fn of(d: &Draw<'_>) -> String {
        let buffers: Vec<(usize, u32)> = d.buffers.iter().map(|b| (b.bytes.len(), b.stride)).collect();
        let uniforms: Vec<(u32, usize)> = d.uniforms.iter().map(|&(b, bytes)| (b, bytes.len())).collect();
        let t = &d.target;
        format!(
            "{buffers:?} {:?} {uniforms:?} {:?} {} {} {} {:?} {:?} {:?} {:?} {} {}x{} {:?}",
            d.elements, d.primitive, d.first, d.count, d.instances, d.viewport, d.depth_range, d.scissor, d.cull, d.front_ccw, t.width, t.height, t.format
        )
    }
}

impl<'x> Device<'x> {
    pub fn new(xe: &'x Xe) -> Device<'x> {
        Device { xe, resident: RefCell::new(BTreeMap::new()), surfaces: RefCell::new(Vec::new()) }
    }

    fn io(e: std::io::Error) -> Error {
        Error(e.to_string())
    }

    /// A new surface of `bytes` bytes, zero
    pub fn surface(&self, bytes: usize) -> Result<Surface, Error> {
        let bo = self.xe.bo((bytes as u64, Caching::WriteCombined).into()).map_err(Device::io)?;
        let mut surfaces = self.surfaces.borrow_mut();
        surfaces.push(bo);
        Ok(Surface(surfaces.len() - 1))
    }

    /// The first `len` bytes of surface `s`
    pub fn read(&self, s: Surface, len: usize) -> Result<Vec<u8>, Error> {
        let mut surfaces = self.surfaces.borrow_mut();
        let bo = surfaces.get_mut(s.0).ok_or_else(|| Error(format!("no surface {s:?}")))?;
        let bytes = bo.bytes().map_err(Device::io)?;
        bytes.get(..len).map(<[u8]>::to_vec).ok_or_else(|| Error(format!("{len} bytes of a surface of {}", bytes.len())))
    }

    /// One draw into surface `target`: the first of its program and shape lowers it with the program's kernels into
    /// an image, makes it resident and sets its context up; every draw loads the target as `d` says, writes its data
    /// and runs the image's batch
    pub fn draw(&self, program: &Compiled<'_>, d: &Draw<'_>, target: Surface) -> Result<(), Error> {
        let key = (program.id, Shape::of(d));
        let mut resident = self.resident.borrow_mut();
        if !resident.contains_key(&key) {
            let mut body = program.body.clone();
            let image = Xe3Draw::image(&mut body, d)?;
            let r = self.load(&image)?;
            resident.insert(key.clone(), r);
        }
        let r = resident.get_mut(&key).ok_or_else(|| Error("an image that was just made resident".into()))?;
        let mut surfaces = self.surfaces.borrow_mut();
        let surface = surfaces.get_mut(target.0).ok_or_else(|| Error(format!("no surface {target:?}")))?;
        let (addr, size, bound) = r.target;
        if bound != Some(target) {
            let done = self.xe.fence(false).map_err(Device::io)?;
            if bound.is_some() {
                r.vm.unbind(addr, size, &[]).map_err(Device::io)?;
            }
            r.vm.bind(surface, 0, addr, size.min(surface.size()), MemType::Streaming, &[&done]).map_err(Device::io)?;
            Device::wait(&done, "binding the target")?;
            r.target.2 = Some(target);
        }
        Device::load_target(surface.bytes().map_err(Device::io)?, d)?;
        for (name, bytes) in Xe3Draw::data(d) {
            let (_, bo) = r.bos.iter_mut().find(|(n, _)| *n == name).ok_or_else(|| Error(format!("no memory {name}")))?;
            let room = bo.bytes().map_err(Device::io)?;
            let len = room.len();
            room.get_mut(..bytes.len()).ok_or_else(|| Error(format!("{name}: {} bytes do not fit {len}", bytes.len())))?.copy_from_slice(bytes);
        }
        self.run(&r.queue, r.batch)
    }

    /// The target's pixels before the draw, as `d.target.load` says
    fn load_target(pixels: &mut [u8], d: &Draw<'_>) -> Result<(), Error> {
        let t = &d.target;
        let len = t.len();
        let pixels = pixels.get_mut(..len).ok_or_else(|| Error(format!("a target of {len} bytes in a smaller surface")))?;
        let clears: &[TargetClear] = match t.load {
            TargetLoad::Keep(c) => c,
            TargetLoad::Fill(texel, c) => {
                TargetFill::apply(pixels, texel);
                c
            }
            TargetLoad::Pixels(p, c) => {
                pixels.copy_from_slice(p.get(..len).ok_or_else(|| Error("fewer pixels than the target holds".into()))?);
                c
            }
        };
        clears.iter().for_each(|c| c.apply(pixels, t.width));
        Ok(())
    }

    fn wait(done: &impl Fence, what: &str) -> Result<(), Error> {
        match done.wait(TIMEOUT).map_err(Device::io)? {
            true => Ok(()),
            false => Err(Error(format!("{what} did not finish: the GPU hung"))),
        }
    }

    /// Command stream `stream` run on `queue`, waited for
    fn run(&self, queue: &XeQueue<'x>, stream: u64) -> Result<(), Error> {
        Native::fence();
        let done = self.xe.fence(false).map_err(Device::io)?;
        queue.submit(&stream, &[], &[&done]).map_err(Device::io)?;
        Device::wait(&done, "the submission")?;
        if let Some(fault) = queue.fault(&done).map_err(Device::io)? {
            return Err(Error(format!("the queue was banned ({fault:#x}) running the stream at {stream:#x}")));
        }
        Ok(())
    }

    /// The image's segments but the target in their own buffer objects at their addresses, in a new address space
    /// with a queue on the render engine, its `INIT` stream run
    fn load(&self, image: &Image) -> Result<Resident<'x>, Error> {
        let xe = self.xe;
        let vm = xe.vm().map_err(Device::io)?;
        let queue = xe.queue(&vm, Engine::Render).map_err(Device::io)?;
        let mut bos = Vec::new();
        let mut target = None;
        for s in &image.segments {
            if s.name == Xe3Draw::TARGET {
                target = Some((s.addr, s.bytes.len() as u64, None));
                continue;
            }
            let mem = MemType::from(s.mem);
            let caching = if mem == MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
            let mut bo = xe.bo((s.bytes.len() as u64, caching).into()).map_err(Device::io)?;
            bo.bytes().map_err(Device::io)?[..s.bytes.len()].copy_from_slice(&s.bytes);
            let done = xe.fence(false).map_err(Device::io)?;
            vm.bind(&bo, 0, s.addr, bo.size(), mem, &[&done]).map_err(Device::io)?;
            Device::wait(&done, &format!("binding {}", s.name))?;
            bos.push((s.name.clone(), bo));
        }
        let stream = |name: &str| image.stream_named(name).ok_or_else(|| Error(format!("no stream {name} in the image")));
        let (init, batch) = (stream(Xe3Draw::INIT)?, stream(Xe3Draw::BATCH)?);
        let target = target.ok_or_else(|| Error(format!("no memory {} in the image", Xe3Draw::TARGET)))?;
        let r = Resident { vm, queue, bos, batch, target };
        self.run(&r.queue, init)?;
        Ok(r)
    }
}
