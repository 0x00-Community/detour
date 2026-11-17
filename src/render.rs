use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Duration;

use mlirformat::Error;

use crate::{Bo, Caching, Compiled, Draw, Engine, Fence, Gpu, Host, Image, MemType, Native, Queue, Vm, Xe, Xe3Draw, XeBo, XeQueue, XeVm};

/// How long a submission may take before the GPU is called hung
const TIMEOUT: Duration = Duration::from_secs(5);

/// The GPU detour submits to, and what it keeps of the draws it ran: each program and draw shape's image, resident
/// in an address space of its own, so that a draw of the same shape only writes its data and submits
pub struct Device<'x> {
    pub xe: &'x Xe,
    resident: RefCell<BTreeMap<(u64, String), Resident<'x>>>,
}

/// An image on the GPU: its memories in buffer objects bound at their addresses, the queue that runs its stream
struct Resident<'x> {
    _vm: XeVm<'x>,
    queue: XeQueue<'x>,
    bos: Vec<(String, XeBo<'x>)>,
    stream: u64,
}

/// What an image depends on in a draw: everything but its bytes
struct Shape;

impl Shape {
    fn of(d: &Draw<'_>) -> String {
        let buffers: Vec<(usize, u32)> = d.buffers.iter().map(|b| (b.bytes.len(), b.stride)).collect();
        let uniforms: Vec<(u32, usize)> = d.uniforms.iter().map(|&(b, bytes)| (b, bytes.len())).collect();
        let t = &d.target;
        format!(
            "{buffers:?} {:?} {uniforms:?} {:?} {} {} {} {:?} {:?} {:?} {:?} {} {}x{} {:?} {}",
            d.elements, d.primitive, d.first, d.count, d.instances, d.viewport, d.depth_range, d.scissor, d.cull, d.front_ccw, t.width, t.height, t.format, t.pixels.len()
        )
    }
}

impl<'x> Device<'x> {
    pub fn new(xe: &'x Xe) -> Device<'x> {
        Device { xe, resident: RefCell::new(BTreeMap::new()) }
    }

    fn io(e: std::io::Error) -> Error {
        Error(e.to_string())
    }

    /// One draw: the first of its program and shape lowers it with the program's kernels into an image and makes it
    /// resident; every draw writes its data there and runs it. The color target's bytes after it
    pub fn draw(&self, program: &Compiled<'_>, d: &Draw<'_>) -> Result<Vec<u8>, Error> {
        let key = (program.id, Shape::of(d));
        let mut resident = self.resident.borrow_mut();
        if !resident.contains_key(&key) {
            let mut body = program.body.clone();
            let image = Xe3Draw::image(&mut body, d)?;
            let r = self.load(&image)?;
            resident.insert(key.clone(), r);
        }
        let r = resident.get_mut(&key).ok_or_else(|| Error("an image that was just made resident".into()))?;
        for (name, bytes) in Xe3Draw::data(d) {
            let (_, bo) = r.bos.iter_mut().find(|(n, _)| *n == name).ok_or_else(|| Error(format!("no memory {name}")))?;
            let room = bo.bytes().map_err(Device::io)?;
            let len = room.len();
            room.get_mut(..bytes.len()).ok_or_else(|| Error(format!("{name}: {} bytes do not fit {len}", bytes.len())))?.copy_from_slice(bytes);
        }
        Native::fence();
        let done = self.xe.fence(false).map_err(Device::io)?;
        r.queue.submit(&r.stream, &[], &[&done]).map_err(Device::io)?;
        if !done.wait(TIMEOUT).map_err(Device::io)? {
            return Err(Error("the submission did not finish: the GPU hung".into()));
        }
        if let Some(fault) = r.queue.fault(&done).map_err(Device::io)? {
            return Err(Error(format!("GPU page fault at {fault:#x}")));
        }
        let (_, bo) = r.bos.iter_mut().find(|(n, _)| n == Xe3Draw::TARGET).ok_or_else(|| Error(format!("no segment {}", Xe3Draw::TARGET)))?;
        let mut target = bo.bytes().map_err(Device::io)?.to_vec();
        target.truncate(d.target.pixels.len());
        Ok(target)
    }

    /// The image's segments in their own buffer objects at their addresses, in a new address space with a queue on
    /// the render engine
    fn load(&self, image: &Image) -> Result<Resident<'x>, Error> {
        let xe = self.xe;
        let vm = xe.vm().map_err(Device::io)?;
        let queue = xe.queue(&vm, Engine::Render).map_err(Device::io)?;
        let mut bos = Vec::new();
        for s in &image.segments {
            let mem = MemType::from(s.mem);
            let caching = if mem == MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
            let mut bo = xe.bo((s.bytes.len() as u64, caching).into()).map_err(Device::io)?;
            bo.bytes().map_err(Device::io)?[..s.bytes.len()].copy_from_slice(&s.bytes);
            let done = xe.fence(false).map_err(Device::io)?;
            vm.bind(&bo, 0, s.addr, bo.size(), mem, &[&done]).map_err(Device::io)?;
            if !done.wait(TIMEOUT).map_err(Device::io)? {
                return Err(Error(format!("binding {} timed out", s.name)));
            }
            bos.push((s.name.clone(), bo));
        }
        Ok(Resident { _vm: vm, queue, bos, stream: image.stream })
    }
}
