use std::time::Duration;

use mlirformat::Error;
use mlirpass::{Draw, Image, Xe3Draw};

use crate::{Bo, Caching, Compiled, Engine, Fence, Gpu, Host, MemType, Native, Queue, Vm, Xe};

/// How long a submission may take before the GPU is called hung
const TIMEOUT: Duration = Duration::from_secs(5);

/// The GPU detour submits to
pub struct Device {
    pub xe: Xe,
}

impl Device {
    pub fn open() -> Result<Device, Error> {
        Xe::open_first().map(|xe| Device { xe }).map_err(|e| Error(format!("opening a GPU: {e}")))
    }

    /// One draw: mlirpass lowers it with the program's kernels into an image, which runs here; the color target's
    /// bytes after it
    pub fn draw(&self, program: &Compiled<'_>, d: &Draw<'_>) -> Result<Vec<u8>, Error> {
        let mut body = program.body.clone();
        let image = Xe3Draw::image(&mut body, d)?;
        let mut target = self.run(&image, Xe3Draw::TARGET)?;
        target.truncate(d.target.pixels.len());
        Ok(target)
    }

    /// The image's segments in their own buffer objects at their addresses, its stream run on the render engine;
    /// the bytes of segment `readback` after it
    fn run(&self, image: &Image, readback: &str) -> Result<Vec<u8>, Error> {
        let io = |e: std::io::Error| Error(e.to_string());
        let xe = &self.xe;
        let vm = xe.vm().map_err(io)?;
        let queue = xe.queue(&vm, Engine::Render).map_err(io)?;
        let mut bos = Vec::new();
        for s in &image.segments {
            let mem = MemType::from(s.mem);
            let caching = if mem == MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
            let mut bo = xe.bo((s.bytes.len() as u64, caching).into()).map_err(io)?;
            bo.bytes().map_err(io)?[..s.bytes.len()].copy_from_slice(&s.bytes);
            let done = xe.fence(false).map_err(io)?;
            vm.bind(&bo, 0, s.addr, bo.size(), mem, &[&done]).map_err(io)?;
            if !done.wait(TIMEOUT).map_err(io)? {
                return Err(Error(format!("binding {} timed out", s.name)));
            }
            bos.push((s.name.as_str(), bo));
        }
        Native::fence();
        let done = xe.fence(false).map_err(io)?;
        queue.submit(&image.stream, &[], &[&done]).map_err(io)?;
        if !done.wait(TIMEOUT).map_err(io)? {
            return Err(Error("the submission did not finish: the GPU hung".into()));
        }
        if let Some(fault) = queue.fault(&done).map_err(io)? {
            return Err(Error(format!("GPU page fault at {fault:#x}")));
        }
        let (_, bo) = bos.iter_mut().find(|(n, _)| *n == readback).ok_or_else(|| Error(format!("no segment {readback}")))?;
        Ok(bo.bytes().map_err(io)?.to_vec())
    }
}
