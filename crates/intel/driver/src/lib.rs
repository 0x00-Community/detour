//! detour on Intel Xe3: an detour MLIR module in, the frame out. The [`Codegen`] pipeline makes the image; a [`Device`]
//! runs it through the kmd traits (any [`Gpu`]: xe) and the host CPU's [`Host`]: a buffer object per memory, bound
//! where the layout put it, the batch submitted, waited for and checked for a fault.

mod copies;
mod filters;
mod fine;
mod frost;
mod pages;
mod paths;
mod raster;
mod runtime;

pub use copies::*;
pub use filters::*;
pub use fine::*;
pub use frost::*;
pub use pages::*;
pub use paths::*;
pub use raster::*;
pub use runtime::*;

use std::fmt;
use std::marker::PhantomData;
use std::time::Duration;

use crate::base::{Bo, Caching, Engine, Fence, Gpu, Host, MemType, Queue, Vm};

pub use crate::base::{Image, Segment};
pub use crate::xe3_codegen::Codegen;

#[derive(Debug)]
pub enum Error {
    /// The compiler refused the module.
    Compile(mlirformat::Error),
    Kmd(std::io::Error),
    /// The job did not finish in time.
    Timeout,
    /// The driver stopped the queue (a fault, a hang): its reason.
    Fault(u64),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Compile(e) => write!(f, "compile: {}", e.0),
            Error::Kmd(e) => write!(f, "kernel driver: {e}"),
            Error::Timeout => f.write_str("the job did not finish"),
            Error::Fault(r) => write!(f, "the queue was stopped (reason {r:#x})"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// A refusal that is not the compiler's own (a missing memory, a full heap, a bad argument), reported as
    /// [`Error::Compile`].
    fn msg(text: impl Into<String>) -> Self {
        Error::Compile(mlirformat::Error(text.into()))
    }
}

impl From<mlirformat::Error> for Error {
    fn from(e: mlirformat::Error) -> Self {
        Error::Compile(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Kmd(e)
    }
}

/// ⭐ A GPU address space and a render queue on `gpu`, the host CPU `H`.
pub struct Device<'d, G: Gpu<Cmd = u64> + 'd, H: Host> {
    gpu: &'d G,
    vm: G::Vm<'d>,
    queue: G::Queue<'d>,
    host: PhantomData<H>,
}

/// What a run left in memory: a buffer object per segment, by name.
pub struct Frame<'d, G: Gpu<Cmd = u64> + 'd> {
    bos: Vec<(String, G::Bo<'d>)>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Frame<'d, G> {
    /// The bytes of memory `name` (a texture's texels, a buffer's contents).
    pub fn bytes(&mut self, name: &str) -> Result<&mut [u8], Error> {
        let bo = self.bos.iter_mut().find(|(n, _)| n == name).map(|(_, b)| b).ok_or_else(|| Error::msg(format!("no memory @{name}")))?;
        Ok(bo.bytes()?)
    }
}

impl<'d, G: Gpu<Cmd = u64> + 'd, H: Host> Device<'d, G, H> {
    pub fn new(gpu: &'d G) -> Result<Self, Error> {
        let vm = gpu.vm()?;
        let queue = gpu.queue(&vm, Engine::Render)?;
        Ok(Device { gpu, vm, queue, host: PhantomData })
    }

    /// ⭐ Runs `image`: its memories in buffer objects at their addresses, the batch, waited for (`timeout`).
    pub fn run(&self, image: &Image, timeout: Duration) -> Result<Frame<'d, G>, Error> {
        let mut bos = Vec::new();
        let mut bound = Vec::new();
        for s in &image.segments {
            let caching = if s.mem == MemType::Coherent { Caching::WriteBack } else { Caching::WriteCombined };
            let mut bo = self.gpu.bo((s.bytes.len() as u64, caching).into())?;
            bo.bytes()?[..s.bytes.len()].copy_from_slice(&s.bytes);
            if caching == Caching::WriteBack {
                H::flush(&bo.bytes()?[..s.bytes.len()]);
            }
            bos.push((s.name.clone(), bo));
        }
        // write-combined stores out before the GPU reads
        H::fence();
        for ((_, bo), s) in bos.iter().zip(&image.segments) {
            let done = self.gpu.fence(false)?;
            self.vm.bind(bo, 0, s.addr, bo.size(), s.mem, &[&done])?;
            bound.push(done);
        }
        let done = self.gpu.fence(false)?;
        self.queue.submit(&image.stream, &bound.iter().collect::<Vec<_>>(), &[&done])?;
        if !done.wait(timeout)? {
            return Err(Error::Timeout);
        }
        if let Some(reason) = self.queue.fault(&done)? {
            return Err(Error::Fault(reason));
        }
        Ok(Frame { bos })
    }
}
