//! Draws on a GPU: a card (an Xe or amdgpu render node) and the device that keeps surfaces and draws on it, the same
//! calls on either (`Device`)

mod amd;
mod target;
mod xe3;

use std::io;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::path::Path;
use std::time::Duration;

use mlirformat::Error;

use crate::{Amd, ColorTarget, Compiled, Draw, Fence, Isa, Node, Syncobj, Xe};
pub(crate) use target::SurfaceMem;

/// How long a submission may take before the GPU is called hung
const TIMEOUT: Duration = Duration::from_secs(5);

fn io(e: io::Error) -> Error {
    Error(e.to_string())
}

fn wait(f: &Syncobj<'_>, what: &str) -> Result<(), Error> {
    if f.wait(TIMEOUT).map_err(io)? { Ok(()) } else { Err(Error(format!("{what} did not finish: the GPU hung"))) }
}

/// DRM's format modifiers for a surface's tiling: linear, and Intel's Tile4 (fourcc_mod_code(INTEL, 9))
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;
pub const I915_FORMAT_MOD_4_TILED: u64 = 1 << 56 | 9;
/// DRM fourccs of the color formats: R8G8B8A8 in memory order (DRM's ABGR8888), R8
pub const DRM_FORMAT_ABGR8888: u32 = u32::from_le_bytes(*b"AB24");
pub const DRM_FORMAT_R8: u32 = u32::from_le_bytes(*b"R8  ");

/// A surface exported as a dma-buf (`Device::export`): plane 0 from `offset`, rows `pitch` bytes apart, laid out as
/// `modifier` says, in DRM format `fourcc`
#[derive(Debug)]
pub struct Exported {
    pub fd: OwnedFd,
    pub offset: u64,
    pub pitch: u32,
    pub modifier: u64,
    pub fourcc: u32,
}

/// A color target the device keeps: its pixels stay on the GPU between draws
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Surface(pub(crate) usize);

/// A GPU detour draws on: an Intel Xe3 (`xe`) or an AMD RDNA2 (`amdgpu`) render node
#[expect(clippy::large_enum_variant, reason = "a card or two in a process")]
pub enum Card {
    Xe(Xe),
    Amd(Amd),
}

impl Card {
    /// The first render node in /dev/dri whose driver detour draws with
    pub fn open_first() -> io::Result<Card> {
        Card::try_from(Node::open_any(&["xe", "amdgpu"])?)
    }

    /// Render node `path`, by its driver
    pub fn open(path: &Path) -> io::Result<Card> {
        Card::try_from(Node::open(path)?)
    }

    pub fn path(&self) -> &Path {
        match self {
            Card::Xe(xe) => xe.path(),
            Card::Amd(amd) => amd.path(),
        }
    }

    /// The instruction set its programs are compiled for (`Toolchain::new`)
    pub fn isa(&self) -> Isa {
        match self {
            Card::Xe(_) => Isa::Xe3,
            Card::Amd(_) => Isa::Rdna2,
        }
    }
}

impl TryFrom<Node> for Card {
    type Error = io::Error;

    fn try_from(node: Node) -> io::Result<Card> {
        match node.driver()?.as_str() {
            "xe" => Xe::try_from(node).map(Card::Xe),
            "amdgpu" => Amd::try_from(node).map(Card::Amd),
            d => Err(io::Error::new(io::ErrorKind::Unsupported, format!("{}: driver {d}, not xe or amdgpu", node.path().display()))),
        }
    }
}

/// A card a device draws on, borrowed: a `Card`, an `Xe` or an `Amd`
#[derive(Clone, Copy)]
pub enum CardRef<'x> {
    Xe(&'x Xe),
    Amd(&'x Amd),
}

impl<'x> From<&'x Xe> for CardRef<'x> {
    fn from(xe: &'x Xe) -> Self {
        CardRef::Xe(xe)
    }
}

impl<'x> From<&'x Amd> for CardRef<'x> {
    fn from(amd: &'x Amd) -> Self {
        CardRef::Amd(amd)
    }
}

impl<'x> From<&'x Card> for CardRef<'x> {
    fn from(card: &'x Card) -> Self {
        match card {
            Card::Xe(xe) => CardRef::Xe(xe),
            Card::Amd(amd) => CardRef::Amd(amd),
        }
    }
}

#[expect(clippy::large_enum_variant, reason = "a device or two in a process")]
enum Backend<'x> {
    Xe(xe3::XeDevice<'x>),
    Amd(amd::AmdDevice<'x>),
}

/// The GPU detour submits to, and what it keeps there in one address space: its surfaces (its own or dma-bufs
/// imported), each program's code, rings of the draws' vertices, indices and uniforms. A draw is recorded, nothing
/// waits for the GPU; the draws go to the GPU, one submission, when a surface they draw into is read, loaded with
/// other pixels, finished, flushed or fenced, or when a ring is full. The same calls on Xe3 and RDNA2
pub struct Device<'x> {
    backend: Backend<'x>,
}

/// `$e` on the device's backend, as `$d`
macro_rules! on {
    ($self:ident, $d:ident => $e:expr) => {
        match &$self.backend {
            Backend::Xe($d) => $e,
            Backend::Amd($d) => $e,
        }
    };
}

impl<'x> Device<'x> {
    pub fn new(card: impl Into<CardRef<'x>>) -> Device<'x> {
        let backend = match card.into() {
            CardRef::Xe(xe) => Backend::Xe(xe3::XeDevice::new(xe)),
            CardRef::Amd(amd) => Backend::Amd(amd::AmdDevice::new(amd)),
        };
        Device { backend }
    }

    /// The card it draws on
    pub fn card(&self) -> CardRef<'x> {
        match &self.backend {
            Backend::Xe(d) => CardRef::Xe(d.xe),
            Backend::Amd(d) => CardRef::Amd(d.amd),
        }
    }

    /// The instruction set the programs it draws are compiled for
    pub fn isa(&self) -> Isa {
        match self.backend {
            Backend::Xe(_) => Isa::Xe3,
            Backend::Amd(_) => Isa::Rdna2,
        }
    }

    /// A new surface for target `t` (its size, format and layout): zero, then loaded as `t.load` says (a texture's
    /// texels: `TargetLoad::Pixels`)
    pub fn surface(&self, t: &ColorTarget<'_>) -> Result<Surface, Error> {
        on!(self, d => d.surface(t))
    }

    /// A new surface for target `t` the display engine can scan out (a compositor's output, flipped to KMS): like
    /// `surface`, its memory write-combined and marked for scanout; `export` gives it to KMS
    pub fn scanout(&self, t: &ColorTarget<'_>) -> Result<Surface, Error> {
        on!(self, d => d.scanout(t))
    }

    /// Surface `s` as a dma-buf for another process or device (a compositor, KMS): its fd and how its pixels lie
    /// in it. Its draws are not waited for: `fence` says when they are done
    pub fn export(&self, s: Surface) -> Result<Exported, Error> {
        on!(self, d => d.export(s))
    }

    /// A surface drawn straight into dma-buf `fd` (a GBM buffer, a udmabuf...): targets like `t` (its size, format:
    /// `Rgba8Unorm` is DRM's ABGR8888; tiling: `Linear`, or on Xe3 `Tile4` for I915_FORMAT_MOD_4_TILED), from byte
    /// `offset` (on a page), rows `stride` bytes apart (on RDNA2 a multiple of 256). What it held stays until a draw
    /// writes it; `fence` says when the draws into it are done
    pub fn import(&self, t: &ColorTarget<'_>, fd: BorrowedFd<'_>, offset: u64, stride: u32) -> Result<Surface, Error> {
        on!(self, d => d.import(t, fd, offset, stride))
    }

    /// Surface `s` gone, once the draws into it are done; using it again is an error
    pub fn free(&self, s: Surface) -> Result<(), Error> {
        on!(self, d => d.free(s))
    }

    /// `pixels` (rows of `rect`'s width, tightly packed) written into rectangle `rect` (x, y, width, height) of
    /// surface `s`, once the draws into it and those sampling it are done: a texture's part updated (a glyph
    /// atlas's)
    pub fn write(&self, s: Surface, rect: [u32; 4], pixels: &[u8]) -> Result<(), Error> {
        on!(self, d => d.write(s, rect, pixels))
    }

    /// How many draw shapes (Xe3) or programs (RDNA2) were made ready to draw so far: each costs a compile or an
    /// upload; a draw of a resident one does not
    pub fn lowered(&self) -> usize {
        on!(self, d => d.lowered())
    }

    /// The first `len` bytes of surface `s`'s pixels (rows), once the draws into it are done
    pub fn read(&self, s: Surface, len: usize) -> Result<Vec<u8>, Error> {
        on!(self, d => d.read(s, len))
    }

    /// Surface `s`'s memory as it is laid out, once the draws into it are done
    pub fn raw(&self, s: Surface) -> Result<Vec<u8>, Error> {
        on!(self, d => d.raw(s))
    }

    /// The draws recorded (into `s` and every other surface) submitted, not waited for
    pub fn flush(&self, s: Surface) -> Result<(), Error> {
        on!(self, d => d.flush(s))
    }

    /// The draws into surface `s` submitted and done (glFinish)
    pub fn finish(&self, s: Surface) -> Result<(), Error> {
        on!(self, d => d.finish(s))
    }

    /// The draws recorded submitted (when some draw into `s`): a sync_file signalled once the draws into `s` are
    /// done, None when nothing ever drew into it
    pub fn fence(&self, s: Surface) -> Result<Option<OwnedFd>, Error> {
        on!(self, d => d.fence(s))
    }

    /// The next submission waits for sync_file `fd` (a compositor done reading a surface, say)
    pub fn after(&self, fd: BorrowedFd<'_>) -> Result<(), Error> {
        on!(self, d => d.after(fd))
    }

    /// One draw into surface `s` (`d.target` its size, format and tiling), recorded with `program` (compiled for
    /// the device's instruction set); the surface is loaded as `d.target.load` says. The draws go to the GPU
    /// together (`Device`)
    pub fn draw(&self, program: &Compiled, d: &Draw<'_>, s: Surface) -> Result<(), Error> {
        on!(self, x => x.draw(program, d, s))
    }
}
