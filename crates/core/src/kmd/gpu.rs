//! The GPU's kernel driver: one implementation per driver (xe, asahi), everything above asks only these traits.

use std::io;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::time::Duration;

pub type Result<T> = io::Result<T>;

/// How the CPU maps a buffer (fixed when it is created).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caching {
    /// Cached and coherent with the GPU: buffers the CPU reads back.
    WriteBack,
    /// Write-combined: buffers the CPU only writes (batches, uploads, scanout).
    WriteCombined,
}

/// How the GPU caches a mapping (the driver turns it into its page attributes); `mem = "coherent"` on an
/// `image.memory`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemType {
    /// Coherent with the CPU's caches; the buffer is `Caching::WriteBack`.
    Coherent,
    /// Not snooped; the buffer is `Caching::WriteCombined`.
    Streaming,
    /// Read by the display engine or another device.
    Scanout,
    /// Compressed (render targets).
    Compressed,
}

/// The engine a queue runs on.
/// How an image's memory (mlirpass `image.memory {mem}`) is cached: its buffer object's.
impl From<mlirpass::MemType> for MemType {
    fn from(m: mlirpass::MemType) -> Self {
        match m {
            mlirpass::MemType::Coherent => Self::Coherent,
            mlirpass::MemType::Streaming => Self::Streaming,
            mlirpass::MemType::Scanout => Self::Scanout,
            mlirpass::MemType::Compressed => Self::Compressed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Render,
    Compute,
    Copy,
}

/// A buffer to create.
#[derive(Clone, Copy, Debug)]
pub struct BoDesc {
    pub size: u64,
    pub caching: Caching,
}

impl From<(u64, Caching)> for BoDesc {
    fn from((size, caching): (u64, Caching)) -> Self {
        Self { size, caching }
    }
}

/// What every driver reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub vendor_id: u16,
    pub device_id: u16,
    pub revision: u8,
    pub has_vram: bool,
    /// GPU virtual address bits.
    pub va_bits: u32,
    /// Buffer sizes and GPU addresses are multiples of this.
    pub min_alignment: u64,
}

/// ⭐ An open GPU. Buffers, address spaces, queues and fences borrow it.
pub trait Gpu: Sized {
    /// What only this driver reports.
    type Info: core::fmt::Debug;
    type Bo<'d>: Bo
    where
        Self: 'd;
    type Vm<'d>: Vm<'d, Self>
    where
        Self: 'd;
    type Queue<'d>: Queue<'d, Self>
    where
        Self: 'd;
    type Fence<'d>: Fence
    where
        Self: 'd;
    /// What a queue runs: xe a batch buffer's GPU address, asahi a list of compute / render commands.
    type Cmd: ?Sized;

    /// The first device this driver drives.
    fn open_first() -> Result<Self>;
    fn info(&self) -> DeviceInfo;
    fn driver_info(&self) -> &Self::Info;
    fn bo(&self, desc: BoDesc) -> Result<Self::Bo<'_>>;
    /// A dma-buf another device or process made (a compositor's buffer), as a buffer of this one. Not mapped: the
    /// CPU does not touch it.
    fn import(&self, fd: BorrowedFd<'_>) -> Result<Self::Bo<'_>>;
    /// A GPU virtual address space; addresses in it are the caller's to place.
    fn vm(&self) -> Result<Self::Vm<'_>>;
    fn queue<'d>(&'d self, vm: &Self::Vm<'d>, engine: Engine) -> Result<Self::Queue<'d>>;
    fn fence(&self, signaled: bool) -> Result<Self::Fence<'_>>;
}

/// A buffer object.
pub trait Bo {
    fn size(&self) -> u64;
    /// The CPU's view of it (mapped on first use).
    fn bytes(&mut self) -> Result<&mut [u8]>;
    /// As a dma-buf.
    fn export(&self) -> Result<OwnedFd>;
}

/// A GPU virtual address space.
pub trait Vm<'d, G: Gpu + 'd> {
    /// Maps `range` bytes of `bo` from `offset` at GPU address `addr`; asynchronous, signals `done`.
    fn bind(&self, bo: &G::Bo<'d>, offset: u64, addr: u64, range: u64, mem: MemType, done: &[&G::Fence<'d>]) -> Result<()>;
    fn unbind(&self, addr: u64, range: u64, done: &[&G::Fence<'d>]) -> Result<()>;
}

/// A queue of command buffers on one engine.
pub trait Queue<'d, G: Gpu + 'd> {
    /// Runs `cmd` after `wait`, signals `signal` when it is done.
    fn submit(&self, cmd: &G::Cmd, wait: &[&G::Fence<'d>], signal: &[&G::Fence<'d>]) -> Result<()>;
    /// Why the job that signalled `done` did not run to the end (a fault, a hang), `None` when it did. A fence
    /// signals even when its job faulted: this is how to tell. Asked after `done` signalled.
    fn fault(&self, done: &G::Fence<'d>) -> Result<Option<u64>>;
}

/// A GPU fence the CPU can wait on and hand on.
pub trait Fence {
    /// `false` on timeout.
    fn wait(&self, timeout: Duration) -> Result<bool>;
    /// As a sync_file (an acquire fence for the compositor).
    fn export(&self) -> Result<OwnedFd>;
}
