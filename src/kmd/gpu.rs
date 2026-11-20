use std::io;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::time::Duration;

pub type Result<T> = io::Result<T>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caching {
    WriteBack,
    WriteCombined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemType {
    Coherent,
    Streaming,
    Scanout,
    Compressed,
}

impl From<crate::ImageMemType> for MemType {
    fn from(m: crate::ImageMemType) -> Self {
        match m {
            crate::ImageMemType::Coherent => Self::Coherent,
            crate::ImageMemType::Streaming => Self::Streaming,
            crate::ImageMemType::Scanout => Self::Scanout,
            crate::ImageMemType::Compressed => Self::Compressed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Render,
    Compute,
    Copy,
}

#[derive(Clone, Copy, Debug)]
pub struct BoDesc {
    pub size: u64,
    pub caching: Caching,
    /// the display engine may scan it out (write-combined)
    pub scanout: bool,
}

impl From<(u64, Caching)> for BoDesc {
    fn from((size, caching): (u64, Caching)) -> Self {
        Self { size, caching, scanout: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub vendor_id: u16,
    pub device_id: u16,
    pub revision: u8,
    pub has_vram: bool,
    pub va_bits: u32,
    pub min_alignment: u64,
}

pub trait Gpu: Sized {
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
    type Cmd: ?Sized;

    fn open_first() -> Result<Self>;
    fn info(&self) -> DeviceInfo;
    fn driver_info(&self) -> &Self::Info;
    fn bo(&self, desc: BoDesc) -> Result<Self::Bo<'_>>;
    fn import(&self, fd: BorrowedFd<'_>) -> Result<Self::Bo<'_>>;
    fn vm(&self) -> Result<Self::Vm<'_>>;
    fn queue<'d>(&'d self, vm: &Self::Vm<'d>, engine: Engine) -> Result<Self::Queue<'d>>;
    fn fence(&self, signaled: bool) -> Result<Self::Fence<'_>>;
}

pub trait Bo {
    fn size(&self) -> u64;
    fn bytes(&mut self) -> Result<&mut [u8]>;
    fn export(&self) -> Result<OwnedFd>;
}

pub trait Vm<'d, G: Gpu + 'd> {
    fn bind(&self, bo: &G::Bo<'d>, offset: u64, addr: u64, range: u64, mem: MemType, done: &[&G::Fence<'d>]) -> Result<()>;
    fn unbind(&self, addr: u64, range: u64, done: &[&G::Fence<'d>]) -> Result<()>;
}

pub trait Queue<'d, G: Gpu + 'd> {
    fn submit(&self, cmd: &G::Cmd, wait: &[&G::Fence<'d>], signal: &[&G::Fence<'d>]) -> Result<()>;
    fn fault(&self, done: &G::Fence<'d>) -> Result<Option<u64>>;
}

pub trait Fence {
    fn wait(&self, timeout: Duration) -> Result<bool>;
    fn export(&self) -> Result<OwnedFd>;
}
