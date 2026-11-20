use std::io;
use std::os::fd::OwnedFd;

pub type Result<T> = io::Result<T>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caching {
    WriteBack,
    WriteCombined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemType {
    Coherent,
    Streaming,
    Scanout,
    Compressed,
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

pub trait Bo {
    fn size(&self) -> u64;
    fn bytes(&mut self) -> Result<&mut [u8]>;
    fn export(&self) -> Result<OwnedFd>;
}
