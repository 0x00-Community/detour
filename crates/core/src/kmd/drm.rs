#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_ulong, c_void};
use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::{Fence, Result};

pub struct Ioctl;

impl Ioctl {
    pub const COMMAND_BASE: c_ulong = 0x40;
    const WRITE: c_ulong = 1;
    const READ: c_ulong = 2;

    const fn ioc<T>(dir: c_ulong, ty: u8, nr: c_ulong) -> c_ulong {
        (dir << 30) | ((size_of::<T>() as c_ulong) << 16) | ((ty as c_ulong) << 8) | nr
    }
    pub const fn w<T>(nr: c_ulong) -> c_ulong {
        Self::ioc::<T>(Self::WRITE, b'd', nr)
    }
    pub const fn wr<T>(nr: c_ulong) -> c_ulong {
        Self::ioc::<T>(Self::READ | Self::WRITE, b'd', nr)
    }
}

#[repr(C)]
struct drm_version {
    version_major: c_int,
    version_minor: c_int,
    version_patchlevel: c_int,
    name_len: usize,
    name: *mut c_char,
    date_len: usize,
    date: *mut c_char,
    desc_len: usize,
    desc: *mut c_char,
}

#[repr(C)]
#[derive(Default)]
struct drm_gem_close {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct drm_prime_handle {
    handle: u32,
    flags: u32,
    fd: i32,
}

#[repr(C)]
#[derive(Default)]
struct drm_syncobj_create {
    handle: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Default)]
struct drm_syncobj_destroy {
    handle: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct drm_syncobj_handle {
    handle: u32,
    flags: u32,
    fd: i32,
    pad: u32,
    point: u64,
}

#[repr(C)]
#[derive(Default)]
struct drm_syncobj_wait {
    handles: u64,
    timeout_nsec: i64,
    count_handles: u32,
    flags: u32,
    first_signaled: u32,
    pad: u32,
    deadline_nsec: u64,
}

#[repr(C)]
#[derive(Default)]
struct drm_syncobj_array {
    handles: u64,
    count_handles: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Default)]
struct sync_file_info {
    name: [u8; 32],
    status: i32,
    flags: u32,
    num_fences: u32,
    pad: u32,
    sync_fence_info: u64,
}

const DRM_SYNCOBJ_CREATE_SIGNALED: u32 = 1 << 0;
const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL: u32 = 1 << 0;
const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 1;
const DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE: u32 = 1 << 0;
const DRM_CLOEXEC: u32 = libc::O_CLOEXEC as u32;
const DRM_RDWR: u32 = libc::O_RDWR as u32;

const DRM_IOCTL_VERSION: c_ulong = Ioctl::wr::<drm_version>(0x00);
const DRM_IOCTL_GEM_CLOSE: c_ulong = Ioctl::w::<drm_gem_close>(0x09);
const DRM_IOCTL_PRIME_HANDLE_TO_FD: c_ulong = Ioctl::wr::<drm_prime_handle>(0x2d);
const DRM_IOCTL_PRIME_FD_TO_HANDLE: c_ulong = Ioctl::wr::<drm_prime_handle>(0x2e);
const DRM_IOCTL_SYNCOBJ_CREATE: c_ulong = Ioctl::wr::<drm_syncobj_create>(0xbf);
const DRM_IOCTL_SYNCOBJ_DESTROY: c_ulong = Ioctl::wr::<drm_syncobj_destroy>(0xc0);
const DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD: c_ulong = Ioctl::wr::<drm_syncobj_handle>(0xc1);
const DRM_IOCTL_SYNCOBJ_WAIT: c_ulong = Ioctl::wr::<drm_syncobj_wait>(0xc3);
const DRM_IOCTL_SYNCOBJ_SIGNAL: c_ulong = Ioctl::wr::<drm_syncobj_array>(0xc5);
const SYNC_IOC_FILE_INFO: c_ulong = Ioctl::ioc::<sync_file_info>(Ioctl::READ | Ioctl::WRITE, b'>', 4);

pub struct Node {
    fd: OwnedFd,
    path: PathBuf,
}

impl Node {
    fn open(path: &Path) -> Result<Node> {
        let file = OpenOptions::new().read(true).write(true).custom_flags(libc::O_CLOEXEC).open(path)?;
        Ok(Node { fd: file.into(), path: path.to_owned() })
    }

    pub fn open_driver(driver: &str) -> Result<Node> {
        let mut nodes: Vec<PathBuf> = std::fs::read_dir("/dev/dri")?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("renderD")))
            .collect();
        nodes.sort();
        nodes
            .iter()
            .filter_map(|p| Node::open(p).ok())
            .find(|n| n.driver().is_ok_and(|d| d == driver))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no {driver} render node in /dev/dri")))
    }

    pub fn open_checked(path: &Path, driver: &str) -> Result<Node> {
        let node = Node::open(path)?;
        match node.driver()? {
            d if d == driver => Ok(node),
            d => Err(io::Error::new(io::ErrorKind::Unsupported, format!("{}: driver {d}, not {driver}", path.display()))),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub unsafe fn ioctl<T>(&self, request: c_ulong, arg: &mut T) -> Result<()> {
        loop {
            let r = unsafe { libc::ioctl(self.fd.as_raw_fd(), request as _, arg as *mut T as *mut c_void) };
            if r != -1 {
                return Ok(());
            }
            let e = io::Error::last_os_error();
            if !matches!(e.raw_os_error(), Some(libc::EINTR | libc::EAGAIN)) {
                return Err(e);
            }
        }
    }

    pub fn driver(&self) -> Result<String> {
        let mut name = [0u8; 64];
        let mut v = drm_version {
            version_major: 0,
            version_minor: 0,
            version_patchlevel: 0,
            name_len: name.len(),
            name: name.as_mut_ptr().cast(),
            date_len: 0,
            date: std::ptr::null_mut(),
            desc_len: 0,
            desc: std::ptr::null_mut(),
        };
        unsafe { self.ioctl(DRM_IOCTL_VERSION, &mut v) }?;
        Ok(String::from_utf8_lossy(&name[..v.name_len.min(name.len())]).into_owned())
    }

    pub fn close(&self, handle: u32) -> Result<()> {
        unsafe { self.ioctl(DRM_IOCTL_GEM_CLOSE, &mut drm_gem_close { handle, pad: 0 }) }
    }

    pub fn export(&self, handle: u32) -> Result<OwnedFd> {
        let mut p = drm_prime_handle { handle, flags: DRM_CLOEXEC | DRM_RDWR, fd: -1 };
        unsafe { self.ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &mut p) }?;
        Ok(unsafe { OwnedFd::from_raw_fd(p.fd) })
    }

    pub fn import(&self, fd: BorrowedFd<'_>) -> Result<(u32, u64)> {
        let end = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
        if end < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut p = drm_prime_handle { handle: 0, flags: 0, fd: fd.as_raw_fd() };
        unsafe { self.ioctl(DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut p) }?;
        Ok((p.handle, end as u64))
    }

    pub fn map(&self, offset: u64, len: usize) -> Result<Map> {
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, self.fd.as_raw_fd(), offset as libc::off_t)
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Map { ptr: p.cast(), len })
    }

    pub fn syncobj(&self, signaled: bool) -> Result<Syncobj<'_>> {
        let mut c = drm_syncobj_create { handle: 0, flags: if signaled { DRM_SYNCOBJ_CREATE_SIGNALED } else { 0 } };
        unsafe { self.ioctl(DRM_IOCTL_SYNCOBJ_CREATE, &mut c) }?;
        Ok(Syncobj { node: self, handle: c.handle })
    }
}

pub struct Map {
    ptr: *mut u8,
    len: usize,
}

impl Map {
    pub fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

pub struct Syncobj<'n> {
    node: &'n Node,
    handle: u32,
}

impl Syncobj<'_> {
    pub fn handle(&self) -> u32 {
        self.handle
    }

    pub fn signal(&self) -> Result<()> {
        let handles = [self.handle];
        let mut a = drm_syncobj_array { handles: handles.as_ptr() as u64, count_handles: 1, pad: 0 };
        unsafe { self.node.ioctl(DRM_IOCTL_SYNCOBJ_SIGNAL, &mut a) }
    }

    pub fn status(&self) -> Result<i32> {
        let file = self.export()?;
        let mut info = sync_file_info::default();
        let r = unsafe { libc::ioctl(file.as_raw_fd(), SYNC_IOC_FILE_INFO as _, &mut info as *mut sync_file_info as *mut c_void) };
        if r == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(info.status)
    }
}

impl Fence for Syncobj<'_> {
    fn wait(&self, timeout: Duration) -> Result<bool> {
        let mut now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
        let deadline = (now.tv_sec * 1_000_000_000 + now.tv_nsec).saturating_add(timeout.as_nanos() as i64);
        let handles = [self.handle];
        let mut w = drm_syncobj_wait {
            handles: handles.as_ptr() as u64,
            timeout_nsec: deadline,
            count_handles: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
            ..Default::default()
        };
        match unsafe { self.node.ioctl(DRM_IOCTL_SYNCOBJ_WAIT, &mut w) } {
            Ok(()) => Ok(true),
            Err(e) if e.raw_os_error() == Some(libc::ETIME) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn export(&self) -> Result<OwnedFd> {
        let mut h = drm_syncobj_handle { handle: self.handle, flags: DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE, fd: -1, ..Default::default() };
        unsafe { self.node.ioctl(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &mut h) }?;
        Ok(unsafe { OwnedFd::from_raw_fd(h.fd) })
    }
}

impl Drop for Syncobj<'_> {
    fn drop(&mut self) {
        let mut d = drm_syncobj_destroy { handle: self.handle, pad: 0 };
        let _ = unsafe { self.node.ioctl(DRM_IOCTL_SYNCOBJ_DESTROY, &mut d) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_kernel() {
        assert_eq!(size_of::<drm_version>(), 64);
        assert_eq!(size_of::<drm_syncobj_wait>(), 40);
        assert_eq!(size_of::<sync_file_info>(), 56);
        assert_eq!(DRM_IOCTL_VERSION, 0xc040_6400);
        assert_eq!(DRM_IOCTL_SYNCOBJ_WAIT, 0xc028_64c3);
        assert_eq!(DRM_IOCTL_SYNCOBJ_SIGNAL, 0xc010_64c5);
        assert_eq!(SYNC_IOC_FILE_INFO, 0xc038_3e04);
    }
}
