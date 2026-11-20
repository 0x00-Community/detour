#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};
use std::fs::OpenOptions;
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rustix::fs::{OFlags, SeekFrom};
use rustix::io::Errno;
use rustix::ioctl::{Ioctl, IoctlOutput, Opcode, opcode};
use rustix::mm::{MapFlags, ProtFlags};
use rustix::time::ClockId;

use super::{Fence, Result};

pub const DRM_IOCTL_BASE: u8 = b'd';
pub const DRM_COMMAND_BASE: u8 = 0x40;

struct Call<'a, T> {
    request: Opcode,
    arg: &'a mut T,
}

unsafe impl<T> Ioctl for Call<'_, T> {
    type Output = ();
    const IS_MUTATING: bool = true;

    fn opcode(&self) -> Opcode {
        self.request
    }

    fn as_ptr(&mut self) -> *mut c_void {
        (self.arg as *mut T).cast()
    }

    unsafe fn output_from_ptr(_: IoctlOutput, _: *mut c_void) -> rustix::io::Result<()> {
        Ok(())
    }
}

unsafe fn call<T>(fd: BorrowedFd<'_>, request: Opcode, arg: &mut T) -> rustix::io::Result<()> {
    loop {
        match unsafe { rustix::ioctl::ioctl(fd, Call { request, arg: &mut *arg }) } {
            Err(Errno::INTR | Errno::AGAIN) => continue,
            r => return r,
        }
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

const DRM_SYNCOBJ_CREATE_SIGNALED: u32 = 1 << 0;
const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL: u32 = 1 << 0;
const DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT: u32 = 1 << 1;
const DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE: u32 = 1 << 0;
const DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE: u32 = 1 << 0;
const DRM_CLOEXEC: u32 = OFlags::CLOEXEC.bits();
const DRM_RDWR: u32 = OFlags::RDWR.bits();

const DRM_IOCTL_VERSION: Opcode = opcode::read_write::<drm_version>(DRM_IOCTL_BASE, 0x00);
const DRM_IOCTL_GEM_CLOSE: Opcode = opcode::write::<drm_gem_close>(DRM_IOCTL_BASE, 0x09);
const DRM_IOCTL_PRIME_HANDLE_TO_FD: Opcode = opcode::read_write::<drm_prime_handle>(DRM_IOCTL_BASE, 0x2d);
const DRM_IOCTL_PRIME_FD_TO_HANDLE: Opcode = opcode::read_write::<drm_prime_handle>(DRM_IOCTL_BASE, 0x2e);
const DRM_IOCTL_SYNCOBJ_CREATE: Opcode = opcode::read_write::<drm_syncobj_create>(DRM_IOCTL_BASE, 0xbf);
const DRM_IOCTL_SYNCOBJ_DESTROY: Opcode = opcode::read_write::<drm_syncobj_destroy>(DRM_IOCTL_BASE, 0xc0);
const DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD: Opcode = opcode::read_write::<drm_syncobj_handle>(DRM_IOCTL_BASE, 0xc1);
const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE: Opcode = opcode::read_write::<drm_syncobj_handle>(DRM_IOCTL_BASE, 0xc2);
const DRM_IOCTL_SYNCOBJ_WAIT: Opcode = opcode::read_write::<drm_syncobj_wait>(DRM_IOCTL_BASE, 0xc3);
const DRM_IOCTL_SYNCOBJ_SIGNAL: Opcode = opcode::read_write::<drm_syncobj_array>(DRM_IOCTL_BASE, 0xc5);

pub struct Node {
    fd: OwnedFd,
    path: PathBuf,
}

impl Node {
    pub fn open(path: &Path) -> Result<Node> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Node { fd: file.into(), path: path.to_owned() })
    }

    pub fn open_driver(driver: &str) -> Result<Node> {
        Node::open_any(&[driver])
    }

    /// The first render node in /dev/dri whose driver is one of `drivers`
    pub fn open_any(drivers: &[&str]) -> Result<Node> {
        Node::all(drivers)?.into_iter().next().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no {} render node in /dev/dri", drivers.join(" or "))))
    }

    /// Every render node in /dev/dri whose driver is one of `drivers`, in order
    pub fn all(drivers: &[&str]) -> Result<Vec<Node>> {
        let mut nodes: Vec<PathBuf> = std::fs::read_dir("/dev/dri")?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("renderD")))
            .collect();
        nodes.sort();
        Ok(nodes.iter().filter_map(|p| Node::open(p).ok()).filter(|n| n.driver().is_ok_and(|d| drivers.contains(&d.as_str()))).collect())
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

    /// # Safety
    /// `request` must be an ioctl of this node whose argument is a `T`, and every pointer inside `arg` valid for it
    pub unsafe fn ioctl<T>(&self, request: Opcode, arg: &mut T) -> Result<()> {
        Ok(unsafe { call(self.fd.as_fd(), request, arg) }?)
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
        let end = rustix::fs::seek(fd, SeekFrom::End(0))?;
        let mut p = drm_prime_handle { handle: 0, flags: 0, fd: fd.as_raw_fd() };
        unsafe { self.ioctl(DRM_IOCTL_PRIME_FD_TO_HANDLE, &mut p) }?;
        Ok((p.handle, end))
    }

    pub fn map(&self, offset: u64, len: usize) -> Result<Map> {
        let p = unsafe { rustix::mm::mmap(std::ptr::null_mut(), len, ProtFlags::READ | ProtFlags::WRITE, MapFlags::SHARED, &self.fd, offset) }?;
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
        let _ = unsafe { rustix::mm::munmap(self.ptr.cast(), self.len) };
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

    /// The fence of `sync_file` put in it: a wait on it waits for that
    pub fn import(&self, sync_file: BorrowedFd<'_>) -> Result<()> {
        let mut h = drm_syncobj_handle { handle: self.handle, flags: DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE, fd: sync_file.as_raw_fd(), ..Default::default() };
        unsafe { self.node.ioctl(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &mut h) }
    }
}

impl Fence for Syncobj<'_> {
    fn wait(&self, timeout: Duration) -> Result<bool> {
        let now = rustix::time::clock_gettime(ClockId::Monotonic);
        let deadline = (now.tv_sec * 1_000_000_000 + now.tv_nsec).saturating_add(timeout.as_nanos() as i64);
        let handles = [self.handle];
        let mut w = drm_syncobj_wait {
            handles: handles.as_ptr() as u64,
            timeout_nsec: deadline,
            count_handles: 1,
            flags: DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
            ..Default::default()
        };
        match unsafe { call(self.node.fd.as_fd(), DRM_IOCTL_SYNCOBJ_WAIT, &mut w) } {
            Ok(()) => Ok(true),
            Err(Errno::TIME) => Ok(false),
            Err(e) => Err(e.into()),
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
        assert_eq!(DRM_IOCTL_VERSION, 0xc040_6400);
        assert_eq!(DRM_IOCTL_SYNCOBJ_WAIT, 0xc028_64c3);
        assert_eq!(DRM_IOCTL_SYNCOBJ_SIGNAL, 0xc010_64c5);
        assert_eq!(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, 0xc018_64c2);
    }
}
