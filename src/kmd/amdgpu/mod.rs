mod sys;

use std::cell::RefCell;
use std::io;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::path::Path;

use crate::{BoDesc, Caching, DeviceInfo, Map, MemType, Node, Result, Syncobj};

pub use sys::{AMDGPU_FAMILY_GC_10_3_6, drm_amdgpu_info_device as AmdgpuInfo};

/// AMD's vendor id
const AMD: u16 = 0x1002;
/// Every buffer object on this alignment (a GPU page)
const ALIGN: u64 = 4096;

fn ioctl<T>(node: &Node, request: rustix::ioctl::Opcode, arg: &mut T) -> Result<()> {
    unsafe { node.ioctl(request, arg) }
}

/// An `amdgpu` render node: its device's facts, and the buffer objects alive on it (each submission's residency list)
pub struct Amdgpu {
    node: Node,
    info: AmdgpuInfo,
    resident: RefCell<Vec<u32>>,
}

impl Amdgpu {
    pub fn open(path: &Path) -> Result<Amdgpu> {
        Amdgpu::try_from(Node::open_checked(path, "amdgpu")?)
    }

    pub fn path(&self) -> &Path {
        self.node.path()
    }

    fn adopt(&self, handle: u32, size: u64) -> AmdgpuBo<'_> {
        self.resident.borrow_mut().push(handle);
        AmdgpuBo { amd: self, handle, size, map: None }
    }
}

impl TryFrom<Node> for Amdgpu {
    type Error = io::Error;

    fn try_from(node: Node) -> Result<Amdgpu> {
        let mut info = AmdgpuInfo::default();
        let mut q = sys::drm_amdgpu_info {
            return_pointer: &mut info as *mut AmdgpuInfo as u64,
            return_size: size_of::<AmdgpuInfo>() as u32,
            query: sys::AMDGPU_INFO_DEV_INFO,
            ..Default::default()
        };
        ioctl(&node, sys::DRM_IOCTL_AMDGPU_INFO, &mut q)?;
        Ok(Amdgpu { node, info, resident: RefCell::new(Vec::new()) })
    }
}

/// A command buffer an `amdgpu` queue runs: an indirect buffer's address and its dwords
#[derive(Clone, Copy, Debug)]
pub struct AmdgpuIb {
    pub addr: u64,
    pub dwords: u32,
}

impl Amdgpu {

    pub fn open_first() -> Result<Amdgpu> {
        Amdgpu::try_from(Node::open_driver("amdgpu")?)
    }

    pub fn info(&self) -> DeviceInfo {
        DeviceInfo {
            vendor_id: AMD,
            device_id: self.info.device_id as u16,
            revision: self.info.pci_rev as u8,
            has_vram: false,
            va_bits: 48,
            min_alignment: u64::from(self.info.virtual_address_alignment).max(ALIGN),
        }
    }

    pub fn driver_info(&self) -> &AmdgpuInfo {
        &self.info
    }

    pub fn bo(&self, desc: BoDesc) -> Result<AmdgpuBo<'_>> {
        let size = desc.size.next_multiple_of(ALIGN);
        let mut c = sys::drm_amdgpu_gem_create {
            bo_size: size,
            alignment: ALIGN,
            domains: sys::AMDGPU_GEM_DOMAIN_GTT,
            domain_flags: match desc.caching {
                Caching::WriteCombined => sys::AMDGPU_GEM_CREATE_CPU_GTT_USWC,
                Caching::WriteBack => 0,
            } | sys::AMDGPU_GEM_CREATE_EXPLICIT_SYNC,
        };
        ioctl(&self.node, sys::DRM_IOCTL_AMDGPU_GEM_CREATE, &mut c)?;
        Ok(self.adopt(c.bo_size as u32, size))
    }

    pub fn import(&self, fd: BorrowedFd<'_>) -> Result<AmdgpuBo<'_>> {
        let (handle, size) = self.node.import(fd)?;
        Ok(self.adopt(handle, size))
    }

    pub fn vm(&self) -> Result<AmdgpuVm<'_>> {
        Ok(AmdgpuVm { amd: self })
    }

    pub fn queue<'d>(&'d self, _vm: &AmdgpuVm<'d>) -> Result<AmdgpuQueue<'d>> {
        let mut c = sys::drm_amdgpu_ctx { op_or_ctx: sys::AMDGPU_CTX_OP_ALLOC_CTX, ..Default::default() };
        ioctl(&self.node, sys::DRM_IOCTL_AMDGPU_CTX, &mut c)?;
        Ok(AmdgpuQueue { amd: self, ctx: c.op_or_ctx })
    }

    pub fn fence(&self, signaled: bool) -> Result<Syncobj<'_>> {
        self.node.syncobj(signaled)
    }
}

pub struct AmdgpuBo<'d> {
    amd: &'d Amdgpu,
    handle: u32,
    size: u64,
    map: Option<Map>,
}

impl crate::Bo for AmdgpuBo<'_> {
    fn size(&self) -> u64 {
        self.size
    }

    fn bytes(&mut self) -> Result<&mut [u8]> {
        if self.map.is_none() {
            let mut m = sys::drm_amdgpu_gem_mmap { handle_or_offset: u64::from(self.handle) };
            ioctl(&self.amd.node, sys::DRM_IOCTL_AMDGPU_GEM_MMAP, &mut m)?;
            self.map = Some(self.amd.node.map(m.handle_or_offset, self.size as usize)?);
        }
        Ok(self.map.as_mut().map(Map::bytes).unwrap_or_default())
    }

    fn export(&self) -> Result<OwnedFd> {
        self.amd.node.export(self.handle)
    }
}

impl Drop for AmdgpuBo<'_> {
    fn drop(&mut self) {
        self.map = None;
        self.amd.resident.borrow_mut().retain(|h| *h != self.handle);
        let _ = self.amd.node.close(self.handle);
    }
}

/// The node's address space (amdgpu has one per open file)
pub struct AmdgpuVm<'d> {
    amd: &'d Amdgpu,
}

impl AmdgpuVm<'_> {
    fn va(&self, mut v: sys::drm_amdgpu_gem_va, done: &[&Syncobj<'_>]) -> Result<()> {
        ioctl(&self.amd.node, sys::DRM_IOCTL_AMDGPU_GEM_VA, &mut v)?;
        done.iter().try_for_each(|f| f.signal())
    }
}

impl<'d> AmdgpuVm<'d> {
    pub fn bind(&self, bo: &AmdgpuBo<'d>, offset: u64, addr: u64, range: u64, mem: MemType, done: &[&Syncobj<'d>]) -> Result<()> {
        let mtype = match mem {
            MemType::Coherent | MemType::Streaming | MemType::Compressed => sys::AMDGPU_VM_MTYPE_DEFAULT,
            MemType::Scanout => sys::AMDGPU_VM_MTYPE_UC,
        };
        let v = sys::drm_amdgpu_gem_va {
            handle: bo.handle,
            operation: sys::AMDGPU_VA_OP_MAP,
            flags: sys::AMDGPU_VM_PAGE_READABLE | sys::AMDGPU_VM_PAGE_WRITEABLE | sys::AMDGPU_VM_PAGE_EXECUTABLE | mtype,
            va_address: addr,
            offset_in_bo: offset,
            map_size: range.next_multiple_of(ALIGN),
            ..Default::default()
        };
        self.va(v, done)
    }

    pub fn unbind(&self, addr: u64, range: u64, done: &[&Syncobj<'d>]) -> Result<()> {
        let v = sys::drm_amdgpu_gem_va { operation: sys::AMDGPU_VA_OP_CLEAR, va_address: addr, map_size: range.next_multiple_of(ALIGN), ..Default::default() };
        self.va(v, done)
    }
}

/// A context on one of the GPU's rings
pub struct AmdgpuQueue<'d> {
    amd: &'d Amdgpu,
    ctx: u32,
}

impl<'d> AmdgpuQueue<'d> {
    pub fn submit(&self, ib: AmdgpuIb, wait: &[&Syncobj<'d>], signal: &[&Syncobj<'d>]) -> Result<()> {
        let resident: Vec<sys::drm_amdgpu_bo_list_entry> =
            self.amd.resident.borrow().iter().map(|&h| sys::drm_amdgpu_bo_list_entry { bo_handle: h, bo_priority: 0 }).collect();
        let list = sys::drm_amdgpu_bo_list_in {
            operation: u32::MAX,
            list_handle: u32::MAX,
            bo_number: resident.len() as u32,
            bo_info_size: size_of::<sys::drm_amdgpu_bo_list_entry>() as u32,
            bo_info_ptr: resident.as_ptr() as u64,
        };
        let chunk_ib = sys::drm_amdgpu_cs_chunk_ib { va_start: ib.addr, ib_bytes: 4 * ib.dwords, ip_type: sys::AMDGPU_HW_IP_GFX, ..Default::default() };
        let sems = |fs: &[&Syncobj<'d>]| -> Vec<u32> { fs.iter().map(|f| f.handle()).collect() };
        let (waits, signals) = (sems(wait), sems(signal));
        let dw = |n: usize| (n / 4) as u32;
        let mut chunks = vec![
            sys::drm_amdgpu_cs_chunk { chunk_id: sys::AMDGPU_CHUNK_ID_IB, length_dw: dw(size_of_val(&chunk_ib)), chunk_data: &chunk_ib as *const _ as u64 },
            sys::drm_amdgpu_cs_chunk { chunk_id: sys::AMDGPU_CHUNK_ID_BO_HANDLES, length_dw: dw(size_of_val(&list)), chunk_data: &list as *const _ as u64 },
        ];
        if !waits.is_empty() {
            chunks.push(sys::drm_amdgpu_cs_chunk {
                chunk_id: sys::AMDGPU_CHUNK_ID_SYNCOBJ_IN,
                length_dw: waits.len() as u32,
                chunk_data: waits.as_ptr() as u64,
            });
        }
        if !signals.is_empty() {
            chunks.push(sys::drm_amdgpu_cs_chunk {
                chunk_id: sys::AMDGPU_CHUNK_ID_SYNCOBJ_OUT,
                length_dw: signals.len() as u32,
                chunk_data: signals.as_ptr() as u64,
            });
        }
        let pointers: Vec<u64> = chunks.iter().map(|c| c as *const _ as u64).collect();
        let mut cs = sys::drm_amdgpu_cs { ctx_id: self.ctx, num_chunks: pointers.len() as u32, chunks: pointers.as_ptr() as u64, ..Default::default() };
        ioctl(&self.amd.node, sys::DRM_IOCTL_AMDGPU_CS, &mut cs)
    }

    pub fn fault(&self) -> Result<Option<u64>> {
        let mut c = sys::drm_amdgpu_ctx { op_or_ctx: sys::AMDGPU_CTX_OP_QUERY_STATE2, ctx_id: self.ctx, ..Default::default() };
        ioctl(&self.amd.node, sys::DRM_IOCTL_AMDGPU_CTX, &mut c)?;
        let flags = u64::from(c.op_or_ctx) | u64::from(c.flags) << 32;
        Ok((flags & sys::AMDGPU_CTX_QUERY2_FLAGS_GUILTY != 0).then_some(flags))
    }
}

impl Drop for AmdgpuQueue<'_> {
    fn drop(&mut self) {
        let mut c = sys::drm_amdgpu_ctx { op_or_ctx: sys::AMDGPU_CTX_OP_FREE_CTX, ctx_id: self.ctx, ..Default::default() };
        let _ = ioctl(&self.amd.node, sys::DRM_IOCTL_AMDGPU_CTX, &mut c);
    }
}
