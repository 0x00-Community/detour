mod sys;

use std::cell::{Cell, RefCell};
use std::ffi::c_ulong;
use std::io;
use std::ops::Range;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::path::Path;

use crate::base::{BoDesc, Caching, DeviceInfo, Engine, Gpu, Map, MemType, Node, Result, Syncobj};

const AMD: u16 = 0x1002;

trait AmdNode {
    fn amdgpu<T>(&self, request: c_ulong, arg: &mut T) -> Result<()>;
    fn query<T: Default>(&self, query: u32, args: [u32; 4]) -> Result<T>;
}

impl AmdNode for Node {
    fn amdgpu<T>(&self, request: c_ulong, arg: &mut T) -> Result<()> {
        unsafe { self.ioctl(request, arg) }
    }

    fn query<T: Default>(&self, query: u32, args: [u32; 4]) -> Result<T> {
        let mut out = T::default();
        let mut i = sys::drm_amdgpu_info { return_pointer: &mut out as *mut T as u64, return_size: size_of::<T>() as u32, query, args };
        self.amdgpu(sys::DRM_IOCTL_AMDGPU_INFO, &mut i)?;
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HwIp {
    pub version: (u32, u32),
    pub rings: u32,
    pub ib_start_alignment: u32,
    pub ib_size_alignment: u32,
}

impl From<sys::drm_amdgpu_info_hw_ip> for HwIp {
    fn from(h: sys::drm_amdgpu_info_hw_ip) -> Self {
        HwIp {
            version: (h.hw_ip_version_major, h.hw_ip_version_minor),
            rings: h.available_rings,
            ib_start_alignment: h.ib_start_alignment,
            ib_size_alignment: h.ib_size_alignment,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AmdInfo {
    pub common: DeviceInfo,
    pub family: u32,
    pub chip_rev: u32,
    pub external_rev: u32,
    pub apu: bool,
    pub shader_engines: u32,
    pub shader_arrays_per_engine: u32,
    pub cus: u32,
    pub wave_size: u32,
    pub max_engine_clock_khz: u64,
    pub va: Range<u64>,
    pub high_va: Range<u64>,
    pub gfx: HwIp,
    pub compute: HwIp,
    pub dma: HwIp,
}

impl AmdInfo {
    fn ip(&self, engine: Engine) -> (u32, &HwIp) {
        match engine {
            Engine::Render => (sys::AMDGPU_HW_IP_GFX, &self.gfx),
            Engine::Compute => (sys::AMDGPU_HW_IP_COMPUTE, &self.compute),
            Engine::Copy => (sys::AMDGPU_HW_IP_DMA, &self.dma),
        }
    }
}

impl TryFrom<&Node> for AmdInfo {
    type Error = io::Error;

    fn try_from(node: &Node) -> Result<AmdInfo> {
        let d: sys::drm_amdgpu_info_device = node.query(sys::AMDGPU_INFO_DEV_INFO, [0; 4])?;
        let ip = |ty: u32| node.query::<sys::drm_amdgpu_info_hw_ip>(sys::AMDGPU_INFO_HW_IP_INFO, [ty, 0, 0, 0]).map(HwIp::from);
        Ok(AmdInfo {
            common: DeviceInfo {
                vendor_id: AMD,
                device_id: d.device_id as u16,
                revision: d.pci_rev as u8,
                has_vram: d.ids_flags & sys::AMDGPU_IDS_FLAGS_FUSION == 0,
                va_bits: 64 - d.virtual_address_max.saturating_sub(1).leading_zeros(),
                min_alignment: u64::from(d.virtual_address_alignment).max(4096),
            },
            family: d.family,
            chip_rev: d.chip_rev,
            external_rev: d.external_rev,
            apu: d.ids_flags & sys::AMDGPU_IDS_FLAGS_FUSION != 0,
            shader_engines: d.num_shader_engines,
            shader_arrays_per_engine: d.num_shader_arrays_per_engine,
            cus: d.cu_active_number,
            wave_size: d.wave_front_size,
            max_engine_clock_khz: d.max_engine_clock,
            va: d.virtual_address_offset..d.virtual_address_max,
            high_va: d.high_va_offset..d.high_va_max,
            gfx: ip(sys::AMDGPU_HW_IP_GFX)?,
            compute: ip(sys::AMDGPU_HW_IP_COMPUTE)?,
            dma: ip(sys::AMDGPU_HW_IP_DMA)?,
        })
    }
}

pub struct Amdgpu {
    node: Node,
    info: AmdInfo,
    vm_taken: Cell<bool>,
    mapped: RefCell<Vec<(u64, u32)>>,
}

impl Amdgpu {
    pub fn open(path: &Path) -> Result<Amdgpu> {
        Amdgpu::try_from(Node::open_checked(path, "amdgpu")?)
    }

    pub fn path(&self) -> &Path {
        self.node.path()
    }

    fn gem_va(&self, mut va: sys::drm_amdgpu_gem_va, done: &[&Syncobj<'_>]) -> Result<()> {
        self.node.amdgpu(sys::DRM_IOCTL_AMDGPU_GEM_VA, &mut va)?;
        done.iter().try_for_each(|f| f.signal())
    }
}

impl TryFrom<Node> for Amdgpu {
    type Error = io::Error;

    fn try_from(node: Node) -> Result<Amdgpu> {
        let info = AmdInfo::try_from(&node)?;
        Ok(Amdgpu { node, info, vm_taken: Cell::new(false), mapped: RefCell::new(Vec::new()) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ib {
    pub addr: u64,
    pub dwords: u32,
}

impl Gpu for Amdgpu {
    type Info = AmdInfo;
    type Bo<'d> = AmdBo<'d>;
    type Vm<'d> = AmdVm<'d>;
    type Queue<'d> = AmdQueue<'d>;
    type Fence<'d> = Syncobj<'d>;
    type Cmd = [Ib];

    fn open_first() -> Result<Amdgpu> {
        Amdgpu::try_from(Node::open_driver("amdgpu")?)
    }

    fn info(&self) -> DeviceInfo {
        self.info.common
    }

    fn driver_info(&self) -> &AmdInfo {
        &self.info
    }

    fn bo(&self, desc: BoDesc) -> Result<AmdBo<'_>> {
        let size = desc.size.next_multiple_of(self.info.common.min_alignment);
        let wc = match desc.caching {
            Caching::WriteBack => 0,
            Caching::WriteCombined => sys::AMDGPU_GEM_CREATE_CPU_GTT_USWC,
        };
        let mut c = sys::drm_amdgpu_gem_create {
            r#in: sys::drm_amdgpu_gem_create_in {
                bo_size: size,
                alignment: self.info.common.min_alignment,
                domains: sys::AMDGPU_GEM_DOMAIN_GTT,
                domain_flags: sys::AMDGPU_GEM_CREATE_EXPLICIT_SYNC | wc,
            },
        };
        self.node.amdgpu(sys::DRM_IOCTL_AMDGPU_GEM_CREATE, &mut c)?;
        let handle = unsafe { c.out.handle };
        Ok(AmdBo { amd: self, handle, size, map: None })
    }

    fn import(&self, fd: BorrowedFd<'_>) -> Result<AmdBo<'_>> {
        let (handle, size) = self.node.import(fd)?;
        Ok(AmdBo { amd: self, handle, size, map: None })
    }

    fn vm(&self) -> Result<AmdVm<'_>> {
        if self.vm_taken.replace(true) {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "amdgpu has one VM per open file, already given"));
        }
        Ok(AmdVm { amd: self })
    }

    fn queue<'d>(&'d self, _vm: &AmdVm<'d>, engine: Engine) -> Result<AmdQueue<'d>> {
        let (ip, hw) = self.info.ip(engine);
        if hw.rings & 1 == 0 {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("no {engine:?} ring")));
        }
        let mut c = sys::drm_amdgpu_ctx {
            r#in: sys::drm_amdgpu_ctx_in { op: sys::AMDGPU_CTX_OP_ALLOC_CTX, priority: sys::AMDGPU_CTX_PRIORITY_NORMAL, ..Default::default() },
        };
        self.node.amdgpu(sys::DRM_IOCTL_AMDGPU_CTX, &mut c)?;
        let ctx = unsafe { c.out.alloc.ctx_id };
        Ok(AmdQueue { amd: self, ctx, ip })
    }

    fn fence(&self, signaled: bool) -> Result<Syncobj<'_>> {
        self.node.syncobj(signaled)
    }
}

pub struct AmdBo<'d> {
    amd: &'d Amdgpu,
    handle: u32,
    size: u64,
    map: Option<Map>,
}

impl crate::base::Bo for AmdBo<'_> {
    fn size(&self) -> u64 {
        self.size
    }

    fn bytes(&mut self) -> Result<&mut [u8]> {
        if self.map.is_none() {
            let mut m = sys::drm_amdgpu_gem_mmap { r#in: sys::drm_amdgpu_gem_mmap_in { handle: self.handle, _pad: 0 } };
            self.amd.node.amdgpu(sys::DRM_IOCTL_AMDGPU_GEM_MMAP, &mut m)?;
            let offset = unsafe { m.out.addr_ptr };
            self.map = Some(self.amd.node.map(offset, self.size as usize)?);
        }
        Ok(self.map.as_mut().map(Map::bytes).unwrap_or_default())
    }

    fn export(&self) -> Result<OwnedFd> {
        self.amd.node.export(self.handle)
    }
}

impl Drop for AmdBo<'_> {
    fn drop(&mut self) {
        self.map = None;
        self.amd.mapped.borrow_mut().retain(|&(_, h)| h != self.handle);
        let _ = self.amd.node.close(self.handle);
    }
}

pub struct AmdVm<'d> {
    amd: &'d Amdgpu,
}

impl<'d> crate::base::Vm<'d, Amdgpu> for AmdVm<'d> {
    fn bind(&self, bo: &AmdBo<'d>, offset: u64, addr: u64, range: u64, _mem: MemType, done: &[&Syncobj<'d>]) -> Result<()> {
        let va = sys::drm_amdgpu_gem_va {
            handle: bo.handle,
            operation: sys::AMDGPU_VA_OP_MAP,
            flags: sys::AMDGPU_VM_PAGE_READABLE | sys::AMDGPU_VM_PAGE_WRITEABLE | sys::AMDGPU_VM_PAGE_EXECUTABLE,
            va_address: addr,
            offset_in_bo: offset,
            map_size: range,
            ..Default::default()
        };
        self.amd.gem_va(va, done)?;
        self.amd.mapped.borrow_mut().push((addr, bo.handle));
        Ok(())
    }

    fn unbind(&self, addr: u64, range: u64, done: &[&Syncobj<'d>]) -> Result<()> {
        let mut mapped = self.amd.mapped.borrow_mut();
        let i = mapped
            .iter()
            .position(|&(a, _)| a == addr)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("nothing mapped at {addr:#x}")))?;
        let (_, handle) = mapped.remove(i);
        drop(mapped);
        let va = sys::drm_amdgpu_gem_va { handle, operation: sys::AMDGPU_VA_OP_UNMAP, va_address: addr, map_size: range, ..Default::default() };
        self.amd.gem_va(va, done)
    }
}

impl Drop for AmdVm<'_> {
    fn drop(&mut self) {
        self.amd.vm_taken.set(false);
    }
}

pub struct AmdQueue<'d> {
    amd: &'d Amdgpu,
    ctx: u32,
    ip: u32,
}

impl<T> From<(u32, &[T])> for sys::drm_amdgpu_cs_chunk {
    fn from((id, items): (u32, &[T])) -> Self {
        sys::drm_amdgpu_cs_chunk { chunk_id: id, length_dw: (size_of_val(items) / 4) as u32, chunk_data: items.as_ptr() as u64 }
    }
}

impl<'d> crate::base::Queue<'d, Amdgpu> for AmdQueue<'d> {
    fn submit(&self, ibs: &[Ib], wait: &[&Syncobj<'d>], signal: &[&Syncobj<'d>]) -> Result<()> {
        if ibs.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "a submission without an indirect buffer"));
        }
        let ib: Vec<sys::drm_amdgpu_cs_chunk_ib> =
            ibs.iter().map(|b| sys::drm_amdgpu_cs_chunk_ib { va_start: b.addr, ib_bytes: b.dwords * 4, ip_type: self.ip, ..Default::default() }).collect();
        let bos: Vec<sys::drm_amdgpu_bo_list_entry> =
            self.amd.mapped.borrow().iter().map(|&(_, h)| sys::drm_amdgpu_bo_list_entry { bo_handle: h, bo_priority: 0 }).collect();
        let list = [sys::drm_amdgpu_bo_list_in {
            operation: !0,
            list_handle: !0,
            bo_number: bos.len() as u32,
            bo_info_size: size_of::<sys::drm_amdgpu_bo_list_entry>() as u32,
            bo_info_ptr: bos.as_ptr() as u64,
        }];
        let sem = |fences: &[&Syncobj<'d>]| fences.iter().map(|f| sys::drm_amdgpu_cs_chunk_sem { handle: f.handle() }).collect::<Vec<_>>();
        let (wait, signal) = (sem(wait), sem(signal));
        let mut chunks: Vec<sys::drm_amdgpu_cs_chunk> = ib.chunks(1).map(|b| (sys::AMDGPU_CHUNK_ID_IB, b).into()).collect();
        chunks.push((sys::AMDGPU_CHUNK_ID_BO_HANDLES, &list[..]).into());
        if !wait.is_empty() {
            chunks.push((sys::AMDGPU_CHUNK_ID_SYNCOBJ_IN, &wait[..]).into());
        }
        if !signal.is_empty() {
            chunks.push((sys::AMDGPU_CHUNK_ID_SYNCOBJ_OUT, &signal[..]).into());
        }
        let pointers: Vec<u64> = chunks.iter().map(|c| c as *const sys::drm_amdgpu_cs_chunk as u64).collect();
        let mut cs = sys::drm_amdgpu_cs {
            r#in: sys::drm_amdgpu_cs_in { ctx_id: self.ctx, num_chunks: pointers.len() as u32, chunks: pointers.as_ptr() as u64, ..Default::default() },
        };
        self.amd.node.amdgpu(sys::DRM_IOCTL_AMDGPU_CS, &mut cs)
    }

    fn fault(&self, done: &Syncobj<'d>) -> Result<Option<u64>> {
        let mut c = sys::drm_amdgpu_ctx { r#in: sys::drm_amdgpu_ctx_in { op: sys::AMDGPU_CTX_OP_QUERY_STATE2, ctx_id: self.ctx, ..Default::default() } };
        self.amd.node.amdgpu(sys::DRM_IOCTL_AMDGPU_CTX, &mut c)?;
        let lost = sys::AMDGPU_CTX_QUERY2_FLAGS_RESET | sys::AMDGPU_CTX_QUERY2_FLAGS_GUILTY | sys::AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST;
        let flags = unsafe { c.out.state.flags } & lost;
        if flags != 0 {
            return Ok(Some(flags));
        }
        let status = done.status()?;
        Ok((status < 0).then(|| status.unsigned_abs().into()))
    }
}

impl Drop for AmdQueue<'_> {
    fn drop(&mut self) {
        let mut c = sys::drm_amdgpu_ctx { r#in: sys::drm_amdgpu_ctx_in { op: sys::AMDGPU_CTX_OP_FREE_CTX, ctx_id: self.ctx, ..Default::default() } };
        let _ = self.amd.node.amdgpu(sys::DRM_IOCTL_AMDGPU_CTX, &mut c);
    }
}
