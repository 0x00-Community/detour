//! The xe kernel driver (Intel Xe2 / Xe3 GPUs: Lunar Lake, Battlemage, Panther Lake) as an [`crate::base::Gpu`], without
//! libdrm: open a render node, query the device, create buffer objects and map them, manage a GPU virtual address
//! space (VM_BIND), submit batch buffers to an exec queue and wait for them with DRM syncobjs.
//!
//! Addresses are the caller's: detour owns its whole VM and places every buffer itself (no relocations in xe).

mod sys;

use std::ffi::c_ulong;
use std::io;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::path::Path;

use crate::base::{BoDesc, Caching, DeviceInfo, Engine, Gpu, Map, MemType, Node, Result, Syncobj};

/// An engine to create exec queues on.
pub use sys::drm_xe_engine_class_instance as EngineInstance;

// ---- the render node --------------------------------------------------------------------------------------------

/// xe's ioctls on a render node.
trait XeNode {
    fn xe<T>(&self, request: c_ulong, arg: &mut T) -> Result<()>;
    /// A device query's answer (asks for the size first).
    fn query(&self, kind: u32) -> Result<Query>;
}

impl XeNode for Node {
    fn xe<T>(&self, request: c_ulong, arg: &mut T) -> Result<()> {
        // SAFETY: sys.rs builds every request from the struct it takes; callers fill in its pointers.
        unsafe { self.ioctl(request, arg) }
    }

    fn query(&self, kind: u32) -> Result<Query> {
        let mut q = sys::drm_xe_device_query { query: kind, ..Default::default() };
        self.xe(sys::DRM_IOCTL_XE_DEVICE_QUERY, &mut q)?;
        let mut data = vec![0u64; (q.size as usize).div_ceil(8)];
        q.data = data.as_mut_ptr() as u64;
        self.xe(sys::DRM_IOCTL_XE_DEVICE_QUERY, &mut q)?;
        Ok(Query(data))
    }
}

/// What a device query wrote.
struct Query(Vec<u64>);

impl Query {
    fn bytes(&self) -> &[u8] {
        // SAFETY: u64 has no padding and every byte pattern is a valid u8.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast(), self.0.len() * 8) }
    }

    /// A header struct followed by `count(header)` items.
    fn items<H: Copy, T: Copy>(&self, count: impl Fn(&H) -> usize) -> Vec<T> {
        let bytes = self.bytes();
        // SAFETY: the kernel filled the buffer with a header and its items; the reads stay inside it (checked below).
        let head: H = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast()) };
        let n = count(&head);
        let start = size_of::<H>();
        assert!(start + n * size_of::<T>() <= bytes.len(), "short device query");
        (0..n).map(|i| unsafe { std::ptr::read_unaligned(bytes[start + i * size_of::<T>()..].as_ptr().cast()) }).collect()
    }

    /// The topology masks: (header, mask bytes) back to back.
    fn topology(&self) -> Vec<Topology> {
        let bytes = self.bytes();
        let head = size_of::<sys::drm_xe_query_topology_mask>();
        let mut out = Vec::new();
        let mut at = 0;
        while at + head <= bytes.len() {
            // SAFETY: bounds checked above.
            let h: sys::drm_xe_query_topology_mask = unsafe { std::ptr::read_unaligned(bytes[at..].as_ptr().cast()) };
            at += head;
            let n = h.num_bytes as usize;
            if h.type_ == 0 || at + n > bytes.len() {
                break;
            }
            out.push(Topology { gt: h.gt_id, kind: h.type_, mask: bytes[at..at + n].to_vec() });
            at += n;
        }
        out
    }

    /// As little-endian dwords.
    fn dwords(&self) -> Vec<u32> {
        self.bytes().as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
    }
}

// ---- what the device reports ------------------------------------------------------------------------------------

/// One GT (graphics or media tile part).
#[derive(Clone, Debug)]
pub struct Gt {
    pub id: u16,
    pub media: bool,
    pub tile: u16,
    /// Timestamp frequency (Hz) of this GT's engines.
    pub reference_clock: u32,
    /// Bit mask of mem region instances close to this GT.
    pub near_mem_regions: u64,
    pub ip_version: (u16, u16, u16),
}

impl From<sys::drm_xe_gt> for Gt {
    fn from(g: sys::drm_xe_gt) -> Self {
        Gt {
            id: g.gt_id,
            media: g.type_ == sys::DRM_XE_QUERY_GT_TYPE_MEDIA,
            tile: g.tile_id,
            reference_clock: g.reference_clock,
            near_mem_regions: g.near_mem_regions,
            ip_version: (g.ip_ver_major, g.ip_ver_minor, g.ip_ver_rev),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemRegion {
    pub vram: bool,
    pub instance: u16,
    pub min_page_size: u32,
    pub total_size: u64,
    pub cpu_visible_size: u64,
}

impl From<sys::drm_xe_mem_region> for MemRegion {
    fn from(r: sys::drm_xe_mem_region) -> Self {
        MemRegion {
            vram: r.mem_class == sys::DRM_XE_MEM_REGION_CLASS_VRAM,
            instance: r.instance,
            min_page_size: r.min_page_size,
            total_size: r.total_size,
            cpu_visible_size: r.cpu_visible_size,
        }
    }
}

/// A topology mask of one GT (`DRM_XE_TOPO_*`).
#[derive(Clone, Debug)]
pub struct Topology {
    pub gt: u16,
    pub kind: u16,
    pub mask: Vec<u8>,
}

/// What only xe reports, read once at open.
#[derive(Clone, Debug)]
pub struct XeInfo {
    pub common: DeviceInfo,
    pub max_exec_queue_priority: u64,
    pub engines: Vec<EngineInstance>,
    pub gts: Vec<Gt>,
    pub mem_regions: Vec<MemRegion>,
    pub topology: Vec<Topology>,
    /// The GuC's hwconfig table: KLVs (u32 key, u32 length in dwords, values), see `hwconfig()`.
    pub hwconfig: Vec<u32>,
}

impl XeInfo {
    /// The hwconfig table as (key, values) pairs.
    pub fn hwconfig(&self) -> impl Iterator<Item = (u32, &[u32])> {
        let t = &self.hwconfig;
        let mut i = 0;
        std::iter::from_fn(move || {
            let (key, len) = (*t.get(i)?, *t.get(i + 1)? as usize);
            let values = t.get(i + 2..i + 2 + len)?;
            i += 2 + len;
            Some((key, values))
        })
    }

    /// The system memory region's placement bit (`gem_create`).
    fn sysmem_placement(&self) -> u32 {
        self.mem_regions.iter().find(|r| !r.vram).map_or(1, |r| 1 << r.instance)
    }

    /// The first engine of `engine`'s class.
    pub fn engine(&self, engine: Engine) -> Option<EngineInstance> {
        let class = match engine {
            Engine::Render => sys::DRM_XE_ENGINE_CLASS_RENDER,
            Engine::Compute => sys::DRM_XE_ENGINE_CLASS_COMPUTE,
            Engine::Copy => sys::DRM_XE_ENGINE_CLASS_COPY,
        };
        self.engines.iter().copied().find(|e| e.engine_class == class)
    }
}

impl TryFrom<&Node> for XeInfo {
    type Error = io::Error;

    fn try_from(node: &Node) -> Result<XeInfo> {
        let params: Vec<u64> = node.query(sys::DRM_XE_DEVICE_QUERY_CONFIG)?.items::<sys::drm_xe_query_config, u64>(|h| h.num_params as usize);
        let param = |i: usize| params.get(i).copied().unwrap_or(0);
        let rev_and_id = param(sys::DRM_XE_QUERY_CONFIG_REV_AND_DEVICE_ID);
        let engines = node.query(sys::DRM_XE_DEVICE_QUERY_ENGINES)?.items::<sys::drm_xe_query_engines, sys::drm_xe_engine>(|h| h.num_engines as usize);
        let gts = node.query(sys::DRM_XE_DEVICE_QUERY_GT_LIST)?.items::<sys::drm_xe_query_gt_list, sys::drm_xe_gt>(|h| h.num_gt as usize);
        let regions = node
            .query(sys::DRM_XE_DEVICE_QUERY_MEM_REGIONS)?
            .items::<sys::drm_xe_query_mem_regions, sys::drm_xe_mem_region>(|h| h.num_mem_regions as usize);
        Ok(XeInfo {
            common: DeviceInfo {
                vendor_id: 0x8086,
                device_id: rev_and_id as u16,
                revision: (rev_and_id >> 16) as u8,
                has_vram: param(sys::DRM_XE_QUERY_CONFIG_FLAGS) & sys::DRM_XE_QUERY_CONFIG_FLAG_HAS_VRAM != 0,
                va_bits: param(sys::DRM_XE_QUERY_CONFIG_VA_BITS) as u32,
                min_alignment: param(sys::DRM_XE_QUERY_CONFIG_MIN_ALIGNMENT).max(4096),
            },
            max_exec_queue_priority: param(sys::DRM_XE_QUERY_CONFIG_MAX_EXEC_QUEUE_PRIORITY),
            engines: engines.into_iter().map(|e| e.instance).collect(),
            gts: gts.into_iter().map(Gt::from).collect(),
            mem_regions: regions.into_iter().map(MemRegion::from).collect(),
            topology: node.query(sys::DRM_XE_DEVICE_QUERY_GT_TOPOLOGY)?.topology(),
            hwconfig: node.query(sys::DRM_XE_DEVICE_QUERY_HWCONFIG).map(|q| q.dwords()).unwrap_or_default(),
        })
    }
}

// ---- the device -------------------------------------------------------------------------------------------------

/// ⭐ An open xe render node.
pub struct Xe {
    node: Node,
    info: XeInfo,
}

impl Xe {
    pub fn open(path: &Path) -> Result<Xe> {
        Xe::try_from(Node::open_checked(path, "xe")?)
    }

    pub fn path(&self) -> &Path {
        self.node.path()
    }
}

impl TryFrom<Node> for Xe {
    type Error = io::Error;

    fn try_from(node: Node) -> Result<Xe> {
        let info = XeInfo::try_from(&node)?;
        Ok(Xe { node, info })
    }
}

impl Gpu for Xe {
    type Info = XeInfo;
    type Bo<'d> = XeBo<'d>;
    type Vm<'d> = XeVm<'d>;
    type Queue<'d> = XeQueue<'d>;
    type Fence<'d> = Syncobj<'d>;
    /// The GPU address of a batch buffer ending with MI_BATCH_BUFFER_END.
    type Cmd = u64;

    /// The first render node driven by xe.
    fn open_first() -> Result<Xe> {
        Xe::try_from(Node::open_driver("xe")?)
    }

    fn info(&self) -> DeviceInfo {
        self.info.common
    }

    fn driver_info(&self) -> &XeInfo {
        &self.info
    }

    fn bo(&self, desc: BoDesc) -> Result<XeBo<'_>> {
        let size = desc.size.next_multiple_of(self.info.common.min_alignment);
        let mut c = sys::drm_xe_gem_create {
            size,
            placement: self.info.sysmem_placement(),
            cpu_caching: match desc.caching {
                Caching::WriteBack => sys::DRM_XE_GEM_CPU_CACHING_WB,
                Caching::WriteCombined => sys::DRM_XE_GEM_CPU_CACHING_WC,
            },
            ..Default::default()
        };
        self.node.xe(sys::DRM_IOCTL_XE_GEM_CREATE, &mut c)?;
        Ok(XeBo { xe: self, handle: c.handle, size, map: None })
    }

    fn import(&self, fd: BorrowedFd<'_>) -> Result<XeBo<'_>> {
        let (handle, size) = self.node.import(fd)?;
        Ok(XeBo { xe: self, handle, size, map: None })
    }

    /// A VM in dma-fence mode (submissions signal syncobjs).
    fn vm(&self) -> Result<XeVm<'_>> {
        let mut c = sys::drm_xe_vm_create::default();
        self.node.xe(sys::DRM_IOCTL_XE_VM_CREATE, &mut c)?;
        Ok(XeVm { xe: self, id: c.vm_id })
    }

    fn queue<'d>(&'d self, vm: &XeVm<'d>, engine: Engine) -> Result<XeQueue<'d>> {
        let instance = self.info.engine(engine).ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no {engine:?} engine")))?;
        let mut c = sys::drm_xe_exec_queue_create {
            width: 1,
            num_placements: 1,
            vm_id: vm.id,
            instances: &instance as *const EngineInstance as u64,
            ..Default::default()
        };
        self.node.xe(sys::DRM_IOCTL_XE_EXEC_QUEUE_CREATE, &mut c)?;
        Ok(XeQueue { xe: self, id: c.exec_queue_id })
    }

    fn fence(&self, signaled: bool) -> Result<Syncobj<'_>> {
        self.node.syncobj(signaled)
    }
}

// ---- buffer objects ---------------------------------------------------------------------------------------------

/// The PAT index of a mapping, Xe2 / Xe3 entries (Mesa XE2_PAT_ENTRIES, BSpec 71582).
struct Pat(u16);

impl From<MemType> for Pat {
    fn from(m: MemType) -> Self {
        Pat(match m {
            MemType::Coherent => 1,   // WB, 1-way coherent with the CPU caches
            MemType::Streaming => 0,  // WB, not coherent
            MemType::Scanout => 6,    // WB + display transient
            MemType::Compressed => 9, // WB + compressed
        })
    }
}

/// A GEM buffer object, mapped into this process on first use.
pub struct XeBo<'d> {
    xe: &'d Xe,
    handle: u32,
    size: u64,
    map: Option<Map>,
}

impl crate::base::Bo for XeBo<'_> {
    fn size(&self) -> u64 {
        self.size
    }

    fn bytes(&mut self) -> Result<&mut [u8]> {
        if self.map.is_none() {
            let mut m = sys::drm_xe_gem_mmap_offset { handle: self.handle, ..Default::default() };
            self.xe.node.xe(sys::DRM_IOCTL_XE_GEM_MMAP_OFFSET, &mut m)?;
            self.map = Some(self.xe.node.map(m.offset, self.size as usize)?);
        }
        Ok(self.map.as_mut().map(Map::bytes).unwrap_or_default())
    }

    fn export(&self) -> Result<OwnedFd> {
        self.xe.node.export(self.handle)
    }
}

impl Drop for XeBo<'_> {
    fn drop(&mut self) {
        self.map = None;
        let _ = self.xe.node.close(self.handle);
    }
}

// ---- syncobjs as exec / bind syncs ------------------------------------------------------------------------------

/// A syncobj as a wait (`false`) or a signal (`true`) of an exec or a bind.
impl From<(&Syncobj<'_>, bool)> for sys::drm_xe_sync {
    fn from((f, signal): (&Syncobj<'_>, bool)) -> Self {
        sys::drm_xe_sync {
            type_: sys::DRM_XE_SYNC_TYPE_SYNCOBJ,
            flags: if signal { sys::DRM_XE_SYNC_FLAG_SIGNAL } else { 0 },
            handle_or_addr: f.handle().into(),
            ..Default::default()
        }
    }
}

/// `wait` as waits, then `signal` as signals.
struct Syncs(Vec<sys::drm_xe_sync>);

impl From<(&[&Syncobj<'_>], &[&Syncobj<'_>])> for Syncs {
    fn from((wait, signal): (&[&Syncobj<'_>], &[&Syncobj<'_>])) -> Self {
        Syncs(wait.iter().map(|&f| (f, false).into()).chain(signal.iter().map(|&f| (f, true).into())).collect())
    }
}

// ---- VMs and exec queues ----------------------------------------------------------------------------------------

/// A GPU virtual address space.
pub struct XeVm<'d> {
    xe: &'d Xe,
    id: u32,
}

impl XeVm<'_> {
    fn bind_op(&self, op: sys::drm_xe_vm_bind_op, done: &[&Syncobj<'_>]) -> Result<()> {
        let Syncs(syncs) = (&[][..], done).into();
        let mut b = sys::drm_xe_vm_bind {
            vm_id: self.id,
            num_binds: 1,
            bind: op,
            num_syncs: syncs.len() as u32,
            syncs: syncs.as_ptr() as u64,
            ..Default::default()
        };
        self.xe.node.xe(sys::DRM_IOCTL_XE_VM_BIND, &mut b)
    }
}

impl<'d> crate::base::Vm<'d, Xe> for XeVm<'d> {
    fn bind(&self, bo: &XeBo<'d>, offset: u64, addr: u64, range: u64, mem: MemType, done: &[&Syncobj<'d>]) -> Result<()> {
        let op = sys::drm_xe_vm_bind_op {
            obj: bo.handle,
            pat_index: Pat::from(mem).0,
            obj_offset: offset,
            range,
            addr,
            op: sys::DRM_XE_VM_BIND_OP_MAP,
            ..Default::default()
        };
        self.bind_op(op, done)
    }

    fn unbind(&self, addr: u64, range: u64, done: &[&Syncobj<'d>]) -> Result<()> {
        self.bind_op(sys::drm_xe_vm_bind_op { range, addr, op: sys::DRM_XE_VM_BIND_OP_UNMAP, ..Default::default() }, done)
    }
}

impl Drop for XeVm<'_> {
    fn drop(&mut self) {
        let mut d = sys::drm_xe_vm_destroy { vm_id: self.id, ..Default::default() };
        let _ = self.xe.node.xe(sys::DRM_IOCTL_XE_VM_DESTROY, &mut d);
    }
}

/// A queue of batch buffers on one engine, in one VM.
pub struct XeQueue<'d> {
    xe: &'d Xe,
    id: u32,
}

impl<'d> crate::base::Queue<'d, Xe> for XeQueue<'d> {
    fn submit(&self, batch: &u64, wait: &[&Syncobj<'d>], signal: &[&Syncobj<'d>]) -> Result<()> {
        let Syncs(syncs) = (wait, signal).into();
        let mut e = sys::drm_xe_exec {
            exec_queue_id: self.id,
            num_syncs: syncs.len() as u32,
            syncs: syncs.as_ptr() as u64,
            address: *batch,
            num_batch_buffer: 1,
            ..Default::default()
        };
        self.xe.node.xe(sys::DRM_IOCTL_XE_EXEC, &mut e)
    }

    /// The queue's ban reason (`DRM_XE_EXEC_QUEUE_BAN_REASON_*`), the same for every job on it; ANV asks the same
    /// after a wait.
    fn fault(&self, _done: &Syncobj<'d>) -> Result<Option<u64>> {
        let mut p = sys::drm_xe_exec_queue_get_property {
            exec_queue_id: self.id,
            property: sys::DRM_XE_EXEC_QUEUE_GET_PROPERTY_BAN,
            ..Default::default()
        };
        self.xe.node.xe(sys::DRM_IOCTL_XE_EXEC_QUEUE_GET_PROPERTY, &mut p)?;
        Ok((p.value != 0).then_some(p.value))
    }
}

impl Drop for XeQueue<'_> {
    fn drop(&mut self) {
        let mut d = sys::drm_xe_exec_queue_destroy { exec_queue_id: self.id, ..Default::default() };
        let _ = self.xe.node.xe(sys::DRM_IOCTL_XE_EXEC_QUEUE_DESTROY, &mut d);
    }
}
