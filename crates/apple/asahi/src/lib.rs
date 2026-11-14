//! The asahi kernel driver (Apple M1 / M2 GPUs, AGX G13 / G14, on Asahi Linux; M4, G16G, on GravityLinux's kernel, same uAPI) as an [`crate::base::Gpu`], without
//! libdrm: open a render node, read the device's parameters, create buffer objects and map them, manage a GPU
//! virtual address space (VM_BIND), submit compute / render commands to a queue and wait for them with DRM syncobjs.
//!
//! Unlike xe, a queue runs commands the kernel hands to the GPU firmware (`drm_asahi_cmd_compute`: a CDM control
//! stream; `drm_asahi_cmd_render`: a VDM control stream and the tiler's state), not a batch buffer. Addresses are the
//! caller's, inside the windows [`AsahiVm`] gives: shaders (USC binaries) in its 4 GiB `usc` window, everything else
//! in `user`. The layout follows Mesa (src/asahi/lib/agx_device.c `agx_open_device`).

mod sys;

use std::ffi::c_ulong;
use std::io;
use std::ops::Range;
use std::os::fd::{BorrowedFd, OwnedFd};
use std::path::Path;

use crate::base::{BoDesc, Caching, DeviceInfo, Engine, Gpu, Map, MemType, Node, Result, Syncobj};

/// The commands a queue runs, as the kernel takes them.
pub use sys::{
    drm_asahi_bg_eot as BgEot, drm_asahi_cmd_compute as Compute, drm_asahi_cmd_render as Render, drm_asahi_helper_program as Helper,
    drm_asahi_timestamp as Timestamp, drm_asahi_timestamps as Timestamps, drm_asahi_zls_buffer as ZlsBuffer,
};

/// Apple's PCI vendor id (the GPU is not on PCI; for `DeviceInfo` only).
const APPLE: u16 = 0x106b;
/// The USC window: shader addresses are 32 bits from its base (`drm_asahi_queue_create::usc_exec_base`).
const USC_SIZE: u64 = 1 << 32;
/// The kernel's range at the top of the VM, at least `vm_kernel_min_size` (Mesa gives it 32 GiB).
const KERNEL_SIZE: u64 = 32 << 30;
/// The GPU's page (UAT) and the kernel's: 16 KiB on Apple silicon.
const PAGE: u64 = 16 << 10;

// ---- the render node --------------------------------------------------------------------------------------------

/// asahi's ioctls on a render node.
trait AsahiNode {
    fn asahi<T>(&self, request: c_ulong, arg: &mut T) -> Result<()>;
}

impl AsahiNode for Node {
    fn asahi<T>(&self, request: c_ulong, arg: &mut T) -> Result<()> {
        // SAFETY: sys.rs builds every request from the struct it takes; callers fill in its pointers.
        unsafe { self.ioctl(request, arg) }
    }
}

// ---- what the device reports ------------------------------------------------------------------------------------

/// Which AGX the device is, as Mesa sorts them (`agx_open_device`'s `dev->chip`). G13 / G14 are from upstream Mesa;
/// G15 / G16G / G17P from GravityLinux's Mesa fork (github.com/GravityLinux/mesa `87ac51d`, MIT,
/// src/asahi/lib/agx_device.c), which runs M4 on the same asahi uAPI. Not checked on any machine of ours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chip {
    /// M1, one cluster.
    G13G,
    /// M1 Pro / Max / Ultra, more than one cluster.
    G13X,
    /// M2 (and M3, generation 15, which GravityLinux drives as G14).
    G14G,
    /// M2 Pro / Max / Ultra (and M3 Pro / Max).
    G14X,
    /// M4, T8132: the Apple9 ISA and the fixed USC base (`agx_apple9_*`).
    G16G,
    /// T8140 (generation 17, variant 'P'): Apple9 like G16G.
    G17P,
}

/// Which shader ISA a [`Chip`] runs: what `detour-apple-dialect`'s `agx.usc` encodes, or Apple9 (not done).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AGX2 (Mesa src/asahi/isa/AGX2.xml).
    Agx2,
    /// Apple9: other encodings (GravityLinux src/asahi/compiler/agx_apple9_*), shaders entered through a table at the
    /// fixed USC base.
    Apple9,
}

impl Chip {
    pub const fn isa(self) -> Isa {
        match self {
            Self::G13G | Self::G13X | Self::G14G | Self::G14X => Isa::Agx2,
            Self::G16G | Self::G17P => Isa::Apple9,
        }
    }
}

impl TryFrom<&sys::drm_asahi_params_global> for Chip {
    type Error = io::Error;

    /// GravityLinux asserts the variant of 16 and 17 (only 'G' and 'P' are known); so do we.
    fn try_from(p: &sys::drm_asahi_params_global) -> Result<Chip> {
        let many = p.num_clusters_total > 1;
        match (p.gpu_generation, char::from_u32(p.gpu_variant)) {
            (13, _) => Ok(if many { Self::G13X } else { Self::G13G }),
            (14 | 15, _) => Ok(if many { Self::G14X } else { Self::G14G }),
            (16, Some('G')) => Ok(Self::G16G),
            (17, Some('P')) => Ok(Self::G17P),
            (generation, variant) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("asahi: AGX generation {generation} variant {variant:?} is not known"),
            )),
        }
    }
}

/// What only asahi reports (`drm_asahi_params_global`), read once at open.
#[derive(Clone, Debug)]
pub struct AsahiInfo {
    pub common: DeviceInfo,
    /// 13 = M1, 14 = M2, 15 = M3, 16 = M4, 17 (T8140).
    pub generation: u32,
    pub chip: Chip,
    /// 'G' (base), 'S' (Pro), 'C' (Max), 'D' (Ultra).
    pub variant: char,
    pub chip_id: u32,
    pub dies: u32,
    pub clusters: u32,
    pub cores_per_cluster: u32,
    pub max_frequency_khz: u32,
    /// Per cluster, the cores that are there.
    pub core_masks: Vec<u64>,
    /// The VA range the VM may use.
    pub vm: Range<u64>,
    pub vm_kernel_min_size: u64,
    pub max_commands_per_submission: u32,
    pub max_attachments: u32,
    pub timestamp_frequency_hz: u64,
    /// Shader loads past a mapping read zero instead of faulting (`DRM_ASAHI_FEATURE_SOFT_FAULTS`).
    pub soft_faults: bool,
}

impl AsahiInfo {
    /// "Apple M1 (G13G B1)", as Mesa names it (`agx_open_device`).
    pub fn name(&self) -> String {
        let variant = match self.variant {
            // Apple9 firmware reports Pro as 'P' (GravityLinux agx_device.c).
            'S' | 'P' => " Pro",
            'C' => " Max",
            'D' => " Ultra",
            _ => "",
        };
        let rev = self.common.revision;
        format!("Apple M{}{variant} (G{}{} {:02X})", self.generation - 12, self.generation, self.variant, u32::from(rev) + 0xa0)
    }
}

impl TryFrom<&Node> for AsahiInfo {
    type Error = io::Error;

    fn try_from(node: &Node) -> Result<AsahiInfo> {
        let mut p = sys::drm_asahi_params_global::default();
        let mut g = sys::drm_asahi_get_params {
            pointer: &mut p as *mut sys::drm_asahi_params_global as u64,
            size: size_of::<sys::drm_asahi_params_global>() as u64,
            ..Default::default()
        };
        node.asahi(sys::DRM_IOCTL_ASAHI_GET_PARAMS, &mut g)?;
        let clusters = (p.num_clusters_total as usize).min(sys::DRM_ASAHI_MAX_CLUSTERS);
        Ok(AsahiInfo {
            common: DeviceInfo {
                vendor_id: APPLE,
                device_id: p.chip_id as u16,
                revision: p.gpu_revision as u8,
                has_vram: false,
                va_bits: 64 - p.vm_end.saturating_sub(1).leading_zeros(),
                min_alignment: PAGE,
            },
            generation: p.gpu_generation,
            chip: Chip::try_from(&p)?,
            variant: char::from_u32(p.gpu_variant).unwrap_or('?'),
            chip_id: p.chip_id,
            dies: p.num_dies,
            clusters: p.num_clusters_total,
            cores_per_cluster: p.num_cores_per_cluster,
            max_frequency_khz: p.max_frequency_khz,
            core_masks: p.core_masks[..clusters].to_vec(),
            vm: p.vm_start..p.vm_end,
            vm_kernel_min_size: p.vm_kernel_min_size,
            max_commands_per_submission: p.max_commands_per_submission,
            max_attachments: p.max_attachments,
            timestamp_frequency_hz: p.command_timestamp_frequency_hz,
            soft_faults: p.features & sys::DRM_ASAHI_FEATURE_SOFT_FAULTS != 0,
        })
    }
}

// ---- the device -------------------------------------------------------------------------------------------------

/// ⭐ An open asahi render node.
pub struct Asahi {
    node: Node,
    info: AsahiInfo,
}

impl Asahi {
    pub fn open(path: &Path) -> Result<Asahi> {
        Asahi::try_from(Node::open_checked(path, "asahi")?)
    }

    pub fn path(&self) -> &Path {
        self.node.path()
    }
}

impl TryFrom<Node> for Asahi {
    type Error = io::Error;

    fn try_from(node: Node) -> Result<Asahi> {
        let info = AsahiInfo::try_from(&node)?;
        Ok(Asahi { node, info })
    }
}

/// One command of a submission.
#[derive(Clone, Debug)]
pub enum Command {
    Compute(Compute),
    Render(Box<Render>),
}

/// A uAPI struct as the bytes the kernel reads.
trait Raw: Sized {
    fn raw(&self) -> &[u8] {
        // SAFETY: only for `repr(C)` uAPI structs without implicit padding (asahi_drm.h's rules, sizes tested).
        unsafe { std::slice::from_raw_parts((self as *const Self).cast(), size_of::<Self>()) }
    }
}

impl Raw for sys::drm_asahi_cmd_header {}
impl Raw for Compute {}
impl Raw for Render {}

/// A submission's command buffer: each command after a header. Every command waits for all the ones before it, on
/// both subqueues (a barrier is the count of earlier commands of that subqueue in this submission, 0 = the earlier
/// submissions): the queue runs in order, as an xe batch does.
struct Cmdbuf(Vec<u8>);

impl From<&[Command]> for Cmdbuf {
    fn from(cmds: &[Command]) -> Self {
        let (mut vdm, mut cdm) = (0u16, 0u16);
        let mut out = Vec::new();
        for c in cmds {
            let (cmd_type, payload) = match c {
                Command::Compute(c) => (sys::DRM_ASAHI_CMD_COMPUTE, c.raw()),
                Command::Render(r) => (sys::DRM_ASAHI_CMD_RENDER, r.raw()),
            };
            let h = sys::drm_asahi_cmd_header { cmd_type, size: payload.len() as u16, vdm_barrier: vdm, cdm_barrier: cdm };
            out.extend_from_slice(h.raw());
            out.extend_from_slice(payload);
            if cmd_type == sys::DRM_ASAHI_CMD_COMPUTE {
                cdm += 1;
            } else {
                vdm += 1;
            }
        }
        Cmdbuf(out)
    }
}

impl Gpu for Asahi {
    type Info = AsahiInfo;
    type Bo<'d> = AsahiBo<'d>;
    type Vm<'d> = AsahiVm<'d>;
    type Queue<'d> = AsahiQueue<'d>;
    type Fence<'d> = Syncobj<'d>;
    /// The commands of one submission, run in order.
    type Cmd = [Command];

    /// The first render node driven by asahi.
    fn open_first() -> Result<Asahi> {
        Asahi::try_from(Node::open_driver("asahi")?)
    }

    fn info(&self) -> DeviceInfo {
        self.info.common
    }

    fn driver_info(&self) -> &AsahiInfo {
        &self.info
    }

    /// Write-back buffers are cached and coherent with the GPU (Mesa `AGX_BO_WRITEBACK`), the rest write-combined.
    fn bo(&self, desc: BoDesc) -> Result<AsahiBo<'_>> {
        let size = desc.size.next_multiple_of(PAGE);
        let flags = match desc.caching {
            Caching::WriteBack => sys::DRM_ASAHI_GEM_WRITEBACK,
            Caching::WriteCombined => 0,
        };
        let mut c = sys::drm_asahi_gem_create { size, flags, ..Default::default() };
        self.node.asahi(sys::DRM_IOCTL_ASAHI_GEM_CREATE, &mut c)?;
        Ok(AsahiBo { asahi: self, handle: c.handle, size, map: None })
    }

    fn import(&self, fd: BorrowedFd<'_>) -> Result<AsahiBo<'_>> {
        let (handle, size) = self.node.import(fd)?;
        Ok(AsahiBo { asahi: self, handle, size, map: None })
    }

    /// A VM: the kernel's range at the top (Mesa: at least 32 GiB), the USC window at the bottom (4 GiB aligned,
    /// above the first 4 GiB so that a null pointer faults), the user range between them.
    fn vm(&self) -> Result<AsahiVm<'_>> {
        let Range { start, end } = self.info.vm;
        let kernel = end - KERNEL_SIZE.max(self.info.vm_kernel_min_size)..end;
        let usc = start.max(USC_SIZE).next_multiple_of(USC_SIZE);
        let usc = usc..usc + USC_SIZE;
        if usc.end >= kernel.start {
            return Err(io::Error::new(io::ErrorKind::Unsupported, format!("VM {:#x?} too small for the USC window", self.info.vm)));
        }
        let mut c = sys::drm_asahi_vm_create { kernel_start: kernel.start, kernel_end: kernel.end, ..Default::default() };
        self.node.asahi(sys::DRM_IOCTL_ASAHI_VM_CREATE, &mut c)?;
        Ok(AsahiVm { asahi: self, id: c.vm_id, user: usc.end..kernel.start, usc })
    }

    /// One queue for every engine: it has a render (VDM) and a compute (CDM) subqueue.
    fn queue<'d>(&'d self, vm: &AsahiVm<'d>, _engine: Engine) -> Result<AsahiQueue<'d>> {
        let mut c = sys::drm_asahi_queue_create {
            vm_id: vm.id,
            priority: sys::DRM_ASAHI_PRIORITY_MEDIUM,
            usc_exec_base: vm.usc.start,
            ..Default::default()
        };
        self.node.asahi(sys::DRM_IOCTL_ASAHI_QUEUE_CREATE, &mut c)?;
        Ok(AsahiQueue { asahi: self, id: c.queue_id })
    }

    fn fence(&self, signaled: bool) -> Result<Syncobj<'_>> {
        self.node.syncobj(signaled)
    }
}

// ---- buffer objects ---------------------------------------------------------------------------------------------

/// A GEM buffer object, mapped into this process on first use.
pub struct AsahiBo<'d> {
    asahi: &'d Asahi,
    handle: u32,
    size: u64,
    map: Option<Map>,
}

impl crate::base::Bo for AsahiBo<'_> {
    fn size(&self) -> u64 {
        self.size
    }

    fn bytes(&mut self) -> Result<&mut [u8]> {
        if self.map.is_none() {
            let mut m = sys::drm_asahi_gem_mmap_offset { handle: self.handle, ..Default::default() };
            self.asahi.node.asahi(sys::DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &mut m)?;
            self.map = Some(self.asahi.node.map(m.offset, self.size as usize)?);
        }
        Ok(self.map.as_mut().map(Map::bytes).unwrap_or_default())
    }

    fn export(&self) -> Result<OwnedFd> {
        self.asahi.node.export(self.handle)
    }
}

impl Drop for AsahiBo<'_> {
    fn drop(&mut self) {
        self.map = None;
        let _ = self.asahi.node.close(self.handle);
    }
}

// ---- syncobjs as submit syncs -----------------------------------------------------------------------------------

impl From<&Syncobj<'_>> for sys::drm_asahi_sync {
    fn from(f: &Syncobj<'_>) -> Self {
        sys::drm_asahi_sync { sync_type: sys::DRM_ASAHI_SYNC_SYNCOBJ, handle: f.handle(), timeline_value: 0 }
    }
}

// ---- VMs and queues ---------------------------------------------------------------------------------------------

/// A GPU virtual address space and its windows.
pub struct AsahiVm<'d> {
    asahi: &'d Asahi,
    id: u32,
    /// Shaders (USC binaries) go here; their addresses are 32 bits from `usc.start`.
    pub usc: Range<u64>,
    /// Everything else.
    pub user: Range<u64>,
}

impl AsahiVm<'_> {
    fn bind_op(&self, op: sys::drm_asahi_gem_bind_op) -> Result<()> {
        let mut b = sys::drm_asahi_vm_bind {
            vm_id: self.id,
            num_binds: 1,
            stride: size_of::<sys::drm_asahi_gem_bind_op>() as u32,
            userptr: &op as *const sys::drm_asahi_gem_bind_op as u64,
            ..Default::default()
        };
        self.asahi.node.asahi(sys::DRM_IOCTL_ASAHI_VM_BIND, &mut b)
    }
}

impl<'d> crate::base::Vm<'d, Asahi> for AsahiVm<'d> {
    /// Synchronous (asahi binds before the ioctl returns), so `done` is signalled right away. The GPU's caching is
    /// the buffer's (`Caching`), not the mapping's: `mem` only says how it is used.
    fn bind(&self, bo: &AsahiBo<'d>, offset: u64, addr: u64, range: u64, _mem: MemType, done: &[&Syncobj<'d>]) -> Result<()> {
        let flags = sys::DRM_ASAHI_BIND_READ | sys::DRM_ASAHI_BIND_WRITE;
        self.bind_op(sys::drm_asahi_gem_bind_op { flags, handle: bo.handle, offset, range, addr })?;
        done.iter().try_for_each(|f| f.signal())
    }

    fn unbind(&self, addr: u64, range: u64, done: &[&Syncobj<'d>]) -> Result<()> {
        self.bind_op(sys::drm_asahi_gem_bind_op { flags: sys::DRM_ASAHI_BIND_UNBIND, handle: 0, offset: 0, range, addr })?;
        done.iter().try_for_each(|f| f.signal())
    }
}

impl Drop for AsahiVm<'_> {
    fn drop(&mut self) {
        let mut d = sys::drm_asahi_vm_destroy { vm_id: self.id, pad: 0 };
        let _ = self.asahi.node.asahi(sys::DRM_IOCTL_ASAHI_VM_DESTROY, &mut d);
    }
}

/// A queue of compute and render commands, in one VM.
pub struct AsahiQueue<'d> {
    asahi: &'d Asahi,
    id: u32,
}

impl<'d> crate::base::Queue<'d, Asahi> for AsahiQueue<'d> {
    fn submit(&self, cmds: &[Command], wait: &[&Syncobj<'d>], signal: &[&Syncobj<'d>]) -> Result<()> {
        let max = self.asahi.info.max_commands_per_submission as usize;
        if cmds.is_empty() || cmds.len() > max {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{} commands, a submission takes 1 ..= {max}", cmds.len())));
        }
        let Cmdbuf(buf) = cmds.into();
        let syncs: Vec<sys::drm_asahi_sync> = wait.iter().chain(signal).map(|&f| f.into()).collect();
        let mut s = sys::drm_asahi_submit {
            syncs: syncs.as_ptr() as u64,
            cmdbuf: buf.as_ptr() as u64,
            queue_id: self.id,
            in_sync_count: wait.len() as u32,
            out_sync_count: signal.len() as u32,
            cmdbuf_size: buf.len() as u32,
            ..Default::default()
        };
        self.asahi.node.asahi(sys::DRM_IOCTL_ASAHI_SUBMIT, &mut s)
    }

    /// The error the job's fence carries (`-errno` of `dma_fence_get_status`). asahi has no ban query; Mesa (hk /
    /// gallium asahi) does not detect faults at all.
    fn fault(&self, done: &Syncobj<'d>) -> Result<Option<u64>> {
        let status = done.status()?;
        Ok((status < 0).then(|| status.unsigned_abs().into()))
    }
}

impl Drop for AsahiQueue<'_> {
    fn drop(&mut self) {
        let mut d = sys::drm_asahi_queue_destroy { queue_id: self.id, pad: 0 };
        let _ = self.asahi.node.asahi(sys::DRM_IOCTL_ASAHI_QUEUE_DESTROY, &mut d);
    }
}
