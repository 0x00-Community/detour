#![allow(non_camel_case_types)]
#![allow(dead_code, reason = "mirrors the uAPI headers, used or not")]

use std::ffi::c_ulong;

use crate::Ioctl;

const DRM_COMMAND_BASE: c_ulong = Ioctl::COMMAND_BASE;

pub const DRM_XE_ENGINE_CLASS_RENDER: u16 = 0;
pub const DRM_XE_ENGINE_CLASS_COPY: u16 = 1;
pub const DRM_XE_ENGINE_CLASS_VIDEO_DECODE: u16 = 2;
pub const DRM_XE_ENGINE_CLASS_VIDEO_ENHANCE: u16 = 3;
pub const DRM_XE_ENGINE_CLASS_COMPUTE: u16 = 4;
pub const DRM_XE_ENGINE_CLASS_VM_BIND: u16 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct drm_xe_engine_class_instance {
    pub engine_class: u16,
    pub engine_instance: u16,
    pub gt_id: u16,
    pub pad: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_engine {
    pub instance: drm_xe_engine_class_instance,
    pub reserved: [u64; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_query_engines {
    pub num_engines: u32,
    pub pad: u32,
}

pub const DRM_XE_MEM_REGION_CLASS_SYSMEM: u16 = 0;
pub const DRM_XE_MEM_REGION_CLASS_VRAM: u16 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_mem_region {
    pub mem_class: u16,
    pub instance: u16,
    pub min_page_size: u32,
    pub total_size: u64,
    pub used: u64,
    pub cpu_visible_size: u64,
    pub cpu_visible_used: u64,
    pub reserved: [u64; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_query_mem_regions {
    pub num_mem_regions: u32,
    pub pad: u32,
}

pub const DRM_XE_QUERY_CONFIG_REV_AND_DEVICE_ID: usize = 0;
pub const DRM_XE_QUERY_CONFIG_FLAGS: usize = 1;
pub const DRM_XE_QUERY_CONFIG_FLAG_HAS_VRAM: u64 = 1 << 0;
pub const DRM_XE_QUERY_CONFIG_MIN_ALIGNMENT: usize = 2;
pub const DRM_XE_QUERY_CONFIG_VA_BITS: usize = 3;
pub const DRM_XE_QUERY_CONFIG_MAX_EXEC_QUEUE_PRIORITY: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_query_config {
    pub num_params: u32,
    pub pad: u32,
}

pub const DRM_XE_QUERY_GT_TYPE_MAIN: u16 = 0;
pub const DRM_XE_QUERY_GT_TYPE_MEDIA: u16 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_gt {
    pub type_: u16,
    pub tile_id: u16,
    pub gt_id: u16,
    pub pad: [u16; 3],
    pub reference_clock: u32,
    pub near_mem_regions: u64,
    pub far_mem_regions: u64,
    pub ip_ver_major: u16,
    pub ip_ver_minor: u16,
    pub ip_ver_rev: u16,
    pub pad2: u16,
    pub reserved: [u64; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_query_gt_list {
    pub num_gt: u32,
    pub pad: u32,
}

pub const DRM_XE_TOPO_DSS_GEOMETRY: u16 = 1;
pub const DRM_XE_TOPO_DSS_COMPUTE: u16 = 2;
pub const DRM_XE_TOPO_L3_BANK: u16 = 3;
pub const DRM_XE_TOPO_EU_PER_DSS: u16 = 4;
pub const DRM_XE_TOPO_SIMD16_EU_PER_DSS: u16 = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_query_topology_mask {
    pub gt_id: u16,
    pub type_: u16,
    pub num_bytes: u32,
}

pub const DRM_XE_DEVICE_QUERY_ENGINES: u32 = 0;
pub const DRM_XE_DEVICE_QUERY_MEM_REGIONS: u32 = 1;
pub const DRM_XE_DEVICE_QUERY_CONFIG: u32 = 2;
pub const DRM_XE_DEVICE_QUERY_GT_LIST: u32 = 3;
pub const DRM_XE_DEVICE_QUERY_HWCONFIG: u32 = 4;
pub const DRM_XE_DEVICE_QUERY_GT_TOPOLOGY: u32 = 5;

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_device_query {
    pub extensions: u64,
    pub query: u32,
    pub size: u32,
    pub data: u64,
    pub reserved: [u64; 2],
}

pub const DRM_XE_GEM_CREATE_FLAG_DEFER_BACKING: u32 = 1 << 0;
pub const DRM_XE_GEM_CREATE_FLAG_SCANOUT: u32 = 1 << 1;
pub const DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM: u32 = 1 << 2;
pub const DRM_XE_GEM_CPU_CACHING_WB: u16 = 1;
pub const DRM_XE_GEM_CPU_CACHING_WC: u16 = 2;

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_gem_create {
    pub extensions: u64,
    pub size: u64,
    pub placement: u32,
    pub flags: u32,
    pub vm_id: u32,
    pub handle: u32,
    pub cpu_caching: u16,
    pub pad: [u16; 3],
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_gem_mmap_offset {
    pub extensions: u64,
    pub handle: u32,
    pub flags: u32,
    pub offset: u64,
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_vm_create {
    pub extensions: u64,
    pub flags: u32,
    pub vm_id: u32,
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_vm_destroy {
    pub vm_id: u32,
    pub pad: u32,
    pub reserved: [u64; 2],
}

pub const DRM_XE_VM_BIND_OP_MAP: u32 = 0x0;
pub const DRM_XE_VM_BIND_OP_UNMAP: u32 = 0x1;
pub const DRM_XE_VM_BIND_OP_MAP_USERPTR: u32 = 0x2;
pub const DRM_XE_VM_BIND_FLAG_READONLY: u32 = 1 << 0;
pub const DRM_XE_VM_BIND_FLAG_IMMEDIATE: u32 = 1 << 1;
pub const DRM_XE_VM_BIND_FLAG_NULL: u32 = 1 << 2;
pub const DRM_XE_VM_BIND_FLAG_DUMPABLE: u32 = 1 << 3;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_xe_vm_bind_op {
    pub extensions: u64,
    pub obj: u32,
    pub pat_index: u16,
    pub pad: u16,
    pub obj_offset: u64,
    pub range: u64,
    pub addr: u64,
    pub op: u32,
    pub flags: u32,
    pub prefetch_mem_region_instance: u32,
    pub pad2: u32,
    pub reserved: [u64; 3],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_vm_bind {
    pub extensions: u64,
    pub vm_id: u32,
    pub exec_queue_id: u32,
    pub pad: u32,
    pub num_binds: u32,
    pub bind: drm_xe_vm_bind_op,
    pub pad2: u32,
    pub num_syncs: u32,
    pub syncs: u64,
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_exec_queue_create {
    pub extensions: u64,
    pub width: u16,
    pub num_placements: u16,
    pub vm_id: u32,
    pub flags: u32,
    pub exec_queue_id: u32,
    pub instances: u64,
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_exec_queue_destroy {
    pub exec_queue_id: u32,
    pub pad: u32,
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_exec_queue_get_property {
    pub extensions: u64,
    pub exec_queue_id: u32,
    pub property: u32,
    pub value: u64,
    pub reserved: [u64; 2],
}

pub const DRM_XE_EXEC_QUEUE_GET_PROPERTY_BAN: u32 = 0;
pub const DRM_XE_EXEC_QUEUE_BAN_REASON_GPU_HANG: u64 = 1 << 0;
pub const DRM_XE_EXEC_QUEUE_BAN_REASON_PAGE_OFFLINE: u64 = 1 << 1;

pub const DRM_XE_SYNC_TYPE_SYNCOBJ: u32 = 0x0;
pub const DRM_XE_SYNC_TYPE_TIMELINE_SYNCOBJ: u32 = 0x1;
pub const DRM_XE_SYNC_TYPE_USER_FENCE: u32 = 0x2;
pub const DRM_XE_SYNC_FLAG_SIGNAL: u32 = 1 << 0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_xe_sync {
    pub extensions: u64,
    pub type_: u32,
    pub flags: u32,
    pub handle_or_addr: u64,
    pub timeline_value: u64,
    pub reserved: [u64; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct drm_xe_exec {
    pub extensions: u64,
    pub exec_queue_id: u32,
    pub num_syncs: u32,
    pub syncs: u64,
    pub address: u64,
    pub num_batch_buffer: u16,
    pub pad: [u16; 3],
    pub reserved: [u64; 2],
}

pub const DRM_IOCTL_XE_DEVICE_QUERY: c_ulong = Ioctl::wr::<drm_xe_device_query>(DRM_COMMAND_BASE);
pub const DRM_IOCTL_XE_GEM_CREATE: c_ulong = Ioctl::wr::<drm_xe_gem_create>(DRM_COMMAND_BASE + 0x01);
pub const DRM_IOCTL_XE_GEM_MMAP_OFFSET: c_ulong = Ioctl::wr::<drm_xe_gem_mmap_offset>(DRM_COMMAND_BASE + 0x02);
pub const DRM_IOCTL_XE_VM_CREATE: c_ulong = Ioctl::wr::<drm_xe_vm_create>(DRM_COMMAND_BASE + 0x03);
pub const DRM_IOCTL_XE_VM_DESTROY: c_ulong = Ioctl::w::<drm_xe_vm_destroy>(DRM_COMMAND_BASE + 0x04);
pub const DRM_IOCTL_XE_VM_BIND: c_ulong = Ioctl::w::<drm_xe_vm_bind>(DRM_COMMAND_BASE + 0x05);
pub const DRM_IOCTL_XE_EXEC_QUEUE_CREATE: c_ulong = Ioctl::wr::<drm_xe_exec_queue_create>(DRM_COMMAND_BASE + 0x06);
pub const DRM_IOCTL_XE_EXEC_QUEUE_DESTROY: c_ulong = Ioctl::w::<drm_xe_exec_queue_destroy>(DRM_COMMAND_BASE + 0x07);
pub const DRM_IOCTL_XE_EXEC_QUEUE_GET_PROPERTY: c_ulong = Ioctl::wr::<drm_xe_exec_queue_get_property>(DRM_COMMAND_BASE + 0x08);
pub const DRM_IOCTL_XE_EXEC: c_ulong = Ioctl::w::<drm_xe_exec>(DRM_COMMAND_BASE + 0x09);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_kernel() {
        assert_eq!(size_of::<drm_xe_device_query>(), 40);
        assert_eq!(size_of::<drm_xe_engine>(), 32);
        assert_eq!(size_of::<drm_xe_mem_region>(), 88);
        assert_eq!(size_of::<drm_xe_gt>(), 96);
        assert_eq!(size_of::<drm_xe_gem_create>(), 56);
        assert_eq!(size_of::<drm_xe_gem_mmap_offset>(), 40);
        assert_eq!(size_of::<drm_xe_vm_create>(), 32);
        assert_eq!(size_of::<drm_xe_vm_bind_op>(), 80);
        assert_eq!(size_of::<drm_xe_vm_bind>(), 136);
        assert_eq!(size_of::<drm_xe_exec_queue_create>(), 48);
        assert_eq!(size_of::<drm_xe_sync>(), 48);
        assert_eq!(size_of::<drm_xe_exec>(), 56);
        assert_eq!(size_of::<drm_xe_exec_queue_get_property>(), 40);
        assert_eq!(DRM_IOCTL_XE_DEVICE_QUERY, 0xc028_6440);
        assert_eq!(DRM_IOCTL_XE_EXEC, 0x4038_6449);
        assert_eq!(DRM_IOCTL_XE_EXEC_QUEUE_GET_PROPERTY, 0xc028_6448);
    }
}
