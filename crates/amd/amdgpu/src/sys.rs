//! Raw uAPI: the structs and ioctl numbers of include/uapi/drm/amdgpu_drm.h (Linux 7.2) that detour uses, laid out as
//! the kernel has them (all `repr(C)`; the header's unions are Rust unions; sizes checked by the tests at the end).

#![allow(non_camel_case_types)]

use std::ffi::c_ulong;

use crate::base::Ioctl;

pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;

pub const AMDGPU_GEM_CREATE_CPU_GTT_USWC: u64 = 1 << 2;
pub const AMDGPU_GEM_CREATE_EXPLICIT_SYNC: u64 = 1 << 7;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_gem_create_in {
    pub bo_size: u64,
    pub alignment: u64,
    pub domains: u64,
    pub domain_flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_gem_create_out {
    pub handle: u32,
    pub _pad: u32,
}

#[repr(C)]
pub union drm_amdgpu_gem_create {
    pub r#in: drm_amdgpu_gem_create_in,
    pub out: drm_amdgpu_gem_create_out,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_gem_mmap_in {
    pub handle: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_gem_mmap_out {
    pub addr_ptr: u64,
}

#[repr(C)]
pub union drm_amdgpu_gem_mmap {
    pub r#in: drm_amdgpu_gem_mmap_in,
    pub out: drm_amdgpu_gem_mmap_out,
}

pub const AMDGPU_CTX_OP_ALLOC_CTX: u32 = 1;
pub const AMDGPU_CTX_OP_FREE_CTX: u32 = 2;
pub const AMDGPU_CTX_OP_QUERY_STATE2: u32 = 4;

pub const AMDGPU_CTX_QUERY2_FLAGS_RESET: u64 = 1 << 0;
pub const AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST: u64 = 1 << 1;
pub const AMDGPU_CTX_QUERY2_FLAGS_GUILTY: u64 = 1 << 2;

pub const AMDGPU_CTX_PRIORITY_NORMAL: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_ctx_in {
    pub op: u32,
    pub flags: u32,
    pub ctx_id: u32,
    pub priority: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_ctx_alloc {
    pub ctx_id: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_ctx_state {
    pub flags: u64,
    pub hangs: u32,
    pub reset_status: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union drm_amdgpu_ctx_out {
    pub alloc: drm_amdgpu_ctx_alloc,
    pub state: drm_amdgpu_ctx_state,
}

#[repr(C)]
pub union drm_amdgpu_ctx {
    pub r#in: drm_amdgpu_ctx_in,
    pub out: drm_amdgpu_ctx_out,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_bo_list_in {
    pub operation: u32,
    pub list_handle: u32,
    pub bo_number: u32,
    pub bo_info_size: u32,
    pub bo_info_ptr: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_bo_list_entry {
    pub bo_handle: u32,
    pub bo_priority: u32,
}

pub const AMDGPU_CHUNK_ID_IB: u32 = 0x01;
pub const AMDGPU_CHUNK_ID_SYNCOBJ_IN: u32 = 0x04;
pub const AMDGPU_CHUNK_ID_SYNCOBJ_OUT: u32 = 0x05;
pub const AMDGPU_CHUNK_ID_BO_HANDLES: u32 = 0x06;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_cs_chunk {
    pub chunk_id: u32,
    pub length_dw: u32,
    pub chunk_data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_cs_in {
    pub ctx_id: u32,
    pub bo_list_handle: u32,
    pub num_chunks: u32,
    pub flags: u32,
    /// A pointer to an array of pointers to `drm_amdgpu_cs_chunk`.
    pub chunks: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_cs_out {
    pub handle: u64,
}

#[repr(C)]
pub union drm_amdgpu_cs {
    pub r#in: drm_amdgpu_cs_in,
    pub out: drm_amdgpu_cs_out,
}

pub const AMDGPU_HW_IP_GFX: u32 = 0;
pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;
pub const AMDGPU_HW_IP_DMA: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_cs_chunk_ib {
    pub _pad: u32,
    pub flags: u32,
    pub va_start: u64,
    pub ib_bytes: u32,
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_cs_chunk_sem {
    pub handle: u32,
}

pub const AMDGPU_VA_OP_MAP: u32 = 1;
pub const AMDGPU_VA_OP_UNMAP: u32 = 2;
pub const AMDGPU_VM_PAGE_READABLE: u32 = 1 << 1;
pub const AMDGPU_VM_PAGE_WRITEABLE: u32 = 1 << 2;
pub const AMDGPU_VM_PAGE_EXECUTABLE: u32 = 1 << 3;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_gem_va {
    pub handle: u32,
    pub _pad: u32,
    pub operation: u32,
    pub flags: u32,
    pub va_address: u64,
    pub offset_in_bo: u64,
    pub map_size: u64,
    pub vm_timeline_point: u64,
    pub vm_timeline_syncobj_out: u32,
    pub num_syncobj_handles: u32,
    pub input_fence_syncobj_handles: u64,
}

pub const AMDGPU_INFO_HW_IP_INFO: u32 = 0x02;
pub const AMDGPU_INFO_DEV_INFO: u32 = 0x16;

/// `drm_amdgpu_info`; `args` is its union of per-query arguments (16 bytes, the largest member).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_info {
    pub return_pointer: u64,
    pub return_size: u32,
    pub query: u32,
    pub args: [u32; 4],
}

pub const AMDGPU_IDS_FLAGS_FUSION: u64 = 0x1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct drm_amdgpu_info_device {
    pub device_id: u32,
    pub chip_rev: u32,
    pub external_rev: u32,
    pub pci_rev: u32,
    pub family: u32,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub gpu_counter_freq: u32,
    pub max_engine_clock: u64,
    pub max_memory_clock: u64,
    pub cu_active_number: u32,
    pub cu_ao_mask: u32,
    pub cu_bitmap: [[u32; 4]; 4],
    pub enabled_rb_pipes_mask: u32,
    pub num_rb_pipes: u32,
    pub num_hw_gfx_contexts: u32,
    pub pcie_gen: u32,
    pub ids_flags: u64,
    pub virtual_address_offset: u64,
    pub virtual_address_max: u64,
    pub virtual_address_alignment: u32,
    pub pte_fragment_size: u32,
    pub gart_page_size: u32,
    pub ce_ram_size: u32,
    pub vram_type: u32,
    pub vram_bit_width: u32,
    pub vce_harvest_config: u32,
    pub gc_double_offchip_lds_buf: u32,
    pub prim_buf_gpu_addr: u64,
    pub pos_buf_gpu_addr: u64,
    pub cntl_sb_buf_gpu_addr: u64,
    pub param_buf_gpu_addr: u64,
    pub prim_buf_size: u32,
    pub pos_buf_size: u32,
    pub cntl_sb_buf_size: u32,
    pub param_buf_size: u32,
    pub wave_front_size: u32,
    pub num_shader_visible_vgprs: u32,
    pub num_cu_per_sh: u32,
    pub num_tcc_blocks: u32,
    pub gs_vgt_table_depth: u32,
    pub gs_prim_buffer_depth: u32,
    pub max_gs_waves_per_vgt: u32,
    pub pcie_num_lanes: u32,
    pub cu_ao_bitmap: [[u32; 4]; 4],
    pub high_va_offset: u64,
    pub high_va_max: u64,
    pub pa_sc_tile_steering_override: u32,
    pub tcc_disabled_mask: u64,
    pub min_engine_clock: u64,
    pub min_memory_clock: u64,
    pub tcp_cache_size: u32,
    pub num_sqc_per_wgp: u32,
    pub sqc_data_cache_size: u32,
    pub sqc_inst_cache_size: u32,
    pub gl1c_cache_size: u32,
    pub gl2c_cache_size: u32,
    pub mall_size: u64,
    pub enabled_rb_pipes_mask_hi: u32,
    pub shadow_size: u32,
    pub shadow_alignment: u32,
    pub csa_size: u32,
    pub csa_alignment: u32,
    pub userq_ip_mask: u32,
    pub pad: u32,
}

impl Default for drm_amdgpu_info_device {
    fn default() -> Self {
        // SAFETY: plain integers, all zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_amdgpu_info_hw_ip {
    pub hw_ip_version_major: u32,
    pub hw_ip_version_minor: u32,
    pub capabilities_flags: u64,
    pub ib_start_alignment: u32,
    pub ib_size_alignment: u32,
    pub available_rings: u32,
    pub ip_discovery_version: u32,
    pub userq_num_slots: u32,
}

const BASE: c_ulong = Ioctl::COMMAND_BASE;

pub const DRM_IOCTL_AMDGPU_GEM_CREATE: c_ulong = Ioctl::wr::<drm_amdgpu_gem_create>(BASE);
pub const DRM_IOCTL_AMDGPU_GEM_MMAP: c_ulong = Ioctl::wr::<drm_amdgpu_gem_mmap>(BASE + 0x01);
pub const DRM_IOCTL_AMDGPU_CTX: c_ulong = Ioctl::wr::<drm_amdgpu_ctx>(BASE + 0x02);
pub const DRM_IOCTL_AMDGPU_CS: c_ulong = Ioctl::wr::<drm_amdgpu_cs>(BASE + 0x04);
pub const DRM_IOCTL_AMDGPU_INFO: c_ulong = Ioctl::w::<drm_amdgpu_info>(BASE + 0x05);
pub const DRM_IOCTL_AMDGPU_GEM_VA: c_ulong = Ioctl::w::<drm_amdgpu_gem_va>(BASE + 0x08);

#[cfg(test)]
mod tests {
    use super::*;

    /// gcc on include/uapi/drm/amdgpu_drm.h (Linux 7.2): `sizeof`, offsets, the `DRM_IOCTL_AMDGPU_*` values.
    #[test]
    fn sizes_match_the_kernel() {
        let sizes = [
            (size_of::<drm_amdgpu_gem_create>(), 32),
            (size_of::<drm_amdgpu_gem_mmap>(), 8),
            (size_of::<drm_amdgpu_ctx>(), 16),
            (size_of::<drm_amdgpu_bo_list_in>(), 24),
            (size_of::<drm_amdgpu_bo_list_entry>(), 8),
            (size_of::<drm_amdgpu_cs_chunk>(), 16),
            (size_of::<drm_amdgpu_cs>(), 24),
            (size_of::<drm_amdgpu_cs_chunk_ib>(), 32),
            (size_of::<drm_amdgpu_cs_chunk_sem>(), 4),
            (size_of::<drm_amdgpu_gem_va>(), 64),
            (size_of::<drm_amdgpu_info>(), 32),
            (size_of::<drm_amdgpu_info_device>(), 448),
            (size_of::<drm_amdgpu_info_hw_ip>(), 40),
            (std::mem::offset_of!(drm_amdgpu_info_device, virtual_address_offset), 144),
            (std::mem::offset_of!(drm_amdgpu_info_device, high_va_offset), 336),
            (std::mem::offset_of!(drm_amdgpu_info_device, tcc_disabled_mask), 360),
            (std::mem::offset_of!(drm_amdgpu_info_device, userq_ip_mask), 436),
            (std::mem::offset_of!(drm_amdgpu_info_hw_ip, available_rings), 24),
        ];
        for (i, (got, want)) in sizes.into_iter().enumerate() {
            assert_eq!(got, want, "entry {i}");
        }
        let ioctls = [
            (DRM_IOCTL_AMDGPU_GEM_CREATE, 0xc020_6440),
            (DRM_IOCTL_AMDGPU_GEM_MMAP, 0xc008_6441),
            (DRM_IOCTL_AMDGPU_CTX, 0xc010_6442),
            (DRM_IOCTL_AMDGPU_CS, 0xc018_6444),
            (DRM_IOCTL_AMDGPU_INFO, 0x4020_6445),
            (DRM_IOCTL_AMDGPU_GEM_VA, 0x4040_6448),
        ];
        for (got, want) in ioctls {
            assert_eq!(got, want);
        }
    }
}
