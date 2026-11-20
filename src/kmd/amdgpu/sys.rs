#![allow(non_camel_case_types)]
#![allow(dead_code, reason = "mirrors the uAPI headers, used or not")]

use rustix::ioctl::{Opcode, opcode};

use crate::{DRM_COMMAND_BASE, DRM_IOCTL_BASE};

pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;
pub const AMDGPU_GEM_CREATE_CPU_GTT_USWC: u64 = 1 << 2;
pub const AMDGPU_GEM_CREATE_EXPLICIT_SYNC: u64 = 1 << 7;

pub const AMDGPU_CTX_OP_ALLOC_CTX: u32 = 1;
pub const AMDGPU_CTX_OP_FREE_CTX: u32 = 2;
pub const AMDGPU_CTX_OP_QUERY_STATE2: u32 = 4;
pub const AMDGPU_CTX_QUERY2_FLAGS_GUILTY: u64 = 1 << 2;

pub const AMDGPU_VA_OP_MAP: u32 = 1;
pub const AMDGPU_VA_OP_CLEAR: u32 = 3;
pub const AMDGPU_VM_PAGE_READABLE: u32 = 1 << 1;
pub const AMDGPU_VM_PAGE_WRITEABLE: u32 = 1 << 2;
pub const AMDGPU_VM_PAGE_EXECUTABLE: u32 = 1 << 3;
pub const AMDGPU_VM_MTYPE_DEFAULT: u32 = 0;
pub const AMDGPU_VM_MTYPE_UC: u32 = 4 << 5;

pub const AMDGPU_HW_IP_GFX: u32 = 0;
pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;
pub const AMDGPU_HW_IP_DMA: u32 = 2;

pub const AMDGPU_CHUNK_ID_IB: u32 = 0x01;
pub const AMDGPU_CHUNK_ID_SYNCOBJ_IN: u32 = 0x04;
pub const AMDGPU_CHUNK_ID_SYNCOBJ_OUT: u32 = 0x05;
pub const AMDGPU_CHUNK_ID_BO_HANDLES: u32 = 0x06;

pub const AMDGPU_INFO_DEV_INFO: u32 = 0x16;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_gem_create {
    pub bo_size: u64,
    pub alignment: u64,
    pub domains: u64,
    pub domain_flags: u64,
}

/// `union drm_amdgpu_gem_mmap`: in, the handle; out, the offset to mmap
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_gem_mmap {
    pub handle_or_offset: u64,
}

/// `union drm_amdgpu_ctx`: in, the op; out, the context or its state
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_ctx {
    pub op_or_ctx: u32,
    pub flags: u32,
    pub ctx_id: u32,
    pub priority: i32,
}

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

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_cs_chunk {
    pub chunk_id: u32,
    pub length_dw: u32,
    pub chunk_data: u64,
}

/// `union drm_amdgpu_cs`: in; out, the submission's sequence number in `ctx_id` and `bo_list_handle`
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_cs {
    pub ctx_id: u32,
    pub bo_list_handle: u32,
    pub num_chunks: u32,
    pub flags: u32,
    pub chunks: u64,
}

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

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_amdgpu_info {
    pub return_pointer: u64,
    pub return_size: u32,
    pub query: u32,
    pub args: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
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

pub const DRM_IOCTL_AMDGPU_GEM_CREATE: Opcode = opcode::read_write::<drm_amdgpu_gem_create>(DRM_IOCTL_BASE, DRM_COMMAND_BASE);
pub const DRM_IOCTL_AMDGPU_GEM_MMAP: Opcode = opcode::read_write::<drm_amdgpu_gem_mmap>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x01);
pub const DRM_IOCTL_AMDGPU_CTX: Opcode = opcode::read_write::<drm_amdgpu_ctx>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x02);
pub const DRM_IOCTL_AMDGPU_CS: Opcode = opcode::read_write::<drm_amdgpu_cs>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x04);
pub const DRM_IOCTL_AMDGPU_INFO: Opcode = opcode::write::<drm_amdgpu_info>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x05);
pub const DRM_IOCTL_AMDGPU_GEM_VA: Opcode = opcode::write::<drm_amdgpu_gem_va>(DRM_IOCTL_BASE, DRM_COMMAND_BASE + 0x08);
