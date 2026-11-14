#![allow(non_camel_case_types)]
#![allow(dead_code, reason = "mirrors the uAPI header, used or not")]

use std::ffi::c_ulong;

use crate::base::Ioctl;

pub const DRM_ASAHI_MAX_CLUSTERS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct drm_asahi_params_global {
    pub features: u64,
    pub gpu_generation: u32,
    pub gpu_variant: u32,
    pub gpu_revision: u32,
    pub chip_id: u32,
    pub num_dies: u32,
    pub num_clusters_total: u32,
    pub num_cores_per_cluster: u32,
    pub max_frequency_khz: u32,
    pub core_masks: [u64; DRM_ASAHI_MAX_CLUSTERS],
    pub vm_start: u64,
    pub vm_end: u64,
    pub vm_kernel_min_size: u64,
    pub max_commands_per_submission: u32,
    pub max_attachments: u32,
    pub command_timestamp_frequency_hz: u64,
}

impl Default for drm_asahi_params_global {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

pub const DRM_ASAHI_FEATURE_SOFT_FAULTS: u64 = 1 << 0;

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_get_params {
    pub param_group: u32,
    pub pad: u32,
    pub pointer: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_vm_create {
    pub kernel_start: u64,
    pub kernel_end: u64,
    pub vm_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_vm_destroy {
    pub vm_id: u32,
    pub pad: u32,
}

pub const DRM_ASAHI_GEM_WRITEBACK: u32 = 1 << 0;
pub const DRM_ASAHI_GEM_VM_PRIVATE: u32 = 1 << 1;

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_gem_create {
    pub size: u64,
    pub flags: u32,
    pub vm_id: u32,
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_gem_mmap_offset {
    pub handle: u32,
    pub flags: u32,
    pub offset: u64,
}

pub const DRM_ASAHI_BIND_UNBIND: u32 = 1 << 0;
pub const DRM_ASAHI_BIND_READ: u32 = 1 << 1;
pub const DRM_ASAHI_BIND_WRITE: u32 = 1 << 2;
pub const DRM_ASAHI_BIND_SINGLE_PAGE: u32 = 1 << 3;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_asahi_gem_bind_op {
    pub flags: u32,
    pub handle: u32,
    pub offset: u64,
    pub range: u64,
    pub addr: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_vm_bind {
    pub vm_id: u32,
    pub num_binds: u32,
    pub stride: u32,
    pub pad: u32,
    pub userptr: u64,
}

pub const DRM_ASAHI_BIND_OBJECT_OP_BIND: u32 = 0;
pub const DRM_ASAHI_BIND_OBJECT_OP_UNBIND: u32 = 1;
pub const DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS: u32 = 1 << 0;

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_gem_bind_object {
    pub op: u32,
    pub flags: u32,
    pub handle: u32,
    pub vm_id: u32,
    pub offset: u64,
    pub range: u64,
    pub object_handle: u32,
    pub pad: u32,
}

pub const DRM_ASAHI_CMD_RENDER: u16 = 0;
pub const DRM_ASAHI_CMD_COMPUTE: u16 = 1;
pub const DRM_ASAHI_SET_VERTEX_ATTACHMENTS: u16 = 2;
pub const DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS: u16 = 3;
pub const DRM_ASAHI_SET_COMPUTE_ATTACHMENTS: u16 = 4;

pub const DRM_ASAHI_PRIORITY_LOW: u32 = 0;
pub const DRM_ASAHI_PRIORITY_MEDIUM: u32 = 1;
pub const DRM_ASAHI_PRIORITY_HIGH: u32 = 2;
pub const DRM_ASAHI_PRIORITY_REALTIME: u32 = 3;

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_queue_create {
    pub flags: u32,
    pub vm_id: u32,
    pub priority: u32,
    pub queue_id: u32,
    pub usc_exec_base: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_queue_destroy {
    pub queue_id: u32,
    pub pad: u32,
}

pub const DRM_ASAHI_SYNC_SYNCOBJ: u32 = 0;
pub const DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_asahi_sync {
    pub sync_type: u32,
    pub handle: u32,
    pub timeline_value: u64,
}

pub const DRM_ASAHI_BARRIER_NONE: u16 = 0xffff;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct drm_asahi_cmd_header {
    pub cmd_type: u16,
    pub size: u16,
    pub vdm_barrier: u16,
    pub cdm_barrier: u16,
}

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_submit {
    pub syncs: u64,
    pub cmdbuf: u64,
    pub flags: u32,
    pub queue_id: u32,
    pub in_sync_count: u32,
    pub out_sync_count: u32,
    pub cmdbuf_size: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_attachment {
    pub pointer: u64,
    pub size: u64,
    pub pad: u32,
    pub flags: u32,
}

pub const DRM_ASAHI_RENDER_VERTEX_SCRATCH: u32 = 1 << 0;
pub const DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES: u32 = 1 << 1;
pub const DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING: u32 = 1 << 2;
pub const DRM_ASAHI_RENDER_DBIAS_IS_INT: u32 = 1 << 18;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_zls_buffer {
    pub base: u64,
    pub comp_base: u64,
    pub stride: u32,
    pub comp_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_timestamp {
    pub handle: u32,
    pub offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_timestamps {
    pub start: drm_asahi_timestamp,
    pub end: drm_asahi_timestamp,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_helper_program {
    pub binary: u32,
    pub cfg: u32,
    pub data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_bg_eot {
    pub usc: u32,
    pub rsrc_spec: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_cmd_render {
    pub flags: u32,
    pub isp_zls_pixels: u32,
    pub vdm_ctrl_stream_base: u64,
    pub vertex_helper: drm_asahi_helper_program,
    pub fragment_helper: drm_asahi_helper_program,
    pub isp_scissor_base: u64,
    pub isp_dbias_base: u64,
    pub isp_oclqry_base: u64,
    pub depth: drm_asahi_zls_buffer,
    pub stencil: drm_asahi_zls_buffer,
    pub zls_ctrl: u64,
    pub ppp_multisamplectl: u64,
    pub sampler_heap: u64,
    pub ppp_ctrl: u32,
    pub width_px: u16,
    pub height_px: u16,
    pub layers: u16,
    pub sampler_count: u16,
    pub utile_width_px: u8,
    pub utile_height_px: u8,
    pub samples: u8,
    pub sample_size_b: u8,
    pub isp_merge_upper_x: u32,
    pub isp_merge_upper_y: u32,
    pub bg: drm_asahi_bg_eot,
    pub eot: drm_asahi_bg_eot,
    pub partial_bg: drm_asahi_bg_eot,
    pub partial_eot: drm_asahi_bg_eot,
    pub isp_bgobjdepth: u32,
    pub isp_bgobjvals: u32,
    pub ts_vtx: drm_asahi_timestamps,
    pub ts_frag: drm_asahi_timestamps,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct drm_asahi_cmd_compute {
    pub flags: u32,
    pub sampler_count: u32,
    pub cdm_ctrl_stream_base: u64,
    pub cdm_ctrl_stream_end: u64,
    pub sampler_heap: u64,
    pub helper: drm_asahi_helper_program,
    pub ts: drm_asahi_timestamps,
}

#[repr(C)]
#[derive(Default)]
pub struct drm_asahi_get_time {
    pub flags: u64,
    pub gpu_timestamp: u64,
}

const BASE: c_ulong = Ioctl::COMMAND_BASE;

pub const DRM_IOCTL_ASAHI_GET_PARAMS: c_ulong = Ioctl::w::<drm_asahi_get_params>(BASE);
pub const DRM_IOCTL_ASAHI_GET_TIME: c_ulong = Ioctl::wr::<drm_asahi_get_time>(BASE + 0x01);
pub const DRM_IOCTL_ASAHI_VM_CREATE: c_ulong = Ioctl::wr::<drm_asahi_vm_create>(BASE + 0x02);
pub const DRM_IOCTL_ASAHI_VM_DESTROY: c_ulong = Ioctl::w::<drm_asahi_vm_destroy>(BASE + 0x03);
pub const DRM_IOCTL_ASAHI_VM_BIND: c_ulong = Ioctl::w::<drm_asahi_vm_bind>(BASE + 0x04);
pub const DRM_IOCTL_ASAHI_GEM_CREATE: c_ulong = Ioctl::wr::<drm_asahi_gem_create>(BASE + 0x05);
pub const DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET: c_ulong = Ioctl::wr::<drm_asahi_gem_mmap_offset>(BASE + 0x06);
pub const DRM_IOCTL_ASAHI_GEM_BIND_OBJECT: c_ulong = Ioctl::wr::<drm_asahi_gem_bind_object>(BASE + 0x07);
pub const DRM_IOCTL_ASAHI_QUEUE_CREATE: c_ulong = Ioctl::wr::<drm_asahi_queue_create>(BASE + 0x08);
pub const DRM_IOCTL_ASAHI_QUEUE_DESTROY: c_ulong = Ioctl::w::<drm_asahi_queue_destroy>(BASE + 0x09);
pub const DRM_IOCTL_ASAHI_SUBMIT: c_ulong = Ioctl::w::<drm_asahi_submit>(BASE + 0x0a);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_kernel() {
        let sizes = [
            (size_of::<drm_asahi_params_global>(), 592),
            (size_of::<drm_asahi_get_params>(), 24),
            (size_of::<drm_asahi_vm_create>(), 24),
            (size_of::<drm_asahi_vm_destroy>(), 8),
            (size_of::<drm_asahi_gem_create>(), 24),
            (size_of::<drm_asahi_gem_mmap_offset>(), 16),
            (size_of::<drm_asahi_gem_bind_op>(), 32),
            (size_of::<drm_asahi_vm_bind>(), 24),
            (size_of::<drm_asahi_gem_bind_object>(), 40),
            (size_of::<drm_asahi_queue_create>(), 24),
            (size_of::<drm_asahi_queue_destroy>(), 8),
            (size_of::<drm_asahi_sync>(), 16),
            (size_of::<drm_asahi_cmd_header>(), 8),
            (size_of::<drm_asahi_submit>(), 40),
            (size_of::<drm_asahi_attachment>(), 24),
            (size_of::<drm_asahi_zls_buffer>(), 24),
            (size_of::<drm_asahi_timestamps>(), 16),
            (size_of::<drm_asahi_helper_program>(), 16),
            (size_of::<drm_asahi_bg_eot>(), 8),
            (size_of::<drm_asahi_cmd_render>(), 240),
            (size_of::<drm_asahi_cmd_compute>(), 64),
            (size_of::<drm_asahi_get_time>(), 16),
            (std::mem::offset_of!(drm_asahi_cmd_render, width_px), 148),
            (std::mem::offset_of!(drm_asahi_cmd_render, bg), 168),
            (std::mem::offset_of!(drm_asahi_cmd_render, ts_frag), 224),
            (std::mem::offset_of!(drm_asahi_cmd_compute, helper), 32),
        ];
        for (i, (got, want)) in sizes.into_iter().enumerate() {
            assert_eq!(got, want, "entry {i}");
        }
        let ioctls = [
            (DRM_IOCTL_ASAHI_GET_PARAMS, 0x4018_6440),
            (DRM_IOCTL_ASAHI_GET_TIME, 0xc010_6441),
            (DRM_IOCTL_ASAHI_VM_CREATE, 0xc018_6442),
            (DRM_IOCTL_ASAHI_VM_DESTROY, 0x4008_6443),
            (DRM_IOCTL_ASAHI_VM_BIND, 0x4018_6444),
            (DRM_IOCTL_ASAHI_GEM_CREATE, 0xc018_6445),
            (DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, 0xc010_6446),
            (DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, 0xc028_6447),
            (DRM_IOCTL_ASAHI_QUEUE_CREATE, 0xc018_6448),
            (DRM_IOCTL_ASAHI_QUEUE_DESTROY, 0x4008_6449),
            (DRM_IOCTL_ASAHI_SUBMIT, 0x4028_644a),
        ];
        for (got, want) in ioctls {
            assert_eq!(got, want);
        }
    }
}
