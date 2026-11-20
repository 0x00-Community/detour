//! A triangle pair on the Radeon (RDNA2, gfx10.3) GFX ring from hand-written PM4: the register state radeonsi leaves at
//! its draw, its shader, vertex and target addresses moved onto ours; ACO's NGG vertex shader and pixel shader as
//! machine words; a 64x64 linear RGBA8 target the CPU zeroes first and reads back after the fence.
//!
//!   cargo run --release --example amd_draw

use std::time::{Duration, Instant};

use detour::{Amd, AmdIb, Bo, Caching, Engine, Fence, Gpu, MemType, Queue, Vm};

/// PM4 type-3 packet header: opcode, then `n` dwords of body
const fn pkt3(op: u32, n: u32) -> u32 {
    3 << 30 | (n - 1) << 16 | op << 8
}
const CLEAR_STATE: u32 = 0x12;
const CONTEXT_CONTROL: u32 = 0x28;
const DRAW_INDEX_AUTO: u32 = 0x2d;
const NUM_INSTANCES: u32 = 0x2f;
const EVENT_WRITE: u32 = 0x46;
/// The register spaces, each by the packet that sets it (its body starts at the dword offset from the space's base)
const CTX: u32 = 0x69;
const SH: u32 = 0x76;
const UC: u32 = 0x79;
/// EVENT_WRITE types
const BREAK_BATCH: u32 = 0xe;
const SQ_NON_EVENT: u32 = 0x26;
/// DRAW_INDEX_AUTO initiator: auto-generated indices
const AUTO_INDEX: u32 = 2;
/// A one-dword NOP the GFX ring takes as padding
const NOP: u32 = 0xffff_1000;
/// s_code_end, the padding after a shader the instruction prefetch may run into
const CODE_END: u32 = 0xbf9f_0000;

/// ACO's NGG vertex shader: position (s[20:23]) and color (s[24:27]) fetched by vertex index, exported
const VS: [u32; 23] = [
    0xbf8f0003, 0x8f008803, 0x93fe00c1, 0xbf880002, 0xf8000941, 0x80808000, 0x92808003, 0xbf8cff0f, 0xbefe0300, 0xbf88000c, 0x4a0a0a0d,
    0xe00c2000, 0x80060005, 0xe0042000, 0x80050405, 0x7e0c0280, 0x7e0e02f2, 0xbf8c3f70, 0xf80008cf, 0x07060504, 0xf800020f, 0x03020100,
    0xbf810000,
];
/// ACO's pixel shader and its epilog: the color interpolated, packed to fp16, exported to MRT0
const PS: [u32; 16] = [
    0xbefc0307, 0xc8000002, 0xc8040102, 0xc8100202, 0xc8080302, 0xc8010003, 0xc8050103, 0xc8110203, 0xc8090303, 0x7e060302, 0x7e040304,
    0x5e000300, 0x5e040702, 0xf8001c0f, 0x80800200, 0xbf810000,
];

/// The register state at radeonsi's draw (space, dword offset, value)
#[rustfmt::skip]
const STATE: &[(u32, u32, u32)] = &[
    (SH, 0x001, 0x00000000), // SPI_SHADER_PGM_RSRC4_PS
    (SH, 0x007, 0x003f0003), // SPI_SHADER_PGM_RSRC3_PS
    (SH, 0x008, 0x00002006), // SPI_SHADER_PGM_LO_PS
    (SH, 0x009, 0x00000080), // SPI_SHADER_PGM_HI_PS
    (SH, 0x00a, 0x002c0001), // SPI_SHADER_PGM_RSRC1_PS
    (SH, 0x00b, 0x0000000e), // SPI_SHADER_PGM_RSRC2_PS
    (SH, 0x00c, 0x00000380), // SPI_SHADER_USER_DATA_PS_0
    (SH, 0x00e, 0x00000000), // SPI_SHADER_USER_DATA_PS_2
    (SH, 0x00f, 0x00000000), // SPI_SHADER_USER_DATA_PS_3
    (SH, 0x010, 0x88888888), // SPI_SHADER_USER_DATA_PS_4
    (SH, 0x030, 0x00000007), // SPI_SHADER_REQ_CTRL_PS
    (SH, 0x032, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_0
    (SH, 0x033, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_1
    (SH, 0x034, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_2
    (SH, 0x035, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_3
    (SH, 0x041, 0x0000ffff), // SPI_SHADER_PGM_RSRC4_VS
    (SH, 0x04c, 0x00000380), // SPI_SHADER_USER_DATA_VS_0
    (SH, 0x070, 0x00000000), // SPI_SHADER_REQ_CTRL_VS
    (SH, 0x072, 0x00000000), // SPI_SHADER_USER_ACCUM_VS_0
    (SH, 0x073, 0x00000000), // SPI_SHADER_USER_ACCUM_VS_1
    (SH, 0x074, 0x00000000), // SPI_SHADER_USER_ACCUM_VS_2
    (SH, 0x075, 0x00000000), // SPI_SHADER_USER_ACCUM_VS_3
    (SH, 0x081, 0x0000ffff), // SPI_SHADER_PGM_RSRC4_GS
    (SH, 0x082, 0x00000000), // SPI_SHADER_USER_DATA_ADDR_LO_GS
    (SH, 0x083, 0x00000000), // SPI_SHADER_USER_DATA_ADDR_HI_GS
    (SH, 0x087, 0x003fffff), // SPI_SHADER_PGM_RSRC3_GS
    (SH, 0x08a, 0x002c0001), // SPI_SHADER_PGM_RSRC1_GS
    (SH, 0x08b, 0x00000028), // SPI_SHADER_PGM_RSRC2_GS
    (SH, 0x08c, 0x00000380), // SPI_SHADER_USER_DATA_GS_0
    (SH, 0x08e, 0x00000000), // SPI_SHADER_USER_DATA_GS_2
    (SH, 0x08f, 0x00000000), // SPI_SHADER_USER_DATA_GS_3
    (SH, 0x090, 0x00000000), // SPI_SHADER_USER_DATA_GS_4
    (SH, 0x091, 0x00000000), // SPI_SHADER_USER_DATA_GS_5
    (SH, 0x098, 0x00800000), // SPI_SHADER_USER_DATA_GS_12
    (SH, 0x099, 0x000c8001), // SPI_SHADER_USER_DATA_GS_13
    (SH, 0x09a, 0x00000006), // SPI_SHADER_USER_DATA_GS_14
    (SH, 0x09b, 0x1104022c), // SPI_SHADER_USER_DATA_GS_15
    (SH, 0x09c, 0x00800008), // SPI_SHADER_USER_DATA_GS_16
    (SH, 0x09d, 0x000c8001), // SPI_SHADER_USER_DATA_GS_17
    (SH, 0x09e, 0x00000006), // SPI_SHADER_USER_DATA_GS_18
    (SH, 0x09f, 0x11038fac), // SPI_SHADER_USER_DATA_GS_19
    (SH, 0x0b2, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_0
    (SH, 0x0b3, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_1
    (SH, 0x0b4, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_2
    (SH, 0x0b5, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_3
    (SH, 0x0c8, 0x00002004), // SPI_SHADER_PGM_LO_ES
    (SH, 0x0c9, 0x00000080), // SPI_SHADER_PGM_HI_ES
    (SH, 0x101, 0x0000ffff), // SPI_SHADER_PGM_RSRC4_HS
    (SH, 0x102, 0x00000000), // SPI_SHADER_USER_DATA_ADDR_LO_HS
    (SH, 0x103, 0x00000000), // SPI_SHADER_USER_DATA_ADDR_HI_HS
    (SH, 0x107, 0xffff003f), // SPI_SHADER_PGM_RSRC3_HS
    (SH, 0x10c, 0x00000380), // SPI_SHADER_USER_DATA_HS_0
    (SH, 0x132, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_0
    (SH, 0x133, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_1
    (SH, 0x134, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_2
    (SH, 0x135, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_3
    (SH, 0x149, 0x00000080), // SPI_SHADER_PGM_HI_LS
    (SH, 0x20d, 0x00000080), // COMPUTE_PGM_HI
    (SH, 0x216, 0xffffffff), // COMPUTE_DESTINATION_EN_SE0
    (SH, 0x217, 0x00000000), // COMPUTE_DESTINATION_EN_SE1
    (SH, 0x219, 0x00000000), // COMPUTE_DESTINATION_EN_SE2
    (SH, 0x21a, 0x00000000), // COMPUTE_DESTINATION_EN_SE3
    (SH, 0x224, 0x00000000), // COMPUTE_USER_ACCUM_0
    (SH, 0x225, 0x00000000), // COMPUTE_USER_ACCUM_1
    (SH, 0x226, 0x00000000), // COMPUTE_USER_ACCUM_2
    (SH, 0x227, 0x00000000), // COMPUTE_USER_ACCUM_3
    (SH, 0x27d, 0x00000000), // COMPUTE_DISPATCH_TUNNEL
    (CTX, 0x004, 0x08000000), // DB_RENDER_OVERRIDE2
    (CTX, 0x00e, 0x00000002), // DB_DFSM_CONTROL
    (CTX, 0x019, 0x00000000), // DB_VRS_OVERRIDE_CNTL
    (CTX, 0x01f, 0x00000040), // DB_RMI_L2_CACHE_CONTROL
    (CTX, 0x020, 0x01002000), // TA_BC_BASE_ADDR
    (CTX, 0x021, 0x00000080), // TA_BC_BASE_ADDR_HI
    (CTX, 0x082, 0x00400040), // PA_SC_WINDOW_SCISSOR_BR
    (CTX, 0x08c, 0xaa959a6a), // PA_SC_EDGERULE
    (CTX, 0x08d, 0x00020002), // PA_SU_HARDWARE_SCREEN_OFFSET
    (CTX, 0x08e, 0x0000000f), // CB_TARGET_MASK
    (CTX, 0x08f, 0x0000000f), // CB_SHADER_MASK
    (CTX, 0x094, 0x80040004), // PA_SC_VPORT_SCISSOR_0_TL
    (CTX, 0x095, 0x003c003c), // PA_SC_VPORT_SCISSOR_0_BR
    (CTX, 0x0b4, 0x00000000), // PA_SC_VPORT_ZMIN_0
    (CTX, 0x0b5, 0x3f800000), // PA_SC_VPORT_ZMAX_0
    (CTX, 0x104, 0x00440044), // CB_RMI_GL2_CACHE_CONTROL
    (CTX, 0x105, 0x00000000), // CB_BLEND_RED
    (CTX, 0x106, 0x00000000), // CB_BLEND_GREEN
    (CTX, 0x107, 0x00000000), // CB_BLEND_BLUE
    (CTX, 0x108, 0x00000000), // CB_BLEND_ALPHA
    (CTX, 0x109, 0x00000418), // CB_DCC_CONTROL
    (CTX, 0x10c, 0x01000000), // DB_STENCILREFMASK
    (CTX, 0x10d, 0x01000000), // DB_STENCILREFMASK_BF
    (CTX, 0x10f, 0x42000000), // PA_CL_VPORT_XSCALE
    (CTX, 0x110, 0x42000000), // PA_CL_VPORT_XOFFSET
    (CTX, 0x111, 0x42000000), // PA_CL_VPORT_YSCALE
    (CTX, 0x112, 0x42000000), // PA_CL_VPORT_YOFFSET
    (CTX, 0x113, 0x3f000000), // PA_CL_VPORT_ZSCALE
    (CTX, 0x114, 0x3f000000), // PA_CL_VPORT_ZOFFSET
    (CTX, 0x191, 0x00000000), // SPI_PS_INPUT_CNTL_0
    (CTX, 0x1b1, 0x00000000), // SPI_VS_OUT_CONFIG
    (CTX, 0x1b3, 0x00000002), // SPI_PS_INPUT_ENA
    (CTX, 0x1b4, 0x00009003), // SPI_PS_INPUT_ADDR
    (CTX, 0x1b5, 0x0000486b), // SPI_INTERP_CONTROL_0
    (CTX, 0x1b6, 0x00000001), // SPI_PS_IN_CONTROL
    (CTX, 0x1c2, 0x00000001), // SPI_SHADER_IDX_FORMAT
    (CTX, 0x1c3, 0x00000004), // SPI_SHADER_POS_FORMAT
    (CTX, 0x1c4, 0x00000000), // SPI_SHADER_Z_FORMAT
    (CTX, 0x1c5, 0x00000004), // SPI_SHADER_COL_FORMAT
    (CTX, 0x1d4, 0x000000ff), // SX_PS_DOWNCONVERT_CONTROL
    (CTX, 0x1d5, 0x00000005), // SX_PS_DOWNCONVERT
    (CTX, 0x1d6, 0x00000006), // SX_BLEND_OPT_EPSILON
    (CTX, 0x1d7, 0x00000000), // SX_BLEND_OPT_CONTROL
    (CTX, 0x1d8, 0x06000600), // SX_MRT0_BLEND_OPT
    (CTX, 0x1e0, 0x00000000), // CB_BLEND0_CONTROL
    (CTX, 0x1ff, 0x00000080), // GE_MAX_OUTPUT_PER_SUBGROUP
    (CTX, 0x201, 0x00130000), // DB_EQAA
    (CTX, 0x202, 0x00cc0010), // CB_COLOR_CONTROL
    (CTX, 0x203, 0x00000010), // DB_SHADER_CONTROL
    (CTX, 0x204, 0x01000000), // PA_CL_CLIP_CNTL
    (CTX, 0x205, 0x00080244), // PA_SU_SC_MODE_CNTL
    (CTX, 0x206, 0x0000043f), // PA_CL_VTE_CNTL
    (CTX, 0x207, 0x60000000), // PA_CL_VS_OUT_CNTL
    (CTX, 0x20b, 0xc0000000), // PA_SU_PRIM_FILTER_CNTL
    (CTX, 0x20c, 0x00000001), // PA_SU_SMALL_PRIM_FILTER_CNTL
    (CTX, 0x20e, 0x00000078), // PA_CL_NGG_CNTL
    (CTX, 0x212, 0x00000201), // PA_CL_VRS_CNTL
    (CTX, 0x280, 0x00080008), // PA_SU_POINT_SIZE
    (CTX, 0x281, 0x00080008), // PA_SU_POINT_MINMAX
    (CTX, 0x282, 0x00000008), // PA_SU_LINE_CNTL
    (CTX, 0x286, 0x42800000), // VGT_HOS_MAX_TESS_LEVEL
    (CTX, 0x291, 0x00040080), // VGT_GS_ONCHIP_CNTL
    (CTX, 0x292, 0x00000022), // PA_SC_MODE_CNTL_0
    (CTX, 0x293, 0x760201b5), // PA_SC_MODE_CNTL_1
    (CTX, 0x29b, 0x00000002), // VGT_GS_OUT_PRIM_TYPE
    (CTX, 0x2ab, 0x00000001), // VGT_ESGS_RING_ITEMSIZE
    (CTX, 0x2ce, 0x00000001), // VGT_GS_MAX_VERT_OUT
    (CTX, 0x2d3, 0x00000001), // GE_NGG_SUBGRP_CNTL
    (CTX, 0x2d4, 0xd8181e0c), // VGT_TESS_DISTRIBUTION
    (CTX, 0x2d5, 0x06412010), // VGT_SHADER_STAGES_EN
    (CTX, 0x2dc, 0x0000aa00), // DB_ALPHA_TO_MASK
    (CTX, 0x2e5, 0x00000000), // VGT_STRMOUT_CONFIG
    (CTX, 0x2e6, 0x00000000), // VGT_STRMOUT_BUFFER_CONFIG
    (CTX, 0x2f5, 0x00000000), // PA_SC_CENTROID_PRIORITY_0
    (CTX, 0x2f6, 0x00000000), // PA_SC_CENTROID_PRIORITY_1
    (CTX, 0x2f7, 0x00000000), // PA_SC_LINE_CNTL
    (CTX, 0x2f8, 0x00000000), // PA_SC_AA_CONFIG
    (CTX, 0x2f9, 0x0000003d), // PA_SU_VTX_CNTL
    (CTX, 0x2fa, 0x42800000), // PA_CL_GB_VERT_CLIP_ADJ
    (CTX, 0x2fb, 0x3f800000), // PA_CL_GB_VERT_DISC_ADJ
    (CTX, 0x2fc, 0x42800000), // PA_CL_GB_HORZ_CLIP_ADJ
    (CTX, 0x2fd, 0x3f800000), // PA_CL_GB_HORZ_DISC_ADJ
    (CTX, 0x2fe, 0x00000000), // PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0
    (CTX, 0x302, 0x00000000), // PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y0_0
    (CTX, 0x306, 0x00000000), // PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y1_0
    (CTX, 0x30a, 0x00000000), // PA_SC_AA_SAMPLE_LOCS_PIXEL_X1Y1_0
    (CTX, 0x311, 0x19fc00a0), // PA_SC_BINNER_CNTL_0
    (CTX, 0x312, 0x03ff0054), // PA_SC_BINNER_CNTL_1
    (CTX, 0x314, 0x00000200), // PA_SC_NGG_MODE_CNTL
    (CTX, 0x316, 0x0000000e), // VGT_VERTEX_REUSE_BLOCK_CNTL
    (CTX, 0x318, 0x01006000), // CB_COLOR0_BASE
    (CTX, 0x319, 0x00000000), // CB_COLOR0_PITCH
    (CTX, 0x31a, 0x00000000), // CB_COLOR0_SLICE
    (CTX, 0x31b, 0x00000000), // CB_COLOR0_VIEW
    (CTX, 0x31c, 0x00028028), // CB_COLOR0_INFO
    (CTX, 0x31d, 0x00000000), // CB_COLOR0_ATTRIB
    (CTX, 0x31e, 0x00100038), // CB_COLOR0_DCC_CONTROL
    (CTX, 0x31f, 0x01006000), // CB_COLOR0_CMASK
    (CTX, 0x320, 0x00000000), // CB_COLOR0_CMASK_SLICE
    (CTX, 0x321, 0x01006000), // CB_COLOR0_FMASK
    (CTX, 0x322, 0x00000000), // CB_COLOR0_FMASK_SLICE
    (CTX, 0x323, 0x00000000), // CB_COLOR0_CLEAR_WORD0
    (CTX, 0x324, 0x00000000), // CB_COLOR0_CLEAR_WORD1
    (CTX, 0x325, 0x00000000), // CB_COLOR0_DCC_BASE
    (CTX, 0x390, 0x00ffff80), // CB_COLOR0_BASE_EXT
    (CTX, 0x398, 0x00ffff80), // CB_COLOR0_CMASK_BASE_EXT
    (CTX, 0x3a0, 0x00ffff80), // CB_COLOR0_FMASK_BASE_EXT
    (CTX, 0x3a8, 0x00000000), // CB_COLOR0_DCC_BASE_EXT
    (CTX, 0x3b0, 0x000fc03f), // CB_COLOR0_ATTRIB2
    (CTX, 0x3b8, 0x0d000000), // CB_COLOR0_ATTRIB3
    (UC, 0x07b, 0x00000020), // CP_COHER_START_DELAY
    (UC, 0x242, 0x00000004), // VGT_PRIMITIVE_TYPE
    (UC, 0x249, 0x00000000), // GE_MIN_VTX_INDX
    (UC, 0x24a, 0x00000000), // GE_INDX_OFFSET
    (UC, 0x24b, 0x00000000), // GE_MULTI_PRIM_IB_RESET_EN
    (UC, 0x259, 0xffffffff), // GE_MAX_VTX_INDX
    (UC, 0x25a, 0x00000000), // VGT_INSTANCE_BASE_ID
    (UC, 0x25b, 0x00010080), // GE_CNTL
    (UC, 0x25f, 0x00000000), // GE_STEREO_CNTL
    (UC, 0x260, 0x000007fe), // GE_PC_ALLOC
    (UC, 0x262, 0x00000000), // GE_USER_VGPR_EN
    (UC, 0x280, 0x00000000), // PA_SU_LINE_STIPPLE_VALUE
    (UC, 0x281, 0x00000000), // PA_SC_LINE_STIPPLE_STATE
    (UC, 0x380, 0x01002000), // TA_CS_BC_BASE_ADDR
    (UC, 0x381, 0x00000080), // TA_CS_BC_BASE_ADDR_HI
];

/// The target's side in pixels
const SIDE: usize = 64;
/// Where each piece sits in the one buffer object
const SHADERS: u64 = 0;
const VERTICES: u64 = 0x1_0000;
const TARGET: u64 = 0x2_0000;
const BORDER: u64 = 0x3_0000;
const IB: u64 = 0x4_0000;
const SIZE: u64 = 0x5_0000;

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn main() {
    let amd = Amd::open_first().expect("an amdgpu render node");
    let base = amd.driver_info().virtual_address_offset.max(1 << 32);
    let (vs, ps, vtx, rt, bc, ib_addr) = (base + SHADERS, base + SHADERS + 0x100, base + VERTICES, base + TARGET, base + BORDER, base + IB);

    let mut state = STATE.to_vec();
    let mut set = |op: u32, off: u32, v: u64| state.iter_mut().find(|r| (r.0, r.1) == (op, off)).expect("a register in the table").2 = v as u32;
    for (lo, a) in [(0x008, ps), (0x0c8, vs)] {
        set(SH, lo, a >> 8);
        set(SH, lo + 1, a >> 40);
    }
    // the two vertex buffer descriptors: base, then base's high bits with the 12-byte stride
    for (d, a) in [(0x098, vtx), (0x09c, vtx + 8)] {
        set(SH, d, a);
        set(SH, d + 1, (a >> 32 & 0xffff) | 12 << 16);
    }
    for (lo, ext) in [(0x318, 0x390), (0x31f, 0x398), (0x321, 0x3a0)] {
        set(CTX, lo, rt >> 8);
        set(CTX, ext, rt >> 40);
    }
    for (space, lo) in [(CTX, 0x020), (UC, 0x380)] {
        set(space, lo, bc >> 8);
        set(space, lo + 1, bc >> 40);
    }

    let mut ib = vec![pkt3(CONTEXT_CONTROL, 2), 0x8000_0000, 0x8000_0000, pkt3(EVENT_WRITE, 1), BREAK_BATCH, pkt3(CLEAR_STATE, 1), 0];
    // each run of consecutive registers in one packet
    let mut k = 0;
    while k < state.len() {
        let (op, off, _) = state[k];
        let n = state[k..].iter().enumerate().take_while(|(j, r)| r.0 == op && r.1 == off + *j as u32).count();
        ib.extend([pkt3(op, 1 + n as u32), off]);
        ib.extend(state[k..k + n].iter().map(|r| r.2));
        k += n;
    }
    ib.extend([pkt3(EVENT_WRITE, 1), SQ_NON_EVENT, pkt3(NUM_INSTANCES, 1), 1, pkt3(DRAW_INDEX_AUTO, 2), 6, AUTO_INDEX]);
    while !ib.len().is_multiple_of(8) {
        ib.push(NOP);
    }

    let mut bo = amd.bo((SIZE, Caching::WriteBack).into()).unwrap();
    let b = bo.bytes().unwrap();
    b.fill(0);
    let code = &mut b[SHADERS as usize..VERTICES as usize];
    code.copy_from_slice(&words(&vec![CODE_END; code.len() / 4]));
    code[..4 * VS.len()].copy_from_slice(&words(&VS));
    code[0x100..0x100 + 4 * PS.len()].copy_from_slice(&words(&PS));
    // x, y, rgba8 per vertex
    let verts: [(f32, f32, [u8; 4]); 6] = [
        (-1.0, -1.0, [255, 0, 0, 255]),
        (1.0, -1.0, [0, 255, 0, 255]),
        (1.0, 1.0, [0, 0, 255, 255]),
        (-1.0, -1.0, [255, 0, 0, 255]),
        (1.0, 1.0, [0, 0, 255, 255]),
        (-1.0, 1.0, [255, 255, 255, 128]),
    ];
    let v: Vec<u8> = verts.iter().flat_map(|(x, y, c)| [x.to_le_bytes(), y.to_le_bytes(), *c].concat()).collect();
    b[VERTICES as usize..][..v.len()].copy_from_slice(&v);
    b[IB as usize..][..4 * ib.len()].copy_from_slice(&words(&ib));

    let vm = amd.vm().unwrap();
    let queue = amd.queue(&vm, Engine::Render).unwrap();
    vm.bind(&bo, 0, base, SIZE, MemType::Streaming, &[]).unwrap();
    let done = amd.fence(false).unwrap();
    let t = Instant::now();
    queue.submit(&AmdIb { addr: ib_addr, dwords: ib.len() as u32 }, &[], &[&done]).unwrap();
    assert!(done.wait(Duration::from_secs(5)).unwrap(), "the draw did not finish");
    let took = t.elapsed();
    println!("{} dwords drawn in {:.3} ms; fault: {:?}", ib.len(), took.as_secs_f64() * 1e3, queue.fault(&done).unwrap());
    let px = &bo.bytes().unwrap()[TARGET as usize..][..SIDE * SIDE * 4];
    let at = |x: usize, y: usize| &px[4 * (y * SIDE + x)..][..4];
    let lit = px.chunks(4).filter(|p| p.iter().any(|&c| c != 0)).count();
    println!("(32,32) {:?}  (1,1) {:?}  (8,56) {:?}  {lit} pixels lit", at(32, 32), at(1, 1), at(8, 56));
    assert_eq!((at(32, 32), at(1, 1)), (&[126, 0, 129, 255][..], &[0, 0, 0, 0][..]), "the pixels radeonsi draws");
    println!("ok   the pixels radeonsi draws");
    vm.unbind(base, SIZE, &[]).unwrap();
}
