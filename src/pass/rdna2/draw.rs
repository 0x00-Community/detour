//! A draw on RDNA2 (gfx10.3) in PM4: the register state radeonsi leaves at a draw (Mesa 26.2 on the Radeon 610M),
//! what a draw's program, target, blend, rasterizer, viewport and scissor set over it, the descriptors its shaders
//! read (V#, T#, S#) and the draw packets. The device (`render::amd`) writes them into its command buffers.

use alloc::format;
use alloc::vec::Vec;

use mlirformat::Error;

use crate::{BlendEquation, BlendFactor, ColorFormat, CullMode, Draw, DrawPrimitive, Filter, IndexFormat, Rdna2Shader, ShaderStage, VertexFormat, Wrap};

/// A register write: the packet that sets its space (`Rdna2Packet::CTX`, `SH`, `UC`), its dword offset in the space,
/// its value
pub type Rdna2Write = (u32, u32, u32);

/// PM4 type-3 packets
pub struct Rdna2Packet;

impl Rdna2Packet {
    pub const NOP: u32 = 0x10;
    pub const CLEAR_STATE: u32 = 0x12;
    pub const DRAW_INDEX_2: u32 = 0x27;
    pub const CONTEXT_CONTROL: u32 = 0x28;
    pub const DRAW_INDEX_AUTO: u32 = 0x2d;
    pub const NUM_INSTANCES: u32 = 0x2f;
    pub const EVENT_WRITE: u32 = 0x46;
    pub const ACQUIRE_MEM: u32 = 0x58;
    /// The register spaces, each by the packet that sets it
    pub const CTX: u32 = 0x69;
    pub const SH: u32 = 0x76;
    pub const UC: u32 = 0x79;
    pub const UC_INDEX: u32 = 0x7a;
    /// EVENT_WRITE types
    pub const BREAK_BATCH: u32 = 0xe;
    pub const SQ_NON_EVENT: u32 = 0x26;
    /// The GFX ring's one-dword padding
    pub const PAD: u32 = 0xffff_1000;

    /// A packet's header: opcode `op`, then `n` dwords of body
    pub const fn header(op: u32, n: u32) -> u32 {
        3 << 30 | (n - 1) << 16 | op << 8
    }

    /// `out` padded to a multiple of 8 dwords (the GFX ring fetches IBs so)
    pub fn pad(out: &mut Vec<u32>) {
        while !out.len().is_multiple_of(8) {
            out.push(Rdna2Packet::PAD);
        }
    }
}

/// The register state at radeonsi's draw (space, dword offset, value): its preamble and a plain draw's state, the
/// addresses, shaders and target each draw sets again
#[rustfmt::skip]
pub const RDNA2_DRAW_STATE: &[Rdna2Write] = &[
    (SH, 0x001, 0x00000000), // SPI_SHADER_PGM_RSRC4_PS
    (SH, 0x007, 0x003f0003), // SPI_SHADER_PGM_RSRC3_PS
    (SH, 0x008, 0x00000000), // SPI_SHADER_PGM_LO_PS
    (SH, 0x009, 0x00000000), // SPI_SHADER_PGM_HI_PS
    (SH, 0x00a, 0x002c0001), // SPI_SHADER_PGM_RSRC1_PS
    (SH, 0x00b, 0x00000000), // SPI_SHADER_PGM_RSRC2_PS
    (SH, 0x030, 0x00000007), // SPI_SHADER_REQ_CTRL_PS
    (SH, 0x032, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_0
    (SH, 0x033, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_1
    (SH, 0x034, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_2
    (SH, 0x035, 0x00000000), // SPI_SHADER_USER_ACCUM_PS_3
    (SH, 0x041, 0x0000ffff), // SPI_SHADER_PGM_RSRC4_VS
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
    (SH, 0x08b, 0x00000000), // SPI_SHADER_PGM_RSRC2_GS
    (SH, 0x0b2, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_0
    (SH, 0x0b3, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_1
    (SH, 0x0b4, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_2
    (SH, 0x0b5, 0x00000000), // SPI_SHADER_USER_ACCUM_ESGS_3
    (SH, 0x0c8, 0x00000000), // SPI_SHADER_PGM_LO_ES
    (SH, 0x0c9, 0x00000000), // SPI_SHADER_PGM_HI_ES
    (SH, 0x101, 0x0000ffff), // SPI_SHADER_PGM_RSRC4_HS
    (SH, 0x102, 0x00000000), // SPI_SHADER_USER_DATA_ADDR_LO_HS
    (SH, 0x103, 0x00000000), // SPI_SHADER_USER_DATA_ADDR_HI_HS
    (SH, 0x107, 0xffff003f), // SPI_SHADER_PGM_RSRC3_HS
    (SH, 0x132, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_0
    (SH, 0x133, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_1
    (SH, 0x134, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_2
    (SH, 0x135, 0x00000000), // SPI_SHADER_USER_ACCUM_LSHS_3
    (CTX, 0x004, 0x08000000), // DB_RENDER_OVERRIDE2
    (CTX, 0x00e, 0x00000002), // DB_DFSM_CONTROL
    (CTX, 0x019, 0x00000000), // DB_VRS_OVERRIDE_CNTL
    (CTX, 0x01f, 0x00000040), // DB_RMI_L2_CACHE_CONTROL
    (CTX, 0x020, 0x00000000), // TA_BC_BASE_ADDR
    (CTX, 0x021, 0x00000000), // TA_BC_BASE_ADDR_HI
    (CTX, 0x082, 0x00400040), // PA_SC_WINDOW_SCISSOR_BR
    (CTX, 0x08c, 0xaa959a6a), // PA_SC_EDGERULE
    (CTX, 0x08d, 0x00020002), // PA_SU_HARDWARE_SCREEN_OFFSET
    (CTX, 0x08e, 0x0000000f), // CB_TARGET_MASK
    (CTX, 0x08f, 0x0000000f), // CB_SHADER_MASK
    (CTX, 0x094, 0x80000000), // PA_SC_VPORT_SCISSOR_0_TL
    (CTX, 0x095, 0x00400040), // PA_SC_VPORT_SCISSOR_0_BR
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
    (CTX, 0x1b4, 0x00000002), // SPI_PS_INPUT_ADDR
    (CTX, 0x1b5, 0x0000486b), // SPI_INTERP_CONTROL_0
    (CTX, 0x1b6, 0x00008001), // SPI_PS_IN_CONTROL
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
    (CTX, 0x204, 0x01080000), // PA_CL_CLIP_CNTL (radeonsi's, and DX_CLIP_SPACE_DEF: z clipped to [0, w])
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
    (CTX, 0x318, 0x00000000), // CB_COLOR0_BASE
    (CTX, 0x319, 0x00000000), // CB_COLOR0_PITCH
    (CTX, 0x31a, 0x00000000), // CB_COLOR0_SLICE
    (CTX, 0x31b, 0x00000000), // CB_COLOR0_VIEW
    (CTX, 0x31c, 0x00028028), // CB_COLOR0_INFO
    (CTX, 0x31d, 0x00000000), // CB_COLOR0_ATTRIB
    (CTX, 0x31e, 0x00100038), // CB_COLOR0_DCC_CONTROL
    (CTX, 0x31f, 0x00000000), // CB_COLOR0_CMASK
    (CTX, 0x320, 0x00000000), // CB_COLOR0_CMASK_SLICE
    (CTX, 0x321, 0x00000000), // CB_COLOR0_FMASK
    (CTX, 0x322, 0x00000000), // CB_COLOR0_FMASK_SLICE
    (CTX, 0x323, 0x00000000), // CB_COLOR0_CLEAR_WORD0
    (CTX, 0x324, 0x00000000), // CB_COLOR0_CLEAR_WORD1
    (CTX, 0x325, 0x00000000), // CB_COLOR0_DCC_BASE
    (CTX, 0x390, 0x00000000), // CB_COLOR0_BASE_EXT
    (CTX, 0x398, 0x00000000), // CB_COLOR0_CMASK_BASE_EXT
    (CTX, 0x3a0, 0x00000000), // CB_COLOR0_FMASK_BASE_EXT
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
    (UC, 0x380, 0x00000000), // TA_CS_BC_BASE_ADDR
    (UC, 0x381, 0x00000000), // TA_CS_BC_BASE_ADDR_HI
];

const CTX: u32 = Rdna2Packet::CTX;
const SH: u32 = Rdna2Packet::SH;
const UC: u32 = Rdna2Packet::UC;

/// The registers a draw sets (dword offsets in their spaces)
mod reg {
    pub const PGM_LO_PS: u32 = 0x008;
    pub const PGM_RSRC1_PS: u32 = 0x00a;
    pub const USER_DATA_PS_0: u32 = 0x00c;
    pub const PGM_RSRC1_GS: u32 = 0x08a;
    pub const USER_DATA_GS_0: u32 = 0x08c;
    pub const PGM_LO_ES: u32 = 0x0c8;
    pub const TA_BC_BASE_ADDR: u32 = 0x020;
    pub const WINDOW_SCISSOR_TL: u32 = 0x081;
    pub const HARDWARE_SCREEN_OFFSET: u32 = 0x08d;
    pub const CB_TARGET_MASK: u32 = 0x08e;
    pub const VPORT_SCISSOR_0_TL: u32 = 0x094;
    pub const VPORT_ZMIN_0: u32 = 0x0b4;
    pub const CB_BLEND_RED: u32 = 0x105;
    pub const VPORT_XSCALE: u32 = 0x10f;
    pub const PS_INPUT_CNTL_0: u32 = 0x191;
    pub const VS_OUT_CONFIG: u32 = 0x1b1;
    pub const PS_INPUT_ENA: u32 = 0x1b3;
    pub const PS_IN_CONTROL: u32 = 0x1b6;
    pub const SX_PS_DOWNCONVERT: u32 = 0x1d5;
    pub const SX_MRT0_BLEND_OPT: u32 = 0x1d8;
    pub const CB_BLEND0_CONTROL: u32 = 0x1e0;
    pub const SC_MODE_CNTL: u32 = 0x205;
    pub const GS_OUT_PRIM_TYPE: u32 = 0x29b;
    pub const VTX_CNTL: u32 = 0x2f9;
    pub const CB_COLOR0_BASE: u32 = 0x318;
    pub const CB_COLOR0_INFO: u32 = 0x31c;
    pub const CB_COLOR0_CMASK: u32 = 0x31f;
    pub const CB_COLOR0_FMASK: u32 = 0x321;
    pub const CB_COLOR0_BASE_EXT: u32 = 0x390;
    pub const CB_COLOR0_CMASK_BASE_EXT: u32 = 0x398;
    pub const CB_COLOR0_FMASK_BASE_EXT: u32 = 0x3a0;
    pub const CB_COLOR0_ATTRIB2: u32 = 0x3b0;
    pub const VGT_PRIMITIVE_TYPE: u32 = 0x242;
    pub const VGT_INDEX_TYPE: u32 = 0x243;
    pub const TA_CS_BC_BASE_ADDR: u32 = 0x380;
}

/// The register spaces' dword offsets a stream may hold (every space's registers lie below)
const SPACE: usize = 0x400;

/// Scissors whose top-left (here) and bottom-right (the next register) are set together: the bottom-right alone set
/// leaves the scissor as it was (radeonsi and RADV always write both)
const PAIRED: [u32; 2] = [reg::WINDOW_SCISSOR_TL, reg::VPORT_SCISSOR_0_TL];

/// What a command stream has set: each register's last value, to set only those a draw changes (a context register
/// written rolls the context, even to the value it has)
pub struct Rdna2Shadow {
    known: Vec<Option<u32>>,
    changed: Vec<Rdna2Write>,
}

impl Default for Rdna2Shadow {
    fn default() -> Self {
        Rdna2Shadow { known: alloc::vec![None; 3 * SPACE], changed: Vec::new() }
    }
}

impl Rdna2Shadow {
    fn slot(op: u32, off: u32) -> usize {
        let space = match op {
            CTX => 0,
            SH => 1,
            _ => 2,
        };
        space * SPACE + off as usize
    }

    /// Nothing known: a stream that starts over (CLEAR_STATE)
    pub fn forget(&mut self) {
        self.known.fill(None);
    }

    /// `regs` set in `out` where the stream does not hold their values: each run of consecutive registers of a space
    /// in one packet
    pub fn set(&mut self, out: &mut Vec<u32>, regs: &[Rdna2Write]) {
        self.changed.clear();
        for &(op, off, v) in regs {
            let slot = &mut self.known[Rdna2Shadow::slot(op, off)];
            if *slot != Some(v) {
                *slot = Some(v);
                self.changed.push((op, off, v));
            }
        }
        for tl in PAIRED {
            let has = |c: &[Rdna2Write], off: u32| c.iter().any(|r| r.0 == CTX && r.1 == off);
            for (set, other) in [(tl, tl + 1), (tl + 1, tl)] {
                if has(&self.changed, set) && !has(&self.changed, other)
                    && let Some(v) = self.known[Rdna2Shadow::slot(CTX, other)]
                {
                    self.changed.push((CTX, other, v));
                }
            }
        }
        self.changed.sort_unstable_by_key(|r| Rdna2Shadow::slot(r.0, r.1));
        let mut k = 0;
        while k < self.changed.len() {
            let (op, off, _) = self.changed[k];
            let n = self.changed[k..].iter().enumerate().take_while(|(j, r)| r.0 == op && r.1 == off + *j as u32).count();
            out.extend([Rdna2Packet::header(op, 1 + n as u32), off]);
            out.extend(self.changed[k..k + n].iter().map(|r| r.2));
            k += n;
        }
    }
}

/// A surface as a draw sees it, drawn into or sampled: where it is, its size, its rows' pitch in pixels (a multiple of
/// 256 bytes) and format
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rdna2Surface {
    pub addr: u64,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub format: ColorFormat,
}

/// A draw's registers, descriptors and packets
pub struct Rdna2Draw;

impl Rdna2Draw {
    /// A surface's rows' bytes: gfx10.3 reads and writes a linear surface whose pitch is a multiple of 256 bytes
    pub const PITCH_ALIGN: u32 = 256;

    /// A command stream's start: the context reset (CLEAR_STATE), the caches the GPU reads invalidated (written
    /// back), radeonsi's state, the border colors at `border` (none is used: a clamp is to the edge)
    pub fn preamble(out: &mut Vec<u32>, shadow: &mut Rdna2Shadow, border: u64) {
        out.extend([Rdna2Packet::header(Rdna2Packet::CONTEXT_CONTROL, 2), 0x8000_0000, 0x8000_0000]);
        out.extend([Rdna2Packet::header(Rdna2Packet::EVENT_WRITE, 1), Rdna2Packet::BREAK_BATCH]);
        out.extend([Rdna2Packet::header(Rdna2Packet::CLEAR_STATE, 1), 0]);
        // ACQUIRE_MEM as radeonsi's: the whole address space; GCR_CNTL: GLI, GLM, GLK, GLV, GL1, GL2 invalidated,
        // GLM and GL2 written back
        out.extend([Rdna2Packet::header(Rdna2Packet::ACQUIRE_MEM, 7), 0x8000_0000, 0xffff_ffff, 0xff, 0, 0, 0xa, 0xc3b1]);
        shadow.forget();
        shadow.set(out, RDNA2_DRAW_STATE);
        let (lo, hi) = ((border >> 8) as u32, (border >> 40) as u32);
        shadow.set(out, &[(CTX, reg::TA_BC_BASE_ADDR, lo), (CTX, reg::TA_BC_BASE_ADDR + 1, hi), (UC, reg::TA_CS_BC_BASE_ADDR, lo), (UC, reg::TA_CS_BC_BASE_ADDR + 1, hi)]);
    }

    /// Whether program `vs`, `fs` draws as the ABI (`Rdna2Shader`) has it: the fragment shader reads varyings the
    /// vertex shader writes, interpolates some (the hardware needs one pair of weights), its user SGPRs fit
    pub fn check(vs: &Rdna2Shader, fs: &Rdna2Shader) -> Result<(), Error> {
        if vs.stage != ShaderStage::Vertex || fs.stage != ShaderStage::Fragment {
            return Err(Error(format!("a program of a {:?} and a {:?} shader", vs.stage, fs.stage)));
        }
        if fs.inputs > vs.varyings {
            return Err(Error(format!("the fragment shader reads {} varyings, the vertex shader writes {}", fs.inputs, vs.varyings)));
        }
        if fs.ps_input & 0x7f == 0 {
            return Err(Error(format!("SPI_PS_INPUT_ENA {:#x}: no interpolation weights (ACO enables PERSP_CENTER then)", fs.ps_input)));
        }
        for s in [vs, fs] {
            if s.user_sgprs() > 32 {
                return Err(Error(format!("a {:?} shader of {} user SGPRs, a stage has 32", s.stage, s.user_sgprs())));
            }
        }
        Ok(())
    }

    /// SPI_SHADER_PGM_RSRC1's VGPRS: wave32 allocates VGPRs 8 at a time
    fn vgpr_blocks(vgprs: u32) -> u32 {
        (vgprs.max(1) - 1) / 8
    }

    /// SPI_SHADER_PGM_RSRC2's USER_SGPR and USER_SGPR_MSB
    fn user_sgpr(n: u32) -> u32 {
        (n & 31) << 1 | (n >> 5) << 27
    }

    /// The program's registers: its shaders at `vs_addr` and `fs_addr` (on 256 bytes), the varyings between them
    pub fn program(out: &mut Vec<Rdna2Write>, vs: &Rdna2Shader, fs: &Rdna2Shader, vs_addr: u64, fs_addr: u64) {
        out.extend([
            (SH, reg::PGM_LO_PS, (fs_addr >> 8) as u32),
            (SH, reg::PGM_LO_PS + 1, (fs_addr >> 40) as u32),
            // radeonsi's FLOAT_MODE (fp16/64 denormals) and DX10_CLAMP
            (SH, reg::PGM_RSRC1_PS, 0x002c_0000 | Rdna2Draw::vgpr_blocks(fs.vgprs)),
            (SH, reg::PGM_RSRC1_PS + 1, Rdna2Draw::user_sgpr(fs.user_sgprs())),
            (SH, reg::PGM_RSRC1_GS, 0x002c_0000 | Rdna2Draw::vgpr_blocks(vs.vgprs)),
            (SH, reg::PGM_RSRC1_GS + 1, Rdna2Draw::user_sgpr(vs.user_sgprs())),
            (SH, reg::PGM_LO_ES, (vs_addr >> 8) as u32),
            (SH, reg::PGM_LO_ES + 1, (vs_addr >> 40) as u32),
        ]);
        out.extend((0..fs.inputs).map(|k| (CTX, reg::PS_INPUT_CNTL_0 + k, k)));
        let exports = vs.varyings.max(1) - 1;
        out.extend([
            (CTX, reg::VS_OUT_CONFIG, exports << 1 | u32::from(vs.varyings == 0) << 7),
            (CTX, reg::PS_INPUT_ENA, fs.ps_input),
            (CTX, reg::PS_INPUT_ENA + 1, fs.ps_input),
            // NUM_INTERP, PS_W32_EN
            (CTX, reg::PS_IN_CONTROL, fs.inputs | 1 << 15),
        ]);
    }

    /// CB_COLOR0_INFO (COLOR_8_8_8_8, COLOR_8, COLOR_32_32_32_32; UNORM or FLOAT; BLEND_CLAMP for UNORM; SIMPLE_FLOAT;
    /// ROUND_MODE for FLOAT) and CB_COLOR0_ATTRIB (FORCE_DST_ALPHA_1 for one channel), as radeonsi's
    fn color_info(format: ColorFormat) -> (u32, u32) {
        match format {
            ColorFormat::Rgba8Unorm => (10 << 2 | 1 << 15 | 1 << 17, 0),
            ColorFormat::R8Unorm => (1 << 2 | 1 << 15 | 1 << 17, 1 << 17),
            ColorFormat::Rgba32Float => (14 << 2 | 7 << 8 | 1 << 17 | 1 << 18, 0),
        }
    }

    /// The target's registers, its color mask's and the export's down-conversion (RB+), as radeonsi's
    pub fn target(out: &mut Vec<Rdna2Write>, t: &Rdna2Surface, mask: [bool; 4]) {
        let (info, attrib) = Rdna2Draw::color_info(t.format);
        let (lo, hi) = ((t.addr >> 8) as u32, (t.addr >> 40) as u32);
        let writes = mask.iter().enumerate().fold(0, |m, (c, on)| m | u32::from(*on) << c);
        // SX_RT_EXPORT_8_8_8_8 and 8BIT_FORMAT_0_5 for 8-bit channels; MRT0_*_OPT_DISABLE where nothing is kept
        let (downconvert, epsilon) = match t.format {
            ColorFormat::Rgba8Unorm | ColorFormat::R8Unorm => (5, 6),
            ColorFormat::Rgba32Float => (0, 0),
        };
        let alpha = t.format != ColorFormat::R8Unorm && mask[3];
        let rgb = mask[..3].iter().any(|m| *m);
        let control = u32::from(!rgb) | u32::from(!alpha) << 1;
        out.extend([
            // the window offset off, the window from (0, 0)
            (CTX, reg::WINDOW_SCISSOR_TL, 1 << 31),
            (CTX, reg::WINDOW_SCISSOR_TL + 1, t.width | t.height << 16),
            (CTX, reg::CB_TARGET_MASK, writes),
            (CTX, reg::SX_PS_DOWNCONVERT, downconvert),
            (CTX, reg::SX_PS_DOWNCONVERT + 1, epsilon),
            (CTX, reg::SX_PS_DOWNCONVERT + 2, control),
            (CTX, reg::CB_COLOR0_BASE, lo),
            (CTX, reg::CB_COLOR0_INFO, info),
            (CTX, reg::CB_COLOR0_INFO + 1, attrib),
            (CTX, reg::CB_COLOR0_CMASK, lo),
            (CTX, reg::CB_COLOR0_FMASK, lo),
            (CTX, reg::CB_COLOR0_BASE_EXT, hi),
            (CTX, reg::CB_COLOR0_CMASK_BASE_EXT, hi),
            (CTX, reg::CB_COLOR0_FMASK_BASE_EXT, hi),
            // a linear target's pitch in MIP0_WIDTH (gfx10.3: a multiple of 256 bytes), MIP0_HEIGHT
            (CTX, reg::CB_COLOR0_ATTRIB2, (t.height - 1) | (t.pitch - 1) << 14),
        ]);
    }

    fn factor(f: BlendFactor) -> u32 {
        match f {
            BlendFactor::Zero => 0,
            BlendFactor::One => 1,
            BlendFactor::SrcColor => 2,
            BlendFactor::OneMinusSrcColor => 3,
            BlendFactor::SrcAlpha => 4,
            BlendFactor::OneMinusSrcAlpha => 5,
            BlendFactor::DstAlpha => 6,
            BlendFactor::OneMinusDstAlpha => 7,
            BlendFactor::DstColor => 8,
            BlendFactor::OneMinusDstColor => 9,
            BlendFactor::SrcAlphaSaturate => 10,
            BlendFactor::ConstantColor => 13,
            BlendFactor::OneMinusConstantColor => 14,
            BlendFactor::ConstantAlpha => 19,
            BlendFactor::OneMinusConstantAlpha => 20,
        }
    }

    /// CB's COMB_FCN and SX's OPT_COMB_FCN
    fn equation(e: BlendEquation) -> (u32, u32) {
        match e {
            BlendEquation::Add => (0, 1),
            BlendEquation::Subtract => (1, 2),
            BlendEquation::ReverseSubtract => (4, 5),
            BlendEquation::Min => (2, 3),
            BlendEquation::Max => (3, 4),
        }
    }

    /// SX_MRT0_BLEND_OPT's SRC_OPT or DST_OPT for factor `f` (radeonsi's)
    fn opt(f: BlendFactor, alpha: bool) -> u32 {
        match f {
            BlendFactor::Zero => 0,
            BlendFactor::One => 1,
            BlendFactor::SrcColor if alpha => 4,
            BlendFactor::SrcColor => 2,
            BlendFactor::OneMinusSrcColor if alpha => 5,
            BlendFactor::OneMinusSrcColor => 3,
            BlendFactor::SrcAlpha => 4,
            BlendFactor::OneMinusSrcAlpha => 5,
            BlendFactor::SrcAlphaSaturate if alpha => 1,
            BlendFactor::SrcAlphaSaturate => 6,
            _ => 7,
        }
    }

    fn uses_dst(f: BlendFactor) -> bool {
        matches!(f, BlendFactor::DstColor | BlendFactor::OneMinusDstColor | BlendFactor::DstAlpha | BlendFactor::OneMinusDstAlpha | BlendFactor::SrcAlphaSaturate)
    }

    /// The blend's registers: CB_BLEND0_CONTROL (RGB and alpha apart), SX's view of it, the constant color
    pub fn blend(out: &mut Vec<Rdna2Write>, d: &Draw<'_>) {
        let Some(b) = d.blend else {
            out.extend([(CTX, reg::SX_MRT0_BLEND_OPT, 0x0600_0600), (CTX, reg::CB_BLEND0_CONTROL, 0)]);
            return;
        };
        let (mut control, mut opt) = (1 << 30 | 1 << 29, 0);
        for c in 0..2 {
            let (comb, sx_comb) = Rdna2Draw::equation(b.equation[c]);
            // min and max take no factors
            let (src, dst) = match b.equation[c] {
                BlendEquation::Min | BlendEquation::Max => (BlendFactor::One, BlendFactor::One),
                _ => (b.src[c], b.dst[c]),
            };
            let alpha = c == 1;
            let src_opt = Rdna2Draw::opt(src, alpha);
            let mut dst_opt = Rdna2Draw::opt(dst, alpha);
            if Rdna2Draw::uses_dst(src) {
                dst_opt = 7;
            }
            if src == BlendFactor::SrcAlphaSaturate && matches!(dst, BlendFactor::Zero | BlendFactor::SrcAlpha | BlendFactor::SrcAlphaSaturate) {
                dst_opt = 6;
            }
            let (src_opt, dst_opt) = match b.equation[c] {
                BlendEquation::Min | BlendEquation::Max => (7, 7),
                _ => (src_opt, dst_opt),
            };
            let shift = 16 * c as u32;
            control |= (Rdna2Draw::factor(src) | comb << 5 | Rdna2Draw::factor(dst) << 8) << shift;
            opt |= (src_opt | dst_opt << 4 | sx_comb << 8) << shift;
        }
        out.extend([(CTX, reg::SX_MRT0_BLEND_OPT, opt), (CTX, reg::CB_BLEND0_CONTROL, control)]);
        out.extend(b.color.iter().enumerate().map(|(k, c)| (CTX, reg::CB_BLEND_RED + k as u32, c.to_bits())));
    }

    /// VGT_PRIMITIVE_TYPE's DI_PT and VGT_GS_OUT_PRIM_TYPE's (what NGG hands the rasterizer)
    fn primitive(p: DrawPrimitive) -> (u32, u32) {
        match p {
            DrawPrimitive::Points => (1, 0),
            DrawPrimitive::Lines => (2, 1),
            DrawPrimitive::LineStrip => (3, 1),
            DrawPrimitive::Triangles => (4, 2),
            DrawPrimitive::TriangleStrip => (6, 2),
        }
    }

    /// The rasterizer's registers: the primitive, faces culled, which is front
    pub fn raster(out: &mut Vec<Rdna2Write>, d: &Draw<'_>) {
        let (cull_front, cull_back) = match d.cull {
            CullMode::None => (0, 0),
            CullMode::Front => (1, 0),
            CullMode::Back => (0, 1),
        };
        // radeonsi's (triangles drawn as triangles, the last vertex provoking); FACE: clockwise front on the screen,
        // whose rows run as the viewport's y does: front_ccw
        let mode = 0x0008_0240 | cull_front | cull_back << 1 | u32::from(d.front_ccw) << 2;
        let (prim, out_prim) = Rdna2Draw::primitive(d.primitive);
        out.extend([(CTX, reg::SC_MODE_CNTL, mode), (CTX, reg::GS_OUT_PRIM_TYPE, out_prim), (UC, reg::VGT_PRIMITIVE_TYPE, prim)]);
    }

    /// The pixels a draw may write: its viewport's, within its scissor and target (x0, y0, x1, y1, exclusive);
    /// None when that is none
    pub fn pixels(d: &Draw<'_>) -> Option<[u32; 4]> {
        let [x, y, w, h] = d.viewport;
        let [sx, sy, sw, sh] = d.scissor.unwrap_or([0, 0, d.target.width, d.target.height]);
        let x0 = (x.min(x + w).floor().max(0.0) as u32).max(sx);
        let y0 = (y.min(y + h).floor().max(0.0) as u32).max(sy);
        let x1 = (x.max(x + w).ceil().max(0.0) as u32).min(sx.saturating_add(sw)).min(d.target.width);
        let y1 = (y.max(y + h).ceil().max(0.0) as u32).min(sy.saturating_add(sh)).min(d.target.height);
        (x0 < x1 && y0 < y1).then_some([x0, y0, x1, y1])
    }

    /// The viewport's registers, the scissor (`pixels`'), the guard band (radeonsi's: the screen offset centred on
    /// the viewport, the subpixel precision its size allows)
    pub fn viewport(out: &mut Vec<Rdna2Write>, d: &Draw<'_>, pixels: [u32; 4]) {
        let [x, y, w, h] = d.viewport;
        let [near, far] = d.depth_range;
        let (sx, tx, sy, ty) = (w / 2.0, x + w / 2.0, h / 2.0, y + h / 2.0);
        let f = f32::to_bits;
        out.extend([
            (CTX, reg::VPORT_SCISSOR_0_TL, pixels[0] | pixels[1] << 16 | 1 << 31),
            (CTX, reg::VPORT_SCISSOR_0_TL + 1, pixels[2] | pixels[3] << 16),
            (CTX, reg::VPORT_ZMIN_0, f(near.min(far))),
            (CTX, reg::VPORT_ZMIN_0 + 1, f(near.max(far))),
            (CTX, reg::VPORT_XSCALE, f(sx)),
            (CTX, reg::VPORT_XSCALE + 1, f(tx)),
            (CTX, reg::VPORT_XSCALE + 2, f(sy)),
            (CTX, reg::VPORT_XSCALE + 3, f(ty)),
            (CTX, reg::VPORT_XSCALE + 4, f(far - near)),
            (CTX, reg::VPORT_XSCALE + 5, f(near)),
        ]);
        // the viewport as a scissor, its largest coordinate choosing the quantization (12.12, 14.10, 16.8)
        let (minx, maxx) = ((tx - sx.abs()).floor() as i32, (tx + sx.abs()).ceil() as i32);
        let (miny, maxy) = ((ty - sy.abs()).floor() as i32, (ty + sy.abs()).ceil() as i32);
        let corner = [minx, maxx, miny, maxy].iter().map(|c| c.unsigned_abs()).max().unwrap_or(0);
        let quant = match corner {
            0..=1024 => 2,
            1025..=4096 => 1,
            _ => 0,
        };
        let range = [65536.0, 16384.0, 4096.0][quant] / 2.0;
        let offset = |lo: i32, hi: i32| ((lo + hi) / 2).clamp(0, 8176) & !15;
        let (ox, oy) = (offset(minx, maxx), offset(miny, maxy));
        let band = |scale: f32, translate: f32| {
            let s = scale.abs().max(0.5);
            ((range + translate) / s).min((range - translate) / s).max(1.0)
        };
        let (gx, gy) = (band(sx, tx - ox as f32), band(sy, ty - oy as f32));
        // points and lines: discarded only once half a pixel wide past the clip region
        let (dx, dy) = match d.primitive {
            DrawPrimitive::Triangles | DrawPrimitive::TriangleStrip => (1.0, 1.0),
            _ => ((1.0 + 0.5 / sx.abs().max(0.5)).min(gx), (1.0 + 0.5 / sy.abs().max(0.5)).min(gy)),
        };
        out.extend([
            (CTX, reg::HARDWARE_SCREEN_OFFSET, (ox as u32 >> 4) | (oy as u32 >> 4) << 16),
            // PIX_CENTER, ROUND_TO_EVEN, QUANT_MODE
            (CTX, reg::VTX_CNTL, 1 | 2 << 1 | (5 + quant as u32) << 3),
            (CTX, reg::VTX_CNTL + 1, f(gy)),
            (CTX, reg::VTX_CNTL + 2, f(dy)),
            (CTX, reg::VTX_CNTL + 3, f(gx)),
            (CTX, reg::VTX_CNTL + 4, f(dx)),
        ]);
    }

    /// The user data of the stage whose first is `first` (`USER_DATA_GS_0`, `USER_DATA_PS_0`)
    pub fn user_data(out: &mut Vec<Rdna2Write>, vertex: bool, data: &[u32]) {
        let first = if vertex { reg::USER_DATA_GS_0 } else { reg::USER_DATA_PS_0 };
        out.extend(data.iter().enumerate().map(|(k, v)| (SH, first + k as u32, *v)));
    }

    /// A vertex input's buffer descriptor: its first byte at `addr`, `stride` apart, `bytes` of them from `addr` on
    /// (fetched by index with `idxen`; one past the last whole vertex reads zero)
    pub fn vertex(addr: u64, stride: u32, bytes: u64, format: VertexFormat) -> [u32; 4] {
        // GFX10_FORMAT and DST_SEL (X 4, Y 5, Z 6, W 7, zero 0, one 1)
        let (fmt, size, sel) = match format {
            VertexFormat::Float32 => (22, 4, 4 | 1 << 9),
            VertexFormat::Float32x2 => (64, 8, 4 | 5 << 3 | 1 << 9),
            VertexFormat::Float32x3 => (74, 12, 4 | 5 << 3 | 6 << 6 | 1 << 9),
            VertexFormat::Float32x4 => (77, 16, 4 | 5 << 3 | 6 << 6 | 7 << 9),
            VertexFormat::Unorm8x4 => (56, 4, 4 | 5 << 3 | 6 << 6 | 7 << 9),
        };
        // OOB_SELECT STRUCTURED (the index checked) by records, RAW by bytes when no stride
        let (records, oob) = match stride {
            0 => (bytes.min(u64::from(u32::MAX)) as u32, 3),
            _ if bytes < size => (0, 1),
            s => (((bytes - size) / u64::from(s) + 1).min(u64::from(u32::MAX)) as u32, 1),
        };
        [addr as u32, (addr >> 32) as u32 & 0xffff | (stride & 0x3fff) << 16, records, sel | fmt << 12 | 1 << 24 | oob << 28]
    }

    /// A 2D linear image's descriptor (T#), its pitch in DEPTH (gfx10.3's custom pitch)
    pub fn image(t: &Rdna2Surface) -> [u32; 8] {
        // GFX10_FORMAT, DST_SEL
        let (fmt, sel) = match t.format {
            ColorFormat::Rgba8Unorm => (56, 4 | 5 << 3 | 6 << 6 | 7 << 9),
            ColorFormat::R8Unorm => (1, 4 | 1 << 9),
            ColorFormat::Rgba32Float => (77, 4 | 5 << 3 | 6 << 6 | 7 << 9),
        };
        let (w, h, pitch) = (t.width - 1, t.height - 1, t.pitch - 1);
        [
            (t.addr >> 8) as u32,
            (t.addr >> 40) as u32 & 0xff | fmt << 20 | (w & 3) << 30,
            w >> 2 | h << 14 | 1 << 31,
            // TYPE 2D, SW_MODE linear
            sel | 9 << 28,
            pitch & 0x1fff | (pitch >> 13 & 1) << 13,
            // PERF_MOD as radeonsi's
            4 << 20,
            0,
            0,
        ]
    }

    /// A sampler's descriptor (S#): one mip level, filtered and wrapped so; coordinates truncated as radeonsi's for
    /// nearest
    pub fn sampler(filter: Filter, wrap: Wrap) -> [u32; 4] {
        // SQ_TEX_WRAP, SQ_TEX_MIRROR, SQ_TEX_CLAMP_LAST_TEXEL
        let clamp = match wrap {
            Wrap::Repeat => 0,
            Wrap::MirroredRepeat => 1,
            Wrap::ClampToEdge => 2,
        };
        let (xy, trunc) = match filter {
            Filter::Nearest => (0, 1),
            Filter::Linear => (1, 0),
        };
        // DISABLE_CUBE_WRAP; MIP_FILTER point, ANISO_OVERRIDE
        [clamp | clamp << 3 | clamp << 6 | trunc << 27 | 1 << 28, 0, xy << 20 | xy << 22 | 1 << 26 | 1 << 29, 0]
    }

    /// The draw packets: `instances` of `count` vertices from `first` (the first vertex is the shader's), or of
    /// `count` indices from index buffer `indices` (its address, its indices' format and number) from index `first`
    pub fn packets(out: &mut Vec<u32>, count: u32, first: u32, instances: u32, indices: Option<(u64, IndexFormat, u32)>) {
        out.extend([Rdna2Packet::header(Rdna2Packet::EVENT_WRITE, 1), Rdna2Packet::SQ_NON_EVENT, Rdna2Packet::header(Rdna2Packet::NUM_INSTANCES, 1), instances]);
        match indices {
            None => out.extend([Rdna2Packet::header(Rdna2Packet::DRAW_INDEX_AUTO, 2), count, 2]),
            Some((addr, format, n)) => {
                let size = match format {
                    IndexFormat::U16 => 2,
                    IndexFormat::U32 => 4,
                };
                let at = addr + u64::from(first) * size;
                out.extend([
                    Rdna2Packet::header(Rdna2Packet::DRAW_INDEX_2, 5),
                    n - first,
                    at as u32,
                    (at >> 32) as u32,
                    count,
                    // DI_SRC_SEL_DMA
                    0,
                ]);
            }
        }
    }

    /// VGT_INDEX_TYPE set (SET_UCONFIG_REG_INDEX, index 2, as radeonsi's on gfx10)
    pub fn index_type(out: &mut Vec<u32>, format: IndexFormat) {
        let t = match format {
            IndexFormat::U16 => 0,
            IndexFormat::U32 => 1,
        };
        out.extend([Rdna2Packet::header(Rdna2Packet::UC_INDEX, 2), reg::VGT_INDEX_TYPE | 2 << 28, t]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Blend, ColorTarget};

    fn regs(f: impl Fn(&mut Vec<Rdna2Write>)) -> Vec<Rdna2Write> {
        let mut out = Vec::new();
        f(&mut out);
        out
    }

    fn at(regs: &[Rdna2Write], off: u32) -> u32 {
        regs.iter().find(|r| r.1 == off).map(|r| r.2).expect("a register set")
    }

    fn draw<'d>() -> Draw<'d> {
        Draw { viewport: [0.0, 0.0, 64.0, 64.0], scissor: Some([4, 4, 56, 56]), target: ColorTarget { width: 64, height: 64, ..Default::default() }, ..Default::default() }
    }

    #[test]
    fn radeonsi_viewport_and_scissor() {
        let d = draw();
        let px = Rdna2Draw::pixels(&d).unwrap();
        assert_eq!(px, [4, 4, 60, 60]);
        let r = regs(|o| Rdna2Draw::viewport(o, &d, px));
        assert_eq!((at(&r, 0x094), at(&r, 0x095)), (0x8004_0004, 0x003c_003c));
        assert_eq!((at(&r, 0x08d), at(&r, 0x2f9)), (0x0002_0002, 0x3d));
        assert_eq!([at(&r, 0x2fa), at(&r, 0x2fb), at(&r, 0x2fc), at(&r, 0x2fd)], [0x4280_0000, 0x3f80_0000, 0x4280_0000, 0x3f80_0000]);
        assert_eq!([at(&r, 0x10f), at(&r, 0x110), at(&r, 0x111), at(&r, 0x112)], [0x4200_0000; 4]);
    }

    #[test]
    fn radeonsi_blend() {
        let mut d = draw();
        d.blend = Some(Blend { src: [BlendFactor::One; 2], dst: [BlendFactor::OneMinusSrcAlpha; 2], ..Default::default() });
        let r = regs(|o| Rdna2Draw::blend(o, &d));
        // radeonsi's 0x40000501 (alpha as RGB), ours with alpha apart
        assert_eq!(at(&r, 0x1e0), 0x4000_0501 | 1 << 29 | 0x0501 << 16);
        assert_eq!(at(&r, 0x1d8), 0x0151_0151);
    }

    #[test]
    fn radeonsi_descriptors() {
        let t = Rdna2Surface { addr: 0x8001_00e0_0000, width: 4, height: 4, pitch: 64, format: ColorFormat::Rgba8Unorm };
        let d = Rdna2Draw::image(&t);
        assert_eq!(d[..4], [0x0100_e000, 0xc380_0080, 0x8000_c000, 0x9000_0fac]);
        assert_eq!(Rdna2Draw::sampler(Filter::Nearest, Wrap::ClampToEdge), [0x1800_0092, 0, 0x2400_0000, 0]);
        assert_eq!(Rdna2Draw::vertex(0x8001_0080_0000, 12, 72, VertexFormat::Float32x2), [0x0080_0000, 0x000c_8001, 6, 0x1104_022c]);
        assert_eq!(Rdna2Draw::vertex(0x8001_0080_0008, 12, 64, VertexFormat::Unorm8x4), [0x0080_0008, 0x000c_8001, 6, 0x1103_8fac]);
    }

    #[test]
    fn shadow_sets_what_changed() {
        let mut s = Rdna2Shadow::default();
        let mut out = Vec::new();
        s.set(&mut out, &[(CTX, 0x10, 1), (CTX, 0x11, 2), (SH, 0x8, 3)]);
        assert_eq!(out, [Rdna2Packet::header(CTX, 3), 0x10, 1, 2, Rdna2Packet::header(SH, 2), 0x8, 3]);
        out.clear();
        s.set(&mut out, &[(CTX, 0x10, 1), (CTX, 0x11, 5), (SH, 0x8, 3)]);
        assert_eq!(out, [Rdna2Packet::header(CTX, 2), 0x11, 5]);
    }

    #[test]
    fn a_scissor_is_set_whole() {
        let (tl, br) = (reg::VPORT_SCISSOR_0_TL, reg::VPORT_SCISSOR_0_TL + 1);
        let mut s = Rdna2Shadow::default();
        let mut out = Vec::new();
        s.set(&mut out, &[(CTX, tl, 1 << 31), (CTX, br, 16 | 32 << 16)]);
        out.clear();
        // only the bottom-right changes: the top-left goes with it
        s.set(&mut out, &[(CTX, tl, 1 << 31), (CTX, br, 256 | 256 << 16)]);
        assert_eq!(out, [Rdna2Packet::header(CTX, 3), tl, 1 << 31, 256 | 256 << 16]);
    }
}
