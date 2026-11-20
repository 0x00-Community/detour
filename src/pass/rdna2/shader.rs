//! detour's graphics ABI on RDNA2 (gfx10.3, wave32, NGG passthrough): what a compiled vertex or fragment shader
//! expects of the draw that runs it, and what it tells the draw.
//!
//! Vertex shader, run as the hardware GS stage (merged ES+GS, `SPI_SHADER_PGM_*_ES`), NGG passthrough
//! (`GE_CNTL` and `VGT_SHADER_STAGES_EN` as radeonsi sets them for a plain VS). Its user SGPRs start at s8
//! (`SPI_SHADER_USER_DATA_GS_0`), each descriptor on its alignment (a V# on 4 SGPRs, an address on 2):
//!   - each vertex input, in location order: a buffer descriptor (V#, 4 SGPRs) whose base is the input's first
//!     byte in the first vertex, its stride the buffer's, its format the input's (fetched with `idxen`)
//!   - then each uniform block of `ubos`, in order: its address, low then high dword (2 SGPRs)
//!   - then the first vertex, added to the vertex index (v5)
//!
//! It begins with ACO's NGG passthrough prologue (s3: merged wave info; v0: the primitive's export data), exports
//! `pos0`, then `param0..varyings`.
//!
//! Fragment shader: its user SGPRs start at s0 (`SPI_SHADER_USER_DATA_PS_0`):
//!   - each sampled texture of `textures`, in order: its image descriptor (T#, 8 SGPRs) then its sampler (S#, 4
//!     SGPRs)
//!   - then each uniform block of `ubos`, in order: its address, low then high dword (2 SGPRs)
//!
//! The primitive mask (for `m0`) is the SGPR after them. Its input VGPRs are those `ps_input` enables
//! (`SPI_PS_INPUT_ENA` and `SPI_PS_INPUT_ADDR` both); it reads `inputs` varyings (`SPI_PS_INPUT_CNTL_0..`, each at
//! its own offset) and exports `mrt0` as four fp16 values (`SPI_SHADER_COL_FORMAT` FP16_ABGR, compressed).

use alloc::format;
use alloc::vec::Vec;

use mlirformat::Error;

use crate::ShaderStage;

/// SPI_PS_INPUT_ENA bits
pub const RDNA2_PERSP_CENTER: u32 = 1 << 1;

/// The first user SGPR of a vertex shader (merged: the system SGPRs come first)
pub const RDNA2_VS_USER_SGPR: u32 = 8;
/// The SGPRs a uniform block's address takes, a texture's descriptors, a vertex input's
pub const RDNA2_UBO_SGPRS: u32 = 2;
pub const RDNA2_TEXTURE_SGPRS: u32 = 12;
pub const RDNA2_INPUT_SGPRS: u32 = 4;
/// The user SGPRs a stage has
pub const RDNA2_USER_SGPRS: u32 = 32;

/// A compiled shader as a draw takes it: its machine code (padded with `s_code_end` as the instruction prefetch
/// runs past it) and what the draw sets up for it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rdna2Shader {
    pub stage: ShaderStage,
    pub code: Vec<u8>,
    /// the SGPRs and VGPRs it uses (for `SPI_SHADER_PGM_RSRC1_*`)
    pub sgprs: u32,
    pub vgprs: u32,
    /// vertex: its vertex inputs; fragment: the varyings it reads
    pub inputs: u32,
    /// vertex: the varyings it exports (after the position)
    pub varyings: u32,
    /// the uniform blocks it reads, by binding, in user SGPR order
    pub ubos: Vec<u32>,
    /// the textures it samples, by binding, in user SGPR order
    pub textures: Vec<u32>,
    /// fragment: `SPI_PS_INPUT_ENA` (and `_ADDR`)
    pub ps_input: u32,
}

impl Rdna2Shader {
    /// The user SGPRs it takes (`SPI_SHADER_PGM_RSRC2_*.USER_SGPR`)
    pub fn user_sgprs(&self) -> u32 {
        let ubos = RDNA2_UBO_SGPRS * self.ubos.len() as u32;
        match self.stage {
            ShaderStage::Vertex => 1 + ubos + RDNA2_INPUT_SGPRS * self.inputs,
            _ => ubos + RDNA2_TEXTURE_SGPRS * self.textures.len() as u32,
        }
    }

    /// The user SGPR (from the stage's first) the first uniform block's address is in
    fn ubos_sgpr(&self) -> u32 {
        match self.stage {
            ShaderStage::Vertex => RDNA2_INPUT_SGPRS * self.inputs,
            _ => RDNA2_TEXTURE_SGPRS * self.textures.len() as u32,
        }
    }

    /// The user SGPR (from the stage's first) a uniform block's address is in
    pub fn ubo_sgpr(&self, binding: u32) -> Option<u32> {
        let k = self.ubos.iter().position(|b| *b == binding)? as u32;
        Some(self.ubos_sgpr() + RDNA2_UBO_SGPRS * k)
    }

    /// The user SGPR a vertex input's V# is in
    pub fn input_sgpr(&self, location: u32) -> u32 {
        RDNA2_INPUT_SGPRS * location
    }

    /// The user SGPR of a vertex shader the first vertex is in
    pub fn first_vertex_sgpr(&self) -> u32 {
        self.ubos_sgpr() + RDNA2_UBO_SGPRS * self.ubos.len() as u32
    }

    /// The user SGPR a texture's T# is in (its S# 8 after)
    pub fn texture_sgpr(&self, binding: u32) -> Option<u32> {
        let k = self.textures.iter().position(|b| *b == binding)? as u32;
        Some(RDNA2_TEXTURE_SGPRS * k)
    }

    /// Its bytes in a program: the fields as little-endian u32s (each list its length, then its items), the code's
    /// length, the code
    pub fn write(&self, out: &mut Vec<u8>) {
        let stage = match self.stage {
            ShaderStage::Vertex => 0,
            ShaderStage::Fragment => 1,
            _ => 2,
        };
        let mut words = alloc::vec![stage, self.sgprs, self.vgprs, self.inputs, self.varyings, self.ps_input];
        for list in [&self.ubos, &self.textures] {
            words.push(list.len() as u32);
            words.extend(list);
        }
        words.push(self.code.len() as u32);
        for w in words {
            out.extend(w.to_le_bytes());
        }
        out.extend(&self.code);
    }

    /// One read from `bytes` (`write`'s), which then start after it
    pub fn read(bytes: &mut &[u8]) -> Result<Rdna2Shader, Error> {
        let mut word = || -> Result<u32, Error> {
            let (head, rest) = bytes.split_first_chunk::<4>().ok_or_else(|| Error("a program cut short".into()))?;
            *bytes = rest;
            Ok(u32::from_le_bytes(*head))
        };
        let [stage, sgprs, vgprs, inputs, varyings, ps_input] = [word()?, word()?, word()?, word()?, word()?, word()?];
        let mut list = || -> Result<Vec<u32>, Error> {
            let n = word()?;
            if n > RDNA2_USER_SGPRS {
                return Err(Error(format!("a shader of {n} bindings")));
            }
            (0..n).map(|_| word()).collect()
        };
        let (ubos, textures) = (list()?, list()?);
        let len = word()? as usize;
        let (code, rest) = bytes.split_at_checked(len).ok_or_else(|| Error("a program cut short".into()))?;
        *bytes = rest;
        let stage = match stage {
            0 => ShaderStage::Vertex,
            1 => ShaderStage::Fragment,
            s => return Err(Error(format!("a shader of stage {s}"))),
        };
        Ok(Rdna2Shader { stage, code: code.to_vec(), sgprs, vgprs, inputs, varyings, ubos, textures, ps_input })
    }
}
