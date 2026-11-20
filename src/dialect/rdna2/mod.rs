//! AMD RDNA2 (gfx1036): the GFX10 instructions, each with its format and opcode (Mesa ACO's table,
//! `scripts/rdna2_opcodes.py`).

#[path = "opcodes.gen.rs"]
mod opcodes;

pub use opcodes::{Format as Rdna2Format, Opcode as Rdna2Opcode};
