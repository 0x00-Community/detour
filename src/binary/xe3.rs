//! Intel Xe3 machine code and batches: each EU instruction of the `xe3` dialect encoded (its fields
//! read back as an `Inst`, packed as gentool packs them), a kernel's code, and an image's batch,
//! state and kernels (`Xe3Image`).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::ShaderStage;
use crate::{Arf, Cond, EuFormat as Format, EuOpcode as Opcode, EuOperand as Operand, File, Inst, Sfid, XE3, Xe3Kernel};
use mlirformat::{Attribute, Error, Op, Operation};

use crate::{ImageBytes, Pack, Pool, Write};
use mlirformat::{Ir, OpId};

/// The EU encoder
pub struct Xe3Isa;

impl Xe3Isa {
    /// Bytes of an encoded instruction (the uncompacted form)
    pub const BYTES: usize = 16;

    /// `i` as gentool encodes it
    pub fn encode(i: &Inst) -> [u8; 16] {
        let mut e = Enc([0; 2]);
        e.inst(i);
        (u128::from(e.0[1]) << 64 | u128::from(e.0[0])).to_le_bytes()
    }

    /// Instruction op `name` with attributes `attrs`, encoded
    pub fn op(name: &str, attrs: &[mlirformat::NamedAttr<'_>]) -> Result<[u8; 16], Error> {
        Inst::try_from((name, attrs)).map(|i| Xe3Isa::encode(&i))
    }
}

#[allow(dead_code)]
mod f {
    pub type F = (u32, u32);
    pub const HW_OPCODE: F = (6, 0);
    pub const SWSB: F = (17, 8);
    pub const EXEC_SIZE: F = (20, 18);
    pub const FLAG_SUBNR: F = (21, 21);
    pub const FLAG_NR: F = (23, 22);
    pub const CHAN_OFFSET: F = (25, 24);
    pub const PRED_CONTROL: F = (27, 26);
    pub const PRED_INV: F = (28, 28);
    pub const DEBUG_CONTROL: F = (30, 30);
    pub const NO_MASK: F = (31, 31);
    pub const ATOMIC_CONTROL: F = (32, 32);
    pub const BRANCH_CONTROL: F = (33, 33);
    pub const DST_OPERAND_EXTRA: F = (33, 33);
    pub const SATURATE: F = (34, 34);
    pub const DST_ADDRESS_MODE: F = (35, 35);
    pub const DST_TYPE: F = (39, 36);
    pub const SRC0_TYPE: F = (43, 40);
    pub const SRC0_ABS: F = (44, 44);
    pub const SRC0_NEGATE: F = (45, 45);
    pub const SRC0_IS_IMM: F = (46, 46);
    pub const SRC1_IS_IMM: F = (47, 47);
    pub const DST_HSTRIDE: F = (49, 48);
    pub const DST_OPERAND: F = (63, 50);
    pub const SRC0_HSTRIDE: F = (65, 64);
    pub const SRC0_OPERAND: F = (79, 66);
    pub const SRC0_ADDRESS_MODE: F = (80, 80);
    pub const SRC0_WIDTH: F = (83, 81);
    pub const SRC0_VSTRIDE: F = (86, 84);
    pub const SRC0_OPERAND_EXTRA: F = (87, 87);
    pub const SRC1_TYPE: F = (91, 88);
    pub const COND_MODIFIER: F = (95, 92);
    pub const SRC1_HSTRIDE: F = (97, 96);
    pub const SRC1_OPERAND: F = (111, 98);
    pub const SRC1_ADDRESS_MODE: F = (112, 112);
    pub const SRC1_WIDTH: F = (115, 113);
    pub const SRC1_VSTRIDE: F = (118, 116);
    pub const SRC1_ABS: F = (120, 120);
    pub const SRC1_NEGATE: F = (121, 121);
    pub const IMM_LO_32: F = (127, 96);
    pub const IMM_HI_32: F = (95, 64);

    pub const THREE_DST_TYPE: F = (38, 36);
    pub const THREE_EXEC_DATA_TYPE: F = (39, 39);
    pub const THREE_SRC0_TYPE: F = (42, 40);
    pub const THREE_SRC0_VSTRIDE: [F; 2] = [(43, 43), (35, 35)];
    pub const THREE_SRC2_IS_IMM: F = (47, 47);
    pub const THREE_DST_HSTRIDE: F = (48, 48);
    pub const THREE_DST_OPERAND: F = (63, 50);
    pub const THREE_SRC0_HSTRIDE: F = (65, 64);
    pub const THREE_SRC0_OPERAND: F = (79, 66);
    pub const THREE_SRC2_TYPE: F = (82, 80);
    pub const THREE_SRC2_ABS: F = (84, 84);
    pub const THREE_SRC2_NEGATE: F = (85, 85);
    pub const THREE_SRC1_ABS: F = (86, 86);
    pub const THREE_SRC1_NEGATE: F = (87, 87);
    pub const THREE_SRC1_TYPE: F = (90, 88);
    pub const THREE_SRC1_VSTRIDE: [F; 2] = [(91, 91), (83, 83)];
    pub const THREE_SRC1_HSTRIDE: F = (97, 96);
    pub const THREE_SRC1_OPERAND: F = (111, 98);
    pub const THREE_SRC2_HSTRIDE: F = (113, 112);
    pub const THREE_SRC2_OPERAND: F = (127, 114);
    pub const THREE_SRC0_IMM: F = (79, 64);
    pub const THREE_SRC2_IMM: F = (127, 112);

    pub const MATH_FC: F = (95, 92);
    pub const SYNC_CTRL: F = (95, 92);
    pub const BFN_COND_MODIFIER: F = (45, 44);

    pub const DPAS_RCOUNT: F = (45, 43);
    pub const DPAS_SDEPTH: F = (49, 48);
    pub const DPAS_SRC2_SUBBYTE: F = (85, 84);
    pub const DPAS_SRC1_SUBBYTE: F = (87, 86);
    pub const BFN_FUNC_CONTROL: [F; 2] = [(95, 92), (87, 84)];

    pub const SEND_EOT: F = (34, 34);
    pub const SEND_EX_BSO: F = (39, 39);
    pub const SEND_DESC_IS_REG: F = (48, 48);
    pub const SEND_EX_DESC_IS_REG: F = (49, 49);
    pub const SEND_SFID: F = (95, 92);
    pub const SEND_SRC1_LEN: F = (103, 99);
}

/// An instruction's 128 bits being filled
struct Enc([u64; 2]);

impl Enc {
    fn cvt(v: u8) -> u64 {
        if v == 0 { 0 } else { v.trailing_zeros() as u64 + 1 }
    }

    fn vstride3(v: u8) -> u64 {
        match v {
            0 => 0,
            1 | 2 => 1,
            4 => 2,
            _ => 3,
        }
    }

    #[inline]
    fn set(&mut self, (hi, lo): f::F, v: u64) {
        debug_assert!(hi / 64 == lo / 64, "field crosses a qword");
        let w = hi - lo + 1;
        debug_assert!(w == 64 || v >> w == 0, "value {v:#x} too wide for bits {hi}:{lo}");
        let mask = if w == 64 { u64::MAX } else { ((1u64 << w) - 1) << (lo % 64) };
        let q = &mut self.0[(hi / 64) as usize];
        *q = (*q & !mask) | (v << (lo % 64));
    }

    fn set2(&mut self, parts: [f::F; 2], v: u64) {
        let lo_bits = parts[1].0 - parts[1].1 + 1;
        self.set(parts[1], v & ((1 << lo_bits) - 1));
        self.set(parts[0], v >> lo_bits);
    }

    fn sub(field: f::F, hi: u32, lo: u32) -> f::F {
        (field.1 + hi, field.1 + lo)
    }

    fn direct(&mut self, field: f::F, o: &Operand, skip_subnr: bool) {
        self.set(Self::sub(field, 0, 0), (o.file == File::Grf) as u64);
        if !skip_subnr {
            self.set(Self::sub(field, 5, 1), (o.subnr >> 1) as u64);
        }
        self.set(Self::sub(field, 13, 6), o.nr as u64);
    }

    fn vstride(v: u8) -> u64 {
        (if v == 0xff { 0xf } else { Self::cvt(v) }) & 0x7
    }

    fn inst(&mut self, i: &Inst) {
        let format = i.op.format();
        self.set(f::HW_OPCODE, i.op.hw());
        self.set(f::SWSB, i.swsb.encode(i.op));
        self.set(f::EXEC_SIZE, if i.exec_size == 0 { 0 } else { Self::cvt(i.exec_size) - 1 });
        self.set(f::FLAG_SUBNR, i.flag.1 as u64);
        self.set(f::FLAG_NR, i.flag.0 as u64);
        self.set(f::PRED_CONTROL, i.pred as u64);
        self.set(f::PRED_INV, i.pred_inv as u64);
        self.set(f::NO_MASK, i.no_mask as u64);
        self.set(f::ATOMIC_CONTROL, i.atomic as u64);
        self.set(f::CHAN_OFFSET, (i.chan_offset / 8) as u64);

        if matches!(format, Format::One | Format::Two | Format::Dpas) || (format == Format::Three && i.op != Opcode::Bfn) {
            self.set(f::SATURATE, i.saturate as u64);
        }
        if i.op == Opcode::Math {
            self.set(f::MATH_FC, i.math.expect("math without a function") as u64);
        } else if i.op == Opcode::Sync {
            self.set(f::SYNC_CTRL, i.sync as u64);
        } else if i.op == Opcode::Bfn {
            let c = match i.cond {
                Cond::Z => 1,
                Cond::Gt => 2,
                Cond::Lt => 3,
                _ => 0,
            };
            self.set(f::BFN_COND_MODIFIER, c);
        } else {
            let has_cond = match format {
                Format::One => !(i.src[0].file == File::Imm && i.src[0].ty.size() >= 8),
                Format::Two | Format::Three | Format::Dpas => true,
                _ => false,
            };
            if has_cond {
                self.set(f::COND_MODIFIER, i.cond as u64);
            }
        }

        match format {
            Format::One | Format::Two => {
                if i.op.has_dst() {
                    let d = &i.dst;
                    self.set(f::DST_TYPE, d.ty.code());
                    self.direct(f::DST_OPERAND, d, false);
                    self.set(f::DST_OPERAND_EXTRA, (d.subnr & 1) as u64);
                }
                self.set(f::DST_HSTRIDE, Self::cvt(i.dst.region.hstride.max(1)));
                self.set(f::SRC0_TYPE, i.src[0].ty.code());
                let mut imm = None;
                let s0 = &i.src[0];
                if s0.file == File::Imm {
                    imm = Some(s0);
                    self.set(f::SRC0_IS_IMM, 1);
                } else {
                    self.set(f::SRC0_NEGATE, s0.negate as u64);
                    self.set(f::SRC0_ABS, s0.abs as u64);
                    self.direct(f::SRC0_OPERAND, s0, false);
                    self.set(f::SRC0_OPERAND_EXTRA, (s0.subnr & 1) as u64);
                    self.set(f::SRC0_VSTRIDE, Self::vstride(s0.region.vstride));
                    self.set(f::SRC0_WIDTH, Self::cvt(s0.region.width) - 1);
                    self.set(f::SRC0_HSTRIDE, Self::cvt(s0.region.hstride));
                }
                if format == Format::Two {
                    let s1 = &i.src[1];
                    self.set(f::SRC1_TYPE, s1.ty.code());
                    if s1.file == File::Imm {
                        assert!(imm.is_none(), "two immediates");
                        imm = Some(s1);
                        self.set(f::SRC1_IS_IMM, 1);
                    } else {
                        self.set(f::SRC1_NEGATE, s1.negate as u64);
                        self.set(f::SRC1_ABS, s1.abs as u64);
                        self.direct(f::SRC1_OPERAND, s1, false);
                        self.set(f::SRC1_VSTRIDE, Self::vstride(s1.region.vstride));
                        self.set(f::SRC1_WIDTH, Self::cvt(s1.region.width) - 1);
                        self.set(f::SRC1_HSTRIDE, Self::cvt(s1.region.hstride));
                    }
                }
                if let Some(o) = imm {
                    self.set(f::IMM_LO_32, o.imm & 0xffff_ffff);
                    if o.ty.size() > 4 {
                        self.set(f::IMM_HI_32, o.imm >> 32);
                    }
                }
            }
            Format::Three => {
                let [s0, s1, s2] = &i.src;
                self.set(f::THREE_EXEC_DATA_TYPE, i.dst.ty.is_float() as u64);
                self.set(f::THREE_DST_TYPE, i.dst.ty.code3());
                self.set(f::THREE_SRC0_TYPE, s0.ty.code3());
                self.set(f::THREE_SRC1_TYPE, s1.ty.code3());
                self.set(f::THREE_SRC2_TYPE, s2.ty.code3());
                self.direct(f::THREE_DST_OPERAND, &i.dst, false);
                self.set(f::THREE_DST_HSTRIDE, (i.dst.region.hstride.max(1) != 1) as u64);
                if s0.file == File::Imm {
                    self.set(f::SRC0_IS_IMM, 1);
                    self.set(f::THREE_SRC0_IMM, s0.imm & 0xffff);
                } else {
                    self.direct(f::THREE_SRC0_OPERAND, s0, false);
                    self.set2(f::THREE_SRC0_VSTRIDE, Self::vstride3(s0.region.vstride));
                    self.set(f::THREE_SRC0_HSTRIDE, Self::cvt(s0.region.hstride));
                }
                self.direct(f::THREE_SRC1_OPERAND, s1, false);
                self.set2(f::THREE_SRC1_VSTRIDE, Self::vstride3(s1.region.vstride));
                self.set(f::THREE_SRC1_HSTRIDE, Self::cvt(s1.region.hstride));
                if s2.file == File::Imm {
                    self.set(f::THREE_SRC2_IS_IMM, 1);
                    self.set(f::THREE_SRC2_IMM, s2.imm & 0xffff);
                } else {
                    self.direct(f::THREE_SRC2_OPERAND, s2, false);
                    self.set(f::THREE_SRC2_HSTRIDE, Self::cvt(s2.region.step()));
                }
                if i.op != Opcode::Bfn {
                    self.set(f::SRC0_NEGATE, s0.negate as u64);
                    self.set(f::THREE_SRC1_NEGATE, s1.negate as u64);
                    self.set(f::THREE_SRC2_NEGATE, s2.negate as u64);
                    self.set(f::SRC0_ABS, s0.abs as u64);
                    self.set(f::THREE_SRC1_ABS, s1.abs as u64);
                    self.set(f::THREE_SRC2_ABS, s2.abs as u64);
                } else {
                    self.set2(f::BFN_FUNC_CONTROL, i.bfn_ctrl as u64);
                }
            }
            Format::Dpas => {
                let [s0, s1, s2] = &i.src;
                self.set(f::DPAS_RCOUNT, (i.dpas.rcount - 1) as u64);
                self.set(f::DPAS_SDEPTH, i.dpas.sdepth_code());
                self.set(f::THREE_EXEC_DATA_TYPE, i.dst.ty.is_float() as u64);
                self.set(f::THREE_DST_TYPE, i.dst.ty.code3());
                self.set(f::THREE_SRC0_TYPE, s0.ty.code3());
                self.set(f::THREE_SRC1_TYPE, s1.ty.code3());
                self.set(f::THREE_SRC2_TYPE, s2.ty.code3());
                self.set(f::DPAS_SRC1_SUBBYTE, i.dpas.src1_subbyte as u64);
                self.set(f::DPAS_SRC2_SUBBYTE, i.dpas.src2_subbyte as u64);
                self.direct(f::THREE_DST_OPERAND, &i.dst, false);
                self.direct(f::THREE_SRC0_OPERAND, s0, false);
                self.direct(f::THREE_SRC1_OPERAND, s1, false);
                self.direct(f::THREE_SRC2_OPERAND, s2, false);
            }
            Format::Send => {
                let s = &i.send;
                self.set(f::SEND_EOT, s.eot as u64);
                self.set(f::SEND_SFID, s.sfid as u64);
                self.direct(f::DST_OPERAND, &i.dst, true);
                self.direct(f::SRC0_OPERAND, &i.src[0], true);
                self.direct(f::SRC1_OPERAND, &i.src[1], true);
                if s.ex_bso && s.sfid != Sfid::Ugm as u8 {
                    self.set(f::SEND_EX_BSO, 1);
                }
                self.set(f::SEND_SRC1_LEN, s.src1_len as u64);
                let d = s.desc as u64;
                self.set((123, 122), (d >> 30) & 0x3);
                self.set((71, 67), (d >> 25) & 0x1f);
                self.set((55, 51), (d >> 20) & 0x1f);
                self.set((121, 113), (d >> 11) & 0x1ff);
                self.set((91, 81), d & 0x7ff);
                let x = s.ex_desc as u64;
                debug_assert!(x & 0x7ff == 0, "ex_desc bits 10:0 must be zero (src1_len has its own field)");
                self.set((127, 124), (x >> 28) & 0xf);
                self.set((97, 96), (x >> 26) & 0x3);
                self.set((65, 64), (x >> 24) & 0x3);
                self.set((47, 35), (x >> 11) & 0x1fff);
            }
            Format::Branch1 | Format::Branch2 => {
                if i.op != Opcode::Join {
                    self.set(f::BRANCH_CONTROL, i.branch_ctrl as u64);
                }
                let s0 = &i.src[0];
                let reg = s0.file == File::Grf || (s0.file == File::Arf && s0.nr != Arf::NULL);
                if reg {
                    self.set(f::SRC0_TYPE, s0.ty.code());
                    self.direct(f::SRC0_OPERAND, s0, false);
                } else {
                    self.set(f::SRC0_IS_IMM, 1);
                    self.set(f::IMM_LO_32, i.jip as u32 as u64);
                    if format == Format::Branch2 {
                        self.set(f::SRC1_IS_IMM, 1);
                        self.set(f::IMM_HI_32, i.uip as u32 as u64);
                    }
                }
            }
            Format::Nop => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xe3Code {
    pub name: String,
    pub stage: Option<ShaderStage>,
    pub bytes: Vec<u8>,
}

impl Xe3Code {
    /// An `xe3.kernel`'s code
    pub fn of(kernel: &Operation<'_>) -> Result<Xe3Code, Error> {
        let head = Xe3Kernel::try_from(kernel)?;
        let ir = Ir::new(kernel)?;
        let mut bytes = Vec::new();
        for op in ir.block_ops(ir.top_block(ir.root)) {
            bytes.extend(Xe3Isa::op(ir[op].name, &ir[op].all_attrs())?);
        }
        let name = match &head.sym_name {
            Attribute::SymbolRef(p) => p[0].trim_start_matches('@').into(),
            a @ Attribute::Str { .. } => a.text()?,
            other => return Err(Error(format!("{other} is not a symbol"))),
        };
        Ok(Xe3Code { name, stage: head.stage, bytes })
    }

    /// The kernel's code and the padding it asks for after it, as an image holds it
    pub fn padded(kernel: &Operation<'_>) -> Result<Vec<u8>, Error> {
        let mut bytes = Xe3Code::of(kernel)?.bytes;
        let pad = Xe3Kernel::try_from(kernel)?.pad.unwrap_or(0);
        bytes.resize(bytes.len() + pad as usize, 0);
        Ok(bytes)
    }
}

/// The `xe3` machine code of a kernel (`xe3.kernel`), or of every kernel in an op, one after the
/// other; `Xe3Code::of` gives one kernel's code
#[derive(Debug, Clone, Copy, Default)]
pub struct Xe3Writer;

impl Xe3Writer {
    pub const NAME: &'static str = "xe3";

    /// Every `xe3.kernel` in `op` (`op` itself if it is one)
    pub fn kernels(op: &Operation<'_>) -> Result<Vec<Xe3Code>, Error> {
        let mut out = Vec::new();
        Self::collect(op, &mut out)?;
        Ok(out)
    }

    fn collect(op: &Operation<'_>, out: &mut Vec<Xe3Code>) -> Result<(), Error> {
        if op.name == Xe3Kernel::NAME {
            out.push(Xe3Code::of(op)?);
            return Ok(());
        }
        for o in op.regions.iter().flat_map(|r| &r.blocks).flat_map(|b| &b.ops) {
            Self::collect(o, out)?;
        }
        Ok(())
    }
}

impl Write for Xe3Writer {
    fn name(&self) -> &'static str {
        Xe3Writer::NAME
    }

    fn write(&self, op: &Operation<'_>, _pool: &dyn Pool) -> Result<Vec<u8>, Error> {
        Ok(Xe3Writer::kernels(op)?.into_iter().flat_map(|k| k.bytes).collect())
    }
}

/// An Xe3 image: memories from 1 MiB up in 64 KiB pages, a batch of genxml commands, state structs
/// and padded kernels
#[derive(Debug, Clone, Copy, Default)]
pub struct Xe3Image;

impl Xe3Image {
    /// Prefix of the state structs' ops (`xe3.state.blend_state`), which an image places in memory
    pub const STATE: &'static str = "xe3.state.";

    fn le(dw: &[u32]) -> Vec<u8> {
        dw.iter().flat_map(|d| d.to_le_bytes()).collect()
    }
}

impl ImageBytes for Xe3Image {
    const BASE: u64 = 0x10_0000;
    const PAGE: u64 = 0x1_0000;

    fn stream(&self, ir: &Ir<'_, '_>, stream: OpId) -> Result<Vec<u8>, Error> {
        let mut dw = Vec::new();
        for c in ir.block_ops(ir.top_block(stream)) {
            dw.extend(Pack(XE3).pack(ir[c].name, &ir[c].all_attrs())?);
        }
        Ok(Xe3Image::le(&dw))
    }

    fn item(&self, ir: &Ir<'_, '_>, op: OpId) -> Result<Vec<u8>, Error> {
        match ir[op].name {
            n if n == Xe3Kernel::NAME => Xe3Code::padded(ir[op].op),
            name if name.starts_with(Xe3Image::STATE) => Ok(Xe3Image::le(&Pack(XE3).pack(name, &ir[op].all_attrs())?)),
            name => Err(Error(format!("{name} cannot be placed in memory"))),
        }
    }
}
