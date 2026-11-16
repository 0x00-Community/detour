//! Intel Xe3 inside detour: the instruction set and genxml batches (the xe3 dialect), SPIR-V and GPU kernels lowered
//! to Xe3 machine code (convert-gpu-to-xe3 and the xe3-* passes), and a draw's image (Xe3Draw)
mod asm;
#[path = "cmd.gen.rs"]
mod cmd;
mod control;
mod convert;
mod draw;
mod end;
mod eu;
mod layout;
mod msg;
mod pad;
mod regalloc;
mod swsb;
mod target;
mod to_xe3;

pub use control::*;
pub use convert::*;
pub use draw::*;
pub use end::*;
pub use layout::*;
pub use pad::*;
pub use regalloc::*;
pub use swsb::*;
pub use target::*;
pub use to_xe3::*;

/// The xe3 passes, added to mlirpass's with `Passes::with(Xe3Passes::add)`
pub struct Xe3Passes;

impl Xe3Passes {
    pub fn add(p: &mut mlirpass::Passes) {
        p.register::<ConvertGpuToXe3>()
            .register::<Xe3Regalloc>()
            .register::<Xe3Control>()
            .register::<Xe3Branches>()
            .register::<Xe3Swsb>()
            .register::<Xe3Pad>()
            .register::<Xe3End>()
            .register::<Xe3Layout>();
    }
}

pub use cmd::*;
pub use eu::{
    Arf, Cond, Dpas, Elem, File, Format as EuFormat, Inst, MathFn, Opcode as EuOpcode, Operand as EuOperand, Pipe, Pred, Region as EuRegion, SbidMode, SendInfo, Swsb, SyncFn,
};
pub use msg::{AddrSize, AddrSurface, CacheStore, DataSize, Lsc, LscOp, Render, Sampler as Xe3Sampler, Sfid, VectSize};

use alloc::format;
use alloc::vec::Vec;

use mlirformat::{Attribute, Type};

use mlirpass::{Layouts, OpRows, RegType, ShaderStage};
use cmd::GenxmlOps;
#[allow(unused_imports, reason = "the instruction modules name each other's types from here")]
use eu::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grf<'e>(pub &'e str);

impl<'a> From<Grf<'_>> for Type<'a> {
    fn from(g: Grf<'_>) -> Self {
        RegType { name: "xe3.grf", elem: g.0 }.into()
    }
}

impl<'t> TryFrom<&'t Type<'_>> for Grf<'t> {
    type Error = mlirformat::Error;

    fn try_from(t: &'t Type<'_>) -> Result<Grf<'t>, mlirformat::Error> {
        match RegType::try_from(t)? {
            RegType { name: "xe3.grf", elem } => Ok(Grf(elem)),
            _ => Err(mlirformat::Error(format!("{t} is not an xe3 register"))),
        }
    }
}

pub const XE3: Layouts<'_> = Layouts(XE3_ITEMS);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Xe3Dialect;

impl mlirformat::Dialect for Xe3Dialect {
    fn namespace(&self) -> &'static str {
        "xe3"
    }

    fn ops(&self) -> Vec<mlirformat::Template> {
        [GenxmlOps.ops(), EuOps.ops(), PseudoOps.ops()].concat()
    }

    fn op_infos(&self) -> Vec<mlirformat::OpInfo> {
        OpRows::of(&self.ops(), &[XE3_INHERENT, Xe3Dialect::INHERENT].concat())
    }
}

#[mlirformat::mlir::op("xe3.eu.payload", format = "attr-dict `->` type($result)", results(result))]
pub struct EuPayload<'a> {
    #[attr(int = "i64")]
    pub grf: i64,
    pub result: Type<'a>,
}

impl Xe3Dialect {
    pub const INHERENT: &[(&str, &[&str])] = &[
        ("xe3.eu.payload", &["grf"]),
        ("xe3.kernel", &["sym_name", "stage", "simd", "payload", "inputs", "varyings", "targets", "omask", "imask", "bindings", "buffers", "push", "reserve", "threads", "shared", "params", "pad", "align"]),
    ];
}

#[mlirformat::mlir::op("xe3.kernel", format = "$sym_name attr-dict-with-keyword $body")]
pub struct Xe3Kernel<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub stage: Option<ShaderStage>,
    pub simd: Option<u32>,
    pub payload: Option<Vec<i64>>,
    pub inputs: Option<u32>,
    pub varyings: Option<u32>,
    pub targets: Option<u32>,
    pub omask: Option<bool>,
    pub imask: Option<bool>,
    pub bindings: Option<u32>,
    pub buffers: Option<u32>,
    pub push: Option<u32>,
    pub reserve: Option<u32>,
    pub threads: Option<u32>,
    pub shared: Option<u32>,
    pub params: Option<u32>,
    pub pad: Option<u64>,
    pub align: Option<u64>,
    pub body: mlirformat::Region<'a>,
}

#[mlirformat::mlir::dialect("xe3", ops = [EuPayload, Xe3Kernel])]
struct PseudoOps;

mlirpass::attr_ops! { "xe3" EuOps;
    EuIllegal "xe3.eu.illegal", EuBfrev "xe3.eu.bfrev", EuCbit "xe3.eu.cbit", EuFbh "xe3.eu.fbh", EuFbl "xe3.eu.fbl",
    EuFrc "xe3.eu.frc", EuLzd "xe3.eu.lzd", EuMov "xe3.eu.mov", EuNot "xe3.eu.not", EuRndd "xe3.eu.rndd",
    EuRnde "xe3.eu.rnde", EuRndu "xe3.eu.rndu", EuRndz "xe3.eu.rndz", EuSync "xe3.eu.sync", EuWait "xe3.eu.wait",
    EuAdd "xe3.eu.add", EuAddc "xe3.eu.addc", EuAnd "xe3.eu.and", EuAsr "xe3.eu.asr", EuAvg "xe3.eu.avg",
    EuBfi1 "xe3.eu.bfi1", EuCmp "xe3.eu.cmp", EuCmpn "xe3.eu.cmpn", EuMac "xe3.eu.mac", EuMach "xe3.eu.mach",
    EuMacl "xe3.eu.macl", EuMath "xe3.eu.math", EuMovi "xe3.eu.movi", EuMul "xe3.eu.mul", EuOr "xe3.eu.or",
    EuRol "xe3.eu.rol", EuRor "xe3.eu.ror", EuSel "xe3.eu.sel", EuShl "xe3.eu.shl", EuShr "xe3.eu.shr",
    EuSmov "xe3.eu.smov", EuSrnd "xe3.eu.srnd", EuSubb "xe3.eu.subb", EuXor "xe3.eu.xor", EuAdd3 "xe3.eu.add3",
    EuBfe "xe3.eu.bfe", EuBfi2 "xe3.eu.bfi2", EuBfn "xe3.eu.bfn", EuCsel "xe3.eu.csel", EuDp4a "xe3.eu.dp4a",
    EuMad "xe3.eu.mad", EuDpas "xe3.eu.dpas", EuMadm "xe3.eu.madm", EuSend "xe3.eu.send", EuSendc "xe3.eu.sendc",
    EuBrd "xe3.eu.brd", EuCall "xe3.eu.call", EuCalla "xe3.eu.calla", EuEndif "xe3.eu.endif", EuJmpi "xe3.eu.jmpi",
    EuJoin "xe3.eu.join", EuRet "xe3.eu.ret", EuWhile "xe3.eu.while", EuBrc "xe3.eu.brc", EuBreak "xe3.eu.break",
    EuContinue "xe3.eu.continue", EuElse "xe3.eu.else", EuGoto "xe3.eu.goto", EuHalt "xe3.eu.halt", EuIf "xe3.eu.if",
    EuNop "xe3.eu.nop",
}
