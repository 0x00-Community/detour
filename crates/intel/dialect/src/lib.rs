mod asm;
mod eu;
mod msg;

pub use eu::*;
pub use msg::*;

use crate::base::Layouts;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grf<'e>(pub &'e str);

impl<'a> From<Grf<'_>> for mlirformat::Type<'a> {
    fn from(g: Grf<'_>) -> Self {
        crate::base::RegType { name: "xe3.grf", elem: g.0 }.into()
    }
}

impl<'t> TryFrom<&'t mlirformat::Type<'_>> for Grf<'t> {
    type Error = mlirformat::Error;

    fn try_from(t: &'t mlirformat::Type<'_>) -> Result<Grf<'t>, mlirformat::Error> {
        match crate::base::RegType::try_from(t)? {
            crate::base::RegType { name: "xe3.grf", elem } => Ok(Grf(elem)),
            _ => Err(mlirformat::Error(format!("{t} is not an xe3 register"))),
        }
    }
}

pub const XE3: Layouts<'_> = Layouts(ITEMS);

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
        crate::base::OpRows::of(&self.ops(), &[cmd::INHERENT, &[("xe3.eu.payload", &["grf"])]].concat())
    }
}

#[mlirformat::mlir::op("xe3.eu.payload", format = "attr-dict `->` type($result)", results(result))]
pub struct EuPayload<'a> {
    #[attr(int = "i64")]
    pub grf: i64,
    pub result: mlirformat::Type<'a>,
}

#[mlirformat::mlir::dialect("xe3", ops = [EuPayload])]
struct PseudoOps;

mod cmd {
    include!("cmd.gen.rs");
}

pub use cmd::*;

crate::attr_ops! { "xe3" EuOps;
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
