//! What the GPU instruction-set dialect (`xe3`) is built from: register types
//! (`!xe3.grf<u32>`), the stage a kernel runs as, their op tables, and `attr_ops!`, which declares
//! instruction ops whose fields are attributes.

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use mlirformat::{Error, Extended, Type};

/// A register type: the dialect's register type name and what it holds (`!xe3.grf<u32>`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegType<'t> {
    pub name: &'t str,
    pub elem: &'t str,
}

impl<'a> From<RegType<'_>> for Type<'a> {
    fn from(r: RegType<'_>) -> Self {
        Type::Extended(Box::new(Extended { name: r.name.to_owned().into(), body: Some(format!("<{}>", r.elem).into()), params: Vec::new() }))
    }
}

impl<'t> TryFrom<&'t Type<'_>> for RegType<'t> {
    type Error = Error;

    fn try_from(t: &'t Type<'_>) -> Result<RegType<'t>, Error> {
        let not = || Error(format!("{t} is not a register type"));
        let Type::Extended(e) = t else { return Err(not()) };
        let elem = e.inner().ok_or_else(not)?;
        Ok(RegType { name: &e.name, elem })
    }
}

/// The stage a kernel runs as
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// An instruction dialect's op infos: each op and its inherent attributes, no traits, effects unknown
pub struct OpRows;

impl OpRows {
    pub fn of(ops: &[mlirformat::Template], inherent: &[(&'static str, &'static [&'static str])]) -> Vec<mlirformat::OpInfo> {
        let inherent: BTreeMap<&str, &'static [&'static str]> = inherent.iter().rev().copied().collect();
        ops.iter()
            .map(|t| {
                let inherent = inherent.get(t.name).copied().unwrap_or_default();
                mlirformat::OpInfo { name: t.name, traits: &[], effects: mlirformat::Effects::Unknown, inherent }
            })
            .collect()
    }
}

/// Instruction ops whose fields are attributes: `"xe3.eu.add"(%a, %b) {dst = 3} : (…) -> …`
#[macro_export]
macro_rules! attr_ops {
    ($ns:tt $set:ident; $($ty:ident $op:literal,)*) => {
        $(
            #[doc = concat!("`", $op, "`")]
            #[::mlirformat::mlir::op($op, format = "(`(` $operands^ `:` type($operands) `)`)? attr-dict (`->` type($results)^)?", results(results))]
            pub struct $ty<'a> {
                #[slot(types)]
                pub operands: Vec<::mlirformat::Value<'a>>,
                pub types: Vec<::mlirformat::Type<'a>>,
                pub results: Vec<::mlirformat::Type<'a>>,
            }
        )*

        #[::mlirformat::mlir::dialect($ns, ops = [$($ty),*])]
        pub struct $set;
    };
}

/// An instruction's integer field: an integer attribute, hex literals too (`{offset = 0x10}`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsaInt(pub i64);

impl TryFrom<&mlirformat::Attribute<'_>> for IsaInt {
    type Error = Error;

    fn try_from(a: &mlirformat::Attribute<'_>) -> Result<IsaInt, Error> {
        match a {
            mlirformat::Attribute::Int { lit, .. } => match lit.strip_prefix("0x") {
                Some(h) => i64::from_str_radix(h, 16),
                None => lit.parse(),
            }
            .map(IsaInt)
            .map_err(|_| Error(format!("bad integer {lit}"))),
            a => Err(Error(format!("{a} is not an integer"))),
        }
    }
}
