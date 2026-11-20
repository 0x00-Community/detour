//! The tables a GPU dialect's packed records are laid out by (`Layouts`: each record's fields, their
//! bits and how a value is encoded), with the values its fields take (`Fill`, `Real`, `Fields`).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use mlirformat::{Attribute, Error, NamedAttr, Type};
use mlirpass::A;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty<'t> {
    Bool,
    Uint,
    Int,
    Float,
    Address,
    Ufixed(u8),
    Sfixed(u8),
    Struct(&'t str),
    Mbo,
    Exact,
    Shr(u8),
    Minus(u32),
    Log2,
    Groups(u32),
    Lod,
    Half,
}

#[derive(Clone, Copy, Debug)]
pub struct Field<'t> {
    pub name: &'t str,
    pub bit: u32,
    pub width: u32,
    pub ty: Ty<'t>,
    pub default: u64,
    pub dims: &'t [(u32, u32)],
}

impl<'t> Field<'t> {
    pub const fn new(name: &'t str, bit: u32, width: u32, ty: Ty<'t>) -> Self {
        Field { name, bit, width, ty, default: 0, dims: &[] }
    }

    pub const fn with_default(self, default: u64) -> Self {
        Field { default, ..self }
    }

    pub const fn with_dims(self, dims: &'t [(u32, u32)]) -> Self {
        Field { dims, ..self }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Group<'t> {
    pub at: u32,
    pub size: u32,
    pub fields: &'t [Field<'t>],
}

#[derive(Clone, Copy, Debug)]
pub struct Item<'t> {
    pub name: &'t str,
    pub op: &'t str,
    pub length: u32,
    pub bias: u32,
    pub num: u32,
    pub fields: &'t [Field<'t>],
    pub group: Option<Group<'t>>,
}

#[derive(Clone, Copy)]
pub struct Layouts<'t>(pub &'t [Item<'t>]);

impl<'t> Layouts<'t> {
    pub fn get(&self, op: &str) -> Option<&'t Item<'t>> {
        self.0.binary_search_by(|i| i.op.cmp(op)).ok().map(|i| &self.0[i])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fill {
    Int(i128),
    Bool(bool),
    Place { path: Vec<String>, add: i64 },
}

impl From<bool> for Fill {
    fn from(v: bool) -> Self {
        Fill::Bool(v)
    }
}

macro_rules! fills {
    ($($t:ty),*) => {$(
        impl From<$t> for Fill {
            fn from(v: $t) -> Self {
                Fill::Int(v.into())
            }
        }
    )*};
}

fills!(i32, u32, i64, u64);

impl From<A> for Fill {
    fn from(a: A) -> Self {
        Fill::try_from(&a).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl TryFrom<&A> for Fill {
    type Error = Error;

    fn try_from(a: &A) -> Result<Fill, Error> {
        match a {
            A::Int(v) => Ok(Fill::Int(*v)),
            A::Bool(b) => Ok(Fill::Bool(*b)),
            A::Sym(s) => Ok(Fill::at(s.clone(), 0)),
            A::In(m, s) => Ok(Fill::Place { path: vec![m.clone(), s.clone()], add: 0 }),
            other => Err(Error(format!("{other:?} is not a field's value"))),
        }
    }
}

impl Fill {
    pub fn at(name: impl Into<String>, add: i64) -> Fill {
        Fill::Place { path: vec![name.into()], add }
    }
}

impl<'a> From<&Fill> for Attribute<'a> {
    fn from(f: &Fill) -> Self {
        match f {
            Fill::Int(v) => Attribute::Int { lit: if *v < 0 { v.to_string() } else { format!("{v:#x}") }.into(), ty: None },
            Fill::Bool(b) => Attribute::Bool(*b),
            Fill::Place { path, add: 0 } => Attribute::SymbolRef(path.iter().map(|s| s.clone().into()).collect()),
            Fill::Place { path, add } => Attribute::Array(vec![Attribute::SymbolRef(path.iter().map(|s| s.clone().into()).collect()), Attribute::Int { lit: add.to_string().into(), ty: None }]),
        }
    }
}

impl TryFrom<&Attribute<'_>> for Fill {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Fill, Error> {
        let path = |s: &[mlirformat::Str<'_>]| s.iter().map(|x| x.to_string()).collect();
        match a {
            Attribute::Int { .. } => Ok(Fill::Int(a.int()?)),
            Attribute::Bool(b) => Ok(Fill::Bool(*b)),
            Attribute::Unit => Ok(Fill::Bool(true)),
            Attribute::SymbolRef(s) => Ok(Fill::Place { path: path(s), add: 0 }),
            Attribute::Array(xs) => match &xs[..] {
                [Attribute::SymbolRef(s), n @ Attribute::Int { .. }] => Ok(Fill::Place { path: path(s), add: n.int()? as i64 }),
                _ => Err(Error(format!("{a}: not a number or [@place, n]"))),
            },
            _ => Err(Error(format!("{a}: not a number or a place"))),
        }
    }
}

pub fn attribute_of<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Attribute<'static>, D::Error> {
    d.deserialize_i128(AttrVisitor)
}

#[derive(Clone, Copy)]
struct AttrVisitor;

impl<'de> serde::de::DeserializeSeed<'de> for AttrVisitor {
    type Value = Attribute<'static>;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        attribute_of(d)
    }
}

impl<'de> serde::de::Visitor<'de> for AttrVisitor {
    type Value = Attribute<'static>;
    fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("an attribute")
    }
    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
        Ok(Attribute::Bool(v))
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
        self.visit_i128(v.into())
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
        self.visit_i128(v.into())
    }
    fn visit_i128<E: serde::de::Error>(self, v: i128) -> Result<Self::Value, E> {
        Ok(Attribute::Int { lit: v.to_string().into(), ty: None })
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Ok(Attribute::Float { lit: format!("{v:?}").into(), ty: None })
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(Attribute::Str { lit: String::from(v).into(), ty: None })
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(Attribute::Unit)
    }
    fn visit_some<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        attribute_of(d)
    }
    fn visit_seq<S: serde::de::SeqAccess<'de>>(self, mut seq: S) -> Result<Self::Value, S::Error> {
        let mut xs = Vec::new();
        while let Some(x) = seq.next_element_seed(AttrVisitor)? {
            xs.push(x);
        }
        Ok(Attribute::Array(xs))
    }
    fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
        let mut d = Vec::new();
        while let Some(k) = map.next_key::<String>()? {
            d.push(NamedAttr { name: k.into(), value: Some(map.next_value_seed(AttrVisitor)?) });
        }
        Ok(Attribute::Dict(d))
    }
    fn visit_enum<A: serde::de::EnumAccess<'de>>(self, e: A) -> Result<Self::Value, A::Error> {
        <Attribute<'static> as serde::Deserialize>::deserialize(serde::de::value::EnumAccessDeserializer::new(e))
    }
}

impl serde::Serialize for Fill {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&Attribute::from(self), s)
    }
}

impl<'de> serde::Deserialize<'de> for Fill {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Fill::try_from(&attribute_of(d)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fields<T>(pub T);

impl<T: serde::Serialize> serde::Serialize for Fields<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl<'de, T: serde::de::DeserializeOwned> serde::Deserialize<'de> for Fields<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use mlirformat::IntoStatic;
        let err = |m: String| <D::Error as serde::de::Error>::custom(m);
        let Attribute::Dict(fields) = attribute_of(d)? else { return Err(err("not a dictionary".into())) };
        let map = fields.into_iter().map(|n| (mlirformat::Content::Str(n.name.to_string()), mlirformat::Content::Attr(n.value.unwrap_or(Attribute::Unit).into_static()))).collect();
        mlirformat::Content::Map(map).to().map(Fields).map_err(|e| err(e.to_string()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Real(pub f64);

impl From<f32> for Real {
    fn from(v: f32) -> Self {
        Real(format!("{v:?}").parse().unwrap_or(v.into()))
    }
}

impl From<f64> for Real {
    fn from(v: f64) -> Self {
        Real(v)
    }
}

impl<'a> From<Real> for Attribute<'a> {
    fn from(r: Real) -> Self {
        let s = format!("{:?}", r.0);
        let lit = match s.find(['e', 'E']) {
            Some(at) if !s[..at].contains('.') => format!("{}.0{}", &s[..at], &s[at..]),
            _ => s,
        };
        Attribute::Float { lit: lit.into(), ty: Some(Type::Float(mlirformat::FloatKind::F32)) }
    }
}

impl serde::Serialize for Real {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&Attribute::from(*self), s)
    }
}

impl<'de> serde::Deserialize<'de> for Real {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let err = <D::Error as serde::de::Error>::custom;
        match attribute_of(d)? {
            Attribute::Float { lit, .. } => lit.parse::<f64>().map(Real).map_err(|e| err(e.to_string())),
            a @ Attribute::Int { .. } => a.int().map(|v| Real(v as f64)).map_err(|e| err(e.0)),
            a => Err(err(format!("{a}: not a float"))),
        }
    }
}
