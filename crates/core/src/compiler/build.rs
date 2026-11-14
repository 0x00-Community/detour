use mlirformat::{Attribute, Error, Extended, NamedAttr, Type};
use mlirpass::OpData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegType<'t> {
    pub name: &'t str,
    pub elem: &'t str,
}

impl<'a> From<RegType<'_>> for Type<'a> {
    fn from(r: RegType<'_>) -> Self {
        Type::Extended(Extended { name: r.name.to_owned().into(), body: Some(format!("<{}>", r.elem).into()), params: Vec::new() }.into())
    }
}

impl<'t> TryFrom<&'t Type<'_>> for RegType<'t> {
    type Error = Error;

    fn try_from(t: &'t Type<'_>) -> Result<RegType<'t>, Error> {
        let not = || Error(format!("{t} is not a register type"));
        let Type::Extended(e) = t else { return Err(not()) };
        let elem = e.body.as_deref().and_then(|b| b.strip_prefix('<')?.strip_suffix('>')).ok_or_else(not)?;
        Ok(RegType { name: &e.name, elem })
    }
}

#[derive(Clone, Debug)]
pub enum A {
    Bool(bool),
    Int(i128),
    F32(f32),
    Sym(String),
    In(String, String),
    Str(String),
    Raw(Attribute<'static>),
}

impl From<bool> for A {
    fn from(v: bool) -> Self {
        A::Bool(v)
    }
}

macro_rules! ints {
    ($($t:ty),*) => {$(
        impl From<$t> for A {
            fn from(v: $t) -> Self {
                A::Int(v.into())
            }
        }
    )*};
}

ints!(u8, u16, u32, u64, i32, i64);

impl From<f32> for A {
    fn from(v: f32) -> Self {
        A::F32(v)
    }
}

impl From<Attribute<'static>> for A {
    fn from(v: Attribute<'static>) -> Self {
        A::Raw(v)
    }
}

impl A {
    pub fn sym(s: impl Into<String>) -> A {
        A::Sym(s.into())
    }

    pub fn within(memory: impl Into<String>, s: impl Into<String>) -> A {
        A::In(memory.into(), s.into())
    }

    pub fn str(s: impl Into<String>) -> A {
        A::Str(s.into())
    }
}

impl<'a> From<A> for Attribute<'a> {
    fn from(v: A) -> Self {
        match v {
            A::Bool(b) => Attribute::Bool(b),
            A::Int(i) => Attribute::Int { lit: if i < 0 { i.to_string() } else { format!("{i:#x}") }.into(), ty: None },
            A::F32(f) => Attribute::Float { lit: format!("{f:?}").into(), ty: Some(Type::Float(mlirformat::FloatKind::F32)) },
            A::Sym(s) => Attribute::SymbolRef(vec![s.into()]),
            A::In(m, s) => Attribute::SymbolRef(vec![m.into(), s.into()]),
            A::Str(s) => Attribute::Str { lit: s.into(), ty: None },
            A::Raw(a) => a,
        }
    }
}

impl A {
    pub fn dict<'a>(entries: Vec<(&str, A)>) -> Attribute<'a> {
        Attribute::Dict(entries.into_iter().map(|(n, v)| NamedAttr { name: String::from(n).into(), value: Some(v.into()) }).collect())
    }
}

pub trait WithAttrs {
    fn with_attrs(self, attrs: Vec<(&str, A)>) -> Self;
}

impl WithAttrs for OpData<'_> {
    fn with_attrs(mut self, attrs: Vec<(&str, A)>) -> Self {
        for (n, v) in attrs {
            self.set(n, v.into());
        }
        self
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

impl Fill {
    pub fn at(name: impl Into<String>, add: i64) -> Fill {
        Fill::Place { path: vec![name.into()], add }
    }
}

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

impl<'a> From<&Fill> for Attribute<'a> {
    fn from(f: &Fill) -> Self {
        match f {
            Fill::Int(v) => A::Int(*v).into(),
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
                [Attribute::SymbolRef(s), n @ Attribute::Int { .. }] => Ok(Fill::Place { path: path(s), add: crate::base::Attrs::as_int(n)? }),
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
