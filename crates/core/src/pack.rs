use mlirformat::{Attribute, Error, NamedAttr};

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

    pub fn pack(&self, op: &str, attrs: &[NamedAttr<'_>]) -> Result<Vec<u32>, Error> {
        let t = self.template(op, attrs)?;
        match t.holes.first() {
            Some(h) => Err(Error(format!("{op}: @{} is not a number yet", h.name))),
            None => Ok(t.dw),
        }
    }

    pub fn template(&self, op: &str, attrs: &[NamedAttr<'_>]) -> Result<Template, Error> {
        let item = self.get(op).ok_or_else(|| Error(format!("{op}: not an item of the table")))?;
        let mut p = Packer::new(*self, item);
        p.pack(attrs)?;
        Ok(Template { dw: p.dw, holes: p.holes })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hole {
    pub name: String,
    bit: u32,
    width: u32,
    shift: u32,
    add: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Template {
    pub dw: Vec<u32>,
    pub holes: Vec<Hole>,
}

impl Template {
    pub fn emit(&self, out: &mut Vec<u32>, value: impl Fn(&str) -> u64) {
        self.emit_indexed(out, |i| value(&self.holes[i].name));
    }

    pub fn emit_indexed(&self, out: &mut Vec<u32>, value: impl Fn(usize) -> u64) {
        let at = out.len();
        out.extend_from_slice(&self.dw);
        for (i, h) in self.holes.iter().enumerate() {
            let v = (value(i) + h.add) >> h.shift;
            let v = if h.width >= 64 { v } else { v & ((1 << h.width) - 1) };
            let mut rest = u128::from(v) << (h.bit % 32);
            for d in &mut out[at + (h.bit / 32) as usize..at + (h.bit + h.width).div_ceil(32) as usize] {
                *d |= rest as u32;
                rest >>= 32;
            }
        }
    }

    pub fn extend(&mut self, other: &Template) {
        let base = self.dw.len() as u32 * 32;
        self.dw.extend_from_slice(&other.dw);
        self.holes.extend(other.holes.iter().map(|h| Hole { bit: h.bit + base, ..h.clone() }));
    }
}

struct Packer<'l, 't> {
    layouts: Layouts<'t>,
    item: &'l Item<'t>,
    dw: Vec<u32>,
    holes: Vec<Hole>,
}

const META: &[&str] = &["elements", "sym_name", "align", "offset"];

impl<'l, 't> Packer<'l, 't> {
    fn new(layouts: Layouts<'t>, item: &'l Item<'t>) -> Self {
        Packer { layouts, item, dw: vec![0; item.length as usize], holes: Vec::new() }
    }

    fn err<T>(&self, msg: impl core::fmt::Display) -> Result<T, Error> {
        Err(Error(format!("{}: {msg}", self.item.op)))
    }

    fn pack(&mut self, attrs: &[NamedAttr<'_>]) -> Result<(), Error> {
        let get = |name: &str| attrs.iter().find(|a| a.name == name).map(|a| a.value.as_ref().unwrap_or(&Attribute::Unit));
        for a in attrs {
            if !META.contains(&&*a.name) && !self.item.fields.iter().any(|f| f.name == a.name) {
                return self.err(format_args!("no field {}", a.name));
            }
        }
        let auto_length = self.item.bias > 0 && get("dword_length").is_none() && self.item.fields.iter().any(|f| f.name == "dword_length");
        for f in self.item.fields {
            if !(auto_length && f.name == "dword_length") {
                self.field(f, 0, get(f.name))?;
            }
        }
        match (self.item.group, get("elements")) {
            (Some(g), Some(Attribute::Array(es))) => {
                self.dw.resize((g.at as usize + es.len() * g.size as usize).div_ceil(32).max(self.dw.len()), 0);
                for (i, e) in es.iter().enumerate() {
                    let Attribute::Dict(es) = e else { return self.err("an element is not a dictionary") };
                    for f in g.fields {
                        let v = es.iter().find(|a| a.name == f.name).map(|a| a.value.as_ref().unwrap_or(&Attribute::Unit));
                        self.field(f, g.at + i as u32 * g.size, v)?;
                    }
                }
            }
            (_, None) => {}
            _ => return self.err("elements: not a variable-length item, or not an array"),
        }
        if auto_length {
            let f = self.item.fields.iter().find(|f| f.name == "dword_length").unwrap();
            let n = self.dw.len() as u64 - u64::from(self.item.bias);
            self.put(f.bit, f.width, n, true);
        }
        Ok(())
    }

    fn field(&mut self, f: &Field<'_>, base: u32, value: Option<&Attribute<'_>>) -> Result<(), Error> {
        self.elements(f, base + f.bit, f.dims, value, value.is_some())
    }

    fn elements(&mut self, f: &Field<'_>, bit: u32, dims: &[(u32, u32)], value: Option<&Attribute<'_>>, set: bool) -> Result<(), Error> {
        let Some(((count, stride), inner)) = dims.split_first() else { return self.scalar(f, bit, value, set) };
        let items = match value {
            Some(Attribute::Array(v)) if v.len() == *count as usize => Some(v),
            None => None,
            _ => return self.err(format_args!("{}: not an array of {count}", f.name)),
        };
        for i in 0..*count {
            self.elements(f, bit + i * stride, inner, items.map(|v| &v[i as usize]), set)?;
        }
        Ok(())
    }

    fn scalar(&mut self, f: &Field<'_>, bit: u32, value: Option<&Attribute<'_>>, set: bool) -> Result<(), Error> {
        let v = match (f.ty, value) {
            (Ty::Struct(op), v) => {
                let Some(item) = self.layouts.get(op) else { return self.err(format_args!("{}: no struct {op}", f.name)) };
                let attrs = match v {
                    Some(Attribute::Dict(a)) => a.as_slice(),
                    None => &[],
                    _ => return self.err(format_args!("{}: not a dictionary", f.name)),
                };
                let mut inner = Packer::new(self.layouts, item);
                inner.pack(attrs)?;
                for (k, w) in inner.dw.iter().enumerate().take(f.width.div_ceil(32) as usize) {
                    let k = k as u32;
                    self.put(bit + 32 * k, (f.width - 32 * k).min(32), u64::from(*w), set);
                }
                self.holes.extend(inner.holes.into_iter().map(|h| Hole { bit: h.bit + bit, ..h }));
                return Ok(());
            }
            (Ty::Uint, Some(Attribute::Array(words))) if f.width > 64 => {
                for (k, w) in words.iter().enumerate() {
                    let (k, w) = (k as u32, Scalar::try_from(w)?.0 as u64);
                    self.put(bit + 32 * k, (f.width - 32 * k).min(32), w, set);
                }
                return Ok(());
            }
            (Ty::Mbo, Some(_)) => return self.err(format_args!("{}: must be one, not settable", f.name)),
            (ty, Some(Attribute::SymbolRef(p))) => return self.hole(f, ty, bit, p, 0),
            (ty, Some(Attribute::Array(v))) if matches!(&v[..], [Attribute::SymbolRef(_), Attribute::Int { .. }]) => {
                let (Attribute::SymbolRef(p), add) = (&v[0], Scalar::try_from(&v[1])?.0 as u64) else { unreachable!() };
                return self.hole(f, ty, bit, p, add);
            }
            (_, None) => f.default,
            (ty, Some(a)) => {
                let s = Scalar::try_from(a)?;
                match ty.encode(s, f.width, bit % 32) {
                    Some(v) => v,
                    None => return self.err(format_args!("{}: {a} does not fit {ty:?} of {} bits at bit {}", f.name, f.width, bit % 32)),
                }
            }
        };
        self.put(bit, f.width, v, set);
        Ok(())
    }

    fn hole(&mut self, f: &Field<'_>, ty: Ty<'_>, bit: u32, path: &[mlirformat::Str<'_>], add: u64) -> Result<(), Error> {
        let shift = match ty {
            Ty::Uint | Ty::Int | Ty::Bool => 0,
            Ty::Address => bit % 32,
            _ => return self.err(format_args!("{}: a symbol in a {ty:?} field", f.name)),
        };
        let name = path.iter().map(|s| s.trim_matches('"')).collect::<Vec<_>>().join("::");
        self.holes.push(Hole { name, bit, width: f.width, shift, add });
        Ok(())
    }

    fn put(&mut self, bit: u32, width: u32, v: u64, set: bool) {
        let v = if width >= 64 { v } else { v & ((1 << width) - 1) };
        let mut rest = u128::from(v) << (bit % 32);
        for k in (bit / 32) as usize..(bit + width).div_ceil(32) as usize {
            let d = rest as u32;
            rest >>= 32;
            if k >= self.dw.len() {
                if !set || d == 0 {
                    continue;
                }
                self.dw.resize(k + 1, 0);
            }
            self.dw[k] |= d;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Scalar(i128, Option<f64>);

impl TryFrom<&Attribute<'_>> for Scalar {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Scalar, Error> {
        let int = |lit: &str| {
            let (neg, l) = lit.strip_prefix('-').map_or((false, lit), |l| (true, l));
            let v = match l.strip_prefix("0x") {
                Some(h) => i128::from_str_radix(h, 16),
                None => l.parse(),
            };
            v.ok().map(|v| if neg { -v } else { v })
        };
        match a {
            Attribute::Unit => Ok(Scalar(1, None)),
            Attribute::Bool(b) => Ok(Scalar((*b).into(), None)),
            Attribute::Int { lit, .. } => int(lit).map(|v| Scalar(v, Some(v as f64))).ok_or_else(|| Error(format!("bad integer {lit}"))),
            Attribute::Float { lit, .. } => lit.parse::<f64>().map(|v| Scalar(v as i128, Some(v))).map_err(|_| Error(format!("bad float {lit}"))),
            _ => Err(Error(format!("{a} is not a number"))),
        }
    }
}

impl Ty<'_> {
    fn encode(self, s: Scalar, width: u32, lo: u32) -> Option<u64> {
        let fits = |v: i128| v >= 0 && (width >= 64 || v >> width == 0);
        let frac = |v: f64, f: u8| (v * (1u64 << f) as f64).round() as i128;
        match self {
            Ty::Bool => (s.0 == 0 || s.0 == 1).then_some(s.0 as u64),
            Ty::Uint => fits(s.0).then_some(s.0 as u64),
            Ty::Int => (s.0 >= -(1i128 << (width - 1)) && s.0 < 1i128 << width).then_some(s.0 as u64),
            Ty::Float => Some(u64::from((s.1? as f32).to_bits())),
            Ty::Ufixed(f) => Some(frac(s.1?, f)).filter(|v| fits(*v)).map(|v| v as u64),
            Ty::Sfixed(f) => Some(frac(s.1?, f) as u64),
            Ty::Address => (s.0 >= 0 && s.0 & ((1 << lo) - 1) == 0 && (width + lo >= 64 || s.0 >> (width + lo) == 0)).then_some((s.0 >> lo) as u64),
            Ty::Struct(_) | Ty::Mbo => None,
        }
    }
}
