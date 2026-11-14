//! ⭐ What a GPU runs, for every vendor: the `image` dialect, its layout (GPU addresses) and its bytes.
//!
//! ```text
//!   image.memory @m {mem = "streaming"} {...}   one buffer object: what is placed in it, in order
//!   image.stream @s {...}                        a command stream of vendor ops (Intel's batch); the vendor ends it
//!   image.kernel @k {...}                        a shader: vendor instructions
//!   image.data @d {bytes = dense<...>}           raw bytes (vertices, uploads), or {size} zeros
//! ```
//!
//! A vendor op placed directly in a memory (a state struct) may carry `sym_name` and `align`. An attribute may name a
//! place: `@x` is x's GPU address, `@m::@x` its offset in memory `m`, `[@x, n]` x's address plus n (n may be negative:
//! an address relative to a window). [`Layout`] gives
//! every memory an address (`addr`) and a size (`size`), every item an offset (`offset`, aligned to its `align`, 64 by
//! default), and turns the symbols into numbers; then [`ImageEmit`] has everything it packs. What bytes a vendor op
//! is, is the vendor's [`ImageTarget`]; what ends a stream or pads a kernel is a vendor pass, before the layout.

use std::collections::BTreeMap;

use mlirformat::{Attribute, Const, Error, FunctionType, NamedAttr, Op, Operation, Region, ResultDef, Role, Value};
use mlirpass::{BlockId, Body, Emit, OpData, OpId, Quiet, Rewriter, ValueId};

use crate::base::{A, Attrs, MemType, Stage, Symbol};

/// ⭐ The `image` dialect: load it with `MlirOpt::load(ImageDialect)` (the [`Compiler`](crate::base::Compiler) does).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageDialect;

impl mlirformat::Dialect for ImageDialect {
    fn namespace(&self) -> &'static str {
        "image"
    }

    fn ops(&self) -> Vec<mlirformat::Template> {
        Containers.ops()
    }

    /// Every op has effects (none may be dropped as dead).
    fn op_infos(&self) -> Vec<mlirformat::OpInfo> {
        ImageDialect::INHERENT.iter().map(|(name, inherent)| mlirformat::OpInfo { name, traits: &[], effects: mlirformat::Effects::Unknown, inherent }).collect()
    }
}

impl ImageDialect {
    /// Each op's properties, by name (checked against the typed ops' fields by the `inherent` test).
    pub const INHERENT: &[(&str, &[&str])] = &[
        ("image.memory", &["sym_name", "mem", "align", "size", "addr"]),
        ("image.stream", &["sym_name", "align", "offset"]),
        ("image.kernel", &["sym_name", "stage", "align", "offset", "pad", "simd", "payload", "inputs", "varyings", "targets", "omask", "imask", "bindings", "buffers", "push", "reserve", "threads", "shared", "params"]),
        ("image.data", &["sym_name", "bytes", "size", "align", "offset"]),
    ];
}

/// `image.memory @m {mem = "coherent"} {...}`: one buffer object; [`Layout`] gives it `addr` and `size` (at least
/// the `size` asked, a page multiple).
#[mlirformat::mlir::op("image.memory", format = "$sym_name attr-dict-with-keyword $body")]
pub struct Memory<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    /// How the GPU caches it (`streaming` when not said).
    pub mem: Option<MemType>,
    pub align: Option<u64>,
    pub size: Option<u64>,
    pub addr: Option<u64>,
    pub body: Region<'a>,
}

/// `image.stream @s {...}`: a command stream of vendor ops.
#[mlirformat::mlir::op("image.stream", format = "$sym_name attr-dict-with-keyword $body")]
pub struct Stream<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub align: Option<u64>,
    pub offset: Option<u64>,
    pub body: Region<'a>,
}

/// `image.kernel @k {stage = "compute"} {...}`: a shader of vendor instructions, with what its vendor says of it
/// (Intel's: `simd`, `payload`, `inputs`, `varyings`, `targets`, `bindings`, `buffers`, `reserve`, `params`; AMD's and
/// Apple's kernels are mlirpass's own `rdna2.kernel` / `agx.kernel`).
#[mlirformat::mlir::op("image.kernel", format = "$sym_name attr-dict-with-keyword $body")]
pub struct Kernel<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub stage: Option<Stage>,
    pub align: Option<u64>,
    pub offset: Option<u64>,
    /// Bytes of padding after its instructions (instruction prefetch).
    pub pad: Option<u64>,
    pub simd: Option<u32>,
    /// The registers its inputs arrive in.
    pub payload: Option<Vec<i64>>,
    pub inputs: Option<u32>,
    pub varyings: Option<u32>,
    /// A fragment shader's color targets (its results; but the sample mask).
    pub targets: Option<u32>,
    /// A fragment shader writes the sample mask (WGSL `@builtin(sample_mask)` out: its last result).
    pub omask: Option<bool>,
    /// A fragment shader reads its input coverage mask (`detour.coverage_mask`: in its payload, past the barycentrics).
    pub imask: Option<bool>,
    pub bindings: Option<u32>,
    pub buffers: Option<u32>,
    /// The push constants' bytes it reads (`detour.push_constant`: the end of the furthest).
    pub push: Option<u32>,
    /// Registers kept from the allocator (the payload's).
    pub reserve: Option<u32>,
    /// A compute kernel's threads a workgroup (`detour.workgroup`; 1 when absent) and its shared memory's bytes
    /// (`detour.shared`).
    pub threads: Option<u32>,
    pub shared: Option<u32>,
    pub params: Option<u32>,
    pub body: Region<'a>,
}

/// `image.data @d {bytes = dense<...>}`: raw bytes, or `{size}` zeros (both: the bytes, then zeros up to the size).
#[mlirformat::mlir::op("image.data", format = "$sym_name attr-dict")]
pub struct Data<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub bytes: Option<Attribute<'a>>,
    pub size: Option<u64>,
    pub align: Option<u64>,
    pub offset: Option<u64>,
}

#[mlirformat::mlir::dialect("image", ops = [Memory, Stream, Kernel, Data])]
struct Containers;

impl<'a> Memory<'a> {
    /// Memory `name`, cached as `mem`.
    pub fn named(name: impl Into<String>, mem: MemType) -> Self {
        Memory::new(A::str(name.into()).into(), Some(mem), None, None, None, Image::region())
    }
}

impl<'a> Stream<'a> {
    pub fn named(name: impl Into<String>) -> Self {
        Stream::new(A::str(name.into()).into(), None, None, Image::region())
    }
}

impl<'a> Kernel<'a> {
    /// Kernel `name` (its `sym_name`) of `stage`, nothing else said.
    pub fn of(name: Attribute<'a>, stage: Stage) -> Self {
        Kernel::new(name, Some(stage), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, Image::region())
    }

    /// Its name.
    pub fn name(&self) -> Result<&str, Error> {
        Symbol::defined(&self.sym_name)
    }
}

impl Kernel<'_> {
    /// ⭐ The `image.kernel`s under `op`, in order.
    pub fn all(body: &Body<'_>, op: OpId) -> Vec<OpId> {
        let mut out = Vec::new();
        body.walk(op, &mut |o| {
            if body[o].name == Kernel::NAME {
                out.push(o);
            }
        });
        out
    }
}

impl<'a> Data<'a> {
    /// `size` zeros.
    pub fn zeros(name: impl Into<String>, size: u64) -> Self {
        Data::new(A::str(name.into()).into(), None, Some(size), None, None)
    }

    /// `bytes` (a dense constant).
    pub fn bytes(name: impl Into<String>, bytes: Attribute<'a>) -> Self {
        Data::new(A::str(name.into()).into(), Some(bytes), None, None, None)
    }

    /// Its bytes, then zeros up to its size.
    fn contents(&self) -> Result<Vec<u8>, Error> {
        let mut bytes = self.bytes.as_ref().map(Image::dense).transpose()?.unwrap_or_default();
        bytes.resize(bytes.len().max(self.size.unwrap_or(0) as usize), 0);
        Ok(bytes)
    }
}

/// ⭐ Typed ops in a [`Body`] by the op alone: a region op (a memory, a kernel) without exporting what is in it, any
/// op without naming every value of the body ([`Body::operation`] does both). Operands and results are named by their
/// [`ValueId`]s.
pub trait Heads<'a> {
    /// Op `op`'s attributes as the typed `O` (its regions read empty).
    fn head<O: Op>(&self, op: OpId) -> Result<O, Error>;
    /// Op `op`'s attributes set to `typed`'s (what `typed` leaves out stays; regions as they are).
    fn store<O>(&mut self, op: OpId, typed: &O) -> Result<(), Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>;
    /// Op `op`'s attributes read as the typed `O`, changed by `f`, stored.
    fn update<O: Op>(&mut self, op: OpId, f: impl FnOnce(&mut O)) -> Result<(), Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>,
    {
        let mut typed = self.head::<O>(op)?;
        f(&mut typed);
        self.store(op, &typed)
    }
    /// Typed op `typed` (a memory, a stream) detached with one region of one empty block: the op and its block.
    fn container<O>(&mut self, typed: &O) -> Result<(OpId, BlockId), Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>;
}

/// [`Heads::update`] through a [`Rewriter`]: the listener hears of the change.
pub trait Updates {
    fn update<O: Op>(&mut self, op: OpId, f: impl FnOnce(&mut O)) -> Result<(), Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>;
}

impl Updates for Rewriter<'_, '_> {
    fn update<O: Op>(&mut self, op: OpId, f: impl FnOnce(&mut O)) -> Result<(), Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>,
    {
        let mut done = Ok(());
        self.modify(op, |b| done = b.update(op, f));
        done
    }
}

/// ⭐ A typed op with regions as a detached [`OpData`]: its name and attributes (its regions are added in the body;
/// [`OpData::of`] refuses them).
pub trait OpHead<'a>: Sized {
    fn head<O>(typed: &O) -> Result<Self, Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>;
}

impl<'a> OpHead<'a> for OpData<'a> {
    fn head<O>(typed: &O) -> Result<Self, Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>,
    {
        let g = Operation::try_from(typed)?;
        let mut data = OpData::build(&g.name);
        if let Some(Attribute::Dict(props)) = g.properties {
            props.into_iter().for_each(|a| data.set(&a.name, a.value.unwrap_or(Attribute::Unit)));
        }
        g.attributes.into_iter().for_each(|a| data.set(&a.name, a.value.unwrap_or(Attribute::Unit)));
        Ok(data)
    }
}

impl<'a> Heads<'a> for Body<'a> {
    fn head<O: Op>(&self, op: OpId) -> Result<O, Error> {
        let d = &self[op];
        // the properties by the typed op's own fields (an attribute set after building may still be discardable)
        let (props, attributes): (Vec<_>, Vec<_>) = d.all_attrs().into_iter().partition(|a| O::FIELDS.iter().any(|f| matches!(f.role, Role::Prop { .. }) && f.name == &*a.name));
        // values named by their ids (as `OpData::of` reads them back)
        let value = |v: &ValueId| Value::from(*v);
        let types = |vs: &[ValueId]| vs.iter().map(|v| self[*v].ty.clone()).collect();
        let g = Operation {
            name: d.name.clone(),
            results: d.results.first().map(|r| ResultDef { name: value(r).name, count: (d.results.len() > 1).then_some(d.results.len() as u32) }).into_iter().collect(),
            operands: d.operands.iter().map(value).collect(),
            properties: (!props.is_empty()).then_some(Attribute::Dict(props)),
            attributes,
            regions: d.regions.iter().map(|_| Image::region()).collect(),
            ty: FunctionType { inputs: types(&d.operands), results: types(&d.results) },
            ..Default::default()
        };
        g.to_op()
    }

    fn store<O>(&mut self, op: OpId, typed: &O) -> Result<(), Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>,
    {
        let data = OpData::head(typed)?;
        for a in data.all_attrs() {
            self[op].set(&a.name, a.value.unwrap_or(Attribute::Unit));
        }
        Ok(())
    }

    fn container<O>(&mut self, typed: &O) -> Result<(OpId, BlockId), Error>
    where
        for<'o> Operation<'static>: TryFrom<&'o O, Error = Error>,
    {
        let op = self.create(OpData::head(typed)?, &[]);
        let r = self.add_region(op);
        Ok((op, self.add_block(r, &[])))
    }
}

/// A placed item's discardable `align` / `offset` (a vendor op placed in a memory carries them as the image ops do).
pub struct Placed;

impl Placed {
    pub const ALIGN: &str = "align";
    pub const OFFSET: &str = "offset";
}

/// Rewrite patterns, a unit struct each, declared as mlirpass declares its own (its `rewrites!` is crate internal):
/// `/// doc` `Name("root", benefit) |rw, op| { .. }` (`_` for any root, `(Op::NAME)` for a typed op's); the body says whether it rewrote.
#[macro_export]
macro_rules! rewrites {
    (@root _) => { None };
    (@root $root:tt) => { Some($root) };
    ($($(#[$m:meta])* $name:ident($root:tt, $benefit:literal) |$rw:ident, $op:ident| $body:block)*) => {$(
        $(#[$m])*
        pub struct $name;

        impl mlirpass::Pattern for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn root(&self) -> Option<&str> {
                $crate::rewrites!(@root $root)
            }

            fn benefit(&self) -> u16 {
                $benefit
            }

            #[allow(unused_variables)]
            fn rewrite(&self, $rw: &mut mlirpass::Rewriter<'_, '_>, $op: mlirpass::OpId) -> bool $body
        }
    )*};
}

/// ⭐ The greedy driver with only the given patterns: no folding, no constant uniquing, no region simplification
/// (a backend's patterns change what they say and nothing else).
pub struct Rewrites;

impl Rewrites {
    /// `patterns` over everything in `op` (a module) until none applies.
    pub fn apply(body: &mut Body<'_>, op: OpId, patterns: &mlirpass::PatternSet) {
        let config = mlirpass::GreedyConfig { region_simplify: mlirpass::RegionSimplify::Disabled, fold: false, cse_constants: false, ..Default::default() };
        let region = body[op].regions[0];
        body.apply_greedily(region, patterns, &config);
    }
}

/// ⭐ A dialect's `op_infos` rows: every op of `ops` with unknown effects (none may be dropped as dead: a command,
/// an instruction), the ones in `inherent` with those attributes declared (they live in `<{...}>`: the custom form
/// prints them).
pub struct OpRows;

impl OpRows {
    pub fn of(ops: &[mlirformat::Template], inherent: &[(&'static str, &'static [&'static str])]) -> Vec<mlirformat::OpInfo> {
        // looked up once an op (a dialect may declare thousands)
        // (the first row of a name wins, as a search would find it)
        let inherent: alloc::collections::BTreeMap<&str, &'static [&'static str]> = inherent.iter().rev().copied().collect();
        ops.iter()
            .map(|t| {
                let inherent = inherent.get(t.name).copied().unwrap_or_default();
                mlirformat::OpInfo { name: t.name, traits: &[], effects: mlirformat::Effects::Unknown, inherent }
            })
            .collect()
    }
}

/// Ops whose fields are attributes (`Type "op"`: a command, a state, a register, an instruction) in dialect `$ns`,
/// gathered in the op table `$set` (`$set.ops()` for the dialect's `ops`). An instruction is virtual before its
/// registers are allocated: its SSA operands (filling the source fields not given, `src0`, `src1` ...) and results
/// are declared too, so both forms are the same op: the virtual one (SSA operands and results) and the allocated one
/// (attributes only).
#[macro_export]
macro_rules! attr_ops {
    ($ns:tt $set:ident; $($ty:ident $op:literal,)*) => {
        $(
            #[doc = concat!("`", $op, "`")]
            #[mlirformat::mlir::op($op, format = "(`(` $operands^ `:` type($operands) `)`)? attr-dict (`->` type($results)^)?", results(results))]
            pub struct $ty<'a> {
                #[slot(types)]
                pub operands: Vec<mlirformat::Value<'a>>,
                pub types: Vec<mlirformat::Type<'a>>,
                pub results: Vec<mlirformat::Type<'a>>,
            }
        )*

        #[mlirformat::mlir::dialect($ns, ops = [$($ty),*])]
        pub(crate) struct $set;
    };
}

/// ⭐ What a vendor says about its image: where memories go, and the bytes of its ops (symbols already numbers).
pub trait ImageTarget {
    /// Where the first memory starts in the GPU's address space.
    const BASE: u64;
    /// Memories are this aligned at least (a buffer object's page).
    const PAGE: u64;

    /// An `image.stream` ⇒ its bytes (its ops packed as they are: what ends it is a vendor pass's).
    fn stream(&self, body: &Body<'_>, stream: OpId) -> Result<Vec<u8>, Error>;
    /// An `image.kernel` ⇒ its bytes.
    fn kernel(&self, body: &Body<'_>, kernel: OpId) -> Result<Vec<u8>, Error>;
    /// A vendor op placed in a memory itself (a state struct) ⇒ bytes.
    fn item(&self, body: &Body<'_>, op: OpId) -> Result<Vec<u8>, Error>;

    /// The bytes of anything placed in a memory.
    fn contents(&self, body: &Body<'_>, op: OpId) -> Result<Vec<u8>, Error> {
        let name = &*body[op].name;
        if name == Stream::NAME {
            self.stream(body, op)
        } else if name == Kernel::NAME {
            self.kernel(body, op)
        } else if name == Data::NAME {
            body.head::<Data>(op)?.contents()
        } else {
            self.item(body, op)
        }
    }
}

/// Where a symbol is.
#[derive(Clone, Copy)]
struct Place<'s> {
    memory: &'s str,
    addr: u64,
    offset: u64,
}

/// Symbols ⇒ numbers.
struct Resolver<'s>(BTreeMap<String, Place<'s>>);

impl Resolver<'_> {
    fn int<'a>(v: u64) -> Attribute<'a> {
        A::from(v).into()
    }

    fn get(&self, s: &str) -> Result<Place<'_>, Error> {
        self.0.get(s.trim_matches('"')).copied().ok_or_else(|| Error(format!("no symbol @{s}")))
    }

    /// `a` with its symbols numbers (every symbol 0 with `zero`: to size things before there are addresses).
    fn attr<'a>(&self, a: &Attribute<'a>, zero: bool) -> Result<Attribute<'a>, Error> {
        Ok(match a {
            _ if zero && matches!(a, Attribute::SymbolRef(_)) => Self::int(0),
            Attribute::SymbolRef(p) if p.len() == 1 => Self::int(self.get(&p[0])?.addr),
            Attribute::SymbolRef(p) if p.len() == 2 => {
                let x = self.get(&p[1])?;
                if x.memory != p[0].trim_matches('"') {
                    return Err(Error(format!("@{} is not in memory @{}", p[1], p[0])));
                }
                Self::int(x.offset)
            }
            // sizing: the whole sum 0 (an offset may be negative)
            Attribute::Array(v) if zero && matches!(&v[..], [Attribute::SymbolRef(_), Attribute::Int { .. }]) => Self::int(0),
            Attribute::Array(v) if matches!(&v[..], [Attribute::SymbolRef(_), Attribute::Int { .. }]) => {
                let base = Attrs::as_int(&self.attr(&v[0], zero)?)? as u64;
                Self::int(base.wrapping_add(Attrs::as_int(&v[1])? as u64))
            }
            Attribute::Array(v) => Attribute::Array(v.iter().map(|e| self.attr(e, zero)).collect::<Result<_, _>>()?),
            Attribute::Dict(d) => Attribute::Dict(d.iter().map(|n| Ok(NamedAttr { name: n.name.clone(), value: n.value.as_ref().map(|v| self.attr(v, zero)).transpose()? })).collect::<Result<_, Error>>()?),
            _ => a.clone(),
        })
    }

    /// Every attribute of `op` and of the ops in it, resolved.
    fn op(&self, rw: &mut Rewriter<'_, '_>, op: OpId, zero: bool) -> Result<(), Error> {
        let mut ops = Vec::new();
        rw.body.walk(op, &mut |o| ops.push(o));
        for o in ops {
            for a in rw.body[o].all_attrs() {
                if let Some(v) = &a.value
                    && a.name != Symbol::ATTR
                {
                    let r = self.attr(v, zero)?;
                    if &r != v {
                        rw.modify(o, |b| b[o].set(&a.name, r));
                    }
                }
            }
        }
        Ok(())
    }
}

/// ⭐ The layout of a module for target `T` (a vendor's `<vendor>-layout` pass runs it).
pub struct Layout<'t, T: ImageTarget>(pub &'t T);

impl<T: ImageTarget> Layout<'_, T> {
    pub fn run(&self, body: &mut Body<'_>, op: OpId) -> Result<(), Error> {
        // sizes first, on a copy whose symbols are 0 (packing does not depend on them)
        let mut sized = body.clone();
        Resolver(BTreeMap::new()).op(&mut Rewriter::new(&mut sized, &mut Quiet), op, true)?;
        let mut quiet = Quiet;
        let mut rw = Rewriter::new(body, &mut quiet);
        let memories = Image::memories(rw.body, op);
        let heads = memories.iter().map(|m| rw.body.head::<Memory>(*m)).collect::<Result<Vec<_>, _>>()?;
        let mut places: Vec<(String, String, u64, u64)> = Vec::new(); // (symbol, memory, address, offset)
        let mut at = T::BASE;
        for (&m, mut head) in memories.iter().zip(heads) {
            let name = Symbol::defined(&head.sym_name)?.to_owned();
            let align = head.align.unwrap_or(T::PAGE).max(T::PAGE);
            let addr = at.next_multiple_of(align);
            let mut cursor = 0u64;
            for item in rw.body.block_ops(rw.body.top_block(m)) {
                cursor = cursor.next_multiple_of(Image::int(rw.body, item, Placed::ALIGN)?.unwrap_or(64));
                if let Some(s) = rw.body[item].get(Symbol::ATTR) {
                    places.push((Symbol::defined(s)?.to_owned(), name.clone(), addr + cursor, cursor));
                }
                rw.modify(item, |b| b[item].set(Placed::OFFSET, A::from(cursor).into()));
                cursor += self.0.contents(&sized, item)?.len() as u64;
            }
            let size = head.size.unwrap_or(0).max(cursor).next_multiple_of(4096).max(4096);
            (head.addr, head.size) = (Some(addr), Some(size));
            rw.update::<Memory>(m, |h| *h = head)?;
            places.push((name.clone(), name, addr, 0));
            at = addr + size;
        }
        let resolver = Resolver(places.iter().map(|(s, m, a, o)| (s.clone(), Place { memory: m, addr: *a, offset: *o })).collect());
        for m in memories {
            for item in rw.body.block_ops(rw.body.top_block(m)) {
                resolver.op(&mut rw, item, false)?;
            }
        }
        Ok(())
    }
}

/// One buffer object's worth: where it goes and what is in it.
#[derive(Debug, Clone)]
pub struct Segment {
    pub name: String,
    pub addr: u64,
    pub mem: MemType,
    pub bytes: Vec<u8>,
}

/// ⭐ What the GPU runs: the memories, and where the (first) command stream starts and ends.
#[derive(Debug, Clone)]
pub struct Image {
    pub segments: Vec<Segment>,
    pub stream: u64,
    /// The address after the stream's last byte.
    pub stream_end: u64,
}

impl Image {
    /// A region with nothing in it (a typed region op's, before its body has its blocks).
    pub fn region<'a>() -> Region<'a> {
        Region { blocks: Vec::new() }
    }

    pub fn segment(&self, name: &str) -> Option<&Segment> {
        self.segments.iter().find(|s| s.name == name)
    }

    /// The memories of the module.
    fn memories(body: &Body<'_>, op: OpId) -> Vec<OpId> {
        body.ops(body.top_block(op)).filter(|m| body[*m].name == Memory::NAME).collect()
    }

    /// A placed item's [`Placed`] attribute `name`.
    fn int(body: &Body<'_>, op: OpId, name: &str) -> Result<Option<u64>, Error> {
        Ok(body[op].get(name).map(Attrs::as_int).transpose()?.map(|v| v as u64))
    }

    /// A dense constant's elements, little endian.
    fn dense(a: &Attribute<'_>) -> Result<Vec<u8>, Error> {
        let c = Const::try_from(a).map_err(|()| Error("bytes: not a dense constant".into()))?;
        let n = c.ty.element_count().unwrap_or(1);
        let width = c.ty.element_type().bit_width().ok_or_else(|| Error("bytes: element without a width".into()))? as usize / 8;
        let mut out = Vec::with_capacity(n * width);
        for i in 0..n {
            let bits = match c.at(i) {
                mlirformat::Scalar::Float(f) => u128::from(f.bits),
                mlirformat::Scalar::Int(v) => v.bits(),
                mlirformat::Scalar::Complex(..) | mlirformat::Scalar::ComplexFloat(..) | mlirformat::Scalar::Wide(_) => {
                    return Err(Error("bytes: complex and wide float elements have no image layout".into()));
                }
            };
            out.extend_from_slice(&bits.to_le_bytes()[..width]);
        }
        Ok(out)
    }
}

/// ⭐ The module (after its layout) as an [`Image`], packed by target `T`.
pub struct ImageEmit<'t, T: ImageTarget>(pub &'t T);

impl<T: ImageTarget> Emit for ImageEmit<'_, T> {
    type Output = Image;

    fn emit(&self, body: &Body<'_>, op: OpId) -> Result<Image, Error> {
        let mut segments = Vec::new();
        let mut stream: Option<(u64, u64)> = None;
        for m in Image::memories(body, op) {
            let head = body.head::<Memory>(m)?;
            let name = Symbol::defined(&head.sym_name)?.to_owned();
            let addr = head.addr.ok_or_else(|| Error(format!("memory @{name} without an address: run the layout")))?;
            let size = head.size.unwrap_or(0);
            let mem = head.mem.unwrap_or(MemType::Streaming);
            let mut bytes = vec![0u8; size as usize];
            for item in body.block_ops(body.top_block(m)) {
                let offset = Image::int(body, item, Placed::OFFSET)?.unwrap_or(0) as usize;
                let b = self.0.contents(body, item)?;
                bytes[offset..offset + b.len()].copy_from_slice(&b);
                if body[item].name == Stream::NAME && stream.is_none() {
                    let at = addr + offset as u64;
                    stream = Some((at, at + b.len() as u64));
                }
            }
            segments.push(Segment { name, addr, mem, bytes });
        }
        let (stream, stream_end) = stream.ok_or_else(|| Error("no image.stream".into()))?;
        Ok(Image { segments, stream, stream_end })
    }
}

#[cfg(test)]
mod tests {
    use mlirformat::{Op, Role};

    use super::*;

    /// The inherent lists are the typed ops' properties, exactly.
    #[test]
    fn inherent() {
        fn props<O: Op>() -> (&'static str, Vec<&'static str>) {
            (O::NAME, O::FIELDS.iter().filter(|f| matches!(f.role, Role::Prop { .. })).map(|f| f.name).collect())
        }
        let typed = [props::<Memory>(), props::<Stream>(), props::<Kernel>(), props::<Data>()];
        assert_eq!(typed.len(), ImageDialect::INHERENT.len());
        for ((name, mut fields), (listed, inherent)) in typed.into_iter().zip(ImageDialect::INHERENT) {
            let mut inherent = inherent.to_vec();
            fields.sort_unstable();
            inherent.sort_unstable();
            assert_eq!((name, fields), (*listed, inherent));
        }
    }
}
