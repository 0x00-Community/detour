#[allow(unused_imports, reason = "the alloc prelude")]
use alloc::{borrow::ToOwned, format, string::{String, ToString}, vec, vec::Vec};

use mlirformat::{Attribute, Const, Error, FloatKind, Op, Scalar, ShapedKind, Type};

use mlirformat::spirv_types::SpirvType;

use crate::{Lanes, Matrices, A, Attrs, Body, ConversionRewriter, Heads, OpData, OpHead, OpId, Quiet, Rewriter, ShaderStage as Stage, SignatureConversion, TypeConverter, Updates, ValueId, WithAttrs, patterns};
use crate::{AddrSize, AddrSurface, Xe3Sampler as Sampler, CacheStore, DataSize, EuPayload, Grf, Lsc, LscOp, Render, SendInfo, Sfid, VectSize, Xe3Kernel as Kernel};

type Cr<'r, 'a, 'p> = ConversionRewriter<'r, 'a, 'p>;

/// Something that writes an EU instruction (name, operands, attributes, result element) and gives its result
type Emit<'e> = dyn FnMut(&str, &[ValueId], Vec<(&str, A)>, &'static str) -> ValueId + 'e;

mod std450;
use std450::Std450;

pub const XE3_BUFFER_BTI: u8 = 32;

pub const XE3_SIMD: u8 = 16;

pub const XE3_VS_ATTR_GRF: u8 = 2;

pub const XE3_MAX_VARYINGS: usize = 16;

/// The binding table entry of a draw's first texture: its one color target's is 0
pub const XE3_TEXTURE_BTI: u8 = 1;

pub struct Interpolate;

impl Interpolate {
    pub const BX: u8 = 1;
    pub const BY: u8 = 2;
    /// The first register of the attributes' setup data (after r0, then the barycentrics in r1..r2)
    pub const SETUP: u8 = 3;

    pub fn plane(setup: u8, k: usize, p: usize) -> String {
        let reg = setup as usize + k / 5;
        let byte = (k % 5) * 12 + 4 * p;
        format!("r{reg}.{byte}<0;1,0>:f")
    }

    pub fn end(setup: u8, components: usize) -> u8 {
        setup + components.div_ceil(5) as u8
    }
}

pub struct Xe3Types;

impl Xe3Types {
    pub fn lanes(t: &Type<'_>) -> Option<usize> {
        Some(Self::of(t)?.0)
    }

    fn elem(t: &Type<'_>) -> Option<&'static str> {
        match t {
            Type::Float(FloatKind::F32) | Type::Integer { width: 1, .. } => Some("f32"),
            Type::Integer { width: 8 | 16 | 32, .. } | Type::Index => Some("i32"),
            Type::Float(FloatKind::F64) => Some("f64"),
            Type::Integer { width: 64, .. } => Some("i64"),
            _ => None,
        }
    }

    pub fn of(t: &Type<'_>) -> Option<(usize, &'static str)> {
        if let Ok(SpirvType::Matrix { cols, column }) = SpirvType::try_from(t) {
            let (rows, elem) = Self::of(&column)?;
            return Some((rows * usize::try_from(cols).ok()?, elem));
        }
        match t {
            Type::Shaped(s) if s.kind == ShapedKind::Vector => match t.dims()?[..] {
                [n] if n > 0 => Some((n as usize, Self::elem(&s.elem)?)),
                _ => None,
            },
            _ => Some((1, Self::elem(t)?)),
        }
    }

    pub fn converter<'c>() -> TypeConverter<'c> {
        let mut tc = TypeConverter::default();
        tc.conversion(|t| Self::of(t).map(|(n, e)| Some(vec![Type::from(Grf(e)); n])));
        tc.conversion(|t| matches!(SpirvType::try_from(t), Ok(SpirvType::Ptr { .. } | SpirvType::Image { .. } | SpirvType::SampledImage(_))).then(|| Some(Vec::new())));
        tc.conversion(|t| Grf::try_from(t).is_ok().then(|| Some(vec![t.clone()])));
        tc
    }
}

pub struct GpuToXe3;

impl GpuToXe3 {
    pub const STAGE: &str = "spirv.execution_model";
    const BINDINGS: &str = "xe3.bindings";
    const BUFFERS: &str = "xe3.buffers";
    const BINDING: &str = "xe3.binding";
    const OFFSET: &str = "xe3.offset";
    const STRIDE: &str = "xe3.matrix_stride";
    const MATRIX_STRIDE: &str = "MatrixStride";

    fn ieee_prologue<'a>() -> [OpData<'a>; 2] {
        let or = vec![("exec", A::from(1u8)), ("dst", A::str("arf128<1>:ud")), ("src0", A::str("arf128<0;1,0>:ud")), ("src1", A::str("0xc0:ud")), ("w", A::Bool(true))];
        let sync = vec![("exec", A::from(1u8)), ("fn", A::str("nop")), ("swsb", A::str("A@1"))];
        [OpData::build("xe3.eu.or").with_attrs(or), OpData::build("xe3.eu.sync").with_attrs(sync)]
    }

    pub fn resources(body: &mut Body<'_>, shaders: &[OpId]) -> Result<(), Error> {
        for &f in shaders {
            let Some(Attribute::Array(attrs)) = body[f].get("arg_attrs").cloned() else { continue };
            let block = body.top_block(f);
            let args = body[block].args.clone();
            let mut res: alloc::collections::BTreeMap<ValueId, (String, i64)> = alloc::collections::BTreeMap::new();
            for (a, d) in args.iter().zip(&attrs) {
                let Attribute::Dict(d) = d else { continue };
                let Some(abi) = d.iter().find(|n| n.name == "spirv.interface_var_abi").and_then(|n| n.value.as_ref()) else { continue };
                let text = abi.to_string();
                let inner = text.split_once("<(").and_then(|(_, r)| r.split_once(')')).ok_or_else(|| Error(format!("{text}: not an interface ABI")))?;
                let mut numbers = inner.0.split(',').map(|n| n.trim().parse::<i64>());
                let (Some(Ok(set)), Some(Ok(binding))) = (numbers.next(), numbers.next()) else { return Err(Error(format!("{text}: no set and binding"))) };
                let class = inner.1.trim_start_matches(',').trim().trim_end_matches('>').to_owned();
                // what a draw binds: set 0's uniform blocks and sampled images
                if set != 0 || !matches!(&*class, "Uniform" | "UniformConstant") {
                    return Err(Error(format!("a {class} resource in set {set}: a draw binds set 0's uniform blocks and sampled images only")));
                }
                res.insert(*a, (class, binding));
            }
            let mut ops = Vec::new();
            body.walk(f, &mut |o| ops.push(o));
            let (mut textures, mut buffers) = (0i64, 0i64);
            let root = |body: &Body<'_>, mut v: ValueId| -> Option<ValueId> {
                for _ in 0..4 {
                    if res.contains_key(&v) {
                        return Some(v);
                    }
                    let crate::Def::Result(d, _) = body[v].def else { return None };
                    v = *body[d].operands.first()?;
                }
                None
            };
            for o in ops {
                let name = body[o].name.to_string();
                if name.starts_with("spirv.Image") {
                    let Some(r) = body[o].operands.first().and_then(|v| root(body, *v)) else { continue };
                    let b = res[&r].1;
                    textures = textures.max(b + 1);
                    body[o].set(Self::BINDING, A::from(b).into());
                } else if name == "spirv.Load" {
                    let p = body[o].operands[0];
                    let crate::Def::Result(chain, _) = body[p].def else { continue };
                    if body[chain].name != "spirv.AccessChain" {
                        continue;
                    }
                    let base = body[chain].operands[0];
                    let Some(&(_, b)) = res.get(&base) else { continue };
                    let (offset, stride) = Self::member_offset(body, base, &body[chain].operands[1..]).ok_or_else(|| Error("a resource access chain: not constant struct members".into()))?;
                    if let Some(stride) = stride {
                        body[o].set(Self::STRIDE, A::from(stride).into());
                    }
                    buffers = buffers.max(b + 1);
                    body[o].set(Self::BINDING, A::from(b).into());
                    body[o].set(Self::OFFSET, A::from(offset).into());
                    body.set_operand(o, 0, base);
                    if body[p].uses.is_empty() {
                        let indices: Vec<ValueId> = body[chain].operands[1..].to_vec();
                        body.erase(chain);
                        for i in indices {
                            if let crate::Def::Result(c, _) = body[i].def
                                && body[i].uses.is_empty()
                            {
                                body.erase(c);
                            }
                        }
                    }
                }
            }
            body[f].set(Self::BINDINGS, A::from(textures).into());
            body[f].set(Self::BUFFERS, A::from(buffers).into());
        }
        Ok(())
    }

    /// The byte offset of a constant access chain and, when it ends at a matrix, the matrix's column stride.
    fn member_offset(body: &Body<'_>, base: ValueId, indices: &[ValueId]) -> Option<(i64, Option<i64>)> {
        let Ok(SpirvType::Ptr { pointee, .. }) = SpirvType::try_from(&body[base].ty) else { return None };
        let mut ty = (*pointee).clone();
        let mut at = 0i64;
        let mut stride = None;
        for i in indices {
            let crate::Def::Result(c, _) = body[*i].def else { return None };
            let k = Attrs::as_int(body[c].get("value")?).ok()? as usize;
            if let Some((n, _)) = Xe3Types::of(&ty).filter(|_| SpirvType::try_from(&ty).is_err()) {
                // a vector's component
                if k >= n {
                    return None;
                }
                at += 4 * k as i64;
                ty = match &ty {
                    Type::Shaped(s) => (*s.elem).clone(),
                    t => t.clone(),
                };
                stride = None;
                continue;
            }
            match SpirvType::try_from(&ty).ok()? {
                SpirvType::Matrix { cols, column } if (k as u64) < u64::from(cols) => {
                    at += stride? * i64::try_from(k).ok()?;
                    ty = (*column).clone();
                    stride = None;
                }
                SpirvType::Struct { members: Some(m), .. } => {
                    at += i64::from(m.get(k)?.offset?);
                    stride = m[k].decorations.iter().find(|d| d.name == Self::MATRIX_STRIDE).and_then(|d| d.value.as_deref()?.parse().ok());
                    ty = m[k].ty.clone();
                }
                SpirvType::Array { elem, stride, .. } if stride != 0 => {
                    at += i64::from(stride) * i64::try_from(k).ok()?;
                    ty = *elem;
                }
                _ => return None,
            }
        }
        Some((at, stride))
    }

    pub fn place_arguments<'a>(body: &mut Body<'a>, module: OpId) -> Result<(), Error> {
        let mut quiet = Quiet;
        let mut rw = Rewriter::new(body, &mut quiet);
        let f32 = Type::from(Grf("f32"));
        let put = |rw: &mut Rewriter<'_, 'a>, data: OpData<'a>, ty: Type<'a>| {
            let p = rw.create(data.with_attrs(vec![("exec", A::from(XE3_SIMD))]), &[ty]);
            rw.body[p].results[0]
        };
        for k in Kernel::all(rw.body, module) {
            let head = rw.body.head::<Kernel>(k)?;
            let regs = head.payload.unwrap_or_default();
            let fragment = head.stage == Some(Stage::Fragment);
            let setup = Interpolate::SETUP;
            let block = rw.body.top_block(k);
            let args = rw.body[block].args.clone();
            rw.set_start(block);
            let bary = if fragment && !args.is_empty() {
                let [bx, by] = [Interpolate::BX, Interpolate::BY].map(|g| OpData::of(&EuPayload::new(g.into(), Grf("f32").into())).map(|(data, _)| data));
                Some((put(&mut rw, bx?, f32.clone()), put(&mut rw, by?, f32.clone())))
            } else {
                None
            };
            for (n, arg) in args.into_iter().enumerate() {
                let g = *regs.get(n).ok_or_else(|| Error("kernel: more arguments than payload entries".into()))?;
                let ty = rw.body[arg].ty.clone();
                if g <= Self::VERTEX_INDEX {
                    let r = Self::VERTEX_INDEX - g;
                    let [id, first] = [r, r + 2].map(|g| OpData::of(&EuPayload::new(g, Grf("i32").into())).map(|(data, _)| data));
                    let (id, first) = (put(&mut rw, id?, ty.clone()), put(&mut rw, first?, ty.clone()));
                    let v = put(&mut rw, OpData::build("xe3.eu.add").operands(&[id, first]), ty);
                    rw.replace_all_uses(arg, v);
                    continue;
                }
                if g < 0 {
                    let used = !rw.body[arg].uses.is_empty();
                    let Some(v) = Self::fragment_builtin(&mut rw, g, used)? else { continue };
                    rw.replace_all_uses(arg, v);
                    continue;
                }
                let v = match bary {
                    Some((bx, by)) => {
                        let k = g as usize;
                        let a = OpData::build("xe3.eu.mad").operands(&[by]).with_attrs(vec![("src0", A::str(Interpolate::plane(setup, k, 0))), ("src2", A::str(Interpolate::plane(setup, k, 2)))]);
                        let v = put(&mut rw, a, f32.clone());
                        put(&mut rw, OpData::build("xe3.eu.mad").operands(&[v, bx]).with_attrs(vec![("src2", A::str(Interpolate::plane(setup, k, 1)))]), f32.clone())
                    }
                    None => put(&mut rw, OpData::of(&EuPayload::new(g, ty.clone()))?.0, ty),
                };
                rw.replace_all_uses(arg, v);
            }
            rw.body[block].args.clear();
            rw.set_start(block);
            for data in Self::ieee_prologue() {
                rw.create(data, &[]);
            }
            let reserve = match fragment {
                true => Interpolate::end(setup, regs.iter().filter(|k| **k >= 0).map(|k| *k as usize + 1).max().unwrap_or(0)),
                false => regs.iter().map(|&g| if g <= Self::VERTEX_INDEX { Self::VERTEX_INDEX - g + 3 } else { g + 1 }).max().unwrap_or(0).max(0) as u8,
            };
            rw.update::<Kernel>(k, |k| k.reserve = Some(u32::from(reserve)))?;
        }
        Ok(())
    }

    /// A fragment built-in's value (`Self::FRAG_COORD`, `Self::FRONT_FACING`) from the SIMD16 thread payload, as
    /// Mesa reads it on Xe2+: each subspan's (2×2 pixels) upper-left x and y as words at r0.40, 44, 48, 52 (y two
    /// bytes after x), lanes in subspan order and in each the upper left, upper right, lower left, lower right
    /// pixel; the polygon's back-facing bit 11 of the word at r0.36. None: a component nothing reads. The offsets
    /// added to each lane are Mesa's: into a dword, lane i takes the immediate vector's half-byte 2i (mod 8)
    fn fragment_builtin<'a>(rw: &mut Rewriter<'_, 'a>, code: i64, used: bool) -> Result<Option<ValueId>, Error> {
        let mut put = |name: &str, operands: &[ValueId], attrs: Vec<(&str, A)>, elem: &'static str| {
            let data = OpData::build(name).operands(operands).with_attrs(attrs).with_attrs(vec![("exec", A::from(XE3_SIMD))]);
            let p = rw.create(data, &[Type::from(Grf(elem))]);
            rw.body[p].results[0]
        };
        let axis = |put: &mut Emit<'_>, at: u8, offsets: u32| {
            let int = put("xe3.eu.add", &[], vec![("src0", A::str(format!("r0.{at}<2;4,0>:uw"))), ("src1", A::str(format!("{offsets:#x}:v")))], "i32");
            let float = put("xe3.eu.mov", &[int], vec![], "f32");
            put("xe3.eu.add", &[float], vec![("src1", A::str("0x3f000000:f"))], "f32")
        };
        Ok(match code {
            c if c == Self::FRAG_COORD => Some(axis(&mut put, 40, 0x0100_0100)),
            c if c == Self::FRAG_COORD - 1 => Some(axis(&mut put, 42, 0x0101_0000)),
            c if c == Self::FRONT_FACING => {
                let back = put("xe3.eu.and", &[], vec![("src0", A::str("r0.36<0;1,0>:uw")), ("src1", A::str("0x800:uw")), ("as", A::str("ud"))], "i32");
                let bit = put("xe3.eu.shr", &[back], vec![("src1", A::str("0xb:ud")), ("unsigned", A::Bool(true))], "i32");
                let front = put("xe3.eu.add", &[bit], vec![("src1", A::str("0xffffffff:d"))], "i32");
                Some(put("xe3.eu.mov", &[front], vec![("as", A::str("ud"))], "f32"))
            }
            _ if used => return Err(Error("gl_FragCoord.z and .w are not supported yet".into())),
            _ => None,
        })
    }

    const URB: Lsc = Lsc {
        op: LscOp::Store,
        surface: AddrSurface::Flat,
        addr_size: AddrSize::A32,
        data_size: DataSize::D32,
        cache: CacheStore::L1UC_L3UC,
        vect_size: VectSize::V4,
        transpose: false,
        cmask: 0,
    };

    fn create<'a>(cr: &mut Cr<'_, 'a, '_>, before: OpId, name: &str, operands: &[ValueId], attrs: Vec<(&str, A)>, types: &[Type<'a>]) -> Vec<ValueId> {
        let mut data = OpData::build(name).operands(operands);
        data.set("exec", A::from(XE3_SIMD).into());
        data = data.with_attrs(attrs);
        cr.rw.set_before(before);
        let op = cr.rw.create(data, types);
        cr.rw.body[op].results.to_vec()
    }

    pub fn eu(cr: &mut Cr<'_, '_, '_>, before: OpId, name: &str, operands: &[ValueId], attrs: Vec<(&str, A)>, elem: &str) -> ValueId {
        Self::create(cr, before, name, operands, attrs, &[Type::from(Grf(elem))])[0]
    }

    fn send(cr: &mut Cr<'_, '_, '_>, before: OpId, name: &str, payload: &[ValueId], attrs: Vec<(&str, A)>) {
        Self::create(cr, before, name, payload, attrs, &[]);
    }

    fn payload(cr: &mut Cr<'_, '_, '_>, before: OpId, grf: i64, elem: &str) -> ValueId {
        let (mut data, types) = OpData::of(&EuPayload::new(grf, Grf(elem).into())).expect("a typed op");
        data.set("exec", A::from(XE3_SIMD).into());
        cr.rw.set_before(before);
        let op = cr.rw.create(data, &types);
        cr.rw.body[op].results[0]
    }

    fn imm(cr: &mut Cr<'_, '_, '_>, before: OpId, bits: u32, elem: &str) -> ValueId {
        Self::imm64(cr, before, bits.into(), elem)
    }

    fn imm64(cr: &mut Cr<'_, '_, '_>, before: OpId, bits: u64, elem: &str) -> ValueId {
        let ty = match elem {
            "f32" => "f",
            "i32" => "d",
            "f64" => "df",
            "i64" => "q",
            _ => "ud",
        };
        Self::eu(cr, before, "xe3.eu.mov", &[], vec![("src0", A::str(format!("{bits:#x}:{ty}")))], elem)
    }

    fn vect_size(n: usize) -> Option<VectSize> {
        [VectSize::V1, VectSize::V2, VectSize::V3, VectSize::V4].get(n.wrapping_sub(1)).copied()
    }

    fn constant_bits(value: &Attribute<'_>, n: usize) -> Option<Vec<u64>> {
        let c = Const::try_from(value).ok()?;
        let bits = |s: Scalar| match s {
            Scalar::Float(f) => matches!(f.kind, FloatKind::F32 | FloatKind::F64).then_some(f.bits),
            Scalar::Int(i) if i.width == 1 => Some(if i.bits() != 0 { 0xffff_ffff } else { 0 }),
            Scalar::Int(i) => (i.width <= 64).then_some(i.bits() as u64),
            Scalar::Complex(..) | Scalar::ComplexFloat(..) | Scalar::Wide(_) => None,
        };
        (0..n).map(|i| bits(c.at(i))).collect()
    }

    pub fn stage_of(body: &Body<'_>, f: OpId) -> Option<Stage> {
        if body[f].name != "func.func" {
            return None;
        }
        match body[f].get(Self::STAGE)? {
            Attribute::Extended { ext, .. } => match ext.body.as_deref()? {
                "<Vertex>" => Some(Stage::Vertex),
                "<Fragment>" => Some(Stage::Fragment),
                _ => None,
            },
            _ => None,
        }
    }

    /// The attribute of a shader module listing a stage's built-in inputs (comma separated): they are the first
    /// arguments of its entry point
    pub fn builtins_attr(stage: Stage) -> String {
        format!("xe3.builtins.{stage:?}").to_lowercase()
    }

    fn builtins(body: &Body<'_>, f: OpId, stage: Stage) -> Vec<String> {
        let names = body.parent_op(f).and_then(|m| body[m].get_str(&Self::builtins_attr(stage)).map(str::to_owned));
        names.map(|n| n.split(',').filter(|s| !s.is_empty()).map(str::to_owned).collect()).unwrap_or_default()
    }

    /// Where a built-in's components come from: a vertex's indices are written by the vertex fetcher into the
    /// input after the attributes (`after`: its first register), the vertex index's component 0, the instance
    /// index's 1; a fragment's position and facing come from the thread payload (`Self::FRAG_COORD`,
    /// `Self::FRONT_FACING`)
    fn builtin_payload(name: &str, stage: Stage, after: usize) -> Option<Vec<i64>> {
        match (stage, name) {
            (Stage::Vertex, "VertexIndex") => Some(vec![Self::VERTEX_INDEX - after as i64]),
            (Stage::Vertex, "InstanceIndex") => Some(vec![after as i64 + 1]),
            (Stage::Fragment, "FragCoord") => Some((0..4).map(|c| Self::FRAG_COORD - c).collect()),
            (Stage::Fragment, "FrontFacing") => Some(vec![Self::FRONT_FACING]),
            _ => None,
        }
    }

    /// Payload entries that are not registers: gl_FragCoord's x (then y, z, w below it), gl_FrontFacing, and the
    /// vertex index (less the register `r` of the vertex fetcher's VertexID: the sum of it and r + 2, the first
    /// vertex)
    pub const FRAG_COORD: i64 = -1;
    pub const FRONT_FACING: i64 = -5;
    pub const VERTEX_INDEX: i64 = -1000;

    fn interface_inputs(body: &Body<'_>, f: OpId, inputs: &[Type<'_>]) -> usize {
        let Some(Attribute::Array(attrs)) = body[f].get("arg_attrs") else { return inputs.len() };
        attrs.iter().position(|a| matches!(a, Attribute::Dict(d) if d.iter().any(|n| n.name == "spirv.interface_var_abi"))).unwrap_or(inputs.len())
    }

    fn stage(body: &Body<'_>, op: OpId) -> Option<Stage> {
        let mut at = body.parent_op(op);
        while let Some(p) = at {
            if body[p].name == Kernel::NAME {
                return body.head::<Kernel>(p).ok()?.stage;
            }
            if let Some(s) = Self::stage_of(body, p) {
                return Some(s);
            }
            at = body.parent_op(p);
        }
        None
    }

    fn lanes(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], to: &str, attrs: &[(&str, A)]) -> bool {
        let Some(elem) = Self::result_elem(cr, op) else { return false };
        cr.lanes_each(op, o, |cr, x| Self::eu(cr, op, to, x, attrs.to_vec(), elem))
    }

    fn convert(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], to: &str, attrs: &[(&str, A)]) -> bool {
        let Some(re) = Self::result_elem(cr, op) else { return false };
        let Some(oe) = o.first().and_then(|v| v.first()).and_then(|v| Grf::try_from(&cr.rw.body[*v].ty).ok()).map(|g| g.0) else { return false };
        let wide = |e: &str| matches!(e, "f64" | "i64");
        let unsigned = attrs.iter().any(|(k, _)| *k == "unsigned");
        let narrow = |e: &str| match e {
            "f32" => "f",
            _ if unsigned => "ud",
            _ => "d",
        };
        match (wide(re), wide(oe)) {
            (true, false) => {
                let t = narrow(oe);
                cr.lanes_each(op, o, |cr, x| {
                    let spread = Self::eu(cr, op, "xe3.eu.mov", x, vec![("as", A::str("ud")), ("dst_wide", A::str("ud"))], "i64");
                    let mut a = attrs.to_vec();
                    a.push(("wide0", A::str(t)));
                    Self::eu(cr, op, to, &[spread], a, re)
                })
            }
            (false, true) => {
                let t = narrow(re);
                cr.lanes_each(op, o, |cr, x| {
                    let mut a = attrs.to_vec();
                    a.push(("dst_wide", A::str(t)));
                    let spread = Self::eu(cr, op, to, x, a, "i64");
                    Self::eu(cr, op, "xe3.eu.mov", &[spread], vec![("wide0", A::str("ud")), ("as", A::str("ud"))], re)
                })
            }
            _ => Self::lanes(cr, op, o, to, attrs),
        }
    }

    fn rounding(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>], to: &str, attrs: &[(&str, A)]) -> bool {
        if Self::result_elem(cr, op) != Some("f64") {
            return Self::lanes(cr, op, o, to, attrs);
        }
        let kind = match to {
            "xe3.eu.rndd" => Rounding::Down,
            "xe3.eu.rndz" => Rounding::Zero,
            _ => Rounding::Even,
        };
        cr.lanes_each(op, o, |cr, x| Exact64::round(cr, op, x[0], kind))
    }

    fn result_elem(cr: &Cr<'_, '_, '_>, op: OpId) -> Option<&'static str> {
        let r = *cr.rw.body[op].results.first()?;
        Xe3Types::of(&cr.rw.body[r].ty).map(|(_, e)| e)
    }

}

#[patterns(converter)]
#[map("arith.addf" => "xe3.eu.add", via = Self::lanes(&[]))]
#[map("arith.subf" => "xe3.eu.add", via = Self::lanes(&[("neg1", A::Bool(true))]))]
#[map("arith.mulf" => "xe3.eu.mul", via = Self::lanes(&[]))]

#[map("arith.negf" => "xe3.eu.mov", via = Self::lanes(&[("neg0", A::Bool(true))]))]

#[map("math.absf" => "xe3.eu.mov", via = Self::lanes(&[("abs0", A::Bool(true))]))]
#[map("math.floor" => "xe3.eu.rndd", via = Self::rounding(&[]))]
#[map("math.trunc" => "xe3.eu.rndz", via = Self::rounding(&[]))]

#[map("math.roundeven" => "xe3.eu.rnde", via = Self::rounding(&[]))]
#[map("arith.minnumf" => "xe3.eu.sel", via = Self::lanes(&[("cond", A::str("lt"))]))]
#[map("arith.maxnumf" => "xe3.eu.sel", via = Self::lanes(&[("cond", A::str("ge"))]))]
#[map("arith.addi" => "xe3.eu.add", via = Self::lanes(&[]))]
#[map("arith.subi" => "xe3.eu.add", via = Self::lanes(&[("neg1", A::Bool(true))]))]
#[map("arith.shli" => "xe3.eu.shl", via = Self::lanes(&[("unsigned", A::Bool(true))]))]
#[map("arith.shrui" => "xe3.eu.shr", via = Self::lanes(&[("unsigned", A::Bool(true))]))]
#[map("arith.shrsi" => "xe3.eu.asr", via = Self::lanes(&[]))]
#[map("arith.andi" => "xe3.eu.and", via = Self::lanes(&[("as", A::str("ud"))]))]
#[map("arith.ori" => "xe3.eu.or", via = Self::lanes(&[("as", A::str("ud"))]))]
#[map("arith.xori" => "xe3.eu.xor", via = Self::lanes(&[("as", A::str("ud"))]))]
#[map("arith.minsi" => "xe3.eu.sel", via = Self::lanes(&[("cond", A::str("lt"))]))]
#[map("arith.maxsi" => "xe3.eu.sel", via = Self::lanes(&[("cond", A::str("ge"))]))]
#[map("arith.minui" => "xe3.eu.sel", via = Self::lanes(&[("cond", A::str("lt")), ("unsigned", A::Bool(true))]))]
#[map("arith.maxui" => "xe3.eu.sel", via = Self::lanes(&[("cond", A::str("ge")), ("unsigned", A::Bool(true))]))]

#[map("arith.extsi" => "xe3.eu.mov", via = Self::lanes(&[("as", A::str("ud"))]))]
#[map("arith.sitofp" => "xe3.eu.mov", via = Self::convert(&[]))]
#[map("arith.uitofp" => "xe3.eu.mov", via = Self::convert(&[("unsigned", A::Bool(true))]))]
#[map("arith.fptosi" => "xe3.eu.mov", via = Self::convert(&[]))]
#[map("arith.fptoui" => "xe3.eu.mov", via = Self::convert(&[("unsigned", A::Bool(true))]))]

#[map("arith.extf" => "xe3.eu.mov", via = Self::convert(&[]))]
#[map("arith.truncf" => "xe3.eu.mov", via = Self::convert(&[]))]

#[map("arith.bitcast" => "xe3.eu.mov", via = Self::lanes(&[("as", A::str("ud"))]))]
impl GpuToXe3 {
    #[convert("func.func")]
    fn func(cr: &mut Cr<'_, '_, '_>, op: OpId, _: &[Vec<ValueId>]) -> bool {
        let Some(stage) = Self::stage_of(cr.rw.body, op) else { return false };
        let Some(Attribute::Type(Type::Function(fty))) = cr.rw.body[op].get("function_type").cloned() else { return false };
        let Some(name) = cr.rw.body[op].get("sym_name").cloned() else { return false };
        let region = cr.rw.body[op].regions[0];
        let mut sig = SignatureConversion::new(fty.inputs.len());
        let mut payload = Vec::new();
        let all = Self::interface_inputs(cr.rw.body, op, &fty.inputs);
        let builtins = Self::builtins(cr.rw.body, op, stage);
        let Some(inputs) = all.checked_sub(builtins.len()) else { return false };
        let mut system = 0u32;
        for (i, t) in fty.inputs.iter().enumerate() {
            if i >= all {
                sig.add_inputs(i, Vec::new());
                continue;
            }
            let Some((n, elem)) = Xe3Types::of(t) else { return false };
            let base = if stage == Stage::Vertex { XE3_VS_ATTR_GRF as usize } else { 0 };
            match builtins.get(i).map(String::as_str) {
                Some(b) => {
                    let Some(at) = Self::builtin_payload(b, stage, base + 4 * inputs) else { return false };
                    if stage == Stage::Vertex {
                        system |= if b == "VertexIndex" { 1 } else { 2 };
                    }
                    payload.extend(at.into_iter().take(n));
                }
                None => {
                    let j = i - builtins.len();
                    payload.extend((0..n).map(|c| (base + 4 * j + c) as i64));
                }
            }
            sig.add_inputs(i, vec![Type::from(Grf(elem)); n]);
        }
        let varyings = if stage == Stage::Vertex { fty.results.len().saturating_sub(1) } else { 0 };
        if varyings > XE3_MAX_VARYINGS || (stage == Stage::Fragment && inputs > XE3_MAX_VARYINGS) {
            return false;
        }
        let count = |attr: &str| cr.rw.body[op].get(attr).map(Attrs::as_int).transpose().map(|n| n.unwrap_or(0) as u32);
        let (Ok(bindings), Ok(buffers)) = (count(Self::BINDINGS), count(Self::BUFFERS)) else { return false };
        let kernel = Kernel {
            simd: Some(u32::from(XE3_SIMD)),
            payload: Some(payload),
            inputs: Some(inputs as u32),
            varyings: Some(varyings as u32),
            bindings: Some(bindings),
            buffers: Some(buffers),
            system: (system != 0).then_some(system),
            ..Kernel::new(name, Some(stage), None, None, None, None, None, None, None, None, None, mlirformat::Region { blocks: Vec::new() })
        };
        let Ok(data) = OpData::head(&kernel) else { return false };
        cr.rw.set_before(op);
        let new = cr.rw.create_with(data, &[], |b, k| {
            b.add_region(k);
        });
        let into = cr.rw.body[new].regions[0];
        cr.inline_region(region, into, None);
        let Some(current) = cr.current else { return false };
        if cr.convert_region_types(into, current, Some(&sig)).is_none() {
            return false;
        }
        cr.erase(op);
        true
    }

    fn grfs<'a>(t: &Type<'a>) -> Option<Vec<Type<'a>>> {
        Xe3Types::of(t).map(|(n, e)| vec![Type::from(Grf(e)); n])
    }

    fn structured(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        let types: Vec<Type<'_>> = cr.rw.body[op].results.iter().map(|r| cr.rw.body[*r].ty.clone()).collect();
        let Some(results) = types.iter().map(Self::grfs).collect::<Option<Vec<_>>>() else { return false };
        let regions = cr.rw.body[op].regions.to_vec();
        let mut sigs = Vec::new();
        for r in &regions {
            let Some(&entry) = cr.rw.body[*r].blocks.first() else {
                sigs.push(None);
                continue;
            };
            let args: Vec<Type<'_>> = cr.rw.body[entry].args.iter().map(|a| cr.rw.body[*a].ty.clone()).collect();
            let mut sig = SignatureConversion::new(args.len());
            for (i, t) in args.iter().enumerate() {
                let Some(g) = Self::grfs(t) else { return false };
                sig.add_inputs(i, g);
            }
            sigs.push(Some(sig));
        }
        let mut data = OpData::build(&name).operands(&o.concat());
        for a in cr.rw.body[op].attributes.clone() {
            data.set(&a.name, a.value.unwrap_or(Attribute::Unit));
        }
        cr.rw.set_before(op);
        let flat: Vec<Type<'_>> = results.concat();
        let n = regions.len();
        let new = cr.rw.create_with(data, &flat, |b, k| {
            for _ in 0..n {
                b.add_region(k);
            }
        });
        let Some(current) = cr.current else { return false };
        for (i, (r, sig)) in regions.iter().zip(&sigs).enumerate() {
            let into = cr.rw.body[new].regions[i];
            cr.inline_region(*r, into, None);
            if cr.convert_region_types(into, current, sig.as_ref()).is_none() {
                return false;
            }
        }
        let mut values = cr.rw.body[new].results.iter().copied();
        let grouped = results.iter().map(|g| values.by_ref().take(g.len()).collect()).collect();
        cr.replace(op, grouped);
        true
    }

    #[convert("scf.for")]
    fn scf_for(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        Self::structured(cr, op, o)
    }

    #[convert("scf.if")]
    fn scf_if(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        Self::structured(cr, op, o)
    }

    #[convert("scf.while")]
    fn scf_while(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        Self::structured(cr, op, o)
    }

    #[convert("scf.condition")]
    fn scf_condition(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.rw.set_before(op);
        cr.rw.create(OpData::build("scf.condition").operands(&o.concat()), &[]);
        cr.erase(op);
        true
    }

    #[convert("scf.yield")]
    fn scf_yield(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.rw.set_before(op);
        cr.rw.create(OpData::build("scf.yield").operands(&o.concat()), &[]);
        cr.erase(op);
        true
    }

    pub fn structured_legal(body: &Body<'_>, op: OpId) -> bool {
        let grf = |v: &ValueId| Grf::try_from(&body[*v].ty).is_ok();
        let args = body[op].regions.iter().flat_map(|r| body[*r].blocks.iter().flat_map(|b| body[*b].args.iter()));
        body[op].operands.iter().all(grf) && body[op].results.iter().all(grf) && args.into_iter().all(grf)
    }

    #[convert("func.return")]
    fn ret(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(values) = o.first() else { return false };
        let Ok(values): Result<[ValueId; 4], _> = values.clone().try_into() else { return false };
        match Self::stage(cr.rw.body, op) {
            Some(Stage::Fragment) => {
                // a draw's one color target
                if o.len() != 1 {
                    return false;
                }
                let write = Render { bti: 0, msg_ctrl: 0x10, msg_type: 12, coarse_write: false };
                let attrs = vec![
                    ("sfid", A::str(<&str>::from(Sfid::Render))),
                    ("desc", A::from(SendInfo::lengths(write.desc(), 4, 0))),
                    ("src0_len", A::from(4u8)),
                    ("eot", A::Bool(true)),
                ];
                Self::send(cr, op, "xe3.eu.sendc", &values, attrs);
            }
            Some(Stage::Vertex) => {
                let handle = Self::payload(cr, op, 1, "ui32");
                for (k, varying) in o[1..].iter().enumerate() {
                    let Some(vect_size) = Self::vect_size(varying.len()) else { return false };
                    let store = Lsc { vect_size, ..Self::URB };
                    let mut payload = vec![handle];
                    payload.extend(varying);
                    let attrs = vec![
                        ("sfid", A::str(<&str>::from(Sfid::Urb))),
                        ("desc", A::from(SendInfo::lengths(store.desc(), 1, 0))),
                        ("ex_desc", A::from(((32 + 16 * k) as u32) << 12)),
                        ("src0_len", A::from(1u8)),
                        ("src1_len", A::from(varying.len() as u8)),
                    ];
                    Self::send(cr, op, "xe3.eu.send", &payload, attrs);
                }
                let mut payload = vec![handle, Self::imm(cr, op, 0x3c00_3c00, "ui32")];
                payload.extend((0..3).map(|_| Self::imm(cr, op, 0, "ui32")));
                payload.extend(values);
                let urb = Lsc { vect_size: VectSize::V8, ..Self::URB };
                let attrs = vec![
                    ("sfid", A::str(<&str>::from(Sfid::Urb))),
                    ("desc", A::from(SendInfo::lengths(urb.desc(), 1, 0))),
                    ("src0_len", A::from(1u8)),
                    ("src1_len", A::from(8u8)),
                    ("eot", A::Bool(true)),
                ];
                Self::send(cr, op, "xe3.eu.send", &payload, attrs);
            }
            _ => return false,
        }
        cr.erase(op);
        true
    }

    #[convert("spirv.Load")]
    fn resource_load(cr: &mut Cr<'_, '_, '_>, op: OpId, _: &[Vec<ValueId>]) -> bool {
        let r = cr.rw.body[op].results[0];
        let int = |cr: &Cr<'_, '_, '_>, n: &str| cr.rw.body[op].get(n).and_then(|a| Attrs::as_int(a).ok());
        let (binding, offset) = (int(cr, Self::BINDING), int(cr, Self::OFFSET));
        let Some((n, elem)) = Xe3Types::of(&cr.rw.body[r].ty) else {
            cr.replace(op, vec![Vec::new()]);
            return true;
        };
        let (Some(b), Some(offset)) = (binding, offset) else { return false };
        let bti = XE3_BUFFER_BTI + b as u8;
        let (cols, stride) = match (SpirvType::try_from(&cr.rw.body[r].ty), int(cr, Self::STRIDE)) {
            (Ok(SpirvType::Matrix { cols, .. }), Some(stride)) => (cols as usize, stride),
            (Ok(SpirvType::Matrix { .. }), None) => return false,
            _ => (1, 0),
        };
        let rows = n / cols;
        let Some(vect_size) = Self::vect_size(rows).filter(|_| offset % 4 == 0 && stride % 4 == 0) else { return false };
        let lsc = Lsc { op: LscOp::Load, surface: AddrSurface::Bti, addr_size: AddrSize::A32, data_size: DataSize::D32, cache: 0, vect_size, transpose: false, cmask: 0 };
        let mut values = Vec::with_capacity(n);
        for c in 0..cols {
            let addr = Self::imm(cr, op, (offset + c as i64 * stride) as u32, "i32");
            let attrs = vec![
                ("sfid", A::str(<&str>::from(Sfid::Ugm))),
                ("desc", A::from(SendInfo::lengths(lsc.desc(), 1, rows as u32))),
                ("ex_desc", A::from(u32::from(bti) << 24)),
                ("src0_len", A::from(1u8)),
            ];
            values.extend(Self::create(cr, op, "xe3.eu.send", &[addr], attrs, &vec![Type::from(Grf(elem)); rows]));
        }
        cr.replace(op, vec![values]);
        true
    }

    #[convert("spirv.AccessChain", "spirv.SampledImage")]
    fn handle(cr: &mut Cr<'_, '_, '_>, op: OpId, _: &[Vec<ValueId>]) -> bool {
        cr.replace(op, vec![Vec::new()]);
        true
    }

    fn binding(cr: &Cr<'_, '_, '_>, op: OpId) -> Option<u8> {
        cr.rw.body[op].get(Self::BINDING).and_then(|a| Attrs::as_int(a).ok()).and_then(|b| u8::try_from(b).ok())
    }

    #[convert("spirv.ImageSampleImplicitLod")]
    /// A sample at coordinates only: image operands (an offset, a bias) are not taken
    fn sample(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let [_, uv] = o else { return false };
        if !matches!(uv.len(), 2..=4) {
            return false;
        }
        let Some(b) = Self::binding(cr, op) else { return false };
        let msg = Sampler { msg_type: 0, simd_mode: 1, bti: XE3_TEXTURE_BTI + b, sampler: b, return_hp: false };
        let n = uv.len() as u32;
        let attrs = vec![("sfid", A::str(<&str>::from(Sfid::Sampler))), ("desc", A::from(SendInfo::lengths(msg.desc(), n, 4))), ("src0_len", A::from(n as u8))];
        let values = Self::create(cr, op, "xe3.eu.send", uv, attrs, &vec![Type::from(Grf("f32")); 4]);
        cr.replace(op, vec![values]);
        true
    }

    #[convert("arith.constant")]
    fn constant(cr: &mut Cr<'_, '_, '_>, op: OpId, _: &[Vec<ValueId>]) -> bool {
        let ty = cr.rw.body[cr.rw.body[op].results[0]].ty.clone();
        let (Some((n, elem)), Some(value)) = (Xe3Types::of(&ty), cr.rw.body[op].get("value").cloned()) else { return false };
        let Some(bits) = Self::constant_bits(&value, n) else { return false };
        let values = bits.into_iter().map(|b| Self::imm64(cr, op, b, elem)).collect();
        cr.replace(op, vec![values]);
        true
    }

    #[convert("math.fma")]
    fn fma(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(elem) = Self::result_elem(cr, op) else { return false };
        cr.lanes_each(op, o, |cr, x| Self::eu(cr, op, "xe3.eu.mad", &[x[2], x[0], x[1]], vec![], elem))
    }

    #[convert("arith.cmpf")]
    fn cmp(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(p) = cr.rw.body[op].get("predicate").map(Attrs::as_int).and_then(Result::ok) else { return false };
        let cond = match p {
            1 => "z",
            2 => "gt",
            3 => "ge",
            4 => "lt",
            5 => "le",
            13 => "nz",
            _ => return false,
        };
        let f64s = o.first().and_then(|v| v.first()).and_then(|v| Grf::try_from(&cr.rw.body[*v].ty).ok()).is_some_and(|g| g.0 == "f64");
        if !f64s {
            return Self::lanes(cr, op, o, "xe3.eu.cmp", &[("cond", A::str(cond))]);
        }
        cr.lanes_each(op, o, |cr, x| {
            let wide = Self::eu(cr, op, "xe3.eu.cmp", x, vec![("cond", A::str(cond))], "f64");
            Self::eu(cr, op, "xe3.eu.mov", &[wide], vec![("wide0", A::str("ud")), ("as", A::str("ud"))], "f32")
        })
    }

    #[convert("arith.select")]
    fn select(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let [c, x, y] = o else { return false };
        let Some(elem) = Self::result_elem(cr, op) else { return false };
        let c = if c.len() == 1 { vec![c[0]; x.len()] } else { c.clone() };
        cr.lanes_each(op, &[c, x.clone(), y.clone()], |cr, v| {
            if elem == "f32" {
                return Self::eu(cr, op, "xe3.eu.csel", &[v[2], v[1], v[0]], vec![("cond", A::str("z"))], "f32");
            }
            if matches!(elem, "f64" | "i64") {
                Self::create(cr, op, "xe3.eu.cmp", &[v[0]], vec![("cond", A::str("nz")), ("dst", A::str("null<1;1,0>:d")), ("src1", A::str("0x0:d")), ("flag", A::str("f0.0")), ("as", A::str("ud"))], &[]);
                Self::create(cr, op, "xe3.eu.sync", &[], vec![("exec", A::from(1u8)), ("fn", A::str("nop")), ("swsb", A::str("A@1"))], &[]);
                return Self::eu(cr, op, "xe3.eu.sel", &[v[1], v[2]], vec![("pred", A::str("normal")), ("flag", A::str("f0.0"))], elem);
            }
            let bits = || vec![("as", A::str("ud"))];
            let t = Self::eu(cr, op, "xe3.eu.and", &[v[1], v[0]], bits(), "i32");
            let not = Self::eu(cr, op, "xe3.eu.not", &[v[0]], bits(), "i32");
            let f = Self::eu(cr, op, "xe3.eu.and", &[v[2], not], bits(), "i32");
            Self::eu(cr, op, "xe3.eu.or", &[t, f], bits(), "i32")
        })
    }

    #[convert("arith.muli")]
    fn muli(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.lanes_each(op, o, |cr, x| {
            let lo = Self::eu(cr, op, "xe3.eu.mul", &[x[0], x[1]], vec![("half1", A::str("lo"))], "i32");
            let hi = Self::eu(cr, op, "xe3.eu.mul", &[x[0], x[1]], vec![("half1", A::str("hi"))], "i32");
            let sixteen = Self::imm(cr, op, 16, "i32");
            let up = Self::eu(cr, op, "xe3.eu.shl", &[hi, sixteen], vec![("unsigned", A::Bool(true))], "i32");
            Self::eu(cr, op, "xe3.eu.add", &[lo, up], vec![], "i32")
        })
    }

    #[convert("arith.cmpi")]
    fn cmpi(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(p) = cr.rw.body[op].get("predicate").map(Attrs::as_int).and_then(Result::ok) else { return false };
        let Some(cond) = ["z", "nz", "lt", "le", "gt", "ge", "lt", "le", "gt", "ge"].get(p as usize) else { return false };
        let mut attrs = vec![("cond", A::str(*cond)), ("dst_as", A::str(if p >= 6 { "ud" } else { "d" }))];
        if p >= 6 {
            attrs.push(("unsigned", A::Bool(true)));
        }
        let masks = o.first().and_then(|v| v.first()).and_then(|v| Grf::try_from(&cr.rw.body[*v].ty).ok()).is_some_and(|g| g.0 == "f32");
        if masks {
            attrs.push(("as", A::str("ud")));
        }
        Self::lanes(cr, op, o, "xe3.eu.cmp", &attrs)
    }

    #[convert("arith.extui")]
    fn extui(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.lanes_each(op, o, |cr, x| {
            let one = Self::imm(cr, op, 1, "i32");
            Self::eu(cr, op, "xe3.eu.and", &[x[0], one], vec![("as", A::str("ud"))], "i32")
        })
    }

    #[convert("arith.index_cast", "arith.index_castui")]
    fn same(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.replace(op, vec![o[0].clone()]);
        true
    }

    #[convert("math.ceil")]
    fn ceil(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(elem) = Self::result_elem(cr, op) else { return false };
        if elem == "f64" {
            return cr.lanes_each(op, o, |cr, x| Exact64::round(cr, op, x[0], Rounding::Up));
        }
        cr.lanes_each(op, o, |cr, x| {
            let down = Self::eu(cr, op, "xe3.eu.rndd", x, vec![("neg0", A::Bool(true))], elem);
            Self::eu(cr, op, "xe3.eu.mov", &[down], vec![("neg0", A::Bool(true))], elem)
        })
    }

    #[convert("arith.divf")]
    fn div(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        if Self::result_elem(cr, op) == Some("f64") {
            return cr.lanes_each(op, o, |cr, x| Exact64::div(cr, op, x[0], x[1]));
        }
        cr.lanes_each(op, o, |cr, x| Exact::div(cr, op, x[0], x[1]))
    }

    #[convert("math.sqrt")]
    fn sqrt(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        if Self::result_elem(cr, op) == Some("f64") {
            return cr.lanes_each(op, o, |cr, x| Exact64::sqrt(cr, op, x[0]));
        }
        cr.lanes_each(op, o, |cr, x| Exact::sqrt(cr, op, x[0]))
    }

    #[convert("vector.extract")]
    fn extract(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let r = cr.rw.body[op].results[0];
        match Xe3Types::lanes(&cr.rw.body[r].ty) {
            Some(1) => cr.lanes_extract(op, o),
            Some(k) => {
                let Some(column) = cr.lane_position(op).and_then(|at| o[0].get(at * k..at * k + k)).map(<[ValueId]>::to_vec) else { return false };
                cr.replace(op, vec![column]);
                true
            }
            None => false,
        }
    }

    /// An undefined value: any bits do, zero is as good as any.
    #[convert("ub.poison", "spirv.Undef")]
    fn undef(cr: &mut Cr<'_, '_, '_>, op: OpId, _: &[Vec<ValueId>]) -> bool {
        let ty = cr.rw.body[cr.rw.body[op].results[0]].ty.clone();
        let Some((n, elem)) = Xe3Types::of(&ty) else { return false };
        let values = (0..n).map(|_| Self::imm(cr, op, 0, elem)).collect();
        cr.replace(op, vec![values]);
        true
    }

    /// Each result lane a sum of products: a mul, then a mad for each further term
    #[convert("spirv.VectorTimesScalar", "spirv.MatrixTimesScalar", "spirv.MatrixTimesVector", "spirv.VectorTimesMatrix", "spirv.MatrixTimesMatrix", "spirv.OuterProduct")]
    fn products(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let ([a, b], Some(elem)) = (o, Self::result_elem(cr, op)) else { return false };
        let body = &*cr.rw.body;
        let Some(terms) = Matrices::products(&body[op].name, &body[body[op].operands[0]].ty, &body[body[op].operands[1]].ty) else { return false };
        let values = terms
            .into_iter()
            .map(|t| {
                let first = Self::eu(cr, op, "xe3.eu.mul", &[a[t[0].0], b[t[0].1]], vec![], elem);
                t[1..].iter().fold(first, |acc, &(x, y)| Self::eu(cr, op, "xe3.eu.mad", &[acc, a[x], b[y]], vec![], elem))
            })
            .collect();
        cr.replace(op, vec![values]);
        true
    }

    #[convert("spirv.Transpose")]
    fn transpose(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let Some(order) = Matrices::transpose(&cr.rw.body[cr.rw.body[op].operands[0]].ty) else { return false };
        let values = order.into_iter().map(|i| o[0][i]).collect();
        cr.replace(op, vec![values]);
        true
    }

    #[convert("vector.insert")]
    fn insert(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.lanes_insert(op, o)
    }

    #[convert("vector.from_elements")]
    fn from_elements(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.lanes_from_elements(op, o)
    }

    #[convert("vector.broadcast")]
    fn broadcast(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.lanes_broadcast(op, o, Xe3Types::lanes)
    }

    #[convert("vector.shuffle")]
    fn shuffle(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        cr.lanes_shuffle(op, o)
    }

    #[convert("spirv.GL.Exp", "spirv.GL.Exp2", "spirv.GL.Log", "spirv.GL.Log2", "spirv.GL.InverseSqrt", "spirv.GL.Sin", "spirv.GL.Cos", "spirv.GL.Tan", "spirv.GL.Atan", "spirv.GL.Fract", "spirv.GL.Trunc", "spirv.GL.Radians", "spirv.GL.Degrees", "spirv.GL.FSign")]
    fn std450_unary(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        Std450::unary(cr, op, o, &name)
    }

    #[convert("spirv.GL.Pow", "spirv.GL.Atan2", "spirv.GL.Step", "spirv.FMod")]
    fn std450_binary(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        Std450::binary(cr, op, o, &name)
    }

    #[convert("spirv.GL.FClamp", "spirv.GL.FMix", "spirv.GL.SmoothStep")]
    fn std450_ternary(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        Std450::ternary(cr, op, o, &name)
    }

    #[convert("spirv.Dot", "spirv.GL.Length", "spirv.GL.Distance")]
    fn std450_reduce(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        Std450::reduce(cr, op, o, &name)
    }

    #[convert("spirv.GL.Normalize", "spirv.GL.Cross", "spirv.GL.Reflect")]
    fn std450_vector(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        Std450::vector(cr, op, o, &name)
    }

    #[convert("spirv.UDiv", "spirv.UMod", "spirv.SDiv", "spirv.SRem", "spirv.SMod")]
    fn std450_divide(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        Std450::divide(cr, op, o, &name)
    }

    #[convert("spirv.SNegate", "spirv.GL.SAbs", "spirv.GL.SSign")]
    fn std450_integer(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let name = cr.rw.body[op].name.to_string();
        Std450::integer(cr, op, o, &name)
    }

    #[convert("spirv.GL.SClamp", "spirv.GL.UClamp")]
    fn std450_clamp(cr: &mut Cr<'_, '_, '_>, op: OpId, o: &[Vec<ValueId>]) -> bool {
        let unsigned = &*cr.rw.body[op].name == "spirv.GL.UClamp";
        Std450::clamp(cr, op, o, unsigned)
    }

}

struct Exact;

impl Exact {
    fn op(cr: &mut Cr<'_, '_, '_>, at: OpId, name: &str, x: &[ValueId], attrs: Vec<(&str, A)>) -> ValueId {
        GpuToXe3::eu(cr, at, name, x, attrs, "f32")
    }

    fn mad(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId, c: ValueId, neg: bool) -> ValueId {
        Self::op(cr, at, "xe3.eu.mad", &[a, b, c], if neg { vec![("neg1", A::Bool(true))] } else { vec![] })
    }

    fn math(cr: &mut Cr<'_, '_, '_>, at: OpId, f: &str, x: ValueId) -> ValueId {
        Self::op(cr, at, "xe3.eu.math", &[x], vec![("fn", A::str(f))])
    }

    fn finite_or(cr: &mut Cr<'_, '_, '_>, at: OpId, refined: ValueId, fallback: ValueId) -> ValueId {
        let d = Self::op(cr, at, "xe3.eu.add", &[refined, refined], vec![("neg1", A::Bool(true))]);
        Self::op(cr, at, "xe3.eu.csel", &[refined, fallback, d], vec![("cond", A::str("z"))])
    }

    fn imm(cr: &mut Cr<'_, '_, '_>, at: OpId, v: f32) -> ValueId {
        GpuToXe3::imm(cr, at, v.to_bits(), "f32")
    }

    fn below(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId, bound: f32, then: ValueId, other: ValueId) -> ValueId {
        let minus = Self::imm(cr, at, -bound);
        let d = Self::op(cr, at, "xe3.eu.add", &[x, minus], vec![("abs0", A::Bool(true))]);
        Self::op(cr, at, "xe3.eu.csel", &[then, other, d], vec![("cond", A::str("lt"))])
    }

    fn div(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        let one = Self::imm(cr, at, 1.0);
        let up = Self::imm(cr, at, pow2f(64));
        let small = Self::below(cr, at, a, pow2f(-100), up, one);
        let s = Self::below(cr, at, b, pow2f(-100), up, small);
        let a = Self::op(cr, at, "xe3.eu.mul", &[a, s], vec![]);
        let b = Self::op(cr, at, "xe3.eu.mul", &[b, s], vec![]);
        let y0 = Self::math(cr, at, "inv", b);
        let e = Self::mad(cr, at, one, b, y0, true);
        let y1 = Self::mad(cr, at, y0, e, y0, false);
        let q0 = Self::op(cr, at, "xe3.eu.mul", &[a, y1], vec![]);
        let r = Self::mad(cr, at, a, b, q0, true);
        let q1 = Self::mad(cr, at, q0, r, y1, false);
        let rough = Self::op(cr, at, "xe3.eu.mul", &[a, y0], vec![]);
        let q = Self::finite_or(cr, at, q1, rough);
        Self::op(cr, at, "xe3.eu.csel", &[rough, q, a], vec![("cond", A::str("z"))])
    }

    fn sqrt(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId) -> ValueId {
        let one = Self::imm(cr, at, 1.0);
        let (up, down) = (Self::imm(cr, at, pow2f(64)), Self::imm(cr, at, pow2f(-32)));
        let s = Self::below(cr, at, a, pow2f(-100), up, one);
        let back = Self::below(cr, at, a, pow2f(-100), down, one);
        let a = Self::op(cr, at, "xe3.eu.mul", &[a, s], vec![]);
        let half = Self::imm(cr, at, 0.5);
        let y0 = Self::math(cr, at, "rsq", a);
        let g0 = Self::op(cr, at, "xe3.eu.mul", &[a, y0], vec![]);
        let h0 = Self::op(cr, at, "xe3.eu.mul", &[half, y0], vec![]);
        let d = Self::mad(cr, at, half, g0, h0, true);
        let g1 = Self::mad(cr, at, g0, g0, d, false);
        let h1 = Self::mad(cr, at, h0, h0, d, false);
        let r = Self::mad(cr, at, a, g1, g1, true);
        let g2 = Self::mad(cr, at, g1, r, h1, false);
        let rough = Self::math(cr, at, "sqrt", a);
        let root = Self::finite_or(cr, at, g2, rough);
        Self::op(cr, at, "xe3.eu.mul", &[root, back], vec![])
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounding {
    Down,
    Up,
    Zero,
    Even,
}

struct Exact64;

impl Exact64 {
    fn f(cr: &mut Cr<'_, '_, '_>, at: OpId, name: &str, x: &[ValueId], attrs: Vec<(&str, A)>) -> ValueId {
        GpuToXe3::eu(cr, at, name, x, attrs, "f64")
    }

    fn i(cr: &mut Cr<'_, '_, '_>, at: OpId, name: &str, x: &[ValueId], attrs: Vec<(&str, A)>) -> ValueId {
        GpuToXe3::eu(cr, at, name, x, attrs, "i32")
    }

    fn fimm(cr: &mut Cr<'_, '_, '_>, at: OpId, v: f64) -> ValueId {
        GpuToXe3::imm64(cr, at, v.to_bits(), "f64")
    }

    fn iimm(cr: &mut Cr<'_, '_, '_>, at: OpId, v: u32) -> ValueId {
        GpuToXe3::imm(cr, at, v, "i32")
    }

    fn mad(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId, c: ValueId, neg: bool) -> ValueId {
        Self::f(cr, at, "xe3.eu.mad", &[a, b, c], if neg { vec![("neg1", A::Bool(true))] } else { vec![] })
    }

    fn mul(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        Self::f(cr, at, "xe3.eu.mul", &[a, b], vec![])
    }

    fn hi(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        Self::i(cr, at, "xe3.eu.mov", &[x], vec![("hi0", A::Bool(true)), ("as", A::str("ud"))])
    }

    fn lo(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        Self::i(cr, at, "xe3.eu.mov", &[x], vec![("wide0", A::str("ud")), ("as", A::str("ud"))])
    }

    fn pack(cr: &mut Cr<'_, '_, '_>, at: OpId, lo: ValueId, hi: ValueId) -> ValueId {
        GpuToXe3::create(cr, at, crate::XE3_PACK64, &[lo, hi], vec![], &[Type::from(Grf("f64"))])[0]
    }

    fn pow2(cr: &mut Cr<'_, '_, '_>, at: OpId, e: ValueId) -> ValueId {
        let twenty = Self::iimm(cr, at, 20);
        let hi = Self::i(cr, at, "xe3.eu.shl", &[e, twenty], vec![("unsigned", A::Bool(true))]);
        let zero = Self::iimm(cr, at, 0);
        Self::pack(cr, at, zero, hi)
    }

    fn and(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        Self::i(cr, at, "xe3.eu.and", &[a, b], vec![("as", A::str("ud"))])
    }

    fn andn(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        let not = Self::i(cr, at, "xe3.eu.not", &[b], vec![("as", A::str("ud"))]);
        Self::and(cr, at, a, not)
    }

    fn or(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        Self::i(cr, at, "xe3.eu.or", &[a, b], vec![("as", A::str("ud"))])
    }

    fn add(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId, neg: bool) -> ValueId {
        Self::i(cr, at, "xe3.eu.add", &[a, b], if neg { vec![("neg1", A::Bool(true))] } else { vec![] })
    }

    fn exponent(cr: &mut Cr<'_, '_, '_>, at: OpId, hi: ValueId) -> ValueId {
        let twenty = Self::iimm(cr, at, 20);
        let e = Self::i(cr, at, "xe3.eu.shr", &[hi, twenty], vec![("unsigned", A::Bool(true))]);
        let m = Self::iimm(cr, at, 0x7ff);
        Self::and(cr, at, e, m)
    }

    fn cmpi(cr: &mut Cr<'_, '_, '_>, at: OpId, cond: &str, a: ValueId, b: ValueId) -> ValueId {
        GpuToXe3::eu(cr, at, "xe3.eu.cmp", &[a, b], vec![("cond", A::str(cond)), ("dst_as", A::str("d"))], "f32")
    }

    fn cmpf(cr: &mut Cr<'_, '_, '_>, at: OpId, cond: &str, a: ValueId, b: ValueId) -> ValueId {
        let wide = Self::f(cr, at, "xe3.eu.cmp", &[a, b], vec![("cond", A::str(cond))]);
        GpuToXe3::eu(cr, at, "xe3.eu.mov", &[wide], vec![("wide0", A::str("ud")), ("as", A::str("ud"))], "f32")
    }

    fn sel(cr: &mut Cr<'_, '_, '_>, at: OpId, m: ValueId, x: ValueId, y: ValueId) -> ValueId {
        GpuToXe3::create(cr, at, "xe3.eu.cmp", &[m], vec![("cond", A::str("nz")), ("dst", A::str("null<1;1,0>:d")), ("src1", A::str("0x0:d")), ("flag", A::str("f0.0")), ("as", A::str("ud"))], &[]);
        GpuToXe3::create(cr, at, "xe3.eu.sync", &[], vec![("exec", A::from(1u8)), ("fn", A::str("nop")), ("swsb", A::str("A@1"))], &[]);
        Self::f(cr, at, "xe3.eu.sel", &[x, y], vec![("pred", A::str("normal")), ("flag", A::str("f0.0"))])
    }

    fn to_f32(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        let spread = GpuToXe3::eu(cr, at, "xe3.eu.mov", &[x], vec![("dst_wide", A::str("f"))], "i64");
        GpuToXe3::eu(cr, at, "xe3.eu.mov", &[spread], vec![("wide0", A::str("ud")), ("as", A::str("ud"))], "f32")
    }

    fn to_f64(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        let spread = GpuToXe3::eu(cr, at, "xe3.eu.mov", &[x], vec![("as", A::str("ud")), ("dst_wide", A::str("ud"))], "i64");
        Self::f(cr, at, "xe3.eu.mov", &[spread], vec![("wide0", A::str("f"))])
    }

    fn split(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> (ValueId, ValueId) {
        let e0 = {
            let h = Self::hi(cr, at, x);
            Self::exponent(cr, at, h)
        };
        let zero = Self::iimm(cr, at, 0);
        let sub = Self::cmpi(cr, at, "z", e0, zero);
        let up = Self::fimm(cr, at, pow2d(54));
        let scaled = Self::mul(cr, at, x, up);
        let x2 = Self::sel(cr, at, sub, scaled, x);
        let h = Self::hi(cr, at, x2);
        let e = Self::exponent(cr, at, h);
        let bias = Self::iimm(cr, at, 1023);
        let e = Self::add(cr, at, e, bias, true);
        let fifty4 = Self::iimm(cr, at, 54);
        let adj = Self::and(cr, at, fifty4, sub);
        let e = Self::add(cr, at, e, adj, true);
        let keep = Self::iimm(cr, at, 0x800f_ffff);
        let one = Self::iimm(cr, at, 0x3ff0_0000);
        let m = Self::and(cr, at, h, keep);
        let m = Self::or(cr, at, m, one);
        let l = Self::lo(cr, at, x2);
        (Self::pack(cr, at, l, m), e)
    }

    fn pow2k(cr: &mut Cr<'_, '_, '_>, at: OpId, k: ValueId) -> ValueId {
        let bias = Self::iimm(cr, at, 1023);
        let e = Self::add(cr, at, k, bias, false);
        Self::pow2(cr, at, e)
    }

    fn div(cr: &mut Cr<'_, '_, '_>, at: OpId, a: ValueId, b: ValueId) -> ValueId {
        let inf = Self::fimm(cr, at, f64::INFINITY);
        let zero = Self::fimm(cr, at, 0.0);
        let general = {
            let aa = Self::f(cr, at, "xe3.eu.mov", &[a], vec![("abs0", A::Bool(true))]);
            let ab = Self::f(cr, at, "xe3.eu.mov", &[b], vec![("abs0", A::Bool(true))]);
            let fa = Self::cmpf(cr, at, "lt", aa, inf);
            let fb = Self::cmpf(cr, at, "lt", ab, inf);
            let na = Self::cmpf(cr, at, "nz", a, zero);
            let nb = Self::cmpf(cr, at, "nz", b, zero);
            let m = Self::and(cr, at, fa, fb);
            let m = Self::and(cr, at, m, na);
            Self::and(cr, at, m, nb)
        };
        let special = {
            let hb = Self::hi(cr, at, b);
            let sign = Self::iimm(cr, at, 0x8000_0000);
            let s = Self::and(cr, at, hb, sign);
            let one = Self::iimm(cr, at, 0x3ff0_0000);
            let shi = Self::or(cr, at, s, one);
            let z = Self::iimm(cr, at, 0);
            let unit = Self::pack(cr, at, z, shi);
            let ab = Self::f(cr, at, "xe3.eu.mov", &[b], vec![("abs0", A::Bool(true))]);
            let fb = Self::cmpf(cr, at, "lt", ab, inf);
            let nb = Self::cmpf(cr, at, "nz", b, zero);
            let finite = Self::and(cr, at, fb, nb);
            let bb = Self::sel(cr, at, finite, unit, b);
            let b32 = Self::to_f32(cr, at, bb);
            let r32 = GpuToXe3::eu(cr, at, "xe3.eu.math", &[b32], vec![("fn", A::str("inv"))], "f32");
            let r = Self::to_f64(cr, at, r32);
            Self::mul(cr, at, a, r)
        };
        let (am, ea) = Self::split(cr, at, a);
        let (bm, eb) = Self::split(cr, at, b);
        let one = Self::fimm(cr, at, 1.0);
        let b32 = Self::to_f32(cr, at, bm);
        let y32 = GpuToXe3::eu(cr, at, "xe3.eu.math", &[b32], vec![("fn", A::str("inv"))], "f32");
        let mut y = Self::to_f64(cr, at, y32);
        for _ in 0..3 {
            let e = Self::mad(cr, at, one, bm, y, true);
            y = Self::mad(cr, at, y, y, e, false);
        }
        let q0 = Self::mul(cr, at, am, y);
        let r = Self::mad(cr, at, am, bm, q0, true);
        let q1 = Self::mad(cr, at, q0, r, y, false);
        let r1 = Self::mad(cr, at, am, bm, q1, true);
        let k = Self::add(cr, at, ea, eb, true);
        let hq = Self::hi(cr, at, q1);
        let eq = Self::exponent(cr, at, hq);
        let big = Self::add(cr, at, eq, k, false);
        let normal = {
            let twenty = Self::iimm(cr, at, 20);
            let ks = Self::i(cr, at, "xe3.eu.shl", &[k, twenty], vec![]);
            let h = Self::add(cr, at, hq, ks, false);
            let l = Self::lo(cr, at, q1);
            Self::pack(cr, at, l, h)
        };
        let over = {
            let sign = Self::iimm(cr, at, 0x8000_0000);
            let s = Self::and(cr, at, hq, sign);
            let e = Self::iimm(cr, at, 0x7ff0_0000);
            let h = Self::or(cr, at, s, e);
            let z = Self::iimm(cr, at, 0);
            Self::pack(cr, at, z, h)
        };
        let under = {
            let lowest = Self::iimm(cr, at, (-1300i32) as u32);
            let kc = Self::i(cr, at, "xe3.eu.sel", &[k, lowest], vec![("cond", A::str("ge"))]);
            let six = Self::iimm(cr, at, 600);
            let k6 = Self::add(cr, at, kc, six, false);
            let p = Self::pow2k(cr, at, k6);
            let m6 = Self::iimm(cr, at, (-600i32) as u32);
            let p6 = Self::pow2k(cr, at, m6);
            let u = Self::mul(cr, at, q1, p);
            let res = Self::mul(cr, at, u, p6);
            let back = {
                let p600 = Self::iimm(cr, at, 600);
                let up = Self::pow2k(cr, at, p600);
                let r600 = Self::mul(cr, at, res, up);
                let zero32 = Self::iimm(cr, at, 0);
                let nk6 = Self::add(cr, at, zero32, k6, true);
                let down = Self::pow2k(cr, at, nk6);
                Self::mul(cr, at, r600, down)
            };
            let d = Self::f(cr, at, "xe3.eu.add", &[q1, back], vec![("neg1", A::Bool(true))]);
            let ad = Self::f(cr, at, "xe3.eu.mov", &[d], vec![("abs0", A::Bool(true))]);
            let h = {
                let c = Self::iimm(cr, at, (-1075i32) as u32);
                let e = Self::add(cr, at, c, kc, true);
                Self::pow2k(cr, at, e)
            };
            let tie = Self::cmpf(cr, at, "z", ad, h);
            let exact = Self::cmpf(cr, at, "z", r1, zero);
            let fix = Self::andn(cr, at, tie, exact);
            let dir = Self::mul(cr, at, r1, y);
            let pos = Self::cmpf(cr, at, "gt", dir, zero);
            let hn = Self::f(cr, at, "xe3.eu.mov", &[h], vec![("neg0", A::Bool(true))]);
            let step = Self::sel(cr, at, pos, h, hn);
            let q1c = Self::f(cr, at, "xe3.eu.add", &[q1, step], vec![]);
            let uc = Self::mul(cr, at, q1c, p);
            let resc = Self::mul(cr, at, uc, p6);
            Self::sel(cr, at, fix, resc, res)
        };
        let lo1 = Self::iimm(cr, at, 1);
        let hi2046 = Self::iimm(cr, at, 2046);
        let is_over = Self::cmpi(cr, at, "gt", big, hi2046);
        let is_under = Self::cmpi(cr, at, "lt", big, lo1);
        let q = Self::sel(cr, at, is_over, over, normal);
        let q = Self::sel(cr, at, is_under, under, q);
        Self::sel(cr, at, general, q, special)
    }

    fn sqrt(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId) -> ValueId {
        let inf = Self::fimm(cr, at, f64::INFINITY);
        let zero = Self::fimm(cr, at, 0.0);
        let general = {
            let pos = Self::cmpf(cr, at, "gt", x, zero);
            let fin = Self::cmpf(cr, at, "lt", x, inf);
            Self::and(cr, at, pos, fin)
        };
        let special = {
            let neg = Self::cmpf(cr, at, "lt", x, zero);
            let nan = Self::fimm(cr, at, f64::NAN);
            Self::sel(cr, at, neg, nan, x)
        };
        let (m, e) = Self::split(cr, at, x);
        let one = Self::iimm(cr, at, 1);
        let odd = Self::and(cr, at, e, one);
        let z = Self::iimm(cr, at, 0);
        let is_odd = Self::cmpi(cr, at, "nz", odd, z);
        let two = Self::fimm(cr, at, 2.0);
        let m2 = Self::mul(cr, at, m, two);
        let m = Self::sel(cr, at, is_odd, m2, m);
        let e = Self::add(cr, at, e, odd, true);
        let half_e = Self::i(cr, at, "xe3.eu.asr", &[e, one], vec![]);
        let m32 = Self::to_f32(cr, at, m);
        let y32 = GpuToXe3::eu(cr, at, "xe3.eu.math", &[m32], vec![("fn", A::str("rsq"))], "f32");
        let y = Self::to_f64(cr, at, y32);
        let half = Self::fimm(cr, at, 0.5);
        let mut g = Self::mul(cr, at, m, y);
        let mut h = Self::mul(cr, at, half, y);
        for _ in 0..3 {
            let r = Self::mad(cr, at, half, g, h, true);
            g = Self::mad(cr, at, g, g, r, false);
            h = Self::mad(cr, at, h, h, r, false);
        }
        let d = Self::mad(cr, at, m, g, g, true);
        let s = Self::mad(cr, at, g, d, h, false);
        let p = Self::pow2k(cr, at, half_e);
        let root = Self::mul(cr, at, s, p);
        Self::sel(cr, at, general, root, special)
    }

    fn round(cr: &mut Cr<'_, '_, '_>, at: OpId, x: ValueId, kind: Rounding) -> ValueId {
        let sign = Self::iimm(cr, at, 0x8000_0000);
        let hx = Self::hi(cr, at, x);
        let sx = Self::and(cr, at, hx, sign);
        let signed = |cr: &mut Cr<'_, '_, '_>, v: ValueId| {
            let h = Self::hi(cr, at, v);
            let m = Self::andn(cr, at, h, sign);
            let h = Self::or(cr, at, m, sx);
            let l = Self::lo(cr, at, v);
            Self::pack(cr, at, l, h)
        };
        let big = Self::fimm(cr, at, pow2d(52));
        let s = signed(cr, big);
        let up = Self::f(cr, at, "xe3.eu.add", &[x, s], vec![]);
        let even = Self::f(cr, at, "xe3.eu.add", &[up, s], vec![("neg1", A::Bool(true))]);
        let one = Self::fimm(cr, at, 1.0);
        let r = match kind {
            Rounding::Even => even,
            Rounding::Down => {
                let past = Self::cmpf(cr, at, "gt", even, x);
                let back = Self::f(cr, at, "xe3.eu.add", &[even, one], vec![("neg1", A::Bool(true))]);
                Self::sel(cr, at, past, back, even)
            }
            Rounding::Up => {
                let past = Self::cmpf(cr, at, "lt", even, x);
                let back = Self::f(cr, at, "xe3.eu.add", &[even, one], vec![]);
                Self::sel(cr, at, past, back, even)
            }
            Rounding::Zero => {
                let ae = Self::f(cr, at, "xe3.eu.mov", &[even], vec![("abs0", A::Bool(true))]);
                let ax = Self::f(cr, at, "xe3.eu.mov", &[x], vec![("abs0", A::Bool(true))]);
                let past = Self::cmpf(cr, at, "gt", ae, ax);
                let unit = signed(cr, one);
                let back = Self::f(cr, at, "xe3.eu.add", &[even, unit], vec![("neg1", A::Bool(true))]);
                Self::sel(cr, at, past, back, even)
            }
        };
        let r = signed(cr, r);
        let ax = Self::f(cr, at, "xe3.eu.mov", &[x], vec![("abs0", A::Bool(true))]);
        let small = Self::cmpf(cr, at, "lt", ax, big);
        Self::sel(cr, at, small, r, x)
    }
}

const fn pow2f(e: i32) -> f32 {
    f32::from_bits(((127 + e) as u32) << 23)
}

const fn pow2d(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}
