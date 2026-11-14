use mlirformat::llvm_types::LlvmType;
use mlirformat::{Attribute, Dim, Error, Op, Shaped, ShapedKind, Type};
use mlirpass::{A, Body, Heads, OpData, OpId, Quiet, Rewriter, ValueId, pass};

use crate::base::{Atomic, AtomicOp, Attrs, Barrier, Lane, Load, Stage, Store, WorkgroupId};

pub const SHADERS: &str = "shaders";

const LANES: i64 = 16;

const GLOBAL: u32 = 1;
const SHARED: u32 = 3;

const MONOTONIC: i64 = 2;

#[pass("convert-detour-to-gpu", on = "builtin.module")]
pub fn convert_detour_to_gpu<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let top = body.top_block(op);
    let shaders: Vec<OpId> = body.ops(top).filter(|f| body[*f].name == "func.func" && Stage::of(body, *f) == Some(Stage::Compute)).collect();
    let Some(&first) = shaders.first() else { return Ok(()) };
    let mut quiet = Quiet;
    let mut rw = Rewriter::new(body, &mut quiet);
    rw.set_before(first);
    let module = rw.create_with(OpData::build("gpu.module").with("sym_name", A::str(SHADERS).into()), &[], |b, m| {
        let r = b.add_region(m);
        b.add_block(r, &[]);
    });
    let block = rw.body.top_block(module);
    for f in shaders {
        rw.move_before(f, block, None);
        ToGpu::kernel(&mut rw, f)?;
        let mut ops = Vec::new();
        rw.body.walk(f, &mut |o| ops.push(o));
        for o in ops {
            ToGpu::op(&mut rw, o)?;
        }
    }
    rw.modify(op, |b| b[op].set("gpu.container_module", Attribute::Unit));
    Ok(())
}

struct ToGpu;

impl ToGpu {
    fn kernel(rw: &mut Rewriter<'_, '_>, f: OpId) -> Result<(), Error> {
        let count = |rw: &Rewriter<'_, '_>, attr: &str| rw.body[f].get(attr).map(Attrs::as_int).transpose().map(|n| n.unwrap_or(0));
        let (threads, shared) = (count(rw, "detour.workgroup")?, count(rw, "detour.shared")?);
        let inputs = match rw.body[f].get("function_type") {
            Some(Attribute::Type(Type::Function(f))) => f.inputs.len(),
            _ => return Err(Error("a shader without a function type".into())),
        };
        if shared > 0 {
            let ty = Type::Shaped(Shaped { kind: ShapedKind::MemRef, shape: Some(vec![Dim::Static(shared)]), elem: Type::int(8).into(), extras: vec![Attribute::dialect("gpu.address_space", "<workgroup>")] }.into());
            let entry = rw.body[rw.body[f].regions[0]].blocks[0];
            rw.body.insert_arg(entry, inputs, ty, None);
        }
        rw.modify(f, |b| {
            b[f].name = "gpu.func".into();
            for a in [Stage::ATTR, "detour.workgroup", "detour.shared"] {
                b[f].remove(a);
            }
            b[f].set("gpu.kernel", Attribute::Unit);
            if threads > 1 {
                b[f].set("known_block_size", Attribute::dense_i32(&[(LANES * threads) as i32, 1, 1]));
            }
            if shared > 0 {
                b[f].set("workgroup_attributions", Attribute::i64(1));
            }
        });
        Ok(())
    }

    fn op(rw: &mut Rewriter<'_, '_>, o: OpId) -> Result<(), Error> {
        match &*rw.body[o].name {
            "func.return" => rw.modify(o, |b| b[o].name = "gpu.return".into()),
            n if n == WorkgroupId::NAME => {
                let dim = rw.body.head::<WorkgroupId>(o)?.dim.unwrap_or(0);
                let case = ["x", "y", "z"].get(dim as usize).ok_or_else(|| Error(format!("detour.workgroup_id dim {dim}")))?;
                Self::id(rw, o, "gpu.block_id", case);
            }
            n if n == Lane::NAME => Self::id(rw, o, "gpu.thread_id", "x"),
            n if n == Load::NAME => {
                let load = rw.body.head::<Load>(o)?;
                Self::unbound(load.binding)?;
                rw.set_before(o);
                let ptr = Self::pointer(rw, rw.body[o].operands[0], load.shared);
                let ty = rw.body[rw.body[o].results[0]].ty.clone();
                let v = match Self::narrow(load.size)? {
                    Some(n) => {
                        let l = rw.create(OpData::build("llvm.load").operands(&[ptr]).with("alignment", Self::align(n.1)), &[Type::int(n.0)]);
                        let l = rw.body[l].results[0];
                        let e = rw.create(OpData::build("arith.extui").operands(&[l]), &[ty]);
                        rw.body[e].results[0]
                    }
                    None => {
                        let align = Self::align(Self::bytes(&ty));
                        let l = rw.create(OpData::build("llvm.load").operands(&[ptr]).with("alignment", align), &[ty]);
                        rw.body[l].results[0]
                    }
                };
                rw.replace_op(o, &[v]);
            }
            n if n == Store::NAME => {
                let store = rw.body.head::<Store>(o)?;
                Self::unbound(store.binding)?;
                rw.set_before(o);
                let [addr, value] = rw.body[o].operands[..] else { return Err(Error("detour.store: two operands".into())) };
                let ptr = Self::pointer(rw, addr, store.shared);
                let (value, align) = match Self::narrow(store.size)? {
                    Some(n) => {
                        let t = rw.create(OpData::build("arith.trunci").operands(&[value]), &[Type::int(n.0)]);
                        (rw.body[t].results[0], Self::align(n.1))
                    }
                    None => (value, Self::align(Self::bytes(&rw.body[value].ty))),
                };
                rw.create(OpData::build("llvm.store").operands(&[value, ptr]).with("alignment", align), &[]);
                rw.erase_op(o);
            }
            n if n == Atomic::NAME => {
                let atomic = rw.body.head::<Atomic>(o)?;
                let bin_op = match atomic.op {
                    AtomicOp::Exchange => 0,
                    AtomicOp::Add => 1,
                    AtomicOp::Sub => 2,
                    AtomicOp::And => 3,
                    AtomicOp::Or => 5,
                    AtomicOp::Xor => 6,
                    AtomicOp::Max => 7,
                    AtomicOp::Min => 8,
                    AtomicOp::Umax => 9,
                    AtomicOp::Umin => 10,
                };
                rw.set_before(o);
                let [addr, value] = rw.body[o].operands[..] else { return Err(Error("detour.atomic: two operands".into())) };
                let ptr = Self::pointer(rw, addr, atomic.shared);
                let ty = rw.body[rw.body[o].results[0]].ty.clone();
                let data = OpData::build("llvm.atomicrmw").operands(&[ptr, value]).with("bin_op", Attribute::i64(bin_op)).with("ordering", Attribute::i64(MONOTONIC));
                let r = rw.create(data, &[ty]);
                let v = rw.body[r].results[0];
                rw.replace_op(o, &[v]);
            }
            n if n == Barrier::NAME => {
                rw.set_before(o);
                rw.create(OpData::build("gpu.barrier"), &[]);
                rw.erase_op(o);
            }
            n if n.starts_with("detour.") => return Err(Error(format!("a compute shader's {n} has no upstream spelling here"))),
            _ => {}
        }
        Ok(())
    }

    fn id(rw: &mut Rewriter<'_, '_>, o: OpId, name: &str, case: &str) {
        rw.set_before(o);
        let dim = Attribute::Extended { ext: mlirformat::Extended { name: "gpu".into(), body: Some(format!("<dim {case}>").into()), params: Vec::new() }.into(), ty: None };
        let id = rw.create(OpData::build(name).with("dimension", dim), &[Type::Index]);
        let id = rw.body[id].results[0];
        let ty = rw.body[rw.body[o].results[0]].ty.clone();
        let cast = rw.create(OpData::build("arith.index_cast").operands(&[id]), &[ty]);
        let v = rw.body[cast].results[0];
        rw.replace_op(o, &[v]);
    }

    fn pointer(rw: &mut Rewriter<'_, '_>, addr: ValueId, shared: Option<bool>) -> ValueId {
        let space = if shared == Some(true) { SHARED } else { GLOBAL };
        let p = rw.create(OpData::build("llvm.inttoptr").operands(&[addr]), &[LlvmType::Ptr(space).into()]);
        rw.body[p].results[0]
    }

    fn narrow(size: Option<u32>) -> Result<Option<(u32, u32)>, Error> {
        match size {
            None | Some(4) => Ok(None),
            Some(1) => Ok(Some((8, 1))),
            Some(2) => Ok(Some((16, 2))),
            Some(n) => Err(Error(format!("a load / store of {n} bytes a component"))),
        }
    }

    fn bytes(t: &Type<'_>) -> u32 {
        let e = match t {
            Type::Shaped(s) => &*s.elem,
            t => t,
        };
        match e {
            Type::Float(mlirformat::FloatKind::F64) | Type::Integer { width: 64, .. } => 8,
            _ => 4,
        }
    }

    fn align(bytes: u32) -> Attribute<'static> {
        Attribute::integer(i128::from(bytes), Type::int(64))
    }

    fn unbound(binding: Option<u32>) -> Result<(), Error> {
        match binding {
            None => Ok(()),
            Some(b) => Err(Error(format!("a compute shader's load / store of binding {b}"))),
        }
    }
}
