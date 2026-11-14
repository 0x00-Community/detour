use mlirformat::{Error, FloatKind, Op, ShapedKind, Type};
use mlirpass::{Body, OpId, pass};

use crate::base::{Lane, Load, Stage, Store, Unpack4xu8, WorkgroupId};
use mlirpass::Heads;

#[pass("rdna2-precheck", on = "builtin.module")]
pub fn rdna2_precheck<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let top = body.top_block(op);
    let shaders: Vec<OpId> = body.ops(top).filter(|f| body[*f].name == "func.func" && Stage::of(body, *f).is_some()).collect();
    for f in shaders {
        Precheck::shader(body, f)?;
    }
    Ok(())
}

struct Precheck;

impl Precheck {
    fn shader(body: &Body<'_>, f: OpId) -> Result<(), Error> {
        let name = body[f].get("sym_name").map(|n| n.to_string()).unwrap_or_default();
        let refuse = |why: String| Err(Error(format!("RDNA2: shader {name}: {why}")));
        match Stage::of(body, f) {
            Some(Stage::Compute) => {}
            s => return refuse(format!("{s:?} shaders are not lowered (compute only)")),
        }
        for attr in ["detour.workgroup", "detour.shared"] {
            if body[f].get(attr).is_some() {
                return refuse(format!("{attr}: workgroups of more threads and shared memory are not lowered"));
            }
        }
        let mut ops = Vec::new();
        body.walk(f, &mut |o| ops.push(o));
        for o in ops {
            let n = &*body[o].name;
            let ok = match n {
                _ if n == Lane::NAME || n == Unpack4xu8::NAME => true,
                _ if n == WorkgroupId::NAME => body.head::<WorkgroupId>(o)?.dim.unwrap_or(0) < 3,
                _ if n == Load::NAME => {
                    let l = body.head::<Load>(o)?;
                    l.binding.is_none() && !l.shared.unwrap_or(false) && Precheck::global(l.size, &l.result)
                }
                _ if n == Store::NAME => {
                    let s = body.head::<Store>(o)?;
                    s.binding.is_none() && !s.shared.unwrap_or(false) && Precheck::global(s.size, &s.value_ty)
                }
                _ if n.starts_with("detour.") => false,
                _ => true,
            };
            if !ok {
                return refuse(format!("{n} is not lowered as written ({})", body[o].all_attrs().iter().map(|a| a.name.to_string()).collect::<Vec<_>>().join(", ")));
            }
        }
        Ok(())
    }

    fn global(size: Option<u32>, ty: &Type<'_>) -> bool {
        let (n, elem) = match ty {
            Type::Shaped(s) if s.kind == ShapedKind::Vector => match ty.dims().as_deref() {
                Some([n]) => (*n as usize, &*s.elem),
                _ => return false,
            },
            t => (1, t),
        };
        let dword = *elem == Type::int(32) || *elem == Type::Float(FloatKind::F32);
        match (size.unwrap_or(4), n) {
            (4, 1..=4) => dword,
            (1 | 2, 1) => *elem == Type::int(32),
            _ => false,
        }
    }
}
