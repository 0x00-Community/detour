use alloc::vec::Vec;

use mlirformat::spirv_types::SpirvType;
use mlirformat::{Attribute, Error, Type};

use crate::{Body, Def, OpData, OpId, Quiet, Rewriter, ValueId, pass};

/// A function's array variable, read and written at computed indices (GLSL ES 3.00 lets every stage index a local
/// array with any integer): one variable an element, each access a chain of compares and selects over them, which
/// mem2reg then promotes. An array stored whole from `vector.from_elements` (its initializer) and accessed through
/// one-index access chains; any other use leaves it as it is
struct LocalArray;

impl LocalArray {
    fn split<'a>(rw: &mut Rewriter<'_, 'a>, var: OpId) -> Result<(), Error> {
        let ptr = rw.body[var].results[0];
        let Ok(SpirvType::Ptr { pointee, storage }) = SpirvType::try_from(&rw.body[ptr].ty) else { return Ok(()) };
        let Ok(SpirvType::Array { len, elem, .. }) = SpirvType::try_from(&*pointee) else { return Ok(()) };
        let (n, elem) = (len as usize, (*elem).clone());
        let mut wholes = Vec::new();
        let mut chains = Vec::new();
        for u in rw.body[ptr].uses.to_vec() {
            let o = &rw.body[u.op];
            match (&*o.name, u.index) {
                ("spirv.Store", 0) => match Self::elements(rw.body, o.operands[1], n) {
                    Some(_) => wholes.push(u.op),
                    None => return Ok(()),
                },
                ("spirv.AccessChain", 0) if o.operands.len() == 2 => {
                    let r = o.results[0];
                    let plain = rw.body[r].uses.iter().all(|c| matches!((&*rw.body[c.op].name, c.index), ("spirv.Load", 0) | ("spirv.Store", 0)));
                    if !plain {
                        return Ok(());
                    }
                    chains.push(u.op);
                }
                _ => return Ok(()),
            }
        }
        let class = rw.body[var].get("storage_class").cloned().ok_or_else(|| Error("spirv.Variable without a storage class".into()))?;
        rw.set_before(var);
        let ptr_ty = SpirvType::ptr(elem.clone(), &storage);
        let cells: Vec<ValueId> = (0..n)
            .map(|_| {
                let c = rw.create(OpData::build("spirv.Variable").with("storage_class", class.clone()), core::slice::from_ref(&ptr_ty));
                rw.body[c].results[0]
            })
            .collect();
        for s in wholes {
            let values = Self::elements(rw.body, rw.body[s].operands[1], n).unwrap_or_default();
            rw.set_before(s);
            for (cell, v) in cells.iter().zip(values) {
                rw.create(OpData::build("spirv.Store").operands(&[*cell, v]), &[]);
            }
            rw.erase_op(s);
        }
        for c in chains {
            let (index, r) = (rw.body[c].operands[1], rw.body[c].results[0]);
            for u in rw.body[r].uses.to_vec() {
                rw.set_before(u.op);
                let load = |rw: &mut Rewriter<'_, 'a>, cell: ValueId| {
                    let l = rw.create(OpData::build("spirv.Load").operands(&[cell]), core::slice::from_ref(&elem));
                    rw.body[l].results[0]
                };
                if rw.body[u.op].name == "spirv.Load" {
                    let mut v = load(rw, cells[0]);
                    for (k, cell) in cells.iter().enumerate().skip(1) {
                        let here = load(rw, *cell);
                        let at = Self::is(rw, index, k, &elem);
                        let s = rw.create(OpData::build("arith.select").operands(&[at, here, v]), core::slice::from_ref(&elem));
                        v = rw.body[s].results[0];
                    }
                    let old = rw.body[u.op].results[0];
                    rw.replace_all_uses(old, v);
                } else {
                    let value = rw.body[u.op].operands[1];
                    for (k, cell) in cells.iter().enumerate() {
                        let was = load(rw, *cell);
                        let at = Self::is(rw, index, k, &elem);
                        let s = rw.create(OpData::build("arith.select").operands(&[at, value, was]), core::slice::from_ref(&elem));
                        let s = rw.body[s].results[0];
                        rw.create(OpData::build("spirv.Store").operands(&[*cell, s]), &[]);
                    }
                }
                rw.erase_op(u.op);
            }
            rw.erase_op(c);
        }
        rw.erase_op(var);
        Ok(())
    }

    /// The `n` elements of an array value built by `vector.from_elements`
    fn elements(body: &Body<'_>, v: ValueId, n: usize) -> Option<Vec<ValueId>> {
        let Def::Result(d, _) = body[v].def else { return None };
        (body[d].name == "vector.from_elements" && body[d].operands.len() == n).then(|| body[d].operands.to_vec())
    }

    /// Whether `index` is `k`: an i1, or a vector of them as wide as `elem`
    fn is<'a>(rw: &mut Rewriter<'_, 'a>, index: ValueId, k: usize, elem: &Type<'a>) -> ValueId {
        let ty = rw.body[index].ty.clone();
        let c = rw.create(OpData::build("arith.constant").with("value", Attribute::of_int(k as i128, ty.clone())), &[ty]);
        let c = rw.body[c].results[0];
        let eq = rw.create(OpData::build("arith.cmpi").operands(&[index, c]).with("predicate", Attribute::i64(0)), &[Type::I1]);
        let eq = rw.body[eq].results[0];
        match elem.dims() {
            Some(dims) if !dims.is_empty() => {
                let b = rw.create(OpData::build("vector.broadcast").operands(&[eq]), &[Type::vector(&dims, &[false], Type::I1)]);
                rw.body[b].results[0]
            }
            _ => eq,
        }
    }
}

#[pass("gpu-local-arrays", on = "builtin.module")]
pub fn gpu_local_arrays<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let mut vars = Vec::new();
    body.walk(op, &mut |o| {
        if body[o].name == "spirv.Variable" {
            vars.push(o);
        }
    });
    let mut quiet = Quiet;
    let mut rw = Rewriter::new(body, &mut quiet);
    for v in vars {
        LocalArray::split(&mut rw, v)?;
    }
    Ok(())
}
