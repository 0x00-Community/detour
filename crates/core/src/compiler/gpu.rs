//! `detour-from-gpu`: shaders written in the `gpu` dialect. Every `gpu.func` of a `gpu.module` becomes a `func.func`
//! of the module (its `detour.stage` kept; a kernel without one is a compute shader), `gpu.return` a `func.return`.

use mlirformat::Error;
use mlirpass::{Body, OpData, OpId, Quiet, Rewriter, pass};

use crate::base::Stage;

/// What a `gpu.func` keeps as a `func.func`.
const KEPT: &[&str] = &["sym_name", "function_type", "arg_attrs", "res_attrs", "sym_visibility", Stage::ATTR];

/// `detour-from-gpu`.
#[pass("detour-from-gpu", on = "builtin.module")]
pub fn detour_from_gpu<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let top = body.top_block(op);
    let mut quiet = Quiet;
    let mut rw = Rewriter::new(body, &mut quiet);
    for m in rw.body.block_ops(top) {
        if rw.body[m].name != "gpu.module" {
            continue;
        }
        let inner = rw.body.top_block(m);
        let funcs: Vec<OpId> = rw.body.ops(inner).filter(|f| rw.body[*f].name == "gpu.func").collect();
        for f in funcs {
            let mut data = OpData::build("func.func");
            for a in rw.body[f].all_attrs().into_iter().filter(|a| KEPT.contains(&&*a.name)) {
                data.set(&a.name, a.value.unwrap_or(mlirformat::Attribute::Unit));
            }
            if rw.body[f].get(Stage::ATTR).is_none() && rw.body[f].get("gpu.kernel").is_some() {
                data.set(Stage::ATTR, Stage::Compute.into());
            }
            // the body moves over whole
            rw.set_before(m);
            let new = rw.create_with(data, &[], |b, new| {
                let regions = core::mem::take(&mut b[f].regions);
                for r in &regions {
                    b[*r].parent = new;
                }
                b[new].regions = regions;
            });
            let mut returns = Vec::new();
            rw.body.walk(new, &mut |o| returns.push(o));
            returns.retain(|r| rw.body[*r].name == "gpu.return");
            for r in returns {
                rw.modify(r, |b| b[r].name = "func.return".into());
            }
        }
        rw.erase_op(m);
    }
    rw.modify(op, |b| {
        b[op].remove("gpu.container_module");
    });
    Ok(())
}
