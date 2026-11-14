//! Constants where they are read: canonicalize gathers every `arith.constant` at a function's entry, where each would
//! hold a register through the whole kernel; every backend's lowering first copies each one next to each of its uses.

use alloc::collections::BTreeSet;

use mlirformat::{Op, arith::Constant};
use mlirpass::{Body, OpData, OpId, Quiet, Rewriter};

/// ⭐ [`Body`]'s constant sinking (a backend calls it on its shaders before lowering them).
pub trait SinkConstants {
    /// Every `arith.constant` in `shaders` copied next to each of its uses, the original gone.
    fn sink_constants(&mut self, shaders: &[OpId]);
}

impl SinkConstants for Body<'_> {
    fn sink_constants(&mut self, shaders: &[OpId]) {
        let mut constants = Vec::new();
        for &f in shaders {
            self.walk(f, &mut |o| {
                if self[o].name == Constant::NAME {
                    constants.push(o);
                }
            });
        }
        let mut quiet = Quiet;
        let mut rw = Rewriter::new(self, &mut quiet);
        for k in constants {
            let v = rw.body[k].results[0];
            let ty = rw.body[v].ty.clone();
            let uses: BTreeSet<OpId> = rw.body[v].uses.iter().map(|u| u.op).collect();
            for user in uses {
                let mut data = OpData::build(Constant::NAME);
                data.properties = rw.body[k].properties.clone();
                data.attributes = rw.body[k].attributes.clone();
                rw.set_before(user);
                let c = rw.create(data, core::slice::from_ref(&ty));
                let cv = rw.body[c].results[0];
                rw.replace_uses_if(v, cv, |_, u| u.op == user);
            }
            rw.erase_op(k);
        }
    }
}
