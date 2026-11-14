use mlirformat::arith::{Constant, DivF, MaxNumF};
use mlirformat::math::AbsF;
use mlirformat::vector::{BroadcastOp, ExtractOp, FromElementsOp};
use mlirformat::{Attribute, Error, FloatKind, Type};
use mlirpass::{Body, OpId, Quiet, Rewriter, pass};

use crate::base::{Heads, Sample};

#[pass("detour-normalize-cube", on = "builtin.module")]
pub fn detour_normalize_cube<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let mut samples = Vec::new();
    body.walk(op, &mut |o| samples.push(o));
    samples.retain(|s| body[*s].name == "detour.sample" && body.head::<Sample>(*s).is_ok_and(|s| s.cube == Some(true)));
    let mut quiet = Quiet;
    let mut rw = Rewriter::new(body, &mut quiet);
    let f32 = Type::Float(FloatKind::F32);
    for s in samples {
        let d = rw.body[s].operands[0];
        let vt = rw.body[d].ty.clone();
        let arrayed = vt.dims().is_some_and(|d| d[..] == [4]);
        rw.set_before(s);
        let component = |rw: &mut Rewriter<'_, 'a>, k: i64| {
            let e = rw.emit(&ExtractOp::new(d.into(), Vec::new(), Attribute::dense_i64(&[k]), vt.clone(), Type::Index, f32.clone()));
            let e = rw.body[e].results[0];
            let a = rw.emit(&AbsF::new(e.into(), None, f32.clone()));
            rw.body[a].results[0]
        };
        let (x, y, z) = (component(&mut rw, 0), component(&mut rw, 1), component(&mut rw, 2));
        let xy = rw.emit(&MaxNumF::new(x.into(), y.into(), None, f32.clone()));
        let xy = rw.body[xy].results[0];
        let m = rw.emit(&MaxNumF::new(xy.into(), z.into(), None, f32.clone()));
        let m = rw.body[m].results[0];
        let divisor = if arrayed {
            let one = rw.emit(&Constant::new(Attribute::Float { lit: "1.0".into(), ty: Some(f32.clone()) }, f32.clone()));
            let one = rw.body[one].results[0];
            let all = rw.emit(&FromElementsOp::new(vec![m.into(), m.into(), m.into(), one.into()], vec![f32.clone(); 4], vt.clone()));
            rw.body[all].results[0]
        } else {
            let m3 = rw.emit(&BroadcastOp::new(m.into(), f32.clone(), vt.clone()));
            rw.body[m3].results[0]
        };
        let n = rw.emit(&DivF::new(d.into(), divisor.into(), None, None, vt.clone()));
        let n = rw.body[n].results[0];
        rw.body.set_operand(s, 0, n);
    }
    Ok(())
}
