use std::collections::BTreeMap;

use crate::base::{A, Backdrop, Compiler, Effect, Shadow, Gpu, Host, Lane, Load, Stage, Store, Unpack4xu8, WorkgroupId};
use mlirformat::arith::{AddF, AddI, AndI, Bitcast, CmpF, CmpFPredicate, CmpI, CmpIPredicate, Constant, DivF, FPToSI, FPToUI, MaxNumF, MaxSI, MinNumF, MinSI, MulF, MulI, OrI, SIToFP, Select, ShLI, ShRUI, SubF, SubI};
use mlirformat::func::Return;
use mlirformat::math::{AbsF, FloorOp, FmaOp, SqrtOp};
use mlirformat::vector::{BroadcastOp, ExtractOp, FromElementsOp};
use mlirformat::{ApFloat, ApInt, Attribute, Const, FloatKind, FunctionType, Scalar, Type};
use mlirpass::{BlockId, Body, Ctx, OpData, OpId, TypedOp, ValueId};

use crate::xe3::{Buffer, Codegen, Context, Error, Kernel, Recorder};

#[derive(Clone, Copy, Debug)]
pub struct Canvas {
    pub addr: u64,
    pub size: [u32; 2],
    pub stride: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub addr: u64,
    pub stride: u32,
    pub premultiplied: bool,
}

impl Canvas {
    pub fn region(&self, rect: [f32; 4]) -> Option<[u32; 4]> {
        let [width, height] = self.size;
        let x0 = rect[0].max(0.0).floor() as u32;
        let y0 = rect[1].max(0.0).floor() as u32;
        let x1 = ((rect[0] + rect[2]).max(0.0).ceil() as u32).min(width);
        let y1 = ((rect[1] + rect[3]).max(0.0).ceil() as u32).min(height);
        (x1 > x0 + 1 && y1 > y0 + 1).then(|| [x0, y0, x1 - x0, y1 - y0])
    }

    fn within(&self, rect: [u32; 4], clip: [u32; 4]) -> Option<[u32; 4]> {
        let x0 = rect[0].max(clip[0]);
        let y0 = rect[1].max(clip[1]);
        let x1 = (rect[0] + rect[2]).min(clip[0] + clip[2]).min(self.size[0]);
        let y1 = (rect[1] + rect[3]).min(clip[1] + clip[3]).min(self.size[1]);
        (x1 > x0 && y1 > y0).then(|| [x0, y0, x1 - x0, y1 - y0])
    }
}

const INV: f32 = 1.0 / 255.0;

type V = ValueId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    F,
    I,
    VF,
    VI,
    V2I,
}

use Ty::{F, I, V2I, VF, VI};

impl From<Ty> for Type<'_> {
    fn from(t: Ty) -> Self {
        match t {
            F => Type::Float(FloatKind::F32),
            I => Type::int(32),
            VF => Type::vector(&[4], &[false], F.into()),
            VI => Type::vector(&[4], &[false], I.into()),
            V2I => Type::vector(&[2], &[false], I.into()),
        }
    }
}

struct Ssa<'a> {
    body: Body<'a>,
    block: BlockId,
}

macro_rules! binary {
    ($($f:ident $op:ident [$($extra:expr),*];)*) => {$(
        fn $f(&mut self, a: &V, b: &V, ty: Ty) -> V {
            self.let_(&$op::new((*a).into(), (*b).into(), $($extra,)* ty.into()))
        }
    )*};
}

macro_rules! unary {
    ($($f:ident $op:ident;)*) => {$(
        fn $f(&mut self, a: &V, ty: Ty) -> V {
            self.let_(&$op::new((*a).into(), None, ty.into()))
        }
    )*};
}

macro_rules! casts {
    ($($f:ident $op:ident;)*) => {$(
        fn $f(&mut self, a: &V, from: Ty, to: Ty) -> V {
            self.let_(&$op::new((*a).into(), from.into(), to.into()))
        }
    )*};
}

#[derive(Clone, Copy)]
struct Columns {
    bytes: V,
    t: V,
}

impl<'a> Ssa<'a> {
    fn new<const N: usize>(c: &'a Compiler<'_>, name: &str) -> (Self, [V; N]) {
        let mut body = Body::new(Ctx::new(&c.opt, &c.interfaces));
        let fty = Type::Function(FunctionType { inputs: vec![I.into(); N], results: Vec::new() }.into());
        let data = OpData::build("func.func").with("sym_name", A::str(name).into()).with("function_type", Attribute::Type(fty)).with(Stage::ATTR, Stage::Compute.into());
        let f = body.create(data, &[]);
        let top = body.top_block(body.root);
        body.insert(f, top, None);
        let region = body.add_region(f);
        let block = body.add_block(region, &vec![(I.into(), None); N]);
        let args = std::array::from_fn(|k| body[block].args[k]);
        (Ssa { body, block }, args)
    }

    fn put(&mut self, op: &impl TypedOp) -> OpId {
        let o = self.body.emit(op);
        self.body.insert(o, self.block, None);
        o
    }

    fn let_(&mut self, op: &impl TypedOp) -> V {
        let o = self.put(op);
        self.body[o].results[0]
    }

    fn finish(mut self) -> Body<'a> {
        self.put(&Return::new(Vec::new(), Vec::new()));
        self.body
    }

    binary! {
        addf AddF [None, None]; subf SubF [None, None]; mulf MulF [None, None]; divf DivF [None, None];
        minf MinNumF [None]; maxf MaxNumF [None];
        addi AddI [None]; subi SubI [None]; muli MulI [None]; shli ShLI [None]; shrui ShRUI [false];
        andi AndI []; ori OrI []; minsi MinSI []; maxsi MaxSI [];
    }

    unary! { floor FloorOp; abs AbsF; sqrt SqrtOp; }

    casts! { sitofp SIToFP; fptosi FPToSI; fptoui FPToUI; bitcast Bitcast; }

    fn cmpi(&mut self, p: CmpIPredicate, a: &V, b: &V) -> V {
        self.let_(&CmpI::new(p, (*a).into(), (*b).into(), I.into(), Type::int(1)))
    }

    fn cmpf(&mut self, p: CmpFPredicate, a: &V, b: &V) -> V {
        self.let_(&CmpF::new(p, (*a).into(), (*b).into(), None, F.into(), Type::int(1)))
    }

    fn select(&mut self, c: &V, a: &V, b: &V, ty: Ty) -> V {
        self.let_(&Select::new((*c).into(), (*a).into(), (*b).into(), Type::int(1), ty.into()))
    }

    fn f(&mut self, v: f32) -> V {
        let bits = ApFloat::from_bits(FloatKind::F32, v.to_bits().into()).expect("f32 bits");
        self.let_(&Constant::new(Const::scalar(Scalar::Float(bits), F.into()).into(), F.into()))
    }

    fn i(&mut self, v: i32) -> V {
        self.let_(&Constant::new(Attribute::integer(v.into(), I.into()), I.into()))
    }

    fn splat(&mut self, a: &V, from: Ty, to: Ty) -> V {
        self.let_(&BroadcastOp::new((*a).into(), from.into(), to.into()))
    }

    fn fv(&mut self, v: f32) -> V {
        let s = self.f(v);
        self.splat(&s, F, VF)
    }

    fn extract(&mut self, v: &V, k: usize, elem: Ty, ty: Ty) -> V {
        self.let_(&ExtractOp::new((*v).into(), Vec::new(), Attribute::dense_i64(&[k as i64]), ty.into(), Type::Index, elem.into()))
    }

    fn load(&mut self, addr: &V, ty: Ty) -> V {
        self.let_(&Load::new((*addr).into(), I.into(), None, None, None, ty.into()))
    }

    fn store(&mut self, addr: &V, v: &V) {
        self.put(&Store::new((*addr).into(), (*v).into(), I.into(), I.into(), None, None, None));
    }
    fn minus(&mut self, a: &V, k: i32) -> V {
        let k = self.i(-k);
        self.addi(a, &k, I)
    }

    fn clampi(&mut self, a: &V, lo: &V, hi: &V) -> V {
        let m = self.minsi(a, hi, I);
        self.maxsi(&m, lo, I)
    }

    fn clampf(&mut self, a: &V, lo: f32, hi: f32, ty: Ty) -> V {
        let (l, h) = if ty == VF { (self.fv(lo), self.fv(hi)) } else { (self.f(lo), self.f(hi)) };
        let m = self.maxf(a, &l, ty);
        self.minf(&m, &h, ty)
    }

    fn halves(&mut self, v: &V) -> (V, V) {
        let lo = self.i(0xffff);
        let sixteen = self.i(16);
        (self.andi(v, &lo, I), self.shrui(v, &sixteen, I))
    }

    fn pixel(&mut self, w: &V) -> (V, V) {
        let gx = self.let_(&WorkgroupId::new(None, I.into()));
        let y = self.let_(&WorkgroupId::new(Some(1), I.into()));
        let lane = self.let_(&Lane::new(I.into()));
        let sixteen = self.i(16);
        let base = self.muli(&gx, &sixteen, I);
        let x = self.addi(&base, &lane, I);
        let last = self.minus(w, 1);
        (self.minsi(&x, &last, I), y)
    }

    fn address(&mut self, base: &V, stride: &V, x: &V, y: &V) -> V {
        let row = self.row(base, stride, y);
        self.column(&row, x)
    }

    fn row(&mut self, base: &V, stride: &V, y: &V) -> V {
        let r = self.muli(y, stride, I);
        let two = self.i(2);
        let r = self.shli(&r, &two, I);
        self.addi(base, &r, I)
    }

    fn column(&mut self, row: &V, x: &V) -> V {
        let two = self.i(2);
        let b = self.shli(x, &two, I);
        self.addi(row, &b, I)
    }

    fn unpack(&mut self, word: &V) -> V {
        self.let_(&Unpack4xu8::new((*word).into(), I.into(), VF.into()))
    }

    fn fma(&mut self, a: &V, b: &V, c: &V, ty: Ty) -> V {
        self.let_(&FmaOp::new((*a).into(), (*b).into(), (*c).into(), None, None, ty.into()))
    }

    fn fetch(&mut self, base: &V, stride: &V, x: &V, y: &V) -> V {
        let at = self.address(base, stride, x, y);
        let word = self.load(&at, I);
        self.unpack(&word)
    }

    fn pack(&mut self, v: &V) -> V {
        let half = self.fv(0.5);
        let r = self.addf(v, &half, VF);
        let r = self.floor(&r, VF);
        let r = self.clampf(&r, 0.0, 255.0, VF);
        let b = self.fptoui(&r, VF, VI);
        let shifts = Const { ty: VI.into(), values: [0, 8, 16, 24].map(|k| Scalar::Int(ApInt::new(32, k))).into(), elements: true };
        let shifts = self.let_(&Constant::new(shifts.into(), VI.into()));
        let b = self.shli(&b, &shifts, VI);
        let parts = [0, 1, 2, 3].map(|k| self.extract(&b, k, I, VI));
        let lo = self.ori(&parts[0], &parts[1], I);
        let hi = self.ori(&parts[2], &parts[3], I);
        self.ori(&lo, &hi, I)
    }

    fn lerp(&mut self, a: &V, b: &V, t: &V) -> V {
        let d = self.subf(b, a, VF);
        self.fma(t, &d, a, VF)
    }

    fn split(&mut self, p: &V) -> (V, V) {
        let fl = self.floor(p, F);
        let frac = self.subf(p, &fl, F);
        (self.fptosi(&fl, F, I), frac)
    }

    fn columns(&mut self, q: &V, fx: &V, first: &V, last_pair: &V) -> Columns {
        let zero = self.i(0);
        let p = self.clampi(q, &zero, last_pair);
        let two = self.i(2);
        let bytes = self.shli(&p, &two, I);
        let (z, one) = (self.f(0.0), self.f(1.0));
        let below = self.cmpi(CmpIPredicate::slt, q, first);
        let above = self.cmpi(CmpIPredicate::sgt, q, last_pair);
        let t = self.select(&above, &one, fx, F);
        let t = self.select(&below, &z, &t, F);
        Columns { t: self.splat(&t, F, VF), bytes }
    }

    fn pair(&mut self, row: &V, cols: &Columns) -> V {
        let at = self.addi(row, &cols.bytes, I);
        let w = self.load(&at, V2I);
        let a = self.extract(&w, 0, I, V2I);
        let b = self.extract(&w, 1, I, V2I);
        let (a, b) = (self.unpack(&a), self.unpack(&b));
        self.lerp(&a, &b, &cols.t)
    }

    fn sample(&mut self, base: &V, stride: &V, last_pair: &V, hl: &V, x: &V, y: &V) -> V {
        let (x0, fx) = self.split(x);
        let (y0, fy) = self.split(y);
        let zero = self.i(0);
        let cols = self.columns(&x0, &fx, &zero, last_pair);
        let one = self.i(1);
        let y1 = self.addi(&y0, &one, I);
        let (r0, r1) = (self.clampi(&y0, &zero, hl), self.clampi(&y1, &zero, hl));
        let (r0, r1) = (self.row(base, stride, &r0), self.row(base, stride, &r1));
        let top = self.pair(&r0, &cols);
        let bottom = self.pair(&r1, &cols);
        let fy = self.splat(&fy, F, VF);
        self.lerp(&top, &bottom, &fy)
    }

    fn blend_over(&mut self, src: &V, dst: &V) -> V {
        let sa = self.extract(src, 3, F, VF);
        let full = self.f(255.0);
        let rest = self.subf(&full, &sa, F);
        let inv = self.f(INV);
        let k = self.mulf(&rest, &inv, F);
        let k = self.splat(&k, F, VF);
        let under = self.mulf(dst, &k, VF);
        self.addf(src, &under, VF)
    }

    fn over(&mut self, src: &V, at: &V) {
        let word = self.load(at, I);
        let dst = self.unpack(&word);
        let out = self.blend_over(src, &dst);
        let word = self.pack(&out);
        self.store(at, &word);
    }

    fn premultiply(&mut self, c: &V) -> V {
        let a = self.extract(c, 3, F, VF);
        let inv = self.f(INV);
        let k = self.mulf(&a, &inv, F);
        let [r, g, b] = [0, 1, 2].map(|i| self.extract(c, i, F, VF)).map(|v| self.mulf(&v, &k, F));
        self.vector([&r, &g, &b, &a])
    }

    fn level(mut self, out: &V, db: &V, w: &V, x: &V, y: &V) -> Body<'a> {
        let word = self.pack(out);
        let at = self.address(db, w, x, y);
        self.store(&at, &word);
        self.finish()
    }

    fn vector(&mut self, [r, g, b, a]: [&V; 4]) -> V {
        self.let_(&FromElementsOp::new([r, g, b, a].map(|v| (*v).into()).into(), vec![F.into(); 4], VF.into()))
    }
}

struct Pass {
    name: &'static str,
    taps: &'static [(i32, i32, f32)],
    norm: f32,
}

impl Pass {
    const DOWN: Pass = Pass { name: "frost_down", taps: &[(0, 0, 4.0), (-1, -1, 1.0), (1, 1, 1.0), (1, -1, 1.0), (-1, 1, 1.0)], norm: 0.125 };
    const UP: Pass = Pass {
        name: "frost_up",
        taps: &[(-2, 0, 1.0), (-1, 1, 2.0), (0, 2, 1.0), (1, 1, 2.0), (2, 0, 1.0), (1, -1, 2.0), (0, -2, 1.0), (-1, -1, 2.0)],
        norm: 1.0 / 12.0,
    };

    fn build<'a>(&self, c: &'a Compiler<'_>) -> Body<'a> {
        let (mut s, [sb, ss, sw, sh, db, dw, kx, ky]) = Ssa::new(c, self.name);
        let (x, y) = s.pixel(&dw);
        let half = s.f(0.5);
        let centre = |s: &mut Ssa<'_>, i: &V, k: &V| {
            let k = s.bitcast(k, I, F);
            let i = s.sitofp(i, I, F);
            let c = s.addf(&i, &half, F);
            let c = s.mulf(&c, &k, F);
            let c = s.subf(&c, &half, F);
            s.split(&c)
        };
        let (x0, fx) = centre(&mut s, &x, &kx);
        let (y0, fy) = centre(&mut s, &y, &ky);
        let fy = s.splat(&fy, F, VF);
        let zero = s.i(0);
        let hl = s.minus(&sh, 1);
        let last_pair = s.minus(&sw, 2);
        let last_pair = s.maxsi(&last_pair, &zero, I);
        let one = s.i(1);
        let wide = s.cmpi(CmpIPredicate::sgt, &sw, &one);
        let never = s.i(i32::MAX);
        let first = s.select(&wide, &zero, &never, I);

        let mut rows = BTreeMap::new();
        let mut cols = BTreeMap::new();
        let mut pairs: BTreeMap<(i32, i32), V> = BTreeMap::new();
        let mut sum: Option<V> = None;
        for &(dx, dy, w) in self.taps {
            let c = cols.entry(dx).or_insert_with(|| {
                let d = s.i(dx);
                let q = s.addi(&x0, &d, I);
                s.columns(&q, &fx, &first, &last_pair)
            });
            let c = *c;
            let mut texels = Vec::new();
            for ry in [dy, dy + 1] {
                let v = pairs.entry((ry, dx)).or_insert_with(|| {
                    let row = rows.entry(ry).or_insert_with(|| {
                        let d = s.i(ry);
                        let r = s.addi(&y0, &d, I);
                        let r = s.clampi(&r, &zero, &hl);
                        s.row(&sb, &ss, &r)
                    });
                    s.pair(row, &c)
                });
                texels.push(*v);
            }
            let t = s.lerp(&texels[0], &texels[1], &fy);
            let k = (w != 1.0).then(|| s.fv(w));
            sum = Some(match (sum, k) {
                (None, None) => t,
                (None, Some(k)) => s.mulf(&t, &k, VF),
                (Some(acc), None) => s.addf(&acc, &t, VF),
                (Some(acc), Some(k)) => s.fma(&t, &k, &acc, VF),
            });
        }
        let sum = sum.expect("taps");
        let norm = s.fv(self.norm);
        let v = s.mulf(&sum, &norm, VF);
        s.level(&v, &db, &dw, &x, &y)
    }
}

#[derive(Clone, Copy)]
enum Blend {
    Backdrop,
    Filter,
    Shadow,
    Place,
}

impl Blend {
    const BLOCK: usize = 48;

    fn name(self) -> &'static str {
        match self {
            Blend::Backdrop => "frost_backdrop",
            Blend::Filter => "frost_filter",
            Blend::Shadow => "frost_shadow",
            Blend::Place => "frost_place",
        }
    }

    fn dwords(self) -> usize {
        match self {
            Blend::Backdrop => 35,
            Blend::Filter => 30,
            Blend::Shadow => 16,
            Blend::Place => 10,
        }
    }

    fn build<'a>(self, c: &'a Compiler<'_>) -> Body<'a> {
        let (mut s, [cb, cs, pa]) = Ssa::new(c, self.name());
        let mut block = Vec::new();
        for k in 0..self.dwords().div_ceil(4) {
            let off = s.i(16 * k as i32);
            let at = s.addi(&pa, &off, I);
            let v = s.load(&at, VI);
            block.extend((0..4).map(|j| s.extract(&v, j, I, VI)));
        }
        let fl = |s: &mut Ssa<'_>, k: usize| s.bitcast(&block[k], I, F);
        let matrix = |s: &mut Ssa<'_>, p: &V, at: usize| {
            let pa = s.extract(p, 3, F, VF);
            let full = s.f(255.0);
            let k = s.divf(&full, &pa, F);
            let fz = s.f(0.0);
            let empty = s.cmpf(CmpFPredicate::oeq, &pa, &fz);
            let c = [0, 1, 2].map(|i| s.extract(p, i, F, VF)).map(|v| s.mulf(&v, &k, F));
            let c = [&c[0], &c[1], &c[2], &pa];
            let c = c.map(|v| s.select(&empty, &fz, v, F));
            let m: Vec<V> = (0..4)
                .map(|i| {
                    let mut acc = None::<V>;
                    for (j, cj) in c.iter().enumerate() {
                        let mij = fl(s, at + 4 * i + j);
                        let t = s.mulf(&mij, cj, F);
                        acc = Some(match acc {
                            Some(a) => s.addf(&a, &t, F),
                            None => t,
                        });
                    }
                    let bias = fl(s, at + 16 + i);
                    let v = s.addf(&acc.expect("4 columns"), &bias, F);
                    s.clampf(&v, 0.0, 255.0, F)
                })
                .collect();
            let m = s.vector([&m[0], &m[1], &m[2], &m[3]]);
            s.premultiply(&m)
        };
        let (x, y) = s.pixel(&block[2]);
        let cx = s.addi(&block[0], &x, I);
        let cy = s.addi(&block[1], &y, I);
        let (sb, ss, sw, sh) = (block[4], block[5], block[6], block[7]);
        let zero = s.i(0);
        let hl = s.minus(&sh, 1);
        let last_pair = s.minus(&sw, 2);
        let last_pair = s.maxsi(&last_pair, &zero, I);
        let src = match self {
            Blend::Backdrop => {
                let [rx, ry, irw, irh, hx, hy, rr] = [8, 9, 10, 11, 12, 13, 14].map(|k| fl(&mut s, k));
                let half = s.f(0.5);
                let local = |s: &mut Ssa<'_>, c: &V, r: &V| {
                    let p = s.sitofp(c, I, F);
                    let p = s.addf(&p, &half, F);
                    s.subf(&p, r, F)
                };
                let lx = local(&mut s, &cx, &rx);
                let ly = local(&mut s, &cy, &ry);
                let at = |s: &mut Ssa<'_>, l: &V, inv: &V, n: &V| {
                    let u = s.mulf(l, inv, F);
                    let u = s.clampf(&u, 0.0, 1.0, F);
                    let n = s.sitofp(n, I, F);
                    let p = s.mulf(&u, &n, F);
                    s.subf(&p, &half, F)
                };
                let sx = at(&mut s, &lx, &irw, &sw);
                let sy = at(&mut s, &ly, &irh, &sh);
                let c = s.sample(&sb, &ss, &last_pair, &hl, &sx, &sy);

                let c = matrix(&mut s, &c, 15);

                let fz = s.f(0.0);
                let edge = |s: &mut Ssa<'_>, l: &V, hw: &V| {
                    let p = s.subf(l, hw, F);
                    let p = s.abs(&p, F);
                    let q = s.subf(&p, hw, F);
                    s.addf(&q, &rr, F)
                };
                let qx = edge(&mut s, &lx, &hx);
                let qy = edge(&mut s, &ly, &hy);
                let mx = s.maxf(&qx, &fz, F);
                let my = s.maxf(&qy, &fz, F);
                let mx2 = s.mulf(&mx, &mx, F);
                let my2 = s.mulf(&my, &my, F);
                let len = s.addf(&mx2, &my2, F);
                let len = s.sqrt(&len, F);
                let inside = s.maxf(&qx, &qy, F);
                let inside = s.minf(&inside, &fz, F);
                let d = s.addf(&len, &inside, F);
                let d = s.subf(&d, &rr, F);
                let mask = s.subf(&half, &d, F);
                let mask = s.clampf(&mask, 0.0, 1.0, F);
                let mask = s.splat(&mask, F, VF);
                let src = s.mulf(&c, &mask, VF);
                s.clampf(&src, 0.0, 255.0, VF)
            }
            Blend::Filter => {
                let tx = s.subi(&cx, &block[8], I);
                let ty = s.subi(&cy, &block[9], I);
                let wl = s.minus(&sw, 1);
                let tx = s.clampi(&tx, &zero, &wl);
                let ty = s.clampi(&ty, &zero, &hl);
                let p = s.fetch(&sb, &ss, &tx, &ty);
                matrix(&mut s, &p, 10)
            }
            Blend::Shadow => {
                let [ex, ey, cr, cg, cb, ca, wf, hf] = [8, 9, 10, 11, 12, 13, 14, 15].map(|k| fl(&mut s, k));
                let at = |s: &mut Ssa<'_>, c: &V, e: &V| {
                    let p = s.sitofp(c, I, F);
                    s.subf(&p, e, F)
                };
                let sx = at(&mut s, &cx, &ex);
                let sy = at(&mut s, &cy, &ey);
                let c = s.sample(&sb, &ss, &last_pair, &hl, &sx, &sy);
                let a = s.extract(&c, 3, F, VF);
                let a = s.mulf(&a, &ca, F);
                let half = s.f(0.5);
                let fz = s.f(0.0);
                let mut a = a;
                for (p, n) in [(&sx, &wf), (&sy, &hf)] {
                    let e = s.addf(p, &half, F);
                    let lo = s.cmpf(CmpFPredicate::oge, &e, &fz);
                    let hi = s.cmpf(CmpFPredicate::ole, &e, n);
                    a = s.select(&hi, &a, &fz, F);
                    a = s.select(&lo, &a, &fz, F);
                }
                let [r, g, b] = [&cr, &cg, &cb].map(|k| s.mulf(k, &a, F));
                let src = s.vector([&r, &g, &b, &a]);
                s.clampf(&src, 0.0, 255.0, VF)
            }
            Blend::Place => {
                let tx = s.subi(&cx, &block[8], I);
                let ty = s.subi(&cy, &block[9], I);
                s.fetch(&sb, &ss, &tx, &ty)
            }
        };
        let at = s.address(&cb, &cs, &cx, &cy);
        s.over(&src, &at);
        s.finish()
    }
}

type Image = [u32; 4];

pub struct Frost<'d, G: Gpu<Cmd = u64> + 'd> {
    down: Kernel,
    up: Kernel,
    blends: [Kernel; 4],
    over: Kernel,
    clear: Kernel,
    flatten: Kernel,
    levels: Buffer<'d, G>,
    max: [u32; 2],
    params: Vec<Buffer<'d, G>>,
    capacity: usize,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Frost<'d, G> {
    pub fn shaders<'a>(c: &'a Compiler<'_>) -> Vec<(&'static str, Body<'a>)> {
        let mut out = vec![(Pass::DOWN.name, Pass::DOWN.build(c)), (Pass::UP.name, Pass::UP.build(c))];
        out.extend([Blend::Backdrop, Blend::Filter, Blend::Shadow, Blend::Place].map(|b| (b.name(), b.build(c))));
        out.push(("frost_over", Self::over(c)));
        out.push(("frost_clear", Self::clear(c)));
        out.push(("frost_flatten", Self::flatten(c)));
        out
    }

    fn over<'a>(c: &'a Compiler<'_>) -> Body<'a> {
        let (mut s, [cb, cs, ub, us, xy, wh, db]) = Ssa::new(c, "frost_over");
        let (w, _) = s.halves(&wh);
        let (x, y) = s.pixel(&w);
        let (x0, y0) = s.halves(&xy);
        let cx = s.addi(&x0, &x, I);
        let cy = s.addi(&y0, &y, I);
        let c = s.fetch(&cb, &cs, &cx, &cy);
        let u = s.fetch(&ub, &us, &cx, &cy);
        let out = s.blend_over(&c, &u);
        s.level(&out, &db, &w, &x, &y)
    }

    fn clear<'a>(c: &'a Compiler<'_>) -> Body<'a> {
        let (mut s, [cb, cs, xy, wh]) = Ssa::new(c, "frost_clear");
        let (w, _) = s.halves(&wh);
        let (x, y) = s.pixel(&w);
        let (x0, y0) = s.halves(&xy);
        let cx = s.addi(&x0, &x, I);
        let cy = s.addi(&y0, &y, I);
        let at = s.address(&cb, &cs, &cx, &cy);
        let zero = s.i(0);
        s.store(&at, &zero);
        s.finish()
    }

    fn flatten<'a>(c: &'a Compiler<'_>) -> Body<'a> {
        let (mut s, [sb, ss, wh, db]) = Ssa::new(c, "frost_flatten");
        let (w, _) = s.halves(&wh);
        let (x, y) = s.pixel(&w);
        let c = s.fetch(&sb, &ss, &x, &y);
        let out = s.premultiply(&c);
        s.level(&out, &db, &w, &x, &y)
    }

    pub fn new<H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>, max: [u32; 2], capacity: usize) -> Result<Self, Error> {
        let mut k = Vec::new();
        for (name, body) in Self::shaders(&codegen.0) {
            k.push(cx.kernel_body(codegen, body, name).map_err(|e| Error::msg(format!("@{name}: {e}")))?);
        }
        let [down, up, backdrop, filter, shadow, place, over, clear, flatten]: [Kernel; 9] = k.try_into().map_err(|_| Error::msg("9 shaders"))?;
        let levels = cx.buffer(&vec![0; Self::offset(max, 6) as usize + 64])?;
        let params = (0..cx.ring()).map(|_| cx.buffer(&vec![0; capacity * Blend::BLOCK * 4])).collect::<Result<_, _>>()?;
        Ok(Frost { down, up, blends: [backdrop, filter, shadow, place], over, clear, flatten, levels, max, params, capacity })
    }

    fn size([w, h]: [u32; 2], i: u32) -> [u32; 2] {
        [(w >> i).max(1), (h >> i).max(1)]
    }

    fn offset(size: [u32; 2], i: u32) -> u64 {
        (0..i).fold(0, |at, l| {
            let [w, h] = Self::size(size, l);
            (at + u64::from(w) * u64::from(h) * 4).next_multiple_of(64)
        })
    }

    pub fn frame(&mut self, slot: usize) -> FrostFrame<'_, 'd, G> {
        FrostFrame { frost: self, slot, used: 0 }
    }
}

pub struct FrostFrame<'f, 'd, G: Gpu<Cmd = u64> + 'd> {
    frost: &'f mut Frost<'d, G>,
    slot: usize,
    used: usize,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> FrostFrame<'_, 'd, G> {
    fn level(&self, size: [u32; 2], i: u32) -> Image {
        let at = self.frost.levels.addr + Frost::<G>::offset(size, i);
        let [w, h] = Frost::<G>::size(size, i);
        [at as u32, w, w, h]
    }

    fn blur<H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, src: Image, n: u32) -> Result<Image, Error> {
        let size = [src[2], src[3]];
        if size[0] > self.frost.max[0] || size[1] > self.frost.max[1] {
            return Err(Error::msg(format!("a blur of {size:?}, room for {:?}", self.frost.max)));
        }
        let pass = |r: &mut Recorder<'_, 'd, G, H>, kernel: &Kernel, s: Image, d: Image| {
            let k = [s[2] as f32 / d[2] as f32, s[3] as f32 / d[3] as f32].map(f32::to_bits);
            r.dispatch(kernel, [d[2].div_ceil(16), d[3]], &[s[0], s[1], s[2], s[3], d[0], d[2], k[0], k[1]])
        };
        let image = |i: u32| if i == 0 { src } else { self.level(size, i) };
        for i in 0..n {
            pass(r, &self.frost.down, image(i), image(i + 1))?;
        }
        for i in (0..n).rev() {
            let dst = if i == 0 { self.level(size, 0) } else { image(i) };
            pass(r, &self.frost.up, image(i + 1), dst)?;
        }
        Ok(self.level(size, 0))
    }

    fn blend<H: Host>(&mut self, r: &mut Recorder<'_, 'd, G, H>, kind: Blend, canvas: &Canvas, pixels: [u32; 4], src: Image, rest: &[u32]) -> Result<(), Error> {
        if self.used == self.frost.capacity {
            return Err(Error::msg(format!("more than {} effects a frame", self.frost.capacity)));
        }
        let mut block = [0u32; Blend::BLOCK];
        block[..4].copy_from_slice(&pixels);
        block[4..8].copy_from_slice(&src);
        block[8..8 + rest.len()].copy_from_slice(rest);
        let at = self.used * Blend::BLOCK * 4;
        let params = &mut self.frost.params[self.slot];
        for (to, v) in params.bytes()?[at..][..Blend::BLOCK * 4].as_chunks_mut::<4>().0.iter_mut().zip(block) {
            *to = v.to_le_bytes();
        }
        let pa = (params.addr + at as u64) as u32;
        self.used += 1;
        r.dispatch(&self.frost.blends[kind as usize], [pixels[2].div_ceil(16), pixels[3]], &[canvas.addr as u32, canvas.stride, pa])
    }

    pub fn backdrop<H: Host>(&mut self, r: &mut Recorder<'_, 'd, G, H>, canvas: &Canvas, under: Option<&Layer>, b: &Backdrop, clips: &[[u32; 4]]) -> Result<(), Error> {
        let Some(region @ [x0, y0, w, h]) = canvas.region(b.rect) else { return Ok(()) };
        let pixels: Vec<[u32; 4]> = clips.iter().filter_map(|c| canvas.within(region, *c)).collect();
        if pixels.is_empty() {
            return Ok(());
        }
        let n = b.levels();
        let region = (canvas.addr + (u64::from(y0) * u64::from(canvas.stride) + u64::from(x0)) * 4) as u32;
        let src = match under {
            Some(u) => {
                let l0 = self.level([w, h], 0);
                r.dispatch(&self.frost.over, [w.div_ceil(16), h], &[canvas.addr as u32, canvas.stride, u.addr as u32, u.stride, x0 | y0 << 16, w | h << 16, l0[0]])?;
                l0
            }
            None => [region, canvas.stride, w, h],
        };
        let blurred = self.blur(r, src, n)?;
        let [rx, ry, rw, rh] = b.rect;
        let (hx, hy) = (rw * 0.5, rh * 0.5);
        let rr = b.radius.min(hx.min(hy));
        let m = b.effect.matrix.unwrap_or(Effect::IDENTITY);
        let mut rest: Vec<u32> = [rx, ry, 1.0 / rw, 1.0 / rh, hx, hy, rr].map(f32::to_bits).to_vec();
        rest.extend((0..16).map(|k| m[(k / 4) * 5 + k % 4].to_bits()));
        rest.extend((0..4).map(|i| (m[i * 5 + 4] * 255.0).to_bits()));
        pixels.into_iter().try_for_each(|p| self.blend(r, Blend::Backdrop, canvas, p, blurred, &rest))
    }

    fn flat<H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, layer: &Layer, [w, h]: [u32; 2]) -> Result<Image, Error> {
        if layer.premultiplied {
            return Ok([layer.addr as u32, layer.stride, w, h]);
        }
        let l0 = self.level([w, h], 0);
        r.dispatch(&self.frost.flatten, [w.div_ceil(16), h], &[layer.addr as u32, layer.stride, w | h << 16, l0[0]])?;
        Ok(l0)
    }

    pub fn filtered<H: Host>(&mut self, r: &mut Recorder<'_, 'd, G, H>, canvas: &Canvas, layer: &Layer, e: &Effect, [x, y, w, h]: [u32; 4], clip: [u32; 4]) -> Result<(), Error> {
        let Some(pixels) = canvas.within(clip, [0, 0, canvas.size[0], canvas.size[1]]) else { return Ok(()) };
        if w < 2 || h < 2 {
            return Ok(());
        }
        let n = e.levels();
        let flat = self.flat(r, layer, [w, h])?;
        let src = if e.blur > 0.0 { self.blur(r, flat, n)? } else { flat };
        let m = e.matrix.unwrap_or(Effect::IDENTITY);
        let mut rest = vec![x, y];
        rest.extend((0..16).map(|k| m[(k / 4) * 5 + k % 4].to_bits()));
        rest.extend((0..4).map(|i| (m[i * 5 + 4] * 255.0).to_bits()));
        self.blend(r, Blend::Filter, canvas, pixels, src, &rest)
    }

    pub fn shadow<H: Host>(&mut self, r: &mut Recorder<'_, 'd, G, H>, canvas: &Canvas, layer: &Layer, s: &Shadow, [x, y, w, h]: [u32; 4], clip: [u32; 4]) -> Result<(), Error> {
        if w < 2 || h < 2 {
            return Ok(());
        }
        let (ex, ey) = (x as f32 + s.dx, y as f32 + s.dy);
        let Some(shifted) = canvas.region([ex, ey, w as f32, h as f32]) else { return Ok(()) };
        let Some(pixels) = canvas.within(shifted, clip) else { return Ok(()) };
        let n = s.levels();
        let blurred = self.blur(r, [layer.addr as u32, layer.stride, w, h], n)?;
        let c = s.color;
        let rest = [ex, ey, c[0], c[1], c[2], c[3], w as f32, h as f32].map(f32::to_bits);
        self.blend(r, Blend::Shadow, canvas, pixels, blurred, &rest)
    }

    pub fn place<H: Host>(&mut self, r: &mut Recorder<'_, 'd, G, H>, canvas: &Canvas, layer: &Layer, origin: [u32; 2], clip: [u32; 4]) -> Result<(), Error> {
        if !layer.premultiplied {
            return Err(Error::msg("frost: placing a straight layer (not yet)"));
        }
        let Some(pixels) = canvas.within(clip, [0, 0, canvas.size[0], canvas.size[1]]) else { return Ok(()) };
        self.blend(r, Blend::Place, canvas, pixels, [layer.addr as u32, layer.stride, 0, 0], &origin)
    }

    pub fn clear<H: Host>(&mut self, r: &mut Recorder<'_, 'd, G, H>, canvas: &Canvas, rect: [u32; 4]) -> Result<(), Error> {
        let Some([x, y, w, h]) = canvas.within(rect, [0, 0, canvas.size[0], canvas.size[1]]) else { return Ok(()) };
        r.dispatch(&self.frost.clear, [w.div_ceil(16), h], &[canvas.addr as u32, canvas.stride, x | y << 16, w | h << 16])
    }
}
