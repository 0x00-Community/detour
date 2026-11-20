//! The graphics shader compiler for RDNA2: a program's vertex and fragment `func.func`s (after the generic SPIR-V
//! pipeline) selected to GFX10.3 machine IR (`gfx_mir`) the way ACO would: uniform data loaded once into SGPRs
//! (`s_load` coalesced), vertex inputs and varyings fetched at the entry, `fdiv` as `v_rcp_f32` + `v_mul_f32`, `fmul`
//! used once by an add fused to `v_fma_f32`, `fneg`/`fabs` folded into source modifiers, saturation into `clamp`, small
//! ifs flattened to `v_cndmask_b32`, larger ones branched (`s_cbranch_vccz` when uniform, exec masking when not).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use mlirformat::spirv_types::SpirvType;
use mlirformat::{Const, Error, Scalar};

use super::gfx_mir::*;
use crate::{Attrs, Body, BlockId, GpuToXe3, OpId, PassManager, Passes, Rdna2Opcode as Op, Rdna2Shader, ShaderStage, ValueId, Xe3Draw, Xe3Types};
use crate::{RDNA2_PERSP_CENTER, RDNA2_VS_USER_SGPR};

/// ACO's NGG passthrough prologue (wave32): export the primitive (s3: merged wave info, v0: its data), then run
/// only the wave's vertices; a branch to the end follows it
const NGG_PROLOGUE: [u32; 9] = [0xbf8f0003, 0x8f008803, 0x93fe00c1, 0xbf880002, 0xf8000941, 0x80808000, 0x92808003, 0xbf8cff0f, 0xbefe0300];
/// The VGPR a merged ES+GS wave's vertex index arrives in
const VERTEX_INDEX: u16 = VGPR + 5;
/// Export targets
const MRT0: u8 = 0;
const POS0: u8 = 12;
const PARAM0: u8 = 32;
/// An if this small (its arms' ALU ops) is flattened to selects; NIR's peephole_select limit
const FLATTEN_DIVERGENT: usize = 8;
const FLATTEN_UNIFORM: usize = 2;

pub struct Rdna2Graphics;

impl Rdna2Graphics {
    pub const SHADERS: &'static str =
        "builtin.module(convert-spirv-entry-points, gpu-inline-calls, convert-spirv-cf-to-cf, sroa, mem2reg, gpu-local-arrays, mem2reg, lift-cf-to-scf, canonicalize, cse)";

    /// The vertex and fragment shader of `body` (MLIR text's top-level `spirv.module`s), compiled
    pub fn compile(body: &mut Body<'_>, passes: &Passes) -> Result<(Rdna2Shader, Rdna2Shader), Error> {
        Xe3Draw::wrap(body);
        let root = body.root;
        PassManager::parse(passes, Self::SHADERS)?.run(body, root)?;
        let mut ops = Vec::new();
        body.walk(root, &mut |o| ops.push(o));
        let shaders: Vec<(OpId, ShaderStage)> = ops.into_iter().filter_map(|o| Some((o, GpuToXe3::stage_of(body, o)?))).collect();
        let fs: Vec<OpId> = shaders.iter().map(|s| s.0).collect();
        GpuToXe3::resources(body, &fs)?;
        let (mut vs, mut ps) = (None, None);
        for (f, stage) in shaders {
            let slot = match stage {
                ShaderStage::Vertex => &mut vs,
                ShaderStage::Fragment => &mut ps,
                s => return Err(Error(format!("RDNA2: a program of vertex and fragment shaders, not {s:?}"))),
            };
            if slot.replace(Lower::shader(body, f, stage)?).is_some() {
                return Err(Error("RDNA2: a program of one vertex and one fragment shader".into()));
            }
        }
        match (vs, ps) {
            (Some(vs), Some(ps)) => Ok((vs, ps)),
            _ => Err(Error("RDNA2: a program needs a vertex and a fragment shader".into())),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Val {
    X(Opnd),
    Mask(Src),
    /// an op deferred to its one user, which may fuse it (`fmul` into `v_fma_f32`, `fmax` into `v_med3_f32`)
    Lazy(Op, Opnd, Opnd),
    None,
}

fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn k(x: f32) -> Opnd {
    Opnd::of(Src::K(x.to_bits()))
}

fn konst_of(o: Opnd) -> Option<f32> {
    match o.src {
        Src::K(b) if o.plain() => Some(f(b)),
        _ => None,
    }
}

fn neg(o: Opnd) -> Opnd {
    match o.src {
        Src::K(b) if o.plain() => Opnd::of(Src::K(b ^ 0x8000_0000)),
        _ => Opnd { neg: !o.neg, ..o },
    }
}

fn abs(o: Opnd) -> Opnd {
    match o.src {
        Src::K(b) if o.plain() => Opnd::of(Src::K(b & 0x7fff_ffff)),
        _ => Opnd { neg: false, abs: true, ..o },
    }
}

/// The components of a value its users read (all, unless only extracted from)
fn used(body: &Body<'_>, v: ValueId) -> Vec<u32> {
    let lanes = Xe3Types::lanes(&body[v].ty).unwrap_or(1) as u32;
    let mut set = BTreeSet::new();
    for u in body[v].uses.iter() {
        let o = u.op;
        if &*body[o].name == "vector.extract"
            && let Some(p) = body.dense_i64(o, "static_position").first()
        {
            set.insert(*p as u32);
            continue;
        }
        return (0..lanes).collect();
    }
    set.into_iter().collect()
}

type Memo = BTreeMap<(Op, Vec<Opnd>, bool), Src>;

struct Lower<'b, 'a> {
    body: &'b Body<'a>,
    /// the shader's ABI (where its textures' descriptors are)
    abi: Rdna2Shader,
    regs: Vec<VReg>,
    code: Vec<Inst>,
    vals: BTreeMap<ValueId, Vec<Val>>,
    memo: Vec<Memo>,
    labels: u32,
    ubo: BTreeMap<(u32, u32), Src>,
}

impl<'b, 'a> Lower<'b, 'a> {
    fn shader(body: &'b Body<'a>, f: OpId, stage: ShaderStage) -> Result<Rdna2Shader, Error> {
        if let Some(names) = body.parent_op(f).and_then(|m| body[m].get_str(&GpuToXe3::builtins_attr(stage)))
            && !names.is_empty()
        {
            return Err(Error(format!("RDNA2: built-in inputs ({names}) not yet")));
        }
        let vertex = stage == ShaderStage::Vertex;
        let block = body.top_block(f);
        let inputs: Vec<ValueId> = body[block].args.iter().copied().filter(|a| SpirvType::try_from(&body[*a].ty).is_err()).collect();
        let mut all = Vec::new();
        body.walk(f, &mut |o| all.push(o));
        let int = |o: OpId, n: &str| body[o].get(n).and_then(|a| Attrs::as_int(a).ok()).map(|v| v as u32);
        let (mut ubos, mut textures) = (BTreeMap::<u32, BTreeSet<u32>>::new(), BTreeSet::new());
        for &o in &all {
            let name = &*body[o].name;
            if name == "spirv.Load"
                && let (Some(b), Some(off)) = (int(o, "xe3.binding"), int(o, "xe3.offset"))
            {
                let r = body[o].results[0];
                let set = ubos.entry(b).or_default();
                for c in used(body, r) {
                    set.insert(off / 4 + c);
                }
            } else if name.starts_with("spirv.Image")
                && let Some(b) = int(o, "xe3.binding")
            {
                textures.insert(b);
            }
        }
        let abi = Rdna2Shader {
            stage,
            code: Vec::new(),
            sgprs: 0,
            vgprs: 0,
            inputs: inputs.len() as u32,
            varyings: 0,
            ubos: ubos.keys().copied().collect(),
            textures: textures.into_iter().collect(),
            ps_input: if vertex { 0 } else { RDNA2_PERSP_CENTER },
        };
        let mut l = Lower { body, abi, regs: Vec::new(), code: Vec::new(), vals: BTreeMap::new(), memo: vec![Memo::new()], labels: 0, ubo: BTreeMap::new() };
        let user = if vertex { RDNA2_VS_USER_SGPR } else { 0 };
        let end = l.label();
        if vertex {
            l.code.push(Inst::Raw(NGG_PROLOGUE.to_vec()));
            l.code.push(Inst::Branch { op: Op::SCbranchExecz, label: end });
        }
        for (b, dwords) in &ubos {
            let base = (user + l.abi.ubo_sgpr(*b).unwrap_or(0)) as u16;
            l.preload(*b, base, dwords);
        }
        if vertex {
            l.fetch(&inputs);
        } else {
            l.interpolate(&inputs, (user + l.abi.user_sgprs()) as u16);
        }
        let ret = l.block(block)?.ok_or_else(|| Error("RDNA2: a shader without a return".into()))?;
        if vertex {
            let types = l.result_lanes(f);
            let mut at = 0;
            let mut exports = Vec::new();
            for (n, lanes) in types.iter().enumerate() {
                let comps: Vec<Src> = ret[at..at + lanes].iter().map(|v| l.vgpr_val(*v)).collect::<Result<_, _>>()?;
                at += lanes;
                let mut src = [None; 4];
                for (c, s) in comps.into_iter().enumerate().take(4) {
                    src[c] = Some(s);
                }
                let target = if n == 0 { POS0 } else { PARAM0 + n as u8 - 1 };
                exports.push(Inst::Exp { target, src, done: n == 0, compr: false, vm: false });
            }
            l.abi.varyings = types.len().saturating_sub(1) as u32;
            l.code.extend(exports);
        } else {
            let c: Vec<Opnd> = ret.iter().map(|v| l.x(*v)).collect::<Result<_, _>>()?;
            if c.len() != 4 {
                return Err(Error("RDNA2: a fragment shader returns one vec4".into()));
            }
            let lo = l.alu(Op::VCvtPkrtzF16F32, vec![c[0], c[1]], false);
            let hi = l.alu(Op::VCvtPkrtzF16F32, vec![c[2], c[3]], false);
            l.code.push(Inst::Exp { target: MRT0, src: [Some(lo.src), Some(hi.src), None, None], done: true, compr: true, vm: true });
        }
        l.code.push(Inst::Label(end));
        l.code.push(Inst::Sopp { op: Op::SEndpgm, imm: 0 });
        fold_clamps(&mut l.code);
        let reserved = user + l.abi.user_sgprs() + (!vertex) as u32;
        let alloc = allocate(&l.code, &l.regs, reserved)?;
        let mut enc = Encoder::new(&alloc.base, &l.regs);
        for i in &l.code {
            enc.emit(i)?;
        }
        let mut shader = l.abi;
        shader.code = enc.finish()?;
        shader.sgprs = alloc.sgprs;
        shader.vgprs = alloc.vgprs.max(if vertex { 6 } else { 2 });
        Ok(shader)
    }

    fn label(&mut self) -> u32 {
        self.labels += 1;
        self.labels
    }

    fn reg(&mut self, class: Class, size: u8, mask: bool, uniform: bool) -> u32 {
        let align = match size {
            1 => 1,
            2 => 2,
            _ => 4,
        };
        let align = if class == Class::V { 1 } else { align };
        self.regs.push(VReg { class, size, align, mask, uniform });
        self.regs.len() as u32 - 1
    }

    fn mask_reg(&mut self, uniform: bool) -> u32 {
        self.reg(Class::S, 1, true, uniform)
    }

    fn uniform(&self, o: Opnd) -> bool {
        match o.src {
            Src::K(_) => true,
            Src::Phys(p) => p < VGPR,
            Src::R(v, _) => self.regs[v as usize].uniform,
        }
    }

    fn vgpr(&self, s: Src) -> bool {
        match s {
            Src::R(v, _) => self.regs[v as usize].class == Class::V,
            Src::Phys(p) => p >= VGPR,
            Src::K(_) => false,
        }
    }

    /// `s_load`s of a uniform block's dwords, coalesced (up to 16 dwords, as far as the vec4 the last is in)
    fn preload(&mut self, binding: u32, base: u16, dwords: &BTreeSet<u32>) {
        let Some(&last) = dwords.last() else { return };
        let limit = (last + 4) & !3;
        let mut at = None;
        for &d in dwords {
            if at.is_some_and(|a| d < a) {
                continue;
            }
            let n = [16u32, 8, 4, 2, 1].into_iter().find(|n| d + n <= limit && (*n == 1 || d + n / 2 <= last)).unwrap_or(1);
            let op = match n {
                16 => Op::SLoadDwordx16,
                8 => Op::SLoadDwordx8,
                4 => Op::SLoadDwordx4,
                2 => Op::SLoadDwordx2,
                _ => Op::SLoadDword,
            };
            let r = self.reg(Class::S, n as u8, false, true);
            self.code.push(Inst::Smem { op, dst: Src::R(r, 0), base, offset: d * 4 });
            for c in 0..n {
                self.ubo.insert((binding, d + c), Src::R(r, c as u8));
            }
            at = Some(d + n);
        }
    }

    /// Each vertex input's used components fetched by vertex index (plus the first vertex)
    fn fetch(&mut self, inputs: &[ValueId]) {
        let first = Src::Phys((RDNA2_VS_USER_SGPR + self.abi.first_vertex_sgpr()) as u16);
        let mut index = None;
        for (loc, &a) in inputs.iter().enumerate() {
            let lanes = Xe3Types::lanes(&self.body[a].ty).unwrap_or(1);
            let used = used(self.body, a);
            let Some(&m) = used.last() else {
                self.vals.insert(a, vec![Val::None; lanes]);
                continue;
            };
            let i = *index.get_or_insert_with(|| self.alu(Op::VAddU32, vec![Opnd::of(first), Opnd::of(Src::Phys(VERTEX_INDEX))], false).src);
            let op = [Op::BufferLoadFormatX, Op::BufferLoadFormatXy, Op::BufferLoadFormatXyz, Op::BufferLoadFormatXyzw][m as usize];
            let r = self.reg(Class::V, m as u8 + 1, false, false);
            let rsrc = (RDNA2_VS_USER_SGPR + self.abi.input_sgpr(loc as u32)) as u16;
            self.code.push(Inst::Mubuf { op, dst: Src::R(r, 0), vindex: i, rsrc });
            let vals = (0..lanes as u32).map(|c| if c <= m { Val::X(Opnd::of(Src::R(r, c as u8))) } else { Val::X(k(if c == 3 { 1.0 } else { 0.0 })) }).collect();
            self.vals.insert(a, vals);
        }
    }

    /// Each varying's used components interpolated (perspective, center): all `p1`s, then all `p2`s
    fn interpolate(&mut self, inputs: &[ValueId], prim_mask: u16) {
        let mut todo = Vec::new();
        for (attr, &a) in inputs.iter().enumerate() {
            let lanes = Xe3Types::lanes(&self.body[a].ty).unwrap_or(1);
            let mut vals = vec![Val::X(k(0.0)); lanes];
            for c in used(self.body, a) {
                let r = self.reg(Class::V, 1, false, false);
                todo.push((Src::R(r, 0), attr as u8, c as u8));
                vals[c as usize] = Val::X(Opnd::of(Src::R(r, 0)));
            }
            self.vals.insert(a, vals);
        }
        if todo.is_empty() {
            return;
        }
        self.code.push(Inst::Alu { op: Op::SMovB32, dst: Src::Phys(M0), src: vec![Opnd::of(Src::Phys(prim_mask))], clamp: false });
        for p2 in [false, true] {
            for &(dst, attr, chan) in &todo {
                self.code.push(Inst::Interp { p2, dst, ij: Src::Phys(VGPR + p2 as u16), attr, chan });
            }
        }
    }

    fn result_lanes(&self, f: OpId) -> Vec<usize> {
        let block = self.body.top_block(f);
        let ret = self.body.ops(block).last();
        ret.map(|r| self.body[r].operands.iter().map(|v| Xe3Types::lanes(&self.body[*v].ty).unwrap_or(1)).collect()).unwrap_or_default()
    }

    /// A VALU op, its operands made legal (at most one literal, two SGPRs or literals in all), reused if the same
    /// op on the same operands is in scope
    fn alu(&mut self, op: Op, mut src: Vec<Opnd>, clamp: bool) -> Opnd {
        let key = (op, src.clone(), clamp);
        if let Some(s) = self.memo.iter().rev().find_map(|m| m.get(&key)) {
            return Opnd::of(*s);
        }
        let mut lit = None;
        let mut sgprs: Vec<Src> = Vec::new();
        for o in src.iter_mut() {
            match o.src {
                Src::K(b) if konst(b) == 255 => match lit {
                    Some(l) if l != b => o.src = self.mov(Opnd::of(o.src)),
                    _ => lit = Some(b),
                },
                s @ (Src::R(..) | Src::Phys(_)) if !self.vgpr(s) && !sgprs.contains(&s) => {
                    if sgprs.len() + lit.is_some() as usize >= 2 {
                        o.src = self.mov(Opnd::of(s));
                    } else {
                        sgprs.push(s);
                    }
                }
                _ => {}
            }
        }
        if sgprs.len() + lit.is_some() as usize > 2 {
            for o in src.iter_mut() {
                if matches!(o.src, Src::K(b) if konst(b) == 255) {
                    let s = self.mov(Opnd::of(o.src));
                    o.src = s;
                    break;
                }
            }
        }
        let uniform = src.iter().all(|o| self.uniform(*o));
        let cmp = op.format() == crate::Rdna2Format::Vopc;
        let r = if cmp { self.mask_reg(uniform) } else { self.reg(Class::V, 1, false, uniform) };
        self.code.push(Inst::Alu { op, dst: Src::R(r, 0), src, clamp });
        self.memo.last_mut().unwrap().insert(key, Src::R(r, 0));
        Opnd::of(Src::R(r, 0))
    }

    /// A VGPR holding the operand's value, modifiers applied
    fn mov(&mut self, o: Opnd) -> Src {
        if o.plain() && self.vgpr(o.src) {
            return o.src;
        }
        let base = Opnd::of(o.src);
        if o.abs {
            let a = self.alu(Op::VAndB32, vec![Opnd::of(Src::K(0x7fff_ffff)), base], false);
            return if o.neg { self.alu(Op::VXorB32, vec![Opnd::of(Src::K(0x8000_0000)), a], false).src } else { a.src };
        }
        if o.neg {
            let v = Opnd::of(if self.vgpr(o.src) { o.src } else { self.mov(base) });
            return self.alu(Op::VXorB32, vec![Opnd::of(Src::K(0x8000_0000)), v], false).src;
        }
        self.alu(Op::VMovB32, vec![base], false).src
    }

    /// The operand without modifiers (for ops that take none)
    fn bare(&mut self, o: Opnd) -> Opnd {
        if o.plain() { o } else { Opnd::of(self.mov(o)) }
    }

    fn x(&mut self, v: Val) -> Result<Opnd, Error> {
        match v {
            Val::X(o) => Ok(o),
            Val::Lazy(op, a, b) => Ok(self.alu(op, vec![a, b], false)),
            Val::Mask(_) => Err(Error("RDNA2: a boolean used as a number".into())),
            Val::None => Err(Error("RDNA2: a value with no components".into())),
        }
    }

    fn vgpr_val(&mut self, v: Val) -> Result<Src, Error> {
        let o = self.x(v)?;
        Ok(self.mov(o))
    }

    fn vals(&self, v: ValueId) -> Result<Vec<Val>, Error> {
        self.vals.get(&v).cloned().ok_or_else(|| Error(format!("RDNA2: a value of {} not lowered", self.def_name(v))))
    }

    fn def_name(&self, v: ValueId) -> alloc::string::String {
        match self.body[v].def {
            crate::Def::Result(o, _) => self.body[o].name.to_string(),
            crate::Def::Arg(..) => "an argument".into(),
        }
    }

    fn mask(&self, v: ValueId) -> Result<Src, Error> {
        match self.vals(v)?.first() {
            Some(Val::Mask(m)) => Ok(*m),
            _ => Err(Error("RDNA2: a condition not a comparison".into())),
        }
    }

    /// The single user of a value, if one
    fn single_user(&self, v: ValueId) -> Option<&str> {
        let u = self.body[v].uses.single()?;
        Some(&self.body[u.op].name)
    }

    fn add(&mut self, a: Val, b: Val) -> Result<Val, Error> {
        if let Val::Lazy(Op::VMulF32, x, y) = a {
            let c = self.x(b)?;
            return Ok(Val::X(self.alu(Op::VFmaF32, vec![x, y, c], false)));
        }
        if let Val::Lazy(Op::VMulF32, x, y) = b {
            let c = self.x(a)?;
            return Ok(Val::X(self.alu(Op::VFmaF32, vec![x, y, c], false)));
        }
        let (a, b) = (self.x(a)?, self.x(b)?);
        Ok(Val::X(match (konst_of(a), konst_of(b)) {
            (Some(p), Some(q)) => k(p + q),
            (Some(0.0), _) => b,
            (_, Some(0.0)) => a,
            _ => self.alu(Op::VAddF32, vec![a, b], false),
        }))
    }

    fn sub(&mut self, a: Val, b: Val) -> Result<Val, Error> {
        if let Val::Lazy(Op::VMulF32, x, y) = a {
            let c = self.x(b)?;
            return Ok(Val::X(self.alu(Op::VFmaF32, vec![x, y, neg(c)], false)));
        }
        if let Val::Lazy(Op::VMulF32, x, y) = b {
            let c = self.x(a)?;
            return Ok(Val::X(self.alu(Op::VFmaF32, vec![neg(x), y, c], false)));
        }
        let (a, b) = (self.x(a)?, self.x(b)?);
        Ok(Val::X(match (konst_of(a), konst_of(b)) {
            (Some(p), Some(q)) => k(p - q),
            (_, Some(_)) => return self.add(Val::X(a), Val::X(neg(b))),
            _ => self.alu(Op::VSubF32, vec![a, b], false),
        }))
    }

    fn mul(&mut self, a: Opnd, b: Opnd, lazy: bool) -> Val {
        match (konst_of(a), konst_of(b)) {
            (Some(p), Some(q)) => Val::X(k(p * q)),
            (Some(1.0), _) => Val::X(b),
            (_, Some(1.0)) => Val::X(a),
            (Some(-1.0), _) => Val::X(neg(b)),
            (_, Some(-1.0)) => Val::X(neg(a)),
            _ if lazy => Val::Lazy(Op::VMulF32, a, b),
            _ => Val::X(self.alu(Op::VMulF32, vec![a, b], false)),
        }
    }

    fn clamp(&mut self, x: Opnd, lo: Opnd, hi: Opnd) -> Opnd {
        match (konst_of(x), konst_of(lo), konst_of(hi)) {
            (Some(v), Some(l), Some(h)) => k(v.max(l).min(h)),
            _ => self.alu(Op::VMed3F32, vec![x, lo, hi], false),
        }
    }

    fn unary(&mut self, op: Op, a: Opnd) -> Opnd {
        if op == Op::VRcpF32
            && let Some(c) = konst_of(a)
        {
            return k(1.0 / c);
        }
        self.alu(op, vec![a], false)
    }

    /// Lowers a block's ops; its terminator's operands' components (`scf.yield`, `func.return`)
    fn block(&mut self, block: BlockId) -> Result<Option<Vec<Val>>, Error> {
        for o in self.body.block_ops(block) {
            let name = self.body[o].name.to_string();
            if name == "scf.yield" || name == "func.return" {
                let mut out = Vec::new();
                for &v in self.body[o].operands.iter() {
                    out.extend(self.vals(v)?);
                }
                return Ok(Some(out));
            }
            self.op(o, &name)?;
        }
        Ok(None)
    }

    fn lanes(&self, v: ValueId) -> usize {
        Xe3Types::lanes(&self.body[v].ty).unwrap_or(1)
    }

    /// ALU work in a block, for the flattening choice
    fn cost(&self, block: BlockId) -> usize {
        let mut n = 0;
        for o in self.body.block_ops(block) {
            let name = &*self.body[o].name;
            if name == "scf.if" {
                n += 2 + self.body[o].regions.iter().map(|r| self.body[*r].blocks.first().map_or(0, |b| self.cost(*b))).sum::<usize>();
            } else if name.starts_with("arith.") && name != "arith.constant" || name.starts_with("math.") || name.starts_with("spirv.GL.") || name == "spirv.Dot" {
                n += self.body[o].results.first().map_or(1, |r| self.lanes(*r));
            }
        }
        n
    }

    fn op(&mut self, o: OpId, name: &str) -> Result<(), Error> {
        let body = self.body;
        let results: Vec<ValueId> = body[o].results.to_vec();
        let operands: Vec<ValueId> = body[o].operands.to_vec();
        if name != "scf.if" && results.iter().all(|r| body[*r].uses.is_empty()) {
            return Ok(());
        }
        let r = results.first().copied();
        let lanes = r.map_or(1, |r| self.lanes(r));
        let set = |l: &mut Self, v: Vec<Val>| {
            if let Some(r) = r {
                l.vals.insert(r, v);
            }
        };
        match name {
            "arith.constant" => {
                let value = body[o].get("value").cloned().ok_or_else(|| Error("RDNA2: a constant without a value".into()))?;
                let c = Const::try_from(&value).map_err(|_| Error(format!("RDNA2: constant {value}")))?;
                let vals = (0..lanes)
                    .map(|i| match c.at(i) {
                        Scalar::Float(x) => Ok(Val::X(Opnd::of(Src::K(x.bits as u32)))),
                        Scalar::Int(i) if i.width == 1 => Ok(Val::Mask(Src::K(if i.bits() != 0 { u32::MAX } else { 0 }))),
                        Scalar::Int(i) => Ok(Val::X(Opnd::of(Src::K(i.bits() as u32)))),
                        _ => Err(Error(format!("RDNA2: constant {value}"))),
                    })
                    .collect::<Result<_, _>>()?;
                set(self, vals);
            }
            "ub.poison" => set(self, vec![Val::X(k(0.0)); lanes]),
            "spirv.AccessChain" => set(self, vec![Val::None]),
            "spirv.Load" => {
                let int = |n: &str| body[o].get(n).and_then(|a| Attrs::as_int(a).ok()).map(|v| v as u32);
                if body[o].get("xe3.matrix_stride").is_some() || body[o].get("xe3.push_offset").is_some() {
                    return Err(Error("RDNA2: matrices and push constants in uniform blocks not yet".into()));
                }
                let vals = match (int("xe3.binding"), int("xe3.offset")) {
                    (Some(b), Some(off)) => (0..lanes as u32).map(|c| self.ubo.get(&(b, off / 4 + c)).map_or(Val::None, |s| Val::X(Opnd::of(*s)))).collect(),
                    _ if matches!(SpirvType::try_from(&body[r.unwrap()].ty), Ok(SpirvType::SampledImage(_))) => vec![Val::None],
                    _ => return Err(Error(format!("RDNA2: a load of {} not yet", body[r.unwrap()].ty))),
                };
                set(self, vals);
            }
            "vector.extract" => {
                let p = body.dense_i64(o, "static_position");
                let v = self.vals(operands[0])?;
                let at = *p.first().ok_or_else(|| Error("RDNA2: a dynamic extract".into()))? as usize;
                set(self, vec![v[at]]);
            }
            "vector.insert" => {
                let p = body.dense_i64(o, "static_position");
                let mut v = self.vals(operands[1])?;
                v[p[0] as usize] = self.vals(operands[0])?[0];
                set(self, v);
            }
            "vector.from_elements" => {
                let mut v = Vec::new();
                for x in operands {
                    v.extend(self.vals(x)?);
                }
                set(self, v);
            }
            "vector.broadcast" => {
                let v = self.vals(operands[0])?[0];
                set(self, vec![v; lanes]);
            }
            "vector.shuffle" => {
                let mut all = self.vals(operands[0])?;
                all.extend(self.vals(operands[1])?);
                let v = body.dense_i64(o, "mask").into_iter().map(|m| all[m as usize]).collect();
                set(self, v);
            }
            "spirv.ImageSampleImplicitLod" | "spirv.ImageSampleExplicitLod" => {
                let b = body[o].get("xe3.binding").and_then(|a| Attrs::as_int(a).ok()).ok_or_else(|| Error("RDNA2: a sample of no binding".into()))? as u32;
                let t = self.abi.texture_sgpr(b).ok_or_else(|| Error("RDNA2: an unbound texture".into()))?;
                let explicit_zero = name == "spirv.ImageSampleExplicitLod"
                    && operands.len() == 3
                    && matches!(self.vals(operands[2])?.first(), Some(Val::X(o)) if konst_of(*o) == Some(0.0));
                let image = body[operands[0]].ty.to_string();
                if !image.contains("Dim2D") || image.contains(" Arrayed") || !(name == "spirv.ImageSampleImplicitLod" && operands.len() == 2 || explicit_zero) {
                    return Err(Error(format!("RDNA2: {name} of {image}: only 2D, non-arrayed, level 0 sampling (one mip level) yet")));
                }
                let coord = self.vals(operands[1])?;
                let addr = coord.into_iter().map(|c| self.vgpr_val(c)).collect::<Result<Vec<_>, _>>()?;
                let used = used(self.body, r.unwrap());
                let dmask = used.iter().fold(0u8, |m, c| m | 1 << c);
                let reg = self.reg(Class::V, used.len().max(1) as u8, false, false);
                self.code.push(Inst::Mimg { op: Op::ImageSampleLz, dst: Src::R(reg, 0), addr, rsrc: t as u16, samp: (t + 8) as u16, dmask: dmask.max(1) });
                let mut vals = vec![Val::X(k(0.0)); 4];
                for (i, c) in used.iter().enumerate() {
                    vals[*c as usize] = Val::X(Opnd::of(Src::R(reg, i as u8)));
                }
                set(self, vals);
            }
            "scf.if" => self.lower_if(o)?,
            "arith.cmpf" => {
                let p = body[o].get("predicate").and_then(|a| Attrs::as_int(a).ok()).ok_or_else(|| Error("RDNA2: cmpf without predicate".into()))?;
                let op = match p {
                    1 => Op::VCmpEqF32,
                    2 => Op::VCmpGtF32,
                    3 => Op::VCmpGeF32,
                    4 => Op::VCmpLtF32,
                    5 => Op::VCmpLeF32,
                    6 => Op::VCmpLgF32,
                    7 => Op::VCmpOF32,
                    8 => Op::VCmpNlgF32,
                    9 => Op::VCmpNleF32,
                    10 => Op::VCmpNltF32,
                    11 => Op::VCmpNgeF32,
                    12 => Op::VCmpNgtF32,
                    13 => Op::VCmpNeqF32,
                    14 => Op::VCmpUF32,
                    _ => return Err(Error(format!("RDNA2: cmpf predicate {p}"))),
                };
                let (a, b) = (self.vals(operands[0])?, self.vals(operands[1])?);
                let mut v = Vec::new();
                for i in 0..lanes {
                    let (x, y) = (self.x(a[i])?, self.x(b[i])?);
                    v.push(Val::Mask(self.alu(op, vec![x, y], false).src));
                }
                set(self, v);
            }
            "arith.select" => {
                let c = self.vals(operands[0])?;
                let (a, b) = (self.vals(operands[1])?, self.vals(operands[2])?);
                let mut v = Vec::new();
                for i in 0..lanes {
                    let Val::Mask(m) = c[if c.len() > 1 { i } else { 0 }] else { return Err(Error("RDNA2: select on a non-condition".into())) };
                    let (t, e) = (self.x(a[i])?, self.x(b[i])?);
                    v.push(Val::X(self.select(m, t, e)));
                }
                set(self, v);
            }
            _ => {
                let args: Vec<Vec<Val>> = operands.iter().map(|v| self.vals(*v)).collect::<Result<_, _>>()?;
                let only = r.and_then(|r| self.single_user(r)).map(alloc::borrow::ToOwned::to_owned);
                let fusable = |u: &Option<alloc::string::String>| matches!(u.as_deref(), Some("arith.addf" | "arith.subf"));
                let mut out = Vec::new();
                for i in 0..lanes {
                    let a = |k: usize| args[k][if args[k].len() > 1 { i } else { 0 }];
                    let v = match name {
                        "arith.addf" => self.add(a(0), a(1))?,
                        "arith.subf" => self.sub(a(0), a(1))?,
                        "arith.mulf" => {
                            let (x, y) = (self.x(a(0))?, self.x(a(1))?);
                            self.mul(x, y, fusable(&only))
                        }
                        "arith.divf" => {
                            let (x, y) = (self.x(a(0))?, self.x(a(1))?);
                            let rcp = self.unary(Op::VRcpF32, y);
                            self.mul(x, rcp, fusable(&only))
                        }
                        "arith.negf" => Val::X(neg(self.x(a(0))?)),
                        "math.absf" | "spirv.GL.FAbs" => Val::X(abs(self.x(a(0))?)),
                        "arith.maxnumf" | "arith.maximumf" | "spirv.GL.FMax" => {
                            let (x, y) = (self.x(a(0))?, self.x(a(1))?);
                            if matches!(only.as_deref(), Some("arith.minnumf" | "arith.minimumf" | "spirv.GL.FMin")) && (konst_of(y).is_some() || konst_of(x).is_some()) {
                                Val::Lazy(Op::VMaxF32, x, y)
                            } else {
                                Val::X(self.alu(Op::VMaxF32, vec![x, y], false))
                            }
                        }
                        "arith.minnumf" | "arith.minimumf" | "spirv.GL.FMin" => match (a(0), a(1)) {
                            (Val::Lazy(Op::VMaxF32, x, lo), hi) | (hi, Val::Lazy(Op::VMaxF32, x, lo)) if konst_of(self.x(hi)?).is_some() => {
                                let hi = self.x(hi)?;
                                let (x, lo) = if konst_of(lo).is_some() { (x, lo) } else { (lo, x) };
                                Val::X(self.clamp(x, lo, hi))
                            }
                            (x, y) => {
                                let (x, y) = (self.x(x)?, self.x(y)?);
                                Val::X(self.alu(Op::VMinF32, vec![x, y], false))
                            }
                        },
                        "spirv.GL.FClamp" => {
                            let (x, lo, hi) = (self.x(a(0))?, self.x(a(1))?, self.x(a(2))?);
                            Val::X(self.clamp(x, lo, hi))
                        }
                        "math.sqrt" | "spirv.GL.Sqrt" => { let x = self.x(a(0))?; Val::X(self.unary(Op::VSqrtF32, x)) },
                        "math.rsqrt" | "spirv.GL.InverseSqrt" => { let x = self.x(a(0))?; Val::X(self.unary(Op::VRsqF32, x)) },
                        "math.floor" | "spirv.GL.Floor" => { let x = self.x(a(0))?; Val::X(self.unary(Op::VFloorF32, x)) },
                        "math.ceil" | "spirv.GL.Ceil" => { let x = self.x(a(0))?; Val::X(self.unary(Op::VCeilF32, x)) },
                        "spirv.GL.Fract" => { let x = self.x(a(0))?; Val::X(self.unary(Op::VFractF32, x)) },
                        "math.exp2" | "spirv.GL.Exp2" => { let x = self.x(a(0))?; Val::X(self.unary(Op::VExpF32, x)) },
                        "math.log2" | "spirv.GL.Log2" => { let x = self.x(a(0))?; Val::X(self.unary(Op::VLogF32, x)) },
                        "math.fma" | "spirv.GL.Fma" => {
                            let (x, y, z) = (self.x(a(0))?, self.x(a(1))?, self.x(a(2))?);
                            Val::X(self.alu(Op::VFmaF32, vec![x, y, z], false))
                        }
                        "math.powf" | "spirv.GL.Pow" => {
                            let (x, y) = (self.x(a(0))?, self.x(a(1))?);
                            let l = self.unary(Op::VLogF32, x);
                            let m = self.mul(l, y, false);
                            let m = self.x(m)?;
                            Val::X(self.unary(Op::VExpF32, m))
                        }
                        "spirv.Dot" => {
                            let mut acc: Option<Val> = None;
                            for (&x, &y) in args[0].iter().zip(&args[1]) {
                                let (x, y) = (self.x(x)?, self.x(y)?);
                                let p = self.mul(x, y, true);
                                acc = Some(match acc {
                                    None => p,
                                    Some(s) => self.add(s, p)?,
                                });
                            }
                            acc.unwrap_or(Val::X(k(0.0)))
                        }
                        _ => return Err(Error(format!("RDNA2: {name} not yet"))),
                    };
                    out.push(v);
                }
                set(self, out);
            }
        }
        Ok(())
    }

    fn select(&mut self, m: Src, t: Opnd, e: Opnd) -> Opnd {
        if t == e {
            return t;
        }
        if let Src::K(c) = m {
            return if c != 0 { t } else { e };
        }
        let (t, e) = (self.bare(t), self.bare(e));
        let t = if self.vgpr(t.src) { t } else { Opnd::of(self.mov(t)) };
        self.alu(Op::VCndmaskB32, vec![e, t, Opnd::of(m)], false)
    }

    fn lower_if(&mut self, o: OpId) -> Result<(), Error> {
        let body = self.body;
        let m = self.mask(body[o].operands[0])?;
        let uniform = self.uniform(Opnd::of(m));
        let arms: Vec<Option<BlockId>> = body[o].regions.iter().map(|r| body[*r].blocks.first().copied()).collect();
        let (then, els) = (arms.first().copied().flatten(), arms.get(1).copied().flatten());
        let results: Vec<ValueId> = body[o].results.to_vec();
        let cost = then.map_or(0, |b| self.cost(b)) + els.map_or(0, |b| self.cost(b));
        if let Src::K(c) = m {
            let arm = if c != 0 { then } else { els };
            if let Some(b) = arm {
                let y = self.block(b)?.unwrap_or_default();
                self.bind(&results, y);
            }
            return Ok(());
        }
        if cost <= if uniform { FLATTEN_UNIFORM } else { FLATTEN_DIVERGENT } {
            let t = match then {
                Some(b) => self.block(b)?.unwrap_or_default(),
                None => Vec::new(),
            };
            let e = match els {
                Some(b) => self.block(b)?.unwrap_or_default(),
                None => Vec::new(),
            };
            let mut out = Vec::new();
            for (a, b) in t.into_iter().zip(e) {
                let (a, b) = (self.x(a)?, self.x(b)?);
                out.push(Val::X(self.select(m, a, b)));
            }
            self.bind(&results, out);
            return Ok(());
        }
        let n: usize = results.iter().map(|r| self.lanes(*r)).sum();
        let rs: Vec<u32> = (0..n).map(|_| self.reg(Class::V, 1, false, false)).collect();
        let (else_label, end) = (self.label(), self.label());
        let mut save = None;
        if uniform {
            self.code.push(Inst::BranchZero { mask: m, label: else_label });
        } else {
            let s = self.mask_reg(true);
            self.regs[s as usize].mask = false;
            self.code.push(Inst::Alu { op: Op::SAndSaveexecB32, dst: Src::R(s, 0), src: vec![Opnd::of(m)], clamp: false });
            self.code.push(Inst::Branch { op: Op::SCbranchExecz, label: else_label });
            save = Some(s);
        }
        let mut all_uniform = uniform;
        all_uniform &= self.arm(then, &rs)?;
        if uniform {
            self.code.push(Inst::Branch { op: Op::SBranch, label: end });
        }
        self.code.push(Inst::Label(else_label));
        if let Some(s) = save {
            let exec = Opnd::of(Src::Phys(EXEC));
            self.code.push(Inst::Alu { op: Op::SAndn2B32, dst: Src::Phys(EXEC), src: vec![Opnd::of(Src::R(s, 0)), exec], clamp: false });
            self.code.push(Inst::Branch { op: Op::SCbranchExecz, label: end });
        }
        all_uniform &= self.arm(els, &rs)?;
        self.code.push(Inst::Label(end));
        if let Some(s) = save {
            let exec = Opnd::of(Src::Phys(EXEC));
            self.code.push(Inst::Alu { op: Op::SOrB32, dst: Src::Phys(EXEC), src: vec![exec, Opnd::of(Src::R(s, 0))], clamp: false });
        }
        for r in &rs {
            self.regs[*r as usize].uniform = all_uniform;
        }
        self.bind(&results, rs.iter().map(|r| Val::X(Opnd::of(Src::R(*r, 0)))).collect());
        Ok(())
    }

    fn bind(&mut self, results: &[ValueId], comps: Vec<Val>) {
        let mut at = 0;
        for r in results {
            let n = self.lanes(*r);
            self.vals.insert(*r, comps[at..(at + n).min(comps.len())].to_vec());
            at += n;
        }
    }

    /// An if's arm lowered in its own scope, its yields written to `rs` (a value made in the arm and yielded once
    /// is made in its result register); whether all it yields is uniform
    fn arm(&mut self, block: Option<BlockId>, rs: &[u32]) -> Result<bool, Error> {
        let Some(block) = block else { return Ok(true) };
        let start = self.code.len();
        self.memo.push(Memo::new());
        let y = self.block(block)?.unwrap_or_default();
        let mut uniform = true;
        let mut renamed: BTreeMap<u32, u32> = BTreeMap::new();
        for (i, v) in y.into_iter().enumerate() {
            let o = self.x(v)?;
            uniform &= self.uniform(o);
            let r = rs[i];
            if o.plain()
                && let Src::R(v, 0) = o.src
                && self.regs[v as usize].class == Class::V
                && self.regs[v as usize].size == 1
                && !renamed.contains_key(&v)
                && !rs.contains(&v)
                && self.code[start..].iter().any(|inst| defines(inst, v))
            {
                for inst in &mut self.code[start..] {
                    rename(inst, v, r);
                }
                renamed.insert(v, r);
                continue;
            }
            let src = match o.src {
                Src::R(v, 0) if o.plain() && renamed.contains_key(&v) => Opnd::of(Src::R(renamed[&v], 0)),
                _ => o,
            };
            let src = self.bare(src);
            self.code.push(Inst::Alu { op: Op::VMovB32, dst: Src::R(r, 0), src: vec![src], clamp: false });
        }
        self.memo.pop();
        Ok(uniform)
    }
}

/// `v_med3_f32 d, x, 0, 1` of an `x` read nowhere else and made once, by an op with a clamp bit in the same straight
/// run of code: that op made `d` with its clamp bit set instead (ACO's saturation folding)
fn fold_clamps(code: &mut Vec<Inst>) {
    let mut reads: BTreeMap<u32, usize> = BTreeMap::new();
    let mut made: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    for (i, inst) in code.iter().enumerate() {
        inst.uses(|s| {
            if let Src::R(v, _) = s {
                *reads.entry(v).or_default() += 1;
            }
        });
        inst.defs(|s| {
            if let Src::R(v, _) = s {
                let e = made.entry(v).or_insert((i, 0));
                e.1 += 1;
            }
        });
    }
    let (zero, one) = (Src::K(0), Src::K(1.0f32.to_bits()));
    let mut drop = BTreeSet::new();
    for i in 0..code.len() {
        let Inst::Alu { op: Op::VMed3F32, dst, src, clamp: false } = &code[i] else { continue };
        let (dst, x) = (*dst, src[0]);
        if !(x.plain() && src[1] == Opnd::of(zero) && src[2] == Opnd::of(one)) {
            continue;
        }
        let Src::R(v, 0) = x.src else { continue };
        let Some(&(j, 1)) = made.get(&v) else { continue };
        let touches_dst = |c: &Inst| {
            let mut hit = false;
            c.uses(|s| hit |= s == dst);
            c.defs(|s| hit |= s == dst);
            hit
        };
        let straight = !code[j..i].iter().any(|c| matches!(c, Inst::Label(_) | Inst::Branch { .. } | Inst::BranchZero { .. }));
        if reads.get(&v) != Some(&1) || !straight || code[j + 1..i].iter().any(touches_dst) {
            continue;
        }
        if let Inst::Alu { op, dst: d, clamp: c @ false, .. } = &mut code[j]
            && matches!(op, Op::VAddF32 | Op::VSubF32 | Op::VSubrevF32 | Op::VMulF32 | Op::VFmaF32 | Op::VMaxF32 | Op::VMinF32 | Op::VMed3F32)
        {
            *d = dst;
            *c = true;
            drop.insert(i);
        }
    }
    let mut i = 0;
    code.retain(|_| {
        i += 1;
        !drop.contains(&(i - 1))
    });
}

fn defines(inst: &Inst, v: u32) -> bool {
    matches!(inst, Inst::Alu { dst: Src::R(d, _), .. } if *d == v)
}

fn rename(inst: &mut Inst, from: u32, to: u32) {
    let f = |s: &mut Src| {
        if let Src::R(v, c) = s
            && *v == from
        {
            *s = Src::R(to, *c);
        }
    };
    match inst {
        Inst::Alu { dst, src, .. } => {
            f(dst);
            src.iter_mut().for_each(|o| f(&mut o.src));
        }
        Inst::Mubuf { vindex, .. } => f(vindex),
        Inst::Mimg { addr, .. } => addr.iter_mut().for_each(f),
        Inst::Exp { src, .. } => src.iter_mut().flatten().for_each(f),
        Inst::Interp { dst, ij, .. } => {
            f(dst);
            f(ij);
        }
        Inst::BranchZero { mask, .. } => f(mask),
        _ => {}
    }
}
