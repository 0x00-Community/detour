//! The graphics shader compiler's machine IR: GFX10.3 instructions over virtual registers, their register
//! allocation (linear scan on the structured, forward-only code), `s_waitcnt` insertion and encoding.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use mlirformat::Error;

use crate::Rdna2Format as Format;
use crate::Rdna2Opcode as Op;

/// Operand codes: VGPRs from 256, and the special SGPRs
pub(crate) const VGPR: u16 = 256;
pub(crate) const VCC: u16 = 106;
pub(crate) const M0: u16 = 124;
pub(crate) const EXEC: u16 = 126;
const LITERAL: u16 = 255;
const SGPRS: usize = 106;
const VGPRS: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Class {
    S,
    V,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VReg {
    pub class: Class,
    pub size: u8,
    pub align: u8,
    /// a lane mask: allocated to VCC when it is free
    pub mask: bool,
    /// the same in every lane
    pub uniform: bool,
}

/// A 32-bit operand: a virtual register's component, a physical register, a constant
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Src {
    R(u32, u8),
    Phys(u16),
    K(u32),
}

/// An operand with float source modifiers
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct Opnd {
    pub src: Src,
    pub neg: bool,
    pub abs: bool,
}

impl Opnd {
    pub fn of(src: Src) -> Opnd {
        Opnd { src, neg: false, abs: false }
    }

    pub fn plain(&self) -> bool {
        !self.neg && !self.abs
    }
}

pub(crate) enum Inst {
    /// SALU and VALU; a VOPC's or VOP3b's lane mask is its `dst`, `v_cndmask`'s mask its third source
    Alu { op: Op, dst: Src, src: Vec<Opnd>, clamp: bool },
    Smem { op: Op, dst: Src, base: u16, offset: u32 },
    Mubuf { op: Op, dst: Src, vindex: Src, rsrc: u16 },
    Mimg { op: Op, dst: Src, addr: Vec<Src>, rsrc: u16, samp: u16, dmask: u8 },
    Exp { target: u8, src: [Option<Src>; 4], done: bool, compr: bool, vm: bool },
    /// `v_interp_p1_f32` (`p2` false) or `v_interp_p2_f32`, whose `dst` is also its source (the p1 result)
    Interp { p2: bool, dst: Src, ij: Src, attr: u8, chan: u8 },
    Sopp { op: Op, imm: u16 },
    Branch { op: Op, label: u32 },
    /// to `label` when no lane of `mask` is set (a uniform condition)
    BranchZero { mask: Src, label: u32 },
    Label(u32),
    Raw(Vec<u32>),
}

impl Inst {
    fn defs(&self, mut f: impl FnMut(Src)) {
        match self {
            Inst::Alu { dst, .. } | Inst::Smem { dst, .. } | Inst::Mubuf { dst, .. } | Inst::Mimg { dst, .. } | Inst::Interp { dst, .. } => f(*dst),
            _ => {}
        }
    }

    fn uses(&self, mut f: impl FnMut(Src)) {
        match self {
            Inst::Alu { src, .. } => src.iter().for_each(|s| f(s.src)),
            Inst::Mubuf { vindex, .. } => f(*vindex),
            Inst::Mimg { addr, .. } => addr.iter().for_each(|s| f(*s)),
            Inst::Exp { src, .. } => src.iter().flatten().for_each(|s| f(*s)),
            Inst::Interp { p2, dst, ij, .. } => {
                f(*ij);
                if *p2 {
                    f(*dst);
                }
            }
            Inst::BranchZero { mask, .. } => f(*mask),
            _ => {}
        }
    }

    /// Whether its destination may take a register a source frees
    fn overlaps(&self) -> bool {
        matches!(self, Inst::Alu { .. } | Inst::Interp { .. })
    }
}

/// Each virtual register's first physical register (`VGPR` added for VGPRs), and the SGPRs and VGPRs used
pub(crate) struct Allocation {
    pub base: Vec<u16>,
    pub sgprs: u32,
    pub vgprs: u32,
}

/// Linear scan in code order. The code branches only forward and in structured ifs, so a register live from its
/// first definition to its last use (or definition) is live wherever it is needed; a value written in both arms of
/// an if is one register. `reserved_sgprs` are the user and system SGPRs, never reused (an SMEM read of an SGPR a
/// VALU then writes is a hazard); physical VGPRs read are taken until their last read
pub(crate) fn allocate(code: &[Inst], regs: &[VReg], reserved_sgprs: u32) -> Result<Allocation, Error> {
    let n = code.len();
    let mut first: Vec<Option<usize>> = vec![None; regs.len()];
    let mut last: Vec<Vec<usize>> = regs.iter().map(|r| vec![0; r.size as usize]).collect();
    let mut phys_last = [0usize; VGPRS];
    let mut phys_max = 0u32;
    for (i, inst) in code.iter().enumerate() {
        inst.defs(|d| {
            if let Src::R(v, _) = d {
                first[v as usize].get_or_insert(i);
                last[v as usize].iter_mut().for_each(|l| *l = (*l).max(i));
            }
        });
        inst.uses(|u| match u {
            Src::R(v, c) => last[v as usize][c as usize] = last[v as usize][c as usize].max(i),
            Src::Phys(p) if p >= VGPR => {
                phys_last[(p - VGPR) as usize] = i;
                phys_max = phys_max.max((p - VGPR) as u32 + 1);
            }
            _ => {}
        });
    }
    let mut starts: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut ends: Vec<Vec<(u32, u8)>> = vec![Vec::new(); n];
    for (v, f) in first.iter().enumerate() {
        let Some(f) = *f else { continue };
        starts[f].push(v as u32);
        for (c, l) in last[v].iter().enumerate() {
            ends[(*l).max(f)].push((v as u32, c as u8));
        }
    }
    let mut sfree = [true; SGPRS + 2];
    sfree[..(reserved_sgprs as usize).min(SGPRS)].iter_mut().for_each(|s| *s = false);
    let mut vfree = [true; VGPRS];
    let mut base = vec![0u16; regs.len()];
    let (mut sgprs, mut vgprs) = (reserved_sgprs, 0u32);
    for i in 0..n {
        let release = |ends: &[(u32, u8)], sfree: &mut [bool; SGPRS + 2], vfree: &mut [bool; VGPRS], base: &[u16], dead: bool| {
            for &(v, c) in ends {
                if (first[v as usize] == Some(i)) != dead {
                    continue;
                }
                let r = base[v as usize] + c as u16;
                match regs[v as usize].class {
                    Class::S => sfree[r as usize] = true,
                    Class::V => vfree[(r - VGPR) as usize] = true,
                }
            }
        };
        let early = code[i].overlaps();
        if early {
            release(&ends[i], &mut sfree, &mut vfree, &base, false);
        }
        for &v in &starts[i] {
            let r = regs[v as usize];
            let (size, align) = (r.size as usize, r.align.max(1) as usize);
            let at = match r.class {
                Class::S => {
                    let vcc = VCC as usize;
                    if r.mask && sfree[vcc..vcc + size].iter().all(|f| *f) {
                        Some(vcc)
                    } else {
                        (0..=SGPRS - size).step_by(align).find(|&b| sfree[b..b + size].iter().all(|f| *f))
                    }
                }
                Class::V => (0..=VGPRS - size).step_by(align).find(|&b| (b..b + size).all(|k| vfree[k] && (phys_last[k] < i || (early && phys_last[k] == i)))),
            };
            let Some(at) = at else { return Err(Error(format!("RDNA2: out of {:?} registers", r.class))) };
            match r.class {
                Class::S => {
                    sfree[at..at + size].iter_mut().for_each(|f| *f = false);
                    if at < SGPRS {
                        sgprs = sgprs.max((at + size) as u32);
                    }
                    base[v as usize] = at as u16;
                }
                Class::V => {
                    vfree[at..at + size].iter_mut().for_each(|f| *f = false);
                    vgprs = vgprs.max((at + size) as u32);
                    base[v as usize] = VGPR + at as u16;
                }
            }
        }
        if !early {
            release(&ends[i], &mut sfree, &mut vfree, &base, false);
        }
        release(&ends[i], &mut sfree, &mut vfree, &base, true);
    }
    Ok(Allocation { base, sgprs, vgprs: vgprs.max(phys_max) })
}

fn inline(bits: u32) -> Option<u16> {
    let i = bits as i32;
    if (0..=64).contains(&i) {
        return Some(128 + i as u16);
    }
    if (-16..0).contains(&i) {
        return Some((192 - i) as u16);
    }
    let floats = [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0];
    if let Some(k) = floats.iter().position(|f| f.to_bits() == bits) {
        return Some(240 + k as u16);
    }
    (bits == 0x3e22_f983).then_some(248)
}

/// A constant's operand code, `LITERAL` if it takes the instruction's literal dword
pub(crate) fn konst(bits: u32) -> u16 {
    inline(bits).unwrap_or(LITERAL)
}

fn commutes(op: Op) -> bool {
    matches!(op, Op::VAddF32 | Op::VMulF32 | Op::VMaxF32 | Op::VMinF32 | Op::VAndB32 | Op::VOrB32 | Op::VXorB32 | Op::VAddU32)
}

fn swapped(op: Op) -> Option<Op> {
    if commutes(op) {
        return Some(op);
    }
    Some(match op {
        Op::VSubF32 => Op::VSubrevF32,
        Op::VSubrevF32 => Op::VSubF32,
        Op::VCmpLtF32 => Op::VCmpGtF32,
        Op::VCmpGtF32 => Op::VCmpLtF32,
        Op::VCmpLeF32 => Op::VCmpGeF32,
        Op::VCmpGeF32 => Op::VCmpLeF32,
        Op::VCmpNltF32 => Op::VCmpNgtF32,
        Op::VCmpNgtF32 => Op::VCmpNltF32,
        Op::VCmpNleF32 => Op::VCmpNgeF32,
        Op::VCmpNgeF32 => Op::VCmpNleF32,
        Op::VCmpEqF32 | Op::VCmpLgF32 | Op::VCmpOF32 | Op::VCmpUF32 | Op::VCmpNeqF32 | Op::VCmpNlgF32 => op,
        _ => return None,
    })
}

#[derive(Clone, Default)]
struct Pending {
    /// each VGPR a load writes: the loads issued after it
    vm: BTreeMap<u16, u8>,
    lgkm: BTreeMap<u16, ()>,
    exp: BTreeMap<u16, ()>,
}

impl Pending {
    fn merge(&mut self, o: &Pending) {
        for (r, a) in &o.vm {
            let e = self.vm.entry(*r).or_insert(*a);
            *e = (*e).min(*a);
        }
        self.lgkm.extend(o.lgkm.iter().map(|(k, v)| (*k, *v)));
        self.exp.extend(o.exp.iter().map(|(k, v)| (*k, *v)));
    }
}

pub(crate) struct Encoder<'c> {
    base: &'c [u16],
    regs: &'c [VReg],
    wave64: bool,
    out: Vec<u32>,
    labels: BTreeMap<u32, usize>,
    fixups: Vec<(usize, u32)>,
    pending: Pending,
    at_label: BTreeMap<u32, Pending>,
    live: bool,
}

impl<'c> Encoder<'c> {
    pub fn new(base: &'c [u16], regs: &'c [VReg], wave64: bool) -> Self {
        Encoder { base, regs, wave64, out: Vec::new(), labels: BTreeMap::new(), fixups: Vec::new(), pending: Pending::default(), at_label: BTreeMap::new(), live: true }
    }

    fn reg(&self, s: Src) -> u16 {
        match s {
            Src::R(v, c) => self.base[v as usize] + c as u16,
            Src::Phys(p) => p,
            Src::K(k) => konst(k),
        }
    }

    fn size(&self, s: Src) -> u16 {
        match s {
            Src::R(v, _) => self.regs[v as usize].size as u16,
            _ => 1,
        }
    }

    fn lit(s: &[Opnd]) -> Option<u32> {
        s.iter().find_map(|o| match o.src {
            Src::K(k) if inline(k).is_none() => Some(k),
            _ => None,
        })
    }

    fn vgpr(&self, s: Src) -> bool {
        self.reg(s) >= VGPR
    }

    /// The registers an instruction reads and writes (tuples whole)
    fn touches(&self, inst: &Inst) -> (Vec<u16>, Vec<u16>) {
        let (mut r, mut w) = (Vec::new(), Vec::new());
        let span = |s: Src, n: u16, into: &mut Vec<u16>| {
            if !matches!(s, Src::K(_)) {
                let b = self.reg(s);
                into.extend(b..b + n);
            }
        };
        match inst {
            Inst::Alu { dst, src, .. } => {
                src.iter().for_each(|o| span(o.src, 1, &mut r));
                span(*dst, self.size(*dst), &mut w);
            }
            Inst::Smem { dst, base, .. } => {
                r.extend([*base, *base + 1]);
                span(*dst, self.size(*dst), &mut w);
            }
            Inst::Mubuf { dst, vindex, rsrc, .. } => {
                span(*vindex, 1, &mut r);
                r.extend(*rsrc..*rsrc + 4);
                span(*dst, self.size(*dst), &mut w);
            }
            Inst::Mimg { dst, addr, rsrc, samp, .. } => {
                addr.iter().for_each(|a| span(*a, 1, &mut r));
                r.extend(*rsrc..*rsrc + 8);
                r.extend(*samp..*samp + 4);
                span(*dst, self.size(*dst), &mut w);
            }
            Inst::Exp { src, .. } => src.iter().flatten().for_each(|s| span(*s, 1, &mut r)),
            Inst::Interp { p2, dst, ij, .. } => {
                span(*ij, 1, &mut r);
                if *p2 {
                    span(*dst, 1, &mut r);
                }
                span(*dst, 1, &mut w);
            }
            Inst::BranchZero { mask, .. } => span(*mask, self.size(*mask), &mut r),
            _ => {}
        }
        (r, w)
    }

    fn waitcnt(&mut self, vm: Option<u8>, lgkm: bool, exp: bool) {
        if vm.is_none() && !lgkm && !exp {
            return;
        }
        let vm = vm.unwrap_or(63) as u32;
        let word = (vm & 0xf) | (vm >> 4) << 14 | (if exp { 0 } else { 7 }) << 4 | (if lgkm { 0 } else { 0x3f }) << 8;
        self.out.push(0xbf8c_0000 | word);
        if let Some(vm) = (vm < 63).then_some(vm as u8) {
            self.pending.vm.retain(|_, a| *a < vm);
        }
        if lgkm {
            self.pending.lgkm.clear();
        }
        if exp {
            self.pending.exp.clear();
        }
    }

    fn wait_for(&mut self, inst: &Inst) {
        let (r, w) = self.touches(inst);
        let mut vm: Option<u8> = None;
        let (mut lgkm, mut exp) = (false, false);
        for x in r.iter().chain(&w) {
            if let Some(a) = self.pending.vm.get(x) {
                vm = Some(vm.map_or(*a, |v| v.min(*a)));
            }
            lgkm |= self.pending.lgkm.contains_key(x);
        }
        for x in &w {
            exp |= self.pending.exp.contains_key(x);
        }
        if matches!(inst, Inst::Alu { op: Op::SAndSaveexecB32 | Op::SAndSaveexecB64 | Op::SAndn2B32 | Op::SAndn2B64 | Op::SOrB32 | Op::SOrB64 | Op::SMovB32, dst: Src::Phys(EXEC), .. }) {
            exp |= !self.pending.exp.is_empty();
        }
        self.waitcnt(vm, lgkm, exp);
    }

    fn issued(&mut self, inst: &Inst) {
        let (r, w) = self.touches(inst);
        match inst {
            Inst::Mubuf { .. } | Inst::Mimg { .. } => {
                self.pending.vm.values_mut().for_each(|a| *a = (*a + 1).min(62));
                self.pending.vm.extend(w.iter().map(|x| (*x, 0)));
            }
            Inst::Smem { .. } => self.pending.lgkm.extend(w.iter().map(|x| (*x, ()))),
            Inst::Exp { .. } => self.pending.exp.extend(r.iter().map(|x| (*x, ()))),
            _ => {}
        }
    }

    fn src(&mut self, o: &Opnd) -> u32 {
        self.reg(o.src) as u32
    }

    fn vop3(&mut self, code: u32, dst: u16, src: &[Opnd], clamp: bool) {
        let (mut abs, mut neg) = (0u32, 0u32);
        let mut s = [0u32; 3];
        for (k, o) in src.iter().enumerate() {
            s[k] = self.src(o);
            abs |= (o.abs as u32) << k;
            neg |= (o.neg as u32) << k;
        }
        self.out.push(0xd400_0000 | code << 16 | (clamp as u32) << 15 | abs << 8 | (dst & 0xff) as u32);
        self.out.push(neg << 29 | s[2] << 18 | s[1] << 9 | s[0]);
    }

    fn alu(&mut self, op: Op, dst: Src, src: &[Opnd], clamp: bool) -> Result<(), Error> {
        let d = self.reg(dst) as u32;
        let lit = Self::lit(src);
        let plain = !clamp && src.iter().all(Opnd::plain);
        match op.format() {
            Format::Sop1 => {
                let s0 = self.src(&src[0]);
                self.out.push(0xbe80_0000 | d << 16 | op.code() << 8 | s0);
            }
            Format::Sop2 => {
                let (s0, s1) = (self.src(&src[0]), self.src(&src[1]));
                self.out.push(0x8000_0000 | op.code() << 23 | d << 16 | s1 << 8 | s0);
            }
            Format::Sopc => {
                let (s0, s1) = (self.src(&src[0]), self.src(&src[1]));
                self.out.push(0xbf00_0000 | op.code() << 16 | s1 << 8 | s0);
            }
            Format::Vop1 => {
                if plain {
                    let s0 = self.src(&src[0]);
                    self.out.push(0x7e00_0000 | (d - VGPR as u32) << 17 | op.code() << 9 | s0);
                } else {
                    self.vop3(0x180 + op.code(), d as u16, src, clamp);
                }
            }
            Format::Vop2 | Format::Vopc => {
                let cmp = op.format() == Format::Vopc;
                let cnd = op == Op::VCndmaskB32;
                let short_dst = if cmp { d == VCC as u32 } else { true };
                let mask_ok = !cnd || self.reg(src[2].src) == VCC;
                let mut e32 = None;
                if plain && short_dst && mask_ok {
                    if self.vgpr(src[1].src) {
                        e32 = Some((op, src[0], src[1]));
                    } else if self.vgpr(src[0].src) && !cnd
                        && let Some(s) = swapped(op)
                    {
                        e32 = Some((s, src[1], src[0]));
                    }
                }
                match e32 {
                    Some((op, a, b)) => {
                        let (s0, s1) = (self.src(&a), self.src(&b) - VGPR as u32);
                        if cmp {
                            self.out.push(0x7c00_0000 | op.code() << 17 | s1 << 9 | s0);
                        } else {
                            self.out.push(op.code() << 25 | (d - VGPR as u32) << 17 | s1 << 9 | s0);
                        }
                    }
                    None => self.vop3(if cmp { op.code() } else { 0x100 + op.code() }, d as u16, src, clamp),
                }
            }
            Format::Vop3 => self.vop3(op.code(), d as u16, src, clamp),
            f => return Err(Error(format!("RDNA2: {op:?} is {f:?}, not an ALU op"))),
        }
        if let Some(k) = lit {
            self.out.push(k);
        }
        Ok(())
    }

    fn branch(&mut self, op: Op, label: u32) {
        self.fixups.push((self.out.len(), label));
        self.out.push(0xbf80_0000 | op.code() << 16);
    }

    pub fn emit(&mut self, inst: &Inst) -> Result<(), Error> {
        match inst {
            Inst::Label(l) => {
                let mut st = self.at_label.remove(l).unwrap_or_default();
                if self.live {
                    st.merge(&self.pending);
                }
                self.pending = st;
                self.live = true;
                self.labels.insert(*l, self.out.len());
                return Ok(());
            }
            Inst::Branch { label, .. } | Inst::BranchZero { label, .. } => {
                self.wait_for(inst);
                let p = self.pending.clone();
                self.at_label.entry(*label).or_default().merge(&p);
                match inst {
                    Inst::Branch { op, .. } => {
                        self.branch(*op, *label);
                        self.live = *op != Op::SBranch;
                    }
                    Inst::BranchZero { mask, .. } => {
                        let m = self.reg(*mask);
                        if m == VCC {
                            self.branch(Op::SCbranchVccz, *label);
                        } else {
                            let cmp = if self.wave64 { Op::SCmpLgU64 } else { Op::SCmpLgU32 };
                            self.out.push(0xbf00_0000 | cmp.code() << 16 | 128 << 8 | m as u32);
                            self.branch(Op::SCbranchScc0, *label);
                        }
                    }
                    _ => {}
                }
                return Ok(());
            }
            _ => {}
        }
        self.wait_for(inst);
        match inst {
            Inst::Alu { op, dst, src, clamp } => self.alu(*op, *dst, src, *clamp)?,
            Inst::Smem { op, dst, base, offset } => {
                let d = self.reg(*dst) as u32;
                self.out.push(0xf400_0000 | op.code() << 18 | d << 6 | (*base as u32) >> 1);
                self.out.push(0x7d << 25 | offset & 0x1f_ffff);
            }
            Inst::Mubuf { op, dst, vindex, rsrc } => {
                let (d, i) = (self.reg(*dst) - VGPR, self.reg(*vindex) - VGPR);
                self.out.push(0xe000_0000 | op.code() << 18 | 1 << 13);
                self.out.push(0x80 << 24 | (*rsrc as u32 >> 2) << 16 | (d as u32) << 8 | i as u32);
            }
            Inst::Mimg { op, dst, addr, rsrc, samp, dmask } => {
                let d = (self.reg(*dst) - VGPR) as u32;
                let a: Vec<u32> = addr.iter().map(|s| (self.reg(*s) - VGPR) as u32).collect();
                let contiguous = a.windows(2).all(|w| w[1] == w[0] + 1);
                let extra = if contiguous { 0 } else { (a.len() as u32 - 1).div_ceil(4) };
                let dim = 1;
                self.out.push(0xf000_0000 | op.code() << 18 | (*dmask as u32) << 8 | dim << 3 | extra << 1);
                self.out.push((*samp as u32 >> 2) << 21 | (*rsrc as u32 >> 2) << 16 | d << 8 | a[0]);
                if !contiguous {
                    for chunk in a[1..].chunks(4) {
                        self.out.push(chunk.iter().enumerate().fold(0, |w, (k, r)| w | r << (8 * k)));
                    }
                }
            }
            Inst::Exp { target, src, done, compr, vm } => {
                let en = if *compr { 0xf } else { src.iter().enumerate().fold(0, |m, (k, s)| m | (s.is_some() as u32) << k) };
                self.out.push(0xf800_0000 | (*vm as u32) << 12 | (*done as u32) << 11 | (*compr as u32) << 10 | (*target as u32) << 4 | en);
                let w = src.iter().enumerate().fold(0u32, |w, (k, s)| w | s.map_or(0x80, |s| (self.reg(s) - VGPR) as u32) << (8 * k));
                self.out.push(w);
            }
            Inst::Interp { p2, dst, ij, attr, chan } => {
                let (d, s) = ((self.reg(*dst) - VGPR) as u32, (self.reg(*ij) - VGPR) as u32);
                self.out.push(0xc800_0000 | d << 18 | (*p2 as u32) << 16 | (*attr as u32) << 10 | (*chan as u32) << 8 | s);
            }
            Inst::Sopp { op, imm } => self.out.push(0xbf80_0000 | op.code() << 16 | *imm as u32),
            Inst::Raw(w) => self.out.extend(w),
            Inst::Label(_) | Inst::Branch { .. } | Inst::BranchZero { .. } => unreachable!(),
        }
        self.issued(inst);
        Ok(())
    }

    /// The code, branches resolved, padded with `s_code_end` past the instruction prefetch
    pub fn finish(mut self) -> Result<Vec<u8>, Error> {
        for (at, label) in &self.fixups {
            let to = *self.labels.get(label).ok_or_else(|| Error(format!("RDNA2: no label {label}")))?;
            let off = to as i64 - *at as i64 - 1;
            self.out[*at] |= (off as i16 as u16) as u32;
        }
        let end = Op::SCodeEnd.code();
        while self.out.len() % 16 != 0 {
            self.out.push(0xbf80_0000 | end << 16);
        }
        self.out.extend([0xbf80_0000 | end << 16; 48]);
        Ok(self.out.iter().flat_map(|w| w.to_le_bytes()).collect())
    }
}
