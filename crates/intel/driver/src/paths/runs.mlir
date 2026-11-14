// strip::render a row run a lane (appended to stripgen.mlir, its helpers): a path's sorted tiles fall into runs of
// one row; a run's strips and alphas depend only on its row (its start: the culled windings' captive strip and
// accumulated winding, or none) and on the row of the run before (the background strips between them).
//
// @runs, a path a lane: pass 0 counts its runs (at rcounts + 4 p; a culled path without tiles one, its
// background), pass 1 writes them from roffsets[p] on: 32 bytes each, (path, first tile (global), tiles, the row
// before (its y, the first's -1), flags: 1 first, 2 last, 4 a run). params: paths (strip rendering's table: first
// line, first tile, tiles, rule), windings, consts (as @strips'), tiles, runs, rcounts, roffsets.
func.func @runs(%params: i32, %npaths: i32, %pass: i32) attributes {detour.stage = "compute"} {
  %q4 = arith.constant 4 : i32
  %paths = "detour.load"(%params) : (i32) -> i32
  %q1 = arith.addi %params, %q4 : i32
  %windings = "detour.load"(%q1) : (i32) -> i32
  %q2 = arith.addi %q1, %q4 : i32
  %consts = "detour.load"(%q2) : (i32) -> i32
  %q3 = arith.addi %q2, %q4 : i32
  %tiles = "detour.load"(%q3) : (i32) -> i32
  %q5 = arith.addi %q3, %q4 : i32
  %runs = "detour.load"(%q5) : (i32) -> i32
  %q6 = arith.addi %q5, %q4 : i32
  %rcounts = "detour.load"(%q6) : (i32) -> i32
  %q7 = arith.addi %q6, %q4 : i32
  %roffsets = "detour.load"(%q7) : (i32) -> i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32
  %c16 = arith.constant 16 : i32
  %c20 = arith.constant 20 : i32
  %c31 = arith.constant 31 : i32
  %c32 = arith.constant 32 : i32
  %m1 = arith.constant -1 : i32
  %rows = "detour.load"(%consts) : (i32) -> i32
  %k2 = arith.constant 8 : i32
  %ka = arith.addi %consts, %k2 : i32
  %wstride = "detour.load"(%ka) : (i32) -> i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %p = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %p, %npaths : i32
  scf.if %inside {
    %po = arith.muli %p, %c16 : i32
    %pa = arith.addi %paths, %po : i32
    %pv = "detour.load"(%pa) : (i32) -> vector<4xi32>
    %ft = vector.extract %pv[1] : i32 from vector<4xi32>
    %nt = vector.extract %pv[2] : i32 from vector<4xi32>
    %p4r = arith.muli %p, %q4 : i32
    %roa = arith.addi %roffsets, %p4r : i32
    %write = arith.cmpi eq, %pass, %c1 : i32
    %slot0 = scf.if %write -> (i32) {
      %v = "detour.load"(%roa) : (i32) -> i32
      scf.yield %v : i32
    } else {
      scf.yield %c0 : i32
    }
    %wb = func.call @wblock(%windings, %consts, %p, %wstride) : (i32, i32, i32, i32) -> i32
    %rw0 = arith.addi %rows, %c31 : i32
    %words = arith.shrui %rw0, %c5 : i32
    %r20 = arith.muli %rows, %c20 : i32
    %w4 = arith.muli %words, %c4 : i32
    %co = arith.addi %r20, %w4 : i32
    %ca = arith.addi %wb, %co : i32
    %culledw = "detour.load"(%ca) : (i32) -> i32
    %culled = arith.cmpi ne, %culledw, %c0 : i32
    %none = arith.cmpi eq, %nt, %c0 : i32
    %c7 = arith.constant 7 : i32
    %nruns = scf.if %none -> (i32) {
      // a culled path without tiles: one run, its background (flags first, last, a run); else none
      %one = arith.select %culled, %c1, %c0 : i32
      %w1 = arith.andi %culled, %write : i1
      scf.if %w1 {
        func.call @run_put(%runs, %slot0, %p, %ft, %c0, %m1, %c7) : (i32, i32, i32, i32, i32, i32, i32) -> ()
      }
      scf.yield %one : i32
    } else {
      %t8 = arith.constant 8 : i32
      %tb0 = arith.muli %ft, %t8 : i32
      %tb = arith.addi %tiles, %tb0 : i32
      %t0:3 = func.call @tileat(%tb, %c0, %nt) : (i32, i32, i32) -> (i32, i32, i32)
      // walk: the run open (its first tile, its row, the row before), runs written
      %w:4 = scf.for %i = %c1 to %nt step %c1 iter_args(%rf = %c0, %ry = %t0#2, %rprev = %m1, %k = %c0) -> (i32, i32, i32, i32) : i32 {
        %t:3 = func.call @tileat(%tb, %i, %nt) : (i32, i32, i32) -> (i32, i32, i32)
        %newrow = arith.cmpi ne, %t#2, %ry : i32
        %o:4 = scf.if %newrow -> (i32, i32, i32, i32) {
          %first = arith.cmpi eq, %k, %c0 : i32
          %fl0 = arith.select %first, %c1, %c0 : i32
          %c4f = arith.constant 4 : i32
          %fl = arith.ori %fl0, %c4f : i32
          %n = arith.subi %i, %rf : i32
          %slot = arith.addi %slot0, %k : i32
          %g0 = arith.addi %ft, %rf : i32
          scf.if %write {
            func.call @run_put(%runs, %slot, %p, %g0, %n, %rprev, %fl) : (i32, i32, i32, i32, i32, i32, i32) -> ()
          }
          %k1 = arith.addi %k, %c1 : i32
          scf.yield %i, %t#2, %ry, %k1 : i32, i32, i32, i32
        } else {
          scf.yield %rf, %ry, %rprev, %k : i32, i32, i32, i32
        }
        scf.yield %o#0, %o#1, %o#2, %o#3 : i32, i32, i32, i32
      }
      // the last run
      %first = arith.cmpi eq, %w#3, %c0 : i32
      %fl0 = arith.select %first, %c1, %c0 : i32
      %c6 = arith.constant 6 : i32
      %fl = arith.ori %fl0, %c6 : i32
      %n = arith.subi %nt, %w#0 : i32
      %slot = arith.addi %slot0, %w#3 : i32
      %g0 = arith.addi %ft, %w#0 : i32
      scf.if %write {
        func.call @run_put(%runs, %slot, %p, %g0, %n, %w#2, %fl) : (i32, i32, i32, i32, i32, i32, i32) -> ()
      }
      %nr = arith.addi %w#3, %c1 : i32
      scf.yield %nr : i32
    }
    %pass0 = arith.cmpi eq, %pass, %c0 : i32
    scf.if %pass0 {
      %rca = arith.addi %rcounts, %p4r : i32
      "detour.store"(%rca, %nruns) : (i32, i32) -> ()
    }
  }
  return
}

func.func private @run_put(%runs: i32, %slot: i32, %p: i32, %first: i32, %n: i32, %prev: i32, %flags: i32) {
  %c32 = arith.constant 32 : i32
  %o = arith.muli %slot, %c32 : i32
  %a = arith.addi %runs, %o : i32
  %v = vector.from_elements %p, %first, %n, %prev : vector<4xi32>
  "detour.store"(%a, %v) : (i32, vector<4xi32>) -> ()
  %c16 = arith.constant 16 : i32
  %a2 = arith.addi %a, %c16 : i32
  "detour.store"(%a2, %flags) : (i32, i32) -> ()
  return
}

// @strips_run, a run slot a lane: its strips and alpha bytes. params: lines, paths, tiles, windings, consts, counts
// (strips, alphas: two i32 a slot), offsets (likewise), strips, alphas, scratch (80 bytes a slot), runs, roffsets (a
// path's first run). Pass 0
// counts; pass 1 writes at the offsets, alpha indices from the path's first slot's alpha offset.
func.func @strips_run(%params: i32, %nslots: i32, %pass: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %lines = "detour.load"(%params) : (i32) -> i32
  %a1 = arith.addi %params, %c4 : i32
  %paths = "detour.load"(%a1) : (i32) -> i32
  %a2 = arith.addi %a1, %c4 : i32
  %tiles = "detour.load"(%a2) : (i32) -> i32
  %a3 = arith.addi %a2, %c4 : i32
  %windings = "detour.load"(%a3) : (i32) -> i32
  %a4 = arith.addi %a3, %c4 : i32
  %consts = "detour.load"(%a4) : (i32) -> i32
  %a5 = arith.addi %a4, %c4 : i32
  %counts = "detour.load"(%a5) : (i32) -> i32
  %a6 = arith.addi %a5, %c4 : i32
  %offsets = "detour.load"(%a6) : (i32) -> i32
  %a7 = arith.addi %a6, %c4 : i32
  %strips = "detour.load"(%a7) : (i32) -> i32
  %a8 = arith.addi %a7, %c4 : i32
  %alphas = "detour.load"(%a8) : (i32) -> i32
  %a9 = arith.addi %a8, %c4 : i32
  %scratch = "detour.load"(%a9) : (i32) -> i32
  %a10 = arith.addi %a9, %c4 : i32
  %runs = "detour.load"(%a10) : (i32) -> i32
  %a11 = arith.addi %a10, %c4 : i32
  %roffsets = "detour.load"(%a11) : (i32) -> i32
  %rows = "detour.load"(%consts) : (i32) -> i32
  %k1 = arith.addi %consts, %c4 : i32
  %vw = "detour.load"(%k1) : (i32) -> i32
  %k2 = arith.addi %k1, %c4 : i32
  %wstride = "detour.load"(%k2) : (i32) -> i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %s = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %s, %nslots : i32
  scf.if %inside {
    %c32 = arith.constant 32 : i32
    %ro = arith.muli %s, %c32 : i32
    %ra = arith.addi %runs, %ro : i32
    %rv = "detour.load"(%ra) : (i32) -> vector<4xi32>
    %c16r = arith.constant 16 : i32
    %rfa = arith.addi %ra, %c16r : i32
    %flags = "detour.load"(%rfa) : (i32) -> i32
    %p = vector.extract %rv[0] : i32 from vector<4xi32>
    %rfirst = vector.extract %rv[1] : i32 from vector<4xi32>
    %rn = vector.extract %rv[2] : i32 from vector<4xi32>
    %rprev = vector.extract %rv[3] : i32 from vector<4xi32>
    %isrun0 = arith.andi %flags, %c4 : i32
    %isrun = arith.cmpi ne, %isrun0, %c0 : i32
    %write = arith.cmpi eq, %pass, %c1 : i32
    %s8 = arith.muli %s, %c8 : i32
    %n:2 = scf.if %isrun -> (i32, i32) {
      %po = arith.muli %p, %c16 : i32
      %pa = arith.addi %paths, %po : i32
      %pv = "detour.load"(%pa) : (i32) -> vector<4xi32>
      %fl = vector.extract %pv[0] : i32 from vector<4xi32>
      %ft = vector.extract %pv[1] : i32 from vector<4xi32>
      %rule = vector.extract %pv[3] : i32 from vector<4xi32>
      // where this slot writes, and the path's first slot's alpha offset (its alpha indices count from there)
      %oa = arith.addi %offsets, %s8 : i32
      %p4s = arith.muli %p, %c4 : i32
      %rpa = arith.addi %roffsets, %p4s : i32
      %slot0 = "detour.load"(%rpa) : (i32) -> i32
      %s08 = arith.muli %slot0, %c8 : i32
      %oa0 = arith.addi %offsets, %s08 : i32
      %ov:3 = scf.if %write -> (i32, i32, i32) {
        %v = "detour.load"(%oa) : (i32) -> vector<2xi32>
        %v0 = "detour.load"(%oa0) : (i32) -> vector<2xi32>
        %so = vector.extract %v[0] : i32 from vector<2xi32>
        %ao = vector.extract %v[1] : i32 from vector<2xi32>
        %ab = vector.extract %v0[1] : i32 from vector<2xi32>
        scf.yield %so, %ao, %ab : i32, i32, i32
      } else {
        scf.yield %c0, %c0, %c0 : i32, i32, i32
      }
      %so8 = arith.muli %ov#0, %c8 : i32
      %sout = arith.addi %strips, %so8 : i32
      // the alphas at aout + na with na path-relative: aout = alphas + the path's base
      %abase = arith.subi %ov#1, %ov#2 : i32
      %aout = arith.addi %alphas, %ov#2 : i32
      %wb = func.call @wblock(%windings, %consts, %p, %wstride) : (i32, i32, i32, i32) -> i32
      %c80 = arith.constant 80 : i32
      %sc0 = arith.muli %s, %c80 : i32
      %sc = arith.addi %scratch, %sc0 : i32
      %l16 = arith.muli %fl, %c16 : i32
      %lb = arith.addi %lines, %l16 : i32
      %tb0 = arith.muli %rfirst, %c8 : i32
      %tb = arith.addi %tiles, %tb0 : i32
      %r:2 = func.call @render_run(%lb, %tb, %rn, %rprev, %flags, %rule, %wb, %rows, %vw, %sc, %sout, %aout, %abase, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
      scf.yield %r#0, %r#1 : i32, i32
    } else {
      scf.yield %c0, %c0 : i32, i32
    }
    %first = arith.cmpi eq, %pass, %c0 : i32
    scf.if %first {
      %ca = arith.addi %counts, %s8 : i32
      %cv = vector.from_elements %n#0, %n#1 : vector<2xi32>
      "detour.store"(%ca, %cv) : (i32, vector<2xi32>) -> ()
    }
  }
  return
}

// one run: its strips (from sout, counted from 0) and alpha bytes (at aout + na, na from abase: the path's alpha
// bytes before it); how many of each it wrote
func.func private @render_run(%lb: i32, %tb: i32, %nt: i32, %prev: i32, %flags: i32, %rule: i32, %wb: i32, %rows: i32, %vw: i32, %sc: i32, %sout: i32, %aout: i32, %abase: i32, %write: i1) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32
  %c16 = arith.constant 16 : i32
  %c20 = arith.constant 20 : i32
  %c31 = arith.constant 31 : i32
  %m1 = arith.constant -1 : i32
  %z = arith.constant 0.0 : f32
  %true = arith.constant true
  %false = arith.constant false
  %rw0 = arith.addi %rows, %c31 : i32
  %words = arith.shrui %rw0, %c5 : i32
  %r20 = arith.muli %rows, %c20 : i32
  %w4 = arith.muli %words, %c4 : i32
  %co = arith.addi %r20, %w4 : i32
  %ca = arith.addi %wb, %co : i32
  %culledw = "detour.load"(%ca) : (i32) -> i32
  %culled = arith.cmpi ne, %culledw, %c0 : i32
  %f1 = arith.andi %flags, %c1 : i32
  %isfirst = arith.cmpi ne, %f1, %c0 : i32
  %f2 = arith.andi %flags, %c2 : i32
  %islast = arith.cmpi ne, %f2, %c0 : i32
  %none = arith.cmpi eq, %nt, %c0 : i32
  %zv = vector.from_elements %z, %z, %z, %z : vector<4xf32>
  %c64 = arith.constant 64 : i32
  %aa = arith.addi %sc, %c64 : i32
  "detour.store"(%aa, %zv) : (i32, vector<4xf32>) -> ()
  func.call @setloc(%sc) : (i32) -> ()
  %res:3 = scf.if %none -> (i32, i32, i32) {
    // a culled path without tiles: its background
    %bg:3 = func.call @emit_bg(%c0, %rows, %vw, %wb, %rows, %rule, %sout, %c0, %aout, %abase, %m1, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32, i32)
    scf.yield %bg#0, %bg#1, %bg#2 : i32, i32, i32
  } else {
    %t0:3 = func.call @tileat(%tb, %c0, %nt) : (i32, i32, i32) -> (i32, i32, i32)
    %y = arith.addi %t0#2, %c0 : i32
    %lv0 = arith.cmpi eq, %t0#1, %c0 : i32
    // its start: the background between the run before and it (from 0 for the first), the captive strip
    %st:4 = scf.if %culled -> (i32, i32, i32, i32) {
      %pv1 = arith.addi %prev, %c1 : i32
      %rmax = arith.minsi %y, %rows : i32
      %from = arith.select %isfirst, %c0, %pv1 : i32
      %to = arith.select %isfirst, %rmax, %y : i32
      %bg:3 = func.call @emit_bg(%from, %to, %vw, %wb, %rows, %rule, %sout, %c0, %aout, %abase, %m1, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32, i32)
      %c:4 = func.call @captive(%y, %lv0, %wb, %rows, %rule, %sc, %sout, %bg#0, %aout, %bg#1, %bg#2, %write) : (i32, i1, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32, i32, i32)
      func.call @setloc(%sc) : (i32) -> ()
      scf.yield %c#0, %c#1, %c#2, %c#3 : i32, i32, i32, i32
    } else {
      scf.yield %c0, %c0, %abase, %m1 : i32, i32, i32, i32
    }
    %fw = func.call @fillp(%st#0, %rule) : (i32, i32) -> i1
    %nlv = arith.xori %lv0, %true : i1
    %sf0 = arith.andi %fw, %nlv : i1
    %sx0 = arith.muli %t0#1, %c4 : i32
    %sy0 = arith.muli %y, %c4 : i32
    %sfi = arith.extui %sf0 : i1 to i32
    // the run's tiles: prev (lo, x), winding delta, strips, alphas, the strip (x, alpha index, fill gap)
    %lp:8 = scf.for %i = %c0 to %nt step %c1 iter_args(%plo = %t0#0, %px = %t0#1, %wd = %st#0, %ns = %st#1, %na = %st#2, %sx = %sx0, %sai = %st#2, %sfl = %sfi) -> (i32, i32, i32, i32, i32, i32, i32, i32) : i32 {
      %t:3 = func.call @tileat(%tb, %i, %nt) : (i32, i32, i32) -> (i32, i32, i32)
      %li = arith.shrui %t#0, %c5 : i32
      %lo16 = arith.muli %li, %c16 : i32
      %la = arith.addi %lb, %lo16 : i32
      %lv = "detour.load"(%la) : (i32) -> vector<4xf32>
      %txf0 = arith.sitofp %t#1 : i32 to f32
      %tyf0 = arith.sitofp %y : i32 to f32
      %four = arith.constant 4.0 : f32
      %tlx = arith.mulf %txf0, %four : f32
      %tty = arith.mulf %tyf0, %four : f32
      %x0 = vector.extract %lv[0] : f32 from vector<4xf32>
      %y0 = vector.extract %lv[1] : f32 from vector<4xf32>
      %x1 = vector.extract %lv[2] : f32 from vector<4xf32>
      %y1 = vector.extract %lv[3] : f32 from vector<4xf32>
      %q0x = arith.subf %x0, %tlx : f32
      %q0y = arith.subf %y0, %tty : f32
      %q1x = arith.subf %x1, %tlx : f32
      %q1y = arith.subf %y1, %tty : f32
      %samex = arith.cmpi eq, %px, %t#1 : i32
      %pxn = arith.addi %px, %c1 : i32
      %nextx = arith.cmpi eq, %pxn, %t#1 : i32
      %newloc = arith.xori %samex, %true : i1
      %na1 = scf.if %newloc -> (i32) {
        %n = func.call @finalize(%sc, %rule, %aout, %na, %write) : (i32, i32, i32, i32, i1) -> i32
        scf.yield %n : i32
      } else {
        scf.yield %na : i32
      }
      %np = arith.xori %nextx, %true : i1
      %newstrip = arith.andi %newloc, %np : i1
      %ss:4 = scf.if %newstrip -> (i32, i32, i32, i32) {
        %sfb = arith.cmpi ne, %sfl, %c0 : i32
        %ns1 = func.call @push(%sout, %ns, %sx, %sy0, %sai, %sfb, %write) : (i32, i32, i32, i32, i32, i1, i1) -> i32
        %wf = arith.sitofp %wd : i32 to f32
        %wv = vector.from_elements %wf, %wf, %wf, %wf : vector<4xf32>
        "detour.store"(%aa, %wv) : (i32, vector<4xf32>) -> ()
        %lvp = arith.cmpi eq, %t#1, %c0 : i32
        %fw3 = func.call @fillp(%wd, %rule) : (i32, i32) -> i1
        %nlv2 = arith.xori %lvp, %true : i1
        %sf2 = arith.andi %fw3, %nlv2 : i1
        %sfi2 = arith.extui %sf2 : i1 to i32
        %sx2 = arith.muli %t#1, %c4 : i32
        scf.yield %ns1, %sx2, %na1, %sfi2 : i32, i32, i32, i32
      } else {
        scf.yield %ns, %sx, %sai, %sfl : i32, i32, i32, i32
      }
      %flat = arith.cmpf oeq, %q0y, %q1y : f32
      %wd2 = scf.if %flat -> (i32) {
        scf.yield %wd : i32
      } else {
        %dd = arith.subf %q0y, %q1y : f32
        %up = arith.cmpf ogt, %dd, %z : f32
        %sgn = arith.select %up, %c1, %m1 : i32
        %wbit0 = arith.shrui %t#0, %c4 : i32
        %wbit = arith.andi %wbit0, %c1 : i32
        %dw = arith.muli %sgn, %wbit : i32
        %wn = arith.addi %wd, %dw : i32
        // the coverage only for the alphas (pass 0 counts: a location is 16 bytes whatever its coverage)
        scf.if %write {
          func.call @area(%sc, %q0x, %q0y, %q1x, %q1y) : (i32, f32, f32, f32, f32) -> ()
        }
        scf.yield %wn : i32
      }
      scf.yield %t#0, %t#1, %wd2, %ss#0, %na1, %ss#1, %ss#2, %ss#3 : i32, i32, i32, i32, i32, i32, i32, i32
    }
    // its end: the last location's alphas, the strip, the winding delta's strip at the viewport's right
    %naf = func.call @finalize(%sc, %rule, %aout, %lp#4, %write) : (i32, i32, i32, i32, i1) -> i32
    %sfb = arith.cmpi ne, %lp#7, %c0 : i32
    %ns1 = func.call @push(%sout, %lp#3, %lp#5, %sy0, %lp#6, %sfb, %write) : (i32, i32, i32, i32, i32, i1, i1) -> i32
    %wnz = arith.cmpi ne, %lp#2, %c0 : i32
    %ns2 = scf.if %wnz -> (i32) {
      %fw2 = func.call @fillp(%lp#2, %rule) : (i32, i32) -> i1
      %n2 = func.call @push(%sout, %ns1, %vw, %sy0, %naf, %fw2, %write) : (i32, i32, i32, i32, i32, i1, i1) -> i32
      scf.yield %n2 : i32
    } else {
      scf.yield %ns1 : i32
    }
    // the last run: the background below it
    %lastbg = arith.andi %islast, %culled : i1
    %e:3 = scf.if %lastbg -> (i32, i32, i32) {
      %y1 = arith.addi %y, %c1 : i32
      %ys = arith.minsi %y1, %rows : i32
      %bg:3 = func.call @emit_bg(%ys, %rows, %vw, %wb, %rows, %rule, %sout, %ns2, %aout, %naf, %sy0, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32, i32)
      scf.yield %bg#0, %bg#1, %bg#2 : i32, i32, i32
    } else {
      scf.yield %ns2, %naf, %sy0 : i32, i32, i32
    }
    scf.yield %e#0, %e#1, %e#2 : i32, i32, i32
  }
  // the last run (a path's only one when without tiles): the sentinel after the path's last strip
  %c7 = arith.constant 0 : i32
  %any = arith.cmpi sge, %res#2, %c7 : i32
  %sent = arith.andi %islast, %any : i1
  %fs = scf.if %sent -> (i32) {
    %m = arith.constant 65535 : i32
    %s = func.call @push(%sout, %res#0, %m, %res#2, %res#1, %false, %write) : (i32, i32, i32, i32, i32, i1, i1) -> i32
    scf.yield %s : i32
  } else {
    scf.yield %res#0 : i32
  }
  %nal = arith.subi %res#1, %abase : i32
  return %fs, %nal : i32, i32
}
