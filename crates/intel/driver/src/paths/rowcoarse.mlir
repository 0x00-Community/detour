// vello_cpu's coarse rasterization of plain fills on Xe3, a strip row a lane, bit for bit (roadmap R8.4, its
// first part): every path in draw order, its strips in this row (visit_strip_fill_segments: a strip's alpha fill,
// the gap to the next filled), a gap of an opaque fill split at the depth buckets (split_opaque_span: the whole
// buckets a depth command, the ends paint fills). params: strips (two u32 a strip: x | y << 16, alpha_idx |
// fill_gap << 31; a path's from its own alpha index 0, its sentinel last), paths (strips' first, count with the
// sentinel, alpha base, attrs, opaque: eight i32 a path), consts (columns, rows, paths, paths a chunk, chunks), counts
// (commands, depth commands: two i32 a lane), offsets (likewise), cmds (x | width << 16, alpha (-1 none), attrs, 0:
// 16 bytes), depth (start | end << 16, attrs: 8 bytes). A lane is a row (workgroup y) and a chunk of the paths
// (workgroup x 16 + lane; its counts at lane row chunks + chunk: plain fills keep no state from path to path, so a
// row's commands are its chunks' in turn). Pass 0 counts, pass 1 writes.
func.func @coarse(%params: i32, %nrows: i32, %pass: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %strips = "detour.load"(%params) : (i32) -> i32
  %a1 = arith.addi %params, %c4 : i32
  %paths = "detour.load"(%a1) : (i32) -> i32
  %a2 = arith.addi %a1, %c4 : i32
  %consts = "detour.load"(%a2) : (i32) -> i32
  %a3 = arith.addi %a2, %c4 : i32
  %counts = "detour.load"(%a3) : (i32) -> i32
  %a4 = arith.addi %a3, %c4 : i32
  %offsets = "detour.load"(%a4) : (i32) -> i32
  %a5 = arith.addi %a4, %c4 : i32
  %cmds = "detour.load"(%a5) : (i32) -> i32
  %a6 = arith.addi %a5, %c4 : i32
  %depth = "detour.load"(%a6) : (i32) -> i32
  %cols = "detour.load"(%consts) : (i32) -> i32
  %k1 = arith.addi %consts, %c4 : i32
  %k2 = arith.addi %k1, %c4 : i32
  %np = "detour.load"(%k2) : (i32) -> i32
  %k3 = arith.addi %k2, %c4 : i32
  %chunk = "detour.load"(%k3) : (i32) -> i32
  %k4 = arith.addi %k3, %c4 : i32
  %nchunks = "detour.load"(%k4) : (i32) -> i32
  %g = "detour.workgroup_id"() {dim = 0 : i32} : () -> i32
  %row = "detour.workgroup_id"() {dim = 1 : i32} : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %ci = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %ci, %nchunks : i32
  scf.if %inside {
    %rc = arith.muli %row, %nchunks : i32
    %lane = arith.addi %rc, %ci : i32
    %pfirst = arith.muli %ci, %chunk : i32
    %pend0 = arith.addi %pfirst, %chunk : i32
    %pend = arith.minui %pend0, %np : i32
    %write = arith.cmpi eq, %pass, %c1 : i32
    %r8 = arith.muli %lane, %c8 : i32
    %oa = arith.addi %offsets, %r8 : i32
    %ov = scf.if %write -> (vector<2xi32>) {
      %v = "detour.load"(%oa) : (i32) -> vector<2xi32>
      scf.yield %v : vector<2xi32>
    } else {
      %z = vector.from_elements %c0, %c0 : vector<2xi32>
      scf.yield %z : vector<2xi32>
    }
    %co = vector.extract %ov[0] : i32 from vector<2xi32>
    %do = vector.extract %ov[1] : i32 from vector<2xi32>
    %res:2 = scf.for %p = %pfirst to %pend step %c1 iter_args(%nc = %co, %nd = %do) -> (i32, i32) : i32 {
      %c32 = arith.constant 32 : i32
      %po = arith.muli %p, %c32 : i32
      %pa = arith.addi %paths, %po : i32
      %pv = "detour.load"(%pa) : (i32) -> vector<4xi32>
      %s0 = vector.extract %pv[0] : i32 from vector<4xi32>
      %ns = vector.extract %pv[1] : i32 from vector<4xi32>
      %ab = vector.extract %pv[2] : i32 from vector<4xi32>
      %at = vector.extract %pv[3] : i32 from vector<4xi32>
      %pa2 = arith.addi %pa, %c16 : i32
      %opq = "detour.load"(%pa2) : (i32) -> i32
      %c8s = arith.constant 8 : i32
      %sb0 = arith.muli %s0, %c8s : i32
      %sb = arith.addi %strips, %sb0 : i32
      // pairs (i, i + 1) for i in 0 .. ns - 1: the first whose strip is in this row or below
      %last = arith.subi %ns, %c1 : i32
      %first = func.call @lower_bound(%sb, %last, %row) : (i32, i32, i32) -> i32
      %r:3 = scf.for %i = %first to %last step %c1 iter_args(%c = %nc, %d = %nd, %go = %c1) -> (i32, i32, i32) : i32 {
        %alive = arith.cmpi ne, %go, %c0 : i32
        %o:3 = scf.if %alive -> (i32, i32, i32) {
          %s:4 = func.call @strip(%sb, %i) : (i32, i32) -> (i32, i32, i32, i32)
          %c2s = arith.constant 2 : i32
          %ty = arith.shrui %s#1, %c2s : i32
          %mine = arith.cmpi eq, %ty, %row : i32
          %q:3 = scf.if %mine -> (i32, i32, i32) {
            %in = arith.addi %i, %c1 : i32
            %t:4 = func.call @strip(%sb, %in) : (i32, i32) -> (i32, i32, i32, i32)
            %e:2 = func.call @pair(%s#0, %s#1, %s#2, %s#3, %t#0, %t#1, %t#2, %t#3, %row, %cols, %ab, %at, %opq, %cmds, %depth, %c, %d, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
            scf.yield %e#0, %e#1, %c1 : i32, i32, i32
          } else {
            scf.yield %c, %d, %c0 : i32, i32, i32
          }
          scf.yield %q#0, %q#1, %q#2 : i32, i32, i32
        } else {
          scf.yield %c, %d, %go : i32, i32, i32
        }
        scf.yield %o#0, %o#1, %o#2 : i32, i32, i32
      }
      scf.yield %r#0, %r#1 : i32, i32
    }
    %first0 = arith.cmpi eq, %pass, %c0 : i32
    scf.if %first0 {
      %ca = arith.addi %counts, %r8 : i32
      %cv = vector.from_elements %res#0, %res#1 : vector<2xi32>
      "detour.store"(%ca, %cv) : (i32, vector<2xi32>) -> ()
    }
  }
  return
}

// strip i of sb: x, y, alpha index, fill gap (0 / 1)
func.func private @strip(%sb: i32, %i: i32) -> (i32, i32, i32, i32) {
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %m = arith.constant 65535 : i32
  %c31 = arith.constant 31 : i32
  %mask = arith.constant 2147483647 : i32
  %o = arith.muli %i, %c8 : i32
  %a = arith.addi %sb, %o : i32
  %v = "detour.load"(%a) : (i32) -> vector<2xi32>
  %xy = vector.extract %v[0] : i32 from vector<2xi32>
  %pk = vector.extract %v[1] : i32 from vector<2xi32>
  %x = arith.andi %xy, %m : i32
  %y = arith.shrui %xy, %c16 : i32
  %ai = arith.andi %pk, %mask : i32
  %fg = arith.shrui %pk, %c31 : i32
  return %x, %y, %ai, %fg : i32, i32, i32, i32
}

// the first of strips 0 .. n whose strip row (y / 4) is row or more (n when none): a binary search (they are sorted)
func.func private @lower_bound(%sb: i32, %n: i32, %row: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %true = arith.constant true
  %r:3 = "scf.while"(%c0, %n, %true) ({
  ^bb0(%lw: i32, %hw: i32, %gw: i1):
    "scf.condition"(%gw, %lw, %hw, %gw) : (i1, i32, i32, i1) -> ()
  }, {
  ^bb0(%lo: i32, %hi: i32, %g0: i1):
    %more = arith.cmpi slt, %lo, %hi : i32
    %nx:3 = scf.if %more -> (i32, i32, i1) {
      %s = arith.addi %lo, %hi : i32
      %mid = arith.shrui %s, %c1 : i32
      %st:4 = func.call @strip(%sb, %mid) : (i32, i32) -> (i32, i32, i32, i32)
      %ty = arith.shrui %st#1, %c2 : i32
      %below = arith.cmpi slt, %ty, %row : i32
      %m1 = arith.addi %mid, %c1 : i32
      %lo2 = arith.select %below, %m1, %lo : i32
      %hi2 = arith.select %below, %hi, %mid : i32
      scf.yield %lo2, %hi2, %true : i32, i32, i1
    } else {
      %f = arith.constant false
      scf.yield %lo, %hi, %f : i32, i32, i1
    }
    "scf.yield"(%nx#0, %nx#1, %nx#2) : (i32, i32, i1) -> ()
  }) : (i32, i32, i1) -> (i32, i32, i1)
  return %r#0 : i32
}

// a paint fill command at cmds + 16 c (when writing): pixel span (tiles x0 .. x1), alpha (-1 none), attrs; c + 1
func.func private @cmd(%cmds: i32, %c: i32, %x0: i32, %x1: i32, %alpha: i32, %attrs: i32, %write: i1) -> i32 {
  scf.if %write {
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %px = arith.muli %x0, %c4 : i32
    %w0 = arith.subi %x1, %x0 : i32
    %pw = arith.muli %w0, %c4 : i32
    %ws = arith.shli %pw, %c16 : i32
    %xw = arith.ori %px, %ws : i32
    %z = arith.constant 0 : i32
    %o = arith.muli %c, %c16 : i32
    %a = arith.addi %cmds, %o : i32
    %v = vector.from_elements %xw, %alpha, %attrs, %z : vector<4xi32>
    "detour.store"(%a, %v) : (i32, vector<4xi32>) -> ()
  }
  %c1 = arith.constant 1 : i32
  %n = arith.addi %c, %c1 : i32
  return %n : i32
}

// a depth command at depth + 8 d (when writing): buckets start .. end, attrs; d + 1
func.func private @dcmd(%depth: i32, %d: i32, %start: i32, %end: i32, %attrs: i32, %write: i1) -> i32 {
  scf.if %write {
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %es = arith.shli %end, %c16 : i32
    %se = arith.ori %start, %es : i32
    %o = arith.muli %d, %c8 : i32
    %a = arith.addi %depth, %o : i32
    %v = vector.from_elements %se, %attrs : vector<2xi32>
    "detour.store"(%a, %v) : (i32, vector<2xi32>) -> ()
  }
  %c1 = arith.constant 1 : i32
  %n = arith.addi %d, %c1 : i32
  return %n : i32
}

// one pair of visit_strip_fill_segments within the row: the strip's alpha fill (clipped to the columns), the gap to
// the next filled when it is in the same row and fills it (an opaque fill's gap split at the depth buckets)
func.func private @pair(%x: i32, %y: i32, %ai: i32, %fg: i32, %nx: i32, %ny: i32, %nai: i32, %nfg: i32, %row: i32, %cols: i32, %ab: i32, %at: i32, %opq: i32, %cmds: i32, %depth: i32, %c: i32, %d: i32, %write: i1) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %m = arith.constant 65535 : i32
  // strip.width_to(next): next column - this column (alpha index / 4), saturating
  %col = arith.shrui %ai, %c2 : i32
  %ncol = arith.shrui %nai, %c2 : i32
  %wd0 = arith.subi %ncol, %col : i32
  %wd = arith.maxsi %wd0, %c0 : i32
  %wdm = arith.andi %wd, %m : i32
  %tx0 = arith.shrui %x, %c2 : i32
  %wt = arith.shrui %wdm, %c2 : i32
  %tx1a = arith.addi %tx0, %wt : i32
  %tx1 = arith.minui %tx1a, %m : i32
  %x0 = arith.maxui %tx0, %c0 : i32
  %x1 = arith.minui %tx1, %cols : i32
  %fills = arith.cmpi ult, %x0, %x1 : i32
  %c1 = scf.if %fills -> (i32) {
    %dx = arith.subi %x0, %tx0 : i32
    %d16 = arith.muli %dx, %c16 : i32
    %a0 = arith.addi %ab, %ai : i32
    %alpha = arith.addi %a0, %d16 : i32
    %n = func.call @cmd(%cmds, %c, %x0, %x1, %alpha, %at, %write) : (i32, i32, i32, i32, i32, i32, i1) -> i32
    scf.yield %n : i32
  } else {
    scf.yield %c : i32
  }
  %gap = arith.cmpi ne, %nfg, %c0 : i32
  %samey = arith.cmpi eq, %ny, %y : i32
  %g = arith.andi %gap, %samey : i1
  %r:2 = scf.if %g -> (i32, i32) {
    %gx0 = arith.maxui %tx1, %c0 : i32
    %nt = arith.shrui %nx, %c2 : i32
    %gx1 = arith.minui %nt, %cols : i32
    %some = arith.cmpi ult, %gx0, %gx1 : i32
    %q:2 = scf.if %some -> (i32, i32) {
      %isopq = arith.cmpi ne, %opq, %c0 : i32
      %w:2 = scf.if %isopq -> (i32, i32) {
        %v:2 = func.call @split(%gx0, %gx1, %at, %cmds, %depth, %c1, %d, %write) : (i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
        scf.yield %v#0, %v#1 : i32, i32
      } else {
        %m1 = arith.constant -1 : i32
        %n = func.call @cmd(%cmds, %c1, %gx0, %gx1, %m1, %at, %write) : (i32, i32, i32, i32, i32, i32, i1) -> i32
        scf.yield %n, %d : i32, i32
      }
      scf.yield %w#0, %w#1 : i32, i32
    } else {
      scf.yield %c1, %d : i32, i32
    }
    scf.yield %q#0, %q#1 : i32, i32
  } else {
    scf.yield %c1, %d : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}

// split_opaque_span(tiles x .. end): the whole depth buckets (32 tiles each) a depth command, the ends paint fills
func.func private @split(%x: i32, %end: i32, %at: i32, %cmds: i32, %depth: i32, %c: i32, %d: i32, %write: i1) -> (i32, i32) {
  %c31 = arith.constant 31 : i32
  %c5 = arith.constant 5 : i32
  %m1 = arith.constant -1 : i32
  %nm = arith.constant -32 : i32
  %xa0 = arith.addi %x, %c31 : i32
  %ax = arith.andi %xa0, %nm : i32
  %ae = arith.andi %end, %nm : i32
  %none = arith.cmpi uge, %ax, %ae : i32
  %r:2 = scf.if %none -> (i32, i32) {
    %n = func.call @cmd(%cmds, %c, %x, %end, %m1, %at, %write) : (i32, i32, i32, i32, i32, i32, i1) -> i32
    scf.yield %n, %d : i32, i32
  } else {
    %lead = arith.cmpi ult, %x, %ax : i32
    %c1 = scf.if %lead -> (i32) {
      %n = func.call @cmd(%cmds, %c, %x, %ax, %m1, %at, %write) : (i32, i32, i32, i32, i32, i32, i1) -> i32
      scf.yield %n : i32
    } else {
      scf.yield %c : i32
    }
    %bs = arith.shrui %ax, %c5 : i32
    %be = arith.shrui %ae, %c5 : i32
    %d1 = func.call @dcmd(%depth, %d, %bs, %be, %at, %write) : (i32, i32, i32, i32, i32, i1) -> i32
    %trail = arith.cmpi ult, %ae, %end : i32
    %c2 = scf.if %trail -> (i32) {
      %n = func.call @cmd(%cmds, %c1, %ae, %end, %m1, %at, %write) : (i32, i32, i32, i32, i32, i32, i1) -> i32
      scf.yield %n : i32
    } else {
      scf.yield %c1 : i32
    }
    scf.yield %c2, %d1 : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}

// @coarse_path, a path a lane: its strips once, its commands per row at the slots of a (row, path) table (rows x
// paths: a row's commands are its paths' in draw order, a prefix sum of the table row by row). params: strips, paths,
// consts (as @coarse's; then the sparse tables' arrays, @tab_at), ctab, dtab (pass 0: its commands' and depth
// commands' count in each row it reaches, at 4 (row paths + p), or its pair's), coffs, doffs (pass 1: those tables'
// prefix sums, in (row, path) order), cmds, depth.
func.func @coarse_path(%params: i32, %npaths: i32, %pass: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %strips = "detour.load"(%params) : (i32) -> i32
  %a1 = arith.addi %params, %c4 : i32
  %paths = "detour.load"(%a1) : (i32) -> i32
  %a2 = arith.addi %a1, %c4 : i32
  %consts = "detour.load"(%a2) : (i32) -> i32
  %a3 = arith.addi %a2, %c4 : i32
  %ctab = "detour.load"(%a3) : (i32) -> i32
  %a4 = arith.addi %a3, %c4 : i32
  %dtab = "detour.load"(%a4) : (i32) -> i32
  %a5 = arith.addi %a4, %c4 : i32
  %coffs = "detour.load"(%a5) : (i32) -> i32
  %a6 = arith.addi %a5, %c4 : i32
  %doffs = "detour.load"(%a6) : (i32) -> i32
  %a7 = arith.addi %a6, %c4 : i32
  %cmds = "detour.load"(%a7) : (i32) -> i32
  %a8 = arith.addi %a7, %c4 : i32
  %depth = "detour.load"(%a8) : (i32) -> i32
  %cols = "detour.load"(%consts) : (i32) -> i32
  %k1 = arith.addi %consts, %c4 : i32
  %rows = "detour.load"(%k1) : (i32) -> i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %p = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %p, %npaths : i32
  scf.if %inside {
    %write = arith.cmpi eq, %pass, %c1 : i32
    %c32 = arith.constant 32 : i32
    %po = arith.muli %p, %c32 : i32
    %pa = arith.addi %paths, %po : i32
    %pv = "detour.load"(%pa) : (i32) -> vector<4xi32>
    %s0 = vector.extract %pv[0] : i32 from vector<4xi32>
    %ns = vector.extract %pv[1] : i32 from vector<4xi32>
    %ab = vector.extract %pv[2] : i32 from vector<4xi32>
    %at = vector.extract %pv[3] : i32 from vector<4xi32>
    %pa2 = arith.addi %pa, %c16 : i32
    %opq = "detour.load"(%pa2) : (i32) -> i32
    %sb0 = arith.muli %s0, %c8 : i32
    %sb = arith.addi %strips, %sb0 : i32
    %last = arith.subi %ns, %c1 : i32
    %m1 = arith.constant -1 : i32
    // the row being done (-1 none yet), its commands' and depth commands' slots (pass 1) or counts (pass 0)
    %fin:3 = scf.for %i = %c0 to %last step %c1 iter_args(%cur = %m1, %c = %c0, %d = %c0) -> (i32, i32, i32) : i32 {
      %s:4 = func.call @strip(%sb, %i) : (i32, i32) -> (i32, i32, i32, i32)
      %ty = arith.shrui %s#1, %c2 : i32
      %inrows = arith.cmpi ult, %ty, %rows : i32
      %o:3 = scf.if %inrows -> (i32, i32, i32) {
        %newrow = arith.cmpi ne, %ty, %cur : i32
        // a new row: the last one's counts (pass 0); this one's slots (pass 1) or 0
        %st:2 = scf.if %newrow -> (i32, i32) {
          %done = arith.cmpi sge, %cur, %c0 : i32
          %true0 = arith.constant true
          %w0 = arith.xori %write, %true0 : i1
          %flush = arith.andi %done, %w0 : i1
          scf.if %flush {
            func.call @tab_put(%consts, %ctab, %dtab, %cur, %npaths, %p, %c, %d) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          }
          %n:2 = scf.if %write -> (i32, i32) {
            %v:2 = func.call @tab_get(%consts, %coffs, %doffs, %ty, %npaths, %p) : (i32, i32, i32, i32, i32, i32) -> (i32, i32)
            scf.yield %v#0, %v#1 : i32, i32
          } else {
            scf.yield %c0, %c0 : i32, i32
          }
          scf.yield %n#0, %n#1 : i32, i32
        } else {
          scf.yield %c, %d : i32, i32
        }
        %in = arith.addi %i, %c1 : i32
        %t:4 = func.call @strip(%sb, %in) : (i32, i32) -> (i32, i32, i32, i32)
        %e:2 = func.call @pair(%s#0, %s#1, %s#2, %s#3, %t#0, %t#1, %t#2, %t#3, %ty, %cols, %ab, %at, %opq, %cmds, %depth, %st#0, %st#1, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
        scf.yield %ty, %e#0, %e#1 : i32, i32, i32
      } else {
        scf.yield %cur, %c, %d : i32, i32, i32
      }
      scf.yield %o#0, %o#1, %o#2 : i32, i32, i32
    }
    %done = arith.cmpi sge, %fin#0, %c0 : i32
    %true = arith.constant true
    %w0 = arith.xori %write, %true : i1
    %flush = arith.andi %done, %w0 : i1
    scf.if %flush {
      func.call @tab_put(%consts, %ctab, %dtab, %fin#0, %npaths, %p, %fin#1, %fin#2) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    }
  }
  return
}

// a (row, path) entry's index: row paths + p, or (sparse: consts + 20 the paths' (first row, rows) (two i32 each),
// consts + 24 their first pairs) its pair's, path p's first pair + row - its first row; `sorted` (tab_get, consts +
// 28): the pair's place in (row, path) order, where the scanned tables have it
func.func private @tab_at(%consts: i32, %row: i32, %np: i32, %p: i32, %sorted: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c20 = arith.constant 20 : i32
  %c24 = arith.constant 24 : i32
  %c28 = arith.constant 28 : i32
  %pra = arith.addi %consts, %c20 : i32
  %prow = "detour.load"(%pra) : (i32) -> i32
  %sparse = arith.cmpi ne, %prow, %c0 : i32
  %k = scf.if %sparse -> (i32) {
    %p8 = arith.muli %p, %c8 : i32
    %ra = arith.addi %prow, %p8 : i32
    %r0 = "detour.load"(%ra) : (i32) -> i32
    %pba = arith.addi %consts, %c24 : i32
    %pbase = "detour.load"(%pba) : (i32) -> i32
    %p4 = arith.muli %p, %c4 : i32
    %ba = arith.addi %pbase, %p4 : i32
    %b = "detour.load"(%ba) : (i32) -> i32
    %dr = arith.subi %row, %r0 : i32
    %slot = arith.addi %b, %dr : i32
    %pos = scf.if %sorted -> (i32) {
      %sa0 = arith.addi %consts, %c28 : i32
      %sarr = "detour.load"(%sa0) : (i32) -> i32
      %s4 = arith.muli %slot, %c4 : i32
      %sa = arith.addi %sarr, %s4 : i32
      %v = "detour.load"(%sa) : (i32) -> i32
      scf.yield %v : i32
    } else {
      scf.yield %slot : i32
    }
    scf.yield %pos : i32
  } else {
    %rn = arith.muli %row, %np : i32
    %d = arith.addi %rn, %p : i32
    scf.yield %d : i32
  }
  return %k : i32
}

func.func private @tab_put(%consts: i32, %ctab: i32, %dtab: i32, %row: i32, %np: i32, %p: i32, %c: i32, %d: i32) {
  %c4 = arith.constant 4 : i32
  %f = arith.constant false
  %k = func.call @tab_at(%consts, %row, %np, %p, %f) : (i32, i32, i32, i32, i1) -> i32
  %o = arith.muli %k, %c4 : i32
  %ca = arith.addi %ctab, %o : i32
  %da = arith.addi %dtab, %o : i32
  "detour.store"(%ca, %c) : (i32, i32) -> ()
  "detour.store"(%da, %d) : (i32, i32) -> ()
  return
}

func.func private @tab_get(%consts: i32, %coffs: i32, %doffs: i32, %row: i32, %np: i32, %p: i32) -> (i32, i32) {
  %c4 = arith.constant 4 : i32
  %t = arith.constant true
  %k = func.call @tab_at(%consts, %row, %np, %p, %t) : (i32, i32, i32, i32, i1) -> i32
  %o = arith.muli %k, %c4 : i32
  %ca = arith.addi %coffs, %o : i32
  %da = arith.addi %doffs, %o : i32
  %c = "detour.load"(%ca) : (i32) -> i32
  %d = "detour.load"(%da) : (i32) -> i32
  return %c, %d : i32, i32
}
