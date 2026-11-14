// vello_cpu's coarse rasterization with layers (CommandBucketer: push_layer, pop_layer, ensure_row_layers,
// generate) on Xe3, an event a lane (after rowcoarse.mlir: @strip, @dcmd, @split, @tab_put, @tab_get). The scene's
// events in order: fills and layer pops (a layer's clip strips made at its push, before what it holds). A fill's
// segments are clipped to its innermost clip's box; inside a layer a fill gets no depth commands. A row's PushBuf of a
// layer goes right before the first command any fill inside it puts in that row (ensure_row_layers: the outermost
// first); the first such fill of each (row, layer): pass 2, an atomic min into first (rows x layers, -1 none). A pop
// puts in each row the layer occupies its LayerFills (without a clip one over the box's span; with one the clip
// strips' segments, clipped to the parent's box) and a PopBuf. Commands (16 bytes): x | width << 16, alpha (-1 none),
// attrs, kind (0 paint fill, 1 layer fill, 2 push, 3 pop).
// params: strips, events (eight i32: kind (0 fill, 1 pop), first strip, strips with the sentinel, alpha base, attrs,
// opaque, layer (a fill's innermost, -1 none; a pop's own), 0), layers (eight i32: parent (-1 none), its box in tiles
// x0, y0, x1, y1 (its clip's and its parent's intersection; without a clip its parent's), 0, 0, 0), consts (columns,
// rows, events, layers, 0, then @tab_at's), first, ctab, dtab, coffs, doffs, cmds, depth. Pass 0 counts, 1 writes.

func.func private @cmdk(%cmds: i32, %c: i32, %x0: i32, %x1: i32, %alpha: i32, %attrs: i32, %kind: i32, %write: i1) -> i32 {
  scf.if %write {
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %px = arith.muli %x0, %c4 : i32
    %w0 = arith.subi %x1, %x0 : i32
    %pw = arith.muli %w0, %c4 : i32
    %ws = arith.shli %pw, %c16 : i32
    %xw = arith.ori %px, %ws : i32
    %o = arith.muli %c, %c16 : i32
    %a = arith.addi %cmds, %o : i32
    %v = vector.from_elements %xw, %alpha, %attrs, %kind : vector<4xi32>
    "detour.store"(%a, %v) : (i32, vector<4xi32>) -> ()
  }
  %c1 = arith.constant 1 : i32
  %n = arith.addi %c, %c1 : i32
  return %n : i32
}

// @pair within tiles bx0 .. bx1 (visit_strip_fill_segments' bounds: an alpha fill's index moved past the tiles cut),
// its commands of kind `kind`
func.func private @pair_b(%x: i32, %y: i32, %ai: i32, %nx: i32, %ny: i32, %nai: i32, %nfg: i32, %bx0: i32, %bx1: i32, %ab: i32, %at: i32, %opq: i32, %kind: i32, %cmds: i32, %depth: i32, %c: i32, %d: i32, %write: i1) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %c16 = arith.constant 16 : i32
  %m = arith.constant 65535 : i32
  %col = arith.shrui %ai, %c2 : i32
  %ncol = arith.shrui %nai, %c2 : i32
  %wd0 = arith.subi %ncol, %col : i32
  %wd = arith.maxsi %wd0, %c0 : i32
  %wdm = arith.andi %wd, %m : i32
  %tx0 = arith.shrui %x, %c2 : i32
  %wt = arith.shrui %wdm, %c2 : i32
  %tx1a = arith.addi %tx0, %wt : i32
  %tx1 = arith.minui %tx1a, %m : i32
  %x0 = arith.maxui %tx0, %bx0 : i32
  %x1 = arith.minui %tx1, %bx1 : i32
  %fills = arith.cmpi ult, %x0, %x1 : i32
  %c1 = scf.if %fills -> (i32) {
    %dx = arith.subi %x0, %tx0 : i32
    %d16 = arith.muli %dx, %c16 : i32
    %a0 = arith.addi %ab, %ai : i32
    %alpha = arith.addi %a0, %d16 : i32
    %n = func.call @cmdk(%cmds, %c, %x0, %x1, %alpha, %at, %kind, %write) : (i32, i32, i32, i32, i32, i32, i32, i1) -> i32
    scf.yield %n : i32
  } else {
    scf.yield %c : i32
  }
  %gap = arith.cmpi ne, %nfg, %c0 : i32
  %samey = arith.cmpi eq, %ny, %y : i32
  %g = arith.andi %gap, %samey : i1
  %r:2 = scf.if %g -> (i32, i32) {
    %gx0 = arith.maxui %tx1, %bx0 : i32
    %nt = arith.shrui %nx, %c2 : i32
    %gx1 = arith.minui %nt, %bx1 : i32
    %some = arith.cmpi ult, %gx0, %gx1 : i32
    %q:2 = scf.if %some -> (i32, i32) {
      %isopq = arith.cmpi ne, %opq, %c0 : i32
      %w:2 = scf.if %isopq -> (i32, i32) {
        %v:2 = func.call @split(%gx0, %gx1, %at, %cmds, %depth, %c1, %d, %write) : (i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
        scf.yield %v#0, %v#1 : i32, i32
      } else {
        %m1 = arith.constant -1 : i32
        %n = func.call @cmdk(%cmds, %c1, %gx0, %gx1, %m1, %at, %kind, %write) : (i32, i32, i32, i32, i32, i32, i32, i1) -> i32
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

// layer M's parent and box (tiles); the viewport's (no parent) for M = -1
func.func private @layer(%layers: i32, %cols: i32, %rows: i32, %lm: i32) -> (i32, i32, i32, i32, i32) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %m1 = arith.constant -1 : i32
  %none = arith.cmpi slt, %lm, %c0 : i32
  %r:5 = scf.if %none -> (i32, i32, i32, i32, i32) {
    scf.yield %m1, %c0, %c0, %cols, %rows : i32, i32, i32, i32, i32
  } else {
    %o = arith.muli %lm, %c32 : i32
    %a = arith.addi %layers, %o : i32
    %v = "detour.load"(%a) : (i32) -> vector<4xi32>
    %p = vector.extract %v[0] : i32 from vector<4xi32>
    %x0 = vector.extract %v[1] : i32 from vector<4xi32>
    %y0 = vector.extract %v[2] : i32 from vector<4xi32>
    %x1 = vector.extract %v[3] : i32 from vector<4xi32>
    %c16 = arith.constant 16 : i32
    %a2 = arith.addi %a, %c16 : i32
    %y1 = "detour.load"(%a2) : (i32) -> i32
    scf.yield %p, %x0, %y0, %x1, %y1 : i32, i32, i32, i32, i32
  }
  return %r#0, %r#1, %r#2, %r#3, %r#4 : i32, i32, i32, i32, i32
}

// M's k-th ancestor (k = 0: M)
func.func private @ancestor(%layers: i32, %lm: i32, %k: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  %a = scf.for %i = %c0 to %k step %c1 iter_args(%m = %lm) -> (i32) : i32 {
    %o = arith.muli %m, %c32 : i32
    %at = arith.addi %layers, %o : i32
    %p = "detour.load"(%at) : (i32) -> i32
    scf.yield %p : i32
  }
  return %a : i32
}

// how many layers M is inside, itself too (0: none)
func.func private @depth_of(%layers: i32, %lm: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c32 = arith.constant 32 : i32
  %true = arith.constant true
  %r:2 = "scf.while"(%lm, %c0) ({
  ^bb0(%m: i32, %n: i32):
    %more = arith.cmpi sge, %m, %c0 : i32
    "scf.condition"(%more, %m, %n) : (i1, i32, i32) -> ()
  }, {
  ^bb0(%m: i32, %n: i32):
    %o = arith.muli %m, %c32 : i32
    %at = arith.addi %layers, %o : i32
    %p = "detour.load"(%at) : (i32) -> i32
    %n1 = arith.addi %n, %c1 : i32
    "scf.yield"(%p, %n1) : (i32, i32) -> ()
  }) : (i32, i32) -> (i32, i32)
  return %r#1 : i32
}

// first + 4 (row layers + M)
func.func private @first_at(%first: i32, %row: i32, %nl: i32, %lm: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %rn = arith.muli %row, %nl : i32
  %k = arith.addi %rn, %lm : i32
  %o = arith.muli %k, %c4 : i32
  %a = arith.addi %first, %o : i32
  return %a : i32
}

// the row's PushBufs this fill (event k, innermost layer M, `depth` deep) makes: of the layers it is inside whose first
// fill in the row it is, outermost first; c + their count
func.func private @pushes(%cmds: i32, %c: i32, %first: i32, %row: i32, %nl: i32, %k: i32, %layers: i32, %lm: i32, %depth: i32, %write: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %m1 = arith.constant -1 : i32
  %r = scf.for %i = %c0 to %depth step %c1 iter_args(%cc = %c) -> (i32) : i32 {
    %up0 = arith.subi %depth, %i : i32
    %up = arith.subi %up0, %c1 : i32
    %a = func.call @ancestor(%layers, %lm, %up) : (i32, i32, i32) -> i32
    %fa = func.call @first_at(%first, %row, %nl, %a) : (i32, i32, i32, i32) -> i32
    %f = "detour.load"(%fa) : (i32) -> i32
    %mine = arith.cmpi eq, %f, %k : i32
    %n = scf.if %mine -> (i32) {
      %v = func.call @cmdk(%cmds, %cc, %c0, %c0, %m1, %c0, %c2, %write) : (i32, i32, i32, i32, i32, i32, i32, i1) -> i32
      scf.yield %v : i32
    } else {
      scf.yield %cc : i32
    }
    scf.yield %n : i32
  }
  return %r : i32
}

// the fill's first fill marks for the row: each layer it is inside, first[row][layer] = min(.., k)
func.func private @mark(%first: i32, %row: i32, %nl: i32, %k: i32, %layers: i32, %lm: i32, %depth: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %depth step %c1 : i32 {
    %a = func.call @ancestor(%layers, %lm, %i) : (i32, i32, i32) -> i32
    %fa = func.call @first_at(%first, %row, %nl, %a) : (i32, i32, i32, i32) -> i32
    %o = "detour.atomic"(%fa, %k) {op = "umin"} : (i32, i32) -> i32
  }
  return
}

func.func @coarse_ev(%params: i32, %nev: i32, %pass: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c12 = arith.constant 12 : i32
  %c16 = arith.constant 16 : i32
  %c32 = arith.constant 32 : i32
  %m1 = arith.constant -1 : i32
  %true = arith.constant true
  %false = arith.constant false
  %strips = "detour.load"(%params) : (i32) -> i32
  %a1 = arith.addi %params, %c4 : i32
  %events = "detour.load"(%a1) : (i32) -> i32
  %a2 = arith.addi %a1, %c4 : i32
  %layers = "detour.load"(%a2) : (i32) -> i32
  %a3 = arith.addi %a2, %c4 : i32
  %consts = "detour.load"(%a3) : (i32) -> i32
  %a4 = arith.addi %a3, %c4 : i32
  %first = "detour.load"(%a4) : (i32) -> i32
  %a5 = arith.addi %a4, %c4 : i32
  %ctab = "detour.load"(%a5) : (i32) -> i32
  %a6 = arith.addi %a5, %c4 : i32
  %dtab = "detour.load"(%a6) : (i32) -> i32
  %a7 = arith.addi %a6, %c4 : i32
  %coffs = "detour.load"(%a7) : (i32) -> i32
  %a8 = arith.addi %a7, %c4 : i32
  %doffs = "detour.load"(%a8) : (i32) -> i32
  %a9 = arith.addi %a8, %c4 : i32
  %cmds = "detour.load"(%a9) : (i32) -> i32
  %a10 = arith.addi %a9, %c4 : i32
  %depth = "detour.load"(%a10) : (i32) -> i32
  %cols = "detour.load"(%consts) : (i32) -> i32
  %k1 = arith.addi %consts, %c4 : i32
  %rows = "detour.load"(%k1) : (i32) -> i32
  %k3 = arith.addi %consts, %c12 : i32
  %nl = "detour.load"(%k3) : (i32) -> i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %k = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %k, %nev : i32
  scf.if %inside {
    %write = arith.cmpi eq, %pass, %c1 : i32
    %marking = arith.cmpi eq, %pass, %c2 : i32
    %eo = arith.muli %k, %c32 : i32
    %ea = arith.addi %events, %eo : i32
    %ev = "detour.load"(%ea) : (i32) -> vector<4xi32>
    %kind = vector.extract %ev[0] : i32 from vector<4xi32>
    %s0 = vector.extract %ev[1] : i32 from vector<4xi32>
    %ns = vector.extract %ev[2] : i32 from vector<4xi32>
    %ab = vector.extract %ev[3] : i32 from vector<4xi32>
    %ea2 = arith.addi %ea, %c16 : i32
    %ev2 = "detour.load"(%ea2) : (i32) -> vector<4xi32>
    %at = vector.extract %ev2[0] : i32 from vector<4xi32>
    %opq = vector.extract %ev2[1] : i32 from vector<4xi32>
    %lm = vector.extract %ev2[2] : i32 from vector<4xi32>
    %sb0 = arith.muli %s0, %c8 : i32
    %sb = arith.addi %strips, %sb0 : i32
    %last = arith.subi %ns, %c1 : i32
    %isfill = arith.cmpi eq, %kind, %c0 : i32
    scf.if %isfill {
      // a fill: its strips' segments within its innermost clip's box, row by row
      %bx:5 = func.call @layer(%layers, %cols, %rows, %lm) : (i32, i32, i32, i32) -> (i32, i32, i32, i32, i32)
      %dep = func.call @depth_of(%layers, %lm) : (i32, i32) -> i32
      %fin:4 = scf.for %i = %c0 to %last step %c1 iter_args(%cur = %m1, %c = %c0, %d = %c0, %open = %false) -> (i32, i32, i32, i1) : i32 {
        %s:4 = func.call @strip(%sb, %i) : (i32, i32) -> (i32, i32, i32, i32)
        %ty = arith.shrui %s#1, %c2 : i32
        %ge = arith.cmpi uge, %ty, %bx#2 : i32
        %lt = arith.cmpi ult, %ty, %bx#4 : i32
        %inr = arith.andi %ge, %lt : i1
        %o:4 = scf.if %inr -> (i32, i32, i32, i1) {
          %newrow = arith.cmpi ne, %ty, %cur : i32
          %st:3 = scf.if %newrow -> (i32, i32, i1) {
            %done = arith.cmpi sge, %cur, %c0 : i32
            %count = arith.cmpi eq, %pass, %c0 : i32
            %flush = arith.andi %done, %count : i1
            scf.if %flush {
              func.call @tab_put(%consts, %ctab, %dtab, %cur, %nev, %k, %c, %d) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            }
            %n:2 = scf.if %write -> (i32, i32) {
              %v:2 = func.call @tab_get(%consts, %coffs, %doffs, %ty, %nev, %k) : (i32, i32, i32, i32, i32, i32) -> (i32, i32)
              scf.yield %v#0, %v#1 : i32, i32
            } else {
              scf.yield %c0, %c0 : i32, i32
            }
            scf.yield %n#0, %n#1, %false : i32, i32, i1
          } else {
            scf.yield %c, %d, %open : i32, i32, i1
          }
          %in = arith.addi %i, %c1 : i32
          %t:4 = func.call @strip(%sb, %in) : (i32, i32) -> (i32, i32, i32, i32)
          // whether the pair makes anything: the row's layers pushed (marked) first
          %dry:2 = func.call @pair_b(%s#0, %s#1, %s#2, %t#0, %t#1, %t#2, %t#3, %bx#1, %bx#3, %ab, %at, %opq, %c0, %cmds, %depth, %c0, %c0, %false) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
          %any0 = arith.addi %dry#0, %dry#1 : i32
          %any = arith.cmpi ne, %any0, %c0 : i32
          %notopen = arith.xori %st#2, %true : i1
          %opening = arith.andi %any, %notopen : i1
          %cp = scf.if %opening -> (i32) {
            scf.if %marking {
              func.call @mark(%first, %ty, %nl, %k, %layers, %lm, %dep) : (i32, i32, i32, i32, i32, i32, i32) -> ()
            }
            %v = func.call @pushes(%cmds, %st#0, %first, %ty, %nl, %k, %layers, %lm, %dep, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> i32
            scf.yield %v : i32
          } else {
            scf.yield %st#0 : i32
          }
          %open2 = arith.ori %st#2, %any : i1
          %e:2 = func.call @pair_b(%s#0, %s#1, %s#2, %t#0, %t#1, %t#2, %t#3, %bx#1, %bx#3, %ab, %at, %opq, %c0, %cmds, %depth, %cp, %st#1, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
          scf.yield %ty, %e#0, %e#1, %open2 : i32, i32, i32, i1
        } else {
          scf.yield %cur, %c, %d, %open : i32, i32, i32, i1
        }
        scf.yield %o#0, %o#1, %o#2, %o#3 : i32, i32, i32, i1
      }
      %done = arith.cmpi sge, %fin#0, %c0 : i32
      %count = arith.cmpi eq, %pass, %c0 : i32
      %flush = arith.andi %done, %count : i1
      scf.if %flush {
        func.call @tab_put(%consts, %ctab, %dtab, %fin#0, %nev, %k, %fin#1, %fin#2) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      }
    } else {
      // a pop of layer M: each row it occupies its layer fills and the pop
      %notmark = arith.xori %marking, %true : i1
      scf.if %notmark {
        %bx:5 = func.call @layer(%layers, %cols, %rows, %lm) : (i32, i32, i32, i32) -> (i32, i32, i32, i32, i32)
        %px:5 = func.call @layer(%layers, %cols, %rows, %bx#0) : (i32, i32, i32, i32) -> (i32, i32, i32, i32, i32)
        %clipped = arith.cmpi sge, %ns, %c2 : i32
        scf.for %y = %bx#2 to %bx#4 step %c1 : i32 {
          %fa = func.call @first_at(%first, %y, %nl, %lm) : (i32, i32, i32, i32) -> i32
          %f = "detour.load"(%fa) : (i32) -> i32
          %occ = arith.cmpi ne, %f, %m1 : i32
          scf.if %occ {
            %cd:2 = scf.if %write -> (i32, i32) {
              %v:2 = func.call @tab_get(%consts, %coffs, %doffs, %y, %nev, %k) : (i32, i32, i32, i32, i32, i32) -> (i32, i32)
              scf.yield %v#0, %v#1 : i32, i32
            } else {
              scf.yield %c0, %c0 : i32, i32
            }
            %cf = scf.if %clipped -> (i32) {
              // the clip strips of row y: lower_bound(y) .. lower_bound(y + 1), within the parent's box
              %lo = func.call @lower_bound(%sb, %last, %y) : (i32, i32, i32) -> i32
              %y1 = arith.addi %y, %c1 : i32
              %hi = func.call @lower_bound(%sb, %last, %y1) : (i32, i32, i32) -> i32
              %r = scf.for %i = %lo to %hi step %c1 iter_args(%c = %cd#0) -> (i32) : i32 {
                %s:4 = func.call @strip(%sb, %i) : (i32, i32) -> (i32, i32, i32, i32)
                %in = arith.addi %i, %c1 : i32
                %t:4 = func.call @strip(%sb, %in) : (i32, i32) -> (i32, i32, i32, i32)
                %e:2 = func.call @pair_b(%s#0, %s#1, %s#2, %t#0, %t#1, %t#2, %t#3, %px#1, %px#3, %ab, %at, %c0, %c1, %cmds, %depth, %c, %c0, %write) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1) -> (i32, i32)
                scf.yield %e#0 : i32
              }
              scf.yield %r : i32
            } else {
              %v = func.call @cmdk(%cmds, %cd#0, %bx#1, %bx#3, %m1, %at, %c1, %write) : (i32, i32, i32, i32, i32, i32, i32, i1) -> i32
              scf.yield %v : i32
            }
            %cpop = func.call @cmdk(%cmds, %cf, %c0, %c0, %m1, %c0, %c3, %write) : (i32, i32, i32, i32, i32, i32, i32, i1) -> i32
            %count = arith.cmpi eq, %pass, %c0 : i32
            scf.if %count {
              func.call @tab_put(%consts, %ctab, %dtab, %y, %nev, %k, %cpop, %c0) : (i32, i32, i32, i32, i32, i32, i32, i32) -> ()
            }
          }
        }
      }
    }
  }
  return
}
