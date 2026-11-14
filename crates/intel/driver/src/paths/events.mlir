// The scene's events and layers for rowcoarse/layers.mlir, made on the GPU from the paths' strips (the pipeline's:
// rowcoarse's path table, `cpaths`, eight i32 a path: first strip, strips with the sentinel, alpha base, ...). The
// CPU gives the scene's order: items (four i32: kind (0 fill, 1 pop), its path (a fill's; a pop's clip, -1 none), its
// layer (a fill's innermost, -1 none; a pop's own), opaque), layers (eight i32: parent, then the box (@lbox), then
// the clip path (-1 none) at word 5). params: cpaths, items, layers, flags (an item a fill with strips), attrs (their
// prefix sum: a fill's paint attrs), events (layers.mlir's eight i32), prow, prcount (an event's first row and rows:
// a fill's strips' rows within its box, a pop's box's), strips, consts (columns, rows), own (four u32 a layer: its
// clip's box in tiles, strip_bbox's; the viewport without). @evflags, @evbuild: an item a lane; @lclip, @lbox: a layer.

func.func private @lane() -> i32 {
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %i = arith.addi %b0, %l : i32
  return %i : i32
}

func.func private @param(%params: i32, %k: i32) -> i32 {
  %c4 = arith.constant 4 : i32
  %o = arith.muli %k, %c4 : i32
  %a = arith.addi %params, %o : i32
  %v = "detour.load"(%a) : (i32) -> i32
  return %v : i32
}

// path p's strips: first, count with the sentinel, alpha base (none: p < 0)
func.func private @path(%cpaths: i32, %p: i32) -> (i32, i32, i32) {
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %some = arith.cmpi sge, %p, %c0 : i32
  %r:3 = scf.if %some -> (i32, i32, i32) {
    %o = arith.muli %p, %c32 : i32
    %a = arith.addi %cpaths, %o : i32
    %v = "detour.load"(%a) : (i32) -> vector<4xi32>
    %s0 = vector.extract %v[0] : i32 from vector<4xi32>
    %ns = vector.extract %v[1] : i32 from vector<4xi32>
    %ab = vector.extract %v[2] : i32 from vector<4xi32>
    scf.yield %s0, %ns, %ab : i32, i32, i32
  } else {
    scf.yield %c0, %c0, %c0 : i32, i32, i32
  }
  return %r#0, %r#1, %r#2 : i32, i32, i32
}

// item i: kind, path, layer, opaque
func.func private @item(%items: i32, %i: i32) -> (i32, i32, i32, i32) {
  %c16 = arith.constant 16 : i32
  %o = arith.muli %i, %c16 : i32
  %a = arith.addi %items, %o : i32
  %v = "detour.load"(%a) : (i32) -> vector<4xi32>
  %k = vector.extract %v[0] : i32 from vector<4xi32>
  %p = vector.extract %v[1] : i32 from vector<4xi32>
  %l = vector.extract %v[2] : i32 from vector<4xi32>
  %q = vector.extract %v[3] : i32 from vector<4xi32>
  return %k, %p, %l, %q : i32, i32, i32, i32
}

func.func @evflags(%params: i32, %n: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %i = func.call @lane() : () -> i32
  %inside = arith.cmpi slt, %i, %n : i32
  scf.if %inside {
    %cpaths = func.call @param(%params, %c0) : (i32, i32) -> i32
    %items = func.call @param(%params, %c1) : (i32, i32) -> i32
    %flags = func.call @param(%params, %c3) : (i32, i32) -> i32
    %it:4 = func.call @item(%items, %i) : (i32, i32) -> (i32, i32, i32, i32)
    %fill = arith.cmpi eq, %it#0, %c0 : i32
    %pp = arith.select %fill, %it#1, %c0 : i32
    %pth:3 = func.call @path(%cpaths, %it#1) : (i32, i32) -> (i32, i32, i32)
    %has = arith.cmpi ne, %pth#1, %c0 : i32
    %both = arith.andi %fill, %has : i1
    %f = arith.select %both, %c1, %c0 : i32
    %i4 = arith.muli %i, %c4 : i32
    %fa = arith.addi %flags, %i4 : i32
    "detour.store"(%fa, %f) : (i32, i32) -> ()
  }
  return
}

// @lclip: layer M's own box: its clip strips' segments' tiles (strip_bbox; (0, 0, 0, 0) when none), or the viewport
func.func @lclip(%params: i32, %nl: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c9 = arith.constant 9 : i32
  %c10 = arith.constant 10 : i32
  %c16 = arith.constant 16 : i32
  %c20 = arith.constant 20 : i32
  %c32 = arith.constant 32 : i32
  %m = arith.constant 65535 : i32
  %big = arith.constant 65535 : i32
  %lm = func.call @lane() : () -> i32
  %inside = arith.cmpi slt, %lm, %nl : i32
  scf.if %inside {
    %cpaths = func.call @param(%params, %c0) : (i32, i32) -> i32
    %layers = func.call @param(%params, %c2) : (i32, i32) -> i32
    %strips = func.call @param(%params, %c8) : (i32, i32) -> i32
    %consts = func.call @param(%params, %c9) : (i32, i32) -> i32
    %own = func.call @param(%params, %c10) : (i32, i32) -> i32
    %cols = "detour.load"(%consts) : (i32) -> i32
    %ra = arith.addi %consts, %c4 : i32
    %rows = "detour.load"(%ra) : (i32) -> i32
    %lo = arith.muli %lm, %c32 : i32
    %la = arith.addi %layers, %lo : i32
    %cpa = arith.addi %la, %c20 : i32
    %cp = "detour.load"(%cpa) : (i32) -> i32
    %clipped = arith.cmpi sge, %cp, %c0 : i32
    %b:4 = scf.if %clipped -> (i32, i32, i32, i32) {
      %pth:3 = func.call @path(%cpaths, %cp) : (i32, i32) -> (i32, i32, i32)
      %sb0 = arith.muli %pth#0, %c8 : i32
      %sb = arith.addi %strips, %sb0 : i32
      %last0 = arith.subi %pth#1, %c1 : i32
      %last = arith.maxsi %last0, %c0 : i32
      // the union of the segments' tile rects, from inverted
      %r:4 = scf.for %i = %c0 to %last step %c1 iter_args(%x0 = %big, %y0 = %big, %x1 = %c0, %y1 = %c0) -> (i32, i32, i32, i32) : i32 {
        %o = arith.muli %i, %c8 : i32
        %a = arith.addi %sb, %o : i32
        %v = "detour.load"(%a) : (i32) -> vector<2xi32>
        %xy = vector.extract %v[0] : i32 from vector<2xi32>
        %pk = vector.extract %v[1] : i32 from vector<2xi32>
        %an = arith.addi %a, %c8 : i32
        %vn = "detour.load"(%an) : (i32) -> vector<2xi32>
        %nxy = vector.extract %vn[0] : i32 from vector<2xi32>
        %npk = vector.extract %vn[1] : i32 from vector<2xi32>
        %x = arith.andi %xy, %m : i32
        %y = arith.shrui %xy, %c16 : i32
        %ty = arith.shrui %y, %c2 : i32
        %nx = arith.andi %nxy, %m : i32
        %ny = arith.shrui %nxy, %c16 : i32
        %mask = arith.constant 2147483647 : i32
        %ai = arith.andi %pk, %mask : i32
        %nai = arith.andi %npk, %mask : i32
        %c31 = arith.constant 31 : i32
        %nfg = arith.shrui %npk, %c31 : i32
        // the alpha segment: strip.x / 4 .. + width / 4 (the width as visit_strip_fill_segments' width_to)
        %col = arith.shrui %ai, %c2 : i32
        %ncol = arith.shrui %nai, %c2 : i32
        %wd0 = arith.subi %ncol, %col : i32
        %wd = arith.maxsi %wd0, %c0 : i32
        %wdm = arith.andi %wd, %m : i32
        %tx0 = arith.shrui %x, %c2 : i32
        %wt = arith.shrui %wdm, %c2 : i32
        %tx1a = arith.addi %tx0, %wt : i32
        %tx1 = arith.minui %tx1a, %m : i32
        %af = arith.cmpi ult, %tx0, %tx1 : i32
        %ty1 = arith.addi %ty, %c1 : i32
        %ax0 = arith.minui %x0, %tx0 : i32
        %ay0 = arith.minui %y0, %ty : i32
        %ax1 = arith.maxui %x1, %tx1 : i32
        %ay1 = arith.maxui %y1, %ty1 : i32
        %bx0 = arith.select %af, %ax0, %x0 : i32
        %by0 = arith.select %af, %ay0, %y0 : i32
        %bx1 = arith.select %af, %ax1, %x1 : i32
        %by1 = arith.select %af, %ay1, %y1 : i32
        // the gap fill: tx1 .. next.x / 4 when the next fills it in the same row
        %gap = arith.cmpi ne, %nfg, %c0 : i32
        %samey = arith.cmpi eq, %ny, %y : i32
        %g0 = arith.andi %gap, %samey : i1
        %gx1 = arith.shrui %nx, %c2 : i32
        %gs = arith.cmpi ult, %tx1, %gx1 : i32
        %g = arith.andi %g0, %gs : i1
        %cx0 = arith.minui %bx0, %tx1 : i32
        %cy0 = arith.minui %by0, %ty : i32
        %cx1 = arith.maxui %bx1, %gx1 : i32
        %cy1 = arith.maxui %by1, %ty1 : i32
        %ex0 = arith.select %g, %cx0, %bx0 : i32
        %ey0 = arith.select %g, %cy0, %by0 : i32
        %ex1 = arith.select %g, %cx1, %bx1 : i32
        %ey1 = arith.select %g, %cy1, %by1 : i32
        scf.yield %ex0, %ey0, %ex1, %ey1 : i32, i32, i32, i32
      }
      %empty0 = arith.cmpi uge, %r#0, %r#2 : i32
      %empty1 = arith.cmpi uge, %r#1, %r#3 : i32
      %empty = arith.ori %empty0, %empty1 : i1
      %z0 = arith.select %empty, %c0, %r#0 : i32
      %z1 = arith.select %empty, %c0, %r#1 : i32
      %z2 = arith.select %empty, %c0, %r#2 : i32
      %z3 = arith.select %empty, %c0, %r#3 : i32
      scf.yield %z0, %z1, %z2, %z3 : i32, i32, i32, i32
    } else {
      scf.yield %c0, %c0, %cols, %rows : i32, i32, i32, i32
    }
    %oo = arith.muli %lm, %c16 : i32
    %oa = arith.addi %own, %oo : i32
    %v = vector.from_elements %b#0, %b#1, %b#2, %b#3 : vector<4xi32>
    "detour.store"(%oa, %v) : (i32, vector<4xi32>) -> ()
  }
  return
}

// @lbox: layer M's box: its own and its ancestors' intersected, within the viewport
func.func @lbox(%params: i32, %nl: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c9 = arith.constant 9 : i32
  %c10 = arith.constant 10 : i32
  %c16 = arith.constant 16 : i32
  %c32 = arith.constant 32 : i32
  %lm = func.call @lane() : () -> i32
  %inside = arith.cmpi slt, %lm, %nl : i32
  scf.if %inside {
    %layers = func.call @param(%params, %c2) : (i32, i32) -> i32
    %consts = func.call @param(%params, %c9) : (i32, i32) -> i32
    %own = func.call @param(%params, %c10) : (i32, i32) -> i32
    %cols = "detour.load"(%consts) : (i32) -> i32
    %ra = arith.addi %consts, %c4 : i32
    %rows = "detour.load"(%ra) : (i32) -> i32
    %r:5 = "scf.while"(%lm, %c0, %c0, %cols, %rows) ({
    ^bb0(%m: i32, %x0: i32, %y0: i32, %x1: i32, %y1: i32):
      %more = arith.cmpi sge, %m, %c0 : i32
      "scf.condition"(%more, %m, %x0, %y0, %x1, %y1) : (i1, i32, i32, i32, i32, i32) -> ()
    }, {
    ^bb0(%m: i32, %x0: i32, %y0: i32, %x1: i32, %y1: i32):
      %oo = arith.muli %m, %c16 : i32
      %oa = arith.addi %own, %oo : i32
      %b = "detour.load"(%oa) : (i32) -> vector<4xi32>
      %b0 = vector.extract %b[0] : i32 from vector<4xi32>
      %b1 = vector.extract %b[1] : i32 from vector<4xi32>
      %b2 = vector.extract %b[2] : i32 from vector<4xi32>
      %b3 = vector.extract %b[3] : i32 from vector<4xi32>
      %n0 = arith.maxui %x0, %b0 : i32
      %n1 = arith.maxui %y0, %b1 : i32
      %n2 = arith.minui %x1, %b2 : i32
      %n3 = arith.minui %y1, %b3 : i32
      %lo = arith.muli %m, %c32 : i32
      %la = arith.addi %layers, %lo : i32
      %p = "detour.load"(%la) : (i32) -> i32
      "scf.yield"(%p, %n0, %n1, %n2, %n3) : (i32, i32, i32, i32, i32) -> ()
    }) : (i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32)
    %lo = arith.muli %lm, %c32 : i32
    %la = arith.addi %layers, %lo : i32
    %c4b = arith.constant 4 : i32
    %ba = arith.addi %la, %c4b : i32
    %v = vector.from_elements %r#1, %r#2, %r#3, %r#4 : vector<4xi32>
    "detour.store"(%ba, %v) : (i32, vector<4xi32>) -> ()
  }
  return
}

// @evbuild: item i's event record and rows
func.func @evbuild(%params: i32, %n: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32
  %c6 = arith.constant 6 : i32
  %c7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : i32
  %c9 = arith.constant 9 : i32
  %c16 = arith.constant 16 : i32
  %c18 = arith.constant 18 : i32
  %c32 = arith.constant 32 : i32
  %m1 = arith.constant -1 : i32
  %i = func.call @lane() : () -> i32
  %inside = arith.cmpi slt, %i, %n : i32
  scf.if %inside {
    %cpaths = func.call @param(%params, %c0) : (i32, i32) -> i32
    %items = func.call @param(%params, %c1) : (i32, i32) -> i32
    %layers = func.call @param(%params, %c2) : (i32, i32) -> i32
    %attrs = func.call @param(%params, %c4) : (i32, i32) -> i32
    %events = func.call @param(%params, %c5) : (i32, i32) -> i32
    %prow = func.call @param(%params, %c6) : (i32, i32) -> i32
    %prcount = func.call @param(%params, %c7) : (i32, i32) -> i32
    %strips = func.call @param(%params, %c8) : (i32, i32) -> i32
    %consts = func.call @param(%params, %c9) : (i32, i32) -> i32
    %cols = "detour.load"(%consts) : (i32) -> i32
    %ra = arith.addi %consts, %c4 : i32
    %rows = "detour.load"(%ra) : (i32) -> i32
    %it:4 = func.call @item(%items, %i) : (i32, i32) -> (i32, i32, i32, i32)
    %pth:3 = func.call @path(%cpaths, %it#1) : (i32, i32) -> (i32, i32, i32)
    %i4 = arith.muli %i, %c4 : i32
    %aa = arith.addi %attrs, %i4 : i32
    %pa = "detour.load"(%aa) : (i32) -> i32
    %fill = arith.cmpi eq, %it#0, %c0 : i32
    // a pop's attrs: its ordinal among the pops (items before it but the fills: i - fills before it - pops... given)
    %at = arith.select %fill, %pa, %it#3 : i32
    %top = arith.cmpi slt, %it#2, %c0 : i32
    %opq0 = arith.andi %fill, %top : i1
    %opq1 = arith.cmpi ne, %it#3, %c0 : i32
    %opqb = arith.andi %opq0, %opq1 : i1
    %opq = arith.select %opqb, %c1, %c0 : i32
    %eo = arith.muli %i, %c32 : i32
    %ea = arith.addi %events, %eo : i32
    %v0 = vector.from_elements %it#0, %pth#0, %pth#1, %pth#2 : vector<4xi32>
    "detour.store"(%ea, %v0) : (i32, vector<4xi32>) -> ()
    %ea2 = arith.addi %ea, %c16 : i32
    %v1 = vector.from_elements %at, %opq, %it#2, %c0 : vector<4xi32>
    "detour.store"(%ea2, %v1) : (i32, vector<4xi32>) -> ()
    // the box: its layer's (the viewport's for none)
    %bx:4 = scf.if %top -> (i32, i32, i32, i32) {
      scf.yield %c0, %c0, %cols, %rows : i32, i32, i32, i32
    } else {
      %lo = arith.muli %it#2, %c32 : i32
      %la = arith.addi %layers, %lo : i32
      %lb = arith.addi %la, %c4 : i32
      %b = "detour.load"(%lb) : (i32) -> vector<4xi32>
      %b0 = vector.extract %b[0] : i32 from vector<4xi32>
      %b1 = vector.extract %b[1] : i32 from vector<4xi32>
      %b2 = vector.extract %b[2] : i32 from vector<4xi32>
      %b3 = vector.extract %b[3] : i32 from vector<4xi32>
      scf.yield %b0, %b1, %b2, %b3 : i32, i32, i32, i32
    }
    %boxok = arith.cmpi ult, %bx#1, %bx#3 : i32
    // a fill's: its strips' rows (but the sentinel) within the box; a pop's: the box's
    %rr:2 = scf.if %fill -> (i32, i32) {
      %some = arith.cmpi sge, %pth#1, %c2 : i32
      %ok = arith.andi %some, %boxok : i1
      %q:2 = scf.if %ok -> (i32, i32) {
        %sb0 = arith.muli %pth#0, %c8 : i32
        %sb = arith.addi %strips, %sb0 : i32
        %w0 = "detour.load"(%sb) : (i32) -> i32
        %y0 = arith.shrui %w0, %c18 : i32
        %l2 = arith.subi %pth#1, %c2 : i32
        %l8 = arith.muli %l2, %c8 : i32
        %la = arith.addi %sb, %l8 : i32
        %w1 = "detour.load"(%la) : (i32) -> i32
        %y1 = arith.shrui %w1, %c18 : i32
        %r0 = arith.maxui %y0, %bx#1 : i32
        %bm = arith.subi %bx#3, %c1 : i32
        %r1 = arith.minui %y1, %bm : i32
        %has = arith.cmpi ule, %r0, %r1 : i32
        %d = arith.subi %r1, %r0 : i32
        %d1 = arith.addi %d, %c1 : i32
        %nr = arith.select %has, %d1, %c0 : i32
        scf.yield %r0, %nr : i32, i32
      } else {
        scf.yield %c0, %c0 : i32, i32
      }
      scf.yield %q#0, %q#1 : i32, i32
    } else {
      %d = arith.subi %bx#3, %bx#1 : i32
      %nr = arith.select %boxok, %d, %c0 : i32
      scf.yield %bx#1, %nr : i32, i32
    }
    %i8 = arith.muli %i, %c8 : i32
    %pra = arith.addi %prow, %i8 : i32
    %pv = vector.from_elements %rr#0, %rr#1 : vector<2xi32>
    "detour.store"(%pra, %pv) : (i32, vector<2xi32>) -> ()
    %pca = arith.addi %prcount, %i4 : i32
    "detour.store"(%pca, %rr#1) : (i32, i32) -> ()
  }
  return
}
