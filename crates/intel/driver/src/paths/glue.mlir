// The joins between the GPU's flatten, tiling and strip rendering, a path a lane.
//
// @ranges: a path's lines (flatten's: its elements' at their offsets), as tiling takes them: (first line, count) at
// out + 8 p; none when a point is NaN (an element's flag, or one of its moves: flags + 4 p). elems: (first element,
// count) a path; offsets / nans the elements'; total the lines'.
func.func @ranges(%elems: i32, %flags: i32, %offsets: i32, %nans: i32, %total: i32, %out: i32, %npaths: i32, %nel: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %p = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %p, %npaths : i32
  scf.if %inside {
    %p8 = arith.muli %p, %c8 : i32
    %ea = arith.addi %elems, %p8 : i32
    %ev = "detour.load"(%ea) : (i32) -> vector<2xi32>
    %e0 = vector.extract %ev[0] : i32 from vector<2xi32>
    %ne = vector.extract %ev[1] : i32 from vector<2xi32>
    %p4 = arith.muli %p, %c4 : i32
    %fa = arith.addi %flags, %p4 : i32
    %moved = "detour.load"(%fa) : (i32) -> i32
    %e1 = arith.addi %e0, %ne : i32
    %bad = scf.for %e = %e0 to %e1 step %c1 iter_args(%acc = %moved) -> (i32) : i32 {
      %eo = arith.muli %e, %c4 : i32
      %na = arith.addi %nans, %eo : i32
      %nv = "detour.load"(%na) : (i32) -> i32
      %a2 = arith.ori %acc, %nv : i32
      scf.yield %a2 : i32
    }
    %e0o = arith.muli %e0, %c4 : i32
    %f0a = arith.addi %offsets, %e0o : i32
    %some = arith.cmpi slt, %e0, %nel : i32
    %first = scf.if %some -> (i32) {
      %v = "detour.load"(%f0a) : (i32) -> i32
      scf.yield %v : i32
    } else {
      %t = "detour.load"(%total) : (i32) -> i32
      scf.yield %t : i32
    }
    %last = arith.cmpi slt, %e1, %nel : i32
    %end = scf.if %last -> (i32) {
      %e1o = arith.muli %e1, %c4 : i32
      %f1a = arith.addi %offsets, %e1o : i32
      %v = "detour.load"(%f1a) : (i32) -> i32
      scf.yield %v : i32
    } else {
      %t = "detour.load"(%total) : (i32) -> i32
      scf.yield %t : i32
    }
    %n0 = arith.subi %end, %first : i32
    %isbad = arith.cmpi ne, %bad, %c0 : i32
    %n = arith.select %isbad, %c0, %n0 : i32
    %oa = arith.addi %out, %p8 : i32
    %ov = vector.from_elements %first, %n : vector<2xi32>
    "detour.store"(%oa, %ov) : (i32, vector<2xi32>) -> ()
  }
  return
}

// @stripin: strip rendering's path table: (first line, first tile, tiles, fill rule) at out + 16 p, from the lines'
// ranges (8 a path), the tiles' offsets and counts (4), the fill rules (4)
func.func @stripin(%ranges: i32, %toffsets: i32, %tcounts: i32, %rules: i32, %out: i32, %npaths: i32) attributes {detour.stage = "compute"} {
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %p = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %p, %npaths : i32
  scf.if %inside {
    %p8 = arith.muli %p, %c8 : i32
    %ra = arith.addi %ranges, %p8 : i32
    %fl = "detour.load"(%ra) : (i32) -> i32
    %p4 = arith.muli %p, %c4 : i32
    %toa = arith.addi %toffsets, %p4 : i32
    %ft = "detour.load"(%toa) : (i32) -> i32
    %tca = arith.addi %tcounts, %p4 : i32
    %nt = "detour.load"(%tca) : (i32) -> i32
    %rua = arith.addi %rules, %p4 : i32
    %ru = "detour.load"(%rua) : (i32) -> i32
    %p16 = arith.muli %p, %c16 : i32
    %oa = arith.addi %out, %p16 : i32
    %v = vector.from_elements %fl, %ft, %nt, %ru : vector<4xi32>
    "detour.store"(%oa, %v) : (i32, vector<4xi32>) -> ()
  }
  return
}

// @split: pairs (a, b) at pairs + 8 p into a + 4 p, b + 4 p
func.func @split(%pairs: i32, %a: i32, %b: i32, %n: i32) attributes {detour.stage = "compute"} {
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %p = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %p, %n : i32
  scf.if %inside {
    %p8 = arith.muli %p, %c8 : i32
    %pa = arith.addi %pairs, %p8 : i32
    %v = "detour.load"(%pa) : (i32) -> vector<2xi32>
    %x = vector.extract %v[0] : i32 from vector<2xi32>
    %y = vector.extract %v[1] : i32 from vector<2xi32>
    %p4 = arith.muli %p, %c4 : i32
    %aa = arith.addi %a, %p4 : i32
    %ba = arith.addi %b, %p4 : i32
    "detour.store"(%aa, %x) : (i32, i32) -> ()
    "detour.store"(%ba, %y) : (i32, i32) -> ()
  }
  return
}

// @pairs: offsets a + 4 p, b + 4 p into pairs + 8 p
func.func @pairs(%a: i32, %b: i32, %pairs: i32, %n: i32) attributes {detour.stage = "compute"} {
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %p = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %p, %n : i32
  scf.if %inside {
    %p4 = arith.muli %p, %c4 : i32
    %aa = arith.addi %a, %p4 : i32
    %ba = arith.addi %b, %p4 : i32
    %x = "detour.load"(%aa) : (i32) -> i32
    %y = "detour.load"(%ba) : (i32) -> i32
    %p8 = arith.muli %p, %c8 : i32
    %pa = arith.addi %pairs, %p8 : i32
    %v = vector.from_elements %x, %y : vector<2xi32>
    "detour.store"(%pa, %v) : (i32, vector<2xi32>) -> ()
  }
  return
}

// @cpaths: coarse rasterization's path table (rowcoarse's: first strip, strips with the sentinel, first alpha byte,
// paint attrs, opaque) at out + 32 p, from the paths' first slots (roffsets, 4 a path), the slots' (strip, alpha)
// offsets (soffsets, 8 a slot), the totals the last slot ends at, the paths' opaque flags (4). A path's paint attrs
// are how many before it have strips (vello_cpu's bucketer takes none for a path without: `generate_fill`): pass 0
// writes the table and whether the path has strips (flags, 4 a path), pass 1 their prefix sum (attrs) into it.
// params: soffsets, roffsets, stotal, atotal, opaque, out, flags, attrs.
func.func @cpaths(%params: i32, %npaths: i32, %nslots: i32, %pass: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %c12 = arith.constant 12 : i32
  %c16 = arith.constant 16 : i32
  %c32 = arith.constant 32 : i32
  %soffsets = "detour.load"(%params) : (i32) -> i32
  %a1 = arith.addi %params, %c4 : i32
  %roffsets = "detour.load"(%a1) : (i32) -> i32
  %a2 = arith.addi %a1, %c4 : i32
  %stotal = "detour.load"(%a2) : (i32) -> i32
  %a3 = arith.addi %a2, %c4 : i32
  %atotal = "detour.load"(%a3) : (i32) -> i32
  %a4 = arith.addi %a3, %c4 : i32
  %opaque = "detour.load"(%a4) : (i32) -> i32
  %a5 = arith.addi %a4, %c4 : i32
  %out = "detour.load"(%a5) : (i32) -> i32
  %a6 = arith.addi %a5, %c4 : i32
  %flags = "detour.load"(%a6) : (i32) -> i32
  %a7 = arith.addi %a6, %c4 : i32
  %attrs = "detour.load"(%a7) : (i32) -> i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %p = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %p, %npaths : i32
  scf.if %inside {
    %p4 = arith.muli %p, %c4 : i32
    %p32 = arith.muli %p, %c32 : i32
    %out0 = arith.addi %out, %p32 : i32
    %second = arith.cmpi eq, %pass, %c1 : i32
    scf.if %second {
      %aa = arith.addi %attrs, %p4 : i32
      %at = "detour.load"(%aa) : (i32) -> i32
      %o12 = arith.addi %out0, %c12 : i32
      "detour.store"(%o12, %at) : (i32, i32) -> ()
    } else {
      %ra = arith.addi %roffsets, %p4 : i32
      %s0 = "detour.load"(%ra) : (i32) -> i32
      %p1 = arith.addi %p, %c1 : i32
      %notlast = arith.cmpi slt, %p1, %npaths : i32
      %s1 = scf.if %notlast -> (i32) {
        %ra1 = arith.addi %ra, %c4 : i32
        %v = "detour.load"(%ra1) : (i32) -> i32
        scf.yield %v : i32
      } else {
        scf.yield %nslots : i32
      }
      %in0 = arith.cmpi slt, %s0, %nslots : i32
      %so, %ao = scf.if %in0 -> (i32, i32) {
        %o = arith.muli %s0, %c8 : i32
        %a = arith.addi %soffsets, %o : i32
        %v = "detour.load"(%a) : (i32) -> vector<2xi32>
        %x = vector.extract %v[0] : i32 from vector<2xi32>
        %y = vector.extract %v[1] : i32 from vector<2xi32>
        scf.yield %x, %y : i32, i32
      } else {
        %x = "detour.load"(%stotal) : (i32) -> i32
        %y = "detour.load"(%atotal) : (i32) -> i32
        scf.yield %x, %y : i32, i32
      }
      %in1 = arith.cmpi slt, %s1, %nslots : i32
      %se = scf.if %in1 -> (i32) {
        %o = arith.muli %s1, %c8 : i32
        %a = arith.addi %soffsets, %o : i32
        %x = "detour.load"(%a) : (i32) -> i32
        scf.yield %x : i32
      } else {
        %x = "detour.load"(%stotal) : (i32) -> i32
        scf.yield %x : i32
      }
      %ns = arith.subi %se, %so : i32
      %oa = arith.addi %opaque, %p4 : i32
      %op = "detour.load"(%oa) : (i32) -> i32
      %v0 = vector.from_elements %so, %ns, %ao, %c0 : vector<4xi32>
      "detour.store"(%out0, %v0) : (i32, vector<4xi32>) -> ()
      %out1 = arith.addi %out0, %c16 : i32
      %v1 = vector.from_elements %op, %c0, %c0, %c0 : vector<4xi32>
      "detour.store"(%out1, %v1) : (i32, vector<4xi32>) -> ()
      %some = arith.cmpi ne, %ns, %c0 : i32
      %f = arith.select %some, %c1, %c0 : i32
      %fa = arith.addi %flags, %p4 : i32
      "detour.store"(%fa, %f) : (i32, i32) -> ()
    }
  }
  return
}

// @zero: n dwords at a zeroed (16 bytes a lane; n a multiple of 4)
func.func @zero(%a: i32, %n: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %i = arith.addi %b0, %l : i32
  %i4 = arith.muli %i, %c4 : i32
  %inside = arith.cmpi slt, %i4, %n : i32
  scf.if %inside {
    %o = arith.muli %i, %c16 : i32
    %at = arith.addi %a, %o : i32
    %z = vector.from_elements %c0, %c0, %c0, %c0 : vector<4xi32>
    "detour.store"(%at, %z) : (i32, vector<4xi32>) -> ()
  }
  return
}

// @fill: n dwords at a set to v (16 bytes a lane; n a multiple of 4)
func.func @fill(%a: i32, %n: i32, %v: i32) attributes {detour.stage = "compute"} {
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %i = arith.addi %b0, %l : i32
  %i4 = arith.muli %i, %c4 : i32
  %inside = arith.cmpi slt, %i4, %n : i32
  scf.if %inside {
    %o = arith.muli %i, %c16 : i32
    %at = arith.addi %a, %o : i32
    %w = vector.from_elements %v, %v, %v, %v : vector<4xi32>
    "detour.store"(%at, %w) : (i32, vector<4xi32>) -> ()
  }
  return
}
