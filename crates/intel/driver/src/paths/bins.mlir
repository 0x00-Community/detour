// The strip rows' coarse commands binned by 128 px bucket as the fine kernel reads them (crates/intel/driver/src/
// fine.rs, `Packed`: plain fills), a (row, bucket) a lane (workgroup y the row, x 16 + lane the bucket; lane row
// buckets + bucket): the row's paint fills whose span overlaps the bucket (0, x | width << 16, alpha, attrs: 16
// bytes), its depth commands covering it (start | end << 16, attrs: 8 bytes), in order; the lane's record (first
// command, count, first depth command, count: 16 bytes). params: cmds, depth (@coarse_path's), coffs, doffs (its
// tables' prefix sums: a row's commands from (row, 0) to (row + 1, 0)), ctotal, dtotal (where their totals are),
// consts (rows, paths, buckets; sparse: rowstart, a row's first pair, rows + 1 of them; layer0: a layer fill's attrs
// past the paints'), ccounts, dcounts (pass 0: the lanes' counts, 4 bytes a lane), coffsets, doffsets
// (pass 1: their prefix sums), out cmds, out depth, out rows. Pass 0 counts, pass 1 writes.
func.func @bins(%params: i32, %nlanes: i32, %pass: i32) attributes {detour.stage = "compute"} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %c128 = arith.constant 128 : i32
  %c65535 = arith.constant 65535 : i32
  %true1 = arith.constant true
  %cin = "detour.load"(%params) : (i32) -> i32
  %a1 = arith.addi %params, %c4 : i32
  %din = "detour.load"(%a1) : (i32) -> i32
  %a2 = arith.addi %a1, %c4 : i32
  %coffs = "detour.load"(%a2) : (i32) -> i32
  %a3 = arith.addi %a2, %c4 : i32
  %doffs = "detour.load"(%a3) : (i32) -> i32
  %a4 = arith.addi %a3, %c4 : i32
  %ctotal = "detour.load"(%a4) : (i32) -> i32
  %a5 = arith.addi %a4, %c4 : i32
  %dtotal = "detour.load"(%a5) : (i32) -> i32
  %a6 = arith.addi %a5, %c4 : i32
  %consts = "detour.load"(%a6) : (i32) -> i32
  %a7 = arith.addi %a6, %c4 : i32
  %ccounts = "detour.load"(%a7) : (i32) -> i32
  %a8 = arith.addi %a7, %c4 : i32
  %dcounts = "detour.load"(%a8) : (i32) -> i32
  %a9 = arith.addi %a8, %c4 : i32
  %coffsets = "detour.load"(%a9) : (i32) -> i32
  %a10 = arith.addi %a9, %c4 : i32
  %doffsets = "detour.load"(%a10) : (i32) -> i32
  %a11 = arith.addi %a10, %c4 : i32
  %cout = "detour.load"(%a11) : (i32) -> i32
  %a12 = arith.addi %a11, %c4 : i32
  %dout = "detour.load"(%a12) : (i32) -> i32
  %a13 = arith.addi %a12, %c4 : i32
  %rout = "detour.load"(%a13) : (i32) -> i32
  %nrows = "detour.load"(%consts) : (i32) -> i32
  %k1 = arith.addi %consts, %c4 : i32
  %np = "detour.load"(%k1) : (i32) -> i32
  %k2 = arith.addi %k1, %c4 : i32
  %nb = "detour.load"(%k2) : (i32) -> i32
  %g = "detour.workgroup_id"() {dim = 0 : i32} : () -> i32
  %row = "detour.workgroup_id"() {dim = 1 : i32} : () -> i32
  %l = "detour.lane"() : () -> i32
  %g16 = arith.muli %g, %c16 : i32
  %b = arith.addi %g16, %l : i32
  %inside = arith.cmpi slt, %b, %nb : i32
  scf.if %inside {
    %rb = arith.muli %row, %nb : i32
    %lane = arith.addi %rb, %b : i32
    %l4 = arith.muli %lane, %c4 : i32
    %write = arith.cmpi eq, %pass, %c1 : i32
    // the row's commands and depth commands: its table row's slots
    // (sparse: the row's pairs rowstart[row] .. rowstart[row + 1], consts + 12; the scans one longer, their totals)
    %c12 = arith.constant 12 : i32
    %rsa = arith.addi %consts, %c12 : i32
    %rowstart = "detour.load"(%rsa) : (i32) -> i32
    %sparse = arith.cmpi ne, %rowstart, %c0 : i32
    %r1 = arith.addi %row, %c1 : i32
    %t0 = arith.muli %row, %np : i32
    %row4 = arith.muli %row, %c4 : i32
    %rs0a = arith.addi %rowstart, %row4 : i32
    %k0 = scf.if %sparse -> (i32) {
      %v = "detour.load"(%rs0a) : (i32) -> i32
      scf.yield %v : i32
    } else {
      scf.yield %t0 : i32
    }
    %t04 = arith.muli %k0, %c4 : i32
    %ca0 = arith.addi %coffs, %t04 : i32
    %cf = "detour.load"(%ca0) : (i32) -> i32
    %da0 = arith.addi %doffs, %t04 : i32
    %df = "detour.load"(%da0) : (i32) -> i32
    %notlast0 = arith.cmpi slt, %r1, %nrows : i32
    %notlast = arith.ori %notlast0, %sparse : i1
    %ce, %de = scf.if %notlast -> (i32, i32) {
      %k1_s1 = scf.if %sparse -> (i32) {
        %rs1a = arith.addi %rs0a, %c4 : i32
        %v = "detour.load"(%rs1a) : (i32) -> i32
        scf.yield %v : i32
      } else {
        %v = arith.addi %t0, %np : i32
        scf.yield %v : i32
      }
      %t14 = arith.muli %k1_s1, %c4 : i32
      %ca1 = arith.addi %coffs, %t14 : i32
      %x = "detour.load"(%ca1) : (i32) -> i32
      %da1 = arith.addi %doffs, %t14 : i32
      %y = "detour.load"(%da1) : (i32) -> i32
      scf.yield %x, %y : i32, i32
    } else {
      %x = "detour.load"(%ctotal) : (i32) -> i32
      %y = "detour.load"(%dtotal) : (i32) -> i32
      scf.yield %x, %y : i32, i32
    }
    %co, %dn = scf.if %write -> (i32, i32) {
      %oa = arith.addi %coffsets, %l4 : i32
      %x = "detour.load"(%oa) : (i32) -> i32
      %ob = arith.addi %doffsets, %l4 : i32
      %y = "detour.load"(%ob) : (i32) -> i32
      scf.yield %x, %y : i32, i32
    } else {
      scf.yield %c0, %c0 : i32, i32
    }
    %lo = arith.shli %b, %c7 : i32
    %hi = arith.addi %lo, %c128 : i32
    // (kinds: 0 paint fill, 1 layer fill (its attrs after the paints': layer0, consts + 16), 2 push, 3 pop; a pop
    // right after a push in the bucket takes it back, as Packed. The pushes the bucket ends with are held (t of them),
    // written before what comes next, or at the end: a push taken back is never written, past the bucket's own)
    %c16b = arith.constant 16 : i32
    %l0a = arith.addi %consts, %c16b : i32
    %layer0 = "detour.load"(%l0a) : (i32) -> i32
    %c2k = arith.constant 2 : i32
    %c3k = arith.constant 3 : i32
    %m1k = arith.constant -1 : i32
    %nct:2 = scf.for %i = %cf to %ce step %c1 iter_args(%k = %c0, %tp = %c0) -> (i32, i32) : i32 {
      %i16 = arith.muli %i, %c16 : i32
      %ia = arith.addi %cin, %i16 : i32
      %v = "detour.load"(%ia) : (i32) -> vector<4xi32>
      %xw = vector.extract %v[0] : i32 from vector<4xi32>
      %kind = vector.extract %v[3] : i32 from vector<4xi32>
      %x = arith.andi %xw, %c65535 : i32
      %w = arith.shrui %xw, %c16 : i32
      %xe = arith.addi %x, %w : i32
      %o1 = arith.cmpi slt, %x, %hi : i32
      %o2 = arith.cmpi sgt, %xe, %lo : i32
      %ov = arith.andi %o1, %o2 : i1
      %isfill = arith.cmpi ult, %kind, %c2k : i32
      %ispush = arith.cmpi eq, %kind, %c2k : i32
      %ispop = arith.cmpi eq, %kind, %c3k : i32
      %haspush = arith.cmpi ne, %tp, %c0 : i32
      %takeback = arith.andi %ispop, %haspush : i1
      %fillin = arith.andi %isfill, %ov : i1
      %notback = arith.xori %takeback, %true1 : i1
      %popin = arith.andi %ispop, %notback : i1
      %emit = arith.ori %fillin, %popin : i1
      // what is emitted: the held pushes first, then it
      %k2_s1 = scf.if %emit -> (i32) {
        %kp = func.call @held(%cout, %co, %k, %tp, %write) : (i32, i32, i32, i32, i1) -> i32
        scf.if %write {
          %at = arith.addi %co, %kp : i32
          %at16 = arith.muli %at, %c16 : i32
          %wa = arith.addi %cout, %at16 : i32
          %alpha = vector.extract %v[1] : i32 from vector<4xi32>
          %attrs0 = vector.extract %v[2] : i32 from vector<4xi32>
          %islayer = arith.cmpi eq, %kind, %c1 : i32
          %la = arith.addi %attrs0, %layer0 : i32
          %attrs = arith.select %islayer, %la, %attrs0 : i32
          %sp = arith.select %isfill, %xw, %c0 : i32
          %o = vector.from_elements %kind, %sp, %alpha, %attrs : vector<4xi32>
          "detour.store"(%wa, %o) : (i32, vector<4xi32>) -> ()
        }
        %kk = arith.addi %kp, %c1 : i32
        scf.yield %kk : i32
      } else {
        scf.yield %k : i32
      }
      %tp1 = arith.addi %tp, %c1 : i32
      %tm1 = arith.subi %tp, %c1 : i32
      %t2 = arith.select %ispush, %tp1, %tp : i32
      %t3 = arith.select %takeback, %tm1, %t2 : i32
      %t4 = arith.select %emit, %c0, %t3 : i32
      scf.yield %k2_s1, %t4 : i32, i32
    }
    %nc = func.call @held(%cout, %co, %nct#0, %nct#1, %write) : (i32, i32, i32, i32, i1) -> i32
    %nd = scf.for %i = %df to %de step %c1 iter_args(%k = %c0) -> (i32) : i32 {
      %i8 = arith.muli %i, %c8 : i32
      %ia = arith.addi %din, %i8 : i32
      %v = "detour.load"(%ia) : (i32) -> vector<2xi32>
      %se = vector.extract %v[0] : i32 from vector<2xi32>
      %s = arith.andi %se, %c65535 : i32
      %e = arith.shrui %se, %c16 : i32
      %o1 = arith.cmpi sle, %s, %b : i32
      %o2 = arith.cmpi slt, %b, %e : i32
      %ov = arith.andi %o1, %o2 : i1
      %k1n = scf.if %ov -> (i32) {
        scf.if %write {
          %at = arith.addi %dn, %k : i32
          %at8 = arith.muli %at, %c8 : i32
          %wa = arith.addi %dout, %at8 : i32
          "detour.store"(%wa, %v) : (i32, vector<2xi32>) -> ()
        }
        %kk = arith.addi %k, %c1 : i32
        scf.yield %kk : i32
      } else {
        scf.yield %k : i32
      }
      scf.yield %k1n : i32
    }
    scf.if %write {
      %l16 = arith.muli %lane, %c16 : i32
      %ra = arith.addi %rout, %l16 : i32
      %o = vector.from_elements %co, %nc, %dn, %nd : vector<4xi32>
      "detour.store"(%ra, %o) : (i32, vector<4xi32>) -> ()
    } else {
      %ca = arith.addi %ccounts, %l4 : i32
      "detour.store"(%ca, %nc) : (i32, i32) -> ()
      %da = arith.addi %dcounts, %l4 : i32
      "detour.store"(%da, %nd) : (i32, i32) -> ()
    }
  }
  return
}

// the t pushes held, written at cout + 16 (co + k ..) when writing; k + t
func.func private @held(%cout: i32, %co: i32, %k: i32, %t: i32, %write: i1) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c16 = arith.constant 16 : i32
  %m1 = arith.constant -1 : i32
  scf.if %write {
    scf.for %j = %c0 to %t step %c1 : i32 {
      %a0 = arith.addi %co, %k : i32
      %a1 = arith.addi %a0, %j : i32
      %a16 = arith.muli %a1, %c16 : i32
      %wa = arith.addi %cout, %a16 : i32
      %o = vector.from_elements %c2, %c0, %m1, %c0 : vector<4xi32>
      "detour.store"(%wa, %o) : (i32, vector<4xi32>) -> ()
    }
  }
  %n = arith.addi %k, %t : i32
  return %n : i32
}
