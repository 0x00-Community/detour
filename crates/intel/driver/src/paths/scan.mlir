// An exclusive prefix sum of counts (counts + 4 i; a negative one 0) into offsets, a workgroup of 16 threads (256
// invocations, 4 counts each) a block of 1024 (BLOCK in common/scan.rs). Pass 0: invocation l's 4 counts summed, the
// 256 sums scanned in shared memory (Hillis-Steele, two buffers, a barrier a round), the block's offsets from its
// start at offsets + 4 i, its total at sums + 4 g. Pass 2: the block's start (sums + 4 g, the next level's offsets)
// added to its offsets. In place (counts the offsets) too: each invocation reads its counts before the first barrier,
// writes after the last. Every invocation takes part (barriers), past n writing nothing.
func.func @scan(%counts: i32, %offsets: i32, %sums: i32, %n: i32, %pass: i32) attributes {detour.stage = "compute", detour.workgroup = 16 : i32, detour.shared = 2048 : i32} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c255 = arith.constant 255 : i32
  %c256 = arith.constant 256 : i32
  %c1024 = arith.constant 1024 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %g1024 = arith.muli %g, %c1024 : i32
  %l4 = arith.muli %l, %c4 : i32
  %base = arith.addi %g1024, %l4 : i32
  %g4 = arith.muli %g, %c4 : i32
  %sa = arith.addi %sums, %g4 : i32
  %second = arith.cmpi eq, %pass, %c2 : i32
  scf.if %second {
    %start = "detour.load"(%sa) : (i32) -> i32
    scf.for %k = %c0 to %c4 step %c1 : i32 {
      %i = arith.addi %base, %k : i32
      %in = arith.cmpi slt, %i, %n : i32
      scf.if %in {
        %i4 = arith.muli %i, %c4 : i32
        %oa = arith.addi %offsets, %i4 : i32
        %o = "detour.load"(%oa) : (i32) -> i32
        %o2 = arith.addi %o, %start : i32
        "detour.store"(%oa, %o2) : (i32, i32) -> ()
      }
    }
  } else {
    %x0 = func.call @count(%counts, %base, %c0, %n) : (i32, i32, i32, i32) -> i32
    %x1 = func.call @count(%counts, %base, %c1, %n) : (i32, i32, i32, i32) -> i32
    %x2 = func.call @count(%counts, %base, %c2, %n) : (i32, i32, i32, i32) -> i32
    %x3 = func.call @count(%counts, %base, %c3, %n) : (i32, i32, i32, i32) -> i32
    %s1 = arith.addi %x0, %x1 : i32
    %s2 = arith.addi %s1, %x2 : i32
    %t = arith.addi %s2, %x3 : i32
    "detour.store"(%l4, %t) {shared = true} : (i32, i32) -> ()
    "detour.barrier"() : () -> ()
    // rounds d = 1, 2, .. 128: next[l] = cur[l] + cur[l - d] (l >= d); the buffers at 0 and 1024 bytes
    %c8 = arith.constant 8 : i32
    %fin:2 = scf.for %r = %c0 to %c8 step %c1 iter_args(%d = %c1, %cur = %c0) -> (i32, i32) : i32 {
        %nxt = arith.subi %c1024, %cur : i32
        %ca = arith.addi %cur, %l4 : i32
        %v = "detour.load"(%ca) {shared = true} : (i32) -> i32
        %has = arith.cmpi uge, %l, %d : i32
        %ld = arith.subi %l, %d : i32
        %lc = arith.select %has, %ld, %c0 : i32
        %lc4 = arith.muli %lc, %c4 : i32
        %ua = arith.addi %cur, %lc4 : i32
        %u0 = "detour.load"(%ua) {shared = true} : (i32) -> i32
        %u = arith.select %has, %u0, %c0 : i32
        %w = arith.addi %v, %u : i32
        %na = arith.addi %nxt, %l4 : i32
        "detour.store"(%na, %w) {shared = true} : (i32, i32) -> ()
        "detour.barrier"() : () -> ()
        %d2 = arith.addi %d, %d : i32
        scf.yield %d2, %nxt : i32, i32
    }
    %ia = arith.addi %fin#1, %l4 : i32
    %incl = "detour.load"(%ia) {shared = true} : (i32) -> i32
    %excl = arith.subi %incl, %t : i32
    %o1 = arith.addi %excl, %x0 : i32
    %o2 = arith.addi %o1, %x1 : i32
    %o3 = arith.addi %o2, %x2 : i32
    func.call @put(%offsets, %base, %c0, %n, %excl) : (i32, i32, i32, i32, i32) -> ()
    func.call @put(%offsets, %base, %c1, %n, %o1) : (i32, i32, i32, i32, i32) -> ()
    func.call @put(%offsets, %base, %c2, %n, %o2) : (i32, i32, i32, i32, i32) -> ()
    func.call @put(%offsets, %base, %c3, %n, %o3) : (i32, i32, i32, i32, i32) -> ()
    %last = arith.cmpi eq, %l, %c255 : i32
    scf.if %last {
      "detour.store"(%sa, %incl) : (i32, i32) -> ()
    }
  }
  return
}

// count base + k (0 past n; a negative one 0)
func.func private @count(%counts: i32, %base: i32, %k: i32, %n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %i = arith.addi %base, %k : i32
  %in = arith.cmpi slt, %i, %n : i32
  %n1 = arith.subi %n, %c1 : i32
  %ic = arith.minsi %i, %n1 : i32
  %i4 = arith.muli %ic, %c4 : i32
  %a = arith.addi %counts, %i4 : i32
  %v = "detour.load"(%a) : (i32) -> i32
  %pos = arith.maxsi %v, %c0 : i32
  %r = arith.select %in, %pos, %c0 : i32
  return %r : i32
}

// offset base + k (when before n)
func.func private @put(%offsets: i32, %base: i32, %k: i32, %n: i32, %v: i32) {
  %c4 = arith.constant 4 : i32
  %i = arith.addi %base, %k : i32
  %in = arith.cmpi slt, %i, %n : i32
  scf.if %in {
    %i4 = arith.muli %i, %c4 : i32
    %a = arith.addi %offsets, %i4 : i32
    "detour.store"(%a, %v) : (i32, i32) -> ()
  }
  return
}
