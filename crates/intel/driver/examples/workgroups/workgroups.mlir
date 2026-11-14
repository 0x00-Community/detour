// Workgroups of 4 threads (64 invocations) with 512 bytes of shared memory and barriers (detour.workgroup,
// detour.shared, detour.barrier): invocation l of workgroup g (i = 64 g + l) writes src[i] into shared[l]; after a
// barrier it reads shared[63 - l] into dst[i] (the workgroup's 64 reversed); it also adds 1 into shared counter
// l & 3 (atomically; zeroed by invocations 0 .. 3 before the first barrier), whose values invocations 0 .. 3 write to
// counts[4 g + l] after a second one. Out of range invocations take part (barriers want them all), writing nothing.
func.func @workgroups(%src: i32, %dst: i32, %counts: i32, %n: i32) attributes {detour.stage = "compute", detour.workgroup = 4 : i32, detour.shared = 512 : i32} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c63 = arith.constant 63 : i32
  %c64 = arith.constant 64 : i32
  %c256 = arith.constant 256 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %g64 = arith.muli %g, %c64 : i32
  %i = arith.addi %g64, %l : i32
  %n1 = arith.subi %n, %c1 : i32
  %ic = arith.minui %i, %n1 : i32
  %i4 = arith.muli %ic, %c4 : i32
  %sa = arith.addi %src, %i4 : i32
  %v = "detour.load"(%sa) : (i32) -> i32
  %l4 = arith.muli %l, %c4 : i32
  "detour.store"(%l4, %v) {shared = true} : (i32, i32) -> ()
  %first4 = arith.cmpi slt, %l, %c4 : i32
  %ca = arith.addi %c256, %l4 : i32
  scf.if %first4 {
    "detour.store"(%ca, %c0) {shared = true} : (i32, i32) -> ()
  }
  "detour.barrier"() : () -> ()
  %k = arith.andi %l, %c3 : i32
  %k4 = arith.muli %k, %c4 : i32
  %kc = arith.addi %c256, %k4 : i32
  %old = "detour.atomic"(%kc, %c1) {op = "add", shared = true} : (i32, i32) -> i32
  %r = arith.subi %c63, %l : i32
  %r4 = arith.muli %r, %c4 : i32
  %w = "detour.load"(%r4) {shared = true} : (i32) -> i32
  %inside = arith.cmpi slt, %i, %n : i32
  scf.if %inside {
    %i4b = arith.muli %i, %c4 : i32
    %da = arith.addi %dst, %i4b : i32
    "detour.store"(%da, %w) : (i32, i32) -> ()
  }
  "detour.barrier"() : () -> ()
  scf.if %first4 {
    %cv = "detour.load"(%ca) {shared = true} : (i32) -> i32
    %g4 = arith.muli %g, %c4 : i32
    %gl = arith.addi %g4, %l : i32
    %gl4 = arith.muli %gl, %c4 : i32
    %oa = arith.addi %counts, %gl4 : i32
    "detour.store"(%oa, %cv) : (i32, i32) -> ()
  }
  return
}
