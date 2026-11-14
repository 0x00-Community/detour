// f64 arithmetic, one operand set (a, b, c, d) per invocation: invocation i reads four f64 at src + 32 i and writes
// seventeen f64 results at dst + 136 i.
func.func @ops(%src: i32, %dst: i32) attributes {detour.stage = "compute"} {
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %c16 = arith.constant 16 : i32
  %base = arith.muli %g, %c16 : i32
  %i = arith.addi %base, %l : i32
  %c32 = arith.constant 32 : i32
  %in = arith.muli %i, %c32 : i32
  %at = arith.addi %src, %in : i32
  %c8 = arith.constant 8 : i32
  %at1 = arith.addi %at, %c8 : i32
  %at2 = arith.addi %at1, %c8 : i32
  %at3 = arith.addi %at2, %c8 : i32
  %a = "detour.load"(%at) : (i32) -> f64
  %b = "detour.load"(%at1) : (i32) -> f64
  %c = "detour.load"(%at2) : (i32) -> f64
  %d = "detour.load"(%at3) : (i32) -> f64
  %r0 = arith.addf %a, %b : f64
  %r1 = arith.subf %a, %b : f64
  %r2 = arith.mulf %a, %b : f64
  %r3 = math.fma %a, %b, %c : f64
  %r4 = arith.negf %a : f64
  %r5 = math.absf %a : f64
  %k = arith.constant 0.1 : f64
  %r6 = arith.mulf %a, %k : f64
  %t = arith.truncf %a : f64 to f32
  %r7a = arith.extf %t : f32 to f64
  %di = arith.fptosi %d : f64 to i32
  %r7b = arith.sitofp %di : i32 to f64
  %r7 = arith.addf %r7a, %r7b : f64
  %lt = arith.cmpf olt, %a, %b : f64
  %r8 = arith.select %lt, %a, %b : f64
  %eq = arith.cmpf oeq, %a, %b : f64
  %r9 = arith.select %eq, %c, %a : f64
  %r10 = arith.maxnumf %a, %b : f64
  %r11 = arith.divf %a, %b : f64
  %r12 = math.sqrt %a : f64
  %r13 = math.floor %a : f64
  %r14 = math.ceil %a : f64
  %r15 = math.trunc %a : f64
  %r16 = math.roundeven %a : f64
  %c64 = arith.constant 136 : i32
  %out = arith.muli %i, %c64 : i32
  %o0 = arith.addi %dst, %out : i32
  %o1 = arith.addi %o0, %c8 : i32
  %o2 = arith.addi %o1, %c8 : i32
  %o3 = arith.addi %o2, %c8 : i32
  %o4 = arith.addi %o3, %c8 : i32
  %o5 = arith.addi %o4, %c8 : i32
  %o6 = arith.addi %o5, %c8 : i32
  %o7 = arith.addi %o6, %c8 : i32
  "detour.store"(%o0, %r0) : (i32, f64) -> ()
  "detour.store"(%o1, %r1) : (i32, f64) -> ()
  "detour.store"(%o2, %r2) : (i32, f64) -> ()
  "detour.store"(%o3, %r3) : (i32, f64) -> ()
  "detour.store"(%o4, %r4) : (i32, f64) -> ()
  "detour.store"(%o5, %r5) : (i32, f64) -> ()
  "detour.store"(%o6, %r6) : (i32, f64) -> ()
  "detour.store"(%o7, %r7) : (i32, f64) -> ()
  %o8 = arith.addi %o7, %c8 : i32
  %o9 = arith.addi %o8, %c8 : i32
  %o10 = arith.addi %o9, %c8 : i32
  "detour.store"(%o8, %r8) : (i32, f64) -> ()
  "detour.store"(%o9, %r9) : (i32, f64) -> ()
  "detour.store"(%o10, %r10) : (i32, f64) -> ()
  %o11 = arith.addi %o10, %c8 : i32
  "detour.store"(%o11, %r11) : (i32, f64) -> ()
  %o12 = arith.addi %o11, %c8 : i32
  "detour.store"(%o12, %r12) : (i32, f64) -> ()
  %o13 = arith.addi %o12, %c8 : i32
  %o14 = arith.addi %o13, %c8 : i32
  %o15 = arith.addi %o14, %c8 : i32
  %o16 = arith.addi %o15, %c8 : i32
  "detour.store"(%o13, %r13) : (i32, f64) -> ()
  "detour.store"(%o14, %r14) : (i32, f64) -> ()
  "detour.store"(%o15, %r15) : (i32, f64) -> ()
  "detour.store"(%o16, %r16) : (i32, f64) -> ()
  return
}
