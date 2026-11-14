// detour.atomic on Xe3: lane i (n lanes) adds 1 to counter i & 7 (its old value at olds + 4 i), signed max of
// (i 37 & 1023) - 512 into max i & 3, unsigned min of i 7919 & 65535 into umin (i >> 4) & 3, exchanges i into the
// one swap slot (the value it took at swaps + 4 i). params: counters, olds, maxes, umins, swap, swaps.
func.func @atomics(%params: i32, %n: i32) attributes {detour.stage = "compute"} {
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %counters = "detour.load"(%params) : (i32) -> i32
  %a1 = arith.addi %params, %c4 : i32
  %olds = "detour.load"(%a1) : (i32) -> i32
  %a2 = arith.addi %a1, %c4 : i32
  %maxes = "detour.load"(%a2) : (i32) -> i32
  %a3 = arith.addi %a2, %c4 : i32
  %umins = "detour.load"(%a3) : (i32) -> i32
  %a4 = arith.addi %a3, %c4 : i32
  %swap = "detour.load"(%a4) : (i32) -> i32
  %a5 = arith.addi %a4, %c4 : i32
  %swaps = "detour.load"(%a5) : (i32) -> i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b0 = arith.muli %g, %c16 : i32
  %i = arith.addi %b0, %l : i32
  %inside = arith.cmpi slt, %i, %n : i32
  scf.if %inside {
    %one = arith.constant 1 : i32
    %c7 = arith.constant 7 : i32
    %k = arith.andi %i, %c7 : i32
    %k4 = arith.muli %k, %c4 : i32
    %ca = arith.addi %counters, %k4 : i32
    %old = "detour.atomic"(%ca, %one) {op = "add"} : (i32, i32) -> i32
    %i4 = arith.muli %i, %c4 : i32
    %oa = arith.addi %olds, %i4 : i32
    "detour.store"(%oa, %old) : (i32, i32) -> ()
    %c3 = arith.constant 3 : i32
    %m = arith.andi %i, %c3 : i32
    %m4 = arith.muli %m, %c4 : i32
    %ma = arith.addi %maxes, %m4 : i32
    %c37 = arith.constant 37 : i32
    %c1023 = arith.constant 1023 : i32
    %c512 = arith.constant 512 : i32
    %v0 = arith.muli %i, %c37 : i32
    %v1 = arith.andi %v0, %c1023 : i32
    %v = arith.subi %v1, %c512 : i32
    %mo = "detour.atomic"(%ma, %v) {op = "max"} : (i32, i32) -> i32
    %u0 = arith.shrui %i, %c4 : i32
    %u = arith.andi %u0, %c3 : i32
    %u4 = arith.muli %u, %c4 : i32
    %ua = arith.addi %umins, %u4 : i32
    %c7919 = arith.constant 7919 : i32
    %c65535 = arith.constant 65535 : i32
    %w0 = arith.muli %i, %c7919 : i32
    %w = arith.andi %w0, %c65535 : i32
    %uo = "detour.atomic"(%ua, %w) {op = "umin"} : (i32, i32) -> i32
    %so = "detour.atomic"(%swap, %i) {op = "exchange"} : (i32, i32) -> i32
    %sa = arith.addi %swaps, %i4 : i32
    "detour.store"(%sa, %so) : (i32, i32) -> ()
  }
  return
}
