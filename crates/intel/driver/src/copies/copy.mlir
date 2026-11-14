// A rectangle of bytes copied, one invocation a unit: rows `h` of `w` units from `src` (rows `ss` bytes apart) to `dst`
// (rows `ds` bytes apart); @copy_words a unit 4 bytes (everything 4-aligned), @copy_bytes 1 byte. Workgroup (x, y):
// units 16 x .. 16 x + 15 of row y, x clamped to the last (the lanes past it copy that unit again).

func.func @copy_words(%src: i32, %ss: i32, %dst: i32, %ds: i32, %wh: i32) attributes {detour.stage = "compute"} {
  %c2 = arith.constant 2 : i32
  %x, %y = func.call @unit(%wh) : (i32) -> (i32, i32)
  %xb = arith.shli %x, %c2 : i32
  %sr = arith.muli %y, %ss : i32
  %s0 = arith.addi %src, %sr : i32
  %s = arith.addi %s0, %xb : i32
  %dr = arith.muli %y, %ds : i32
  %d0 = arith.addi %dst, %dr : i32
  %d = arith.addi %d0, %xb : i32
  %v = "detour.load"(%s) : (i32) -> i32
  "detour.store"(%d, %v) : (i32, i32) -> ()
  return
}

func.func @copy_bytes(%src: i32, %ss: i32, %dst: i32, %ds: i32, %wh: i32) attributes {detour.stage = "compute"} {
  %x, %y = func.call @unit(%wh) : (i32) -> (i32, i32)
  %sr = arith.muli %y, %ss : i32
  %s0 = arith.addi %src, %sr : i32
  %s = arith.addi %s0, %x : i32
  %dr = arith.muli %y, %ds : i32
  %d0 = arith.addi %dst, %dr : i32
  %d = arith.addi %d0, %x : i32
  %v = "detour.load"(%s) {size = 1 : i32} : (i32) -> i32
  "detour.store"(%d, %v) {size = 1 : i32} : (i32, i32) -> ()
  return
}

// This invocation's unit of a grid `w | h << 16` (x clamped to the last).
func.func private @unit(%wh: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %cffff = arith.constant 65535 : i32
  %gx = "detour.workgroup_id"() {dim = 0 : i32} : () -> i32
  %gy = "detour.workgroup_id"() {dim = 1 : i32} : () -> i32
  %l = "detour.lane"() : () -> i32
  %w = arith.andi %wh, %cffff : i32
  %b = arith.muli %gx, %c16 : i32
  %x0 = arith.addi %b, %l : i32
  %last = arith.subi %w, %c1 : i32
  %x = arith.minsi %x0, %last : i32
  return %x, %gy : i32, i32
}
