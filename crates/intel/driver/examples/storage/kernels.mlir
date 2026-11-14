// Storage textures (WGSL texture_storage_2d): one invocation per texel of a w x h grid (workgroups of 16 along x, one
// row of them per y; the lanes past w are outside every texture, their stores dropped).
func.func @fill(%w: i32) attributes {detour.stage = "compute"} {
  %gx = "detour.workgroup_id"() {dim = 0 : i64} : () -> i32
  %y = "detour.workgroup_id"() {dim = 1 : i64} : () -> i32
  %l = "detour.lane"() : () -> i32
  %c16 = arith.constant 16 : i32
  %base = arith.muli %gx, %c16 : i32
  %x = arith.addi %base, %l : i32
  %xy = vector.from_elements %x, %y : vector<2xi32>
  %xf = arith.sitofp %x : i32 to f32
  %yf = arith.sitofp %y : i32 to f32
  %p = arith.mulf %xf, %yf : f32
  %one = arith.constant 1.0 : f32
  %rgba = vector.from_elements %xf, %yf, %p, %one : vector<4xf32>
  "detour.texture_store"(%xy, %rgba) {binding = 0 : i64} : (vector<2xi32>, vector<4xf32>) -> ()
  %half = arith.constant 0.5 : f32
  %r = arith.addf %p, %half : f32
  "detour.texture_store"(%xy, %r) {binding = 1 : i64} : (vector<2xi32>, f32) -> ()
  %k = arith.constant 0.003921568627 : f32
  %xn = arith.mulf %xf, %k : f32
  %yn = arith.mulf %yf, %k : f32
  %s = arith.addf %xf, %yf : f32
  %sn = arith.mulf %s, %k : f32
  %unorm = vector.from_elements %xn, %yn, %sn, %one : vector<4xf32>
  "detour.texture_store"(%xy, %unorm) {binding = 2 : i64} : (vector<2xi32>, vector<4xf32>) -> ()
  return
}

// Read-write: texel (x, y) of binding 0 doubled into binding 1.
func.func @double(%w: i32) attributes {detour.stage = "compute"} {
  %gx = "detour.workgroup_id"() {dim = 0 : i64} : () -> i32
  %y = "detour.workgroup_id"() {dim = 1 : i64} : () -> i32
  %l = "detour.lane"() : () -> i32
  %c16 = arith.constant 16 : i32
  %base = arith.muli %gx, %c16 : i32
  %x = arith.addi %base, %l : i32
  %xy = vector.from_elements %x, %y : vector<2xi32>
  %v = "detour.texture_load"(%xy) {binding = 0 : i64} : (vector<2xi32>) -> vector<4xf32>
  %d = arith.addf %v, %v : vector<4xf32>
  "detour.texture_store"(%xy, %d) {binding = 1 : i64} : (vector<2xi32>, vector<4xf32>) -> ()
  return
}
