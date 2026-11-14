// Push constants next to a buffer binding: the translation pushed, the color from buffer binding 0 scaled by the
// pushed scalar at byte 8.
detour.pipeline @mixed {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}

func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %t = "detour.push_constant"() {offset = 0} : () -> vector<2xf32>
  %q = arith.addf %p, %t : vector<2xf32>
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%q, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}

func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c0 = arith.constant 0 : i32
  %c = "detour.load"(%c0) {binding = 0} : (i32) -> vector<4xf32>
  %s = "detour.push_constant"() {offset = 8} : () -> f32
  %v = vector.broadcast %s : f32 to vector<4xf32>
  %out = arith.mulf %c, %v : vector<4xf32>
  return %out : vector<4xf32>
}
