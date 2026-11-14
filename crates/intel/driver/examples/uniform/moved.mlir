// Uniform buffers: the vertex shader reads a translation from binding 0, the fragment shader its color from binding 1
// (`detour.load {binding}`: a byte offset into a buffer bound with the draw, the binding table's buffer entries).
detour.pipeline @moved {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}

func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %c0 = arith.constant 0 : i32
  %t = "detour.load"(%c0) {binding = 0} : (i32) -> vector<2xf32>
  %q = arith.addf %p, %t : vector<2xf32>
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%q, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}

func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c0 = arith.constant 0 : i32
  %c = "detour.load"(%c0) {binding = 1} : (i32) -> vector<4xf32>
  return %c : vector<4xf32>
}
