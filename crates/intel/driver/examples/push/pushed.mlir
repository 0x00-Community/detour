// Push constants (WGSL var<immediate>): the vertex shader's translation at byte 0, the fragment shader's color at byte
// 16 (`detour.push_constant {offset}`: the draw's push constants, binding table entry PUSH_BTI).
detour.pipeline @pushed {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}

func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %t = "detour.push_constant"() {offset = 0} : () -> vector<2xf32>
  %q = arith.addf %p, %t : vector<2xf32>
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%q, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}

func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = "detour.push_constant"() {offset = 16} : () -> vector<4xf32>
  return %c : vector<4xf32>
}
