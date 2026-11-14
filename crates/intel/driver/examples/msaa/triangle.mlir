// A 4x multisampled triangle: positions in clip space, one color (252, 128, 64, 255 as unorm8).
detour.pipeline @tri {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], samples = 4 : i32,
                     vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}

func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}

func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[0.98823529, 0.50196078, 0.25098039, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
