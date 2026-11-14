// Two color targets: result 0 a constant (252, 128, 64, 255 as unorm8), result 1 texture binding 0 sampled at the
// varying's coordinates (nearest).
detour.pipeline @two {vertex = @vs, fragment = @fs, topology = "triangle-strip", targets = ["rgba8unorm", "rgba8unorm"], filter = "nearest",
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x2", offset = 0},
                                                {location = 1, format = "float32x2", offset = 8}]}]}

func.func @vs(%p: vector<2xf32>, %uv: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uv : vector<4xf32>, vector<2xf32>
}

func.func @fs(%uv: vector<2xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[0.98823529, 0.50196078, 0.25098039, 1.0]> : vector<4xf32>
  %t = "detour.sample"(%uv) {binding = 0 : i32} : (vector<2xf32>) -> vector<4xf32>
  return %c, %t : vector<4xf32>, vector<4xf32>
}
