// The window example's triangle: each vertex has a color (location 1), the vertex shader passes it on as a
// varying, the fragment shader returns what it gets.
detour.pipeline @triangle {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 24, attributes = [{location = 0, format = "float32x2", offset = 0},
                                                {location = 1, format = "float32x4", offset = 8}]}]}

func.func @vs(%p: vector<2xf32>, %c: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %c : vector<4xf32>, vector<4xf32>
}

func.func @fs(%c: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  return %c : vector<4xf32>
}
