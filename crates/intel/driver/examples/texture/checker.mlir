// A texture over the whole target: a quad (two triangles as a strip) with texture coordinates, passed to the
// fragment shader as a varying, which samples binding 0 with nearest filtering.
detour.pipeline @textured {vertex = @vs, fragment = @fs, topology = "triangle-strip", targets = ["rgba8unorm"], filter = "nearest",
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x2", offset = 0},
                                                {location = 1, format = "float32x2", offset = 8}]}]}

func.func @vs(%p: vector<2xf32>, %uv: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uv : vector<4xf32>, vector<2xf32>
}

func.func @fs(%uv: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = "detour.sample"(%uv) {binding = 0 : i64} : (vector<2xf32>) -> vector<4xf32>
  return %c : vector<4xf32>
}
