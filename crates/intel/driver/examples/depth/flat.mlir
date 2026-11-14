// A depth-tested pipeline: positions already in clip space (x, y, z, w), each vertex's color interpolated (here every
// vertex of a draw has the same one). `write` is set per module copy by the example (true / false).
detour.pipeline @flat {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  depth = {compare = "less", write = WRITE},
  vertex_buffers = [{stride = 32, attributes = [{location = 0, format = "float32x4", offset = 0},
                                                {location = 1, format = "float32x4", offset = 16}]}]}

func.func @vs(%p: vector<4xf32>, %c: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  return %p, %c : vector<4xf32>, vector<4xf32>
}

func.func @fs(%c: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  return %c : vector<4xf32>
}
