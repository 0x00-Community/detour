// Instanced rectangles: a unit quad (per vertex, a triangle strip), and per instance its rectangle (x, y, w, h in
// clip space) and color.
detour.pipeline @rects {vertex = @vs, fragment = @fs, topology = "triangle-strip", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]},
                    {stride = 32, step_mode = "instance", attributes = [{location = 1, format = "float32x4", offset = 0},
                                                                         {location = 2, format = "float32x4", offset = 16}]}]}

func.func @vs(%corner: vector<2xf32>, %rect: vector<4xf32>, %color: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  %origin = "vector.shuffle"(%rect, %rect) <{mask = array<i64: 0, 1>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %size = "vector.shuffle"(%rect, %rect) <{mask = array<i64: 2, 3>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %scaled = arith.mulf %corner, %size : vector<2xf32>
  %xy = arith.addf %origin, %scaled : vector<2xf32>
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%xy, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %color : vector<4xf32>, vector<4xf32>
}

func.func @fs(%c: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  return %c : vector<4xf32>
}
