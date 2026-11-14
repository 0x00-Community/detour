// The strip compositor (detour-raster): a unit quad per vertex; per instance a run's rectangle (clip space), its
// alpha atlas coordinates, its color and its paint atlas coordinates (u affine in the position, v a row). The
// fragment shader returns paint * color * alpha (both atlases nearest, clamped: the paint's pad extend), blended
// over the target as premultiplied alpha.
detour.pipeline @strips {vertex = @strip_vs, fragment = @strip_fs, topology = "triangle-strip", targets = ["rgba8unorm"],
  blend = "premultiplied", filter = "nearest",
  vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]},
                    {stride = 64, step_mode = "instance", attributes = [{location = 1, format = "float32x4", offset = 0},
                                                                         {location = 2, format = "float32x4", offset = 16},
                                                                         {location = 3, format = "float32x4", offset = 32},
                                                                         {location = 4, format = "float32x4", offset = 48}]}]}

func.func @strip_vs(%corner: vector<2xf32>, %rect: vector<4xf32>, %uv: vector<4xf32>, %color: vector<4xf32>, %paint: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  %origin = "vector.shuffle"(%rect, %rect) <{mask = array<i64: 0, 1>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %size = "vector.shuffle"(%rect, %rect) <{mask = array<i64: 2, 3>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %scaled = arith.mulf %corner, %size : vector<2xf32>
  %xy = arith.addf %origin, %scaled : vector<2xf32>
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%xy, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  %uv0 = "vector.shuffle"(%uv, %uv) <{mask = array<i64: 0, 1>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %uv1 = "vector.shuffle"(%uv, %uv) <{mask = array<i64: 2, 3>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %span = arith.subf %uv1, %uv0 : vector<2xf32>
  %along = arith.mulf %corner, %span : vector<2xf32>
  %tc = arith.addf %uv0, %along : vector<2xf32>
  %cx = vector.extract %corner[0] : f32 from vector<2xf32>
  %cy = vector.extract %corner[1] : f32 from vector<2xf32>
  %u0 = vector.extract %paint[0] : f32 from vector<4xf32>
  %dux = vector.extract %paint[1] : f32 from vector<4xf32>
  %duy = vector.extract %paint[2] : f32 from vector<4xf32>
  %pv = vector.extract %paint[3] : f32 from vector<4xf32>
  %ux = arith.mulf %cx, %dux : f32
  %uy = arith.mulf %cy, %duy : f32
  %u1 = arith.addf %u0, %ux : f32
  %pu = arith.addf %u1, %uy : f32
  %tc0 = vector.extract %tc[0] : f32 from vector<2xf32>
  %tc1 = vector.extract %tc[1] : f32 from vector<2xf32>
  %coords = "vector.from_elements"(%tc0, %tc1, %pu, %pv) : (f32, f32, f32, f32) -> vector<4xf32>
  return %pos, %coords, %color : vector<4xf32>, vector<4xf32>, vector<4xf32>
}

func.func @strip_fs(%coords: vector<4xf32>, %color: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %tc = "vector.shuffle"(%coords, %coords) <{mask = array<i64: 0, 1>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %pc = "vector.shuffle"(%coords, %coords) <{mask = array<i64: 2, 3>}> : (vector<4xf32>, vector<4xf32>) -> vector<2xf32>
  %s = "detour.sample"(%tc) {binding = 0 : i64} : (vector<2xf32>) -> vector<4xf32>
  %p = "detour.sample"(%pc) {binding = 1 : i64} : (vector<2xf32>) -> vector<4xf32>
  %a = vector.extract %s[0] : f32 from vector<4xf32>
  %aaaa = vector.broadcast %a : f32 to vector<4xf32>
  %pcol = arith.mulf %p, %color : vector<4xf32>
  %c = arith.mulf %pcol, %aaaa : vector<4xf32>
  return %c : vector<4xf32>
}
