// The shader math, one texel of operands (a, b, c, d) per pixel: the quad and its texture coordinates as in
// `texture`; each pipeline's fragment shader returns four results, written to an rgba32float target. The operands
// are binding 0 times binding 1 (the sampler flushes denormals: a denormal operand comes as a normal one times 2^-100).
detour.pipeline @arith {vertex = @vs, fragment = @arith_fs, topology = "triangle-strip", targets = ["rgba32float"], filter = "nearest",
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x2", offset = 0},
                                                {location = 1, format = "float32x2", offset = 8}]}]}
detour.pipeline @select {vertex = @vs, fragment = @select_fs, topology = "triangle-strip", targets = ["rgba32float"], filter = "nearest",
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x2", offset = 0},
                                                {location = 1, format = "float32x2", offset = 8}]}]}

func.func @vs(%p: vector<2xf32>, %uv: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uv : vector<4xf32>, vector<2xf32>
}

// a / b, sqrt(|a|), fma(a, b, c), floor(c)
func.func @arith_fs(%uv: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %raw = "detour.sample"(%uv) {binding = 0 : i64} : (vector<2xf32>) -> vector<4xf32>
  %scale = "detour.sample"(%uv) {binding = 1 : i64} : (vector<2xf32>) -> vector<4xf32>
  %t = arith.mulf %raw, %scale : vector<4xf32>
  %a = vector.extract %t[0] : f32 from vector<4xf32>
  %b = vector.extract %t[1] : f32 from vector<4xf32>
  %c = vector.extract %t[2] : f32 from vector<4xf32>
  %q = arith.divf %a, %b : f32
  %m = math.absf %a : f32
  %s = math.sqrt %m : f32
  %f = math.fma %a, %b, %c : f32
  %fl = math.floor %c : f32
  %r = "vector.from_elements"(%q, %s, %f, %fl) : (f32, f32, f32, f32) -> vector<4xf32>
  return %r : vector<4xf32>
}

// min(a, b), max(a, b), a < b ? c : d, a >= b ? c : d
func.func @select_fs(%uv: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %raw = "detour.sample"(%uv) {binding = 0 : i64} : (vector<2xf32>) -> vector<4xf32>
  %scale = "detour.sample"(%uv) {binding = 1 : i64} : (vector<2xf32>) -> vector<4xf32>
  %t = arith.mulf %raw, %scale : vector<4xf32>
  %a = vector.extract %t[0] : f32 from vector<4xf32>
  %b = vector.extract %t[1] : f32 from vector<4xf32>
  %c = vector.extract %t[2] : f32 from vector<4xf32>
  %d = vector.extract %t[3] : f32 from vector<4xf32>
  %lo = arith.minnumf %a, %b : f32
  %hi = arith.maxnumf %a, %b : f32
  %lt = arith.cmpf olt, %a, %b : f32
  %x = arith.select %lt, %c, %d : f32
  %ge = arith.cmpf oge, %a, %b : f32
  %y = arith.select %ge, %c, %d : f32
  %r = "vector.from_elements"(%lo, %hi, %x, %y) : (f32, f32, f32, f32) -> vector<4xf32>
  return %r : vector<4xf32>
}
