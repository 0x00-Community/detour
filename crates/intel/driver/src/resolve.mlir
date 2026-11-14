// The resolve of a 4x target: a triangle over the whole target whose corners carry their pixel coordinates; each pixel
// the mean of its four samples (binding 0, a multisampled texture, read unfiltered).
detour.pipeline @resolve {vertex = @resolve_vs, fragment = @resolve_fs, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 24, attributes = [{location = 0, format = "float32x4", offset = 0},
                                                {location = 1, format = "float32x2", offset = 16}]}]}

func.func @resolve_vs(%p: vector<4xf32>, %px: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {detour.stage = "vertex"} {
  return %p, %px : vector<4xf32>, vector<2xf32>
}

func.func @resolve_fs(%px: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %f = math.floor %px : vector<2xf32>
  %xy = arith.fptosi %f : vector<2xf32> to vector<2xi32>
  %s0 = arith.constant 0 : i32
  %s1 = arith.constant 1 : i32
  %s2 = arith.constant 2 : i32
  %s3 = arith.constant 3 : i32
  %c0 = "detour.load_sample"(%xy, %s0) {binding = 0 : i32} : (vector<2xi32>, i32) -> vector<4xf32>
  %c1 = "detour.load_sample"(%xy, %s1) {binding = 0 : i32} : (vector<2xi32>, i32) -> vector<4xf32>
  %c2 = "detour.load_sample"(%xy, %s2) {binding = 0 : i32} : (vector<2xi32>, i32) -> vector<4xf32>
  %c3 = "detour.load_sample"(%xy, %s3) {binding = 0 : i32} : (vector<2xi32>, i32) -> vector<4xf32>
  %a = arith.addf %c0, %c1 : vector<4xf32>
  %b = arith.addf %c2, %c3 : vector<4xf32>
  %sum = arith.addf %a, %b : vector<4xf32>
  %q = arith.constant dense<0.25> : vector<4xf32>
  %mean = arith.mulf %sum, %q : vector<4xf32>
  return %mean : vector<4xf32>
}
