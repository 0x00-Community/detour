// A pipeline alone (no resources, no pass): what Context::pipeline compiles once.
detour.pipeline @tri {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
                     vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}

func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %x = vector.extract %p[0] : f32 from vector<2xf32>
  %y = vector.extract %p[1] : f32 from vector<2xf32>
  %z = arith.constant 0.0 : f32
  %w = arith.constant 1.0 : f32
  %pos = "vector.from_elements"(%x, %y, %z, %w) : (f32, f32, f32, f32) -> vector<4xf32>
  return %pos : vector<4xf32>
}

func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[1.0, 0.5, 0.25, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
