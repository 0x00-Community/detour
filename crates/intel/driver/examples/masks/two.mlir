// Two targets, each its own color target state: the first opaque writing only red and alpha, the second blending
// premultiplied. Result 0 (1, 1, 0, 1); result 1 (100/255, 0, 0, 0): premultiplied with alpha 0 adds it to what is there.
detour.pipeline @two {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm", "rgba8unorm"],
  blends = ["none", "premultiplied"], write_masks = [9, 15],
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x4", offset = 0}]}]}

func.func @vs(%p: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  return %p : vector<4xf32>
}

func.func @fs() -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "fragment"} {
  %a = arith.constant dense<[1.0, 1.0, 0.0, 1.0]> : vector<4xf32>
  %b = arith.constant dense<[0.39215686, 0.0, 0.0, 0.0]> : vector<4xf32>
  return %a, %b : vector<4xf32>, vector<4xf32>
}
