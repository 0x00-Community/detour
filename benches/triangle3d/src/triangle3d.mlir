// The 3D example's triangle: positions arrive already projected (clip space x, y, z, w; the CPU turns and projects
// them every frame), the hardware divides by w; each vertex's color is interpolated.
detour.pipeline @triangle3d {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 32, attributes = [{location = 0, format = "float32x4", offset = 0},
                                                {location = 1, format = "float32x4", offset = 16}]}]}

func.func @vs(%p: vector<4xf32>, %c: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  return %p, %c : vector<4xf32>, vector<4xf32>
}

func.func @fs(%c: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  return %c : vector<4xf32>
}
