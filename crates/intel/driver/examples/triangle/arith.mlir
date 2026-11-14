// The triangle of func.mlir, computed: the vertices in [0, 1] mapped to clip space in the vertex shader (p * 2 - 1),
// the color doubled in the fragment shader. Same picture; exercises add / mul / sub and their dependencies.
detour.buffer @vb {data = dense<[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]> : tensor<6xf32>}
detour.texture @rt {width = 64, height = 64, format = "rgba8unorm"}
detour.buffer @stats {size = 96, readback}
detour.pipeline @tri {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
                     vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}
detour.render_pass attributes {color = @rt, load = "clear", clear = [0.0, 0.0, 1.0, 1.0], statistics = @stats} {
  detour.draw {pipeline = @tri, vertex_buffers = [@vb], vertices = 3}
}

func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %two = arith.constant dense<2.0> : vector<2xf32>
  %one = arith.constant dense<1.0> : vector<2xf32>
  %scaled = arith.mulf %p, %two : vector<2xf32>
  %xy = arith.subf %scaled, %one : vector<2xf32>
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%xy, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos : vector<4xf32>
}

func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %half = arith.constant dense<[0.5, 0.25, 0.125, 0.5]> : vector<4xf32>
  %c = arith.addf %half, %half : vector<4xf32>
  return %c : vector<4xf32>
}
