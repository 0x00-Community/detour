// The triangle of func.mlir, its shaders in the gpu dialect.
module attributes {gpu.container_module} {
  detour.buffer @vb {data = dense<[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]> : tensor<6xf32>}
  detour.texture @rt {width = 64, height = 64, format = "rgba8unorm"}
  detour.buffer @stats {size = 96, readback}
  detour.pipeline @tri {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
                       vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}
  detour.render_pass attributes {color = @rt, load = "clear", clear = [0.0, 0.0, 1.0, 1.0], statistics = @stats} {
    detour.draw {pipeline = @tri, vertex_buffers = [@vb], vertices = 3}
  }

  gpu.module @shaders {
    gpu.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
      %x = vector.extract %p[0] : f32 from vector<2xf32>
      %y = vector.extract %p[1] : f32 from vector<2xf32>
      %z = arith.constant 0.0 : f32
      %w = arith.constant 1.0 : f32
      %pos = "vector.from_elements"(%x, %y, %z, %w) : (f32, f32, f32, f32) -> vector<4xf32>
      gpu.return %pos : vector<4xf32>
    }

    gpu.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
      %c = arith.constant dense<[1.0, 0.5, 0.25, 1.0]> : vector<4xf32>
      gpu.return %c : vector<4xf32>
    }
  }
}
