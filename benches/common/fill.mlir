// `fill`: instanced quads blended over the target as premultiplied alpha (src + dst * (1 - src.a)); the same as
// common::FILL_VS / FILL_FS (rectangle in clip space and premultiplied color per instance).
detour.pipeline @fill {vertex = @vs, fragment = @fs, topology = "triangle-strip", targets = ["rgba8unorm"], blend = "premultiplied",
  vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]},
                    {stride = 32, step_mode = "instance", attributes = [{location = 1, format = "float32x4", offset = 0},
                                                                         {location = 2, format = "float32x4", offset = 16}]}]}

func.func @vs(%corner: vector<2xf32>, %rect: vector<4xf32>, %color: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) attributes {detour.stage = "vertex"} {
  %cx = vector.extract %corner[0] : f32 from vector<2xf32>
  %cy = vector.extract %corner[1] : f32 from vector<2xf32>
  %ox = vector.extract %rect[0] : f32 from vector<4xf32>
  %oy = vector.extract %rect[1] : f32 from vector<4xf32>
  %w = vector.extract %rect[2] : f32 from vector<4xf32>
  %h = vector.extract %rect[3] : f32 from vector<4xf32>
  %sx = arith.mulf %cx, %w : f32
  %sy = arith.mulf %cy, %h : f32
  %x = arith.addf %ox, %sx : f32
  %y = arith.addf %oy, %sy : f32
  %z = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %pos = "vector.from_elements"(%x, %y, %z, %one) : (f32, f32, f32, f32) -> vector<4xf32>
  return %pos, %color : vector<4xf32>, vector<4xf32>
}

func.func @fs(%c: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  return %c : vector<4xf32>
}
