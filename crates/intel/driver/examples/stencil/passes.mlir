// Stencil passes over vec4 corners (the clear's layout), depth always passing and never written:
// @mask writes the reference where it draws, @incr adds one (clamped), @eq0 / @eq1 / @eq2 draw where the stencil
// equals the reference, each its own color; @sided replaces on front faces and inverts on back faces (double sided).
detour.pipeline @mask {vertex = @vs, fragment = @gray, topology = "triangle-strip", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x4", offset = 0}]}],
  depth = {compare = "always", write = false, stencil = {compare = "always", pass_op = "replace"}}}
detour.pipeline @incr {vertex = @vs, fragment = @gray, topology = "triangle-strip", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x4", offset = 0}]}],
  depth = {compare = "always", write = false, stencil = {compare = "always", pass_op = "increment-clamp"}}}
detour.pipeline @sided {vertex = @vs, fragment = @gray, topology = "triangle-strip", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x4", offset = 0}]}],
  depth = {compare = "always", write = false, stencil = {compare = "always", pass_op = "replace", back = {compare = "always", pass_op = "invert"}}}}
detour.pipeline @eq0 {vertex = @vs, fragment = @white, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x4", offset = 0}]}],
  depth = {compare = "always", write = false, stencil = {compare = "equal", write_mask = 0}}}
detour.pipeline @eq1 {vertex = @vs, fragment = @purple, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x4", offset = 0}]}],
  depth = {compare = "always", write = false, stencil = {compare = "equal", write_mask = 0}}}
detour.pipeline @eq2 {vertex = @vs, fragment = @orange, topology = "triangle-list", targets = ["rgba8unorm"],
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x4", offset = 0}]}],
  depth = {compare = "always", write = false, stencil = {compare = "equal", write_mask = 0}}}

func.func @vs(%p: vector<4xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  return %p : vector<4xf32>
}
func.func @gray() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[0.5, 0.5, 0.5, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
func.func @white() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[1.0, 1.0, 1.0, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
func.func @purple() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[0.5019608, 0.0, 1.0, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
func.func @orange() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[1.0, 0.5019608, 0.0, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
