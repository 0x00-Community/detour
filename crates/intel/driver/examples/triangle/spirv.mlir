// The triangle of func.mlir, its shaders in the spirv dialect (generic form, as a SPIR-V binary deserializes).
detour.buffer @vb {data = dense<[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]> : tensor<6xf32>}
detour.texture @rt {width = 64, height = 64, format = "rgba8unorm"}
detour.buffer @stats {size = 96, readback}
detour.pipeline @tri {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
                     vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}
detour.render_pass attributes {color = @rt, load = "clear", clear = [0.0, 0.0, 1.0, 1.0], statistics = @stats} {
  detour.draw {pipeline = @tri, vertex_buffers = [@vb], vertices = 3}
}

"spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>}> ({
  "spirv.GlobalVariable"() <{sym_name = "in_pos", type = !spirv.ptr<vector<2xf32>, Input>}> {location = 0 : i32} : () -> ()
  "spirv.GlobalVariable"() <{sym_name = "out_pos", type = !spirv.ptr<vector<4xf32>, Output>}> {built_in = "Position"} : () -> ()
  "spirv.GlobalVariable"() <{sym_name = "out_color", type = !spirv.ptr<vector<4xf32>, Output>}> {location = 0 : i32} : () -> ()
  "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "vs"}> ({
    %0 = "spirv.mlir.addressof"() <{variable = @in_pos}> : () -> !spirv.ptr<vector<2xf32>, Input>
    %1 = "spirv.Load"(%0) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
    %2 = "spirv.CompositeExtract"(%1) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
    %3 = "spirv.CompositeExtract"(%1) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
    %4 = "spirv.Constant"() <{value = 0.0 : f32}> : () -> f32
    %5 = "spirv.Constant"() <{value = 1.0 : f32}> : () -> f32
    %6 = "spirv.CompositeConstruct"(%2, %3, %4, %5) : (f32, f32, f32, f32) -> vector<4xf32>
    %7 = "spirv.mlir.addressof"() <{variable = @out_pos}> : () -> !spirv.ptr<vector<4xf32>, Output>
    "spirv.Store"(%7, %6) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
    "spirv.Return"() : () -> ()
  }) : () -> ()
  "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "fs"}> ({
    %0 = "spirv.Constant"() <{value = dense<[1.0, 0.5, 0.25, 1.0]> : vector<4xf32>}> : () -> vector<4xf32>
    %1 = "spirv.mlir.addressof"() <{variable = @out_color}> : () -> !spirv.ptr<vector<4xf32>, Output>
    "spirv.Store"(%1, %0) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
    "spirv.Return"() : () -> ()
  }) : () -> ()
  "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Vertex>, fn = @vs, interface = [@in_pos, @out_pos]}> : () -> ()
  "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @fs, interface = [@out_color]}> : () -> ()
}) : () -> ()
