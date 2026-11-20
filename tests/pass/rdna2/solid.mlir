module {
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "a_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 1 : i32, sym_name = "a_color", type = !spirv.ptr<vector<4xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 0 : i32, descriptor_set = 0 : i32, sym_name = "u", type = !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_color", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "Position", sym_name = "gl_Position", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %a_color_addr = spirv.mlir.addressof @a_color : !spirv.ptr<vector<4xf32>, Input>
      %0 = "spirv.Load"(%a_color_addr) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %v_color_addr = spirv.mlir.addressof @v_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%v_color_addr, %0) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      %a_pos_addr = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %1 = "spirv.Load"(%a_pos_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %2 = "spirv.CompositeExtract"(%1) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>
      %3 = spirv.AccessChain %u_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %4 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %5 = "spirv.CompositeExtract"(%4) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %6 = spirv.FDiv %2, %5 : f32
      %cst_f32 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %7 = spirv.FMul %6, %cst_f32 : f32
      %cst_f32_0 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %8 = spirv.FSub %7, %cst_f32_0 : f32
      %cst_f32_1 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %a_pos_addr_2 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %9 = "spirv.Load"(%a_pos_addr_2) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %10 = "spirv.CompositeExtract"(%9) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32_3 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr_4 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>
      %11 = spirv.AccessChain %u_addr_4[%cst0_i32_3] : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %12 = "spirv.Load"(%11) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %13 = "spirv.CompositeExtract"(%12) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %14 = spirv.FDiv %10, %13 : f32
      %cst_f32_5 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %15 = spirv.FMul %14, %cst_f32_5 : f32
      %16 = spirv.FSub %15, %cst_f32_1 : f32
      %cst_f32_6 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_7 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %17 = spirv.CompositeConstruct %8, %16, %cst_f32_6, %cst_f32_7 : (f32, f32, f32, f32) -> vector<4xf32>
      %gl_Position_addr = spirv.mlir.addressof @gl_Position : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%gl_Position_addr, %17) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Vertex>, fn = @main, interface = [@a_pos, @a_color, @v_color, @gl_Position]}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_color", type = !spirv.ptr<vector<4xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "o_color", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %v_color_addr = spirv.mlir.addressof @v_color : !spirv.ptr<vector<4xf32>, Input>
      %0 = "spirv.Load"(%v_color_addr) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %0) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_color, @o_color]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
