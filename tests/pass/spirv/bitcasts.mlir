// Bit casts both ways between floats and integers
module {
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "a_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_uv", type = !spirv.ptr<vector<2xf32>, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "Position", sym_name = "gl_Position", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %a_pos_addr = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %0 = "spirv.Load"(%a_pos_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %v_uv_addr = spirv.mlir.addressof @v_uv : !spirv.ptr<vector<2xf32>, Output>
      "spirv.Store"(%v_uv_addr, %0) : (!spirv.ptr<vector<2xf32>, Output>, vector<2xf32>) -> ()
      %a_pos_addr_0 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %1 = "spirv.Load"(%a_pos_addr_0) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %cst_f32 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %2 = spirv.CompositeConstruct %cst_f32, %cst_f32 : (f32, f32) -> vector<2xf32>
      %3 = spirv.FMul %1, %2 : vector<2xf32>
      %cst_f32_1 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %4 = spirv.CompositeConstruct %cst_f32_1, %cst_f32_1 : (f32, f32) -> vector<2xf32>
      %5 = spirv.FSub %3, %4 : vector<2xf32>
      %cst_f32_2 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_3 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %6 = spirv.CompositeConstruct %5, %cst_f32_2, %cst_f32_3 : (vector<2xf32>, f32, f32) -> vector<4xf32>
      %gl_Position_addr = spirv.mlir.addressof @gl_Position : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%gl_Position_addr, %6) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Vertex>, fn = @main, interface = [@a_pos, @v_uv, @gl_Position]}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_uv", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "o_color", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "FragCoord", sym_name = "gl_FragCoord", type = !spirv.ptr<vector<4xf32>, Input>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %gl_FragCoord_addr = spirv.mlir.addressof @gl_FragCoord : !spirv.ptr<vector<4xf32>, Input>
      %3 = "spirv.Load"(%gl_FragCoord_addr) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %4 = spirv.VectorShuffle [0 : i32, 1 : i32] %3, %3 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
      "spirv.Store"(%2, %4) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
      %5 = "spirv.Load"(%2) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %6 = "spirv.CompositeExtract"(%5) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 3.200000e+01 : f32}> : () -> f32
      %7 = spirv.FSub %6, %cst_f32 : f32
      %cst_f32_0 = "spirv.Constant"() <{value = 1.300000e+01 : f32}> : () -> f32
      %8 = spirv.FDiv %7, %cst_f32_0 : f32
      "spirv.Store"(%1, %8) : (!spirv.ptr<f32, Function>, f32) -> ()
      %9 = "spirv.Load"(%2) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %10 = "spirv.CompositeExtract"(%9) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %11 = "spirv.Load"(%2) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %12 = "spirv.CompositeExtract"(%11) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %13 = spirv.FMul %10, %12 : f32
      %cst_f32_1 = "spirv.Constant"() <{value = 1.500000e+03 : f32}> : () -> f32
      %14 = spirv.FSub %13, %cst_f32_1 : f32
      %cst_f32_2 = "spirv.Constant"() <{value = 7.000000e+02 : f32}> : () -> f32
      %15 = spirv.FDiv %14, %cst_f32_2 : f32
      "spirv.Store"(%0, %15) : (!spirv.ptr<f32, Function>, f32) -> ()
      %16 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %17 = spirv.Bitcast %16 : f32 to i32
      %cst1_i32 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %18 = spirv.BitwiseXor %17, %cst1_i32 : i32
      %19 = spirv.Bitcast %18 : i32 to f32
      %20 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      %21 = spirv.Bitcast %20 : f32 to i32
      %cst3_i32 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
      %22 = spirv.IAdd %21, %cst3_i32 : i32
      %23 = spirv.Bitcast %22 : i32 to f32
      %cst_f32_3 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_4 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %24 = spirv.CompositeConstruct %19, %23, %cst_f32_3, %cst_f32_4 : (f32, f32, f32, f32) -> vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %24) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_uv, @o_color, @gl_FragCoord]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
