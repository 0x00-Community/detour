// Functions called from the entry point, one from another, with in and out parameters: inlined before the lowering
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
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = (f32) -> f32, sym_name = "smooth01"}> ({
    ^bb0(%arg0: f32):
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      "spirv.Store"(%1, %arg0) : (!spirv.ptr<f32, Function>, f32) -> ()
      %2 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_0 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %3 = "spirv.GL.FClamp"(%2, %cst_f32, %cst_f32_0) : (f32, f32, f32) -> f32
      "spirv.Store"(%0, %3) : (!spirv.ptr<f32, Function>, f32) -> ()
      %4 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      %5 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      %6 = spirv.FMul %4, %5 : f32
      %cst_f32_1 = "spirv.Constant"() <{value = 3.000000e+00 : f32}> : () -> f32
      %cst_f32_2 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %7 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      %8 = spirv.FMul %7, %cst_f32_2 : f32
      %9 = spirv.FSub %cst_f32_1, %8 : f32
      %10 = spirv.FMul %6, %9 : f32
      spirv.ReturnValue %10 : f32
    }) : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = (vector<2xf32>, !spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> (), sym_name = "split"}> ({
    ^bb0(%arg0: vector<2xf32>, %arg1: !spirv.ptr<f32, Function>, %arg2: !spirv.ptr<f32, Function>):
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      "spirv.Store"(%0, %arg0) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
      %1 = "spirv.Load"(%0) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %2 = "spirv.CompositeExtract"(%1) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %3 = spirv.FMul %2, %cst_f32 : f32
      %cst_f32_0 = "spirv.Constant"() <{value = 5.000000e-01 : f32}> : () -> f32
      %4 = spirv.FSub %3, %cst_f32_0 : f32
      %5 = spirv.FunctionCall @smooth01(%4) : (f32) -> f32
      "spirv.Store"(%arg1, %5) : (!spirv.ptr<f32, Function>, f32) -> ()
      %6 = "spirv.Load"(%0) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %7 = "spirv.CompositeExtract"(%6) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %8 = "spirv.Load"(%arg1) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_1 = "spirv.Constant"() <{value = 2.500000e-01 : f32}> : () -> f32
      %9 = spirv.FMul %8, %cst_f32_1 : f32
      %10 = spirv.FAdd %7, %9 : f32
      %11 = spirv.FunctionCall @smooth01(%10) : (f32) -> f32
      "spirv.Store"(%arg2, %11) : (!spirv.ptr<f32, Function>, f32) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %v_uv_addr = spirv.mlir.addressof @v_uv : !spirv.ptr<vector<2xf32>, Input>
      %4 = "spirv.Load"(%v_uv_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      spirv.FunctionCall @split(%4, %1, %0) : (vector<2xf32>, !spirv.ptr<f32, Function>, !spirv.ptr<f32, Function>) -> ()
      %5 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      "spirv.Store"(%3, %5) : (!spirv.ptr<f32, Function>, f32) -> ()
      %6 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      "spirv.Store"(%2, %6) : (!spirv.ptr<f32, Function>, f32) -> ()
      %false = "spirv.Constant"() <{value = false}> : () -> i1
      %7 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %8 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %9 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %10 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %11 = spirv.FMul %9, %10 : f32
      %12 = spirv.FunctionCall @smooth01(%11) : (f32) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %13 = spirv.CompositeConstruct %7, %8, %12, %cst_f32 : (f32, f32, f32, f32) -> vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %13) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_uv, @o_color]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
