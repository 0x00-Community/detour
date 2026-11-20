// Uniform blocks: members of vector, scalar, integer and matrix types at std140 offsets
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
    "spirv.GlobalVariable"() <{binding = 7 : i32, descriptor_set = 0 : i32, sym_name = "Defaults", type = !spirv.ptr<!spirv.struct<(vector<4xf32> [0], f32 [16], si32 [20]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 1 : i32, descriptor_set = 0 : i32, sym_name = "params", type = !spirv.ptr<!spirv.struct<(vector<4xf32> [0], !spirv.matrix<4 x vector<4xf32>> [16, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<4xf32>, Function>
      %v_uv_addr = spirv.mlir.addressof @v_uv : !spirv.ptr<vector<2xf32>, Input>
      %1 = "spirv.Load"(%v_uv_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %cst1_i32 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %Defaults_addr = spirv.mlir.addressof @Defaults : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], f32 [16], si32 [20]), Block>, Uniform>
      %2 = spirv.AccessChain %Defaults_addr[%cst1_i32] : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], f32 [16], si32 [20]), Block>, Uniform>, i32 -> !spirv.ptr<f32, Uniform>
      %3 = "spirv.Load"(%2) : (!spirv.ptr<f32, Uniform>) -> f32
      %4 = spirv.CompositeConstruct %3, %3 : (f32, f32) -> vector<2xf32>
      %5 = spirv.FMul %1, %4 : vector<2xf32>
      %cst2_i32 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %Defaults_addr_0 = spirv.mlir.addressof @Defaults : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], f32 [16], si32 [20]), Block>, Uniform>
      %6 = spirv.AccessChain %Defaults_addr_0[%cst2_i32] : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], f32 [16], si32 [20]), Block>, Uniform>, i32 -> !spirv.ptr<si32, Uniform>
      %7 = "spirv.Load"(%6) : (!spirv.ptr<si32, Uniform>) -> si32
      %8 = spirv.Bitcast %7 : si32 to i32
      %9 = spirv.ConvertSToF %8 : i32 to f32
      %cst_f32 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %10 = spirv.CompositeConstruct %5, %9, %cst_f32 : (vector<2xf32>, f32, f32) -> vector<4xf32>
      "spirv.Store"(%0, %10) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      %cst1_i32_1 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %params_addr = spirv.mlir.addressof @params : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], !spirv.matrix<4 x vector<4xf32>> [16, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %11 = spirv.AccessChain %params_addr[%cst1_i32_1] : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], !spirv.matrix<4 x vector<4xf32>> [16, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<!spirv.matrix<4 x vector<4xf32>>, Uniform>
      %12 = "spirv.Load"(%11) : (!spirv.ptr<!spirv.matrix<4 x vector<4xf32>>, Uniform>) -> !spirv.matrix<4 x vector<4xf32>>
      %13 = "spirv.Load"(%0) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %params_addr_2 = spirv.mlir.addressof @params : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], !spirv.matrix<4 x vector<4xf32>> [16, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %14 = spirv.AccessChain %params_addr_2[%cst0_i32] : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], !spirv.matrix<4 x vector<4xf32>> [16, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %15 = "spirv.Load"(%14) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %16 = spirv.FAdd %13, %15 : vector<4xf32>
      %17 = spirv.MatrixTimesVector %12, %16 : !spirv.matrix<4 x vector<4xf32>>, vector<4xf32> -> vector<4xf32>
      %cst0_i32_3 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %Defaults_addr_4 = spirv.mlir.addressof @Defaults : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], f32 [16], si32 [20]), Block>, Uniform>
      %18 = spirv.AccessChain %Defaults_addr_4[%cst0_i32_3] : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], f32 [16], si32 [20]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %19 = "spirv.Load"(%18) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %20 = spirv.FMul %17, %19 : vector<4xf32>
      %cst2_i32_5 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %params_addr_6 = spirv.mlir.addressof @params : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], !spirv.matrix<4 x vector<4xf32>> [16, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %21 = spirv.AccessChain %params_addr_6[%cst2_i32_5] : !spirv.ptr<!spirv.struct<(vector<4xf32> [0], !spirv.matrix<4 x vector<4xf32>> [16, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<f32, Uniform>
      %22 = "spirv.Load"(%21) : (!spirv.ptr<f32, Uniform>) -> f32
      %23 = spirv.CompositeConstruct %22, %22, %22, %22 : (f32, f32, f32, f32) -> vector<4xf32>
      %24 = spirv.FMul %20, %23 : vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %24) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_uv, @o_color]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
