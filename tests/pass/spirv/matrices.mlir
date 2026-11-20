// Matrix arithmetic: a matrix times a vector and a vector times a matrix, matrix products, an outer product, a transpose,
// a matrix times a scalar, from a uniform block's mat3 and mat2
module {
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "a_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_uv", type = !spirv.ptr<vector<2xf32>, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 0 : i32, descriptor_set = 0 : i32, sym_name = "Transform", type = !spirv.ptr<!spirv.struct<(!spirv.matrix<4 x vector<4xf32>> [0, ColMajor, MatrixStride=16]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "Position", sym_name = "gl_Position", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %a_pos_addr = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %0 = "spirv.Load"(%a_pos_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %v_uv_addr = spirv.mlir.addressof @v_uv : !spirv.ptr<vector<2xf32>, Output>
      "spirv.Store"(%v_uv_addr, %0) : (!spirv.ptr<vector<2xf32>, Output>, vector<2xf32>) -> ()
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %Transform_addr = spirv.mlir.addressof @Transform : !spirv.ptr<!spirv.struct<(!spirv.matrix<4 x vector<4xf32>> [0, ColMajor, MatrixStride=16]), Block>, Uniform>
      %1 = spirv.AccessChain %Transform_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(!spirv.matrix<4 x vector<4xf32>> [0, ColMajor, MatrixStride=16]), Block>, Uniform>, i32 -> !spirv.ptr<!spirv.matrix<4 x vector<4xf32>>, Uniform>
      %2 = "spirv.Load"(%1) : (!spirv.ptr<!spirv.matrix<4 x vector<4xf32>>, Uniform>) -> !spirv.matrix<4 x vector<4xf32>>
      %a_pos_addr_0 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %3 = "spirv.Load"(%a_pos_addr_0) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %cst_f32 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_1 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %4 = spirv.CompositeConstruct %3, %cst_f32, %cst_f32_1 : (vector<2xf32>, f32, f32) -> vector<4xf32>
      %5 = spirv.MatrixTimesVector %2, %4 : !spirv.matrix<4 x vector<4xf32>>, vector<4xf32> -> vector<4xf32>
      %gl_Position_addr = spirv.mlir.addressof @gl_Position : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%gl_Position_addr, %5) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Vertex>, fn = @main, interface = [@a_pos, @v_uv, @gl_Position]}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_uv", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "o_color", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 1 : i32, descriptor_set = 0 : i32, sym_name = "Colors", type = !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<3xf32>, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %4 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %cst1_i32 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %Colors_addr = spirv.mlir.addressof @Colors : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %5 = spirv.AccessChain %Colors_addr[%cst1_i32] : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>
      %6 = "spirv.Load"(%5) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>) -> !spirv.matrix<2 x vector<2xf32>>
      %v_uv_addr = spirv.mlir.addressof @v_uv : !spirv.ptr<vector<2xf32>, Input>
      %7 = "spirv.Load"(%v_uv_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %8 = spirv.MatrixTimesVector %6, %7 : !spirv.matrix<2 x vector<2xf32>>, vector<2xf32> -> vector<2xf32>
      "spirv.Store"(%4, %8) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
      %v_uv_addr_0 = spirv.mlir.addressof @v_uv : !spirv.ptr<vector<2xf32>, Input>
      %9 = "spirv.Load"(%v_uv_addr_0) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %cst1_i32_1 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %Colors_addr_2 = spirv.mlir.addressof @Colors : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %10 = spirv.AccessChain %Colors_addr_2[%cst1_i32_1] : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>
      %11 = "spirv.Load"(%10) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>) -> !spirv.matrix<2 x vector<2xf32>>
      %12 = spirv.VectorTimesMatrix %9, %11 : vector<2xf32>, !spirv.matrix<2 x vector<2xf32>> -> vector<2xf32>
      "spirv.Store"(%3, %12) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
      %cst1_i32_3 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %Colors_addr_4 = spirv.mlir.addressof @Colors : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %13 = spirv.AccessChain %Colors_addr_4[%cst1_i32_3] : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>
      %14 = "spirv.Load"(%13) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>) -> !spirv.matrix<2 x vector<2xf32>>
      %cst1_i32_5 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %Colors_addr_6 = spirv.mlir.addressof @Colors : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %15 = spirv.AccessChain %Colors_addr_6[%cst1_i32_5] : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>
      %16 = "spirv.Load"(%15) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Uniform>) -> !spirv.matrix<2 x vector<2xf32>>
      %17 = spirv.Transpose %16 : !spirv.matrix<2 x vector<2xf32>> -> !spirv.matrix<2 x vector<2xf32>>
      %18 = spirv.MatrixTimesMatrix %14, %17 : !spirv.matrix<2 x vector<2xf32>>, !spirv.matrix<2 x vector<2xf32>> -> !spirv.matrix<2 x vector<2xf32>>
      "spirv.Store"(%2, %18) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Function>, !spirv.matrix<2 x vector<2xf32>>) -> ()
      %19 = "spirv.Load"(%4) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %20 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %21 = spirv.OuterProduct %19, %20 : vector<2xf32>, vector<2xf32> -> !spirv.matrix<2 x vector<2xf32>>
      "spirv.Store"(%1, %21) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Function>, !spirv.matrix<2 x vector<2xf32>>) -> ()
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %Colors_addr_7 = spirv.mlir.addressof @Colors : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %22 = spirv.AccessChain %Colors_addr_7[%cst0_i32] : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<!spirv.matrix<3 x vector<3xf32>>, Uniform>
      %23 = "spirv.Load"(%22) : (!spirv.ptr<!spirv.matrix<3 x vector<3xf32>>, Uniform>) -> !spirv.matrix<3 x vector<3xf32>>
      %cst2_i32 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %Colors_addr_8 = spirv.mlir.addressof @Colors : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>
      %24 = spirv.AccessChain %Colors_addr_8[%cst2_i32] : !spirv.ptr<!spirv.struct<(!spirv.matrix<3 x vector<3xf32>> [0, ColMajor, MatrixStride=16], !spirv.matrix<2 x vector<2xf32>> [48, ColMajor, MatrixStride=16], f32 [80]), Block>, Uniform>, i32 -> !spirv.ptr<f32, Uniform>
      %25 = "spirv.Load"(%24) : (!spirv.ptr<f32, Uniform>) -> f32
      %26 = spirv.MatrixTimesScalar %23, %25 : !spirv.matrix<3 x vector<3xf32>>, f32
      %27 = "spirv.Load"(%4) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %cst_f32 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %28 = spirv.CompositeConstruct %27, %cst_f32 : (vector<2xf32>, f32) -> vector<3xf32>
      %29 = spirv.MatrixTimesVector %26, %28 : !spirv.matrix<3 x vector<3xf32>>, vector<3xf32> -> vector<3xf32>
      "spirv.Store"(%0, %29) : (!spirv.ptr<vector<3xf32>, Function>, vector<3xf32>) -> ()
      %30 = "spirv.Load"(%0) : (!spirv.ptr<vector<3xf32>, Function>) -> vector<3xf32>
      %31 = "spirv.Load"(%2) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Function>) -> !spirv.matrix<2 x vector<2xf32>>
      %32 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %33 = spirv.MatrixTimesVector %31, %32 : !spirv.matrix<2 x vector<2xf32>>, vector<2xf32> -> vector<2xf32>
      %34 = "spirv.CompositeExtract"(%33) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %35 = "spirv.Load"(%1) : (!spirv.ptr<!spirv.matrix<2 x vector<2xf32>>, Function>) -> !spirv.matrix<2 x vector<2xf32>>
      %36 = "spirv.Load"(%4) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %37 = spirv.MatrixTimesVector %35, %36 : !spirv.matrix<2 x vector<2xf32>>, vector<2xf32> -> vector<2xf32>
      %38 = "spirv.CompositeExtract"(%37) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %39 = spirv.FAdd %34, %38 : f32
      %40 = spirv.CompositeConstruct %30, %39 : (vector<3xf32>, f32) -> vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %40) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_uv, @o_color]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
