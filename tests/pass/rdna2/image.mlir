module {
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "a_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 1 : i32, sym_name = "a_cover", type = !spirv.ptr<f32, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 0 : i32, descriptor_set = 0 : i32, sym_name = "u", type = !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_pos", type = !spirv.ptr<vector<2xf32>, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 1 : i32, sym_name = "v_cover", type = !spirv.ptr<f32, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "Position", sym_name = "gl_Position", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %a_pos_addr = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %0 = "spirv.Load"(%a_pos_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %v_pos_addr = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Output>
      "spirv.Store"(%v_pos_addr, %0) : (!spirv.ptr<vector<2xf32>, Output>, vector<2xf32>) -> ()
      %a_cover_addr = spirv.mlir.addressof @a_cover : !spirv.ptr<f32, Input>
      %1 = "spirv.Load"(%a_cover_addr) : (!spirv.ptr<f32, Input>) -> f32
      %v_cover_addr = spirv.mlir.addressof @v_cover : !spirv.ptr<f32, Output>
      "spirv.Store"(%v_cover_addr, %1) : (!spirv.ptr<f32, Output>, f32) -> ()
      %a_pos_addr_0 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %2 = "spirv.Load"(%a_pos_addr_0) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %3 = "spirv.CompositeExtract"(%2) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>
      %4 = spirv.AccessChain %u_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %5 = "spirv.Load"(%4) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %6 = "spirv.CompositeExtract"(%5) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %7 = spirv.FDiv %3, %6 : f32
      %cst_f32 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %8 = spirv.FMul %7, %cst_f32 : f32
      %cst_f32_1 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %9 = spirv.FSub %8, %cst_f32_1 : f32
      %a_pos_addr_2 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %10 = "spirv.Load"(%a_pos_addr_2) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %11 = "spirv.CompositeExtract"(%10) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32_3 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr_4 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>
      %12 = spirv.AccessChain %u_addr_4[%cst0_i32_3] : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %13 = "spirv.Load"(%12) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %14 = "spirv.CompositeExtract"(%13) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %15 = spirv.FDiv %11, %14 : f32
      %cst_f32_5 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %16 = spirv.FMul %15, %cst_f32_5 : f32
      %cst_f32_6 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %17 = spirv.FSub %16, %cst_f32_6 : f32
      %cst_f32_7 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_8 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %18 = spirv.CompositeConstruct %9, %17, %cst_f32_7, %cst_f32_8 : (f32, f32, f32, f32) -> vector<4xf32>
      %gl_Position_addr = spirv.mlir.addressof @gl_Position : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%gl_Position_addr, %18) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Vertex>, fn = @main, interface = [@a_pos, @a_cover, @v_pos, @v_cover, @gl_Position]}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{binding = 1 : i32, descriptor_set = 0 : i32, sym_name = "u", type = !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 2 : i32, descriptor_set = 0 : i32, sym_name = "tex", type = !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 1 : i32, sym_name = "v_cover", type = !spirv.ptr<f32, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "o_color", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<3xf32>, Function>
      %v_pos_addr = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Input>
      %2 = "spirv.Load"(%v_pos_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %cst_f32 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %3 = spirv.CompositeConstruct %2, %cst_f32 : (vector<2xf32>, f32) -> vector<3xf32>
      "spirv.Store"(%1, %3) : (!spirv.ptr<vector<3xf32>, Function>, vector<3xf32>) -> ()
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>
      %4 = spirv.AccessChain %u_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %5 = "spirv.Load"(%4) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %6 = "spirv.Load"(%1) : (!spirv.ptr<vector<3xf32>, Function>) -> vector<3xf32>
      %7 = spirv.Dot %5, %6 : vector<3xf32> -> f32
      %cst1_i32 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %u_addr_0 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>
      %8 = spirv.AccessChain %u_addr_0[%cst1_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %9 = "spirv.Load"(%8) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %10 = "spirv.Load"(%1) : (!spirv.ptr<vector<3xf32>, Function>) -> vector<3xf32>
      %11 = spirv.Dot %9, %10 : vector<3xf32> -> f32
      %12 = spirv.CompositeConstruct %7, %11 : (f32, f32) -> vector<2xf32>
      "spirv.Store"(%0, %12) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
      %tex_addr = spirv.mlir.addressof @tex : !spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant>
      %13 = "spirv.Load"(%tex_addr) : (!spirv.ptr<!spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, UniformConstant>) -> !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
      %14 = "spirv.Load"(%0) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %cst2_i32 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %u_addr_1 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>
      %15 = spirv.AccessChain %u_addr_1[%cst2_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %16 = "spirv.Load"(%15) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %17 = spirv.VectorShuffle [0 : i32, 1 : i32] %16, %16 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
      %18 = spirv.FDiv %14, %17 : vector<2xf32>
      %19 = "spirv.ImageSampleImplicitLod"(%13, %18) : (!spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>, vector<2xf32>) -> vector<4xf32>
      %cst2_i32_2 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %u_addr_3 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>
      %20 = spirv.AccessChain %u_addr_3[%cst2_i32_2] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %21 = "spirv.Load"(%20) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %22 = "spirv.CompositeExtract"(%21) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
      %v_cover_addr = spirv.mlir.addressof @v_cover : !spirv.ptr<f32, Input>
      %23 = "spirv.Load"(%v_cover_addr) : (!spirv.ptr<f32, Input>) -> f32
      %24 = spirv.FMul %22, %23 : f32
      %25 = spirv.CompositeConstruct %24, %24, %24, %24 : (f32, f32, f32, f32) -> vector<4xf32>
      %26 = spirv.FMul %19, %25 : vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %26) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_pos, @v_cover, @o_color]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
