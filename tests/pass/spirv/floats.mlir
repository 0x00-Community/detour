// Float arithmetic per pixel: multiply-add, min / max, floor, round to even, sqrt, abs, negation, division,
// comparisons and selects, float / int conversions
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
    "spirv.GlobalVariable"() <{binding = 7 : i32, descriptor_set = 0 : i32, sym_name = "Defaults", type = !spirv.ptr<!spirv.struct<(f32 [0]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "FragCoord", sym_name = "gl_FragCoord", type = !spirv.ptr<vector<4xf32>, Input>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %4 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %5 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %6 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %7 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %8 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %gl_FragCoord_addr = spirv.mlir.addressof @gl_FragCoord : !spirv.ptr<vector<4xf32>, Input>
      %9 = "spirv.Load"(%gl_FragCoord_addr) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %10 = spirv.VectorShuffle [0 : i32, 1 : i32] %9, %9 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
      "spirv.Store"(%8, %10) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
      %11 = "spirv.Load"(%8) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %12 = "spirv.CompositeExtract"(%11) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 3.150000e+01 : f32}> : () -> f32
      %13 = spirv.FSub %12, %cst_f32 : f32
      %cst_f32_0 = "spirv.Constant"() <{value = 9.000000e+00 : f32}> : () -> f32
      %14 = spirv.FDiv %13, %cst_f32_0 : f32
      "spirv.Store"(%7, %14) : (!spirv.ptr<f32, Function>, f32) -> ()
      %15 = "spirv.Load"(%8) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
      %16 = "spirv.CompositeExtract"(%15) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %Defaults_addr = spirv.mlir.addressof @Defaults : !spirv.ptr<!spirv.struct<(f32 [0]), Block>, Uniform>
      %17 = spirv.AccessChain %Defaults_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(f32 [0]), Block>, Uniform>, i32 -> !spirv.ptr<f32, Uniform>
      %18 = "spirv.Load"(%17) : (!spirv.ptr<f32, Uniform>) -> f32
      %19 = spirv.FMul %16, %18 : f32
      %cst_f32_1 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %20 = spirv.FSub %19, %cst_f32_1 : f32
      "spirv.Store"(%6, %20) : (!spirv.ptr<f32, Function>, f32) -> ()
      %21 = "spirv.Load"(%7) : (!spirv.ptr<f32, Function>) -> f32
      %22 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %23 = spirv.FMul %21, %22 : f32
      %cst_f32_2 = "spirv.Constant"() <{value = 5.000000e-01 : f32}> : () -> f32
      %24 = spirv.FSub %23, %cst_f32_2 : f32
      "spirv.Store"(%5, %24) : (!spirv.ptr<f32, Function>, f32) -> ()
      %25 = "spirv.Load"(%5) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_3 = "spirv.Constant"() <{value = 1.500000e+00 : f32}> : () -> f32
      %cst_f32_4 = "spirv.Constant"() <{value = -1.500000e+00 : f32}> : () -> f32
      %cst_f32_5 = "spirv.Constant"() <{value = 2.500000e+00 : f32}> : () -> f32
      %26 = "spirv.GL.FClamp"(%25, %cst_f32_4, %cst_f32_5) : (f32, f32, f32) -> f32
      %27 = "spirv.Load"(%7) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_6 = "spirv.Constant"() <{value = 3.700000e+00 : f32}> : () -> f32
      %28 = spirv.FMul %27, %cst_f32_6 : f32
      %29 = spirv.GL.Floor %28 : f32
      %30 = spirv.FAdd %26, %29 : f32
      %31 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_7 = "spirv.Constant"() <{value = 1.300000e+00 : f32}> : () -> f32
      %32 = spirv.FMul %31, %cst_f32_7 : f32
      %33 = spirv.GL.RoundEven %32 : f32
      %34 = spirv.FSub %30, %33 : f32
      "spirv.Store"(%4, %34) : (!spirv.ptr<f32, Function>, f32) -> ()
      %35 = "spirv.Load"(%5) : (!spirv.ptr<f32, Function>) -> f32
      %36 = spirv.GL.FAbs %35 : f32
      %cst_f32_8 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %37 = spirv.FAdd %36, %cst_f32_8 : f32
      %38 = spirv.GL.Sqrt %37 : f32
      %39 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %40 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %41 = spirv.FMul %39, %40 : f32
      %cst_f32_9 = "spirv.Constant"() <{value = 2.500000e-01 : f32}> : () -> f32
      %42 = spirv.FAdd %41, %cst_f32_9 : f32
      %43 = spirv.FDiv %38, %42 : f32
      "spirv.Store"(%3, %43) : (!spirv.ptr<f32, Function>, f32) -> ()
      %44 = "spirv.Load"(%7) : (!spirv.ptr<f32, Function>) -> f32
      %45 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %46 = spirv.FOrdGreaterThan %44, %45 : f32
      %47 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %48 = spirv.FNegate %47 : f32
      %49 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %50 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %51 = spirv.GL.FMax %49, %50 : f32
      %52 = spirv.Select %46, %48, %51 : i1, f32
      "spirv.Store"(%2, %52) : (!spirv.ptr<f32, Function>, f32) -> ()
      %53 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_10 = "spirv.Constant"() <{value = 1.000000e+02 : f32}> : () -> f32
      %54 = spirv.FMul %53, %cst_f32_10 : f32
      %cst_f32_11 = "spirv.Constant"() <{value = 1.000000e+03 : f32}> : () -> f32
      %55 = spirv.FAdd %54, %cst_f32_11 : f32
      %cst_f32_12 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_13 = "spirv.Constant"() <{value = 4.000000e+03 : f32}> : () -> f32
      %56 = "spirv.GL.FClamp"(%55, %cst_f32_12, %cst_f32_13) : (f32, f32, f32) -> f32
      %57 = spirv.ConvertFToU %56 : f32 to i32
      "spirv.Store"(%1, %57) : (!spirv.ptr<i32, Function>, i32) -> ()
      %58 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %59 = spirv.FNegate %58 : f32
      %cst_f32_14 = "spirv.Constant"() <{value = 5.000000e+01 : f32}> : () -> f32
      %60 = spirv.FMul %59, %cst_f32_14 : f32
      %61 = spirv.ConvertFToS %60 : f32 to i32
      "spirv.Store"(%0, %61) : (!spirv.ptr<i32, Function>, i32) -> ()
      %62 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %63 = "spirv.Load"(%1) : (!spirv.ptr<i32, Function>) -> i32
      %cst2_i32 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %64 = spirv.ShiftRightLogical %63, %cst2_i32 : i32, i32
      %65 = spirv.ConvertUToF %64 : i32 to f32
      %66 = "spirv.Load"(%0) : (!spirv.ptr<i32, Function>) -> i32
      %67 = spirv.ConvertSToF %66 : i32 to f32
      %68 = "spirv.Load"(%5) : (!spirv.ptr<f32, Function>) -> f32
      %69 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %70 = spirv.GL.FMin %68, %69 : f32
      %71 = spirv.CompositeConstruct %62, %65, %67, %70 : (f32, f32, f32, f32) -> vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %71) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_uv, @o_color, @gl_FragCoord]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
