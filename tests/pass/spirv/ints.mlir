// Integer arithmetic per pixel: add, sub, mul, shifts (logical and arithmetic), and, or, xor, signed and unsigned min / max, conversions both ways, bit casts, selects
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
    "spirv.GlobalVariable"() <{binding = 7 : i32, descriptor_set = 0 : i32, sym_name = "Defaults", type = !spirv.ptr<!spirv.struct<(si32 [0]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "FragCoord", sym_name = "gl_FragCoord", type = !spirv.ptr<vector<4xf32>, Input>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %4 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %5 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %6 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %7 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %8 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %gl_FragCoord_addr = spirv.mlir.addressof @gl_FragCoord : !spirv.ptr<vector<4xf32>, Input>
      %9 = "spirv.Load"(%gl_FragCoord_addr) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %10 = "spirv.CompositeExtract"(%9) <{indices = [0 : i32]}> : (vector<4xf32>) -> f32
      %11 = spirv.ConvertFToS %10 : f32 to i32
      %cst64_i32 = "spirv.Constant"() <{value = 64 : i32}> : () -> i32
      %gl_FragCoord_addr_0 = spirv.mlir.addressof @gl_FragCoord : !spirv.ptr<vector<4xf32>, Input>
      %12 = "spirv.Load"(%gl_FragCoord_addr_0) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %13 = "spirv.CompositeExtract"(%12) <{indices = [1 : i32]}> : (vector<4xf32>) -> f32
      %14 = spirv.ConvertFToS %13 : f32 to i32
      %15 = spirv.IMul %14, %cst64_i32 : i32
      %16 = spirv.IAdd %11, %15 : i32
      %cst2048_i32 = "spirv.Constant"() <{value = 2048 : i32}> : () -> i32
      %17 = spirv.ISub %16, %cst2048_i32 : i32
      "spirv.Store"(%8, %17) : (!spirv.ptr<i32, Function>, i32) -> ()
      %18 = "spirv.Load"(%8) : (!spirv.ptr<i32, Function>) -> i32
      %cst-1640531535_i32 = "spirv.Constant"() <{value = -1640531535 : i32}> : () -> i32
      %19 = spirv.IMul %18, %cst-1640531535_i32 : i32
      "spirv.Store"(%7, %19) : (!spirv.ptr<i32, Function>, i32) -> ()
      %20 = "spirv.Load"(%8) : (!spirv.ptr<i32, Function>) -> i32
      %cst7_i32 = "spirv.Constant"() <{value = 7 : i32}> : () -> i32
      %21 = spirv.IMul %20, %cst7_i32 : i32
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %Defaults_addr = spirv.mlir.addressof @Defaults : !spirv.ptr<!spirv.struct<(si32 [0]), Block>, Uniform>
      %22 = spirv.AccessChain %Defaults_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(si32 [0]), Block>, Uniform>, i32 -> !spirv.ptr<si32, Uniform>
      %23 = "spirv.Load"(%22) : (!spirv.ptr<si32, Uniform>) -> si32
      %24 = spirv.Bitcast %23 : si32 to i32
      %25 = spirv.ISub %21, %24 : i32
      "spirv.Store"(%6, %25) : (!spirv.ptr<i32, Function>, i32) -> ()
      %26 = "spirv.Load"(%8) : (!spirv.ptr<i32, Function>) -> i32
      %cst3_i32 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
      %27 = spirv.ShiftLeftLogical %26, %cst3_i32 : i32, i32
      %28 = "spirv.Load"(%6) : (!spirv.ptr<i32, Function>) -> i32
      %cst2_i32 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %29 = spirv.ShiftRightArithmetic %28, %cst2_i32 : i32, i32
      %30 = spirv.BitwiseXor %27, %29 : i32
      "spirv.Store"(%5, %30) : (!spirv.ptr<i32, Function>, i32) -> ()
      %31 = "spirv.Load"(%7) : (!spirv.ptr<i32, Function>) -> i32
      %cst5_i32 = "spirv.Constant"() <{value = 5 : i32}> : () -> i32
      %32 = spirv.ShiftRightLogical %31, %cst5_i32 : i32, i32
      %33 = "spirv.Load"(%5) : (!spirv.ptr<i32, Function>) -> i32
      %cst65280_i32 = "spirv.Constant"() <{value = 65280 : i32}> : () -> i32
      %34 = spirv.BitwiseAnd %33, %cst65280_i32 : i32
      %35 = spirv.BitwiseOr %32, %34 : i32
      "spirv.Store"(%4, %35) : (!spirv.ptr<i32, Function>, i32) -> ()
      %36 = "spirv.Load"(%6) : (!spirv.ptr<i32, Function>) -> i32
      %37 = "spirv.Load"(%5) : (!spirv.ptr<i32, Function>) -> i32
      %38 = spirv.GL.SMin %36, %37 : i32
      %cst1000_i32 = "spirv.Constant"() <{value = 1000 : i32}> : () -> i32
      %cst0_i32_1 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %cst-1000_i32 = "spirv.Constant"() <{value = -1000 : i32}> : () -> i32
      %39 = spirv.GL.SMax %38, %cst-1000_i32 : i32
      %40 = "spirv.Load"(%8) : (!spirv.ptr<i32, Function>) -> i32
      %cst9_i32 = "spirv.Constant"() <{value = 9 : i32}> : () -> i32
      %41 = spirv.ISub %40, %cst9_i32 : i32
      %42 = spirv.GL.SAbs %41 : i32
      %43 = spirv.IAdd %39, %42 : i32
      "spirv.Store"(%3, %43) : (!spirv.ptr<i32, Function>, i32) -> ()
      %44 = "spirv.Load"(%4) : (!spirv.ptr<i32, Function>) -> i32
      %cst-294967296_i32 = "spirv.Constant"() <{value = -294967296 : i32}> : () -> i32
      %45 = spirv.GL.UMin %44, %cst-294967296_i32 : i32
      %cst12_i32 = "spirv.Constant"() <{value = 12 : i32}> : () -> i32
      %46 = spirv.GL.UMax %45, %cst12_i32 : i32
      "spirv.Store"(%2, %46) : (!spirv.ptr<i32, Function>, i32) -> ()
      %47 = "spirv.Load"(%3) : (!spirv.ptr<i32, Function>) -> i32
      %48 = spirv.ConvertSToF %47 : i32 to f32
      %cst_f32 = "spirv.Constant"() <{value = 2.500000e-01 : f32}> : () -> f32
      %49 = spirv.FMul %48, %cst_f32 : f32
      %50 = "spirv.Load"(%2) : (!spirv.ptr<i32, Function>) -> i32
      %51 = spirv.ConvertUToF %50 : i32 to f32
      %cst_f32_2 = "spirv.Constant"() <{value = 4.2949673E+9 : f32}> : () -> f32
      %52 = spirv.FDiv %51, %cst_f32_2 : f32
      %53 = spirv.FAdd %49, %52 : f32
      "spirv.Store"(%1, %53) : (!spirv.ptr<f32, Function>, f32) -> ()
      %54 = "spirv.Load"(%8) : (!spirv.ptr<i32, Function>) -> i32
      %cst0_i32_3 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %55 = spirv.SLessThan %54, %cst0_i32_3 : i32
      %56 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %57 = spirv.ConvertFToS %56 : f32 to i32
      %58 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_4 = "spirv.Constant"() <{value = 3.000000e+00 : f32}> : () -> f32
      %59 = spirv.FMul %58, %cst_f32_4 : f32
      %60 = spirv.ConvertFToU %59 : f32 to i32
      %61 = spirv.Select %55, %57, %60 : i1, i32
      "spirv.Store"(%0, %61) : (!spirv.ptr<i32, Function>, i32) -> ()
      %62 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %63 = "spirv.Load"(%0) : (!spirv.ptr<i32, Function>) -> i32
      %64 = spirv.ConvertSToF %63 : i32 to f32
      %65 = "spirv.Load"(%5) : (!spirv.ptr<i32, Function>) -> i32
      %cst1073741823_i32 = "spirv.Constant"() <{value = 1073741823 : i32}> : () -> i32
      %66 = spirv.BitwiseAnd %65, %cst1073741823_i32 : i32
      %67 = spirv.Bitcast %66 : i32 to f32
      %68 = "spirv.Load"(%4) : (!spirv.ptr<i32, Function>) -> i32
      %cst2_i32_5 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %69 = spirv.ShiftRightLogical %68, %cst2_i32_5 : i32, i32
      %cst1065353216_i32 = "spirv.Constant"() <{value = 1065353216 : i32}> : () -> i32
      %70 = spirv.BitwiseOr %69, %cst1065353216_i32 : i32
      %71 = spirv.Bitcast %70 : i32 to f32
      %72 = spirv.CompositeConstruct %62, %64, %67, %71 : (f32, f32, f32, f32) -> vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %72) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_uv, @o_color, @gl_FragCoord]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
