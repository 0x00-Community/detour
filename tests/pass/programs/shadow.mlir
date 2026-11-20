module {
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "a_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 0 : i32, descriptor_set = 0 : i32, sym_name = "u", type = !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_pos", type = !spirv.ptr<vector<2xf32>, Output>}> : () -> ()
    "spirv.GlobalVariable"() <{built_in = "Position", sym_name = "gl_Position", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %a_pos_addr = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %0 = "spirv.Load"(%a_pos_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %v_pos_addr = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Output>
      "spirv.Store"(%v_pos_addr, %0) : (!spirv.ptr<vector<2xf32>, Output>, vector<2xf32>) -> ()
      %a_pos_addr_0 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %1 = "spirv.Load"(%a_pos_addr_0) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %2 = "spirv.CompositeExtract"(%1) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>
      %3 = spirv.AccessChain %u_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %4 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %5 = "spirv.CompositeExtract"(%4) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %6 = spirv.FDiv %2, %5 : f32
      %cst_f32 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %7 = spirv.FMul %6, %cst_f32 : f32
      %cst_f32_1 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %8 = spirv.FSub %7, %cst_f32_1 : f32
      %cst_f32_2 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %a_pos_addr_3 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %9 = "spirv.Load"(%a_pos_addr_3) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %10 = "spirv.CompositeExtract"(%9) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32_4 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr_5 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>
      %11 = spirv.AccessChain %u_addr_5[%cst0_i32_4] : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %12 = "spirv.Load"(%11) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %13 = "spirv.CompositeExtract"(%12) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %14 = spirv.FDiv %10, %13 : f32
      %cst_f32_6 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %15 = spirv.FMul %14, %cst_f32_6 : f32
      %16 = spirv.FSub %15, %cst_f32_2 : f32
      %cst_f32_7 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %cst_f32_8 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %17 = spirv.CompositeConstruct %8, %16, %cst_f32_7, %cst_f32_8 : (f32, f32, f32, f32) -> vector<4xf32>
      %gl_Position_addr = spirv.mlir.addressof @gl_Position : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%gl_Position_addr, %17) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Vertex>, fn = @main, interface = [@a_pos, @v_pos, @gl_Position]}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
  "spirv.module"() <{addressing_model = #spirv.addressing_model<Logical>, memory_model = #spirv.memory_model<GLSL450>, vce_triple = #spirv.vce<v1.0, [Shader, Matrix], []>}> ({
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 1 : i32, descriptor_set = 0 : i32, sym_name = "u", type = !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "o_color", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %1 = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Input>
      %2 = "spirv.Load"(%1) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %3 = "spirv.CompositeExtract"(%2) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %4 = "spirv.CompositeExtract"(%2) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %5 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %6 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>
      %7 = spirv.AccessChain %6[%5] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %8 = "spirv.Load"(%7) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %9 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %10 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>
      %11 = spirv.AccessChain %10[%9] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %12 = "spirv.Load"(%11) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %13 = "spirv.CompositeExtract"(%8) <{indices = [0 : i32]}> : (vector<3xf32>) -> f32
      %14 = spirv.FMul %13, %3 : f32
      %15 = "spirv.CompositeExtract"(%8) <{indices = [1 : i32]}> : (vector<3xf32>) -> f32
      %16 = spirv.FMul %15, %4 : f32
      %17 = spirv.FAdd %14, %16 : f32
      %18 = "spirv.CompositeExtract"(%8) <{indices = [2 : i32]}> : (vector<3xf32>) -> f32
      %19 = spirv.FAdd %17, %18 : f32
      %20 = "spirv.CompositeExtract"(%12) <{indices = [0 : i32]}> : (vector<3xf32>) -> f32
      %21 = spirv.FMul %20, %3 : f32
      %22 = "spirv.CompositeExtract"(%12) <{indices = [1 : i32]}> : (vector<3xf32>) -> f32
      %23 = spirv.FMul %22, %4 : f32
      %24 = spirv.FAdd %21, %23 : f32
      %25 = "spirv.CompositeExtract"(%12) <{indices = [2 : i32]}> : (vector<3xf32>) -> f32
      %26 = spirv.FAdd %24, %25 : f32
      %27 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
      %28 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>
      %29 = spirv.AccessChain %28[%27] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %30 = "spirv.Load"(%29) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %31 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
      %32 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>
      %33 = spirv.AccessChain %32[%31] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %34 = "spirv.Load"(%33) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %35 = "spirv.Constant"() <{value = 4 : i32}> : () -> i32
      %36 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>
      %37 = spirv.AccessChain %36[%35] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %38 = "spirv.Load"(%37) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %39 = "spirv.CompositeExtract"(%30) <{indices = [0 : i32]}> : (vector<4xf32>) -> f32
      %40 = spirv.FSub %19, %39 : f32
      %41 = spirv.GL.FAbs %40 : f32
      %42 = "spirv.CompositeExtract"(%30) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
      %43 = spirv.FSub %41, %42 : f32
      %44 = "spirv.CompositeExtract"(%30) <{indices = [1 : i32]}> : (vector<4xf32>) -> f32
      %45 = spirv.FSub %26, %44 : f32
      %46 = spirv.GL.FAbs %45 : f32
      %47 = "spirv.CompositeExtract"(%30) <{indices = [3 : i32]}> : (vector<4xf32>) -> f32
      %48 = spirv.FSub %46, %47 : f32
      %c0 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %49 = spirv.GL.FMax %43, %c0 : f32
      %50 = spirv.GL.FMax %48, %c0 : f32
      %51 = "spirv.CompositeExtract"(%34) <{indices = [1 : i32]}> : (vector<4xf32>) -> f32
      %52 = "spirv.CompositeExtract"(%34) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
      %53 = "spirv.GL.Pow"(%49, %51) : (f32, f32) -> f32
      %54 = "spirv.GL.Pow"(%50, %51) : (f32, f32) -> f32
      %55 = spirv.FAdd %53, %54 : f32
      %56 = "spirv.GL.Pow"(%55, %52) : (f32, f32) -> f32
      %57 = spirv.GL.FMax %43, %48 : f32
      %58 = spirv.GL.FMin %57, %c0 : f32
      %59 = spirv.FAdd %56, %58 : f32
      %60 = "spirv.CompositeExtract"(%34) <{indices = [0 : i32]}> : (vector<4xf32>) -> f32
      %61 = spirv.FSub %59, %60 : f32
      %62 = "spirv.CompositeExtract"(%38) <{indices = [0 : i32]}> : (vector<4xf32>) -> f32
      %63 = "spirv.CompositeExtract"(%38) <{indices = [1 : i32]}> : (vector<4xf32>) -> f32
      %64 = "spirv.CompositeExtract"(%34) <{indices = [3 : i32]}> : (vector<4xf32>) -> f32
      %65 = spirv.FAdd %63, %61 : f32
      %66 = spirv.FMul %62, %65 : f32
      %c1 = "spirv.Constant"() <{value = 1.128379e+00 : f32}> : () -> f32
      %67 = spirv.FMul %66, %c1 : f32
      %c2 = "spirv.Constant"() <{value = -1.000000e+02 : f32}> : () -> f32
      %c3 = "spirv.Constant"() <{value = 1.000000e+02 : f32}> : () -> f32
      %68 = "spirv.GL.FClamp"(%67, %c2, %c3) : (f32, f32, f32) -> f32
      %69 = spirv.FMul %68, %68 : f32
      %c4 = "spirv.Constant"() <{value = 3.395000e-02 : f32}> : () -> f32
      %c5 = "spirv.Constant"() <{value = 1.040000e-02 : f32}> : () -> f32
      %70 = spirv.FMul %c5, %69 : f32
      %71 = spirv.FAdd %c4, %70 : f32
      %c6 = "spirv.Constant"() <{value = 2.429500e-01 : f32}> : () -> f32
      %72 = spirv.FMul %71, %69 : f32
      %73 = spirv.FAdd %c6, %72 : f32
      %74 = spirv.FMul %68, %69 : f32
      %75 = spirv.FMul %73, %74 : f32
      %76 = spirv.FAdd %68, %75 : f32
      %c7 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %77 = spirv.FMul %76, %76 : f32
      %78 = spirv.FAdd %c7, %77 : f32
      %79 = spirv.GL.Sqrt %78 : f32
      %80 = spirv.FDiv %76, %79 : f32
      %81 = spirv.FMul %62, %61 : f32
      %82 = spirv.FMul %81, %c1 : f32
      %83 = "spirv.GL.FClamp"(%82, %c2, %c3) : (f32, f32, f32) -> f32
      %84 = spirv.FMul %83, %83 : f32
      %85 = spirv.FMul %c5, %84 : f32
      %86 = spirv.FAdd %c4, %85 : f32
      %87 = spirv.FMul %86, %84 : f32
      %88 = spirv.FAdd %c6, %87 : f32
      %89 = spirv.FMul %83, %84 : f32
      %90 = spirv.FMul %88, %89 : f32
      %91 = spirv.FAdd %83, %90 : f32
      %92 = spirv.FMul %91, %91 : f32
      %93 = spirv.FAdd %c7, %92 : f32
      %94 = spirv.GL.Sqrt %93 : f32
      %95 = spirv.FDiv %91, %94 : f32
      %96 = spirv.FSub %80, %95 : f32
      %97 = spirv.FMul %64, %96 : f32
      %98 = "spirv.GL.FClamp"(%97, %c0, %c7) : (f32, f32, f32) -> f32
      %99 = "spirv.Constant"() <{value = 5 : i32}> : () -> i32
      %100 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>
      %101 = spirv.AccessChain %100[%99] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<4xf32> [48], vector<4xf32> [64], vector<4xf32> [80]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %102 = "spirv.Load"(%101) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %103 = spirv.CompositeConstruct %98, %98, %98, %98 : (f32, f32, f32, f32) -> vector<4xf32>
      %104 = spirv.FMul %102, %103 : vector<4xf32>
      %105 = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%105, %104) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_pos, @o_color]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
