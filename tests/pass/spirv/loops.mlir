// Structured control flow per pixel: a loop the same for every pixel, one of a different length per pixel, an if on a per-pixel condition, an if inside a loop, a while loop
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
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %4 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %5 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %6 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %7 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %8 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %9 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<i32, Function>
      %gl_FragCoord_addr = spirv.mlir.addressof @gl_FragCoord : !spirv.ptr<vector<4xf32>, Input>
      %10 = "spirv.Load"(%gl_FragCoord_addr) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %11 = "spirv.CompositeExtract"(%10) <{indices = [0 : i32]}> : (vector<4xf32>) -> f32
      %12 = spirv.ConvertFToS %11 : f32 to i32
      %cst64_i32 = "spirv.Constant"() <{value = 64 : i32}> : () -> i32
      %gl_FragCoord_addr_0 = spirv.mlir.addressof @gl_FragCoord : !spirv.ptr<vector<4xf32>, Input>
      %13 = "spirv.Load"(%gl_FragCoord_addr_0) : (!spirv.ptr<vector<4xf32>, Input>) -> vector<4xf32>
      %14 = "spirv.CompositeExtract"(%13) <{indices = [1 : i32]}> : (vector<4xf32>) -> f32
      %15 = spirv.ConvertFToS %14 : f32 to i32
      %16 = spirv.IMul %15, %cst64_i32 : i32
      %17 = spirv.IAdd %12, %16 : i32
      "spirv.Store"(%9, %17) : (!spirv.ptr<i32, Function>, i32) -> ()
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      "spirv.Store"(%8, %cst0_i32) : (!spirv.ptr<i32, Function>, i32) -> ()
      %cst0_i32_1 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      "spirv.Store"(%7, %cst0_i32_1) : (!spirv.ptr<i32, Function>, i32) -> ()
      "spirv.mlir.loop"() <{loop_control = #spirv.loop_control<None>}> ({
        spirv.Branch ^bb1
      ^bb1:
        %38 = "spirv.Load"(%7) : (!spirv.ptr<i32, Function>) -> i32
        %cst10_i32 = "spirv.Constant"() <{value = 10 : i32}> : () -> i32
        %39 = spirv.SLessThan %38, %cst10_i32 : i32
        "spirv.BranchConditional"(%39)[^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb2:
        %40 = "spirv.Load"(%8) : (!spirv.ptr<i32, Function>) -> i32
        %41 = "spirv.Load"(%7) : (!spirv.ptr<i32, Function>) -> i32
        %42 = "spirv.Load"(%9) : (!spirv.ptr<i32, Function>) -> i32
        %43 = spirv.IMul %41, %42 : i32
        %cst3_i32 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
        %44 = spirv.IAdd %43, %cst3_i32 : i32
        %45 = spirv.IAdd %40, %44 : i32
        "spirv.Store"(%8, %45) : (!spirv.ptr<i32, Function>, i32) -> ()
        %46 = "spirv.Load"(%7) : (!spirv.ptr<i32, Function>) -> i32
        %cst1_i32_8 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %47 = spirv.IAdd %46, %cst1_i32_8 : i32
        "spirv.Store"(%7, %47) : (!spirv.ptr<i32, Function>, i32) -> ()
        spirv.Branch ^bb1
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %cst1_i32 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      "spirv.Store"(%6, %cst1_i32) : (!spirv.ptr<i32, Function>, i32) -> ()
      %cst0_i32_2 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      "spirv.Store"(%5, %cst0_i32_2) : (!spirv.ptr<i32, Function>, i32) -> ()
      "spirv.mlir.loop"() <{loop_control = #spirv.loop_control<None>}> ({
        spirv.Branch ^bb1
      ^bb1:
        %38 = "spirv.Load"(%5) : (!spirv.ptr<i32, Function>) -> i32
        %39 = "spirv.Load"(%9) : (!spirv.ptr<i32, Function>) -> i32
        %cst7_i32 = "spirv.Constant"() <{value = 7 : i32}> : () -> i32
        %40 = spirv.BitwiseAnd %39, %cst7_i32 : i32
        %41 = spirv.SLessThan %38, %40 : i32
        "spirv.BranchConditional"(%41)[^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb2:
        %42 = "spirv.Load"(%6) : (!spirv.ptr<i32, Function>) -> i32
        %cst3_i32 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
        %43 = spirv.IMul %42, %cst3_i32 : i32
        %44 = "spirv.Load"(%5) : (!spirv.ptr<i32, Function>) -> i32
        %45 = spirv.IAdd %43, %44 : i32
        "spirv.Store"(%6, %45) : (!spirv.ptr<i32, Function>, i32) -> ()
        %46 = "spirv.Load"(%5) : (!spirv.ptr<i32, Function>) -> i32
        %cst1_i32_8 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %47 = spirv.IAdd %46, %cst1_i32_8 : i32
        "spirv.Store"(%5, %47) : (!spirv.ptr<i32, Function>, i32) -> ()
        spirv.Branch ^bb1
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %18 = "spirv.Load"(%9) : (!spirv.ptr<i32, Function>) -> i32
      %cst1_i32_3 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %19 = spirv.BitwiseAnd %18, %cst1_i32_3 : i32
      %cst1_i32_4 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %20 = spirv.IEqual %19, %cst1_i32_4 : i32
      %21 = "spirv.Load"(%9) : (!spirv.ptr<i32, Function>) -> i32
      %cst5_i32 = "spirv.Constant"() <{value = 5 : i32}> : () -> i32
      %22 = spirv.IMul %21, %cst5_i32 : i32
      %23 = "spirv.Load"(%9) : (!spirv.ptr<i32, Function>) -> i32
      %cst100_i32 = "spirv.Constant"() <{value = 100 : i32}> : () -> i32
      %24 = spirv.ISub %23, %cst100_i32 : i32
      %25 = spirv.Select %20, %22, %24 : i1, i32
      "spirv.Store"(%4, %25) : (!spirv.ptr<i32, Function>, i32) -> ()
      %cst0_i32_5 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      "spirv.Store"(%3, %cst0_i32_5) : (!spirv.ptr<i32, Function>, i32) -> ()
      %cst0_i32_6 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      "spirv.Store"(%2, %cst0_i32_6) : (!spirv.ptr<i32, Function>, i32) -> ()
      "spirv.mlir.loop"() <{loop_control = #spirv.loop_control<None>}> ({
        spirv.Branch ^bb1
      ^bb1:
        %38 = "spirv.Load"(%2) : (!spirv.ptr<i32, Function>) -> i32
        %cst5_i32_8 = "spirv.Constant"() <{value = 5 : i32}> : () -> i32
        %39 = spirv.SLessThan %38, %cst5_i32_8 : i32
        "spirv.BranchConditional"(%39)[^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb2:
        %40 = "spirv.Load"(%9) : (!spirv.ptr<i32, Function>) -> i32
        %41 = "spirv.Load"(%2) : (!spirv.ptr<i32, Function>) -> i32
        %42 = spirv.ShiftRightArithmetic %40, %41 : i32, i32
        %cst1_i32_9 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %43 = spirv.BitwiseAnd %42, %cst1_i32_9 : i32
        %cst1_i32_10 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %44 = spirv.IEqual %43, %cst1_i32_10 : i32
        "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
          "spirv.BranchConditional"(%44)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
        ^bb1:
          %47 = "spirv.Load"(%3) : (!spirv.ptr<i32, Function>) -> i32
          %48 = "spirv.Load"(%2) : (!spirv.ptr<i32, Function>) -> i32
          %49 = "spirv.Load"(%2) : (!spirv.ptr<i32, Function>) -> i32
          %50 = spirv.IMul %48, %49 : i32
          %51 = spirv.IAdd %47, %50 : i32
          "spirv.Store"(%3, %51) : (!spirv.ptr<i32, Function>, i32) -> ()
          spirv.Branch ^bb3
        ^bb2:
          %52 = "spirv.Load"(%3) : (!spirv.ptr<i32, Function>) -> i32
          %cst1_i32_12 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
          %53 = spirv.ISub %52, %cst1_i32_12 : i32
          "spirv.Store"(%3, %53) : (!spirv.ptr<i32, Function>, i32) -> ()
          spirv.Branch ^bb3
        ^bb3:
          spirv.mlir.merge
        }) : () -> ()
        %45 = "spirv.Load"(%2) : (!spirv.ptr<i32, Function>) -> i32
        %cst1_i32_11 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %46 = spirv.IAdd %45, %cst1_i32_11 : i32
        "spirv.Store"(%2, %46) : (!spirv.ptr<i32, Function>, i32) -> ()
        spirv.Branch ^bb1
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %26 = "spirv.Load"(%9) : (!spirv.ptr<i32, Function>) -> i32
      "spirv.Store"(%1, %26) : (!spirv.ptr<i32, Function>, i32) -> ()
      %cst0_i32_7 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      "spirv.Store"(%0, %cst0_i32_7) : (!spirv.ptr<i32, Function>, i32) -> ()
      "spirv.mlir.loop"() <{loop_control = #spirv.loop_control<None>}> ({
        spirv.Branch ^bb1
      ^bb1:
        %38 = "spirv.Load"(%1) : (!spirv.ptr<i32, Function>) -> i32
        %cst1_i32_8 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %39 = spirv.SGreaterThan %38, %cst1_i32_8 : i32
        %40 = "spirv.Load"(%0) : (!spirv.ptr<i32, Function>) -> i32
        %cst64_i32_9 = "spirv.Constant"() <{value = 64 : i32}> : () -> i32
        %41 = spirv.SLessThan %40, %cst64_i32_9 : i32
        %42 = spirv.LogicalAnd %39, %41 : i1
        "spirv.BranchConditional"(%42)[^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb2:
        %43 = "spirv.Load"(%1) : (!spirv.ptr<i32, Function>) -> i32
        %cst1_i32_10 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %44 = spirv.BitwiseAnd %43, %cst1_i32_10 : i32
        %cst0_i32_11 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
        %45 = spirv.IEqual %44, %cst0_i32_11 : i32
        %46 = "spirv.Load"(%1) : (!spirv.ptr<i32, Function>) -> i32
        %cst1_i32_12 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %47 = spirv.ShiftRightArithmetic %46, %cst1_i32_12 : i32, i32
        %cst3_i32 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
        %48 = "spirv.Load"(%1) : (!spirv.ptr<i32, Function>) -> i32
        %49 = spirv.IMul %48, %cst3_i32 : i32
        %cst1_i32_13 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %50 = spirv.IAdd %49, %cst1_i32_13 : i32
        %51 = spirv.Select %45, %47, %50 : i1, i32
        "spirv.Store"(%1, %51) : (!spirv.ptr<i32, Function>, i32) -> ()
        %52 = "spirv.Load"(%0) : (!spirv.ptr<i32, Function>) -> i32
        %cst1_i32_14 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
        %53 = spirv.IAdd %52, %cst1_i32_14 : i32
        "spirv.Store"(%0, %53) : (!spirv.ptr<i32, Function>, i32) -> ()
        spirv.Branch ^bb1
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %27 = "spirv.Load"(%8) : (!spirv.ptr<i32, Function>) -> i32
      %28 = spirv.ConvertSToF %27 : i32 to f32
      %29 = "spirv.Load"(%6) : (!spirv.ptr<i32, Function>) -> i32
      %30 = spirv.ConvertSToF %29 : i32 to f32
      %31 = "spirv.Load"(%4) : (!spirv.ptr<i32, Function>) -> i32
      %32 = "spirv.Load"(%3) : (!spirv.ptr<i32, Function>) -> i32
      %33 = spirv.IAdd %31, %32 : i32
      %34 = spirv.ConvertSToF %33 : i32 to f32
      %35 = "spirv.Load"(%0) : (!spirv.ptr<i32, Function>) -> i32
      %36 = spirv.ConvertSToF %35 : i32 to f32
      %37 = spirv.CompositeConstruct %28, %30, %34, %36 : (f32, f32, f32, f32) -> vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %37) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_uv, @o_color, @gl_FragCoord]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
