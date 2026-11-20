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
      %cst_f32_2 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %a_pos_addr_3 = spirv.mlir.addressof @a_pos : !spirv.ptr<vector<2xf32>, Input>
      %10 = "spirv.Load"(%a_pos_addr_3) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %11 = "spirv.CompositeExtract"(%10) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %cst0_i32_4 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr_5 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>
      %12 = spirv.AccessChain %u_addr_5[%cst0_i32_4] : !spirv.ptr<!spirv.struct<(vector<2xf32> [0]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %13 = "spirv.Load"(%12) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %14 = "spirv.CompositeExtract"(%13) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %15 = spirv.FDiv %11, %14 : f32
      %cst_f32_6 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
      %16 = spirv.FMul %15, %cst_f32_6 : f32
      %17 = spirv.FSub %16, %cst_f32_2 : f32
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
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "v_pos", type = !spirv.ptr<vector<2xf32>, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 1 : i32, sym_name = "v_cover", type = !spirv.ptr<f32, Input>}> : () -> ()
    "spirv.GlobalVariable"() <{binding = 1 : i32, descriptor_set = 0 : i32, sym_name = "u", type = !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>}> : () -> ()
    "spirv.GlobalVariable"() <{location = 0 : i32, sym_name = "o_color", type = !spirv.ptr<vector<4xf32>, Output>}> : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = (f32) -> f32, sym_name = "unit"}> ({
    ^bb0(%arg0: f32):
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      "spirv.Store"(%0, %arg0) : (!spirv.ptr<f32, Function>, f32) -> ()
      %1 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %2 = spirv.GL.FMax %1, %cst_f32 : f32
      %cst_f32_0 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %3 = spirv.GL.FMin %2, %cst_f32_0 : f32
      spirv.ReturnValue %3 : f32
    }) : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = (f32, f32) -> f32, sym_name = "angle"}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %4 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %5 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %6 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      "spirv.Store"(%6, %arg0) : (!spirv.ptr<f32, Function>, f32) -> ()
      "spirv.Store"(%5, %arg1) : (!spirv.ptr<f32, Function>, f32) -> ()
      %7 = "spirv.Load"(%5) : (!spirv.ptr<f32, Function>) -> f32
      %8 = spirv.GL.FAbs %7 : f32
      "spirv.Store"(%4, %8) : (!spirv.ptr<f32, Function>, f32) -> ()
      %9 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %10 = spirv.GL.FAbs %9 : f32
      "spirv.Store"(%3, %10) : (!spirv.ptr<f32, Function>, f32) -> ()
      %11 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %12 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %13 = spirv.GL.FMin %11, %12 : f32
      %14 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %15 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %16 = spirv.GL.FMax %14, %15 : f32
      %cst_f32 = "spirv.Constant"() <{value = 9.99999996E-13 : f32}> : () -> f32
      %17 = spirv.GL.FMax %16, %cst_f32 : f32
      %18 = spirv.FDiv %13, %17 : f32
      "spirv.Store"(%2, %18) : (!spirv.ptr<f32, Function>, f32) -> ()
      %19 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %20 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %21 = spirv.FMul %19, %20 : f32
      "spirv.Store"(%1, %21) : (!spirv.ptr<f32, Function>, f32) -> ()
      %cst_f32_0 = "spirv.Constant"() <{value = 0.0464964733 : f32}> : () -> f32
      %cst_f32_1 = "spirv.Constant"() <{value = -0.0464964733 : f32}> : () -> f32
      %22 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %23 = spirv.FMul %cst_f32_1, %22 : f32
      %cst_f32_2 = "spirv.Constant"() <{value = 0.159314215 : f32}> : () -> f32
      %24 = spirv.FAdd %23, %cst_f32_2 : f32
      %25 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %26 = spirv.FMul %24, %25 : f32
      %cst_f32_3 = "spirv.Constant"() <{value = 0.327622771 : f32}> : () -> f32
      %27 = spirv.FSub %26, %cst_f32_3 : f32
      %28 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
      %29 = spirv.FMul %27, %28 : f32
      %30 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %31 = spirv.FMul %29, %30 : f32
      %32 = "spirv.Load"(%2) : (!spirv.ptr<f32, Function>) -> f32
      %33 = spirv.FAdd %31, %32 : f32
      "spirv.Store"(%0, %33) : (!spirv.ptr<f32, Function>, f32) -> ()
      %34 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %35 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %36 = spirv.FOrdGreaterThan %34, %35 : f32
      "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
        "spirv.BranchConditional"(%36)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb1:
        %cst_f32_6 = "spirv.Constant"() <{value = 1.57079637 : f32}> : () -> f32
        %42 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
        %43 = spirv.FSub %cst_f32_6, %42 : f32
        "spirv.Store"(%0, %43) : (!spirv.ptr<f32, Function>, f32) -> ()
        spirv.Branch ^bb3
      ^bb2:
        spirv.Branch ^bb3
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %37 = "spirv.Load"(%5) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_4 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %38 = spirv.FOrdLessThan %37, %cst_f32_4 : f32
      "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
        "spirv.BranchConditional"(%38)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb1:
        %cst_f32_6 = "spirv.Constant"() <{value = 3.14159274 : f32}> : () -> f32
        %42 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
        %43 = spirv.FSub %cst_f32_6, %42 : f32
        "spirv.Store"(%0, %43) : (!spirv.ptr<f32, Function>, f32) -> ()
        spirv.Branch ^bb3
      ^bb2:
        spirv.Branch ^bb3
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %39 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %cst_f32_5 = "spirv.Constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %40 = spirv.FOrdLessThan %39, %cst_f32_5 : f32
      "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
        "spirv.BranchConditional"(%40)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb1:
        %cst_f32_6 = "spirv.Constant"() <{value = 6.28318548 : f32}> : () -> f32
        %42 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
        %43 = spirv.FSub %cst_f32_6, %42 : f32
        "spirv.Store"(%0, %43) : (!spirv.ptr<f32, Function>, f32) -> ()
        spirv.Branch ^bb3
      ^bb2:
        spirv.Branch ^bb3
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %41 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      spirv.ReturnValue %41 : f32
    }) : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = (vector<4xf32>, f32, f32, f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>, sym_name = "between"}> ({
    ^bb0(%arg0: vector<4xf32>, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: vector<4xf32>, %arg5: vector<4xf32>):
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<4xf32>, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<4xf32>, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %4 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %5 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %6 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<4xf32>, Function>
      "spirv.Store"(%6, %arg0) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      "spirv.Store"(%5, %arg1) : (!spirv.ptr<f32, Function>, f32) -> ()
      "spirv.Store"(%4, %arg2) : (!spirv.ptr<f32, Function>, f32) -> ()
      "spirv.Store"(%3, %arg3) : (!spirv.ptr<f32, Function>, f32) -> ()
      "spirv.Store"(%2, %arg4) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      "spirv.Store"(%1, %arg5) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      %7 = "spirv.Load"(%5) : (!spirv.ptr<f32, Function>) -> f32
      %8 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %9 = spirv.FOrdGreaterThan %7, %8 : f32
      "spirv.BranchConditional"(%9)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> {selection_merge = 2 : i64} : (i1) -> ()
    ^bb1:
      %10 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %11 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %12 = spirv.FOrdGreaterThan %10, %11 : f32
      %13 = "spirv.Load"(%5) : (!spirv.ptr<f32, Function>) -> f32
      %14 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %15 = spirv.FSub %13, %14 : f32
      %16 = "spirv.Load"(%3) : (!spirv.ptr<f32, Function>) -> f32
      %17 = "spirv.Load"(%4) : (!spirv.ptr<f32, Function>) -> f32
      %18 = spirv.FSub %16, %17 : f32
      %19 = spirv.FDiv %15, %18 : f32
      %20 = spirv.FunctionCall @unit(%19) : (f32) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
      %21 = spirv.Select %12, %20, %cst_f32 : i1, f32
      "spirv.Store"(%0, %21) : (!spirv.ptr<f32, Function>, f32) -> ()
      %22 = "spirv.Load"(%2) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %23 = "spirv.Load"(%1) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %24 = "spirv.Load"(%2) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %25 = spirv.FSub %23, %24 : vector<4xf32>
      %26 = "spirv.Load"(%0) : (!spirv.ptr<f32, Function>) -> f32
      %27 = spirv.CompositeConstruct %26, %26, %26, %26 : (f32, f32, f32, f32) -> vector<4xf32>
      %28 = spirv.FMul %25, %27 : vector<4xf32>
      %29 = spirv.FAdd %22, %28 : vector<4xf32>
      spirv.ReturnValue %29 : vector<4xf32>
    ^bb2:
      %30 = "spirv.Load"(%6) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      spirv.ReturnValue %30 : vector<4xf32>
    }) : () -> ()
    "spirv.func"() <{function_control = #spirv.function_control<None>, function_type = () -> (), sym_name = "main"}> ({
      %0 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<4xf32>, Function>
      %1 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %2 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %3 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %4 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %5 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %6 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<f32, Function>
      %7 = "spirv.Variable"() <{storage_class = #spirv.storage_class<Function>}> : () -> !spirv.ptr<vector<2xf32>, Function>
      %cst0_i32 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %8 = spirv.AccessChain %u_addr[%cst0_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %9 = "spirv.Load"(%8) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %10 = "spirv.CompositeExtract"(%9) <{indices = [0 : i32]}> : (vector<3xf32>) -> f32
      %v_pos_addr = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Input>
      %11 = "spirv.Load"(%v_pos_addr) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %12 = "spirv.CompositeExtract"(%11) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %13 = spirv.FMul %10, %12 : f32
      %cst0_i32_0 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr_1 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %14 = spirv.AccessChain %u_addr_1[%cst0_i32_0] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %15 = "spirv.Load"(%14) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %16 = "spirv.CompositeExtract"(%15) <{indices = [1 : i32]}> : (vector<3xf32>) -> f32
      %v_pos_addr_2 = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Input>
      %17 = "spirv.Load"(%v_pos_addr_2) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %18 = "spirv.CompositeExtract"(%17) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %19 = spirv.FMul %16, %18 : f32
      %20 = spirv.FAdd %13, %19 : f32
      %cst0_i32_3 = "spirv.Constant"() <{value = 0 : i32}> : () -> i32
      %u_addr_4 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %21 = spirv.AccessChain %u_addr_4[%cst0_i32_3] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %22 = "spirv.Load"(%21) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %23 = "spirv.CompositeExtract"(%22) <{indices = [2 : i32]}> : (vector<3xf32>) -> f32
      %24 = spirv.FAdd %20, %23 : f32
      %cst1_i32 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %u_addr_5 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %25 = spirv.AccessChain %u_addr_5[%cst1_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %26 = "spirv.Load"(%25) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %27 = "spirv.CompositeExtract"(%26) <{indices = [0 : i32]}> : (vector<3xf32>) -> f32
      %v_pos_addr_6 = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Input>
      %28 = "spirv.Load"(%v_pos_addr_6) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %29 = "spirv.CompositeExtract"(%28) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %30 = spirv.FMul %27, %29 : f32
      %cst1_i32_7 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %u_addr_8 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %31 = spirv.AccessChain %u_addr_8[%cst1_i32_7] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %32 = "spirv.Load"(%31) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %33 = "spirv.CompositeExtract"(%32) <{indices = [1 : i32]}> : (vector<3xf32>) -> f32
      %v_pos_addr_9 = spirv.mlir.addressof @v_pos : !spirv.ptr<vector<2xf32>, Input>
      %34 = "spirv.Load"(%v_pos_addr_9) : (!spirv.ptr<vector<2xf32>, Input>) -> vector<2xf32>
      %35 = "spirv.CompositeExtract"(%34) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %36 = spirv.FMul %33, %35 : f32
      %37 = spirv.FAdd %30, %36 : f32
      %cst1_i32_10 = "spirv.Constant"() <{value = 1 : i32}> : () -> i32
      %u_addr_11 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %38 = spirv.AccessChain %u_addr_11[%cst1_i32_10] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<3xf32>, Uniform>
      %39 = "spirv.Load"(%38) : (!spirv.ptr<vector<3xf32>, Uniform>) -> vector<3xf32>
      %40 = "spirv.CompositeExtract"(%39) <{indices = [2 : i32]}> : (vector<3xf32>) -> f32
      %41 = spirv.FAdd %37, %40 : f32
      %42 = spirv.CompositeConstruct %24, %41 : (f32, f32) -> vector<2xf32>
      "spirv.Store"(%7, %42) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
      %cst3_i32 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
      %u_addr_12 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %43 = spirv.AccessChain %u_addr_12[%cst3_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %44 = "spirv.Load"(%43) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %45 = "spirv.CompositeExtract"(%44) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
      %cst_f32 = "spirv.Constant"() <{value = 5.000000e-01 : f32}> : () -> f32
      %46 = spirv.FOrdLessThan %45, %cst_f32 : f32
      "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
        "spirv.BranchConditional"(%46)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb1:
        %cst2_i32 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
        %u_addr_37 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
        %96 = spirv.AccessChain %u_addr_37[%cst2_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
        %97 = "spirv.Load"(%96) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
        %98 = spirv.VectorShuffle [2 : i32, 3 : i32] %97, %97 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
        %cst2_i32_38 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
        %u_addr_39 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
        %99 = spirv.AccessChain %u_addr_39[%cst2_i32_38] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
        %100 = "spirv.Load"(%99) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
        %101 = spirv.VectorShuffle [0 : i32, 1 : i32] %100, %100 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
        %102 = spirv.FSub %98, %101 : vector<2xf32>
        "spirv.Store"(%5, %102) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
        %103 = "spirv.Load"(%7) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %cst2_i32_40 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
        %u_addr_41 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
        %104 = spirv.AccessChain %u_addr_41[%cst2_i32_40] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
        %105 = "spirv.Load"(%104) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
        %106 = spirv.VectorShuffle [0 : i32, 1 : i32] %105, %105 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
        %107 = spirv.FSub %103, %106 : vector<2xf32>
        "spirv.Store"(%4, %107) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
        %108 = "spirv.Load"(%4) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %109 = "spirv.CompositeExtract"(%108) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
        %110 = "spirv.Load"(%5) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %111 = "spirv.CompositeExtract"(%110) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
        %112 = spirv.FMul %109, %111 : f32
        %113 = "spirv.Load"(%4) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %114 = "spirv.CompositeExtract"(%113) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
        %115 = "spirv.Load"(%5) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %116 = "spirv.CompositeExtract"(%115) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
        %117 = spirv.FMul %114, %116 : f32
        %118 = spirv.FAdd %112, %117 : f32
        %119 = "spirv.Load"(%5) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %120 = "spirv.CompositeExtract"(%119) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
        %121 = "spirv.Load"(%5) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %122 = "spirv.CompositeExtract"(%121) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
        %123 = spirv.FMul %120, %122 : f32
        %124 = "spirv.Load"(%5) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %125 = "spirv.CompositeExtract"(%124) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
        %126 = "spirv.Load"(%5) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
        %127 = "spirv.CompositeExtract"(%126) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
        %128 = spirv.FMul %125, %127 : f32
        %129 = spirv.FAdd %123, %128 : f32
        %cst_f32_42 = "spirv.Constant"() <{value = 9.99999996E-13 : f32}> : () -> f32
        %130 = spirv.GL.FMax %129, %cst_f32_42 : f32
        %131 = spirv.FDiv %118, %130 : f32
        "spirv.Store"(%6, %131) : (!spirv.ptr<f32, Function>, f32) -> ()
        spirv.Branch ^bb3
      ^bb2:
        %cst3_i32_43 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
        %u_addr_44 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
        %132 = spirv.AccessChain %u_addr_44[%cst3_i32_43] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
        %133 = "spirv.Load"(%132) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
        %134 = "spirv.CompositeExtract"(%133) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
        %cst_f32_45 = "spirv.Constant"() <{value = 1.500000e+00 : f32}> : () -> f32
        %135 = spirv.FOrdLessThan %134, %cst_f32_45 : f32
        "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
          "spirv.BranchConditional"(%135)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
        ^bb1:
          %136 = "spirv.Load"(%7) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %cst2_i32_46 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
          %u_addr_47 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
          %137 = spirv.AccessChain %u_addr_47[%cst2_i32_46] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
          %138 = "spirv.Load"(%137) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
          %139 = spirv.VectorShuffle [0 : i32, 1 : i32] %138, %138 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
          %140 = spirv.FSub %136, %139 : vector<2xf32>
          "spirv.Store"(%3, %140) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
          %141 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %142 = "spirv.CompositeExtract"(%141) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
          %143 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %144 = "spirv.CompositeExtract"(%143) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
          %145 = spirv.FMul %142, %144 : f32
          %146 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %147 = "spirv.CompositeExtract"(%146) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
          %148 = "spirv.Load"(%3) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %149 = "spirv.CompositeExtract"(%148) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
          %150 = spirv.FMul %147, %149 : f32
          %151 = spirv.FAdd %145, %150 : f32
          %152 = spirv.GL.Sqrt %151 : f32
          %cst2_i32_48 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
          %u_addr_49 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
          %153 = spirv.AccessChain %u_addr_49[%cst2_i32_48] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
          %154 = "spirv.Load"(%153) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
          %155 = "spirv.CompositeExtract"(%154) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
          %cst_f32_50 = "spirv.Constant"() <{value = 9.99999997E-7 : f32}> : () -> f32
          %156 = spirv.GL.FMax %155, %cst_f32_50 : f32
          %157 = spirv.FDiv %152, %156 : f32
          "spirv.Store"(%6, %157) : (!spirv.ptr<f32, Function>, f32) -> ()
          spirv.Branch ^bb3
        ^bb2:
          %158 = "spirv.Load"(%7) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %cst2_i32_51 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
          %u_addr_52 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
          %159 = spirv.AccessChain %u_addr_52[%cst2_i32_51] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
          %160 = "spirv.Load"(%159) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
          %161 = spirv.VectorShuffle [0 : i32, 1 : i32] %160, %160 : vector<4xf32>, vector<4xf32> -> vector<2xf32>
          %162 = spirv.FSub %158, %161 : vector<2xf32>
          "spirv.Store"(%2, %162) : (!spirv.ptr<vector<2xf32>, Function>, vector<2xf32>) -> ()
          %163 = "spirv.Load"(%2) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %164 = "spirv.CompositeExtract"(%163) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
          %165 = "spirv.Load"(%2) : (!spirv.ptr<vector<2xf32>, Function>) -> vector<2xf32>
          %166 = "spirv.CompositeExtract"(%165) <{indices = [0 : i32]}> : (vector<2xf32>) -> f32
          %167 = spirv.FunctionCall @angle(%164, %166) : (f32, f32) -> f32
          %cst2_i32_53 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
          %u_addr_54 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
          %168 = spirv.AccessChain %u_addr_54[%cst2_i32_53] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
          %169 = "spirv.Load"(%168) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
          %170 = "spirv.CompositeExtract"(%169) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
          %171 = spirv.FSub %167, %170 : f32
          %cst2_i32_55 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
          %u_addr_56 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
          %172 = spirv.AccessChain %u_addr_56[%cst2_i32_55] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
          %173 = "spirv.Load"(%172) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
          %174 = "spirv.CompositeExtract"(%173) <{indices = [3 : i32]}> : (vector<4xf32>) -> f32
          %cst2_i32_57 = "spirv.Constant"() <{value = 2 : i32}> : () -> i32
          %u_addr_58 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
          %175 = spirv.AccessChain %u_addr_58[%cst2_i32_57] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
          %176 = "spirv.Load"(%175) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
          %177 = "spirv.CompositeExtract"(%176) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
          %178 = spirv.FSub %174, %177 : f32
          %cst_f32_59 = "spirv.Constant"() <{value = 9.99999997E-7 : f32}> : () -> f32
          %179 = spirv.GL.FMax %178, %cst_f32_59 : f32
          %180 = spirv.FDiv %171, %179 : f32
          "spirv.Store"(%6, %180) : (!spirv.ptr<f32, Function>, f32) -> ()
          spirv.Branch ^bb3
        ^bb3:
          spirv.mlir.merge
        }) : () -> ()
        spirv.Branch ^bb3
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %cst3_i32_13 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
      %u_addr_14 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %47 = spirv.AccessChain %u_addr_14[%cst3_i32_13] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
      %48 = "spirv.Load"(%47) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
      %49 = "spirv.CompositeExtract"(%48) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
      %cst_f32_15 = "spirv.Constant"() <{value = 1.500000e+00 : f32}> : () -> f32
      %50 = spirv.FOrdGreaterThan %49, %cst_f32_15 : f32
      "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
        "spirv.BranchConditional"(%50)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
      ^bb1:
        %96 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
        %cst_f32_37 = "spirv.Constant"() <{value = 2.000000e+00 : f32}> : () -> f32
        %97 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
        %cst_f32_38 = "spirv.Constant"() <{value = 5.000000e-01 : f32}> : () -> f32
        %98 = spirv.FMul %97, %cst_f32_38 : f32
        %99 = spirv.GL.Floor %98 : f32
        %100 = spirv.FMul %99, %cst_f32_37 : f32
        %101 = spirv.FSub %96, %100 : f32
        "spirv.Store"(%1, %101) : (!spirv.ptr<f32, Function>, f32) -> ()
        %cst_f32_39 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
        %102 = "spirv.Load"(%1) : (!spirv.ptr<f32, Function>) -> f32
        %cst_f32_40 = "spirv.Constant"() <{value = 1.000000e+00 : f32}> : () -> f32
        %103 = spirv.FSub %102, %cst_f32_40 : f32
        %104 = spirv.GL.FAbs %103 : f32
        %105 = spirv.FSub %cst_f32_39, %104 : f32
        "spirv.Store"(%6, %105) : (!spirv.ptr<f32, Function>, f32) -> ()
        spirv.Branch ^bb3
      ^bb2:
        %cst3_i32_41 = "spirv.Constant"() <{value = 3 : i32}> : () -> i32
        %u_addr_42 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
        %106 = spirv.AccessChain %u_addr_42[%cst3_i32_41] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<2xf32>, Uniform>
        %107 = "spirv.Load"(%106) : (!spirv.ptr<vector<2xf32>, Uniform>) -> vector<2xf32>
        %108 = "spirv.CompositeExtract"(%107) <{indices = [1 : i32]}> : (vector<2xf32>) -> f32
        %cst_f32_43 = "spirv.Constant"() <{value = 5.000000e-01 : f32}> : () -> f32
        %109 = spirv.FOrdGreaterThan %108, %cst_f32_43 : f32
        "spirv.mlir.selection"() <{selection_control = #spirv.selection_control<None>}> ({
          "spirv.BranchConditional"(%109)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
        ^bb1:
          %110 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
          %111 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
          %112 = spirv.GL.Floor %111 : f32
          %113 = spirv.FSub %110, %112 : f32
          "spirv.Store"(%6, %113) : (!spirv.ptr<f32, Function>, f32) -> ()
          spirv.Branch ^bb3
        ^bb2:
          %114 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
          %115 = spirv.FunctionCall @unit(%114) : (f32) -> f32
          "spirv.Store"(%6, %115) : (!spirv.ptr<f32, Function>, f32) -> ()
          spirv.Branch ^bb3
        ^bb3:
          spirv.mlir.merge
        }) : () -> ()
        spirv.Branch ^bb3
      ^bb3:
        spirv.mlir.merge
      }) : () -> ()
      %cst5_i32 = "spirv.Constant"() <{value = 5 : i32}> : () -> i32
      %u_addr_16 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %51 = spirv.AccessChain %u_addr_16[%cst5_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %52 = "spirv.Load"(%51) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      "spirv.Store"(%0, %52) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      %53 = "spirv.Load"(%0) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %54 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %cst4_i32 = "spirv.Constant"() <{value = 4 : i32}> : () -> i32
      %u_addr_17 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %55 = spirv.AccessChain %u_addr_17[%cst4_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %56 = "spirv.Load"(%55) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %57 = "spirv.CompositeExtract"(%56) <{indices = [0 : i32]}> : (vector<4xf32>) -> f32
      %cst4_i32_18 = "spirv.Constant"() <{value = 4 : i32}> : () -> i32
      %u_addr_19 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %58 = spirv.AccessChain %u_addr_19[%cst4_i32_18] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %59 = "spirv.Load"(%58) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %60 = "spirv.CompositeExtract"(%59) <{indices = [1 : i32]}> : (vector<4xf32>) -> f32
      %cst5_i32_20 = "spirv.Constant"() <{value = 5 : i32}> : () -> i32
      %u_addr_21 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %61 = spirv.AccessChain %u_addr_21[%cst5_i32_20] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %62 = "spirv.Load"(%61) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %cst6_i32 = "spirv.Constant"() <{value = 6 : i32}> : () -> i32
      %u_addr_22 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %63 = spirv.AccessChain %u_addr_22[%cst6_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %64 = "spirv.Load"(%63) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %65 = spirv.FunctionCall @between(%53, %54, %57, %60, %62, %64) : (vector<4xf32>, f32, f32, f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
      "spirv.Store"(%0, %65) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      %66 = "spirv.Load"(%0) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %67 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %cst4_i32_23 = "spirv.Constant"() <{value = 4 : i32}> : () -> i32
      %u_addr_24 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %68 = spirv.AccessChain %u_addr_24[%cst4_i32_23] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %69 = "spirv.Load"(%68) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %70 = "spirv.CompositeExtract"(%69) <{indices = [1 : i32]}> : (vector<4xf32>) -> f32
      %cst4_i32_25 = "spirv.Constant"() <{value = 4 : i32}> : () -> i32
      %u_addr_26 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %71 = spirv.AccessChain %u_addr_26[%cst4_i32_25] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %72 = "spirv.Load"(%71) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %73 = "spirv.CompositeExtract"(%72) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
      %cst6_i32_27 = "spirv.Constant"() <{value = 6 : i32}> : () -> i32
      %u_addr_28 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %74 = spirv.AccessChain %u_addr_28[%cst6_i32_27] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %75 = "spirv.Load"(%74) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %cst7_i32 = "spirv.Constant"() <{value = 7 : i32}> : () -> i32
      %u_addr_29 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %76 = spirv.AccessChain %u_addr_29[%cst7_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %77 = "spirv.Load"(%76) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %78 = spirv.FunctionCall @between(%66, %67, %70, %73, %75, %77) : (vector<4xf32>, f32, f32, f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
      "spirv.Store"(%0, %78) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      %79 = "spirv.Load"(%0) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %80 = "spirv.Load"(%6) : (!spirv.ptr<f32, Function>) -> f32
      %cst4_i32_30 = "spirv.Constant"() <{value = 4 : i32}> : () -> i32
      %u_addr_31 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %81 = spirv.AccessChain %u_addr_31[%cst4_i32_30] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %82 = "spirv.Load"(%81) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %83 = "spirv.CompositeExtract"(%82) <{indices = [2 : i32]}> : (vector<4xf32>) -> f32
      %cst4_i32_32 = "spirv.Constant"() <{value = 4 : i32}> : () -> i32
      %u_addr_33 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %84 = spirv.AccessChain %u_addr_33[%cst4_i32_32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %85 = "spirv.Load"(%84) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %86 = "spirv.CompositeExtract"(%85) <{indices = [3 : i32]}> : (vector<4xf32>) -> f32
      %cst7_i32_34 = "spirv.Constant"() <{value = 7 : i32}> : () -> i32
      %u_addr_35 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %87 = spirv.AccessChain %u_addr_35[%cst7_i32_34] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %88 = "spirv.Load"(%87) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %cst8_i32 = "spirv.Constant"() <{value = 8 : i32}> : () -> i32
      %u_addr_36 = spirv.mlir.addressof @u : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>
      %89 = spirv.AccessChain %u_addr_36[%cst8_i32] : !spirv.ptr<!spirv.struct<(vector<3xf32> [0], vector<3xf32> [16], vector<4xf32> [32], vector<2xf32> [48], vector<4xf32> [64], vector<4xf32> [80], vector<4xf32> [96], vector<4xf32> [112], vector<4xf32> [128]), Block>, Uniform>, i32 -> !spirv.ptr<vector<4xf32>, Uniform>
      %90 = "spirv.Load"(%89) : (!spirv.ptr<vector<4xf32>, Uniform>) -> vector<4xf32>
      %91 = spirv.FunctionCall @between(%79, %80, %83, %86, %88, %90) : (vector<4xf32>, f32, f32, f32, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
      "spirv.Store"(%0, %91) : (!spirv.ptr<vector<4xf32>, Function>, vector<4xf32>) -> ()
      %92 = "spirv.Load"(%0) : (!spirv.ptr<vector<4xf32>, Function>) -> vector<4xf32>
      %v_cover_addr = spirv.mlir.addressof @v_cover : !spirv.ptr<f32, Input>
      %93 = "spirv.Load"(%v_cover_addr) : (!spirv.ptr<f32, Input>) -> f32
      %94 = spirv.CompositeConstruct %93, %93, %93, %93 : (f32, f32, f32, f32) -> vector<4xf32>
      %95 = spirv.FMul %92, %94 : vector<4xf32>
      %o_color_addr = spirv.mlir.addressof @o_color : !spirv.ptr<vector<4xf32>, Output>
      "spirv.Store"(%o_color_addr, %95) : (!spirv.ptr<vector<4xf32>, Output>, vector<4xf32>) -> ()
      spirv.Return
    }) : () -> ()
    "spirv.EntryPoint"() <{execution_model = #spirv.execution_model<Fragment>, fn = @main, interface = [@v_pos, @v_cover, @o_color]}> : () -> ()
    "spirv.ExecutionMode"() <{execution_mode = #spirv.execution_mode<OriginUpperLeft>, fn = @main, values = []}> : () -> ()
  }) {spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader, ImageQuery], []>, #spirv.resource_limits<>>} : () -> ()
}
