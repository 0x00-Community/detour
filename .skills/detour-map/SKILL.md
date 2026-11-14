---
name: detour-map
description: detour 仓库地图：每个 crate 管什么（kmd trait、xe、detour 方言与前端、xe3 方言、codegen、driver 运行时、cmdgen）、一个三角形从 MLIR 到 GPU 经过哪些文件和 pass、数据文件和对照文件在哪。第一次进库、找某件事该改哪里时读。
---

# detour 地图

## crate

| 路径 | crate | 管什么 |
| --- | --- | --- |
| `crates/core/src/kmd*` | `detour-core` | 内核接口的 trait: `Gpu` (`Bo` / `Vm` / `Queue` / `Fence`, GAT 借用设备), CPU 侧 `Host` (`hosts!` 宏, `Native`) |
| `crates/core/src/image.rs`, `pack.rs` | `detour-core` | 厂商无关的镜像: `image` 方言 (`Kernel::all`)、指令 op 宏 `attr_ops!`、`OpRows`、`rewrites!` / `Rewrites`、`ImageTarget`、`Layout`、`ImageEmit`; 表驱动打包 `Layouts` / `Template` |
| `crates/core/src/compiler/build.rs` | `detour-core` | 构造: `A`、`WithAttrs`、`Slots`、`RegType` |
| `crates/core/src/regalloc.rs` | `detour-core` | 线性扫描 `LinearScan` + `RegTarget` (各家的寄存器分配都用它; Intel 的 `xe3-regalloc` 只做消息载荷和改写) |
| `crates/core/src/raster.rs` | `detour-core` | 2D 光栅化的 CPU 一半: vello_common 条带 ⇒ 实例 + alpha 图集 + paint 图集 (`Strips`) |
| `crates/core/src/scene.rs`, `filter.rs` | `detour-core` | acde 画进来的 2D 场景: `PaintScene`、`Scene` / `RenderCommand`、`Paint`、CSS 滤镜 `Filter` |
| `src/` (仓库根) | `detour` | 聚合导出: 场景 + 一个 GPU 型号一个 feature (`xe3` / `agx-g13` / `rdna2`) + `fallback-vulkan` 回退; `Backend::pick` |
| `src/main.rs`, `src/cli/` | `detour` (bin, feature `cli`) | 调试工具: `main.rs` 只组装, 一个 clap 子命令一个文件 (`devices`: render node ⇒ 驱动、型号、支持到哪一步) |
| `benches/triangle`, `benches/triangle3d`, `benches/interop` | `triangle`, `triangle3d`, `interop` | 窗口示例 (winit + softbuffer, Xe3): 三角形; 转动的 3D 三角形; interop: fallback 的 dma-buf 图在 Xe3 页面里画, 对 vello_cpu. `triangle` 的 lib 是窗口骨架 (`Scene`, `Shown`) |
| `crates/fallback` | `detour-fallback` | Vulkan 回退 (`fallback-vulkan`): `Vello`、`Composer`、`Device` (`DeviceRequest` 分档)、`Raster`、`Offscreen` |
| `crates/core/src/compiler*` | `detour-core` | `detour` 方言 (`compiler/dialect.rs`)、`Stage`、`Compiler`、前端 pass: `detour-from-spirv` (`spirv.rs`)、`detour-from-gpu` (`gpu.rs`)、`detour-expand-clear` (`clear.rs`); `FRONTEND` + `FrontendPasses` |
| `crates/amd/amdgpu` | `detour-amd-amdgpu` | amdgpu 的 ioctl (`sys.rs` 私有), `impl Gpu for Amdgpu` |
| `crates/apple/codegen` | `detour-apple-codegen` | 桥接 `convert-detour-to-gpu` (detour 着色器 ⇒ `gpu.func`) + mlirpass 的 AGX 目标 (`agx` 方言、`convert-gpu-to-agx`、寄存器、`AgxDispatch`、`AGX_IMAGE` 都在 mlirpass) |
| `crates/amd/codegen` | `detour-amd-codegen` | `rdna2-precheck` + detour-core 的桥接 + mlirpass 的 RDNA2 目标 (`rdna2` 方言、`convert-gpu-to-rdna2`、寄存器、PM4、`Rdna2Dispatch`、`RDNA2_IMAGE` 都在 mlirpass) |
| `crates/apple/asahi` | `detour-apple-asahi` | asahi 的 ioctl (`sys.rs` 私有), `impl Gpu for Asahi` (Apple M1 / M2, Asahi Linux) |
| `crates/intel/xe` | `detour-intel-xe` | xe 的 ioctl (`sys.rs` 私有), `impl Gpu for Xe` |
| `crates/intel/dialect` | `detour-intel-dialect` | `xe3` 方言: genxml 的 op 和布局表 (`cmd.gen.rs`, 生成的)、`XE3` 表、EU 指令 (`eu.rs` 编码, `asm.rs` ⇄ op, `msg.rs` 消息描述符); 打包引擎和容器在 detour-core (`pack.rs`、`image.rs`) |
| `crates/intel/codegen` | `detour-intel-codegen` | `BACKEND`: `convert-detour-to-xe3` (`lib.rs` + `shader.rs` + `render.rs`)、`xe3-regalloc` (`regalloc.rs`)、`xe3-swsb` (`swsb.rs`)、`xe3-pad` / `xe3-end` / `xe3-layout` + `Xe3` (`ImageTarget`) + `XE3_IMAGE` (`layout.rs`); `Codegen` |
| `crates/intel/driver` | `detour-intel` | 运行时 (`runtime.rs`)、条带合成器 (`raster.rs`)、模糊效果 (`frost.rs`: 毛玻璃、filter、drop-shadow, compute, 生成 MLIR)、一次性的 `Device` (`lib.rs`); 示例 |
| `crates/vello_cpu` | (模块 `vello_cpu`, xe3) | crates.io vello_cpu 0.3.0 原样拷进来, 导出粗光栅 (`RenderContext::coarse`); 改动处标 `detour:`, 说明在 `DETOUR.md` |
| `scripts/intel-cmdgen.py` | — | `crates/intel/dialect/genxml/xe3.xml` ⇒ `cmd.gen.rs` |

## 一个三角形的路

```
examples/triangle/func.mlir      detour.* + func.func {detour.stage}
  Compiler::parse                 (mlirformat, 加载 detour + xe3 方言)
  FRONTEND                        spirv / gpu ⇒ func; clear ⇒ 清屏 draw; canonicalize, cse
  convert-detour-to-xe3            shader.rs: func ⇒ image.kernel (虚拟 xe3.eu, !xe3.grf<f32>)
                                  render.rs: Render + Cmds ⇒ image.memory / image.stream / xe3.state.*
  xe3-regalloc, xe3-swsb          物理寄存器; regdist
  xe3-pad, xe3-end                kernel 的 pad; batch 加 MI_BATCH_BUFFER_END (和 MI_NOOP 凑 qword)
  xe3-layout                      地址, 符号 ⇒ 数
  XE3_IMAGE (ImageEmit<Xe3>)      每个 memory 一段字节 (XE3.pack, Inst::encode)
  Device::run / Context           kmd trait ⇒ xe ⇒ GPU
```

运行时 (`frames` 示例) 只在 `Context::pipeline` 里跑到 `xe3-swsb`, 之后每帧用模板.

## 数据和对照

| 文件 | 是什么 |
| --- | --- |
| `crates/intel/dialect/genxml/xe3.xml` | Mesa genxml (展平), cmdgen 的输入 |
| `crates/intel/dialect/tests/oracle.{mlir,expected}` | 290 个条目, Mesa C 打包的结果 |
| `crates/intel/dialect/tests/eu.{mlir,expected}` | 67 条 EU 指令, gentool 汇编的结果 |
| `crates/intel/driver/examples/triangle/*.mlir` | 三角形的四种着色器写法 |
| `crates/intel/driver/examples/frames/triangle.mlir` | 只有 pipeline 的模块 (运行时用) |
