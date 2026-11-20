<a id="top"></a>
<div align="center">

# detour

### 绕开 Mesa. 绕开 libdrm.
**Rust 写的 GPU 用户态驱动, 经 MLIR 编译, 只针对少数几种 GPU 做到极致 (目前是 Xe3).**

[![Rust](https://img.shields.io/badge/Rust-100%25-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
[![MLIR](https://img.shields.io/badge/IR-MLIR-262D3A?style=for-the-badge&logo=llvm)](https://mlir.llvm.org)
[![Mesa](https://img.shields.io/badge/Mesa-%E5%AF%B9%E7%85%A7%E5%9F%BA%E5%87%86-red?style=for-the-badge)](#-对照-mesa-核对)
[![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen?style=for-the-badge)](#-许可证)

[![Intel Xe3](https://img.shields.io/badge/Intel-Xe3-0071C5?style=flat-square&logo=intel)](#-支持的-gpu)
[![Linux](https://img.shields.io/badge/Linux-xe-FCC624?style=flat-square&logo=linux&logoColor=black)](#-支持的-gpu)

[🇺🇸 English](README.md) · **🇨🇳 简体中文**

</div>

---

## ⚡ detour 是什么

detour 是**我们自己的 GPU 驱动用户态**, 只为少数几种 GPU 而做: 不模拟任何图形 API, 只提供绘制 (`Device::draw`).
着色器到机器码的编译、命令流都由 detour 自己负责, **经内核 GPU 驱动的 ioctl 直接提交**. 下面没有 Mesa, 没有 libdrm.
图形 API (Vulkan、WebGPU) 是应用那一层的事, 不归 detour.

- 🧩 **绘制** (`src/render.rs`、`src/shader.rs`): SPIR-V 着色器 (GLSL 经 [mlirpass](https://github.com/Jabberwocky238/mlirpass))
  编成 Xe3 或 RDNA2 机器码, 可预编译也可现编; 目标、顶点缓冲、uniform、采样纹理 (任何 surface, 所以也能渲染到纹理)、
  混合、scissor、dma-buf.
- 🔵 **Intel Xe3**: xe3 方言 (EU 指令、genxml 命令和状态)、SPIR-V 着色器下降到 Xe3 机器码、寄存器分配、
  记分板、一次绘制的镜像.
- 🔴 **AMD RDNA2** (gfx1036): SPIR-V 着色器直接编成 GFX10 机器码 (wave32、NGG passthrough; 自己的指令选择、寄存器分配
  和等待)、一次绘制的 PM4 状态, 以及 `amdgpu` 内核接口 (buffer object、地址空间、带 syncobj 的命令提交、dma-buf).
- 🔌 **以插件方式接 mlirformat 和 mlirpass**: `src/dialect` 给 mlirformat 加 xe3 方言 (`MlirOpt::load`), `src/pass`
  给 mlirpass 加下降 (`Passes::with(DetourPasses::add)`), `src/binary` 编码 Xe3 机器码、打包 genxml 记录.
- 🐧 **内核接口** (`src/kmd`, `xe` 驱动在 `src/kmd/xe`): DRM render node、BO、VM 绑定、执行队列、syncobj, 走 Linux 的 `xe` 驱动.

## 🎯 对照 Mesa 核对

部署了不算成功: 每个结果都和本机 Mesa (iris, 25.2.8) 逐字节对照.

| ✔️ | 项目 | 结果 |
| --- | --- | --- |
| ✅ | 正确性 (`examples/correct`): 经 `Device::draw` 绘制, 对照 CPU 参考 (像素中心、8 位子像素对齐、左上规则、GL ES 混合、着色器数学按 Vulkan 的精度) 和同一 GPU 上的 Mesa: 覆盖、共享边、细的 / 退化的 / 极大的三角形、strip、first / count、剔除、裁剪、视口、实例化、所有混合因子和方程、颜色掩码、顶点格式和步长、插值、std140 块、内建变量、15 个 varying、局部数组、20 个数学函数 (含 NaN、无穷、非规格化数) | **752 / 752** 例: 固定功能**和 Mesa 逐字节相同**; 只在 Xe3 混合或转 unorm8 处与 f32 参考差 ±1 (Mesa 相同) |
| 📤 | 导出和扫描输出 (`examples/export`): 可扫描输出的 surface (Tile4 和 Linear, 1920×1080) 画好后导出成 dma-buf (`Device::export`: fd、pitch、modifier、fourcc), 由另一个设备导入读回, 再交给 KMS 建 framebuffer (带 modifier 的 `ADDFB2`) | **2 / 2** |
| 🔢 | 索引绘制和字节颜色 (`examples/indexed`): 3000 个重叠的四边形, 每个 4 个顶点加 6 个 u16 / u32 索引, 颜色用 `Unorm8x4`, 从中间的索引开始, 索引和非索引绘制交替 | **5 / 5** 例和非索引绘制**逐字节一致** |
| 🖼️ | 纹理 (`examples/texture`): 四边形采样一个 surface, 对照 CPU 参考: nearest 和 linear、放大和缩小、clamp / repeat / mirrored repeat、RGBA8 和 R8、linear 和 Tile4; 同一批里先画进再采样、先采样再覆盖; 局部写入 (`Device::write`) | **15 / 15** 例, nearest **逐字节一致**, linear 与 f32 参考差 2 以内 |
| 🔣 | 编码 (`tests/binary`、`src/pass/rdna2`) | Xe3 EU 对 gentool、genxml 对 Mesa, RDNA2 对 llvm-mc |
| 🪜 | 下降 (`tests/pass`) | 手写 SPIR-V、GLSL 和 acdes 页面程序的 Xe3、RDNA2 机器码, 对照读过的基准 |

## 🎮 支持的 GPU

| GPU | 内核驱动 | 状态 |
| --- | --- | --- |
| 🔵 **Intel Xe3** (Panther Lake, PCI 8086:b0a0) | `xe` | 绘制 (见上) |
| 🔴 **AMD RDNA2** (gfx1036, Granite Ridge Radeon 610M, PCI 1002:13c0) | `amdgpu` | 绘制 (acdes 的页面; `examples/programs`、`examples/amdgpu_draw`) |

detour 只驱动 x86_64 Linux 上的这两种核显. 没有任何回退: 没有 CPU 光栅, 也不转给系统的驱动.

## 🗂️ 目录

| 路径 | 内容 |
| --- | --- |
| `src/dialect` | `xe3` 方言 (mlirformat 插件)、RDNA2 指令表、打包记录用的布局表 |
| `src/pass` | mlirpass 插件: `convert-gpu-to-xe3`、寄存器分配、控制流、SWSB、`Xe3Draw`; RDNA2 着色器编译器 (`Rdna2Graphics`) 和绘制状态 |
| `src/binary` | Xe3 机器码、按布局表打包记录 |
| `src/shader.rs`、`src/render.rs` | 工具链 (mlirpass + detour 的 pass) 和执行绘制的设备 |
| `src/kmd` | 内核接口: DRM、`Gpu` trait、`xe` 驱动 (`src/kmd/xe`)、`amdgpu` 驱动 (`src/kmd/amdgpu`) |
| `examples/` | `correct` (正确性对 CPU 参考和 Mesa; `support/glref.c`)、`texture` (采样纹理对 CPU 参考)、`indexed` (索引绘制)、`export` (dma-buf 和扫描输出)、`program` (预编译的程序)、`shapes` 和 `frame` (一帧里的绘制, 计时)、`programs` (画 acdes 的程序)、`xe_info`、`amdgpu_info`、`amdgpu_draw`、`amdgpu_device` |
| `tests/binary`、`tests/pass` | 编码、下降 |
| `scripts/` | xe3 方言生成器 (`xe3gen`) 和它读的 Mesa 数据 (`data/intel`)、RDNA2 opcode 表 (`rdna2_opcodes.py`、`data/amd`), 以及各测试期望值的来源 (`*_oracle.*`) |

## 🚀 快速开始

```sh
# 🔍 本机的 Xe3
cargo run --example xe_info

# 🧪 Xe3 编码和下降
cargo test --test binary --test pass

# ✅ 正确性对 CPU 参考和 Mesa
cc -O1 -o glref examples/support/glref.c -lEGL -lGLESv2
GLREF=./glref cargo run --release --example correct

# 🏁 一帧里的绘制, 计时
cargo run --release --example frame
```

## 📜 许可证

Apache-2.0. `scripts/data/intel/xe3.xml` 来自 Mesa 的 genxml (MIT, Intel).

<div align="center">

**🛣️ 走 detour. 绕路才是捷径.**

[⬆️ 回到顶部](#top) · [🇺🇸 English](README.md)

</div>
