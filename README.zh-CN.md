<a id="top"></a>
<div align="center">

# detour

### 绕开 Mesa. 绕开 libdrm.
**OpenGL、Vulkan、Direct3D 11 和 Direct3D 12 跑在我们自己的 GPU 后端上: Rust 写的 GPU 用户态驱动, 经 MLIR 编译.**

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

detour 是**我们自己的 GPU 驱动用户态**. GL 状态、着色器到机器码的编译、命令流都由 detour 自己负责, **经内核 GPU
驱动的 ioctl 直接提交**. 下面没有 Mesa, 没有 libdrm.

- 🧩 **GL ES 3.0** (`src/gl`): 对象、绑定、状态; GLSL 先由 [mlirpass](https://github.com/Jabberwocky238/mlirpass) 编成 SPIR-V,
  再由 detour 编成 Xe3 机器码.
- 🌋 **Vulkan** (`src/vk`): 下一步.
- 🪟 **Direct3D 11 和 12**: 待做, 用同一个后端 (着色器和 GL、Vulkan 一样经 SPIR-V).
- 🔵 **Intel Xe3**: xe3 方言 (EU 指令、genxml 命令和状态)、SPIR-V 和 GPU kernel 下降到 Xe3 机器码、寄存器分配、
  记分板、一次绘制的镜像.
- 🧱 **更多 GPU 后端**: AMD RDNA2、Apple AGX、Apple9 (M3 / M4) 的方言、下降和编码器, 分别对照 llvm-mc、Mesa 和 M4.
- 🔌 **以插件方式接 mlirformat 和 mlirpass**: `src/dialect` 给 mlirformat 加 GPU 方言 (`MlirOpt::load`), `src/pass`
  给 mlirpass 加下降 (`Passes::with(DetourPasses::add)`), `src/binary` 写出机器码和镜像 (`Binaries`、`Read` / `Write`).
- 🐧 **内核接口** (`src/kmd`, `xe` 驱动在 `src/kmd/xe`): DRM render node、BO、VM 绑定、执行队列、syncobj, 走 Linux 的 `xe` 驱动.

## 🎯 对照 Mesa 核对

部署了不算成功: 每个结果都和本机 Mesa (iris, 25.2.8) 逐字节对照.

| ✔️ | 项目 | 结果 |
| --- | --- | --- |
| 🎨 | 绘制 (`examples/compare`): 同一份 GLSL 和顶点分别经 detour 在 Xe3 上画、经 Mesa 画, 64×64 RGBA32F | 10 个用例里 6 个**逐字节一致** (三角形、两个阶段的 uniform、向量 `?:`、`\|\|`、写分量); 其余的等 Mesa 的 NIR 数学 (`mod`、整数除法、GLSL 内建函数、Mesa 改值的 NIR 优化) |
| 📞 | GL 调用 (`examples/glcalls`): 纹理、renderbuffer、framebuffer 对象的调用脚本分别在 detour 和 Mesa 上跑 | **203 / 203** 次调用: 每次 `glGetError` 和每个查询值都相同 |
| 🔣 | 编码 (`tests/binary`) | Xe3 EU 对 gentool、genxml 对 Mesa, RDNA2 对 llvm-mc 和 Mesa 寄存器宏, AGX 对 Mesa AGX2 用例和 `agx_pack`, Apple9 对 M4 跑过的编码: **7 / 7** |
| 🪜 | 下降 (`tests/pass`) | kernel 对 acgpu (Xe3、RDNA2、AGX)、GLSL 到 Xe3、RDNA2 IB 和 AGX CDM 流对 Mesa、Apple9 kernel 对 M4: **7 / 7** |

## 🎮 支持的 GPU

| GPU | 内核驱动 | 状态 |
| --- | --- | --- |
| 🔵 **Intel Xe3** (Panther Lake, PCI 8086:b0a0) | `xe` | GL ES 3.0 绘制进行中 (见上) |

## 🗂️ 目录

| 路径 | 内容 |
| --- | --- |
| `src/gl` | GL ES 3.0: 上下文、buffer、VAO、shader 和 program (默认 uniform 块)、纹理、renderbuffer、framebuffer、GLenum (`enums.gen.rs` 由 `GLES3/gl3.h` 生成) |
| `src/vk` | Vulkan (待做) |
| — | Direct3D 11 和 12 (待做) |
| `src/dialect` | mlirformat 插件: `xe3`、`rdna2`、`agx`、`apple9`、`image` 方言 |
| `src/pass` | mlirpass 插件: `convert-gpu-to-*`、寄存器分配、等待、命令流、镜像布局、`Xe3Draw`、支持这些 GPU 的 `gpu-module-to-binary` |
| `src/binary` | 机器码和镜像: 各 GPU 的编码器、按布局表打包记录、写出镜像 |
| `src/shader.rs`、`src/render.rs` | 工具链 (mlirpass + detour 的 pass) 和执行绘制的设备 |
| `src/kmd` | 内核接口: DRM、`Gpu` trait、`xe` 驱动 (`src/kmd/xe`) |
| `examples/` | `compare` (绘制对 Mesa)、`glcalls` (GL 调用对 Mesa)、`gl_link`、`xe_info` |
| `tests/binary`、`tests/pass` | 编码、下降 |
| `scripts/` | 方言生成器 (`xe3gen`、`rdna2gen`、`agxgen`) 和它们读的 Mesa 数据 (`data/{intel,amd,apple}`)、`gl_enums.py`、opcode 表, 以及各测试期望值的来源 (`*_oracle.*`、`apple9_check.py`) |

## 🚀 快速开始

```sh
# 🔍 本机的 Xe3
cargo run --example xe_info

# 🧪 Xe3 编码和下降
cargo test --test xe3

# 🎨 绘制对 Mesa (glrun: mlirpass 的 tests/support/glrun.c)
cc -O1 -o glrun path/to/mlirpass/tests/support/glrun.c -lEGL -lGL
cargo run --example compare -- ./glrun

# 📞 GL 调用对 Mesa
cc -O1 -o glcalls examples/support/glcalls.c -lEGL -lGLESv2
cargo run --example glcalls -- ./glcalls
```

## 📜 许可证

Apache-2.0. `scripts/data/intel/xe3.xml` 来自 Mesa 的 genxml (MIT, Intel).

<div align="center">

**🛣️ 走 detour. 绕路才是捷径.**

[⬆️ 回到顶部](#top) · [🇺🇸 English](README.md)

</div>
