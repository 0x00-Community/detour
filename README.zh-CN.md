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
- 🔵 **Intel Xe3** (`src/passes/xe3`): xe3 方言 (EU 指令、genxml 命令和状态)、SPIR-V 和 GPU kernel 下降到 Xe3 机器码、
  寄存器分配、记分板、一次绘制的镜像.
- 🐧 **内核接口** (`src/kmd`、`src/xe.rs`): DRM render node、BO、VM 绑定、执行队列、syncobj, 走 Linux 的 `xe` 驱动.

## 🎯 对照 Mesa 核对

部署了不算成功: 每个结果都和本机 Mesa (iris, 25.2.8) 逐字节对照.

| ✔️ | 项目 | 结果 |
| --- | --- | --- |
| 🎨 | 绘制 (`examples/compare`): 同一份 GLSL 和顶点分别经 detour 在 Xe3 上画、经 Mesa 画, 64×64 RGBA32F | 10 个用例里 6 个**逐字节一致** (三角形、两个阶段的 uniform、向量 `?:`、`\|\|`、写分量); 其余的等 Mesa 的 NIR 数学 (`mod`、整数除法、GLSL 内建函数、Mesa 改值的 NIR 优化) |
| 📞 | GL 调用 (`examples/glcalls`): 纹理、renderbuffer、framebuffer 对象的调用脚本分别在 detour 和 Mesa 上跑 | **203 / 203** 次调用: 每次 `glGetError` 和每个查询值都相同 |
| 🔣 | Xe3 编码和下降 (`tests/xe3.rs`) | EU 编码对 gentool、genxml 打包对 Mesa、kernel、GLSL 到 Xe3: **4 / 4** |

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
| `src/passes/xe3` | Xe3: 方言、`convert-gpu-to-xe3`、`xe3-*` pass、`Xe3Draw` |
| `src/shader.rs`、`src/render.rs` | 工具链 (mlirpass + Xe3 的 pass) 和执行绘制的设备 |
| `src/kmd`、`src/xe.rs` | 内核接口和 `xe` |
| `examples/` | `compare` (绘制对 Mesa)、`glcalls` (GL 调用对 Mesa)、`gl_link`、`xe_info` |
| `tests/xe3.rs` | Xe3 编码和下降 |
| `tools/xe3gen`、`data/intel` | genxml 生成器和 Mesa 的 Xe3 genxml |
| `scripts/` | `gl_enums.py`、`xe3_eu_oracle.sh` |

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

Apache-2.0. `data/intel/xe3.xml` 来自 Mesa 的 genxml (MIT, Intel).

<div align="center">

**🛣️ 走 detour. 绕路才是捷径.**

[⬆️ 回到顶部](#top) · [🇺🇸 English](README.md)

</div>
