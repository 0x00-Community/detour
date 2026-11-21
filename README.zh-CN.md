<a id="top"></a>
<div align="center">

# detour

### 绕开 Mesa. 绕开 libdrm.
**Rust 写的 GPU 用户态驱动, 经 MLIR 编译, 只为两块核显做到极致.**

[![Rust](https://img.shields.io/badge/Rust-100%25-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
[![MLIR](https://img.shields.io/badge/IR-MLIR-262D3A?style=for-the-badge&logo=llvm)](https://mlir.llvm.org)
[![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen?style=for-the-badge)](#-许可证)

[![Intel Xe3](https://img.shields.io/badge/Intel-Xe3-0071C5?style=flat-square&logo=intel)](#-支持的-gpu)
[![AMD RDNA2](https://img.shields.io/badge/AMD-RDNA2-ED1C24?style=flat-square&logo=amd)](#-支持的-gpu)
[![Linux](https://img.shields.io/badge/Linux-xe%20%C2%B7%20amdgpu-FCC624?style=flat-square&logo=linux&logoColor=black)](#-支持的-gpu)

[🇺🇸 English](README.md) · **🇨🇳 简体中文**

</div>

---

## ⚡ detour 是什么

detour 只在两块 GPU 上绘制. 它接收 SPIR-V, 编成 GPU 的机器码, 组好命令流, **直接经内核驱动的 ioctl** 提交: 没有
Mesa, 没有 libdrm, 不提供任何图形 API (那是应用那一层的事). 评判它的只有一条: 画得多快.

- 🧩 **绘制** (`src/render.rs`、`src/shader.rs`): 程序 (一个 SPIR-V 顶点着色器加一个片元着色器) 可预编译也可现编;
  目标、顶点和索引缓冲、uniform 块、采样纹理 (任何 surface, 所以也能渲染到纹理)、混合、scissor、dma-buf 的导入导出.
  绘制给不了的东西, 编译时就拒绝.
- 🔵 **Intel Xe3**: xe3 方言 (EU 指令、genxml 命令和状态)、SPIR-V 下降到 EU 代码、寄存器分配、记分板, 一次绘制的状态
  作为常驻的二级批次.
- 🔴 **AMD RDNA2** (gfx1036): SPIR-V 直接编成 GFX10 代码 (wave32、NGG passthrough; 自己的指令选择、寄存器分配和等待),
  一次绘制的 PM4 状态只设变了的.
- 🔌 **以插件方式接 mlirformat 和 mlirpass**: `src/dialect` 加 xe3 方言, `src/pass` 加下降
  (`Passes::with(DetourPasses::add)`), `src/binary` 编码机器码.
- 🐧 **内核接口** (`src/kmd`): DRM render node、buffer object、地址空间、带 syncobj 的提交, 走 Linux 的 `xe` 和
  `amdgpu` 驱动.

## 🎯 核对

两块 GPU 都对照 CPU 参考; Xe3 的固定功能另与同一 GPU 上的 Mesa (iris 25.2.8) 逐字节对照.

| ✔️ | 内容 | Xe3 | RDNA2 |
| --- | --- | --- | --- |
| ✅ | `examples/correct`: 覆盖、填充规则、剔除、裁剪、视口、实例化、所有混合因子和方程、顶点格式、插值、std140 块、内建变量、varying、局部数组、20 个数学函数 (含 NaN、无穷、非规格化数) | **752 / 752**, 固定功能与 Mesa 相同 | **711 / 752**; 40 例拒绝 (尚不支持), 1 例小数视口差一列 |
| 🖼️ | `examples/texture`: nearest 和 linear、放大和缩小、三种 wrap、RGBA8 和 R8、渲染到纹理、局部写入 | **15 / 15** | **15 / 15** |
| 🔢 | `examples/indexed`: u16 / u32 索引、`Unorm8x4` 颜色、索引与非索引绘制交替 | **5 / 5**, 与非索引绘制相同 | **5 / 5** |
| 📤 | `examples/export`: scanout surface 导出为 dma-buf, 经另一个设备读回, 交给 KMS | **2 / 2** | – |
| 🔣 | `tests/binary`: 编码 | EU 对 gentool、genxml 对 Mesa | 对 llvm-mc |
| 🪜 | `tests/pass`: 每个 SPIR-V 用例的机器码或拒绝原因, 对照仓库里的基准 | ✓ | ✓ |

## 🎮 支持的 GPU

| GPU | 内核驱动 |
| --- | --- |
| 🔵 **Intel Xe3** (Panther Lake, PCI 8086:b0a0) | `xe` |
| 🔴 **AMD RDNA2** (gfx1036, Radeon 610M, PCI 1002:13c0) | `amdgpu` |

只支持 Linux. 绘制跑在按 x86-64-v3 (AVX2, BMI2, FMA; `.cargo/config.toml`) 编译的 x86_64 主机上: 更老的 CPU 会遇到非法
指令而崩溃. 工具链 (编译, 编码, `tests/`) 在 aarch64 上也能编过并通过. 没有任何回退: 没有 CPU 光栅, 也不转给系统的驱动.

## 🗂️ 目录

| 路径 | 内容 |
| --- | --- |
| `src/dialect` | `xe3` 方言、RDNA2 指令表、记录打包用的布局 |
| `src/pass` | 下降: SPIR-V 到 Xe3 (`convert-gpu-to-xe3`、寄存器、控制流、SWSB、`Xe3Draw`) 和到 RDNA2 (`Rdna2Graphics`、`Rdna2Draw`) |
| `src/binary` | 机器码和记录的编码 |
| `src/shader.rs`、`src/render.rs` | 工具链, 以及绘制所在的设备 (`src/render`: Xe、amdgpu、两者共用的环形缓冲) |
| `src/kmd` | 内核接口: DRM、`xe`、`amdgpu` |
| `examples/` | `correct`、`texture`、`indexed`、`program`、`shapes`、`frame` (任一 GPU); `export`、`xe_info` (Xe3); `amdgpu_info`、`amdgpu_draw`、`amdgpu_device` (RDNA2); `programs` (acdes 的程序) |
| `tests/` | `binary` (编码)、`pass` (每个用例在两块 GPU 上的机器码) |
| `scripts/` | xe3 方言生成器 (`xe3gen`)、RDNA2 操作码表、预期字节所依据的 oracle |

## 🚀 快速上手

```sh
cargo test --release --lib --tests                       # 编码和机器码
cargo run --release --example correct                    # 在本机 GPU 上核对正确性
cargo run --release --example shapes                     # 一帧绘制, 计时
```

Xe3 与 Mesa 的对照需要 `examples/support/glref.c` (`cc -O1 -o glref examples/support/glref.c -lEGL -lGLESv2`, 再设
`GLREF=./glref`).

## 📜 许可证

Apache-2.0. `scripts/data/intel/xe3.xml` 是 Mesa 的 genxml (MIT, Intel).

<div align="center">

**🛣️ 走 detour. 绕路才是捷径.**

[⬆️ 回到顶部](#top) · [🇺🇸 English](README.md)

</div>
