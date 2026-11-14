<a id="top"></a>
<div align="center">

# 🛣️ detour

### 绕开 Mesa. 绕开 Vulkan. 绕开 OpenGL. 绕开 libdrm.
**从 MLIR 直达硅片: 100% Rust 写的 GPU 用户态驱动.**

[![Rust](https://img.shields.io/badge/Rust-100%25-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
[![MLIR](https://img.shields.io/badge/IR-MLIR-262D3A?style=for-the-badge&logo=llvm)](https://mlir.llvm.org)
[![Mesa](https://img.shields.io/badge/Mesa-%E4%B8%8D%E9%9C%80%E8%A6%81-red?style=for-the-badge)](#-detour-是什么)
[![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen?style=for-the-badge)](#-许可证)

[![Intel Xe3](https://img.shields.io/badge/Intel-Xe3-0071C5?style=flat-square&logo=intel)](#-支持的-gpu)
[![Apple](https://img.shields.io/badge/Apple-M1%20%7C%20M4-000000?style=flat-square&logo=apple)](#-支持的-gpu)
[![AMD](https://img.shields.io/badge/AMD-RDNA2-ED1C24?style=flat-square&logo=amd)](#-支持的-gpu)
[![Linux](https://img.shields.io/badge/Linux-xe%20%7C%20asahi%20%7C%20amdgpu-FCC624?style=flat-square&logo=linux&logoColor=black)](#-支持的-gpu)

[🇺🇸 English](README.md) · **🇨🇳 简体中文**

</div>

---

## ⚡ detour 是什么

detour 是**我们自己的 GPU 驱动用户态**. 命令缓冲、状态、着色器机器码全部由 detour 自己生成, **经内核 GPU 驱动的 ioctl
直接提交**. 中间没有 Mesa, 没有 Vulkan, 没有 OpenGL, 没有 libdrm.

🎯 **目标:** 比 Mesa 路径更短. 没有 GL / Vulkan 的状态机, 没有驱动层的猜测: 每个命令、每个状态位都是我们定的.

🪟 **为 acde 而生:** acde 每个窗口里网页图层的光栅化、WebGPU (`navigator.gpu`) 和 `<canvas>` 的
`webgl` / `webgl2` / `webgpu` 上下文都走 detour.

## 🔥 凭什么不一样

| | |
| --- | --- |
| 🦀 **全 Rust** | 从 IR 到 ioctl, 下面没有 C 写的驱动栈. |
| 🧬 **所有 GPU 一套 MLIR** | 每一家的命令流和着色器二进制都出自同一条 [mlirformat](https://github.com/Jabberwocky238/mlirformat) / [mlirpass](https://github.com/Jabberwocky238/mlirpass) 流水线. 不是兼容层. |
| 🎨 **Vulkan 能做的都能做** | 完整的 3D 图形管线和 compute, 以 Vulkan 的能力为准. 接口是自己的 (`detour` 方言 + 运行时), 不是 Vulkan API. |
| ⏱️ **每帧不跑 MLIR** | pipeline 只编一次, 每帧只拷模板、填地址: 录一帧约 **17 µs** (debug 构建). |
| 🎯 **逐位对上才算数** | 部署了不算成功. 像素和读回的数逐个核对, 对照 Mesa、gentool、LLVM 或 Rust 模型. |
| 🛟 **总能画出来** | GPU 不支持? 回退到 vello on wgpu (Vulkan). 根本没有 GPU? CPU 直接画页面. |

## 🧬 流水线

```text
detour.* (资源、pipeline、render pass) + 着色器 (func / spirv / gpu)
  ⇒ 🧠 前端    着色器统一、清屏展开、canonicalize / cse
  ⇒ ⚙️ 后端    image.kernel / xe3.cmd / xe3.state, 寄存器分配, 软件记分板, 定址
  ⇒ 📦 镜像    ⇒ kmd trait (xe ioctl)
  ⇒ 🖥️ GPU
```

## 🎮 支持的 GPU

| GPU | 内核驱动 | 状态 |
| --- | --- | --- |
| 🔵 **Intel Xe3** (Panther Lake, PCI 8086:b0a0) | `xe` | ✅ **主力.** 渲染、compute、运行时、2D 合成、模糊效果都在真机上跑通 |
| 🍎 **Apple M1** (AGX G13) | `asahi` (Asahi Linux) | 🚧 内核接口、`agx` 方言、USC 编码和 compute 已写完, 对 Mesa 逐位核对; 还没上真机 |
| 🍏 **Apple M3 / M4** (Apple9) | 无 (Metal binary archive) | 🧪 compute 编码在 M4 上验过 |
| 🔴 **AMD RDNA2** (gfx1036) | `amdgpu` | 🚧 PM4、指令编码、compute 派发已完成, 对 Mesa / `llvm-mc` 核对; 还没上真机 |
| ⚪ 其他 | | 🛟 `fallback-vulkan`: vello on wgpu |

## 🏆 逐位验证

| ✔️ 项目 | 结果 |
| --- | --- |
| 📦 genxml 打包 vs Mesa 的 C 打包 | **290** 个条目, 逐位一致 |
| 🔣 EU 编码 vs gentool (`asm -p ptl`) | **67** 条 `xe3.eu` op 逐位一致, 含 DPAS 和 LSC 2D block |
| 🧮 着色器数值 (`math`) | **65,536** 组操作数 (非规格化数、±∞、NaN …), 除法、平方根、fma、floor、min / max、比较选择: **0 错**, 正确舍入 |
| 🔺 三角形 (`triangle`) | 四种着色器写法 (func / gpu / spirv / arith): **0** 个错像素 |
| 🌫️ Dual Kawase 模糊 (`frost`) | 13 个叠加效果对 Rust 模型逐字节比: **0 错**; 1200×700 面板 GPU **1.02 ms** |
| 🔢 compute 和整数 (`compute`) | 65,536 次调用, 20 种结果对 Rust: **0 错** |
| 🎞️ 运行时 (`frames`) | 背靠背 1000 帧 1920×1080, 录制 p50 **17 µs**, 最后一帧 **207 万像素 0 错** |

完整表格和测量条件: [docs/verification.md](docs/verification.md) (英文).

## 🗺️ 路线

| | 阶段 | 状态 |
| --- | --- | --- |
| 🏁 | M1 运行时: 常驻 Context、pipeline 一次编译、模板录帧、dma-buf / sync_file 导出 | ✅ |
| 🎨 | M2 着色器能力: 插值、实例化、采样、混合、精确 f32 数学 | ✅ |
| 🔢 | R1 compute 和整数 · R2 控制流 · R3 vello_cpu 拆开 | ✅ |
| 🖌️ | R4 GPU 细光栅 · R5 滤镜层 · R6 接 acde · R7 零拷贝 | 🚧 |
| 🚄 | R8 粗光栅上 GPU (Xe3 的 f64 / i64) | 🚧 |
| 🌐 | M6 WebGPU: WGSL 前端、共享内存、barrier、原子 | 🔜 |

详见 [docs/roadmap.md](docs/roadmap.md) (英文).

## 🚀 快速开始

```sh
# 🔍 本机有哪些 GPU、detour 支持到哪一步 (-v 打印驱动报的参数)
cargo run --features cli,xe3,agx-g13,rdna2 -- devices

# 🧪 不用 GPU: 对照 Mesa 的打包、对照 gentool 的编码、前端
cargo test --workspace

# 🔺 要 Xe3: 四种着色器写法画三角形, 逐像素核对
cargo run --features xe3 --example triangle

# 🎞️ 运行时: 1000 帧, 录制 / 吞吐 / 延迟, 逐像素核对
cargo run --features xe3 --example frames

# 🔬 打印下降后的 MLIR
cargo run --features xe3 --example lower -- crates/intel/driver/examples/triangle/func.mlir

# 🧱 Xe3 上的更多示例, 都逐像素核对
cargo run --features xe3 --example indexed   # 索引绘制 (u16 / u32、first index、base vertex)
cargo run --features xe3 --example depth     # 深度测试 (比较、写开关、清深度)
cargo run --features xe3 --example uniform   # 渲染着色器读 uniform 缓冲
cargo run --features xe3 --example mipmap    # mipmap (由导数定级)

# 🥊 同样的帧 / dispatch 在 Mesa anv 和 iris 上跑, 两边逐像素核对
cargo bench --features xe3 --bench against_vulkan
cargo bench --features xe3 --bench against_opengl

# 🪟 窗口里
cargo run -p triangle      # 三角形 (读回经 softbuffer 上屏)
cargo run -p triangle3d    # 转动的 3D 三角形
```

## 📚 文档 (英文)

| | 文件 | 内容 |
| --- | --- | --- |
| 🤝 | [AGENTS.md](AGENTS.md) | 协作约定 (人和 agent 都先读) |
| ✍️ | [CONTRIBUTING.md](CONTRIBUTING.md) | 代码风格、验收、提交和推送 |
| 🏗️ | [docs/architecture.md](docs/architecture.md) | 分层、crate 分工、和内核的分工 |
| 🔵 | [docs/xe3.md](docs/xe3.md) | xe3 方言、Xe3 硬件上的坑、从 Mesa 读出来的 Xe2+ 细节 |
| ⚙️ | [docs/runtime.md](docs/runtime.md) | Context / Pipeline / Target / Recorder, 模板和洞 |
| ✅ | [docs/verification.md](docs/verification.md) | 怎么验证, 已验证的结果 |
| 🚀 | [docs/optimizations.md](docs/optimizations.md) | 精确着色器数值、SWSB 和消息寄存器、URB、2D 合成、渐变、模糊 |
| 🗺️ | [docs/roadmap.md](docs/roadmap.md) | 里程碑, 接入 acde |
| 🧰 | [.skills/](.skills/README.md) | 按任务的操作手册 |

## 📜 许可证

Apache-2.0. `crates/intel/dialect/genxml/xe3.xml` 来自 Mesa (MIT, Intel).

<div align="center">

**🛣️ 走 detour. 绕路才是捷径.**

[⬆️ 回到顶部](#top) · [🇺🇸 English](README.md)

</div>
