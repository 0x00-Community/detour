<a id="top"></a>
<div align="center">

# detour

### Skip Mesa. Skip libdrm.
**OpenGL, Vulkan, Direct3D 11 and Direct3D 12 on our own GPU backend: a GPU userspace driver in Rust, compiled through MLIR.**

[![Rust](https://img.shields.io/badge/Rust-100%25-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
[![MLIR](https://img.shields.io/badge/IR-MLIR-262D3A?style=for-the-badge&logo=llvm)](https://mlir.llvm.org)
[![Mesa](https://img.shields.io/badge/Mesa-the%20reference-red?style=for-the-badge)](#-checked-against-mesa)
[![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen?style=for-the-badge)](#-license)

[![Intel Xe3](https://img.shields.io/badge/Intel-Xe3-0071C5?style=flat-square&logo=intel)](#-gpus)
[![Linux](https://img.shields.io/badge/Linux-xe-FCC624?style=flat-square&logo=linux&logoColor=black)](#-gpus)

**🇺🇸 English** · [🇨🇳 简体中文](README.zh-CN.md)

</div>

---

## ⚡ What is detour

detour is **our own GPU userspace driver**. It keeps the GL state, compiles the shaders to the GPU's machine code,
builds the command streams and submits them **through the kernel driver's ioctls**. No Mesa, no libdrm underneath.

- 🧩 **GL ES 3.0** (`src/gl`): objects, bindings and state; GLSL compiled by [mlirpass](https://github.com/Jabberwocky238/mlirpass)
  to SPIR-V, then by detour to Xe3.
- 🌋 **Vulkan** (`src/vk`): next.
- 🪟 **Direct3D 11 and 12**: to come, on the same backend (shaders through SPIR-V like GL's and Vulkan's).
- 🔵 **Intel Xe3**: the xe3 dialect (EU instructions, genxml commands and states), SPIR-V and GPU kernels lowered to
  Xe3 machine code, register allocation, scoreboard, a draw's image.
- 🧱 **More GPUs' backends**: AMD RDNA2, Apple AGX and Apple9 (M3 / M4) dialects, lowerings and encoders, checked
  against llvm-mc, Mesa and an M4.
- 🔌 **On mlirformat and mlirpass as plugins**: `src/dialect` adds the GPU dialects to mlirformat (`MlirOpt::load`),
  `src/pass` adds the lowerings to mlirpass (`Passes::with(DetourPasses::add)`), `src/binary` writes machine code and
  images (`Binaries`, `Read` / `Write`).
- 🐧 **Kernel interface** (`src/kmd`, `src/xe.rs`): DRM render nodes, buffer objects, VM binds, exec queues, syncobjs,
  on Linux's `xe` driver.

## 🎯 Checked against Mesa

Deploying is not success: every result is compared with the local Mesa (iris, 25.2.8), byte for byte.

| ✔️ | What | Result |
| --- | --- | --- |
| 🎨 | Draws (`examples/compare`): the same GLSL and vertices through detour on Xe3 and through Mesa, a 64×64 RGBA32F target | 6 of 10 cases **byte for byte** (triangle, uniforms in both stages, vector `?:`, `\|\|`, component stores); the rest wait for Mesa's NIR math (`mod`, integer division, GLSL built-ins, Mesa's value-changing NIR rewrites) |
| 📞 | GL calls (`examples/glcalls`): scripts of textures, renderbuffers and framebuffer objects run on detour and on Mesa | **203 / 203** calls: every `glGetError` and every queried value the same |
| 🔣 | Encodings (`tests/binary`) | Xe3 EU vs gentool and genxml vs Mesa, RDNA2 vs llvm-mc and Mesa's registers, AGX vs Mesa's AGX2 cases and `agx_pack`, Apple9 vs what an M4 ran: **7 / 7** |
| 🪜 | Lowerings (`tests/pass`) | kernels as acgpu compiled them (Xe3, RDNA2, AGX), GLSL reaching Xe3, an RDNA2 IB and an AGX CDM stream as Mesa writes them, Apple9 kernels as the M4 ran them: **7 / 7** |

## 🎮 GPUs

| GPU | Kernel driver | Status |
| --- | --- | --- |
| 🔵 **Intel Xe3** (Panther Lake, PCI 8086:b0a0) | `xe` | GL ES 3.0 draws in progress (above) |

## 🗂️ Layout

| Path | What |
| --- | --- |
| `src/gl` | GL ES 3.0: context, buffers, vertex arrays, shaders and programs (default uniform block), textures, renderbuffers, framebuffers, GLenums (`enums.gen.rs` from `GLES3/gl3.h`) |
| `src/vk` | Vulkan (to come) |
| — | Direct3D 11 and 12 (to come) |
| `src/dialect` | mlirformat plugins: the `xe3`, `rdna2`, `agx`, `apple9` and `image` dialects |
| `src/pass` | mlirpass plugins: `convert-gpu-to-*`, register allocation, waits, command streams, image layout, `Xe3Draw`, `gpu-module-to-binary` with these GPUs |
| `src/binary` | machine code and images: each GPU's encoder, records packed by layout tables, image writing |
| `src/shader.rs`, `src/render.rs` | the toolchain (mlirpass + detour's passes) and the device a draw runs on |
| `src/kmd`, `src/xe.rs` | the kernel interface and `xe` |
| `examples/` | `compare` (draws vs Mesa), `glcalls` (GL calls vs Mesa), `gl_link`, `xe_info` |
| `tests/binary`, `tests/pass` | encodings, and lowerings |
| `tools/{xe3gen,rdna2gen,agxgen}`, `data/{intel,amd,apple}` | the dialects' generators and the Mesa data they read |
| `scripts/` | `gl_enums.py`, opcode tables, and the oracles each test's expected bytes come from (`*_oracle.*`, `apple9_check.py`) |

## 🚀 Quick start

```sh
# 🔍 the Xe3 this machine has
cargo run --example xe_info

# 🧪 Xe3 encodings and lowerings
cargo test --test xe3

# 🎨 draws vs Mesa (glrun: mlirpass's tests/support/glrun.c)
cc -O1 -o glrun path/to/mlirpass/tests/support/glrun.c -lEGL -lGL
cargo run --example compare -- ./glrun

# 📞 GL calls vs Mesa
cc -O1 -o glcalls examples/support/glcalls.c -lEGL -lGLESv2
cargo run --example glcalls -- ./glcalls
```

## 📜 License

Apache-2.0. `data/intel/xe3.xml` is Mesa's genxml (MIT, Intel).

<div align="center">

**🛣️ Take the detour. The long way round is the short cut.**

[⬆️ Back to top](#top) · [🇨🇳 简体中文](README.zh-CN.md)

</div>
