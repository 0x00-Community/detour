<a id="top"></a>
<div align="center">

# detour

### Skip Mesa. Skip libdrm.
**A GPU userspace driver in Rust, compiled through MLIR, tuned to the last cycle for two integrated GPUs.**

[![Rust](https://img.shields.io/badge/Rust-100%25-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
[![MLIR](https://img.shields.io/badge/IR-MLIR-262D3A?style=for-the-badge&logo=llvm)](https://mlir.llvm.org)
[![License](https://img.shields.io/badge/License-Apache--2.0-brightgreen?style=for-the-badge)](#-license)

[![Intel Xe3](https://img.shields.io/badge/Intel-Xe3-0071C5?style=flat-square&logo=intel)](#-gpus)
[![AMD RDNA2](https://img.shields.io/badge/AMD-RDNA2-ED1C24?style=flat-square&logo=amd)](#-gpus)
[![Linux](https://img.shields.io/badge/Linux-xe%20%C2%B7%20amdgpu-FCC624?style=flat-square&logo=linux&logoColor=black)](#-gpus)

**🇺🇸 English** · [🇨🇳 简体中文](README.zh-CN.md)

</div>

---

## ⚡ What is detour

detour draws on two GPUs and nothing else. It takes SPIR-V, compiles it to the GPU's machine code, builds the command
streams and submits them **through the kernel driver's ioctls**: no Mesa, no libdrm, no graphics API (that is the
application's layer). It is judged by one thing: how fast it draws.

- 🧩 **Draws** (`src/render.rs`, `src/shader.rs`): a program (a vertex and a fragment shader in SPIR-V) compiled ahead
  of time or in place; targets, vertex and index buffers, uniform blocks, sampled textures (any surface, so render to
  texture too), blending, scissors, dma-bufs in and out. What a draw cannot be given is refused when compiled.
- 🔵 **Intel Xe3**: the xe3 dialect (EU instructions, genxml commands and states), SPIR-V lowered to EU code, register
  allocation, scoreboard, a draw's state as a second-level batch kept resident.
- 🔴 **AMD RDNA2** (gfx1036): SPIR-V compiled straight to GFX10 code (wave32, NGG passthrough; its own selection,
  register allocation and waits), a draw's PM4 state set where it changes.
- 🔌 **On mlirformat and mlirpass as plugins**: `src/dialect` adds the xe3 dialect, `src/pass` the lowerings
  (`Passes::with(DetourPasses::add)`), `src/binary` encodes the machine code.
- 🐧 **Kernel interface** (`src/kmd`): DRM render nodes, buffer objects, address spaces, submission with syncobjs, on
  Linux's `xe` and `amdgpu` drivers.

## 🎯 Checked

Each GPU against a CPU reference, and Xe3's fixed function against Mesa (iris 25.2.8) on the same GPU, byte for byte.

| ✔️ | What | Xe3 | RDNA2 |
| --- | --- | --- | --- |
| ✅ | `examples/correct`: coverage, fill rule, culling, scissors, viewports, instancing, every blend factor and equation, vertex formats, interpolation, std140 blocks, built-ins, varyings, local arrays, 20 math functions with NaN, infinities and denormals | **752 / 752**, fixed function as Mesa | **711 / 752**; 40 refused (not yet compiled), 1 fractional viewport a column wide |
| 🖼️ | `examples/texture`: nearest and linear, magnified and minified, the three wraps, RGBA8 and R8, render to texture, a part written | **15 / 15** | **15 / 15** |
| 🔢 | `examples/indexed`: u16 / u32 indices, `Unorm8x4` colors, indexed and plain draws taking turns | **5 / 5**, as the plain draw | **5 / 5** |
| 📤 | `examples/export`: a scanout surface exported as a dma-buf, read through another device, taken by KMS | **2 / 2** | – |
| 🔣 | `tests/binary`: encodings | EU vs gentool, genxml vs Mesa | vs llvm-mc |
| 🪜 | `tests/pass`: every SPIR-V case's code, or why the GPU refuses it, against the code checked in | ✓ | ✓ |

## 🎮 GPUs

| GPU | Kernel driver |
| --- | --- |
| 🔵 **Intel Xe3** (Panther Lake, PCI 8086:b0a0) | `xe` |
| 🔴 **AMD RDNA2** (gfx1036, Radeon 610M, PCI 1002:13c0) | `amdgpu` |

x86_64 Linux only. There is no fallback: no CPU rasterizer, no hand-off to the system's drivers.

## 🗂️ Layout

| Path | What |
| --- | --- |
| `src/dialect` | the `xe3` dialect, RDNA2's instruction table, the layouts records are packed by |
| `src/pass` | the lowerings: SPIR-V to Xe3 (`convert-gpu-to-xe3`, registers, control flow, SWSB, `Xe3Draw`) and to RDNA2 (`Rdna2Graphics`, `Rdna2Draw`) |
| `src/binary` | machine code and records encoded |
| `src/shader.rs`, `src/render.rs` | the toolchain, and the device a draw runs on (`src/render`: Xe, amdgpu, the ring they share) |
| `src/kmd` | the kernel interface: DRM, `xe`, `amdgpu` |
| `examples/` | `correct`, `texture`, `indexed`, `program`, `shapes`, `frame` (on either GPU); `export`, `xe_info` (Xe3); `amdgpu_info`, `amdgpu_draw`, `amdgpu_device` (RDNA2); `programs` (acdes's) |
| `tests/` | `binary` (encodings), `pass` (each GPU's code for every case) |
| `scripts/` | the xe3 dialect's generator (`xe3gen`), the RDNA2 opcode table, the oracles expected bytes come from |

## 🚀 Quick start

```sh
cargo test --release --lib --tests                       # encodings and code
cargo run --release --example correct                    # correctness on this machine's GPU
cargo run --release --example shapes                     # a frame of draws, timed
```

Xe3's comparison with Mesa needs `examples/support/glref.c` (`cc -O1 -o glref examples/support/glref.c -lEGL
-lGLESv2`, then `GLREF=./glref`).

## 📜 License

Apache-2.0. `scripts/data/intel/xe3.xml` is Mesa's genxml (MIT, Intel).

<div align="center">

**🛣️ Take the detour. The long way round is the short cut.**

[⬆️ Back to top](#top) · [🇨🇳 简体中文](README.zh-CN.md)

</div>
