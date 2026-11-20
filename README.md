<a id="top"></a>
<div align="center">

# detour

### Skip Mesa. Skip libdrm.
**A GPU userspace driver in Rust, compiled through MLIR, tuned to the last cycle for a few GPUs (Xe3 so far).**

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

detour is **our own GPU userspace driver**, made for a few GPUs and nothing else: no graphics API to emulate, just
draws (`Device::draw`) whose shaders it compiles to the GPU's machine code, whose command streams it builds and submits
**through the kernel driver's ioctls**. No Mesa, no libdrm underneath. Graphics APIs (Vulkan, WebGPU) are the
application's layer, not detour's.

- 🧩 **Draws** (`src/render.rs`, `src/shader.rs`): SPIR-V shaders (GLSL through [mlirpass](https://github.com/Jabberwocky238/mlirpass))
  compiled to Xe3 or RDNA2, ahead of time or in place; targets, vertex buffers, uniforms, sampled textures (any surface,
  so render to texture too), blending, scissors, dma-bufs.
- 🔵 **Intel Xe3**: the xe3 dialect (EU instructions, genxml commands and states), SPIR-V shaders lowered to Xe3
  machine code, register allocation, scoreboard, a draw's image.
- 🔴 **AMD RDNA2** (gfx1036): SPIR-V shaders compiled straight to GFX10 machine code (wave32, NGG passthrough; its
  own selection, register allocation and waits), a draw's PM4 state, and the `amdgpu` kernel interface (buffer objects,
  the address space, command submission with syncobjs, dma-bufs).
- 🔌 **On mlirformat and mlirpass as plugins**: `src/dialect` adds the xe3 dialect to mlirformat (`MlirOpt::load`),
  `src/pass` adds the lowerings to mlirpass (`Passes::with(DetourPasses::add)`), `src/binary` encodes Xe3 machine code
  and packs genxml records.
- 🐧 **Kernel interface** (`src/kmd`, its `xe` driver in `src/kmd/xe`): DRM render nodes, buffer objects, VM binds, exec queues, syncobjs,
  on Linux's `xe` driver.

## 🎯 Checked against Mesa

Deploying is not success: every result is compared with the local Mesa (iris, 25.2.8), byte for byte.

| ✔️ | What | Result |
| --- | --- | --- |
| ✅ | Correctness (`examples/correct`): draws through `Device::draw` against a CPU reference (pixel centres, 8-bit sub-pixel snapping, the top-left rule, GL ES blending, Vulkan's precision for shader math) and against Mesa on the same GPU: coverage, shared edges, thin, degenerate and huge triangles, strips, first / count, culling, scissors, viewports, instancing, every blend factor and equation, the color mask, vertex formats and strides, interpolation, std140 blocks, built-ins, 15 varyings, local arrays, 20 math functions with NaN, infinities and denormals | **752 / 752** cases: fixed function **byte for byte as Mesa**; the f32 reference ±1 only where Xe3 blends or rounds to unorm8 (Mesa the same) |
| 📤 | Export and scanout (`examples/export`): a scanout surface (Tile4 and Linear, 1920×1080) drawn into, exported as a dma-buf (`Device::export`: fd, pitch, modifier, fourcc), imported by another device and read back, and taken by KMS as a framebuffer (`ADDFB2` with the modifier) | **2 / 2** |
| 🔢 | Indexed draws and byte colors (`examples/indexed`): 3000 overlapping quads as four vertices and six u16 / u32 indices each, colors as `Unorm8x4`, from a middle index, indexed and plain draws taking turns | **5 / 5** cases **byte for byte** as the plain draw |
| 🖼️ | Textures (`examples/texture`): a quad sampling a surface against a CPU reference: nearest and linear, magnified and minified, clamp / repeat / mirrored repeat, RGBA8 and R8, linear and Tile4; drawn into then sampled, and sampled then drawn over, in one batch; a part written (`Device::write`) | **15 / 15** cases, nearest **byte for byte**, linear within 2 of the f32 reference |
| 🔣 | Encodings (`tests/binary`, `src/pass/rdna2`) | Xe3 EU vs gentool and genxml vs Mesa, RDNA2 vs llvm-mc |
| 🪜 | Lowerings (`tests/pass`) | Xe3 and RDNA2 code of hand-written SPIR-V, GLSL and acdes's page programs, held against a reading of it |

## 🎮 GPUs

| GPU | Kernel driver | Status |
| --- | --- | --- |
| 🔵 **Intel Xe3** (Panther Lake, PCI 8086:b0a0) | `xe` | draws (above) |
| 🔴 **AMD RDNA2** (gfx1036, Granite Ridge Radeon 610M, PCI 1002:13c0) | `amdgpu` | draws (acdes's pages; `examples/programs`, `examples/amdgpu_draw`) |

detour drives these integrated GPUs on x86_64 Linux and nothing else. There is no fallback: no CPU
rasterizer and no hand-off to the system's drivers.

## 🗂️ Layout

| Path | What |
| --- | --- |
| `src/dialect` | the `xe3` dialect (a mlirformat plugin), RDNA2's instruction table, the layout tables records are packed by |
| `src/pass` | mlirpass plugins: `convert-gpu-to-xe3`, register allocation, control flow, SWSB, `Xe3Draw`; RDNA2's shader compiler (`Rdna2Graphics`) and draw state |
| `src/binary` | Xe3 machine code, records packed by layout tables |
| `src/shader.rs`, `src/render.rs` | the toolchain (mlirpass + detour's passes) and the device a draw runs on |
| `src/kmd` | the kernel interface: DRM, the `Gpu` trait, the `xe` driver (`src/kmd/xe`) and the `amdgpu` driver (`src/kmd/amdgpu`) |
| `examples/` | `correct` (correctness vs a CPU reference and Mesa; `support/glref.c`), `texture` (sampled textures vs a CPU reference), `indexed` (indexed draws), `export` (dma-bufs and scanout), `program` (programs compiled ahead of time), `shapes` and `frame` (draws in a frame, timed), `programs` (acdes's programs drawn), `xe_info`, `amdgpu_info`, `amdgpu_draw`, `amdgpu_device` |
| `tests/binary`, `tests/pass` | encodings, and lowerings |
| `scripts/` | the xe3 dialect's generator (`xe3gen`) and the Mesa data it reads (`data/intel`), the RDNA2 opcode table (`rdna2_opcodes.py`, `data/amd`), and the oracles each test's expected bytes come from (`*_oracle.*`) |

## 🚀 Quick start

```sh
# 🔍 the Xe3 this machine has
cargo run --example xe_info

# 🧪 Xe3 encodings and lowerings
cargo test --test binary --test pass

# ✅ correctness vs a CPU reference and Mesa
cc -O1 -o glref examples/support/glref.c -lEGL -lGLESv2
GLREF=./glref cargo run --release --example correct

# 🏁 a frame of draws, timed
cargo run --release --example frame
```

## 📜 License

Apache-2.0. `scripts/data/intel/xe3.xml` is Mesa's genxml (MIT, Intel).

<div align="center">

**🛣️ Take the detour. The long way round is the short cut.**

[⬆️ Back to top](#top) · [🇨🇳 简体中文](README.zh-CN.md)

</div>
