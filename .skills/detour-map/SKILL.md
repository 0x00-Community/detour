---
name: detour-map
description: "Map of the detour repository: what each crate does (kmd trait, xe, detour dialect and frontend, xe3 dialect, codegen, driver runtime, cmdgen), which files and passes a triangle goes through from MLIR to the GPU, where the data files and comparison files are. Read when first entering the repo, or when looking for where to change something."
---

# detour map

## Crates

| Path | Crate | What it does |
| --- | --- | --- |
| `crates/core/src/kmd*` | `detour-core` | Kernel interface traits: `Gpu` (`Bo` / `Vm` / `Queue` / `Fence`, GAT borrows of the device), CPU side `Host` (`hosts!` macro, `Native`) |
| `crates/core/src/image.rs`, `pack.rs` | `detour-core` | Vendor-neutral image: `image` dialect (`Kernel::all`), instruction op macro `attr_ops!`, `OpRows`, `rewrites!` / `Rewrites`, `ImageTarget`, `Layout`, `ImageEmit`; table-driven packing `Layouts` / `Template` |
| `crates/core/src/compiler/build.rs` | `detour-core` | Construction: `A`, `WithAttrs`, `Slots`, `RegType` |
| `crates/core/src/regalloc.rs` | `detour-core` | Linear scan `LinearScan` + `RegTarget` (every vendor's register allocation uses it; Intel's `xe3-regalloc` only handles message payloads and rewriting) |
| `crates/core/src/raster.rs` | `detour-core` | The CPU half of 2D rasterization: vello_common strips ⇒ instances + alpha atlas + paint atlas (`Strips`) |
| `crates/core/src/scene.rs`, `filter.rs` | `detour-core` | The 2D scene acde draws into: `PaintScene`, `Scene` / `RenderCommand`, `Paint`, CSS filters `Filter` |
| `src/` (repo root) | `detour` | Aggregate exports: scene + one feature per GPU model (`xe3` / `agx-g13` / `rdna2`) + `fallback-vulkan` fallback; `Backend::pick` |
| `src/main.rs`, `src/cli/` | `detour` (bin, feature `cli`) | Debug tool: `main.rs` only assembles, one file per clap subcommand (`devices`: render node ⇒ driver, model, how far it is supported) |
| `benches/triangle`, `benches/triangle3d`, `benches/interop` | `triangle`, `triangle3d`, `interop` | Window examples (winit + softbuffer, Xe3): triangle; rotating 3D triangle; interop: the fallback's dma-buf image drawn in an Xe3 page, against vello_cpu. `triangle`'s lib is the window skeleton (`Scene`, `Shown`) |
| `crates/fallback` | `detour-fallback` | Vulkan fallback (`fallback-vulkan`): `Vello`, `Composer`, `Device` (`DeviceRequest` tiers), `Raster`, `Offscreen` |
| `crates/core/src/compiler*` | `detour-core` | `detour` dialect (`compiler/dialect.rs`), `Stage`, `Compiler`, frontend passes: `detour-from-spirv` (`spirv.rs`), `detour-from-gpu` (`gpu.rs`), `detour-expand-clear` (`clear.rs`); `FRONTEND` + `FrontendPasses` |
| `crates/amd/amdgpu` | `detour-amd-amdgpu` | amdgpu ioctls (`sys.rs` private), `impl Gpu for Amdgpu` |
| `crates/apple/codegen` | `detour-apple-codegen` | Bridge `convert-detour-to-gpu` (detour shaders ⇒ `gpu.func`) + mlirpass's AGX target (`agx` dialect, `convert-gpu-to-agx`, registers, `AgxDispatch`, `AGX_IMAGE` are all in mlirpass) |
| `crates/amd/codegen` | `detour-amd-codegen` | `rdna2-precheck` + detour-core's bridge + mlirpass's RDNA2 target (`rdna2` dialect, `convert-gpu-to-rdna2`, registers, PM4, `Rdna2Dispatch`, `RDNA2_IMAGE` are all in mlirpass) |
| `crates/apple/asahi` | `detour-apple-asahi` | asahi ioctls (`sys.rs` private), `impl Gpu for Asahi` (Apple M1 / M2, Asahi Linux) |
| `crates/intel/xe` | `detour-intel-xe` | xe ioctls (`sys.rs` private), `impl Gpu for Xe` |
| `crates/intel/dialect` | `detour-intel-dialect` | `xe3` dialect: genxml ops and layout table (`cmd.gen.rs`, generated), `XE3` table, EU instructions (`eu.rs` encoding, `asm.rs` ⇄ op, `msg.rs` message descriptors); the packing engine and container are in detour-core (`pack.rs`, `image.rs`) |
| `crates/intel/codegen` | `detour-intel-codegen` | `BACKEND`: `convert-detour-to-xe3` (`lib.rs` + `shader.rs` + `render.rs`), `xe3-regalloc` (`regalloc.rs`), `xe3-swsb` (`swsb.rs`), `xe3-pad` / `xe3-end` / `xe3-layout` + `Xe3` (`ImageTarget`) + `XE3_IMAGE` (`layout.rs`); `Codegen` |
| `crates/intel/driver` | `detour-intel` | Runtime (`runtime.rs`), strip compositor (`raster.rs`), blur effects (`frost.rs`: frosted glass, filter, drop-shadow, compute, generates MLIR), one-shot `Device` (`lib.rs`); examples |
| `crates/vello_cpu` | (module `vello_cpu`, xe3) | crates.io vello_cpu 0.3.0 copied in as is, exports coarse rasterization (`RenderContext::coarse`); changes are marked `detour:`, explained in `DETOUR.md` |
| `scripts/intel-cmdgen.py` | — | `crates/intel/dialect/genxml/xe3.xml` ⇒ `cmd.gen.rs` |

## The path of a triangle

```
examples/triangle/func.mlir      detour.* + func.func {detour.stage}
  Compiler::parse                 (mlirformat, loads the detour + xe3 dialects)
  FRONTEND                        spirv / gpu ⇒ func; clear ⇒ clear draw; canonicalize, cse
  convert-detour-to-xe3            shader.rs: func ⇒ image.kernel (virtual xe3.eu, !xe3.grf<f32>)
                                  render.rs: Render + Cmds ⇒ image.memory / image.stream / xe3.state.*
  xe3-regalloc, xe3-swsb          physical registers; regdist
  xe3-pad, xe3-end                kernel pad; batch gets MI_BATCH_BUFFER_END (with MI_NOOP to fill a qword)
  xe3-layout                      addresses, symbols ⇒ numbers
  XE3_IMAGE (ImageEmit<Xe3>)      one byte block per memory (XE3.pack, Inst::encode)
  Device::run / Context           kmd trait ⇒ xe ⇒ GPU
```

The runtime (`frames` example) only runs up to `xe3-swsb` in `Context::pipeline`; after that each frame uses templates.

## Data and comparisons

| File | What it is |
| --- | --- |
| `crates/intel/dialect/genxml/xe3.xml` | Mesa genxml (flattened), input to cmdgen |
| `crates/intel/dialect/tests/oracle.{mlir,expected}` | 290 entries, results of Mesa C packing |
| `crates/intel/dialect/tests/eu.{mlir,expected}` | 67 EU instructions, results of gentool assembly |
| `crates/intel/driver/examples/triangle/*.mlir` | the triangle's four shader forms |
| `crates/intel/driver/examples/frames/triangle.mlir` | a module with only the pipeline (used by the runtime) |
