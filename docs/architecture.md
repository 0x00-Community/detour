# Architecture

## Layers

MLIR all the way: dialects and text format use [mlirformat](https://github.com/Jabberwocky238/mlirformat), transformations use
[mlirpass](https://github.com/Jabberwocky238/mlirpass) (both from crates.io, the same mlirformat). Every layer is MLIR and can be printed to look at
(`cargo run --features xe3 --example lower -- FILE [PIPELINE]`).

```
detour.* (one WebGPU submit)   detour.buffer / texture / pipeline / render_pass { draw }; shaders: func.func {detour.stage},
   │                           or spirv.module (generic form deserialized from SPIR-V), or gpu.module
   │  detour-core (compiler)    detour-from-spirv, detour-from-gpu      shaders unified into func.func {detour.stage} (arith / vector)
   │  (FRONTEND)               detour-expand-clear                    load = clear ⇒ one clearing draw at the start
   │                           func.func(canonicalize, cse)          upstream
   │  detour-intel-codegen      convert-detour-to-xe3                  shaders ⇒ image.kernel (virtual xe3.eu); resources, passes ⇒
   │  (BACKEND)                                                      image.memory / xe3.state.* / image.stream (xe3.cmd.*)
   │                           xe3-regalloc, xe3-swsb                registers, software scoreboard
   │                           xe3-pad, xe3-end                      zero padding after kernels (pad), batch ending (MI_BATCH_BUFFER_END / MI_NOOP)
   │                           xe3-layout                            GPU addresses, symbols ⇒ numbers
   ▼
XE3_IMAGE (ImageEmit<Xe3>)    one byte segment per image.memory + the address of image.stream (batch)
   │  detour-intel (driver)     Device<G: Gpu, H: Host>: one BO per segment, VM_BIND, submit, wait on fence, check fault
   ▼
detour-core (kmd trait)        Gpu (device / Bo / Vm / Queue / Fence), Host (CPU cache flush / barriers)
   │  detour-intel-xe           impl Gpu for Xe: xe ioctls (/dev/dri/renderD*)
   ▼
xe (kernel)
```

| crate | contents |
| --- | --- |
| `crates/core` `detour-core` | One vendor-independent crate; three private modules, re-exported flat: |
| ↳ `src/kmd*` | `Gpu` (GAT-borrowed device: `Bo<'d>` `Vm<'d>` `Queue<'d>` `Fence<'d>`; what a queue runs is decided by the driver: `Gpu::Cmd`), `Host` (`hosts!` macro, x86-64 / aarch64), vendor-independent `Caching` / `MemType` / `Engine` / `DeviceInfo`; a layer shared by all Linux DRM drivers (`drm.rs`): `Node` (render node, ioctl, GEM close / mmap / dma-buf), `Syncobj` (fences for all drivers, `status()` reads the sync_file's error) |
| ↳ `src/image.rs` | The `image` dialect (`image.memory` / `stream` / `kernel` / `data` / `words` (dwords that may carry symbols), a vendor-independent image container; `Kernel::all` finds the kernels in a module), the instruction op macro `attr_ops!`, op rows `OpRows`, rewrite patterns `rewrites!` / `Rewrites::apply`, `ImageTarget` (vendor: command stream ending, kernel encoding, state packing), addressing `Layout`, `Image` / `Segment` / `ImageEmit` |
| ↳ `src/pack.rs` | Table-driven packing `Layouts` (`Item` / `Field` / `Ty` / `Group`) and templates `Template` (symbols as holes); each vendor's tables come from its own generator (Intel `XE3`; AMD's `RDNA2`, Apple's `AGX` and the AGX-specific `Ty` are in mlirpass) |
| ↳ `src/regalloc.rs` | Register allocation `LinearScan` (linear scan over SSA in one block: multiple register files, how many registers a value takes and its alignment, precoloring, contiguous ranges, sources and results kept apart); vendors implement `RegTarget` (Intel's `GrfFile`, AMD's `Files`; Apple's is in mlirpass) |
| ↳ `src/raster.rs` | The CPU half of 2D rasterization, vendor-independent: vello_common 0.3 sparse strips ⇒ quad instances (float: NDC rect, atlas uv, premultiplied color, paint atlas coordinates) + R8 alpha atlas + RGBA8 paint atlas (gradient LUTs; `Strips`) |
| ↳ `src/scene.rs`, `src/filter.rs` | The 2D scene acde's web layer paints into, vendor-independent, no GPU: `PaintScene` (the painting interface), `Scene` / `RenderCommand` (recorded commands), `Paint` / `Glyph`, CSS filters `Filter` / `FilterEffect` / `ColorMatrix`; the interface shape comes from anyrender 0.14 |
| root `src/` `detour` | Aggregate exports for users (acde) (canvas 3D features `webgl` / `webgl2` / `webgpu`, on by default) (the repository root is this package, a git dependency gets it directly): the 2D scene is always exported; one feature per GPU model (`xe3` Intel Panther Lake, `agx-g13` Apple M1, `rdna2` AMD), only the enabled vendor's crates are compiled; `fallback-vulkan` is the Vulkan fallback (on by default); `Backend::pick(Capability)` decides who does the work, panics when there is neither driver nor fallback |
| `crates/fallback` `detour-fallback` | Vulkan fallback (feature `fallback-vulkan`, on by default): what the driver cannot do, it does on Vulkan -- scenes painted by vello on wgpu (`Vello`), layer compositing (`Composer`: filters, drop shadows and frosted glass are all color matrices on straight alpha, blur plans the levels and offsets of dual Kawase via `Blur`), the Vulkan device on Linux (`Device`; opened in tiers by `DeviceRequest`: all WebGPU features → no optional features → downlevel limits) / dma-buf canvas / sync_file, full-page rasterization (`Raster`), shared device and GPU images (`Shared`, `GpuImage`, `ImageSet`, `Overrides`), offscreen canvas painting (`Offscreen`: paint, read back, write, shadow blur `blur.wgsl`, CSS filter `matrix.wgsl`). The fallback and acde's acde-vello are the same code (moved back from acde on 2026-11-14: vello 0.12, `Op` batched compositing, `Blur` planning), tests in `crates/fallback/tests/compose.rs`; exports `vello` / `wgpu` |
| ↳ `src/compiler*` | The detour dialect, `Stage`, the `Compiler` context, the three passes of `FRONTEND` |
| `crates/intel/xe` `detour-intel-xe` | xe uAPI (`sys.rs` private), `Xe` |
| `crates/intel/dialect` `detour-intel-dialect` | The xe3 dialect: `xe3.cmd.*` / `xe3.state.*` / `xe3.reg.*` (generated by cmdgen from genxml, table-driven packing `XE3.pack`), `xe3.eu.*` (`Inst` ⇄ op, `Inst::encode`), containers `image.memory` / `batch` / `kernel` / `data` |
| `crates/intel/codegen` `detour-intel-codegen` | The four passes of `BACKEND`, `Xe3` (`ImageTarget`), `XE3_IMAGE`, `Codegen` |
| `crates/intel/driver` `detour-intel` | Strip compositor `Compositor` (`raster.rs`, pipeline in `raster/strips.mlir`); runtime `Context` / `Pipeline` / `Target` / `Buffer` / `Recorder` (`runtime.rs`); one-shot `Device` / `Frame`; examples `frames`, `triangle`, `store`, `lower` |
| `scripts/intel-cmdgen.py` | `crates/intel/dialect/genxml/xe3.xml` ⇒ `crates/intel/dialect/src/cmd.gen.rs` |
| `crates/amd/codegen` `detour-amd-codegen` | Runs on mlirpass's RDNA2 target (feature `target.rdna2`; moved there from detour on 2026-11-08: the `rdna2` dialect (register tables, PKT3, ISA encoding), `convert-gpu-to-rdna2`, `rdna2-regalloc` / `-waitcnt` / `-pm4` / `-pad` / `-layout`, `Rdna2Dispatch`, `RDNA2_IMAGE`); only these remain here: `rdna2-precheck` (precheck.rs: accepts only compute, single-thread workgroups, flat memory accesses of 1–4 dwords or a single byte / short without binding / shared, and `workgroup_id` / `lane` / `unpack4xu8`; everything else is rejected, so it is not dropped or compiled into something else), the detour-core bridge `convert-detour-to-gpu`, `Codegen` (`image`: one dispatch into an image), example `rdna2-compute`; `tests/kernels.rs`: the images of `ints` / `copy` are byte-identical to detour's own RDNA2 backend before the move |
| `crates/amd/amdgpu` `detour-amd-amdgpu` | amdgpu uAPI (`sys.rs` private), `Amdgpu` (Radeon, starting with the Radeon 610M of Ryzen 9000 HX, GC 10.3.6); submission follows radv (IB + BO list + syncobj); example `amdgpu-info` |
| `crates/apple/codegen` `detour-apple-codegen` | Runs on mlirpass's AGX target (feature `target.agx`; moved there from detour on 2026-11-08: the `agx` dialect (Mesa genxml structs, AGX2 USC instructions), `convert-gpu-to-agx`, `agx-regalloc` / `agx-wait` / `agx-end` / `agx-layout`, `AgxDispatch`, `AGX_IMAGE`); only these remain here: the bridge `convert-detour-to-gpu` (detour compute shaders ⇒ `gpu.func` kernels in `gpu.module @shaders`: `detour.workgroup_id` / `lane` ⇒ `gpu.block_id` / `thread_id` + `index_cast`, `detour.load` / `store` ⇒ `llvm.inttoptr` + `llvm.load` / `store`; other stages, binding / shared / non-4-byte accesses are rejected), `Codegen` (`image`: one dispatch into an image), example `agx-compute`; `tests/ints.rs`: the image is byte-identical to detour's own AGX backend before the move. The `apple9` module: Apple M3 / M4 runs on mlirpass's Apple9 target (feature `target.apple9`): `apple9-check` (accepts only compute, workgroups of one or two threads; with one thread, `known_block_size` 16 is stated) + bridge + `convert-gpu-to-apple9` / `apple9-regalloc` / `-control` / `-wait`, `Apple9::kernels` produces `Apple9Code`; `tests/apple9.rs`: machine code of `ints`; with `APPLE9_METAL` / `APPLE9_DONOR` set it runs on the M4 and checks each value |
| `crates/apple/metal` `detour-apple-metal` | Runtime for Apple M3 / M4 on macOS: `Apple9Code` runs through Metal (the code is put into a donor kernel's binary archive replacing `_agc.main`, the pipeline is built with `failOnBinaryArchiveMiss`; the archive is made by build.rs), `M4::open` / `kernel` / `buffer` / `dispatch` (binding 0 memory, binding 1 arguments and grid size, workgroups of 16 are dispatched as 16·G); compiled only on macOS; `tests/ints.rs` checks each value on the M4 |
| `crates/apple/asahi` `detour-apple-asahi` | asahi uAPI (`sys.rs` private), `Asahi` (Apple M1 / M2, Asahi Linux); example `asahi-info` |

The runtime (no MLIR per frame, only filling templates) is in [runtime.md](runtime.md).

Status: a triangle (clear + one draw) runs end to end from MLIR to image; four ways of writing the shader (func / gpu / spirv / func with arithmetic) paint correctly pixel by pixel on the local machine;
the minimal xe3.cmd job (`store`) runs. Compute pipelines (compute stage) are recognized by the frontend, the backend is not done yet (rejected).

## Writing IR (the standard use of mlirformat / mlirpass)

All three backends are written the same way; follow it for anything new:

- **Dialects**: ops are declared with `#[mlir::op]` / `#[mlir::dialect]`. Instruction ops (`attr_ops!`) have variadic operands and variadic results, the format is
  `` (`(` $operands^ `:` type($operands) `)`)? attr-dict (`->` type($results)^)? ``, so both forms below are valid spellings of the same op,
  and both print and parse back unchanged. Each dialect's `op_infos` declares effects (all `Unknown`: cannot be removed as dead code) and inherent attributes with `OpRows::of`.
- **Two IR forms**: before allocation, instructions are virtual: SSA operands + a `slots` attribute (core `Slots`: which field each operand fills), results are values of register type;
  after allocation, operands and results are cleared, and registers are written in each vendor's assembly syntax into the attributes named by `slots` (Intel gentool strings, AMD LLVM syntax `v[4:7]`, Apple Mesa
  encoding dictionaries), and `slots` is removed.
  ```
  %5 = rdna2.isa.v_add_u32(%3, %4 : !rdna2.v<i32>, !rdna2.v<i32>) {slots = ["src0", "src1"]} -> !rdna2.v<i32>   before allocation
  rdna2.isa.v_add_u32 {dst = "v11", src0 = "v11", src1 = "v0"}                                                 after allocation
  ```
  AMD instructions are split into ops by encoding family, the mnemonic is the `op` attribute, `slots` is the `Slot` enum (the encoding hot path does not go through typed op serde: construction uses
  `Build`, reading uses `Isa::read` ⇒ `Fields`):
  ```
  %5 = rdna2.isa.valu(%3, %4 : !rdna2.v<i32>, !rdna2.v<i32>) {op = "v_add_u32", slots = ["src0", "src1"]} -> !rdna2.v<i32>   before allocation
  rdna2.isa.valu {op = "v_add_u32", dst = "v11", src0 = "v11", src1 = "v0"}                                                 after allocation
  ```
  Intel has no `slots`: SSA operands fill, in order, the source slots (`src0` / `src1` / `src2`) not given by attributes; the result is `dst`.
- **Register types**: `!xe3.grf<..>` / `!rdna2.v<..>` `!rdna2.s<..>` / `!agx.r<..>` `!agx.u<..>` are each dialect's typed wrappers (`Grf`,
  `Rdna2Type`, `AgxType`, with core `RegType` underneath), converted with `From` / `TryFrom<&Type>`, no string concatenation.
- **Pseudo ops**: values already in registers when the thread starts, `xe3.eu.payload {grf}`, `rdna2.isa.input {reg}`, `agx.usc.input {uniform}`, are declared in the dialect,
  and built with typed structs `rw.emit(&EuPayload::new(..))`; register allocation pins them there, then deletes them.
- **Lowering**: `#[patterns(converter)]` + `ShaderTypes::converter` (1:N vector splitting). One-to-one mappings are written as `#[map("arith.addf" => "xe3.eu.add",
  via = Self::lanes(..))]`, one line each (the template `lanes` builds an instruction per component); only those needing logic are written as `#[convert]`.
- **Changing IR**: always through mlirpass `Rewriter` (`create` / `emit` / `set_before` / `modify` / `erase_op` / `move_before`); in fn passes use
  `Rewriter::new(body, &mut Quiet)`; endings and padding (`xe3-end` / `xe3-pad` / `rdna2-pad` / `agx-end`) are idempotent `Pattern`s
  (`rewrites!`) run by the greedy driver through `Rewrites::apply`. No direct `body.create` / `insert` / `erase`.
- **Pass registration**: grouped with `detour_core::pass_set!` (`FrontendPasses`, `Xe3Passes`, `Rdna2Passes` + `Rdna2ImagePasses`, `AgxPasses` +
  `AgxImagePasses`), `passes.with(Xe3Passes::add)`; each group corresponds one to one with its pipeline string (`FRONTEND` / `BACKEND`).
- **Typed ops (detour, image dialects)**: one hand-written `#[mlir::op]` struct per op, attributes are typed fields (serde enums, `Option<u32>`,
  `Vec<..>`, `bool` is a unit attribute); the dialect's `INHERENT` table lists each op's attributes, and the unit test `inherent` checks it matches the struct fields.
  Read with `Heads::head::<O>(op)` (looks only at this op: does not export regions, does not name the whole body, ⛔ `body.operation(op).to_op()`), modify with
  `Heads::update` / `Updates::update` (through `Rewriter`, so the listener knows), build region-less ops with `rw.emit(&O::new(..))`, ops with regions
  (`image.memory` / `stream` / `kernel`) with `OpData::head(&o)` + `create_with` adding regions, or `body.container(&o)`.
  ⛔ fetching attributes by name string and then matching, ⛔ `OpData::build("image.data").with(..)`; compare names with `O::NAME`, symbol names with `Symbol::defined`.
  The `align` / `offset` of each vendor's in-memory state structs (`xe3.state.*` etc.) are discardable attributes (`Placed::ALIGN` / `OFFSET`).
- **Building attributes**: `A` (`A::str` / `A::sym` / `A::within` / `A::dict`), `OpData::with_attrs(vec![..])` (`WithAttrs`).

## Division of work with the kernel

detour is in user space, at the same layer as Mesa, and uses only the kernel's existing uAPI (xe: VM_BIND, exec queue, syncobj, userptr). Its speed comes from its own compiler and a shorter
submission path, not from the kernel. ackernel does not build in anything of detour (patches, config, modules, firmware): when hitting a hard uAPI limit, tell the user first. Tuning
(GT frequency, preemption and timeouts) goes through sysfs, set manually on the test machine. The test machine's kernel ships xe.

## Plugging into the kernel (asahi)

- Apple M1 / M2 (AGX G13 / G14) run Asahi Linux, kernel driver asahi (uAPI in mainline `include/uapi/drm/asahi_drm.h`, the same in Linux 7.2 and Mesa's
  `include/drm-uapi`; the driver itself is in Asahi's kernel branch). Structs are in `crates/apple/asahi/src/sys.rs`; tests check them against gcc's
  `sizeof` / offsets / ioctl numbers for the header.
- Unlike xe, a queue does not run batches but commands handed to firmware: `Compute` (`drm_asahi_cmd_compute`: start and end of the CDM control stream) and `Render`
  (`drm_asahi_cmd_render`: VDM control stream and tiler state). `Gpu::Cmd = [Command]`; within one submit each command waits for all earlier commands (the
  barriers of both subqueues are filled with the count of earlier commands), in order like one xe batch.
- VM layout follows Mesa (`agx_device.c` `agx_open_device`): the top goes to the kernel (at least 32 GiB), at the bottom a 4 GiB-aligned USC window (shader addresses are 32-bit relative to
  `usc_exec_base`), the middle is the user's. The first 4 GiB are unmapped, so null pointers fault. Pages are 16 KiB.
- VM_BIND is synchronous (no sync); after `bind` the CPU signals `done` directly. GPU-side cache attributes follow the BO (WRITEBACK or WC), not the mapping.
- Errors: asahi has no ban query like xe, and Mesa (hk / gallium asahi) does not check either. `fault(done)` reads the status of the fence's sync_file (negative is errno);
  whether the kernel sets an error on the fence on a GPU page fault has not been confirmed on real hardware yet.

## Plugging into the kernel (xe)

- Opens the xe driver's render node directly (`/dev/dri/renderD*`, driver identified with `DRM_IOCTL_VERSION`; `Node` in `detour-core`), no libdrm.
  Structs are in `crates/intel/xe/src/sys.rs`, hand-written from `include/uapi/drm/xe_drm.h` (`repr(C)`, tests check sizes and ioctl numbers); those from drm.h
  (syncobj, GEM close, prime) are in `detour-core`'s `drm.rs`.
- One VM and one render-engine exec queue per GPUDevice: a hung context only bans its own queue. The VM uses dma-fence mode,
  submission signals through syncobj (the acquire fence handed to the compositor is exported from it as a sync_file).
- GPU virtual addresses are allocated by detour itself; a BO is bound once and stays valid (xe has no relocations); submissions do not bind again.
- Integrated graphics have no VRAM, everything is system memory. CPU-written, GPU-read memory (batches, uploads) uses WC + PAT 0; memory read back by the CPU uses WB + PAT 1 (1-way coherent);
  render targets will later use a compressed PAT (9). The PAT table is Mesa's `XE2_PAT_ENTRIES` (Xe3 is the same as Xe2).
