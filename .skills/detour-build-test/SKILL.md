---
name: detour-build-test
description: "Building and verifying detour: cargo build / test (three GPU-free comparisons: Mesa packing, gentool EU encoding, frontend equivalence), running triangle / frames / store / xe-info on the local Xe3, regenerating cmd.gen.rs and eu.expected, debug vs release, acceptance criteria (per pixel, fault check, performance numbers with their conditions). Read before building, testing or delivering."
---

# Building and verifying

## Building

```sh
cargo build --workspace --all-targets      # zero warnings required
```

- A debug build is enough for verification. mlirformat in release (opt-level 3) compiles very slowly (several minutes or more); use it only when measuring performance, and say so when you do.
- Dependencies come only from crates.io: `mlirformat`, `mlirpass` (the same mlirformat), `libc`.
- Toolchain 1.96 (pinned by `rust-toolchain.toml`, `rust-version`): mlirformat (from 0.6) uses `if let` guards, which 1.91 fails to compile (E0658).
- A crates.io mirror may lag (aliyun often lacks the latest mlirformat): switch to another, e.g. `cargo --config 'source.crates-io.replace-with="ustc"' ...`.

## Verification without a GPU

```sh
cargo test --workspace
```

| Test | Compares what |
| --- | --- |
| `crates/intel/dialect/tests/oracle.rs` | the 290 entries of `oracle.mlir` packed == Mesa C packing (`oracle.expected`) |
| `crates/intel/dialect/tests/eu.rs` | the 67 encodings of `eu.mlir` == `gentool asm -p ptl` (`eu.expected`) |
| `crates/core/tests/frontend.rs` | the triangle's three forms (func / gpu / spirv) yield the same module through FRONTEND |
| `sizes_match_the_kernel` in `crates/intel/xe`, `crates/apple/asahi`, `detour-core` (`drm.rs`) | uAPI struct sizes, offsets, ioctl numbers == gcc on the kernel headers |

| `crates/amd/codegen/tests/kernels.rs` | the whole image of `ints` / `copy` through `rdna2-precheck`, the bridge and mlirpass's RDNA2 target (1 / 4 / 1024 workgroups) == detour's own RDNA2 backend before it moved (`tests/kernels/*.expected`) |

| `crates/apple/codegen/tests/ints.rs` | the whole image of `compute/ints` through the bridge and mlirpass's AGX target (1 / 4 / 1024 workgroups) == detour's own AGX backend before it moved (`tests/ints/g*.expected`) |


The whole Apple compute chain (no GPU): `cargo run --features agx-g13 --example agx-compute -- FILE [GROUPS] --bin OUT`,
check the kernel bytes by disassembling them with Mesa's `cd ~/coding/mesa/src/asahi/isa && PYTHONPATH=. python3 test/disasm.py OUT`.

The AGX comparisons (struct packing, CDM stream, USC encoding against Mesa) and their regeneration (`scripts/agx_*`, `tools/agxgen`) are in mlirpass (`tests/agx.rs`).

The whole AMD compute chain (no GPU): `cargo run --features rdna2 --example rdna2-compute -- FILE [GROUPS]` prints the IR after the whole lowering and the kernel's location;
add `--hex` to output only the kernel bytes, and check them with `llvm-mc -arch=amdgcn -mcpu=gfx1036 -mattr=+wavefrontsize32 --disassemble`.

The AMD comparisons (registers / packets against Mesa, ISA against llvm-mc, IB against Mesa macros) and their regeneration (`scripts/rdna2_*`, `tools/rdna2gen`) are in mlirpass (`tests/rdna2.rs`).

The aarch64 (Apple) parts are cross-compiled locally, and the tests run under qemu:

```sh
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64-static -L /usr/aarch64-linux-gnu" \
  cargo test --features agx-g13 --target aarch64-unknown-linux-gnu
```

If you change packing, encoding or the frontend, all of these groups must still pass.

## Verification that needs a GPU (this machine is Xe3)

```sh
cargo run --features xe3 --example triangle        # four forms, each: pipeline statistics + 64x64 per pixel
cargo run --features xe3 --example frames [N]      # runtime: recording / throughput / latency, 1920x1080 per pixel, dma-buf / sync_file export
cargo run --features xe3 --example store           # MI_STORE_DATA_IMM written in the xe3 dialect, read back
cargo run --features xe3 --example frost           # blur effect (compute), byte for byte against a Rust model; `frost K` runs only the K-th, `--mlir` prints the shader
cargo run --features xe3 --example xe-info      # device query
cargo run --features xe3 --example lower -- FILE [PIPELINE]   # only look at the MLIR, no GPU
```

First make sure nobody else is using the GPU (other sessions, running examples). Results must match the table in `docs/verification.md`; if the numbers change (better or worse), update the table.

## Regenerating

- Layout table: `python3 scripts/intel-cmdgen.py crates/intel/dialect/genxml/xe3.xml crates/intel/dialect/src/cmd.gen.rs`, then run the oracle test.
- EU comparison: add an entry to `tests/eu.mlir` (with the gentool assembly in a comment on the line above), `GENTOOL=<gentool path> scripts/intel-eu-oracle.sh`.
  How to build gentool is in `docs/verification.md`.

## Acceptance

- ⛔ Running is not enough: it only counts when the pixels and read-back numbers match one by one. `Device::run` / `Context::wait` already check `Queue::fault()`; paths you write yourself must check it too.
- Write the conditions with performance numbers (debug / release, resolution, what each frame draws, back to back or frame by frame, etc.).
- Report separately: verified (what ran, results), not verified, limited by the environment.
