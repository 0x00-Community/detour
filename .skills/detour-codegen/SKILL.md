---
name: detour-codegen
description: "Working on the detour compiler: frontend passes (spirv / gpu => func, clear expansion), shader lowering shader.rs (TypeConverter 1:N, #[convert] in #[patterns], conventions of virtual xe3.eu, payload, EOT messages), register allocation, SWSB pipes and distances, 3D state and commands Cmds (render.rs), xe3-pad / xe3-end finishing, xe3-layout addressing. Read when adding shader capabilities, changing draw state, or changing passes."
---

# Compiler

Pipeline: `FRONTEND` (`detour-core`) then `BACKEND` (`detour-intel-codegen`), see `docs/architecture.md`. mlirpass usage follows
`~/coding/calcc/src/lower.rs` (also mlirpass ⇒ a custom ISA). Macro expansion needs `pub(crate) use mlirpass::{...}` at the crate root.

## Frontend (`crates/core/src/compiler*`)

- The unified form of a shader: `func.func` with `detour.stage`; a vertex shader's arguments are the vertex attributes (by location), its result is the position; a fragment shader's result is the RT0 color.
- `detour-from-spirv`: the input is spirv's generic form (mlirformat does not understand spirv's custom form). Interface variables ⇒ arguments / results; arithmetic via `#[patterns]` + `#[map]`.
- `detour-from-gpu`: `gpu.func` ⇒ `func.func`.
- `detour-expand-clear`: `load = "clear"` ⇒ a clear draw at the start of the pass (full-screen triangle + a constant-color fragment shader).
- New ops go in `dialect.rs`; ones with `sym_name` must be marked inherent in `op_infos`, otherwise ops you build yourself print in the generic form.
- Test `tests/frontend.rs`: the three forms yield the same module.

## Shader lowering (`shader.rs`)

- One SIMD16 register is `!xe3.grf<f32>` (`Grf("f32").into()`); `ShaderTypes::converter` splits `vector<Nxf32>` into N of them (1:N).
- To add a one-to-one arith / math op: add a line `#[map("arith.addf" => "xe3.eu.add", via = Self::lanes(&[]))]` on `ShaderToXe3`
  (extra attributes go in the argument of `lanes`, e.g. `&[("neg1", A::Bool(true))]`). Only write `#[convert("op")]` when logic is needed; use
  `Self::eu(cr, before, "xe3.eu.add", &operands, attrs, "f32")` to build a virtual instruction, and `cr.replace(op, vec![values])` to hand over the results.
  Only accept what can be bit-exact (EU add / mul / mad / neg are exact; divf is rejected). AMD (`Self::valu(cr, before, Opcode::VAddU32, (&operands, &[Src0, Src1]), &consts, vop3, ty)` / `#[map(.. => "rdna2.isa.valu", via = Self::lanes(Opcode::.., &[Src0, Src1], &[]))]`) and
  Apple (`Self::usc(cr, before, UscOp::Fadd, &operands, &[S0, S1], |b| b.neg(S1), ty)` / `#[map(.. => "agx.usc.fadd", via = Self::lanes(UscOp::Fadd, &[S0, S1], |b| b))]`) are written the same way, see "Writing IR" in `docs/architecture.md`.
- Virtual instructions: SSA operands fill, in order, the source slots not given by attributes; the result is dst; `neg0` / `neg1` negate; a send's operands are the message payload, `src0_len` of them go into src0.
- `xe3.eu.payload {grf}` (a pseudo op declared in the dialect, built with `rw.emit(&EuPayload::new(..))`): values already in registers when the thread starts (vertex attributes from r2, 4 registers each; URB handle r1).
- Epilogue: vertex ⇒ URB write (handle + VUE header 4 dwords + position), fragment ⇒ RT write (sendc, 4 channels), both with EOT.
- For how the fragment shader payload and interpolation work, see "Xe2 / Xe3 fragment shader" in `docs/xe3.md` (after Mesa brw).

## Register allocation (`regalloc.rs`)

All changes go through the `Rewriter` (held by `Alloc`): insert movs with `set_before` + `create`, rewrite with `modify`, delete pseudo ops with `erase_op`. The two IR shapes before and after allocation are in `docs/architecture.md`.

Single-block SSA linear scan. The EOT message payload is precolored starting at r112 (values used only by it are generated there directly; the rest get a mov inserted). Temporaries go from after the payload up to r111.
Non-EOT send: a single-register payload uses the value itself, a multi-register one is copied into a contiguous run of registers; both are freed after the send (the result is allocated first, not overlapping the payload).
Multi-block control flow is not done yet.

## SWSB (`swsb.rs`)

Pipes are chosen by dst type (float / int / long / math); read/write dependencies (RAW, WAW, and **cross-pipe WAR**) compute the distance on that pipe (at most 7); multiple pipes use
`A@minimum`. A non-EOT send takes an SBID (32 in rotation; one still in flight is waited on first); writing a register it reads waits on `$n.src`, touching one it writes waits on `$n.dst`. RegDist and SBID
are combined on one instruction only in combinations Mesa allows (`inferred`); otherwise a sync.nop is emitted. Rules and pitfalls are in section 2 of `docs/optimizations.md`.

## 3D state and commands (`render.rs`)

- `Cmds`: only produces ops (batch / code heap state / surface pool state), never touches Body; `Render` puts them into the module through the `Rewriter`. `context()`, `null_surface()`, `pass_state()`, `draw()`,
  `statistics()`, `pc_flush()`. Module lowering (`Render`) and the runtime share it.
- Command order follows iris's first batch + each draw (the old hand-written triangle, git `b9af1c0`). Before changing state, first look up how Mesa iris / anv set the same field.
- All state base addresses are 0; pointers are GPU addresses, binding table pointers are offsets within the pool (`A::within("surfaces", ...)`).

## Finishing and registration

`xe3-pad` / `xe3-end` are idempotent patterns written with `rewrites!` (`PadKernel` / `EndBatch`), run by `Rewrites::apply`. New passes go into
`detour_core::pass_set!(Xe3Passes: ..)` in `lib.rs` and the `BACKEND` string.

## Addressing (`layout.rs`, engine in detour-core `image.rs`)

Memory starts at 1 MB, 64 KB aligned; items by `align`; symbols ⇒ numbers. `Xe3.contents` (`ImageTarget`) gives the bytes of an item (stream / kernel / data / state).
