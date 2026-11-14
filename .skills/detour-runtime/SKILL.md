---
name: detour-runtime
description: "Working on the detour runtime (crates/intel/driver/src/runtime.rs): what Context / Pipeline / Target / Buffer / Recorder each manage, heaps and addresses, the template hole naming convention and who fills them, the three places to change when adding a draw parameter, the batch ring, fences and sync_file / dma-buf export, how to measure with the frames example. Read when changing the per-frame path, hooking up a compositor, or adding a resource type."
---

# Runtime

Details and the hole name table are in `docs/runtime.md`.

## Rules

- ⛔ The per-frame path (`Context::record`, `Recorder::*`, `Template::emit`, `Context::submit`) runs no MLIR, reads no attributes, creates no BOs.
- Things built once (pipelines, targets, buffers) go into `Context::pipeline / target / buffer`; they can be slow.
- A missed hole becomes address 0 ⇒ GPU page fault, banned queue: every new hole must be recognized by the check in `Context::pipeline`.

## Adding a parameter known only at draw time

1. `crates/intel/codegen/src/render.rs` `Cmds::draw` (or `DrawArgs`): the corresponding field takes `A::sym("name")`.
2. `fill` in `runtime.rs` `Context::pipeline`: recognize this name as a kind of `Fill` (unrecognized names are resolved against `Context`'s own names; if that fails, it reports an error).
3. `value` in `runtime.rs` `Recorder::draw_at`: return the value according to the `Fill`.

The module lowering side (`Render::draw`) still gives numbers.

## Adding per-target state

Add it in `Cmds::pass_state` (and add to the returned name array); `Context::target` puts it into the heap and records the address; add the name used in the draw template to `SHARED`.

## Submit, wait, export

`submit(dw)` ⇒ the next slot in the ring (waits first if full), returns the slot number; `wait(n)` waits and checks for faults; `export_fence(n)` ⇒ sync_file; `Target::export()` ⇒ dma-buf
(linear, `width * 4` bytes per row). After the CPU finishes writing WC memory, `H::fence()` before submitting (already done).

## Measuring

`cargo run --features xe3 --example frames [N]`: recording p50 / p99, back-to-back throughput, per-frame latency, per pixel, export. Rerun after changing the per-frame path, and update the results in
`docs/verification.md`.
