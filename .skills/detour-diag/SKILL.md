---
name: detour-diag
description: "How to investigate when detour goes wrong: GPU fault / banned queue (Error::Fault, ban), job timeout, wrong pixels, rejected commands (packing errors, unfilled holes); reading the page fault address in journalctl -k, store-and-read-back markers in the batch, bisecting by disabling stages, using the lower example to see the MLIR after each pass, checking against known pitfalls."
---

# When things go wrong

## Banned queue (`Error::Fault`), page fault

1. `journalctl -k | grep -i xe | tail` to see the faulting address and engine. Where the address falls: compare with the `addr` / `offset` after `xe3-layout` printed by `lower`;
   for the runtime, compare with `Context`'s heaps (from 1 MB: code heap 1 MB, pool 64 KB, batch ring) and each BO's address.
2. Address 0 or very small: most likely an **unfilled hole / unresolved symbol** (e.g. the binding table pool base address, a state pointer). The runtime's template check reports
   `nothing fills @x`; for the module path, check whether `xe3-layout` missed a SymbolRef.
3. Locating the command it hangs on: insert `xe3.cmd.mi_store_data_imm {address = [@x, n], immediate_data = ...}` markers in the batch, read back to see which ones landed.
4. Bisect: disable a stage (e.g. PS) or delete a stretch of commands and run again.

## Wrong pixels

- First use the `lower` example to see the MLIR after each pass: `cargo run --features xe3 --example lower -- FILE 'builtin.module(<up to some pass>)'`.
- Shaders: look at `xe3.eu` after `xe3-swsb` (registers, regdist). Waiting too little on a dependency (swsb) reads stale values, intermittently.
- Read the pipeline statistics (the render pass's `statistics = @q`): whether the IA / VS / clipping / PS counts are right tells you whether nothing was drawn or it was drawn wrong.

## Known pitfalls (see "Pitfalls" in docs/xe3.md)

- The binding table must be in the pool; every stage must emit a binding table pointer (stages that bind nothing get a null table).
- `STATE_SYSTEM_MEM_FENCE_ADDRESS` must be 4 KB aligned.
- The EOT message sources are in r112 ..= r127.
- A signaled fence does not mean it finished: check `Queue::fault()`.
- There is no `3DSTATE_PUSH_CONSTANT_ALLOC_PS` (from Xe-HP on).

## Packing / compile errors

- `no field x`: wrong attribute name; look up the field name in `cmd.gen.rs`.
- `does not fit`: the value exceeds the field width, or the address is not aligned (the low bits of an address field must be 0).
- `@x is not a number yet`: `pack` hit a symbol; assign addresses first or use `template`.
