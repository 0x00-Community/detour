# Runtime

`crates/intel/driver/src/runtime.rs`. MLIR runs only once, when a pipeline is built; each frame only copies templates and fills holes, no mlirpass, no reading attributes.
Same approach as anv: the pipeline's state is packed at build time and merged with the dynamic part at draw time (`_inst_data` | `_partial` in
`~/coding/mesa/src/intel/vulkan/genX_gfx_state.c`).

## Objects

| Type | What it is | Lifetime |
| --- | --- | --- |
| `Context<'d, G: Gpu, H: Host>` | One VM, one render queue; a code heap (1 MB, kernels, dynamic state, fences) and a binding table pool (64 KB, render surface states live there too), one resident BO each; a ring of 4 batches (64 KB); context commands (filled at build time, copied as is each frame) | One per device |
| `Pipeline` | The kernels' addresses in the code heap; all commands of one draw (`Cmds::draw`) packed into one template | Built once, used forever |
| `Kernel` | The compute shader's address in the code heap; the COMPUTE_WALKER + flush for each of three dispatch kinds (direct, with binding table, indirect) packed into one template each, holes (group counts `groups_x/y`, parameters `param0..7`, `table`, `args0..2`) recognized as `Walk` at build time; `Recorder::dispatch*` only copies the template and fills numbers | Built once, used forever |
| `Target<'d, G>` | Render target, its own BO (linear, can `export()` as a dma-buf); at build time its pass state (viewport, blend, color calc, surface, binding table) goes into the heap | One per canvas |
| `Buffer<'d, G>` | Vertices and other things a draw reads, its own BO | |
| `Image2d<'d, G>` | A texture sampled by fragment shaders (`detour.sample`), its own BO (linear), surface state in the pool | Built once |
| `Recorder` | One frame: `Context::record` → `target()` → `draw()` / `dispatch()` …→ `finish()` gives dwords, `Context::submit` submits | Per frame |

Addresses: BOs are laid out upward from 1 MB, 64 KB aligned, bound once and valid for a long time (xe has no relocations). All state base addresses are 0, pointers are GPU addresses
(state pointers are only 32 bits: everything is below 4 GB). The heap is a bump allocator and does not reclaim for now (pipelines and targets are long-lived).

## Templates and holes

`XE3.template(op, attrs)` treats symbols in attributes as holes when packing (recording the bit, width, address alignment shift, and the n of `[@x, n]`), filling 0 first;
`Template::emit(out, value)` copies the dwords and asks `value` by hole name for what to fill in. Hole name conventions:

| Hole name | Who fills it |
| --- | --- |
| `vs` and other kernel function names | The `Pipeline`'s kernel addresses |
| `vb0`, `vb0_size`, `vb1` … | The draw's vertex buffer addresses, byte sizes |
| `vertices`, `instances`, `first` | The draw's counts |
| `cc`, `sf`, `blend`, `ccs`, `rss` | The current `Target`'s pass state (`SHARED`; `rss` is the target's surface state) |
| `surfaces::bt` | This draw's binding table (built on the spot by `Recorder::draw`: the target, then textures), in the ring slot's 4 KB area |
| `samplers` | The `Pipeline`'s sampler state (put in the code heap when the pipeline is built, if it samples) |
| `fence`, `surfaces`, `bt_null` (`surfaces::bt_null`) | The `Context` itself |

`surfaces::x` is x's offset in the binding table pool (binding table pointers are offsets within the pool); the rest are addresses. When `Context::pipeline` builds templates
it checks that every hole has someone to fill it, otherwise it errors: an unfilled hole becomes address 0, the GPU page faults and the queue is banned.

The `Context`'s own names are only set in `Context::new` and never change afterwards: the context commands are filled at build time, and holes in a pipeline that use them are resolved to numbers when the pipeline is built
(`Fill::Value`), so each frame no longer looks them up by name.

To add a parameter only known at draw time: in `Cmds::draw` make the corresponding field take `A::sym("name")`, in `Context::pipeline`'s `fill` recognize the name as a kind of
`Fill`, and in `Recorder::draw_at`'s `value` fill the number according to that `Fill`.

## Submission and synchronization

One binding table per draw: `Recorder` only records it; `submit`, after waiting for this ring slot, writes it into the slot's area in the pool (4 KB per slot), then submits.

`submit` appends PIPE_CONTROL (flush) and MI_BATCH_BUFFER_END after the dwords and writes them into the next slot of the ring (if that slot's previous frame is not done, it waits for it first); after the CPU writes,
`H::fence()`, then submit. One fence per slot: `wait(n)` waits for it and checks `Queue::fault()`, `export_fence(n)` exports a sync_file for the compositor to use as an acquire fence.

## Measuring

`cargo run --features xe3 --example frames [N]`: results in [verification.md](verification.md).

## Indexed drawing

Each pipeline compiles two draw templates: a sequential one, and one with `3DSTATE_INDEX_BUFFER` / RANDOM access (`Pipeline::indexed`). `Recorder::draw_indexed`
fills the holes for the index buffer's address and size, format (`IndexFormat` ⇒ `INDEX_WORD` / `INDEX_DWORD`), first index, and base vertex (32-bit two's complement);
the fields are set after iris (the index buffer in `iris_upload_render_state`, 3DPRIMITIVE in `iris_draw_vbo`).

## Depth

`Context::depth_buffer(w, h)`: depth32float, Tile4 (one tile is 128 bytes × 32 rows; from Xe2 on depth surfaces can only be Tile4 / Tile64, isl), for GPU use only.
`Recorder::depth(&d)` (after `target`) makes subsequent draws use this depth attachment; a pipeline with `depth = {compare, write}` and a pass with a depth attachment must go together
(mismatch is an error, same as WebGPU). Templates of pipelines with depth have four more holes: address, pitch − 1, width − 1, height − 1. Clear depth with `clear_depth_pipeline`
(compare always, write depth) plus `clear_depth_vertices(z)`, the same draw as clearing color. No HiZ.

## Buffer bindings (uniform / storage)

`detour.load(offset) {binding = k}` / `detour.store` in shaders: LSC addresses by BTI (bits 31:24 of the extended descriptor are the table entry, Mesa
`gen_lsc_ex_desc_encode`), offset is in bytes. Every `Buffer` gets a RAW buffer surface at build time (filled as isl_buffer_fill_state_s does).
`Recorder::bind(&[..])` sets the buffers for subsequent draws: the binding table is the target, textures, null surfaces padded up to `BUFFER_BTI` (32), then the buffers;
the vertex shader of a pipeline that uses buffers also points to this table (pipelines without buffers still use `bt_null`). A pipeline's `buffers` is the larger of its two kernels'.

## mipmap

`Context::texture_levels(w, h, format, &[level0, level1, ..])`: tightly packed pixels per level, placed into isl's linear mip layout (`Texture::layout` gives the row pitch, row count,
and each level's position). Pipeline `mipmap = "nearest" / "linear"` sets mip filtering; sampling picks the level from derivatives.
