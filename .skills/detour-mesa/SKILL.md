---
name: detour-mesa
description: "Finding Xe2/Xe3 hardware usage in Mesa (~/coding/mesa, read-only): how iris (OpenGL) and anv (Vulkan) emit state and commands, the brw compiler's payload / interpolation / messages, genxml layouts, the gentool assembler; plus the conclusions already read out and written into docs/xe3.md. Read when implementing a new hardware feature or unsure how to set some state bit."
---

# Finding hardware usage in Mesa

Intel has not published a PRM for Xe3. Register and command layouts, and how to use them, are all read from Mesa (`~/coding/mesa`, shallow clone, read-only, not built, nothing sent upstream).
Khronos specs are in `~/coding/OpenGL-Registry`, `~/coding/Vulkan-Docs`, `~/coding/Vulkan-Headers`.

## Where to look

| To learn | Look at |
| --- | --- |
| Fields of a command / state | `src/intel/genxml/xe3.xml` (and the xe2 / gen125 ... it imports); our `crates/intel/dialect/genxml/xe3.xml` is flattened |
| How OpenGL sets some state | `src/gallium/drivers/iris/iris_state.c` (`genX(...)`), first batch of commands `iris_init_render_context` |
| How Vulkan sets some state | `src/intel/vulkan/genX_gfx_state.c` (merging at draw time), `genX_pipeline.c`, `genX_cmd_draw.c`, `genX_cmd_buffer.c`, `genX_state.c` (first batch) |
| Shader payload, interpolation, messages | `src/intel/compiler/brw/`: `brw_thread_payload.cpp` (payload per stage), `brw_compile_fs.cpp` (`remap_attr_reg`), `brw_from_nir.cpp`, `brw_lower_logical_sends.cpp` (how URB / sampling / RT writes become sends) |
| What is done first at the NIR level | `src/intel/compiler/brw/brw_nir.c`, `src/compiler/nir/nir_lower_*.c` |
| Instruction encoding, assembly syntax | `src/intel/compiler/gen/` (gentool: `gentool asm|disasm -p ptl`), `gen_validate.cpp` (which instructions / operands are legal on which generation) |
| Clear, copy, blit | `src/intel/blorp/` |
| Surface format values | `src/intel/isl/isl_format_layout.csv`, `isl.h` |

`src/intel/compiler/jay` (the new Xe2 / Xe3 compiler) can also be read, but Mesa's rules say it must not be modified.

## Already read out (written in docs/xe3.md)

- PS SIMD16 payload (Xe2+): r0, r1, then 2 registers for each enabled kind of barycentric, then the attributes' plane parameters.
- Plane parameters: 12 bytes per scalar component (a0, a1-a0, a2-a0), 5 components per 64-byte register.
- Interpolation: no PLN, two MADs: `b.y * (a1-a0) + a0`, then `b.x * (a2-a0) + previous step`.
- URB write: LSC URB store, the address is the URB handle.
- Pipeline state: anv packs it when building the pipeline and ORs it bitwise into the dynamic part at draw time (our templates do the same).

When you read out a new conclusion, add it to `docs/xe3.md` with its source (file, function).
