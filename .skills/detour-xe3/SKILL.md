---
name: detour-xe3
description: "Working on the xe3 dialect: xe3.cmd / xe3.state / xe3.reg ops generated from genxml (attribute naming rules, struct fields, arrays, variable-length group elements, automatic dword_length), regenerating with cmdgen, XE3.pack and XE3.template (symbols as holes), EU instructions xe3.eu.* (operand strings in gentool syntax, Inst ⇄ op, encoding comparison), the image container and symbol references in detour-core. Read when adding commands/state/instructions or changing the packing encoding."
---

# xe3 dialect

Details are in `docs/xe3.md`; this is the procedure.

## Using a genxml command / state

1. Search for the name in `crates/intel/dialect/src/cmd.gen.rs` (genxml name in lowercase with underscores, `3DSTATE_VS` ⇒ `xe3.cmd.3dstate_vs`), look at its fields
   (`f("name", bit, width, Ty, default, dims)`) and variable-length groups (`group`).
2. Write the op: `xe3!("cmd.3dstate_vs", kernel_start_pointer = A::sym("vs"), enable = true, ...)` (in codegen) or as MLIR text
   `xe3.cmd.3dstate_vs {enable, kernel_start_pointer = @vs}`.
   - The field name is the attribute name; names starting with a digit get a `_` prefix (`_3d_command_sub_opcode`).
   - Struct fields are written as dictionaries, fixed-length arrays as arrays, variable-length groups as `elements = [{field = ...}, ...]`.
   - `dword_length` is computed automatically if omitted. Fields beyond the declared length only make the command longer when they have nonzero bits.
   - Address fields are written as symbols: `@x` (address), `@m::@x` (offset within m), `[@x, n]` (address plus n).
3. Enum values (`SURFTYPE_2D`, `_3DPRIM_TRILIST` ...) are constants at the end of `cmd.gen.rs`; surface formats are not in genxml, they are in `SurfaceFormat` in codegen `render.rs` (isl values).

Entries not in genxml: do not hand-write the packing. First check whether Mesa's `src/intel/genxml` has it; if so, update `crates/intel/dialect/genxml/xe3.xml` (Mesa `genxml_import.py --flatten`) and rerun cmdgen.

## Packing and templates

- `XE3.pack(op, attrs) -> Vec<u32>`: no symbols allowed.
- `XE3.template(op, attrs) -> Template`: symbols become holes; `Template::emit(out, |name| value)` copies and fills the holes; `extend` concatenates.
- Comparison with Mesa: `cargo test --features xe3 --test oracle`. There is only one intentional difference from Mesa: alternative address fields starting in the same dword are ORed bitwise.

## EU instructions

- Physical form: `xe3.eu.<mnemonic> {exec = 16, dst = "r10<1;1,0>:f", src0 = "...", src1 = "0x3f800000:f", swsb = "F@1", ...}`,
  the full attribute table is at the start of `asm.rs`. Operands: `[-][(abs)]r<nr>[.<byte>]<v;w,h>:<type>`, `null`, `arf<nr>`, immediate `<bits>:<type>`.
- `Inst::op()` turns an instruction into an op, `Inst::try_from((name, attrs))` goes the other way, `Inst::encode()` produces 16 bytes.
- To add a new instruction form: confirm the encoding path in `eu.rs` (after Mesa `src/intel/compiler/gen`), add an entry with a gentool comment to `tests/eu.mlir`,
  regenerate `eu.expected` with `scripts/intel-eu-oracle.sh`, run `--test eu`.
- Message descriptors (`msg.rs`): `Lsc`, `Render`, `Sampler`, `SendInfo::lengths(desc, mlen, rlen)`; the `Sfid` enum.

## Container

The container is detour-core's `image` dialect (vendor-neutral): `image.memory @m {mem = "streaming" | "coherent", size?, align?} { ... }` is one BO; inside it go `image.stream @b {xe3.cmd...}` (Xe3's batch),
`image.kernel @k {xe3.eu...}`, `image.data @d {bytes = dense<...> | size}`, and `xe3.state.*` with `sym_name` / `align`.
`xe3-layout` assigns addresses and replaces symbols with numbers.
