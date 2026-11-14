# Optimization log

Each optimization states what was done, why it was done this way, how it was verified, and which edge cases are not covered. Hardware details in [xe3.md](xe3.md),
the summary table of verification results in [verification.md](verification.md).

General principle: **if it cannot be bit-exact, reject it**. Shaders only accept operations whose IEEE f32 exact result can be computed on the EU. Approximate instructions (inv / rsq /
sqrt) are not used directly as results; they are first corrected to correct rounding. Cases that cannot be corrected are rejected at compile time, never silently degraded. "Correct" for 2D compositing follows the same standard:
the GPU result must match a per-layer quantized exact model channel by channel, and the only allowed deviations are explainable rounding boundaries.

## 1. Shader numerics: how each operation is made exact

The lowering is in `crates/intel/codegen/src/shader.rs`; exactness is checked by the `math` example on real hardware, bit by bit against Rust's f32.

| MLIR op | EU instruction | Why it is exact |
| --- | --- | --- |
| `arith.addf` / `subf` / `mulf` / `negf` | `add` / `add -src1` / `mul` / `mov -x` | A single IEEE operation; negation only changes the sign bit |
| `math.fma` | `mad` (src0 + src1 * src2) | Measured on real hardware to be fused multiply-add: rounds only once, bit-identical to `f32::mul_add` |
| `math.absf` | `mov (abs)x` | Only changes the sign bit |
| `math.floor` | `rndd` | Rounding to an integer is itself exact |
| `arith.minnumf` / `maxnumf` | `sel.l` / `sel.ge` | A selection, no arithmetic. When one side is NaN, takes the other side (same as Rust `f32::min` / `max`). Which of `-0` and `+0` is taken is unspecified (both MLIR and Rust allow either) |
| `arith.cmpf` (oeq ogt oge olt ole une) | `cmp.<cond>` written to a register | i1 is held as an f32 register: true is all ones (a NaN), false is +0. Ordered predicates are false on NaN, `une` is true, matching cmp's condition code semantics |
| `arith.select` | `csel.z`, with the condition as src2 | When the condition is +0 (false) it picks the second value, otherwise (the all-ones NaN is not equal to 0) the first. No flag register, so no SWSB tracking of flag dependencies is needed |
| `arith.divf` | inv + 4 mad + scaling / fallback | See 1.2 |
| `math.sqrt` | rsq + 5 mad + scaling / fallback | See 1.3 |

Others (`minimumf`, `cmpf one/ord/uno/ueq...`, transcendental functions) fail to lower and are a compile error.

### 1.1 Preserving f32 denormals (kernel prologue)

- Symptom: at thread start the f32 denorm bit of cr0 is 0, i.e. **flush to zero**. Then add / mul / mad turn denormal inputs or results into 0, which is not IEEE.
  Measured: when the quotient falls in the denormal range (e.g. 2.4e-16 / 2.1e23 = 1.15e-39) the GPU returns 0.
- Approach: the first instruction of every kernel is `or(1) cr0.0 cr0.0 0x80 {NoMask}`, setting bit 7 (`BRW_CR0_FP32_DENORM_PRESERVE`),
  followed by a `sync.nop A@1` to wait for it to complete, because the EU does not track cr0 dependencies. Written after Mesa: brw `generate_float_controls_mode`,
  jay `set_cr0`. The code is `ieee_prologue` in `shader.rs`, inserted by `place_arguments` at the very start of every kernel, before the fragment shader's interpolation mad,
  so interpolation itself also preserves denormals.
- `xe3-swsb` keeps hand-written SWSB (the `A@1` of the sync in the prologue), provided it has not computed a dependency itself.
- ⚠ **The sampler flushes to zero**: denormals in sampled rgba32float texels all become 0, regardless of cr0. Constants (`mov` immediates) written to an
  rgba32float render target do keep denormals, so the problem is in the sampler, not the write path. To pass denormals in through the sampler, they must be split into
  "normal number × 2^-100" and multiplied back in the shader (this is what the `math` example does).

### 1.2 Exact division `a / b`

From Xe-HPG on the EU has no f32 FDIV, only the approximate `math.inv` (error of a few ulp). It is corrected with the IA-64 Markstein sequence
(Cornea, Harrison, Tang, *Scientific Computing on Itanium-based Systems*). All mads are fused and round only once:

```text
y0 = inv(b)                 approximate reciprocal
e  = 1 - b * y0             mad: residual of the reciprocal
y1 = y0 + e * y0            mad: one Newton step, reciprocal to ~0.5 ulp
q0 = a * y1                 approximate quotient
r  = a - b * q0             mad: residual of the quotient, exactly representable (when no underflow)
q1 = q0 + r * y1            mad: correctly rounded quotient
```

Three edge cases (all handled with `csel`, no branches):

1. **Overflow to NaN**: when b is 0 or ∞, or an input is NaN, the sequence yields NaN (e.g. `1 - 0 * ∞`). Then the EU's own `a * inv(b)` is taken instead,
   which is exact in these cases (±∞, ±0, NaN). How it is detected: `q1 - q1` is +0 means q1 is finite, otherwise it is NaN (`finite_or`).
2. **Zero dividend loses its sign**: when a = -0, `q1 = -0 + (+0) = +0`. So when a equals 0, `a * inv(b)` is taken directly; it keeps the sign.
3. **Residual or reciprocal leaves the normal range**: when a < 2^-100 the residual `a - b*q0` (magnitude about a * 2^-24) is below denormal precision,
   and the correction loses bits. When b is denormal, `inv(b)` overflows to ∞. Approach: when |a| or |b| is less than 2^-100, **multiply both operands by 2^64**.
   This is exact (multiplying by a power of 2) and the quotient is unchanged, so there is no scaling back and no double rounding. Scaling may make one operand overflow (a small and b large,
   or b small and a large), but then the true quotient is already out of range (0 or ∞ respectively), and the sequence / fallback gives exactly that value.
   "|x| < bound" is decided by the sign of `add (abs)x, -bound`: the sign of a subtraction result is exact.

### 1.3 Exact square root `sqrt(a)`

The EU's `math.sqrt` / `math.rsq` are both approximations. The approach likewise follows the IA-64 sequence:

```text
y0 = rsq(a)
g0 = a * y0,  h0 = 0.5 * y0            g ≈ √a, h ≈ 1/(2√a)
d  = 0.5 - g0 * h0                     mad
g1 = g0 + g0 * d,  h1 = h0 + h0 * d    mad ×2: one Newton step
r  = a - g1 * g1                       mad: residual
g2 = g1 + r * h1                       mad: correctly rounded root
```

Edge cases:

- When a is 0, ∞ or negative the sequence yields NaN (`0 * ∞`), and the EU's `math.sqrt` is taken instead. It is exact at these points: ±0, ∞, NaN.
- When a < 2^-100 the residual underflows. Multiply by 2^64 first, and after computing the root multiply by 2^-32 (both powers of 2, exact). The result is at least 2^-75, a normal number, so multiplying back does not double-round.

### 1.4 Verification: the `math` example

- 65536 operand groups (a, b, c, d), one pixel per group. The operands are the product of texels from two rgba32float textures, to avoid the sampler's flush to zero.
  Each pipeline writes four results per pixel to an rgba32float target, compared **bit by bit** against Rust f32. Any NaN counts as equal;
  both ±0 results of min / max are accepted.
- How the operands are built: random normal numbers over the full exponent range (±126), denormals (appearing alone in the a, b, c positions), equal pairs,
  pairs differing by 1 ulp, integers and half-integers (to test floor), numbers with exponents within ±2 (quotients and roots near 1), plus two rows of all combinations of special values:
  0, -0, ±1, ±∞, NaN, 0.5, 3, 1e-30, 1e30.
- Result: 6 random seeds, 65536 groups each; a/b, sqrt, fma, floor, min, max and both selects all have **0 errors**.
- Comparison before and after the fixes (all measured on real hardware):

  | Version | Exponent ±60 | Exponent ±126 | With denormal operands |
  | --- | --- | --- | --- |
  | Sequence only | Division only gets the sign of -0 wrong (21 cases) | Division wrong ~3%, sqrt wrong ~1% (residual underflow, denorm flushed to zero) | All wrongly 0 |
  | + zero-dividend fallback + cr0 denorm preserve | 0 errors | Division ~50, sqrt ~50 (operands < 2^-100) | Dividing by a denorm gives ∞ (~400 cases) |
  | + 2^64 scaling (dividend, root) | 0 errors | 0 errors | Dividing by a denorm still gives ∞ |
  | + scaling when the divisor is below 2^-100 too (final) | 0 errors | 0 errors | 0 errors |

### 1.5 Integers and compute (R1)

Integer arithmetic in compute kernels is checked bit by bit against Rust by the `compute` example: 65536 invocations (4096 workgroups), operands are random words, small numbers, equal pairs,
and all combinations of 0 / ±1 / i32::MIN / MAX; all 20 kinds of results have 0 errors.

| MLIR op | EU instruction | Notes |
| --- | --- | --- |
| `arith.addi` / `subi` | `add` / `add -src1` | Two's complement, wrapping |
| `arith.muli` | two `mul` (32 × 16) + `shl` + `add` | ⚠ From Xe-HP on the EU has no 32 × 32 integer multiply (Mesa `has_integer_dword_mul = false`). Writing `mul :d :d :d` directly gives wrong results in odd channels; the computed address goes out of bounds, the engine reports a CAT error and then hangs. After Mesa `brw_lower_integer_multiplication`: `a * b.lo16 + (a * b.hi16) << 16`, with the half-words taken through the regions `r.0<16;8,2>:uw` / `r.2<16;8,2>:uw` (writing `<2;1,0>` is also wrong) |
| `shli` / `shrui` / `shrsi` | `shl` / `shr` (:ud) / `asr` | |
| `andi` / `ori` / `xori` | `and` / `or` / `xor` (:ud) | |
| `minsi` / `maxsi` / `minui` / `maxui` | `sel.l` / `sel.ge` (:d / :ud) | |
| `cmpi` | `cmp`, destination of the same type as the sources | ⚠ `cmp` cannot mix integer sources with a float destination: written as `:f`, the destination register is not written at all. i1 is still stored in an f32 register; cmp just writes it as :d / :ud (all ones or 0) |
| `select` (i32) | `(x & c) \| (y & !c)` | Bitwise select, not through the float csel, so any bit pattern (including NaN-like ones) is preserved as is |
| `extui` / `extsi` (i1) | `and 1` / `mov` | The mask is all ones: extsi is just -1 |
| `sitofp` / `uitofp` / `fptosi` / `fptoui` | `mov` with type conversion | Measured bit-identical to Rust `as` within range (round to nearest even, truncation) |
| `bitcast` | `mov`, both sides as :ud | |

- How a register is read is decided by the instruction's attributes, without changing the value's type (`regalloc.rs` `Retype`): `unsigned` reads i32 as :ud, `as = "ud"` reads
  all operands as the same type, `dst_as` changes only the destination, `half1` takes one half-word of each dword of source 1.
- **Dispatch** (`render.rs` `compute`, after iris `iris_emit_compute_walker`): from Xe2 on PIPELINE_SELECT is not used; the render batch directly emits
  CFE_STATE + COMPUTE_WALKER. One workgroup is one SIMD16 thread (16 invocations); local id x is generated by the hardware into r1 (16 uw),
  inline data (8 dwords, the kernel's arguments) is in r2, the workgroup id is in r0.1. A thread ends with an EOT sent to the message gateway
  (Mesa `emit_cs_terminate`, from Xe-HPG on). After the walker comes a PIPE_CONTROL with untyped dataport / HDC flush and CS stall,
  so the next command sees what it wrote.
- **Memory access**: LSC untyped global memory, flat surface, **32-bit addresses** (A32). This machine's GPU virtual addresses are only 30 bits (`xe-info`'s
  va_bits), all buffers are below 4 GB, so neither A64 nor surface state is needed; an address is just an i32. One address per invocation; each read or write covers
  1 to 4 consecutive dwords, or 1 / 2 bytes (zero-extended).
- CFE_STATE's thread count is 320: this machine has 4 DSS × 8 EUs × 10 threads (iris uses max_cs_threads × subslice_total).

## 2. Scheduling and registers: software scoreboard (`xe3-swsb`) and message registers (`xe3-regalloc`)

- **Waiting on several SBIDs**: an instruction depends on the results of two out-of-order sends at once (e.g. multiplying the results of two samples), but an instruction's SWSB field can hold only one
  SBID; a send sets its own new SBID and cannot wait on another either. After Mesa `brw_lower_scoreboard`: the instruction keeps one wait, and the rest (plus all waits a send
  needs) go on `sync.nop $n.dst` / `$n.src` before it, one per token.
- **Message register reuse** (prompted by frosted glass, 2026-10-29): previously the payloads of non-EOT sends were allocated downward from r112 and never reused, so a kernel with
  32 or more reads/writes ran out of registers. Now a single-register payload (an address) is the value itself, and a multi-register one is copied into a contiguous block; both are freed after the send;
  later instructions that write them wait on `$n.src`. ⚠ A send's result must not overlap its own payload: allocate the result first, then free the payload.
- Three scoreboard bugs exposed by register reuse (all showed up as a few random wrong pixels, often wrong odd channels or whole patches of 0 / 255; running a kernel alone was also wrong):
  1. Waiting on `$n.src` removed the token from pending, so later instructions reading this send's result no longer waited on `$n.dst`. Now `.src` only clears its read set,
     and only `.dst` removes it.
  2. Write-after-read (WAR) across pipes was not tracked: the int pipe's `mov r48 ← r50` had not read yet when the float pipe's `mov r50 ← ...` wrote. Now each register
     records the last instruction on each pipe that read it, and writing it adds a regdist for readers on other pipes (the same pipe is in order, so not needed).
  3. When RegDist and SBID are on the same instruction, the encoding carries no pipe number (Xe2+: `0b01` / `0b10` are the pipe inferred from the source type + `.dst` / `.src`,
     `0b11` is All + `.dst`, Mesa `gen_swsb_encode`). They are only combined when the RegDist's pipe is the inferred one (float / int), or it is All and waits on
     `.dst` (Mesa `baked_ordered_dependency_mode`); otherwise the token wait goes into a sync.nop. The RegDist of a send that sets a token can only
     be on float / int / All; other pipes are changed to All.
- There are 32 SBIDs, used in rotation: before setting a token that is still in flight, wait on it first (`sync.nop $t.dst`).
- Cost: one sync.nop per extra token, inserted only when there really are several dependencies.

## 3. VS URB entry size

- Problem: the VF first writes the fetched vertices into the VS's URB entry (one vec4 per input), then the VS writes the VUE (header, position, varyings) into the same entry.
  The entry size (`3DSTATE_URB_ALLOC_VS`, in units of 64 bytes, minus 1) defaults to 0, which only fits 4 vec4s. With the paint attributes added, strips have
  5 inputs; each vertex overflows into the adjacent entry, and other vertices / instances read wrong attributes. The hardware reports no error, it just draws wrong (only about 6% of channels correct).
- Approach: entry units = ceil(max(inputs, 2 + varyings) / 4). Entry count = 5888 / units, rounded down to a multiple of 8,
  i.e. the same URB split into fewer, larger entries. The code is in `render.rs` `draw`.
- Debugging method (can be reused when attributes get scrambled): temporarily replace the fragment shader's output with some intermediate value (e.g. the interpolated coordinates), turn off blending,
  and read it back pixel by pixel. The conclusion this time was that attributes leaked between instances, not that the shader computed wrong.

## 4. 2D compositing: linear gradients and clipping (committed)

- **Instances are all floats**: the CPU precomputes the NDC rectangle, alpha atlas uv, premultiplied color and paint coordinates; the shader does not unpack integers or read uniforms.
  Solid segments take coverage from a column in the alpha atlas that is all 255, sharing one shader with alpha segments, with no branch.
- **Paint atlas**: RGBA8, 4096 texels per row. Row 0 is all white; a solid-color instance's color is in color and its paint samples white. A gradient instance's color is 1,
  and its paint samples the LUT in its row. All paints share one pipeline and one draw.
- **The LUT uses vello_common's f32 LUT**: `dispatch!(Level::new(), simd => g.f32_lut(simd))`, generated at the same SIMD
  level as vello_cpu, so whether mul_add is fused is the same on both sides. The LUT is rounded to unorm8 and put in the atlas.
- **A linear gradient's t is an affine function of pixel position**: u is computed at the quad's corners and interpolated by the hardware to pixel centers. This matches vello_cpu's use of
  `PIXEL_CENTER_OFFSET 0.5` to take pixel centers. The shader does no gradient arithmetic at all.
- **Clipping**: vello_common's `ClipContext` already intersects with the clip path when generating strips, so nothing changes on the GPU side.
- Result: in the `strips` scene, compared with the exact model (texel × color × coverage, quantized per layer), 96.6% of channels are equal and the rest all differ by 1, and all of them fall on
  rounding boundaries or paint texel edges. Compared with the vello_cpu f32 pipeline, the difference is at most 2.

## 5. Radial / sweep gradients (in progress, not yet committed)

Sections 1 to 3 were groundwork for these. The design:

- A radial gradient's t is not an affine function of position, so vertex interpolation no longer works. Instead interpolate the **transformed position** (gx, gy) (which is still affine),
  and compute t per pixel in the fragment shader after vello_cpu `fine/common/gradient`:
  - Linear: `t = gx`.
  - Radial (concentric, `RadialKind::Radial`): `t = sqrt(gx² + gy²) * scale + bias`. Uses the exact sqrt.
  - Sweep: vello's `x_y_to_unit_angle`, i.e. a polynomial approximation of atan, plus quadrant selection and NaN clearing:
    `slope = min(|x|,|y|) / max(|x|,|y|)`, then three mads, then three selects for the quadrant, finally `(φ·2π − start) · inv_delta`.
    Uses exact division, abs, min/max, cmpf/select.
  - The three kinds are summed weighted by per-instance coefficients; a coefficient is 0 or 1 times a finite value, so the result is exact and no branch is needed.
    `RadialKind::Strip` / `Focal` (two-point conical) have undefined regions (NaN ⇒ transparent) and are rejected for now.
- Extend modes are also computed in the shader, following the operation order of vello `apply_extend`: pad is `clamp(t, 0, 1)`, repeat is `t − floor(t)`,
  reflect is `|(t−1) − 2·floor((t−1)·0.5) − 1|` then clamp. All are exact operations, summed with per-instance 0/1 weights.
- **LUT index changed to align bit for bit with vello**: vello takes entry `trunc(t·(n−1))` (n is the LUT length, 256 to 4096). The new approach puts the n entries
  into one atlas row as is (no longer resampled to 4096), and the shader computes `u = (floor(t·(n−1)) + 0.5) / 4096`, landing exactly on the texel center. The sampler's fixed-point coordinates
  cannot pick the wrong texel. The old approach resampled the n entries to 4096, and since 4095 is not divisible by n−1, it was one entry off from vello at some t.
- Instances need three gy coefficients, the kind coefficients and the extend weights, so the fragment shader's inputs exceed the current limit of 2 varyings. The URB entry size is already computed from the
  varying count (section 3), so the limit can be lifted; after lifting it, interpolation must be verified on real hardware.
- The model's judgment must be extended accordingly: t is computed per pixel in f64, and `t·(n−1)` within 0.01 of an integer counts as a texel edge. For sweep gradients on the diagonal |x| ≈ |y|,
  the polynomial jumps by about 1e-5 between the two quadrant branches; those pixels also count as edges.

## 6. Runtime (M1, details in [runtime.md](runtime.md))

- A pipeline is compiled only once; state and commands become templates with holes, and recording a frame is just filling holes and copying: 1920x1080 frame recording p50 about 9 µs.
- 4 batches form a ring, each slot with its own binding table area. The CPU writing the next frame does not have to wait for the GPU to finish reading the previous one: about 1180 fps, latency about 1.1 ms.
- Shared state (cc / sf / ccs / rss) is emitted once per render pass, not re-emitted on every draw.

## 7. Blur effects: dual Kawase on compute (`frost.rs`, the `frost` example)

- Follows the three effects of the acde compositor (acde-base `crates/shell/src/compositor/web.wgsl` + `compose.rs`): frosted glass (`backdrop-filter:
  blur() saturate() brightness()`, including the wallpaper layer `under` beneath native windows), CSS `filter` (blur then multiply by a color matrix), `drop-shadow`.
  Level count `clamp(ceil(log2(max(σ,1)/2)), 1, 5)`, 5 sample points for down and 8 for up, RGBA8 per level (quantized every pass, like the compositor's Rgba8Unorm).
  The definition (operation by operation) is in the module docs of `frost.rs`; the reference in the example is written separately from the definition and shares no code with the generator.
- **No sampler**: hardware bilinear weights are fixed-point and cannot match Rust bit for bit. Bilinear filtering is computed in the shader in f32; all values are f32 in byte units
  (0..255), and lerp is `fma(t, b - a, a)` (the EU's mad rounds once, matching Rust `mul_add`, verified by the `math` example).
- Shaders are MLIR generated by Rust (`Ssa`: one SSA value per line, `binary!` / `unary!` generate the operation methods), compiled once when `Frost` is built (7 of them:
  down, up, backdrop, filter, shadow, plus the compositing for `under` and the premultiply of straight-alpha layers). Constants are hexadecimal f32, so the bit patterns are fixed.
  One invocation per pixel, grid (ceil(w / 16), h); extra channels at the end of a row are clamped to the last column.
- Per-frame parameters: down / up go through inline data (8 dwords); the blending passes use one 128-byte parameter block per effect (one per ring slot, `Frost::frame`).
- **Performance** (this machine's Xe3, a 1200x700 panel on 1920x1080, 4 levels, GPU time per panel, a frame with 11 panels minus one with 1):
  | Version | Per panel | down0 | up0 | backdrop |
  | --- | --- | --- | --- | --- |
  | First version (4 gathers per sample point, per-pixel division) | 1.55 ms | 0.41 ms | 0.77 ms | 0.40 ms |
  | Sample points share fractions and row addresses, two adjacent texels in one 8-byte read, coordinate ratios and reciprocals computed on the CPU | 1.39 ms | 0.13 ms | 0.67 ms | 0.31 ms |
  | Unpack by byte (`detour.unpack4xu8`), lerp and weighted sums with fma, ÷12 changed to ×(1/12) | 1.02 ms | 0.11 ms | 0.46 ms | 0.27 ms |

  One frame with one panel takes about 1.5 ms from submit to completion. up0 is about 700 instructions per thread; the time is spent in the ALU (memory bandwidth is small).
  - Reading two adjacent texels at once: at the image edge the fraction is replaced by 0 or 1 (values in the x direction are integer bytes, `fma(1, b - a, a)` is exactly b), giving results
    bit-identical to clamping each one; for a 1-wide image the fraction is always 0 (the second texel read lies outside the image; 64 extra bytes are left at the end of scratch).
  - ⚠ Byte unpacking: `mov :f ← r.k<32;8,4>:ub` is only legal for k = 0 (a float destination requires the source region to keep the bit position of the lowest channel, checked by gentool);
    for k = 1..3, first `mov :d ← :ub`, then convert to f.
  - Measure time with back-to-back runs: if the CPU spends a few seconds checking between two runs, the GPU downclocks and the measured time more than doubles.
- Still possible: for 2x upsampling, one channel computes two adjacent pixels (sharing reads and unpacking, but for odd sizes the ratio is not 0.5, so two sets of shaders are needed); one dispatch per level of compositing
  (currently every pass is followed by a PIPE_CONTROL with CS stall).

## 8. Compile time: the cost of typed instruction ops

Only when building a pipeline / compiling, not on the per-frame path. How it is measured: release, `perf stat -e instructions:u`; put the same shader 1 time and 21 times in one module,
and the difference / 20 is the compile cost per shader (startup removed).

- After the AMD instruction dialect was changed to typed ops (`1de94ab`): the full flow for a single shader is 100.3 million instructions (before: 103.1 million; startup registers 1300 fewer ops); compiling each shader
  is 11.4 million (before: 9.8 million, +16%).
- A pitfall hit first: encoding and the passes after allocation each called `head::<Valu>()` once per instruction, going through the typed op's serde generic form, 1.7 times the cost before the change;
  deserializing the large serde-derived enum (`Opcode`, 1237 mnemonics) compares strings one by one. Changed to: construction uses `Build` to write `OpData` directly, reading uses
  `Isa::read` to read into `Fields` in a single pass, and mnemonics are looked up in a generated FNV-1a hash table. Typed ops are only used to parse and print modules.
- The remaining +16% is mostly in mlirpass 0.4.1's `Body::create` ⇒ `place()`: for every op built, each attribute is compared as a string linearly against that op's `inherent` list;
  typed ops have a dozen or so inherent attributes (the old generic ops had none). Two options, both requiring a change to mlirpass and a new release (the user's repository, needs the user's consent):
  1. Sort `OpInfo.inherent` and binary search (or look up by name once when building the op to get a bitmap);
  2. Give `OpData` already split into inherent / discardable a fast path that skips `place()` (like `push_op`, but going through the Rewriter / conversion listeners).
