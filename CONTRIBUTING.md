# Contributing

People and agents both follow this one. Read [AGENTS.md](AGENTS.md) first, then [.skills/](.skills/README.md) according to the task.

## Acceptance: what counts as done

- **The numbers must be right**. Without a GPU: packing matches Mesa bit for bit (`crates/intel/dialect/tests/oracle.rs`), EU encoding matches gentool bit for bit
  (`tests/eu.rs`). With a GPU: run on Xe3, check the read-back pixels / numbers one by one. **Running (no hang, no page fault) is not success**; only results that match one by one count.
- Waiting on the fence after submission does not mean it ran to completion: also check `Queue::fault()` (`Device::run` and `Context::wait` already do).
- State the conditions of performance numbers: debug / release, resolution, what each frame draws.
- Report "verified / not verified / limited by the environment" separately.

## Rules for changes

- New commands, state and registers come from genxml (`scripts/intel-cmdgen.py` regenerates `cmd.gen.rs`); packing is not written by hand. Generated files are not edited by hand.
- A change to EU encoding must pass `tests/eu.rs`; when adding a new instruction form, add a line to `tests/eu.mlir` with the gentool assembly in a comment, and regenerate `eu.expected` with `scripts/intel-eu-oracle.sh`
  (needs gentool).
- Hardware behavior follows Mesa (`~/coding/mesa`: `src/intel/compiler/brw`, `src/intel/vulkan` (anv), `src/gallium/drivers/iris`, `src/intel/genxml`);
  write the source (file / function) in a comment or in docs. Only read Mesa; do not change it or send anything to its upstream.
- Write for Xe3 only. Other GPUs get a new `crates/<vendor>/`; do not add branches into the Intel ones.
- The per-frame path (runtime, `Template::emit`) does not run MLIR, does not read attributes, does not allocate large blocks of memory.
- Newly learned hardware facts (payload layout, alignment requirements, which state must be emitted) go into "Pitfalls" in [docs/xe3.md](docs/xe3.md).

## Code style (Rust)

- edition 2024, toolchain 1.96 (`rust-toolchain.toml`; required from mlirformat 0.6 on). Use references, borrows, slices; use named lifetimes (`'d` device, `'a` IR), not `'static` (except for const).
- No `pub mod`: submodules are private, the parent module flattens them with `pub use`.
- Write few free functions: a function belongs to the type it mainly operates on (the only mutable parameter, the largest struct, the one with the most complex lifetime).
- Type conversions use `From` / `TryFrom` / `Into`, not hand-written conversions like `to_xxx` (except when extra parameters are needed).
- Repeated structure uses macros (`names!`, `attr_ops!`, `hosts!`, `xe3!`) or code generation (cmdgen), not copy and paste.
- Write unit tests only when backed by real data (Mesa / gentool oracles, read-back on real hardware); do not write tests that pin down uncertain behavior.
- Comments give reasons and sources; they do not restate the code.

## Commits

- English, Conventional Commits (`feat(runtime): ...`, `fix(codegen): ...`, `docs: ...`), at most 72 characters per line; the body gives the reason and the verification.
- Add `Co-Authored-By` when AI took part.
- Stage only with `git add <path>`, without picking up other people's changes; ask first when other people's changes must go into the same commit.
- Push only to `origin` (`AC/detour` on `gitea.dopamine`; do not push to GitHub), after the user agrees. No force push, no rewriting pushed commits.
- acde-base references this repository as a git dependency (see AGENTS.md), not copied into its `vendors/`; changing `PaintScene` / `Scene` / `Filter` changes acde's interface.

## Test machines

This machine (`192.168.100.180`) is the Xe3. For the other one see [docs/verification.md](docs/verification.md). For troubleshooting a stuck GPU / page fault see
[.skills/detour-diag](.skills/detour-diag/SKILL.md).
