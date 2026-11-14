# detour skills

Before changing anything in this repository, read `AGENTS.md`, `README.md` and `CONTRIBUTING.md` in the root, then pick a skill below according to the task. `.claude/skills` is a symlink to this directory.

| skill | When to read |
| --- | --- |
| [detour-map](detour-map/SKILL.md) | First time in the repository; which crate handles what, which files an op passes through from MLIR to the GPU |
| [detour-build-test](detour-build-test/SKILL.md) | Building, running tests, running examples on Xe3; how to regenerate the oracles (Mesa packing, gentool encoding); acceptance criteria |
| [detour-xe3](detour-xe3/SKILL.md) | Changing the xe3 dialect: new commands / state (cmdgen), packing and templates, EU instructions and their assembly syntax |
| [detour-codegen](detour-codegen/SKILL.md) | Changing the compiler: shader lowering (new arith / vector ops), register allocation, SWSB, 3D state and commands (`Cmds`), addressing |
| [detour-runtime](detour-runtime/SKILL.md) | Changing the runtime: Context / Pipeline / Target / Recorder, template holes, adding draw parameters, submission and fences |
| [detour-mesa](detour-mesa/SKILL.md) | Need to know how to use the hardware: where to look in Mesa (iris / anv / brw / genxml), conclusions already read out |
| [detour-diag](detour-diag/SKILL.md) | Something went wrong: GPU fault / queue banned, wrong pixels, hangs; how to locate it |

General discipline (every skill follows it by default):

- ⛔ Deployed ≠ success. Running without hanging does not count; only read-back pixels / numbers that match one by one count; report verified / not verified separately.
- ⛔ The per-frame path does not run MLIR or read attributes: MLIR runs only when compiling / building a pipeline.
- ⛔ Packing and encoding are not written by hand, generated files are not edited by hand; the Mesa / gentool oracles must stay bit-for-bit identical.
- Optimize only for Xe3 (8086:b0a0); hardware details follow Mesa (`~/coding/mesa`, read-only), with the source written down.
- This machine (`192.168.100.180`) is the Xe3, GPU tests run directly; other machines need the user's consent.
- Push only to `origin`, after the user agrees; no force push.
