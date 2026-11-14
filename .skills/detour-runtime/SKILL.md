---
name: detour-runtime
description: 动 detour 运行时（crates/intel/driver/src/runtime.rs）：Context / Pipeline / Target / Buffer / Recorder 各管什么、堆和地址、模板的洞名约定和谁来填、加一个 draw 参数要改的三处、batch 环、fence 和 sync_file / dma-buf 导出、frames 示例怎么量。改每帧路径、接合成器、加资源类型时读。
---

# 运行时

细节和洞名表在 `docs/runtime.md`.

## 规矩

- ⛔ 每帧的路径 (`Context::record`、`Recorder::*`、`Template::emit`、`Context::submit`) 不跑 MLIR、不读属性、不建 BO.
- 建一次的东西 (pipeline、目标、缓冲) 放进 `Context::pipeline / target / buffer`, 它们可以慢.
- 漏填的洞会变成地址 0 ⇒ GPU 缺页、队列被封: 新的洞一定要在 `Context::pipeline` 的检查里认得.

## 加一个 draw 时才知道的参数

1. `crates/intel/codegen/src/render.rs` `Cmds::draw` (或 `DrawArgs`): 对应字段取 `A::sym("名字")`.
2. `runtime.rs` `Context::pipeline` 的 `fill`: 把这个名字认成一种 `Fill` (没认出的名字按 `Context` 自己的名字解析, 解析不了就报错).
3. `runtime.rs` `Recorder::draw_at` 的 `value`: 按 `Fill` 返回值.

模块下降那边 (`Render::draw`) 照旧给数.

## 加一种按目标的状态

在 `Cmds::pass_state` 里加 (返回的名字数组跟着加), `Context::target` 会把它放进堆并记地址; draw 模板里的名字加进 `SHARED`.

## 提交、等待、导出

`submit(dw)` ⇒ 环里下一个槽 (满了先等), 返回槽号; `wait(n)` 等并查 fault; `export_fence(n)` ⇒ sync_file; `Target::export()` ⇒ dma-buf
(linear, 一行 `width * 4` 字节). CPU 写完 WC 内存后 `H::fence()` 再提交 (已经做了).

## 量

`cargo run --features xe3 --example frames [N]`: 录制 p50 / p99、背靠背吞吐、逐帧延迟、逐像素、导出. 改了每帧路径就重跑, 结果更新到
`docs/verification.md`.
