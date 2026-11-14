---
name: detour-diag
description: detour 出事了怎么查：GPU fault / 队列被封（Error::Fault、ban）、作业超时、像素不对、命令被拒（打包报错、洞没人填）；看 journalctl -k 的缺页地址、在 batch 里打点读回、二分关阶段、用 lower 示例看每个 pass 之后的 MLIR、对照已知的坑。
---

# 出事了

## 队列被封 (`Error::Fault`)、缺页

1. `journalctl -k | grep -i xe | tail` 看缺页地址和引擎. 地址落在哪里: 对照 `lower` 打印的 `xe3-layout` 之后的 `addr` / `offset`,
   运行时对照 `Context` 的堆 (1 MB 起: 代码堆 1 MB、pool 64 KB、batch 环) 和各 BO 的地址.
2. 地址 0 或很小: 多半是**没填的洞 / 没解析的符号** (例如 binding table pool 基址、状态指针). 运行时的模板检查会报
   `nothing fills @x`; 模块路径看 `xe3-layout` 有没有漏掉的 SymbolRef.
3. 定位卡在哪条命令: 在 batch 里插 `xe3.cmd.mi_store_data_imm {address = [@x, n], immediate_data = ...}` 打点, 读回看到了哪个.
4. 二分: 关掉一个阶段 (例如 PS) 或删掉一段命令再跑.

## 像素不对

- 先用 `lower` 示例看每个 pass 之后的 MLIR: `cargo run --features xe3 --example lower -- FILE 'builtin.module(<到某个 pass 为止>)'`.
- 着色器: 看 `xe3-swsb` 之后的 `xe3.eu` (寄存器、regdist). 依赖少等 (swsb) 会读到旧值, 时好时坏.
- 读管线统计 (render pass 的 `statistics = @q`): IA / VS / 裁剪 / PS 次数对不对, 判断是没画还是画错.

## 已知的坑 (见 docs/xe3.md「踩过的坑」)

- binding table 必须在 pool 里; 每个阶段都要发 binding table 指针 (不绑东西的给 null 表).
- `STATE_SYSTEM_MEM_FENCE_ADDRESS` 要 4 KB 对齐.
- EOT 消息的源在 r112 ..= r127.
- fence 到了不代表跑完: 要查 `Queue::fault()`.
- 没有 `3DSTATE_PUSH_CONSTANT_ALLOC_PS` (Xe-HP 起).

## 打包 / 编译报错

- `no field x`: 属性名不对, 去 `cmd.gen.rs` 查字段名.
- `does not fit`: 值超出字段宽度, 或地址没对齐 (地址字段的低位必须是 0).
- `@x is not a number yet`: `pack` 遇到了符号, 应该先定址或用 `template`.
