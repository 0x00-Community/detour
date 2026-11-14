# detour skills

进本仓库改东西前先读根目录的 `AGENTS.md`、`README.md`、`CONTRIBUTING.md`, 再按任务挑下面的 skill. `.claude/skills` 是指向本目录的软链.

| skill | 什么时候读 |
| --- | --- |
| [detour-map](detour-map/SKILL.md) | 第一次进库; 查哪个 crate 管什么、一个 op 从 MLIR 到 GPU 经过哪些文件 |
| [detour-build-test](detour-build-test/SKILL.md) | 编译、跑测试、在 Xe3 上跑示例; 对照 (Mesa 打包、gentool 编码) 怎么重生成; 验收标准 |
| [detour-xe3](detour-xe3/SKILL.md) | 动 xe3 方言: 新命令 / 状态 (cmdgen)、打包和模板、EU 指令和它的汇编语法 |
| [detour-codegen](detour-codegen/SKILL.md) | 动编译器: 着色器下降 (新 arith / vector op)、寄存器分配、SWSB、3D 状态和命令 (`Cmds`)、定址 |
| [detour-runtime](detour-runtime/SKILL.md) | 动运行时: Context / Pipeline / Target / Recorder, 模板的洞, 加 draw 参数, 提交和 fence |
| [detour-mesa](detour-mesa/SKILL.md) | 要知道硬件怎么用: 去 Mesa 的哪里找 (iris / anv / brw / genxml), 已经读出来的结论 |
| [detour-diag](detour-diag/SKILL.md) | 出事了: GPU fault / 队列被封、像素不对、挂起; 怎么定位 |

通用纪律 (每个 skill 都默认遵守):

- ⛔ 部署 ≠ 成功. 跑起来不挂不算, 读回的像素 / 数逐个对上才算; 汇报分开写已验证 / 未验证.
- ⛔ 每帧的路径不跑 MLIR、不读属性: MLIR 只在编译 / 建 pipeline 时跑.
- ⛔ 打包、编码不手写, 生成的文件不手改; Mesa / gentool 的对照必须保持逐位一致.
- 只为 Xe3 (8086:b0a0) 优化; 硬件细节照 Mesa (`~/coding/mesa`, 只读), 写出处.
- 本机 (`192.168.100.180`) 就是 Xe3, GPU 测试直接跑; 别的机器要用户同意.
- 只推 `origin`, 推之前用户同意; 不 force push.
