# 贡献

人和 agent 都按这一份. 先读 [AGENTS.md](AGENTS.md), 再按任务读 [.skills/](.skills/README.md).

## 验收: 什么算做完

- **数要对**. 不需要 GPU 的: 打包和 Mesa 逐位一致 (`crates/intel/dialect/tests/oracle.rs`), EU 编码和 gentool 逐位一致
  (`tests/eu.rs`). 需要 GPU 的: 在 Xe3 上跑, 读回的像素 / 数逐个核对. **跑起来 (没挂、没缺页) 不算成功**, 只有结果逐个对上才算.
- 提交后等到 fence 不代表跑完: 还要看 `Queue::fault()` (`Device::run`、`Context::wait` 已查).
- 性能数字写清楚条件: debug / release、分辨率、每帧画了什么.
- 汇报分开写「已验证 / 未验证 / 受环境限制」.

## 改动的规矩

- 新的命令、状态、寄存器从 genxml 来 (`scripts/intel-cmdgen.py` 重生成 `cmd.gen.rs`), 不手写打包. 生成的文件不手改.
- EU 编码改了要过 `tests/eu.rs`; 加新指令形式时在 `tests/eu.mlir` 加一条, 注释写 gentool 汇编, 用 `scripts/intel-eu-oracle.sh`
  重生成 `eu.expected` (要 gentool).
- 硬件行为照 Mesa (`~/coding/mesa`: `src/intel/compiler/brw`、`src/intel/vulkan` (anv)、`src/gallium/drivers/iris`、`src/intel/genxml`),
  在注释或 docs 里写出处 (文件 / 函数). 只读 Mesa, 不改它、不往它的上游提东西.
- 只为 Xe3 写. 别的 GPU 加新的 `crates/<厂商>/`, 不往 Intel 的里塞分支.
- 每帧走的路径 (运行时、`Template::emit`) 不跑 MLIR、不读属性、不分配大块内存.
- 新知道的硬件事实 (payload 布局、对齐要求、哪个状态必须发) 写进 [docs/xe3.md](docs/xe3.md) 的「踩过的坑」.

## 代码风格 (Rust)

- edition 2024, 工具链 1.96 (`rust-toolchain.toml`; mlirformat 0.6 起要的). 用引用、借用、切片; 生命周期用具名的 (`'d` 设备, `'a` IR), 不用 `'static` (const 除外).
- 不用 `pub mod`: 子模块私有, 父模块 `pub use` 平铺导出.
- 少写自由函数: 函数归到它主要操作的类型上 (唯一的可变参数、最大的结构体、生命周期最复杂的那个).
- 类型转换用 `From` / `TryFrom` / `Into`, 不写 `to_xxx` 这类手动转换 (要额外参数时例外).
- 重复的结构用宏 (`names!`、`attr_ops!`、`hosts!`、`xe3!`) 或代码生成 (cmdgen), 不复制粘贴.
- 单元测试只写有真实数据背书的 (Mesa / gentool 的对照、真机读回); 不确定的行为不写测试去钉住它.
- 注释写原因和出处, 不复述代码.

## 提交

- 英文, Conventional Commits (`feat(runtime): ...`、`fix(codegen): ...`、`docs: ...`), 每行不超过 72 字符; 正文写原因和验证.
- AI 参与的加 `Co-Authored-By`.
- 暂存只用 `git add <路径>`, 不带上别人的改动; 别人的改动要一起提交时先问.
- 只推 `origin` (`gitea.dopamine` 的 `AC/detour`; 不推 GitHub), 推之前用户同意. 不 force push, 不改写已推送的提交.
- acde-base 以 git 依赖引用本仓库 (见 AGENTS.md), 不拷进它的 `vendors/`; 改了 `PaintScene` / `Scene` / `Filter` 就是改了 acde 的接口.

## 测试机

本机 (`192.168.100.180`) 就是 Xe3. 另一台见 [docs/verification.md](docs/verification.md). GPU 卡住 / 缺页时的排查见
[.skills/detour-diag](.skills/detour-diag/SKILL.md).
