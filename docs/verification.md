# 开发和验证

- 本机 (`192.168.100.180`) 就是 Xe3 (8086:b0a0, xe, `/dev/dri/renderD128`, 用户在 render 组): 要 GPU 的直接跑.
  另一台测试机: Core Ultra 7 356H (Xe3, IP 30.01, 4 个 Xe core、每个 8 个 SIMD16 EU), AgentOS (ackernel 7.2, xe),
  `ac@192.168.33.139` (`ac.local`), `DETOUR_HOST=ac@192.168.33.139 scripts/on-host.sh '<命令>'` (rsync 到远端 `~/detour` 再执行);
  用户要在 `render` 组里, rust 来自 pacman, 没有 `~/.cargo/env`.
- `cargo test --workspace`: 前端 (三种着色器写法经 FRONTEND 得到同一个模块)、xe3 打包对照 Mesa、EU 编码对照 gentool、uAPI 结构体大小.
- 要 GPU 的: `cargo run --features xe3 --example triangle` (四种写法, 逐像素 + 管线统计), `--example frames` (运行时: 录制 / 吞吐 / 延迟, 逐像素), `--example store`, xe 的 `--example xe-info`.
  部署跑起来不算成功, 像素 / 读回的数逐个对上才算.
- mlirformat 的 release (opt-level 3) 编译很慢: 验证用 debug 构建就够.
- EU 编码的对照工具 gentool (Mesa `src/intel/compiler/gen`, `gentool asm|disasm -p ptl`) 要在 Linux 上从 Mesa 树编:
  `meson setup build -Dtools=intel -Dvulkan-drivers= -Dgallium-drivers= -Dplatforms= -Dglx=disabled -Degl=disabled -Dgles1=disabled
  -Dgles2=disabled -Dopengl=false -Dllvm=disabled -Dintel-elk=false`, `ninja -C build src/intel/compiler/gen/gentool`.

## 结果

| 内容 | 结果 |
| --- | --- |
| `crates/intel/xe` 结构体大小、ioctl 号 | `cargo test`, 和内核头文件一致 |
| 设备查询 (`xe-info`, 经 kmd trait) | 本机: device 0xb0a0, VA 48 位, 引擎 render / copy / compute + 视频 |
| genxml 打包 (`crates/intel/dialect/tests/oracle.rs`) | 290 个条目从 MLIR 解析、打包, 和 Mesa 的 C 打包逐位一致 |
| EU 编码 (`crates/intel/dialect/tests/eu.rs`) | 67 条 `xe3.eu` op 编码和 gentool `asm -p ptl` 逐位一致, 含 DPAS、LSC 2D block 和字节区域的解包 |
| 前端 (`crates/core/tests/frontend.rs`) | func / gpu / spirv 三种着色器写法经 FRONTEND 得到同一个模块 |
| 最小作业 (`store`) | 本机: xe3 方言写的 MI_STORE_DATA_IMM 经 layout / Emit / kmd 提交, 读回 0xacd0c0de, 约 1 ms |
| 对 Mesa (`benches/against_vulkan.rs`、`against_opengl.rs`) | 本机 (release, 2026-11-12, 桌面会话开着): 三种负载各 API 画同一份 1920x1080 RGBA8, 每个 API 的像素 / 读回的数逐个核对 (tri 207 万像素 0 错; fill 全部 [254, 126, 65, 255], 和逐步 unorm8 模型相同; dispatch 每个字对). tri (清屏 + 半屏三角形, 背靠背 4 帧在飞): detour 1230 帧/s, anv optimal 10950, anv linear 1340, iris 10560; 提交到完成 p50 0.99 / 0.18 / 0.90 / 0.15 ms. fill (32 层全屏预乘混合, 一次实例化 draw): 21.2 / 3.1 / 22.5 / 3.2 ms 一帧. 同为线性目标时 detour 和 anv 持平 (fill 还快 6 %), 差距是 anv / iris 的 tiled + 压缩目标和快速清屏 (detour 的目标只有线性). dispatch (16 通道一组, 每个之后刷写): 起初 16 / 4096 通道 detour 每个 11–24 µs, 几乎全是 CPU 录制 (`Recorder::dispatch` 每次建 `Cmds` 再按属性 `XE3.template` 打包); 改成 `Context::kernel` 时把三种 walker 打包成带洞的模板 (`Kernel` / `Walk`) 后录制 ~0.1 µs 一个, 每个 0.9–1.9 µs, anv 2.9–4.1 µs (录制 0.4–0.9 µs)、iris 2.2 µs; 100 万通道 detour 57 µs, anv 138 (linear 129), iris 144. `dispatch_cost` (不计录制): 16 / 4096 通道 0.8 µs 一个 (原 2.6), 100 万 61.5 µs. 用到 dispatch 的 16 个示例 (compute、indirect、storage、async_compute、atomics、workgroups、copy_engine、events、control、math64、flatten、tiles、stripgen、rowcoarse、layercoarse、pipeline) 照旧逐位对 |
| 运行时 (`frames`) | 本机 (debug 构建): 2 个 pipeline 编译 5.7 ms (一次); 每帧 (清屏 + 三角形, 1920x1080, 533 dword) 录制 p50 17 µs / p99 33 µs; 背靠背 1000 帧 860 ms (约 1160 帧/s); 每帧提交到完成 p50 1.04 ms / p99 1.30 ms; 最后一帧 207 万像素 0 错; dma-buf、sync_file 导出 |
| varying (`varying`) | 本机: 顶点颜色红 / 绿 / 蓝经 varying 插值, 256x256 三角形内 32640 像素和重心混合比, 最大差 1 (unorm8 舍入), 0 错 (走运行时) |
| 实例化 (`instances`) | 本机: 一次 draw 16 个矩形 (单位四边形按顶点, 位置和颜色按实例), 256x256 逐像素 0 错 |
| 纹理采样 (`texture`) | 本机: 16x16 棋盘纹理最近点采样铺满 128x128 (每个纹素 8x8 像素), uv 走 varying, 16384 像素 0 错 |
| 混合 (`blend`) | 本机: 3 个半透明矩形 (预乘, alpha 0.5 / 0.25 / 0.75) 依次叠在不透明底色上, 128x128 逐像素和 `src + dst * (1 - a)` 比, 最大差 0.75 (unorm8), 0 错 |
| compute 和整数 (`compute`) | 本机: COMPUTE_WALKER 派发 4096 个工作组 (65536 次调用), LSC 读写缓冲; i32 加减乘、移位、位运算、有 / 无符号 min / max、比较选择、i1 扩展、i32 ⇄ f32 转换, 和 Rust 逐位比较: 20 种结果全部 0 错; 提交到完成约 1 ms |
| 模糊效果 (`frost`) | 本机: compute 上的 dual Kawase (7 个着色器, 自己算的 f32 双线性), 和按同一定义写的 Rust 模型逐字节比: 512x320 上 13 个效果依次叠加 (毛玻璃 1 到 5 层、小数 / 出界的矩形、半径超过半高、9x7 小区域、毛玻璃过颜色矩阵 (saturate ∘ brightness)、两个重叠、壁纸层上的一个; filter: 直通 / 预乘图层、有无模糊、带偏移的颜色矩阵、clip、出界; drop-shadow: 小数偏移、clip、出界) 0 错; 1920x1080 上 1200x700 面板 (4 层) 0 错. 每块面板的 GPU 时间 1.02 ms, 一帧一块提交到完成约 1.5 ms (见 [optimizations.md](optimizations.md) 第 7 节) |
| 着色器数值 (`math`) | 本机: 65536 组操作数 (全指数范围、非规格化数、相邻 / 相等数对、0 / ±∞ / NaN 全组合), a / b、sqrt、fma、floor、min / max、cmpf + select 和 Rust f32 逐位比较: 6 个种子全部 0 错 (除法 / 平方根是 inv / rsq 修正到正确舍入, 见 [optimizations.md](optimizations.md)) |
| 2D 条带 (`strips`) | 本机: 512x384, 圆、非零规则五角星 (三色线性渐变)、圆角矩形、细斜线、半透明叠加 (两色、四色带硬边的线性渐变, 起止不在 0 / 1 时 pad), 两层嵌套裁剪路径 (vello_common `ClipContext`) 里画渐变; vello_common 条带 ⇒ detour-core (raster) ⇒ Xe3 合成器 (1103 个实例). 和 GPU 的精确模型 (paint 图集纹素 × 颜色 × 覆盖率, 每层量化到 unorm8) 比: 96.6% 相等, 其余差 1, 全部在舍入分界点或 paint 纹素边上 (u 离纹素边 0.01 内, 插值是 f32); 和 vello_cpu f32 管线最多差 2 (逐层量化), u8 管线最多差 3. CPU 生成条带和实例 8 ms (debug), GPU 提交到完成 1.2 ms |
| 三角形 (`triangle`) | 本机: func / gpu / spirv / arith 四种写法都画对: 2016 个内部像素橙色、外部蓝色 (清屏 draw), 0 错; 管线统计 IA 6 顶点 2 图元、VS 6、裁剪 2 / 2、PS 6208 (清屏 4096 + 三角形 2112); 编译约 8 ms, 提交加等待约 2 ms |
