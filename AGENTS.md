# detour

我们自己的 GPU 显示驱动用户态: 不经过 Mesa、Vulkan、OpenGL、libdrm, 命令缓冲、状态、着色器的机器码都由 detour 自己生成,
经内核 GPU 驱动的 ioctl 直接提交. 全部用 Rust. 目标是比 Mesa 强: 路径更短 (没有 GL / Vulkan 的状态机和驱动层的猜测,
每个命令、每个状态位都是我们定的), 性能更高.

用它的是 acde 的每一个窗口: 网页图层的光栅化 (现在是 vello → wgpu → Vulkan → Mesa) 和网页里的 WebGPU (`navigator.gpu`、
`<canvas>` 的 `webgpu` 上下文) 都要走 detour. 只为 Intel Xe3 (Panther Lake) 这一款优化, 实现参考 Mesa 的 iris (OpenGL) 和
anv (Vulkan) (`~/coding/mesa`). acde-base 现在的 WebGPU 走
wgpu → Vulkan → Mesa (ANV), 由 acde-dom 经 `GpuNative` 契约 (acde-dom `src/native.ts`) 调 crates/acde `webgpu/`;
detour 在同一个契约下面把 wgpu / Vulkan / Mesa 这一整段换掉. WebGPU 的 render 和 compute 管线都在这里.
canvas 3D (用户定, 2026-11-03, 推翻了之前的「不做 WebGL」): `<canvas>` 的 `webgl` / `webgl2` / `webgpu` 三种上下文都由
detour 实现 (驱动做不了的走 Vulkan 回退), 3D 一定要支持好. WebGL 是 GLSL ES 前端 + GL 的状态机语义映射到 detour 的管线.
能力范围 (用户定, 2026-11-03): 不只是 2D. **Vulkan 能做的 detour 都要能做** —— 完整的 3D 图形管线和 compute, 以 Vulkan 的
能力全集为准 (见 docs/roadmap.md「3D: Vulkan 的能力全集」); 2D 场景只是其中一个用户. 但不实现 Vulkan 的 API 本身: 能力对齐,
接口是我们自己的 (MLIR 的 `detour` 方言 + 运行时). 只管 GPU, NPU 不在这里 (另有项目, 不碰).

目的 (用户定, 2026-10-30): 用 mlirformat / mlirpass 这一套直接生成每一家 GPU 的算子二进制 (命令流和着色器机器码), 所有显卡
走同一个体系, 不是做兼容层. 各家的内核接口只是把生成的镜像交给内核的薄管道; 方言和 pass 是主体 (见 docs/roadmap.md
「多厂商: 同一套 MLIR」).

detour 是独立仓库, 别的仓库里不放它的拷贝 (用户定, 2026-11-02). acde-base 用 Rust crate 依赖它 (git 依赖
`gitea.dopamine` 的 `AC/detour`, 钉 rev; 用户定, 2026-11-03; 11-12 起不再推 GitHub), 不拷进 `vendors/`. 接口归 detour (用户定, 2026-11-03):

- acde 的网页图层只认 detour-core 的 2D 场景 (`PaintScene` / `Scene` / `Filter`, `crates/core/src/scene.rs` / `filter.rs`):
  blitz-paint 和 acde 的 crates/render 直接往它上面画, 中间没有 anyrender 这一层.
- 回退归 detour, 不归 acde: 有 Xe3 走 detour 自己的光栅, 没有 (虚拟机、别的 GPU、Windows / macOS 上的调试模式) 走
  `detour-fallback` (feature `fallback-vulkan`, 缺省开; vello on wgpu on Vulkan). 谁来做由 `detour::Backend::pick` 定, 不由 acde 选;
  关掉 `fallback-vulkan` 时, 驱动做不了的事直接 panic, 不悄悄降级.
  feature `compiler` (缺省开, 每个驱动都隐含它) 是 base 里的编译器、image、pack、regalloc、raster、kmd (MLIR、vello_common);
  关掉它 detour 只剩 2D 场景 / 滤镜 / 合成类型 (和开着的回退), 不带 mlirformat (及其 jemalloc), Windows / macOS 上 acde 这样依赖它. 回退的设备真 GPU 优先 (`Device::open`, 没有时才收 lavapipe); 页面另有 `Device::open_hardware` (只收真 GPU).
  没有真 GPU 时页面由 CPU 直接画 (`CpuRaster`, `src/cpu_raster.rs`, xe3 + Linux: vello_cpu 光栅、合成照 `compose.wgsl` 在 CPU 上做,
  直接写进映射的 dma-buf 画布; `PageRaster::cpu()`, 和 Vulkan / Xe3 并列的一种 `Painters`, 不是 Vulkan 的回退), 完全不开 Vulkan 设备
  (用户定 2026-11-14: lavapipe 导出 sync_file 时空指针崩, 虚拟机里拖窗口就 SIGSEGV). 模糊是三次方框模糊逼近高斯 (GPU 是 dual Kawase), 看起来一样, 字节不同.

acde-base 以前拷过一份 `vendors/detour` (2026-10-29 到 11-02, crates/paint `detour.rs`, 最后一版 `7b9181d`), 已删, 那份上的改动
搬回了这里. detour 不往别的仓库里提交.

## 厂商

一个厂商一个目录, crate 名带厂商: `crates/intel/*` 是 `detour-intel-*`, `scripts/intel-*`.
换平台 (例如 AMD 核显, 内核驱动 amdgpu) 就加 `crates/amd/` 一组, 不改 Intel 的. 和厂商无关的部分 (内核接口的 trait、
WebGPU 语义的 detour 方言、着色器前端) 放在 `crates/core` (一个 crate `detour-core`), 厂商的后端实现它们的接口 (kmd 的 trait, 方言的下降).

现在支持 Intel Xe3 核显 (Panther Lake, 例如 Core Ultra 7 356H 的 Intel Graphics, PCI 8086:b0a0, 内核驱动 xe). 第二个厂商
Apple M1 (AGX G13, Asahi Linux, 内核驱动 asahi) 在做: `crates/apple/*`, 进度见 [docs/roadmap.md](docs/roadmap.md) 的 A 阶段
(Apple M3 / M4 的 compute 在 `detour-apple-codegen::apple9`, mlirpass 的 Apple9 目标, 本机 M4 上验过);
参考 Mesa 的 asahi (`src/asahi`, 只读; 那个目录不接受 AI 写的改动, 我们只读不提交). 别的 GPU 不管.

## 先读什么

进本仓库改东西之前: 本文件 → [README.md](README.md) → [CONTRIBUTING.md](CONTRIBUTING.md) → 按任务挑 [.skills/](.skills/README.md)
(`.claude/skills` 是指向它的软链). 内容写在 [docs/](docs/): 架构 [architecture.md](docs/architecture.md)、方言和硬件
[xe3.md](docs/xe3.md)、运行时 [runtime.md](docs/runtime.md)、验证 [verification.md](docs/verification.md)、优化记录 [optimizations.md](docs/optimizations.md)、路线 [roadmap.md](docs/roadmap.md).
说明只放在这几处, 子目录不放 README / AGENTS.md; 改了行为就改对应的文档.

## 硬规矩

- ⛔ 部署 ≠ 成功: 像素 / 读回的数逐个对上才算 (见 CONTRIBUTING).
- ⛔ 每帧的路径不跑 MLIR: MLIR 只在建 pipeline / 编译时跑.
- ⛔ 打包、编码不手写: genxml 生成, Mesa / gentool 逐位对照.
- 只为 Xe3 优化; 硬件细节照 Mesa (`~/coding/mesa`, 只读) 并写出处.
- 只推 origin, 推之前用户同意.
