# 架构

## 分层

全程走 MLIR: 方言和文本格式用 [mlirformat](https://github.com/Jabberwocky238/mlirformat), 变换用
[mlirpass](https://github.com/Jabberwocky238/mlirpass) (都从 crates.io 取, 同一份 mlirformat). 每一层都是 MLIR, 都能打印出来看
(`cargo run --features xe3 --example lower -- FILE [PIPELINE]`).

```
detour.* (WebGPU 一次提交)     detour.buffer / texture / pipeline / render_pass { draw }; 着色器: func.func {detour.stage},
   │                           或 spirv.module (SPIR-V 反序列化出来的通用形式), 或 gpu.module
   │  detour-core (compiler)    detour-from-spirv, detour-from-gpu      着色器统一成 func.func {detour.stage} (arith / vector)
   │  (FRONTEND)               detour-expand-clear                    load = clear ⇒ 开头一个清屏 draw
   │                           func.func(canonicalize, cse)          上游的
   │  detour-intel-codegen      convert-detour-to-xe3                  着色器 ⇒ image.kernel (虚拟 xe3.eu); 资源、pass ⇒
   │  (BACKEND)                                                      image.memory / xe3.state.* / image.stream (xe3.cmd.*)
   │                           xe3-regalloc, xe3-swsb                寄存器, 软件记分板
   │                           xe3-pad, xe3-end                      kernel 后补零 (pad), batch 收尾 (MI_BATCH_BUFFER_END / MI_NOOP)
   │                           xe3-layout                            GPU 地址, 符号 ⇒ 数
   ▼
XE3_IMAGE (ImageEmit<Xe3>)    每个 image.memory 一段字节 + image.stream (batch) 的地址
   │  detour-intel (driver)     Device<G: Gpu, H: Host>: 每段一个 BO, VM_BIND, 提交, 等 fence, 查 fault
   ▼
detour-core (kmd trait)        Gpu (设备 / Bo / Vm / Queue / Fence), Host (CPU 缓存刷新 / 屏障)
   │  detour-intel-xe           impl Gpu for Xe: xe 的 ioctl (/dev/dri/renderD*)
   ▼
xe (内核)
```

| crate | 内容 |
| --- | --- |
| `crates/core` `detour-core` | 厂商无关的一个 crate, 三个私有模块平铺导出: |
| ↳ `src/kmd*` | `Gpu` (GAT 借用设备: `Bo<'d>` `Vm<'d>` `Queue<'d>` `Fence<'d>`; 队列跑什么由驱动定: `Gpu::Cmd`)、`Host` (`hosts!` 宏, x86-64 / aarch64), 厂商无关的 `Caching` / `MemType` / `Engine` / `DeviceInfo`; Linux DRM 各驱动共用的一层 (`drm.rs`): `Node` (render node、ioctl、GEM close / mmap / dma-buf)、`Syncobj` (所有驱动的 fence, `status()` 读 sync_file 的错误) |
| ↳ `src/image.rs` | `image` 方言 (`image.memory` / `stream` / `kernel` / `data` / `words` (可以带符号的 dword), 厂商无关的镜像容器; `Kernel::all` 找模块里的 kernel)、指令 op 宏 `attr_ops!`、op 行 `OpRows`、改写模式 `rewrites!` / `Rewrites::apply`、`ImageTarget` (厂商: 命令流收尾、kernel 编码、状态打包)、定址 `Layout`、`Image` / `Segment` / `ImageEmit` |
| ↳ `src/pack.rs` | 表驱动打包 `Layouts` (`Item` / `Field` / `Ty` / `Group`) 和模板 `Template` (符号当洞); 各家的表由各自的生成器产出 (Intel `XE3`; AMD 的 `RDNA2`、Apple 的 `AGX` 和 AGX 专用的 `Ty` 在 mlirpass) |
| ↳ `src/regalloc.rs` | 寄存器分配 `LinearScan` (一个块的 SSA 线性扫描: 多个寄存器堆、值占几个寄存器和对齐、预着色、连续段、源和结果分开); 厂商实现 `RegTarget` (Intel 的 `GrfFile`, AMD 的 `Files`; Apple 的在 mlirpass) |
| ↳ `src/raster.rs` | 2D 光栅化的 CPU 一半, 厂商无关: vello_common 0.3 的稀疏条带 ⇒ 四边形实例 (浮点: NDC 矩形、图集 uv、预乘颜色、paint 图集坐标) + R8 alpha 图集 + RGBA8 paint 图集 (渐变 LUT; `Strips`) |
| ↳ `src/scene.rs`, `src/filter.rs` | acde 的网页图层画进来的 2D 场景, 厂商无关、不碰 GPU: `PaintScene` (画的接口)、`Scene` / `RenderCommand` (录下来的命令)、`Paint` / `Glyph`、CSS 滤镜 `Filter` / `FilterEffect` / `ColorMatrix`; 接口形状来自 anyrender 0.14 |
| 根目录 `src/` `detour` | 给用的人 (acde) 的聚合导出 (canvas 3D 的 feature `webgl` / `webgl2` / `webgpu`, 缺省开) (仓库根就是这个包, git 依赖直接拿到它): 2D 场景总是导出; 一个 GPU 型号一个 feature (`xe3` Intel Panther Lake、`agx-g13` Apple M1、`rdna2` AMD), 只编开了的那家的 crate; `fallback-vulkan` 是 Vulkan 回退 (缺省开); `Backend::pick(Capability)` 定谁来做, 没驱动又没回退就 panic |
| `crates/fallback` `detour-fallback` | Vulkan 回退 (feature `fallback-vulkan`, 缺省开): 驱动做不了的由它在 Vulkan 上做 —— 场景由 vello on wgpu 画 (`Vello`)、图层合成 (`Composer`: 滤镜、投影、毛玻璃都是直通 alpha 上的颜色矩阵, 模糊按 `Blur` 规划 dual Kawase 的层数和偏移)、Linux 上的 Vulkan 设备 (`Device`; 按 `DeviceRequest` 分档开: WebGPU 特性全要 → 不要可选特性 → downlevel 限制) / dma-buf 画布 / sync_file、整页光栅 (`Raster`)、共享设备和 GPU 图片 (`Shared`、`GpuImage`、`ImageSet`、`Overrides`)、离屏画 canvas (`Offscreen`: 画、读回、写入、阴影模糊 `blur.wgsl`、CSS filter `matrix.wgsl`). 回退和 acde 的 acde-vello 是同一份 (2026-11-14 从 acde 搬回: vello 0.12、`Op` 批量合成、`Blur` 规划), 测试 `crates/fallback/tests/compose.rs`; 导出 `vello` / `wgpu` |
| ↳ `src/compiler*` | detour 方言、`Stage`、`Compiler` 上下文、`FRONTEND` 的三个 pass |
| `crates/intel/xe` `detour-intel-xe` | xe uAPI (`sys.rs` 私有), `Xe` |
| `crates/intel/dialect` `detour-intel-dialect` | xe3 方言: `xe3.cmd.*` / `xe3.state.*` / `xe3.reg.*` (cmdgen 从 genxml 生成, 表驱动打包 `XE3.pack`)、`xe3.eu.*` (`Inst` ⇄ op, `Inst::encode`)、容器 `image.memory` / `batch` / `kernel` / `data` |
| `crates/intel/codegen` `detour-intel-codegen` | `BACKEND` 的四个 pass、`Xe3` (`ImageTarget`)、`XE3_IMAGE`、`Codegen` |
| `crates/intel/driver` `detour-intel` | 条带合成器 `Compositor` (`raster.rs`, pipeline 在 `raster/strips.mlir`); 运行时 `Context` / `Pipeline` / `Target` / `Buffer` / `Recorder` (`runtime.rs`); 一次性的 `Device` / `Frame`; 示例 `frames`、`triangle`、`store`、`lower` |
| `scripts/intel-cmdgen.py` | `crates/intel/dialect/genxml/xe3.xml` ⇒ `crates/intel/dialect/src/cmd.gen.rs` |
| `crates/amd/codegen` `detour-amd-codegen` | 跑在 mlirpass 的 RDNA2 目标上 (feature `target.rdna2`; 2026-11-08 从 detour 搬过去: `rdna2` 方言 (寄存器表、PKT3、ISA 编码)、`convert-gpu-to-rdna2`、`rdna2-regalloc` / `-waitcnt` / `-pm4` / `-pad` / `-layout`、`Rdna2Dispatch`、`RDNA2_IMAGE`); 这里只剩 `rdna2-precheck` (precheck.rs: 只收 compute、一个线程的工作组、不带 binding / shared 的 1–4 dword 或单个 byte / short 的平坦访存, 和 `workgroup_id` / `lane` / `unpack4xu8`; 别的拒收, 不让它们被丢掉或编成别的)、detour-core 的桥接 `convert-detour-to-gpu`、`Codegen` (`image`: 一次派发进镜像)、示例 `rdna2-compute`; `tests/kernels.rs`: `ints` / `copy` 的镜像和搬走前 detour 自己的 RDNA2 后端逐字节一致 |
| `crates/amd/amdgpu` `detour-amd-amdgpu` | amdgpu uAPI (`sys.rs` 私有), `Amdgpu` (Radeon, 先做 Ryzen 9000 HX 的 Radeon 610M, GC 10.3.6); 提交照 radv (IB + BO 列表 + syncobj); 示例 `amdgpu-info` |
| `crates/apple/codegen` `detour-apple-codegen` | 跑在 mlirpass 的 AGX 目标上 (feature `target.agx`; 2026-11-08 从 detour 搬过去: `agx` 方言 (Mesa genxml 的结构体、AGX2 的 USC 指令)、`convert-gpu-to-agx`、`agx-regalloc` / `agx-wait` / `agx-end` / `agx-layout`、`AgxDispatch`、`AGX_IMAGE`); 这里只剩桥接 `convert-detour-to-gpu` (detour 的 compute 着色器 ⇒ `gpu.module @shaders` 里的 `gpu.func` kernel: `detour.workgroup_id` / `lane` ⇒ `gpu.block_id` / `thread_id` + `index_cast`, `detour.load` / `store` ⇒ `llvm.inttoptr` + `llvm.load` / `store`; 别的 stage、binding / shared / 非 4 字节的访存拒收)、`Codegen` (`image`: 一次派发进镜像)、示例 `agx-compute`; `tests/ints.rs`: 镜像和搬走前 detour 自己的 AGX 后端逐字节一致. `apple9` 模块: Apple M3 / M4 跑在 mlirpass 的 Apple9 目标上 (feature `target.apple9`): `apple9-check` (只收 compute、一或两个线程的工作组, 一个线程时写明 `known_block_size` 16) + 桥接 + `convert-gpu-to-apple9` / `apple9-regalloc` / `-control` / `-wait`, `Apple9::kernels` 出 `Apple9Code`; `tests/apple9.rs`: `ints` 的机器码, 设了 `APPLE9_METAL` / `APPLE9_DONOR` 就在 M4 上跑并逐个核对 |
| `crates/apple/metal` `detour-apple-metal` | Apple M3 / M4 在 macOS 上的运行时: `Apple9Code` 经 Metal 跑 (代码放进捐赠 kernel 的 binary archive 里替换 `_agc.main`, `failOnBinaryArchiveMiss` 建 pipeline; archive 由 build.rs 做), `M4::open` / `kernel` / `buffer` / `dispatch` (binding 0 内存、binding 1 参数和网格大小, 16 的工作组按 16·G 派发); 只在 macOS 上编; `tests/ints.rs` 在 M4 上逐个核对 |
| `crates/apple/asahi` `detour-apple-asahi` | asahi uAPI (`sys.rs` 私有), `Asahi` (Apple M1 / M2, Asahi Linux); 示例 `asahi-info` |

运行时 (每帧不跑 MLIR, 只填模板) 见 [runtime.md](runtime.md).

现状: 三角形 (清屏 + 一个 draw) 从 MLIR 到出图全程跑通, 着色器四种写法 (func / gpu / spirv / 带运算的 func) 在本机逐像素画对;
xe3.cmd 的最小作业 (`store`) 跑通. 计算管线 (compute stage) 前端认得, 后端还没做 (拒收).

## IR 的写法 (mlirformat / mlirpass 的标准用法)

三家后端同一套写法, 新加的东西照着来:

- **方言**: op 用 `#[mlir::op]` / `#[mlir::dialect]` 声明. 指令 op (`attr_ops!`) 有变长操作数、变长结果, format 是
  `` (`(` $operands^ `:` type($operands) `)`)? attr-dict (`->` type($results)^)? ``, 所以下面两种形态都是同一个 op 的合法写法,
  都能打印、原样解析回来. 每个方言的 `op_infos` 用 `OpRows::of` 声明 effects (都是 `Unknown`: 不能当死代码删) 和 inherent 属性.
- **两种 IR 形态**: 分配前是虚拟指令, SSA 操作数 + `slots` 属性 (core `Slots`: 每个操作数填哪个字段), 结果是寄存器类型的值;
  分配后操作数和结果清空, 寄存器按各家汇编语法写进 `slots` 说的属性里 (Intel gentool 串、AMD LLVM 写法 `v[4:7]`、Apple Mesa
  编码字典), `slots` 去掉.
  ```
  %5 = rdna2.isa.v_add_u32(%3, %4 : !rdna2.v<i32>, !rdna2.v<i32>) {slots = ["src0", "src1"]} -> !rdna2.v<i32>   分配前
  rdna2.isa.v_add_u32 {dst = "v11", src0 = "v11", src1 = "v0"}                                                 分配后
  ```
  AMD 的指令按编码家族分 op, 助记符是 `op` 属性, `slots` 是 `Slot` 枚举 (编码热路径不走类型化 op 的 serde: 构造用
  `Build`, 读取用 `Isa::read` ⇒ `Fields`):
  ```
  %5 = rdna2.isa.valu(%3, %4 : !rdna2.v<i32>, !rdna2.v<i32>) {op = "v_add_u32", slots = ["src0", "src1"]} -> !rdna2.v<i32>   分配前
  rdna2.isa.valu {op = "v_add_u32", dst = "v11", src0 = "v11", src1 = "v0"}                                                 分配后
  ```
  Intel 没有 `slots`: SSA 操作数按顺序填属性没给的源槽 (`src0` / `src1` / `src2`), 结果是 `dst`.
- **寄存器类型**: `!xe3.grf<..>` / `!rdna2.v<..>` `!rdna2.s<..>` / `!agx.r<..>` `!agx.u<..>` 是各方言的类型化包装 (`Grf`、
  `Rdna2Type`、`AgxType`, 底下是 core `RegType`), `From` / `TryFrom<&Type>` 转换, 不拼字符串.
- **伪 op**: 线程开始就在寄存器里的值 `xe3.eu.payload {grf}`、`rdna2.isa.input {reg}`、`agx.usc.input {uniform}` 在方言里声明,
  用类型化结构 `rw.emit(&EuPayload::new(..))` 造; 寄存器分配把它们定在那里, 然后删掉.
- **下降**: `#[patterns(converter)]` + `ShaderTypes::converter` (1:N 拆向量). 一对一的写成 `#[map("arith.addf" => "xe3.eu.add",
  via = Self::lanes(..))]` 一行一条 (模板 `lanes` 逐分量造指令); 只有要逻辑的才写 `#[convert]`.
- **改 IR**: 一律经 mlirpass `Rewriter` (`create` / `emit` / `set_before` / `modify` / `erase_op` / `move_before`), fn pass 里用
  `Rewriter::new(body, &mut Quiet)`; 收尾和补齐 (`xe3-end` / `xe3-pad` / `rdna2-pad` / `agx-end`) 是幂等的 `Pattern`
  (`rewrites!`) 经 `Rewrites::apply` 跑贪心驱动. 不直接 `body.create` / `insert` / `erase`.
- **pass 注册**: `detour_core::pass_set!` 分组 (`FrontendPasses`、`Xe3Passes`、`Rdna2Passes` + `Rdna2ImagePasses`、`AgxPasses` +
  `AgxImagePasses`), `passes.with(Xe3Passes::add)`; 每组和对应的管线字符串 (`FRONTEND` / `BACKEND`) 一一对应.
- **类型化 op (detour、image 方言)**: 每个 op 一个手写的 `#[mlir::op]` 结构体, 属性是有类型的字段 (serde 枚举、`Option<u32>`、
  `Vec<..>`、`bool` 是 unit 属性); 方言的 `INHERENT` 表列出每个 op 的属性, 单元测试 `inherent` 核对它和结构体字段一致.
  读用 `Heads::head::<O>(op)` (只看这一个 op: 不导出区域、不给整个 body 起名, ⛔ `body.operation(op).to_op()`), 改用
  `Heads::update` / `Updates::update` (经 `Rewriter`, listener 知道), 造无区域的用 `rw.emit(&O::new(..))`, 有区域的
  (`image.memory` / `stream` / `kernel`) 用 `OpData::head(&o)` + `create_with` 加区域, 或 `body.container(&o)`.
  ⛔ 按名字字符串取属性再 match、⛔ `OpData::build("image.data").with(..)`; 名字比较用 `O::NAME`, 符号名用 `Symbol::defined`.
  各厂商放在内存里的状态结构 (`xe3.state.*` 等) 的 `align` / `offset` 是可丢弃属性 (`Placed::ALIGN` / `OFFSET`).
- **属性构造**: `A` (`A::str` / `A::sym` / `A::within` / `A::dict`), `OpData::with_attrs(vec![..])` (`WithAttrs`).

## 和内核的分工

detour 在用户态, 和 Mesa 同一层, 只用内核已有的 uAPI (xe: VM_BIND、exec queue、syncobj、userptr). 快在自己的编译器和更短的
提交路径, 不在内核里. ackernel 不内置 detour 的任何东西 (补丁、config、模块、固件): 碰到 uAPI 的硬限制先告诉用户. 调优
(GT 频率、抢占和超时) 走 sysfs, 在测试机上手动设. 测试机的内核自带 xe.

## 接入内核 (asahi)

- Apple M1 / M2 (AGX G13 / G14) 跑 Asahi Linux, 内核驱动 asahi (uAPI 在主线 `include/uapi/drm/asahi_drm.h`, Linux 7.2 和 Mesa
  `include/drm-uapi` 的一样; 驱动本身在 Asahi 的内核分支). 结构体在 `crates/apple/asahi/src/sys.rs`, 测试核对 gcc 对头文件的
  `sizeof` / 偏移 / ioctl 号.
- 和 xe 不同, 队列跑的不是 batch, 是交给固件的命令: `Compute` (`drm_asahi_cmd_compute`: CDM 控制流的起止) 和 `Render`
  (`drm_asahi_cmd_render`: VDM 控制流和 tiler 状态). `Gpu::Cmd = [Command]`; 一次提交里每条命令等前面所有命令 (两个子队列的
  barrier 都填前面的条数), 和 xe 的一个 batch 一样按顺序.
- VM 的布局照 Mesa (`agx_device.c` `agx_open_device`): 顶上给内核 (至少 32 GiB), 底下 4 GiB 对齐的 USC 窗口 (着色器地址是相对
  `usc_exec_base` 的 32 位), 中间是用户的. 前 4 GiB 不映射, 空指针会缺页. 页 16 KiB.
- VM_BIND 是同步的 (没有 sync), `bind` 之后 CPU 直接 signal `done`. GPU 侧的缓存属性跟 BO (WRITEBACK 或 WC), 不跟映射.
- 出错: asahi 没有 xe 那种 ban 查询, Mesa (hk / gallium asahi) 也不查. `fault(done)` 读 fence 的 sync_file 状态 (负数是 errno);
  内核在 GPU 缺页时是否给 fence 置错, 还没在真机上确认.

## 接入内核 (xe)

- 直接打开 xe 驱动的 render node (`/dev/dri/renderD*`, 用 `DRM_IOCTL_VERSION` 认驱动; `detour-core` 的 `Node`), 不用 libdrm.
  结构体在 `crates/intel/xe/src/sys.rs`, 按 `include/uapi/drm/xe_drm.h` 手写 (`repr(C)`, 测试核对大小和 ioctl 号); drm.h 的
  (syncobj、GEM close、prime) 在 `detour-core` 的 `drm.rs`.
- 每个 GPUDevice 一个 VM 和一个 render 引擎的 exec queue: 一个上下文挂死只封掉它自己的 queue. VM 用 dma-fence 模式,
  提交用 syncobj 发信号 (交给合成器的 acquire fence 由它导出 sync_file).
- GPU 虚拟地址由 detour 自己分配, BO 绑定一次长期有效 (xe 没有重定位); 每次提交不再 bind.
- 核显没有显存, 都是系统内存. CPU 写、GPU 读的 (批处理、上传) 用 WC + PAT 0; CPU 读回的用 WB + PAT 1 (1-way 一致);
  渲染目标以后用压缩 PAT (9). PAT 表见 Mesa `XE2_PAT_ENTRIES` (Xe3 同 Xe2).
