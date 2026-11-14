# detour

我们自己的 GPU 用户态驱动, 全部用 Rust: 不经过 Mesa、Vulkan、OpenGL、libdrm, 命令、状态、着色器机器码都由 detour 生成, 经内核
GPU 驱动的 ioctl 直接提交. 以后 acde 的每一个窗口 (网页图层的光栅化、网页里的 WebGPU) 都走它. 只为 Intel Xe3 (Panther Lake,
PCI 8086:b0a0, 内核驱动 xe) 一款优化, 实现参考 Mesa 的 iris / anv. 第二个厂商 Apple M1 (Asahi Linux, 内核驱动 asahi) 在做,
见 [docs/roadmap.md](docs/roadmap.md).

整条链路是 MLIR ([mlirformat](https://github.com/Jabberwocky238/mlirformat) / [mlirpass](https://github.com/Jabberwocky238/mlirpass)):

```
detour.* (资源、pipeline、render pass) + 着色器 (func / spirv / gpu)
  ⇒ 前端 (着色器统一、清屏展开、canonicalize / cse)
  ⇒ Xe3 后端 (image.kernel / xe3.cmd / xe3.state, 寄存器分配, 软件记分板, 定址)
  ⇒ 镜像 ⇒ kmd trait (xe ioctl) ⇒ GPU
```

运行时把 pipeline 编译一次, 之后每帧只拷模板、填地址 (约 17 µs 一帧, debug 构建).

## 快速开始

```sh
cargo run --features cli,xe3,agx-g13,rdna2 -- devices    # 调试工具: 本机有哪些 GPU、detour 支持到哪一步 (-v 打印驱动报的参数)
cargo test --workspace                                   # 不用 GPU: 对照 Mesa 的打包、对照 gentool 的编码、前端
cargo run --features xe3 --example triangle              # 要 Xe3: 四种着色器写法画三角形, 逐像素核对
cargo run --features xe3 --example frames                # 运行时: 1000 帧, 录制 / 吞吐 / 延迟, 逐像素核对
cargo run --features xe3 --example lower -- crates/intel/driver/examples/triangle/func.mlir   # 打印下降后的 MLIR
cargo run --features xe3 --example indexed               # 要 Xe3: 索引绘制 (u16 / u32、first index、base vertex), 逐像素
cargo run --features xe3 --example depth                 # 要 Xe3: 深度测试 (比较、写开关、清深度), 逐像素
cargo run --features xe3 --example uniform               # 要 Xe3: 渲染着色器读缓冲 (uniform), 逐像素
cargo run --features xe3 --example mipmap                # 要 Xe3: mipmap (由导数定级), 逐像素
cargo bench --features xe3 --bench against_vulkan        # 要 Xe3: 同样的帧 / dispatch 对 Mesa anv (optimal 和 linear 目标), 两边逐像素
cargo bench --features xe3 --bench against_opengl        # 要 Xe3: 同上对 Mesa iris (EGL surfaceless, GL 4.6 core)
cargo run -p triangle                                    # 要 Xe3: 窗口里画三角形 (读回经 softbuffer 上屏, 首帧核对像素)
cargo run -p triangle3d                                  # 要 Xe3: 窗口里转动的 3D 三角形
```

## 去哪里读

| 文件 | 内容 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 协作约定 (人和 agent 都先读) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 代码风格、验收、提交和推送 |
| [docs/architecture.md](docs/architecture.md) | 分层、crate 分工、和内核的分工 |
| [docs/xe3.md](docs/xe3.md) | xe3 方言、Xe3 硬件上的坑、从 Mesa 读出来的 Xe2+ 细节 |
| [docs/runtime.md](docs/runtime.md) | 运行时: Context / Pipeline / Target / Recorder, 模板和洞 |
| [docs/verification.md](docs/verification.md) | 怎么验证, 已验证的结果 |
| [docs/optimizations.md](docs/optimizations.md) | 优化记录: 精确着色器数值 (除法 / 平方根 / 非规格化数)、SWSB 和消息寄存器、URB、2D 合成、渐变、模糊效果 |
| [docs/roadmap.md](docs/roadmap.md) | M1 到 M6, 接入 acde |
| [.skills/](.skills/README.md) | 按任务的操作手册 |

私有仓库. 许可证 Apache-2.0 (`crates/intel/dialect/genxml/xe3.xml` 来自 Mesa, MIT, Intel).
