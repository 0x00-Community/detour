---
name: detour-build-test
description: 编译和验证 detour：cargo build / test（不用 GPU 的三组对照：Mesa 打包、gentool EU 编码、前端等价）、在本机 Xe3 上跑 triangle / frames / store / xe-info、重生成 cmd.gen.rs 和 eu.expected、debug 和 release 的取舍、验收标准（逐像素、fault 检查、性能数字写条件）。要编、要测、要交付前读。
---

# 编译和验证

## 编译

```sh
cargo build --workspace --all-targets      # 要求零警告
```

- 用 debug 构建验证就够. mlirformat 的 release (opt-level 3) 编译很慢 (几分钟以上), 只在量性能时用, 量的时候写明.
- 依赖只从 crates.io 来: `mlirformat`、`mlirpass` (同一份 mlirformat), `libc`.
- 工具链 1.96 (`rust-toolchain.toml` 钉住, `rust-version`): mlirformat (0.6 起) 用 `if let` 守卫, 1.91 编不过 (E0658).
- crates.io 的镜像可能落后 (aliyun 常常没有最新的 mlirformat): 换一个, 例如 `cargo --config 'source.crates-io.replace-with="ustc"' ...`.

## 不用 GPU 的验证

```sh
cargo test --workspace
```

| 测试 | 对照什么 |
| --- | --- |
| `crates/intel/dialect/tests/oracle.rs` | `oracle.mlir` 的 290 个条目打包后 == Mesa C 打包 (`oracle.expected`) |
| `crates/intel/dialect/tests/eu.rs` | `eu.mlir` 的 67 条编码 == `gentool asm -p ptl` (`eu.expected`) |
| `crates/core/tests/frontend.rs` | 三角形的 func / gpu / spirv 三种写法经 FRONTEND 得到同一个模块 |
| `crates/intel/xe`、`crates/apple/asahi`、`detour-core` (`drm.rs`) 的 `sizes_match_the_kernel` | uAPI 结构体大小、偏移、ioctl 号 == gcc 对内核头文件 |

| `crates/amd/codegen/tests/kernels.rs` | `ints` / `copy` 经 `rdna2-precheck`、桥接和 mlirpass 的 RDNA2 目标的整个镜像 (1 / 4 / 1024 个工作组) == 搬走前 detour 自己的 RDNA2 后端 (`tests/kernels/*.expected`) |

| `crates/apple/codegen/tests/ints.rs` | `compute/ints` 经桥接和 mlirpass 的 AGX 目标的整个镜像 (1 / 4 / 1024 个工作组) == 搬走前 detour 自己的 AGX 后端 (`tests/ints/g*.expected`) |


Apple 的整条 compute 链 (不上 GPU): `cargo run --features agx-g13 --example agx-compute -- FILE [GROUPS] --bin OUT`,
kernel 字节用 Mesa 的 `cd ~/coding/mesa/src/asahi/isa && PYTHONPATH=. python3 test/disasm.py OUT` 反汇编核对.

AGX 的对照 (结构体打包、CDM 流、USC 编码对 Mesa) 和它们的重生成 (`scripts/agx_*`、`tools/agxgen`) 在 mlirpass (`tests/agx.rs`).

AMD 的整条 compute 链 (不上 GPU): `cargo run --features rdna2 --example rdna2-compute -- FILE [GROUPS]` 打印整条下降之后的 IR 和 kernel 的位置;
加 `--hex` 只出 kernel 字节, 交给 `llvm-mc -arch=amdgcn -mcpu=gfx1036 -mattr=+wavefrontsize32 --disassemble` 核对.

AMD 的对照 (寄存器 / 包对 Mesa、ISA 对 llvm-mc、IB 对 Mesa 宏) 和它们的重生成 (`scripts/rdna2_*`、`tools/rdna2gen`) 在 mlirpass (`tests/rdna2.rs`).

aarch64 (Apple) 的部分在本机交叉编译, 测试用 qemu 跑:

```sh
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64-static -L /usr/aarch64-linux-gnu" \
  cargo test --features agx-g13 --target aarch64-unknown-linux-gnu
```

改了打包、编码、前端, 这几组都必须照旧通过.

## 要 GPU 的验证 (本机就是 Xe3)

```sh
cargo run --features xe3 --example triangle        # 四种写法, 每种: 管线统计 + 64x64 逐像素
cargo run --features xe3 --example frames [N]      # 运行时: 录制 / 吞吐 / 延迟, 1920x1080 逐像素, dma-buf / sync_file 导出
cargo run --features xe3 --example store           # xe3 方言写的 MI_STORE_DATA_IMM, 读回
cargo run --features xe3 --example frost           # 模糊效果 (compute), 逐字节对 Rust 模型; `frost K` 只跑第 K 个, `--mlir` 打印着色器
cargo run --features xe3 --example xe-info      # 设备查询
cargo run --features xe3 --example lower -- FILE [PIPELINE]   # 只看 MLIR, 不上 GPU
```

先确认没人在用 GPU (别的会话、跑着的示例). 结果要和 `docs/verification.md` 的表对得上; 数变了 (更好或更差) 就更新表.

## 重生成

- 布局表: `python3 scripts/intel-cmdgen.py crates/intel/dialect/genxml/xe3.xml crates/intel/dialect/src/cmd.gen.rs`, 然后跑 oracle 测试.
- EU 对照: 在 `tests/eu.mlir` 加一条 (上面一行注释写 gentool 汇编), `GENTOOL=<gentool 路径> scripts/intel-eu-oracle.sh`.
  gentool 的编法见 `docs/verification.md`.

## 验收

- ⛔ 跑起来不算: 像素、读回的数逐个对上才算. `Device::run` / `Context::wait` 已经查 `Queue::fault()`; 自己写的路径也要查.
- 性能数字写条件 (debug / release、分辨率、每帧画了什么、背靠背还是逐帧等).
- 汇报分开写: 已验证 (跑了什么、结果)、未验证、受环境限制.
