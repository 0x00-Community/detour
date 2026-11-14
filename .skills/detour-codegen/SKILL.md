---
name: detour-codegen
description: 动 detour 编译器：前端 pass（spirv / gpu ⇒ func、清屏展开）、着色器下降 shader.rs（TypeConverter 1:N、#[patterns] 的 #[convert]、虚拟 xe3.eu 的约定、payload、EOT 消息）、寄存器分配、SWSB 的管线和距离、3D 状态和命令 Cmds（render.rs）、xe3-pad / xe3-end 收尾、xe3-layout 定址。加着色器能力、改 draw 状态、改 pass 时读。
---

# 编译器

管线: `FRONTEND` (`detour-core`) 然后 `BACKEND` (`detour-intel-codegen`), 见 `docs/architecture.md`. mlirpass 的写法照
`~/coding/calcc/src/lower.rs` (同样是 mlirpass ⇒ 自定义 ISA). 宏展开要 crate 根上的 `pub(crate) use mlirpass::{...}`.

## 前端 (`crates/core/src/compiler*`)

- 着色器的统一形式: `func.func` 带 `detour.stage`; vertex 的参数是顶点属性 (按 location), 结果是位置; fragment 的结果是 RT0 颜色.
- `detour-from-spirv`: 输入是 spirv 的通用形式 (mlirformat 不认 spirv 的自定义形式). 接口变量 ⇒ 参数 / 结果, 算术用 `#[patterns]` + `#[map]`.
- `detour-from-gpu`: `gpu.func` ⇒ `func.func`.
- `detour-expand-clear`: `load = "clear"` ⇒ pass 开头一个清屏 draw (全屏三角形 + 常量颜色的 fragment shader).
- 新 op 写在 `dialect.rs`; 有 `sym_name` 的要在 `op_infos` 里标 inherent, 否则自己建的 op 打印成通用形式.
- 测试 `tests/frontend.rs`: 三种写法得到同一个模块.

## 着色器下降 (`shader.rs`)

- 一个 SIMD16 寄存器是 `!xe3.grf<f32>` (`Grf("f32").into()`); `ShaderTypes::converter` 把 `vector<Nxf32>` 拆成 N 个 (1:N).
- 加一个一对一的 arith / math op: 在 `ShaderToXe3` 上加一行 `#[map("arith.addf" => "xe3.eu.add", via = Self::lanes(&[]))]`
  (额外属性写在 `lanes` 的参数里, 如 `&[("neg1", A::Bool(true))]`). 要逻辑的才写 `#[convert("op")]`, 用
  `Self::eu(cr, before, "xe3.eu.add", &operands, attrs, "f32")` 造虚拟指令, `cr.replace(op, vec![values])` 交出结果.
  只收能逐位精确的 (EU 的 add / mul / mad / neg 是精确的; divf 拒收). AMD (`Self::valu(cr, before, Opcode::VAddU32, (&operands, &[Src0, Src1]), &consts, vop3, ty)` / `#[map(.. => "rdna2.isa.valu", via = Self::lanes(Opcode::.., &[Src0, Src1], &[]))]`) 和
  Apple (`Self::usc(cr, before, UscOp::Fadd, &operands, &[S0, S1], |b| b.neg(S1), ty)` / `#[map(.. => "agx.usc.fadd", via = Self::lanes(UscOp::Fadd, &[S0, S1], |b| b))]`) 同样写法, 见 `docs/architecture.md`「IR 的写法」.
- 虚拟指令: SSA 操作数按顺序填属性没给的源槽; 结果是 dst; `neg0` / `neg1` 取负; send 的操作数是消息 payload, `src0_len` 个进 src0.
- `xe3.eu.payload {grf}` (方言里声明的伪 op, `rw.emit(&EuPayload::new(..))` 造): 线程开始就在寄存器里的值 (vertex 属性从 r2 起每个 4 个寄存器, URB 句柄 r1).
- epilogue: vertex ⇒ URB 写 (句柄 + VUE 头 4 dword + 位置), fragment ⇒ RT 写 (sendc, 4 个通道), 都带 EOT.
- 片段着色器的 payload 和插值怎么做, 见 `docs/xe3.md` 的「Xe2 / Xe3 片段着色器」(照 Mesa brw).

## 寄存器分配 (`regalloc.rs`)

所有改动经 `Rewriter` (`Alloc` 持有它): 插 mov 用 `set_before` + `create`, 改写用 `modify`, 删伪 op 用 `erase_op`. 分配前后两种 IR 形态见 `docs/architecture.md`.

单块 SSA 线性扫描. EOT 消息的 payload 预着色到 r112 起 (只给它用的值直接生成在那里, 其余插 mov). 临时寄存器从 payload 之后到 r111.
非 EOT send: 单寄存器 payload 就用值本身, 多寄存器的拷进一段连续寄存器; 都在 send 之后释放 (结果先分配, 不和 payload 重叠).
多块控制流还没做.

## SWSB (`swsb.rs`)

按 dst 类型分管线 (float / int / long / math); 读写依赖 (RAW、WAW, 以及**跨管线的 WAR**) 算该管线上的距离 (最多 7); 多个管线用
`A@最小`. 非 EOT send 取 SBID (32 个轮流, 还在途中的先等); 写它读的寄存器等 `$n.src`, 碰它写的等 `$n.dst`. RegDist 和 SBID
只在 Mesa 允许的组合下合并在一条指令上 (`inferred`), 其余放 sync.nop. 规则和踩过的坑见 `docs/optimizations.md` 第 2 节.

## 3D 状态和命令 (`render.rs`)

- `Cmds`: 只产出 op (batch / 代码堆状态 / surface 池状态), 不碰 Body; `Render` 经 `Rewriter` 把它们放进模块. `context()`、`null_surface()`、`pass_state()`、`draw()`、
  `statistics()`、`pc_flush()`. 模块下降 (`Render`) 和运行时共用它.
- 命令顺序照 iris 的第一批 + 每个 draw (旧的手写 triangle, git `b9af1c0`). 改状态前先在 Mesa iris / anv 里找同一个字段怎么设.
- 所有 state base address 是 0; 指针是 GPU 地址, binding table 指针是 pool 内偏移 (`A::within("surfaces", ...)`).

## 收尾和注册

`xe3-pad` / `xe3-end` 是 `rewrites!` 写的幂等模式 (`PadKernel` / `EndBatch`), `Rewrites::apply` 跑. 新 pass 加进 `lib.rs` 的
`detour_core::pass_set!(Xe3Passes: ..)` 和 `BACKEND` 字符串.

## 定址 (`layout.rs`, 引擎在 detour-core `image.rs`)

memory 从 1 MB 起 64 KB 对齐; 项按 `align`; 符号 ⇒ 数. `Xe3.contents` (`ImageTarget`) 给出一项的字节 (stream / kernel / data / state).
