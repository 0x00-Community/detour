# 优化记录

每一条优化都写清楚做了什么、为什么这样做、怎么验证、还有哪些边界没覆盖. 硬件细节见 [xe3.md](xe3.md),
验证结果的总表见 [verification.md](verification.md).

总原则: **做不到逐位精确就拒**. 着色器只接受在 EU 上能算出 IEEE f32 精确结果的操作. 近似指令 (inv / rsq /
sqrt) 不直接当结果用, 先修正到正确舍入. 修正不了的情况一律拒编译, 不静默降级. 2D 合成的「对」也按这个标准:
GPU 结果要和一个逐层量化的精确模型逐通道对上, 允许的偏差只有可解释的舍入分界点.

## 1. 着色器数值: 每个操作怎么做到精确

下降在 `crates/intel/codegen/src/shader.rs`, 精确性由 `math` 示例在真机上逐位对照 Rust 的 f32.

| MLIR 操作 | EU 指令 | 为什么精确 |
| --- | --- | --- |
| `arith.addf` / `subf` / `mulf` / `negf` | `add` / `add -src1` / `mul` / `mov -x` | 单次 IEEE 运算, 取反只改符号位 |
| `math.fma` | `mad` (src0 + src1 * src2) | 真机实测是融合乘加: 只舍入一次, 和 `f32::mul_add` 逐位一致 |
| `math.absf` | `mov (abs)x` | 只改符号位 |
| `math.floor` | `rndd` | 取整本身精确 |
| `arith.minnumf` / `maxnumf` | `sel.l` / `sel.ge` | 选择, 不做运算. 一边是 NaN 时取另一边 (同 Rust `f32::min` / `max`). `-0` 和 `+0` 取哪个不确定 (MLIR 和 Rust 都允许两种) |
| `arith.cmpf` (oeq ogt oge olt ole une) | `cmp.<cond>` 写进寄存器 | i1 当作一个 f32 寄存器: 真是全 1 (一个 NaN), 假是 +0. 有序谓词碰到 NaN 为假, `une` 为真, 和 cmp 的条件码语义一致 |
| `arith.select` | `csel.z`, 以条件为 src2 | 条件是 +0 (假) 时选第二个值, 其他 (全 1 的 NaN 不等于 0) 选第一个值. 不用 flag 寄存器, 不需要 SWSB 跟踪 flag 依赖 |
| `arith.divf` | inv + 4 次 mad + 缩放 / 兜底 | 见 1.2 |
| `math.sqrt` | rsq + 5 次 mad + 缩放 / 兜底 | 见 1.3 |

其他 (`minimumf`、`cmpf one/ord/uno/ueq...`、超越函数) 下降失败, 编译报错.

### 1.1 保留 f32 非规格化数 (kernel 序言)

- 现象: 线程启动时 cr0 的 f32 denorm 位是 0, 即**冲零**. 这时 add / mul / mad 碰到非规格化输入或结果都会变成 0, 不是 IEEE.
  实测: 商落在非规格化范围时 (例如 2.4e-16 / 2.1e23 = 1.15e-39) GPU 返回 0.
- 做法: 每个 kernel 的第一条指令是 `or(1) cr0.0 cr0.0 0x80 {NoMask}`, 置上 bit 7 (`BRW_CR0_FP32_DENORM_PRESERVE`),
  接一条 `sync.nop A@1` 等它完成, 因为 EU 不跟踪 cr0 的依赖. 写法照 Mesa: brw `generate_float_controls_mode`,
  jay `set_cr0`. 代码在 `shader.rs` `ieee_prologue`, 由 `place_arguments` 插在每个 kernel 的最前面, 在片段着色器插值的 mad 之前,
  所以插值本身也保留非规格化数.
- `xe3-swsb` 会保留手写的 SWSB (序言里 sync 的 `A@1`), 前提是它自己没算出依赖.
- ⚠ **采样器会冲零**: 采样返回的 rgba32float 纹素里的非规格化数都变成 0, 跟 cr0 无关. 常量 (`mov` 立即数) 写进
  rgba32float 渲染目标能保留非规格化数, 所以问题出在采样器, 不在写回路径. 要用采样器传进非规格化数, 只能拆成
  「规格化数 × 2^-100」, 在着色器里乘回来 (`math` 示例就是这么做的).

### 1.2 精确除法 `a / b`

EU 从 Xe-HPG 起没有 f32 FDIV, 只有近似的 `math.inv` (误差几个 ulp). 用 IA-64 的 Markstein 序列修正
(Cornea, Harrison, Tang, *Scientific Computing on Itanium-based Systems*). 所有 mad 都是融合的, 只舍入一次:

```text
y0 = inv(b)                 近似倒数
e  = 1 - b * y0             mad: 倒数的残差
y1 = y0 + e * y0            mad: 一步牛顿, 倒数到 ~0.5 ulp
q0 = a * y1                 近似商
r  = a - b * q0             mad: 商的残差, 精确可表示 (不下溢时)
q1 = q0 + r * y1            mad: 正确舍入的商
```

三处边界 (全部用 `csel` 处理, 不分支):

1. **溢出成 NaN**: b 是 0 或 ∞, 或输入有 NaN 时, 序列算出 NaN (例如 `1 - 0 * ∞`). 这时改取 EU 自己的 `a * inv(b)`,
   这些情况下它是精确的 (±∞、±0、NaN). 判断方法: `q1 - q1` 是 +0 说明 q1 有限, 否则是 NaN (`finite_or`).
2. **零被除数丢符号**: a = -0 时 `q1 = -0 + (+0) = +0`. 所以 a 等于 0 时直接取 `a * inv(b)`, 它保留了符号.
3. **残差或倒数离开规格化范围**: a < 2^-100 时残差 `a - b*q0` (量级约 a * 2^-24) 低于非规格化数的精度,
   修正会丢位. b 是非规格化数时 `inv(b)` 溢出成 ∞. 做法: |a| 或 |b| 小于 2^-100 时, **两个操作数同乘 2^64**.
   这是精确的 (乘 2 的幂), 商不变, 所以不用再缩放回来, 也没有二次舍入. 缩放可能让一个操作数溢出 (a 小 b 大、
   或 b 小 a 大), 但这时真正的商本来就超出范围 (分别是 0 或 ∞), 序列 / 兜底给的正是这个值.
   「|x| < 界」用 `add (abs)x, -界` 的符号判断: 减法结果的符号是精确的.

### 1.3 精确平方根 `sqrt(a)`

EU 的 `math.sqrt` / `math.rsq` 都是近似值. 做法同样照 IA-64 的序列:

```text
y0 = rsq(a)
g0 = a * y0,  h0 = 0.5 * y0            g ≈ √a, h ≈ 1/(2√a)
d  = 0.5 - g0 * h0                     mad
g1 = g0 + g0 * d,  h1 = h0 + h0 * d    mad ×2: 一步牛顿
r  = a - g1 * g1                       mad: 残差
g2 = g1 + r * h1                       mad: 正确舍入的根
```

边界:

- a 是 0、∞ 或负数时序列得 NaN (`0 * ∞`), 改取 EU 的 `math.sqrt`. 它在这些点上精确: ±0、∞、NaN.
- a < 2^-100 时残差下溢. 先乘 2^64, 算出根后乘 2^-32 (都是 2 的幂, 精确). 结果至少 2^-75, 是规格化数, 乘回去不会二次舍入.

### 1.4 验证: `math` 示例

- 65536 组操作数 (a, b, c, d), 每组一个像素. 两个 rgba32float 纹理的纹素相乘得到操作数, 以避开采样器冲零.
  每条管线每个像素写四个结果到 rgba32float 目标, 和 Rust f32 **逐位比较**. 任何 NaN 都算相等;
  min / max 的 ±0 两种结果都接受.
- 操作数的构成: 全指数范围 (±126) 的随机规格化数, 非规格化数 (单独出现在 a、b、c 位置), 相等的数对,
  只差 1 ulp 的数对, 整数和半整数 (测 floor), 指数在 ±2 内的数 (商和根接近 1), 再加两行特殊值的全组合:
  0、-0、±1、±∞、NaN、0.5、3、1e-30、1e30.
- 结果: 6 个随机种子, 每个 65536 组, a/b、sqrt、fma、floor、min、max、两种 select 全部 **0 错**.
- 修正前后的对比 (都在真机上测):

  | 版本 | 指数 ±60 | 指数 ±126 | 有非规格化操作数 |
  | --- | --- | --- | --- |
  | 只用序列 | 除法只错 -0 的符号 (21 个) | 除法错约 3%, sqrt 错约 1% (残差下溢、denorm 冲零) | 全错成 0 |
  | + 零被除数兜底 + cr0 保留 denorm | 0 错 | 除法约 50 个、sqrt 约 50 个 (操作数 < 2^-100) | 除以 denorm 得 ∞ (约 400 个) |
  | + 2^64 缩放 (被除数、根) | 0 错 | 0 错 | 除以 denorm 仍是 ∞ |
  | + 除数小于 2^-100 也缩放 (最终) | 0 错 | 0 错 | 0 错 |

### 1.5 整数和 compute (R1)

compute 内核的整数运算由 `compute` 示例逐位对照 Rust: 65536 次调用 (4096 个工作组), 操作数是随机字、小数、相等的数对、
0 / ±1 / i32::MIN / MAX 的全组合; 20 种结果全部 0 错.

| MLIR 操作 | EU 指令 | 说明 |
| --- | --- | --- |
| `arith.addi` / `subi` | `add` / `add -src1` | 补码, 回绕 |
| `arith.muli` | 两次 `mul` (32 × 16) + `shl` + `add` | ⚠ Xe-HP 起 EU 没有 32 × 32 整数乘 (Mesa `has_integer_dword_mul = false`). 直接写 `mul :d :d :d` 时奇数通道的结果是错的, 算出的地址越界, 引擎报 CAT error 后挂起. 照 Mesa `brw_lower_integer_multiplication`: `a * b.lo16 + (a * b.hi16) << 16`, 半字用区域 `r.0<16;8,2>:uw` / `r.2<16;8,2>:uw` 取 (写成 `<2;1,0>` 也不对) |
| `shli` / `shrui` / `shrsi` | `shl` / `shr` (:ud) / `asr` | |
| `andi` / `ori` / `xori` | `and` / `or` / `xor` (:ud) | |
| `minsi` / `maxsi` / `minui` / `maxui` | `sel.l` / `sel.ge` (:d / :ud) | |
| `cmpi` | `cmp`, 目的操作数和源同类型 | ⚠ `cmp` 不能混用整数源和浮点目的操作数: 写成 `:f` 时目的寄存器根本没被写. i1 仍用 f32 寄存器存, 只是 cmp 按 :d / :ud 写进去 (全 1 或 0) |
| `select` (i32) | `(x & c) \| (y & !c)` | 按位选, 不经过浮点 csel, 任何位模式 (包括像 NaN 的) 都原样保留 |
| `extui` / `extsi` (i1) | `and 1` / `mov` | 掩码是全 1: extsi 就是 -1 |
| `sitofp` / `uitofp` / `fptosi` / `fptoui` | `mov` 带类型转换 | 实测和 Rust `as` 在范围内逐位一致 (舍入到最近偶数、截断) |
| `bitcast` | `mov`, 两边都按 :ud | |

- 寄存器的读法由指令的属性决定, 不改值的类型 (`regalloc.rs` `Retype`): `unsigned` 把 i32 读成 :ud, `as = "ud"` 把
  所有操作数按同一类型读, `dst_as` 只改目的操作数, `half1` 取源 1 每个 dword 的一个半字.
- **派发** (`render.rs` `compute`, 照 iris `iris_emit_compute_walker`): Xe2 起不用 PIPELINE_SELECT, 渲染批次里直接发
  CFE_STATE + COMPUTE_WALKER. 一个工作组是一个 SIMD16 线程 (16 次调用), 局部 id x 由硬件生成到 r1 (16 个 uw),
  inline 数据 (8 个 dword, 内核的参数) 在 r2, 工作组号在 r0.1. 线程用发给消息网关 (message gateway) 的 EOT 结束
  (Mesa `emit_cs_terminate`, Xe-HPG 起). walker 之后一条带 untyped dataport / HDC flush 和 CS stall 的 PIPE_CONTROL,
  所以下一条命令能看到它写的东西.
- **访存**: LSC untyped global memory, flat 表面, **32 位地址** (A32). 本机 GPU 虚拟地址只有 30 位 (`xe-info` 的
  va_bits), 所有缓冲都在 4 GB 以下, 所以不需要 A64 和 surface state, 地址就是一个 i32. 每次调用一个地址, 一次读写
  1 到 4 个连续 dword, 或者 1 / 2 字节 (零扩展).
- CFE_STATE 的线程数取 320: 本机 4 个 DSS × 8 个 EU × 10 个线程 (iris 用 max_cs_threads × subslice_total).

## 2. 调度和寄存器: 软件记分板 (`xe3-swsb`) 和消息寄存器 (`xe3-regalloc`)

- **多个 SBID 的等待**: 一条指令同时依赖两个乱序 send 的结果 (例如两次采样的结果相乘), 而一条指令的 SWSB 字段只能写一个
  SBID; send 自己要设新的 SBID, 也不能再等别的. 照 Mesa `brw_lower_scoreboard`: 指令自己保留一个等待, 其余的 (以及 send
  需要的所有等待) 放到它前面的 `sync.nop $n.dst` / `$n.src` 上, 每个 token 一条.
- **消息寄存器复用** (毛玻璃引出, 2026-10-29): 以前非 EOT send 的 payload 从 r112 往下分配、永不复用, 一个 kernel 有
  32 次以上读写就 out of registers. 现在单寄存器的 payload (地址) 就是它的值本身, 多寄存器的拷进一段连续寄存器, 都在 send
  之后释放; 之后写它们的指令等 `$n.src`. ⚠ send 的结果不能和自己的 payload 重叠: 先给结果分配, 再释放 payload.
- 复用寄存器后暴露的三个记分板漏洞 (现象都是随机的少量像素错, 常见奇数通道错或整片 0 / 255; 单独跑某个 kernel 也会错):
  1. 等过 `$n.src` 就把 token 从 pending 里删掉了, 之后读这个 send 结果的指令不再等 `$n.dst`. 现在 `.src` 只清掉它的读集合,
     `.dst` 才删.
  2. 跨管线的写后读 (WAR) 没有跟踪: int 管线的 `mov r48 ← r50` 还没读, float 管线的 `mov r50 ← ...` 就写了. 现在每个寄存器
     记下每条管线上最后一个读它的指令, 写它时对别的管线的读者加 regdist (同一管线按序, 不用).
  3. RegDist 和 SBID 写在同一条指令上时编码里不带管线号 (Xe2+: `0b01` / `0b10` 是按源类型推断的管线 + `.dst` / `.src`,
     `0b11` 是 All + `.dst`, Mesa `gen_swsb_encode`). 只有 RegDist 的管线就是推断出来的那条 (float / int), 或是 All 且等
     `.dst` 时才合并 (Mesa `baked_ordered_dependency_mode`), 否则 token 的等待放进 sync.nop. 设 token 的 send 的 RegDist 只能
     在 float / int / All 上, 别的管线改成 All.
- SBID 有 32 个, 轮流用: 再次设一个还在途中的 token 前先等它 (`sync.nop $t.dst`).
- 代价: 每个多出来的 token 一条 sync.nop, 只在真有多个依赖时才插.

## 3. VS 的 URB 条目大小

- 问题: VF 先把取到的顶点写进 VS 的 URB 条目 (每个输入一个 vec4), VS 再往同一个条目里写 VUE (头、位置、varying).
  条目大小 (`3DSTATE_URB_ALLOC_VS`, 以 64 字节为单位, 减 1) 默认是 0, 只装得下 4 个 vec4. 条带加上 paint 属性后有
  5 个输入, 每个顶点溢出到相邻条目, 别的顶点 / 实例读到的属性是错的. 硬件不报任何错, 只是画错 (只有约 6% 的通道对).
- 做法: 条目单位数 = ceil(max(输入数, 2 + varying 数) / 4). 条目数 = 5888 / 单位数, 向下取 8 的倍数,
  即同样大的 URB 分成更少、更大的条目. 代码在 `render.rs` `draw`.
- 排查方法 (以后遇到属性错乱可以照做): 把片段着色器的输出临时换成某个中间量 (例如插值后的坐标), 并关掉混合,
  逐像素读出来看. 这次的结论是属性在实例之间串了位, 不是着色器算错.

## 4. 2D 合成: 线性渐变和裁剪 (已提交)

- **实例全是浮点**: CPU 把 NDC 矩形、alpha 图集 uv、预乘颜色、paint 坐标都算好, 着色器不解包整数、不读 uniform.
  实心段从 alpha 图集里一列全 255 的列取覆盖率, 和 alpha 段共用同一个着色器, 不需要分支.
- **paint 图集**: RGBA8, 每行 4096 个纹素. 第 0 行全白, 纯色实例的颜色在 color 里, paint 取白色. 渐变实例的 color 是 1,
  paint 取它那一行的 LUT. 所有 paint 共用一条管线、一次 draw.
- **LUT 用 vello_common 的 f32 LUT**: `dispatch!(Level::new(), simd => g.f32_lut(simd))`, 和 vello_cpu 在同一个 SIMD
  级别上生成, 所以 mul_add 是否融合两边一致. LUT 舍入到 unorm8 放进图集.
- **线性渐变的 t 是像素位置的仿射函数**: 在四边形的角上算好 u, 由硬件插值到像素中心. 这和 vello_cpu 用
  `PIXEL_CENTER_OFFSET 0.5` 取像素中心一致. 着色器里不用做任何渐变运算.
- **裁剪**: vello_common 的 `ClipContext` 生成条带时已经和裁剪路径求交, GPU 侧不用改.
- 结果: `strips` 场景里和精确模型 (纹素 × 颜色 × 覆盖率, 每层量化) 对比, 96.6% 的通道相等, 其余都差 1, 而且全部落在
  舍入分界点或 paint 纹素的边上. 和 vello_cpu f32 管线比最多差 2.

## 5. 径向 / 扫描渐变 (进行中, 尚未提交)

1 到 3 节是为它们准备的. 设计如下:

- 径向渐变的 t 不是位置的仿射函数, 不能再靠顶点插值. 改为插值**变换后的位置** (gx, gy) (它仍是仿射的),
  在片段着色器里照 vello_cpu `fine/common/gradient` 逐像素算 t:
  - 线性: `t = gx`.
  - 径向 (同心, `RadialKind::Radial`): `t = sqrt(gx² + gy²) * scale + bias`. 用到精确 sqrt.
  - 扫描: vello 的 `x_y_to_unit_angle`, 即 atan 的多项式近似, 加上象限选择和清除 NaN:
    `slope = min(|x|,|y|) / max(|x|,|y|)`, 然后三次 mad, 再用三次 select 选象限, 最后 `(φ·2π − start) · inv_delta`.
    用到精确除法、abs、min/max、cmpf/select.
  - 三种类型用逐实例的系数加权求和, 系数是 0 或 1 乘一个有限值, 结果精确, 不需要分支.
    `RadialKind::Strip` / `Focal` (两点锥形) 有未定义区域 (NaN ⇒ 透明), 先拒.
- 扩展模式 (extend) 也在着色器里照 vello `apply_extend` 的运算顺序算: pad 是 `clamp(t, 0, 1)`, repeat 是 `t − floor(t)`,
  reflect 是 `|(t−1) − 2·floor((t−1)·0.5) − 1|` 再 clamp. 都是精确运算, 按逐实例的 0/1 权重相加.
- **LUT 下标改成逐位对齐 vello**: vello 取第 `trunc(t·(n−1))` 项 (n 是 LUT 长度, 256 到 4096). 新做法是图集一行
  放 n 项原样 (不再重采样到 4096), 着色器算 `u = (floor(t·(n−1)) + 0.5) / 4096`, 正好落在纹素中心. 采样器的定点坐标
  不会选错纹素. 旧做法把 n 项重采样到 4096 项, 而 4095 不能被 n−1 整除, 所以在某些 t 上和 vello 差一项.
- 实例要加 gy 的三个系数、类型系数和扩展权重, 片段着色器的输入会超过现在 2 个 varying 的上限. URB 条目大小已经按
  varying 数计算 (第 3 节), 上限可以放开, 放开后要在真机上验证插值.
- 模型的判定也要相应扩展: t 在 f64 里逐像素算, `t·(n−1)` 离整数 0.01 以内算纹素边. 扫描渐变在 |x| ≈ |y| 的对角线上,
  多项式在两个象限分支之间有约 1e-5 的跳变, 这些像素也算边.

## 6. 运行时 (M1, 详见 [runtime.md](runtime.md))

- pipeline 只编译一次, 状态和命令做成带洞的模板, 录一帧只是填洞和拷贝: 1920x1080 录帧 p50 约 9 µs.
- 4 个 batch 组成环, 每个槽有自己的 binding table 区. CPU 写下一帧时不必等 GPU 读完上一帧: 约 1180 fps, 延迟约 1.1 ms.
- 共享状态 (cc / sf / ccs / rss) 每个 render pass 发一次, 不在每个 draw 上重发.

## 7. 模糊效果: compute 上的 dual Kawase (`frost.rs`, `frost` 示例)

- 照 acde 合成器 (acde-base `crates/shell/src/compositor/web.wgsl` + `compose.rs`) 的三种效果: 毛玻璃 (`backdrop-filter:
  blur() saturate() brightness()`, 包括原生窗口下面那层壁纸 `under`)、CSS `filter` (模糊后乘颜色矩阵)、`drop-shadow`.
  层数 `clamp(ceil(log2(max(σ,1)/2)), 1, 5)`, down 5 个采样点、up 8 个, 每层 RGBA8 (和合成器的 Rgba8Unorm 一样每趟量化).
  定义 (逐条运算) 写在 `frost.rs` 的模块文档里; 示例里的参照按定义另写, 不和生成器共用代码.
- **不用采样器**: 硬件双线性的权重是定点的, 没法和 Rust 逐位一致. 双线性在着色器里用 f32 自己算, 值一律是字节单位的
  f32 (0..255), lerp 是 `fma(t, b - a, a)` (EU 的 mad 单次舍入, 和 Rust `mul_add` 一致, `math` 示例验过).
- 着色器由 Rust 生成 MLIR (`Ssa`: 一行一个 SSA 值, `binary!` / `unary!` 生成运算方法), 建 `Frost` 时编译一次 (7 个:
  down、up、backdrop、filter、shadow, 以及 `under` 的合成和直通 alpha 图层的预乘). 常量用十六进制 f32, 位模式确定.
  每像素一个调用, 网格 (ceil(w / 16), h), 行尾多出的通道钳到最后一列.
- 每帧的参数: down / up 走 inline (8 个 dword), 混合的几趟每个效果一块 128 字节的参数块 (每个 ring 槽一份, `Frost::frame`).
- **性能** (本机 Xe3, 1920x1080 上 1200x700 的面板, 4 层, 每块面板的 GPU 时间, 一帧 11 块对 1 块相减):
  | 版本 | 每块 | down0 | up0 | backdrop |
  | --- | --- | --- | --- | --- |
  | 第一版 (每个采样点 4 次 gather, 逐像素除法) | 1.55 ms | 0.41 ms | 0.77 ms | 0.40 ms |
  | 采样点共用小数和行地址、相邻两个纹素一次 8 字节读、坐标比例和倒数在 CPU 上算 | 1.39 ms | 0.13 ms | 0.67 ms | 0.31 ms |
  | 按字节解包 (`detour.unpack4xu8`)、lerp 和加权和用 fma、÷12 改成 ×(1/12) | 1.02 ms | 0.11 ms | 0.46 ms | 0.27 ms |

  一帧一块面板从提交到完成约 1.5 ms. up0 每线程约 700 条指令, 时间在 ALU 上 (显存带宽很小).
  - 相邻两个纹素一次读: 在图像边上把小数换成 0 或 1 (x 方向的值是整数字节, `fma(1, b - a, a)` 就是 b), 和逐个夹紧的结果
    逐位相同; 1 宽的图像小数恒为 0 (读到的第二个纹素在图像外, scratch 末尾多留 64 字节).
  - ⚠ 字节解包: `mov :f ← r.k<32;8,4>:ub` 只有 k = 0 合法 (浮点目的操作数要求源区域保持最低通道的位位置, gentool 校验),
    k = 1..3 先 `mov :d ← :ub` 再转 f.
  - 测时间要背靠背连跑: 两次之间 CPU 做几秒核对的话 GPU 会降频, 测出来的时间翻倍还多.
- 还能做: 2 倍上采样时一个通道算相邻两个像素 (共用读和解包, 但奇数尺寸的比例不是 0.5, 要两套着色器); 每层合成一次 dispatch
  (现在每趟后都有一个带 CS stall 的 PIPE_CONTROL).

## 8. 编译期: 类型化指令 op 的开销

只在建 pipeline / 编译时, 不在每帧的路径上. 量法: release, `perf stat -e instructions:u`; 一个模块里放同一个着色器 1 份和 21 份,
差值 / 20 就是每个着色器的编译开销 (把启动去掉).

- AMD 指令方言改成类型化 op 后 (`1de94ab`): 单个着色器整条流程 1.003 亿条 (改前 1.031 亿, 启动少注册了 1300 个 op); 每个着色器的
  编译 1140 万条 (改前 980 万, +16%).
- 先踩过的坑: 编码和分配后的各个 pass 每条指令都 `head::<Valu>()` 一次, 走类型化 op 的 serde 通用形式, 是改前的 1.7 倍;
  serde 派生的大枚举 (`Opcode`, 1237 个助记符) 反序列化是逐个比较字符串. 改成: 构造用 `Build` 直接写 `OpData`, 读取用
  `Isa::read` 单遍读成 `Fields`, 助记符用生成的 FNV-1a 哈希表查. 类型化 op 只用来解析和打印模块.
- 剩下的 +16% 主要在 mlirpass 0.4.1 的 `Body::create` ⇒ `place()`: 每建一个 op, 每个属性都在该 op 的 `inherent` 列表里
  线性比字符串; 类型化 op 有十几个 inherent 属性 (旧的通用 op 没有). 两个办法, 都要改 mlirpass 发新版 (用户的仓库, 要用户同意):
  1. `OpInfo.inherent` 排序后二分查找 (或者建 op 时按名字查一次得到位图);
  2. 给已经按 inherent / discardable 分好的 `OpData` 一条不再 `place()` 的快速路径 (像 `push_op`, 但走 Rewriter / 转换的监听).
