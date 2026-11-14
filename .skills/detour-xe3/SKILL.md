---
name: detour-xe3
description: 动 xe3 方言：genxml 生成的 xe3.cmd / xe3.state / xe3.reg op（属性名规则、结构体字段、数组、变长组 elements、dword_length 自动算）、cmdgen 重生成、XE3.pack 和 XE3.template（符号当洞）、EU 指令 xe3.eu.*（gentool 语法的操作数串、Inst ⇄ op、编码对照）、放进 detour-core 的 image 容器和符号引用。加命令/状态/指令或改打包编码时读。
---

# xe3 方言

细节在 `docs/xe3.md`, 这里是操作步骤.

## 用一条 genxml 命令 / 状态

1. 在 `crates/intel/dialect/src/cmd.gen.rs` 里搜名字 (genxml 名小写下划线, `3DSTATE_VS` ⇒ `xe3.cmd.3dstate_vs`), 看它的字段
   (`f("name", bit, width, Ty, default, dims)`) 和变长组 (`group`).
2. 写 op: `xe3!("cmd.3dstate_vs", kernel_start_pointer = A::sym("vs"), enable = true, ...)` (codegen 里) 或 MLIR 文本
   `xe3.cmd.3dstate_vs {enable, kernel_start_pointer = @vs}`.
   - 字段名就是属性名; 数字打头的前面加 `_` (`_3d_command_sub_opcode`).
   - 结构体字段写字典, 定长数组写数组, 变长组写 `elements = [{字段 = ...}, ...]`.
   - `dword_length` 不写就自动算. 越过声明长度的字段只在有非零位时把命令变长.
   - 地址字段写符号: `@x` (地址)、`@m::@x` (在 m 里的偏移)、`[@x, n]` (地址加 n).
3. 枚举值 (`SURFTYPE_2D`、`_3DPRIM_TRILIST` ...) 在 `cmd.gen.rs` 末尾是常量; 表面格式不在 genxml 里, 在 codegen `render.rs` 的 `SurfaceFormat` (isl 的值).

genxml 里没有的条目: 不手写打包. 先确认 Mesa 的 `src/intel/genxml` 里有没有, 有就更新 `crates/intel/dialect/genxml/xe3.xml` (Mesa `genxml_import.py --flatten`) 再跑 cmdgen.

## 打包和模板

- `XE3.pack(op, attrs) -> Vec<u32>`: 不能有符号.
- `XE3.template(op, attrs) -> Template`: 符号变成洞; `Template::emit(out, |name| value)` 拷贝并填洞; `extend` 拼接.
- 对照 Mesa: `cargo test --features xe3 --test oracle`. 和 Mesa 有意不同的只有一处: 同一 dword 起始的替代地址字段按位或.

## EU 指令

- 物理形式: `xe3.eu.<助记符> {exec = 16, dst = "r10<1;1,0>:f", src0 = "...", src1 = "0x3f800000:f", swsb = "F@1", ...}`,
  完整的属性表在 `asm.rs` 开头. 操作数: `[-][(abs)]r<nr>[.<byte>]<v;w,h>:<type>`、`null`、`arf<nr>`、立即数 `<bits>:<type>`.
- `Inst::op()` 把指令变成 op, `Inst::try_from((name, attrs))` 反过来, `Inst::encode()` 出 16 字节.
- 加一种新指令形式: 在 `eu.rs` 确认编码路径 (照 Mesa `src/intel/compiler/gen`), 在 `tests/eu.mlir` 加一条带 gentool 注释的,
  `scripts/intel-eu-oracle.sh` 重生成 `eu.expected`, 跑 `--test eu`.
- 消息描述符 (`msg.rs`): `Lsc`、`Render`、`Sampler`, `SendInfo::lengths(desc, mlen, rlen)`; `Sfid` 枚举.

## 容器

容器是 detour-core 的 `image` 方言 (厂商无关): `image.memory @m {mem = "streaming" | "coherent", size?, align?} { ... }` 一个 BO; 里面放 `image.stream @b {xe3.cmd...}` (Xe3 的 batch)、
`image.kernel @k {xe3.eu...}`、`image.data @d {bytes = dense<...> | size}`、带 `sym_name` / `align` 的 `xe3.state.*`.
`xe3-layout` 定址并把符号换成数.
