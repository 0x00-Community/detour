---
name: detour-mesa
description: 去 Mesa（~/coding/mesa，只读）里找 Xe2/Xe3 的硬件用法：iris（OpenGL）、anv（Vulkan）的状态和命令怎么发，brw 编译器的 payload / 插值 / 消息，genxml 布局，gentool 汇编器；以及已经读出来、写进 docs/xe3.md 的结论。要实现新硬件功能、不确定某个状态位怎么设时读。
---

# 照 Mesa 找硬件用法

Intel 没有公开 Xe3 的 PRM. 寄存器和命令的布局、怎么用, 都从 Mesa 读 (`~/coding/mesa`, 浅克隆, 只读, 不编译, 不往上游提东西).
Khronos 的规范在 `~/coding/OpenGL-Registry`、`~/coding/Vulkan-Docs`、`~/coding/Vulkan-Headers`.

## 去哪里找

| 想知道 | 看 |
| --- | --- |
| 命令 / 状态的字段 | `src/intel/genxml/xe3.xml` (和它导入的 xe2 / gen125 ...); 我们的 `crates/intel/dialect/genxml/xe3.xml` 是展平的 |
| 某个状态 OpenGL 怎么设 | `src/gallium/drivers/iris/iris_state.c` (`genX(...)`), 首批命令 `iris_init_render_context` |
| 某个状态 Vulkan 怎么设 | `src/intel/vulkan/genX_gfx_state.c` (draw 时的合并)、`genX_pipeline.c`、`genX_cmd_draw.c`、`genX_cmd_buffer.c`、`genX_state.c` (首批) |
| 着色器的 payload、插值、消息 | `src/intel/compiler/brw/`: `brw_thread_payload.cpp` (各阶段 payload)、`brw_compile_fs.cpp` (`remap_attr_reg`)、`brw_from_nir.cpp`、`brw_lower_logical_sends.cpp` (URB / 采样 / RT 写怎么变成 send) |
| NIR 层面先做了什么 | `src/intel/compiler/brw/brw_nir.c`, `src/compiler/nir/nir_lower_*.c` |
| 指令编码、汇编语法 | `src/intel/compiler/gen/` (gentool: `gentool asm|disasm -p ptl`)、`gen_validate.cpp` (哪些指令 / 操作数在哪代合法) |
| 清屏、拷贝、blit | `src/intel/blorp/` |
| 表面格式的值 | `src/intel/isl/isl_format_layout.csv`、`isl.h` |

`src/intel/compiler/jay` (Xe2 / Xe3 的新编译器) 也可以读, 但 Mesa 规定不能改它.

## 已经读出来的 (写在 docs/xe3.md)

- PS SIMD16 payload (Xe2+): r0、r1, 然后每种开了的重心坐标 2 个寄存器, 再往后是属性的平面参数.
- 平面参数: 每个标量分量 12 字节 (a0, a1-a0, a2-a0), 5 个分量一个 64 字节寄存器.
- 插值: 没有 PLN, 两次 MAD: `b.y * (a1-a0) + a0`, 再 `b.x * (a2-a0) + 前一步`.
- URB 写: LSC 的 URB store, 地址是 URB 句柄.
- pipeline 状态: anv 建 pipeline 时打包、draw 时按位或合进动态部分 (我们的模板同理).

读出新结论就补进 `docs/xe3.md`, 写清楚出处 (文件、函数).
