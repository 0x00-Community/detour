# xe3 方言和 Xe3 硬件

## 方言

- genxml 的每个 instruction / struct / register 是一个只带属性的 op: `xe3.cmd.<名>` / `xe3.state.<名>` / `xe3.reg.<名>`, 名字是
  genxml 名的小写下划线形式 (`3DSTATE_VS` ⇒ `xe3.cmd.3dstate_vs`), 字段是同名属性 (小写下划线, 数字打头的前面加 `_`). 缺省 =
  genxml 的 default. 结构体字段是字典, 定长数组是数组 (多维嵌套), 变长组是 `elements = [{...}, ...]`. `dword_length` 不写就按
  打出来的长度算 (变长组、越过声明长度的长形式). 表驱动打包 (detour-core 的 `pack.rs`), 表由 `scripts/intel-cmdgen.py` 从 `crates/intel/dialect/genxml/xe3.xml`
  生成 (`python3 scripts/intel-cmdgen.py crates/intel/dialect/genxml/xe3.xml crates/intel/dialect/src/cmd.gen.rs`). xe3.xml 是 Mesa 的
  `src/intel/genxml/xe3.xml` 连同它导入的 xe2 / gen125 / ... 用 `genxml_import.py --flatten` 展平的结果 (MIT, Intel). 视频引擎的不生成.
- 对照: `tests/oracle.mlir` 每个条目每个字段一个伪随机值 (和当初 Mesa C 打包函数拿到的相同), `tests/oracle.expected` 是 Mesa 打出来的;
  290 条逐位一致. 和 Mesa 有意不同的: 同一 dword 起始的替代地址字段按位或 (Mesa 只取最后一个).
- `xe3.eu.<助记符>`: 一条 EU 指令. 操作数是 gentool 语法的字符串属性 (`dst = "r10<1;1,0>:f"`, `src1 = "-r12<1;1,0>:f"`,
  立即数 `"0x3f800000:f"`), 其余见 `asm.rs`. 寄存器分配之前操作数可以是 SSA 值 (类型 `!xe3.grf<f32>` 等, 一个 SIMD16 寄存器),
  空着的源槽按顺序填; `xe3.eu.payload {grf}` 是线程开始时就在寄存器里的值. 对照: `tests/eu.mlir` 每条上面的注释是 gentool 汇编,
  `tests/eu.expected` 是 `gentool asm -p ptl` 汇编出来的 (`scripts/intel-eu-oracle.sh` 重新生成); 67 条逐位一致.
- 收尾 (`xe3-pad`, `xe3-end`): kernel 带 `pad = 128` (后面补 128 字节 0, EU 会预取过尾); 每个 batch 末尾加
  `xe3.cmd.mi_batch_buffer_end`, 总长奇数再加 `xe3.cmd.mi_noop`. 输出只按 op 一个个打包, 不再自己加东西.
- 定址 (`xe3-layout`): memory 从 1 MB 起, 64 KB 对齐 (可用 `align` 加大); 项按 `align` (缺省 64) 排. 属性里的符号: `@x` 是 x 的 GPU
  地址, `@m::@x` 是 x 在 memory m 里的偏移, `[@x, n]` 是 x 的地址加 n. 所有 state base address 都是 0, 指针就是 GPU 地址
  (必须在 4 GB 以下); binding table 指针是 pool 内偏移.

## 踩过的坑

- 提交后等到 fence 不代表作业跑完了: 作业出错或挂起时 xe 照样 signal. 等完要看 `Queue::fault()` (`Device::run` 已查).
- GPU 挂起 / 缺页时: `journalctl -k` 里有缺页地址和引擎. 定位卡在哪条命令: 在 batch 里插 `xe3.cmd.mi_store_data_imm` 打点读回;
  再关掉某个阶段二分.
- binding table 只能放在 binding table pool 里 (`3DSTATE_BINDING_TABLE_POOL_ALLOC` 的大小以 4 KB 为单位), 超出 pool 的表指针会回绕;
  每个阶段都要发 binding table 指针 (不绑东西的阶段给一张 entry 0 是 null surface 的表). Xe-HP 起没有 `3DSTATE_PUSH_CONSTANT_ALLOC_PS`.
- `STATE_SYSTEM_MEM_FENCE_ADDRESS` 的地址要 4 KB 对齐.
- EOT 消息的源必须在 r112 ..= r127.
- gentool 读 `load_block2d` 的缓存后缀时查的是 store 的名字表 (load 的 L1C_L3C = 8 写成 `wt.wb`).
- mlirformat 认不得的自定义形式 (`vector.shuffle`、spirv 全部) 写通用形式. `vector.from_elements` 的自定义形式 0.5.0 起能读.
- ⚠ 线程启动时 f32 非规格化数冲零 (cr0 bit 7 = 0); 每个 kernel 的序言置上它. 采样器返回的 f32 纹素里非规格化数总是冲零.
  EU 没有 f32 除法、精确平方根, 只有近似的 inv / rsq: 都修正到正确舍入. 详见 [optimizations.md](optimizations.md) 第 1 节.
- ⚠ send 的目的寄存器不能和它的 payload 重叠 (Mesa 的寄存器分配让两者互相干扰); 重叠时读回随机错.
- ⚠ 软件记分板: 跨管线的写后读也要等 (int 管线还没读, float 管线就写了); RegDist 和 SBID 合在一条指令上时不带管线号
  (按源类型推断), 只有特定组合能合并; 等过 `$n.src` 不代表结果到了. 见 [optimizations.md](optimizations.md) 第 2 节.
- ⚠ 别人的 dma-buf (udmabuf 之类, xe 里是 sg buffer object) 只能用一致的 PAT 绑进 VM (`MemType::Coherent`, PAT 1); 用不一致的
  (`Streaming`, PAT 0) vm_bind 报 EINVAL. `Context::import_canvas` 用 Coherent (`compose` 示例: udmabuf 画布逐字节对).
- ⚠ compute shader 最多 8 个参数 (`COMPUTE_PARAMS`); 超了报的是 "failed to legalize operation 'func.func'" (同一模块里任何一个入口超了, 编哪个入口都报). `i1` 参数的私有函数也编不过, 用 select 展开.
- ⚠ SCISSOR_RECT 要 64 字节对齐 (Wa_1409725701, anv); 32 字节对齐时开了 scissor test 什么都画不出来.
- ⚠ sRGB 格式的转换是近似的 (在 D3D 容差内, 不等于传递函数的精确值): 解码 256 个字节最多差 0.29 个字节步长 (194 处 1.1e-3), 编码线性 0.5 得 187 (公式 188), 256 个值里 20 个差 1 (`formats` 示例). 要逐位对照的东西别走 sRGB 格式.
- ⚠ cube 采样: 采样器先把方向坐标钳到 −1 ..= 1 再选面, 所以方向要先除以最大分量的绝对值 (`detour-normalize-cube`); (3, 2.25, 2.25) 不除会落到 +z 的角上.
- ⚠ Xe2 Tile64 多重采样的 tile (32 位纹素): 2x 是 64 宽 × 128 高 (不是 128 × 64), 4x 64 × 64, 8x 64 × 32 (isl 的 swizzle: U 8 位, V 7 / 6 / 5 位); 尺寸算错时最后一行 tile 越出缓冲, 画到别处.
- alpha to coverage (不开 dither): 盖住 ceil(alpha · n − 1/2) 个采样 (4x / 8x 各扫 257 个 alpha 量出来的).
- compute 的工作组号: x 在 r0.1, y 在 r0.6, z 在 r0.7 (Mesa `emit_work_group_id_setup`).
- ⚠ 字节转浮点 `mov :f ← r.k<32;8,4>:ub` 只有 k = 0 合法 (浮点目的操作数的源区域要保持最低通道的位位置, gentool 校验报错);
  其余字节先转 `:d` (`detour.unpack4xu8`).
- ⚠ 没有 32 × 32 位整数乘 (`mul :d :d` 奇数通道错, 可能导致越界写和 CAT error); `cmp` 不能混用整数源和浮点目的操作数
  (目的寄存器不被写). 见 [optimizations.md](optimizations.md) 1.5.

## Xe2 / Xe3 片段着色器的 payload 和插值 (照 Mesa brw)

Intel 没有公开 Xe2+ 的 PRM, 下面是从 Mesa 的编译器读出来的 (路径相对 `~/coding/mesa/src/intel/compiler/brw`), 实现 varying 时照它:

- PS SIMD16 payload (`brw_thread_payload.cpp` `setup_fs_payload_gfx20`): r0 线程头, r1 掩码和像素坐标; 然后按 `3DSTATE_WM` 的
  Barycentric Interpolation Mode 打开的每种重心坐标各 2 个寄存器 (b.x 16 个通道一个, b.y 一个); 再往后是可选的源深度、源 W、
  覆盖掩码等. payload 之后 (`dispatch_grf_start_register_for_constant_setup_data_0`) 是属性的平面参数.
- 平面参数 (`brw_compile_fs.cpp` `remap_attr_reg`): Gfx20+ 每个标量分量 12 字节 (a0, a1-a0, a2-a0), 每 5 个分量打进一个 64 字节寄存器;
  分量 i 在 setup 区第 i / 5 个寄存器的第 (i % 5) * 12 字节.
- 插值: Xe2+ 没有 PLN (`gen_validate.cpp`: "PLN is Gfx9 only"), Mesa 在 NIR 里先变成两次 FMA (`nir_lower_interpolation.c`).
  ⚠ 真机验证过的对应关系 (`varying` 示例): `v = b.x * (a1 - a0) + a0`, 再 `v = b.y * (a2 - a0) + v` (b.x 在 r1, 跟着顶点 1;
  b.y 在 r2, 跟着顶点 2). 照字面读 Mesa 那段 NIR 会把 b.x / b.y 弄反.
- 我们的实现 (`shader.rs` `Interpolate` 和 `place_arguments`): 开 `3DSTATE_WM` 的 Barycentric Interpolation Mode bit 0
  (透视像素), PS 的 setup 数据从 r3 开始 (`dispatch_grf_start_register_for_constant_setup_data_0 = 3`), 每个 varying 4 个分量.
- 顶点着色器写 VUE (`brw_lower_logical_sends.cpp` `lower_urb_write_logical_send_xe2`): LSC 的 URB store, 地址是 URB 句柄 (r1),
  数据按分量排; VUE 里是头 (4 dword)、位置, 然后每个输出一个 vec4 (16 字节一槽). Xe2+ 的 URB 偏移按字节, 是 FLAT 地址的立即数偏移,
  放在扩展描述符 bit 31:12 (`gen_lsc_ex_desc_encode`): 第 k 个 varying 在 32 + 16k. 我们每个 varying 一条非 EOT 的 URB 写 (取 SBID),
  最后一条 EOT 写头和位置.
- SBE (`iris_emit_sbe`): 读偏移 1 (256 位, 跳过头和位置)、读长度 ceil(n / 2)、`number_of_sf_output_attributes = n`; `3DSTATE_SBE_SWIZ`
  的第 i 个属性 source_attribute = i (默认全是 0: 每个属性都读第 0 个 varying). 现在最多 2 个 varying.
- ⚠ VS 的 URB 条目先放 VF 取来的顶点 (每个输入一个 vec4), VS 再在同一条目里写 VUE: `3DSTATE_URB_ALLOC_VS` 的条目大小
  (64 字节为单位, 减 1) 要够 max(输入数, 2 + varying 数) 个 vec4. 默认 0 只够 4 个: 5 个输入时顶点溢进相邻条目, 别的顶点 / 实例的
  属性读出来是错的, 没有任何报错 (本机实测, `strips` 加 paint 属性时). 条目变大时条目数按比例减 (8 的倍数), `render.rs` `draw`.

## 纹理采样 (照 Mesa brw)

- `detour.sample` ⇒ SAMPLE_LZ (消息类型 24, 0 级、不要导数), SIMD16 (Xe2 的 simd_mode 1), 不带头 (`lower_sampler_logical_send`):
  payload u、v 各一个寄存器, 回来 r、g、b、a 四个. 描述符 `gen_sampler_desc_encode` (和 `msg::Sampler` 一样). 不带头时 sampler 状态的
  指针来自 `3DSTATE_SAMPLER_STATE_POINTERS_PS`.
- mipmap 纹理 (线性): isl 的 GFX4 2D 布局 (0 级在上, 1 级在下, 2 级在 1 级右边, 之后依次往下), 对齐 128 字节 × 4 行 (Xe2+ 线性面
  HALIGN_128 / VALIGN_4, isl_gfx12 / gfx20 choose_image_alignment_el); 没有 mip tail (MipTailStartLOD = 15). 一级的纹理仍按 `width * 4` 紧排
  (渲染目标和合成器的 dma-buf 是这样). 采样器: MIPFILTER, LOD 0 ..= 14, CLAMP_MODE_OGL (iris).
- binding table: entry 0 是渲染目标, 第 b 个纹理是 entry 1 + b, sampler 状态 b. `3DSTATE_PS` 的 binding_table_entry_count = 1 + n,
  sampler_count = ceil(n / 4).
- SAMPLER_STATE 照 iris `iris_create_sampler_state`: clamp、线性时开地址舍入、边框颜色指针指向一块清零的 SAMPLER_BORDER_COLOR_STATE.
- `detour.sample` 是 SAMPLE (消息类型 0, 由 2x2 块的导数定级, WGSL `textureSample`), 不是 SAMPLE_LZ; 一级的纹理照样取 0 级.
- ⚠ **采样要 Vector Mask** (本机实测, 2026-11-03): 片段着色器采样时 3DSTATE_PS 的 Vector Mask Enable 要开. 用 dispatch mask 时,
  被三角形边切开的 2x2 块里的辅助像素不执行 (它们的 uv 是寄存器里的残值), 导数错, 对角线上取到 6 ~ 8 级 (`mipmap` 示例 928 个错);
  开了就对. 写渲染目标仍然只写覆盖的像素 (红绿两个三角形共边, 后画的辅助像素没覆盖先画的, 0 错). Mesa 在 XeHP+ 上只在 coarse /
  宽子组时开 VMask, 这里和它不同, 以真机为准. 注意: 开了以后辅助像素也执行缓冲的 store (Mesa 用 sample mask 谓词挡掉, 我们还没做),
  所以有采样的片段着色器里的 store 会多写.
- 有返回的 send 是乱序的: 取 SBID, 用它结果的指令等 `$n.dst` (`xe3-swsb`); 结果占连续 4 个寄存器 (`xe3-regalloc`).

- **只有 compute 的帧没有 binding table pool**: 3DSTATE_BINDING_TABLE_POOL_ALLOC 原来只跟着 draw 发, compute 的 interface
  descriptor 带 binding table 指针 (pool 里的偏移) 但这一帧没 draw 时 pool 没设, 第一次 TGM 消息就 GPU fault (2026-11-04).
  带 binding table 的 compute walker 前先发一次 pool (`Cmds::compute_bound`).
- **storage texture 走 TGM 的 typed 消息** (Mesa brw `lower_lsc_surface_logical_send`): SFID TGM, LOAD_CMASK / STORE_CMASK,
  BTI 寻址 (扩展描述符 31:24), A32、D32, cmask 是通道数的低位全 1; 2D 是 u、v 两个坐标各一个寄存器 (SIMD16, Xe2 起不拆成
  SIMD8), 数据另起 src1. 越出表面的写由数据端口丢掉; 格式转换 (float → unorm 等) 也是它做.

- **整数向量立即数 (`:v` / `:uv`) 的目的要是 word**: `mov r<1>:d 0x44440000:uv` 不报错但每个通道都得 0 (sample_index
  起初取到的全是相邻槽的号). 照 Mesa 整段在 packed word 里算 (`dst_as = "uw"` / `as = "uw"`), 最后再放宽成 dword
  (`packed0`).

- **双源 RT 写在 Xe2+ 是 SIMD16**: 只是 SIMD32 不行 (Mesa brw_lower_simd_width; SIMD8 拆分是 Gfx12 及以前的). message
  control 是 2 (SIMD8_DUAL_SOURCE_SUBSPAN01 这个名字), 不是 1 (SIMD16_SINGLE_SOURCE_REPLICATED: 拿 8 个寄存器的
  payload 给它就 GPU fault).

## 混合

- 混合属于 pipeline (`blend = "premultiplied"`): BLEND_STATE 加一个 BLEND_STATE_ENTRY (背靠背), 颜色和 alpha 都是 ONE / INV_SRC_ALPHA / ADD;
  `3DSTATE_PS_BLEND` 同样的因子 (iris 的 PS_BLEND 和 entry 0 一致). 开混合时关掉 dither (iris 跟 GL 的 dither 走, GL 默认开):
  dither 会在 unorm8 结果上加噪声.
- ⚠ 混合的算术 (本机实测, `strips` 示例): 对 unorm8 目标, 着色器的输出先舍入成 unorm8, 再 `src + dst * (255 - src.a) / 255`
  舍入回 unorm8, 每画一层都量化一次. 和这个模型 (f64) 逐通道比, 92% 相等, 其余差 1 的全部是模型里某次舍入落在 .5 附近 (0.01 内):
  GPU 自己的转换 (unorm8 → float、乘法) 不是精确的 f64, 只在分界点上露出来. 所以和「最后才量化一次」的渲染器 (vello_cpu 的 f32
  管线) 比, 层数多的地方会差到 2.

## 参考

Intel 没有公开 Xe3 的 PRM. 寄存器和命令布局来自 Mesa 的 genxml, 指令编码和管线状态的写法参照 Mesa 的
`src/intel/compiler/gen` (编码器 / 汇编器 / 反汇编器)、`compiler/jay` (Xe2 / Xe3 的新编译器)、`vulkan` (anv)、`blorp`.
本地有一份 Mesa (`~/coding/mesa`, 浅克隆) 只作参考, 不进构建, 只读; 怎么找见 [.skills/detour-mesa](../.skills/detour-mesa/SKILL.md).

- **z 的约定** (2026-11-03): 视口 m22 = 1、m32 = 0, 3DSTATE_CLIP API Mode = D3D: NDC z ∈ [0, 1] 就是深度 (WebGPU / Vulkan; anv 的
  `depth_clip_negative_one_to_one` 为假时). 之前是 GL 的 m22 = m32 = 0.5. 深度 1.0 的清屏三角形在远平面上, 不被 z 裁掉 (`depth` 示例验过).
- **深度面是 Tile4**: 3DSTATE_DEPTH_BUFFER 的 Tiled Mode 只有 TILE64 / TILE4; pitch 按 128 字节、行数按 32 对齐; 有深度面时
  Depth Write Enable 恒为 true (isl_emit_depth_stencil_hiz_s), 真正的写开关在 3DSTATE_WM_DEPTH_STENCIL.

- **MSAA 面只能是 Tile64** (isl_gfx20 filter_tiling: RENDER_SURFACE_STATE::NumberofMultisamples 只在 Tile Mode = Tile64 时能大于 1).
  Xe2 的 Tile64 把采样号编进 tile 内地址 (`xe2_tile64_2d_32_16bpp_4msaa_swiz`: V5 V4 U7 U6 V3 V2 S1 S0 U5 U4 V1 V0 U3 U2 U1 U0):
  32 位像素 4x 时一块 64 KB 是 64 × 64 像素 × 4 个采样, 物理 256 字节 × 256 行. MSAA 面不在 CPU 上读 (resolve 去读), 只要尺寸、
  行距、surface 字段对.
- **LD2DMS_W 只收 16 位参数** (gfx12.5+, Mesa brw_lower_logical_sends.cpp: "ld2dms_w SIMD8H and SIMD16H Only"): SIMD Mode 用
  SIMD16H (5), 每个参数 (采样号、MCS 0..3、u、v) 各占一个 GRF, 16 个通道的 16 位值在它的低 32 字节. 给 32 位参数时采样器把它们
  当成 16 位读, 读到的位置错乱 (2026-11-03 上板: 三角形放大一倍、隔列错).
- **控制流照 Mesa 结构化** (brw_to_binary.cpp / gen_control_flow.cpp): `(f0.0) if` / `else` / `endif`, 循环没有 DO 指令,
  `(f0.0) break` 让做完的通道离开, `while` 跳回循环头; JIP / UIP 是从指令自己算的字节数 (16 一条): if 有 else 时 JIP 到
  else + 1、UIP 到 endif; else 两个都到 endif; endif 的 JIP 到外层作用域的结束 (外层的 else / endif / while), 没有外层就下一条;
  break 的 UIP 到 while、JIP 到最内层作用域的结束; while 的 JIP 回循环头 (负数). 2026-11-03 上板: 各通道不同的条件和次数都对.
- **MRT 的 BLEND_STATE 项由 RT 写消息的 Render Target Index 选**, 不是 binding table 下标: Xe2+ 在扩展描述符 bit 24:21
  (Mesa lower_fb_write_logical_send: `ex_desc = target << 21`). 不填时每个目标都用第 0 项 (2026-11-03 上板: 第二个目标的结果
  正好是第 0 项的写掩码和不混合).
- **dword 打包成 word 不能 `mov r<1>:uw r:d`**: 区域规则要求目的的步长等于执行类型和目的类型大小之比 (这里是 2), 硬件按步长 2
  写, 奇数通道错. 照 Mesa 写成同类型的 word 搬运, 源取每个 dword 的低半: `mov r<1>:uw r<16;8,2>:uw`.
