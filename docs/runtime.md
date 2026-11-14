# 运行时

`crates/intel/driver/src/runtime.rs`. MLIR 只在建 pipeline 时跑一次; 每帧只拷模板、填洞, 不跑 mlirpass、不读属性.
做法同 anv: pipeline 的状态建时打包, draw 时合进动态部分 (`~/coding/mesa/src/intel/vulkan/genX_gfx_state.c` 的
`_inst_data` | `_partial`).

## 对象

| 类型 | 是什么 | 生命周期 |
| --- | --- | --- |
| `Context<'d, G: Gpu, H: Host>` | 一个 VM、一个 render 队列; 代码堆 (1 MB, kernel、动态状态、fence)、binding table pool (64 KB, render surface state 也在里面), 各一个常驻 BO; 4 个 batch (64 KB) 的环; 上下文命令 (建时填好, 每帧原样拷) | 一个设备一个 |
| `Pipeline` | kernel 在代码堆里的地址; 一个 draw 的全部命令 (`Cmds::draw`) 打成的一个模板 | 建一次, 一直用 |
| `Kernel` | compute 着色器在代码堆里的地址; 三种 dispatch (直接、带 binding table、间接) 的 COMPUTE_WALKER + 刷写各打成一个模板, 洞 (组数 `groups_x/y`、参数 `param0..7`、`table`、`args0..2`) 建时认成 `Walk`, `Recorder::dispatch*` 只拷模板填数 | 建一次, 一直用 |
| `Target<'d, G>` | 渲染目标, 自己的 BO (linear, 可 `export()` 成 dma-buf); 建时把它的 pass 状态 (视口、blend、color calc、surface、binding table) 放进堆 | 一个画布一个 |
| `Buffer<'d, G>` | 顶点等 draw 要读的东西, 自己的 BO | |
| `Image2d<'d, G>` | 片段着色器采样的纹理 (`detour.sample`), 自己的 BO (linear), surface state 在 pool 里 | 建一次 |
| `Recorder` | 一帧: `Context::record` → `target()` → `draw()` / `dispatch()` …→ `finish()` 得到 dword, `Context::submit` 提交 | 每帧 |

地址: BO 从 1 MB 起按 64 KB 对齐往上排, 绑定一次长期有效 (xe 没有重定位). 所有 state base address 都是 0, 指针就是 GPU 地址
(状态指针只有 32 位: 都在 4 GB 以下). 堆是 bump 分配, 现在不回收 (pipeline、目标都是长期的).

## 模板和洞

`XE3.template(op, attrs)` 打包时把属性里的符号当成洞 (记下位、宽、地址的对齐移位、`[@x, n]` 的 n), 先填 0;
`Template::emit(out, value)` 拷贝 dword、按洞名问 `value` 填进去. 洞名约定:

| 洞名 | 谁填 |
| --- | --- |
| `vs` 等 kernel 函数名 | `Pipeline` 的 kernel 地址 |
| `vb0`、`vb0_size`、`vb1` … | draw 的顶点缓冲地址、字节数 |
| `vertices`、`instances`、`first` | draw 的计数 |
| `cc`、`sf`、`blend`、`ccs`、`rss` | 当前 `Target` 的 pass 状态 (`SHARED`; `rss` 是目标的 surface state) |
| `surfaces::bt` | 这个 draw 的 binding table (`Recorder::draw` 当场建: 目标, 再是纹理), 在环槽的 4 KB 区里 |
| `samplers` | `Pipeline` 的 sampler 状态 (有采样时建 pipeline 时放进代码堆) |
| `fence`、`surfaces`、`bt_null` (`surfaces::bt_null`) | `Context` 自己 |

`surfaces::x` 是 x 在 binding table pool 里的偏移 (binding table 指针是 pool 内偏移), 其余是地址. `Context::pipeline` 建模板时
检查每个洞都有人填, 没有就报错: 漏填的洞会变成地址 0, GPU 缺页、队列被封.

`Context` 自己的名字只在 `Context::new` 里放, 之后不动: 上下文命令建时就填好, pipeline 里用到它们的洞建 pipeline 时就解析成数
(`Fill::Value`), 每帧不再按名字查.

加一个 draw 时才知道的参数: 在 `Cmds::draw` 里让对应字段取 `A::sym("名字")`, 在 `Context::pipeline` 的 `fill` 里把名字认成一种
`Fill`, 在 `Recorder::draw_at` 的 `value` 里按这种 `Fill` 填数.

## 提交和同步

binding table 每个 draw 一张: `Recorder` 只记下来, `submit` 在等过这个环槽之后写进它在 pool 里的区 (每槽 4 KB), 再提交.

`submit` 在 dword 后面补 PIPE_CONTROL (刷新) 和 MI_BATCH_BUFFER_END, 写进环里下一个槽 (这个槽上一帧没完就先等它), CPU 写完
`H::fence()` 再提交. 每个槽一个 fence: `wait(n)` 等它并查 `Queue::fault()`, `export_fence(n)` 导出 sync_file 给合成器当 acquire fence.

## 量

`cargo run --features xe3 --example frames [N]`: 结果见 [verification.md](verification.md).

## 索引绘制

每条管线编两份 draw 模板: 顺序的, 和带 `3DSTATE_INDEX_BUFFER` / RANDOM 访问的 (`Pipeline::indexed`). `Recorder::draw_indexed`
填索引缓冲的地址和大小、格式 (`IndexFormat` ⇒ `INDEX_WORD` / `INDEX_DWORD`)、第一个索引、base vertex (32 位补码) 这几个洞;
字段的设法照 iris (`iris_upload_render_state` 的 index buffer, `iris_draw_vbo` 的 3DPRIMITIVE).

## 深度

`Context::depth_buffer(w, h)`: depth32float, Tile4 (128 字节 × 32 行一块; Xe2 起深度面只能是 Tile4 / Tile64, isl), 只给 GPU 用.
`Recorder::depth(&d)` (在 `target` 之后) 让之后的 draw 带这个深度附件; 管线有 `depth = {compare, write}` 和 pass 有深度附件必须一起
(不一致就报错, 同 WebGPU). 带深度的管线的模板多四个洞: 地址、pitch − 1、宽 − 1、高 − 1. 清深度用 `clear_depth_pipeline`
(比较 always、写深度) 加 `clear_depth_vertices(z)`, 和清颜色是同一个 draw. 没有 HiZ.

## 缓冲绑定 (uniform / storage)

着色器里 `detour.load(offset) {binding = k}` / `detour.store`: LSC 按 BTI 寻址 (扩展描述符 31:24 位是表项, Mesa
`gen_lsc_ex_desc_encode`), offset 是字节. 每个 `Buffer` 建时就有一个 RAW 缓冲 surface (isl_buffer_fill_state_s 的填法).
`Recorder::bind(&[..])` 设之后 draw 的缓冲: binding table 是目标、纹理、空 surface 填到 `BUFFER_BTI` (32)、再缓冲们;
用缓冲的管线的顶点着色器也指向这张表 (不用缓冲的仍是 `bt_null`). 管线的 `buffers` 是两个 kernel 的较大者.

## mipmap

`Context::texture_levels(w, h, format, &[level0, level1, ..])`: 每级紧排的像素, 放进 isl 的线性 mip 布局 (`Texture::layout` 给行距、行数、
每级的位置). 管线 `mipmap = "nearest" / "linear"` 定 mip 过滤; 采样由导数定级.
