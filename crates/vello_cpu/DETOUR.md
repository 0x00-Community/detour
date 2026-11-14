# vello_cpu 0.3.0, vendored

The crates.io release of [vello_cpu](https://github.com/linebender/vello) 0.3.0 (Apache-2.0 OR MIT), copied as it is
published (its `Cargo.toml` is the normalized one). detour's changes are marked with `detour:` comments; everything else is
upstream's. What detour adds: `src/detour.rs`, the coarse rasterization of a scene as plain data (per strip row the render
and depth commands, the paint and layer attributes, the alpha masks, the encoded paints) for detour's GPU fine
rasterization (roadmap R3 / R4), and `RenderContext::coarse`, which makes it (its filter layers rendered and filtered
with the u8 kernel, as `RenderMode::OptimizeSpeed`). `RenderContext::coarse_layered` makes it with the filter layers left to the caller
(`detour::Layered`: each one's own coarse rasterization; the scenes compositing them paint with an image by id,
`detour::FILTER_IMAGE_BASE` + the layer's id: `FilterContext::elsewhere`, `filter_source`, the bucketer's
`generate_filter_layer_fill` taking an image source), and `detour::filter` runs a layer's filter as the u8 kernel does.

Changed behaviour (each marked `detour:`):

- Gradient positions (`fine/common/gradient/mod.rs`): every pixel's own, `fma(x, x_advance, fma(y, y_advance, p0))`
  in f32 from the first pixel center's image (`detour::Positions`), where upstream adds `2 x_advance` in f64 from the
  start of each painted span (so a position depends on where the span, split by the depth buffer, starts). The GPU
  computes the same; positions differ from upstream's by a few ulps.
- Image positions (`fine/common/image.rs`, `fine/lowp/image.rs`): the same, each pixel's own from
  `Positions::of_image`, in every image painter (nearest neighbor, lowp's bilinear, the filtered one: bilinear f32 and
  bicubic), where upstream adds the advance in f32 (or f64) from the span's start; the axis-aligned painters are the
  general ones. `ImagePainterData` keeps no running position.
- Box shadows (`fine/common/rounded_blurred_rect.rs`): each pixel's position on its own (`Positions`), and
  `detour::pow` (exp2 / log2 from the bits and fused polynomials, 1.5e-6 relative error at worst) in place of libm's
  `powf`, which the GPU cannot reproduce bit for bit.
