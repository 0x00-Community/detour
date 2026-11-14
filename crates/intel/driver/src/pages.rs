use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::base::{Content, Dmabuf, Pass, Submitted, Filter, FilterEffect, Gpu, Glyph, Host, Part, Px, NormalizedCoord, Paint, PaintRef, PaintScene, Scene};
use crate::vello_cpu::detour::FontEmbolden;
use crate::vello_cpu::kurbo::{Affine, Diagonal2, Rect, Shape, Stroke, Vec2};
use crate::vello_cpu::peniko::{BlendMode, Color, Fill, FontData, ImageAlphaType, ImageFormat, StyleRef};
use crate::vello_cpu::filter_effects::{EdgeMode, Filter as CpuFilter, FilterGraph, FilterPrimitive};
use crate::vello_cpu::{Image, ImageSource, PaintType, Pixmap, RenderContext, Resources};
use crate::base::vello_common::paint::ImageId;

use crate::xe3::{Buffer, Canvas, Codegen, Context, Error, Filters, Fine, Frost, GpuTexels, ImportedCanvas, Layer, Plane, Tiles, TilesAt};
use crate::base::vello_common::filter::PreparedFilter;
use crate::base::vello_common::filter::drop_shadow::DropShadow;
use crate::base::vello_common::filter::gaussian_blur::GaussianBlur;

pub trait XeImages {
    fn find(&self, id: u64) -> Option<GpuTexels>;
}

impl XeImages for () {
    fn find(&self, _: u64) -> Option<GpuTexels> {
        None
    }
}

pub struct CpuScene<'c> {
    ctx: &'c mut RenderContext,
    resources: &'c mut Resources,
    images: &'c dyn XeImages,
    used: Vec<(u64, GpuTexels)>,
    refused: Option<&'static str>,
}

impl<'c> CpuScene<'c> {
    pub fn new(ctx: &'c mut RenderContext, resources: &'c mut Resources) -> Self {
        Self::with_images(ctx, resources, &())
    }

    pub fn reference(scene: Scene, (width, height): (u16, u16)) -> Vec<u8> {
        let mut ctx = RenderContext::new(width, height);
        let mut resources = Resources::new();
        CpuScene::new(&mut ctx, &mut resources).append_scene(scene, Affine::IDENTITY);
        let mut pixmap = Pixmap::new(width, height);
        ctx.render_with(&mut pixmap, &mut resources, crate::vello_cpu::RasterizerSettings { render_mode: crate::vello_cpu::RenderMode::OptimizeSpeed, ..Default::default() });
        pixmap.data_as_u8_slice().to_vec()
    }

    pub fn with_images(ctx: &'c mut RenderContext, resources: &'c mut Resources, images: &'c dyn XeImages) -> Self {
        CpuScene { ctx, resources, images, used: Vec::new(), refused: None }
    }

    pub fn gpu_images(&self) -> Vec<GpuTexels> {
        self.used.iter().map(|(_, t)| *t).collect()
    }

    pub fn refused(&self) -> Option<&'static str> {
        self.refused
    }

    fn refuse(&mut self, what: &'static str) {
        self.refused.get_or_insert(what);
    }

    fn paint(&mut self, p: PaintRef<'_>) -> PaintType {
        match p {
            Paint::Solid(color) => color.into(),
            Paint::Gradient(g) => g.clone().into(),
            Paint::Image(i) if (i.image.data.data().len() as u64) < u64::from(i.image.width) * u64::from(i.image.height) * 4 => {
                let id = i.image.data.id();
                let found = self.images.find(id).filter(|t| t.size == (i.image.width as u16, i.image.height as u16) && i.image.width <= u32::from(u16::MAX) && i.image.height <= u32::from(u16::MAX));
                match found {
                    None => self.refuse("a GPU image (a canvas or a WebGPU texture) not on Xe3"),
                    Some(t) => {
                        let k = self.used.iter().position(|(u, _)| *u == id).unwrap_or_else(|| {
                            self.used.push((id, t));
                            self.used.len() - 1
                        });
                        let source = ImageSource::OpaqueId { id: ImageId::new(k as u32), may_have_transparency: true };
                        return Image { image: source, sampler: i.sampler }.into();
                    }
                }
                Color::TRANSPARENT.into()
            }
            Paint::Image(i) => {
                let image = i.image;
                let mut pixmap = Pixmap::new(image.width as u16, image.height as u16);
                for (out, px) in pixmap.data_as_u8_slice_mut().as_chunks_mut::<4>().0.iter_mut().zip(image.data.data().as_chunks::<4>().0) {
                    let [r, g, b, a] = match image.format {
                        ImageFormat::Rgba8 => *px,
                        ImageFormat::Bgra8 => [px[2], px[1], px[0], px[3]],
                        _ => *px,
                    };
                    *out = match image.alpha_type {
                        ImageAlphaType::AlphaPremultiplied => [r, g, b, a],
                        ImageAlphaType::Alpha => [r, g, b].map(|c| ((u16::from(c) * u16::from(a) + 127) / 255) as u8).into_iter().chain([a]).collect::<Vec<_>>().try_into().expect("four"),
                    };
                }
                Image { image: ImageSource::Pixmap(Arc::new(pixmap)), sampler: i.sampler }.into()
            }
        }
    }

    fn filter(&mut self, f: &Filter) -> Option<CpuFilter> {
        if f.effects().len() > 1 || f.effects().iter().any(|e| matches!(e, FilterEffect::ColorMatrix(_))) {
            self.refuse("a filter of more than one effect, or a colour matrix (vello_cpu runs neither)");
            return None;
        }
        let mut graph = FilterGraph::new();
        let mut last = None;
        for e in f.effects() {
            let primitive = match *e {
                FilterEffect::GaussianBlur(b) => FilterPrimitive::GaussianBlur { std_deviation: b.std_deviation, edge_mode: EdgeMode::None },
                FilterEffect::DropShadow(d) => FilterPrimitive::DropShadow { dx: d.dx, dy: d.dy, std_deviation: d.std_deviation, color: d.color, edge_mode: EdgeMode::None },
                FilterEffect::ColorMatrix(m) => FilterPrimitive::ColorMatrix { matrix: m.0 },
            };
            last = Some(graph.add(primitive, None));
        }
        graph.set_output(last?);
        Some(CpuFilter { graph: Arc::new(graph) })
    }
}

impl PaintScene for CpuScene<'_> {
    fn reset(&mut self) {
        self.ctx.reset();
        self.used.clear();
        self.refused = None;
    }

    fn push_layer(&mut self, fill: Fill, blend: impl Into<BlendMode>, alpha: f32, transform: Affine, clip: &impl Shape, filter: Option<Arc<Filter>>, _backdrop_filter: Option<Arc<Filter>>) {
        self.ctx.set_transform(transform);
        self.ctx.set_fill_rule(fill);
        let filter = filter.as_deref().and_then(|f| self.filter(f));
        self.ctx.push_layer(Some(&clip.to_path(0.1)), Some(blend.into()), Some(alpha), None, filter);
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        self.ctx.set_transform(transform);
        self.ctx.set_fill_rule(fill);
        self.ctx.push_clip_layer(&clip.to_path(0.1));
    }

    fn pop_layer(&mut self) {
        self.ctx.pop_layer();
    }

    fn stroke<'a>(&mut self, style: &Stroke, transform: Affine, paint: impl Into<PaintRef<'a>>, paint_transform: Option<Affine>, shape: &impl Shape) {
        self.ctx.set_transform(transform);
        let paint = self.paint(paint.into());
        self.ctx.set_paint(paint);
        self.ctx.set_paint_transform(paint_transform.unwrap_or(Affine::IDENTITY));
        self.ctx.set_stroke(style.clone());
        self.ctx.stroke_path(&shape.to_path(0.1));
    }

    fn fill<'a>(&mut self, fill: Fill, transform: Affine, paint: impl Into<PaintRef<'a>>, paint_transform: Option<Affine>, shape: &impl Shape) {
        self.ctx.set_transform(transform);
        self.ctx.set_fill_rule(fill);
        let paint = self.paint(paint.into());
        self.ctx.set_paint(paint);
        self.ctx.set_paint_transform(paint_transform.unwrap_or(Affine::IDENTITY));
        self.ctx.fill_path(&shape.to_path(0.1));
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'s mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        paint_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    ) {
        self.ctx.set_transform(transform);
        let paint = paint.into();
        let layered = paint_alpha != 1.0 && !matches!(paint, Paint::Solid(_));
        match paint {
            Paint::Solid(color) => self.ctx.set_paint(color.multiply_alpha(paint_alpha)),
            p => {
                let paint = self.paint(p);
                self.ctx.set_paint(paint);
            }
        }
        if layered {
            self.ctx.push_opacity_layer(paint_alpha);
        }
        self.ctx.reset_paint_transform();
        let mut run = self.ctx.glyph_run(self.resources, font).font_size(font_size).hint(hint).normalized_coords(normalized_coords);
        if let Some(t) = glyph_transform {
            run = run.glyph_transform(t);
        }
        if embolden != Vec2::ZERO {
            run = run.font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)));
        }
        let glyphs = glyphs.map(|g| crate::vello_cpu::Glyph { id: g.id, x: g.x, y: g.y });
        let drawn = match style.into() {
            StyleRef::Fill(_) => run.fill_glyphs(glyphs),
            StyleRef::Stroke(_) => run.stroke_glyphs(glyphs),
        };
        if drawn.is_err() {
            self.refuse("a glyph run vello_cpu could not render");
        }
        if layered {
            self.ctx.pop_layer();
        }
    }

    fn draw_box_shadow(&mut self, transform: Affine, rect: Rect, color: Color, radius: f64, std_dev: f64) {
        self.ctx.set_transform(transform);
        self.ctx.set_paint(color);
        self.ctx.fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32, false);
    }
}

struct Piece {
    filters: Vec<Filtering>,
    root: Tiles,
}

enum OnGpu {
    Blur(GaussianBlur),
    DropShadow(DropShadow),
}

impl OnGpu {
    fn blur(&self) -> GaussianBlur {
        match self {
            OnGpu::Blur(b) => GaussianBlur { std_deviation: b.std_deviation, n_decimations: b.n_decimations, kernel: b.kernel, kernel_size: b.kernel_size, edge_mode: b.edge_mode },
            OnGpu::DropShadow(d) => Filters::shadow_blur(d),
        }
    }
}

struct Filtering {
    image: u32,
    content: Tiles,
    size: (u16, u16),
    filter: crate::vello_cpu::filter_effects::Filter,
    transform: Affine,
    gpu: Option<OnGpu>,
}

impl Filtering {
    fn pixels(&self) -> usize {
        (usize::from(self.size.0) * usize::from(self.size.1) * 4).next_multiple_of(64)
    }

    fn bytes(&self) -> usize {
        self.pixels() + self.gpu.as_ref().map_or(0, |g| Filters::weights(&g.blur()))
    }
}

struct FilteringAt {
    tiles: TilesAt,
    slot: u64,
    weights: u64,
}

struct PieceAt {
    filters: Vec<FilteringAt>,
    root: TilesAt,
}

pub struct Pages<'d, G: Gpu<Cmd = u64> + 'd> {
    fine: Fine,
    filters: Filters,
    frost: Frost<'d, G>,
    resources: Resources,
    arena: Option<Buffer<'d, G>>,
    inflight: Option<usize>,
    scratch: Option<Buffer<'d, G>>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> Pages<'d, G> {
    pub fn new<H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>, max: [u32; 2]) -> Result<Self, Error> {
        Ok(Pages { fine: Fine::new(cx, codegen)?, filters: Filters::new(cx, codegen)?, frost: Frost::new(cx, codegen, max, 256)?, resources: Resources::new(), arena: None, inflight: None, scratch: None })
    }

    fn piece(&mut self, scene: Scene, place: Affine, (width, height): (u16, u16), images: &dyn XeImages) -> Result<Piece, Error> {
        let mut ctx = RenderContext::new(width, height);
        let mut cpu = CpuScene::with_images(&mut ctx, &mut self.resources, images);
        cpu.append_scene(scene, place);
        if let Some(what) = cpu.refused() {
            return Err(Error::msg(format!("pages: {what} (not yet)")));
        }
        let used = cpu.gpu_images();
        let layered = ctx.coarse_layered(&mut self.resources).ok_or_else(|| Error::msg("pages: a multi-threaded context"))?;
        let mut sizes = Vec::with_capacity(layered.filters.len());
        let mut filters = Vec::with_capacity(layered.filters.len());
        for f in layered.filters {
            let size = (f.coarse.width, f.coarse.height);
            let content = Tiles::new(&f.coarse, &used, &sizes)?;
            sizes.push((f.image, size));
            let gpu = match PreparedFilter::new(&f.filter, &f.transform) {
                PreparedFilter::GaussianBlur(b) if b.std_deviation > 0.0 && b.edge_mode == EdgeMode::None => Some(OnGpu::Blur(b)),
                PreparedFilter::DropShadow(d) if d.edge_mode == EdgeMode::None => Some(OnGpu::DropShadow(d)),
                _ => None,
            };
            filters.push(Filtering { image: f.image, content, size, filter: f.filter, transform: f.transform, gpu });
        }
        Ok(Piece { root: Tiles::new(&layered.root, &used, &sizes)?, filters })
    }

    fn upload<H: Host>(&mut self, cx: &mut Context<'d, G, H>, pieces: &[Piece]) -> Result<Vec<PieceAt>, Error> {
        let len: usize = pieces.iter().map(|p| p.root.bytes() + p.filters.iter().map(|f| f.content.bytes() + f.bytes()).sum::<usize>()).sum();
        if self.arena.as_ref().is_none_or(|a| a.size < len as u64) {
            self.arena = Some(cx.buffer(&vec![0; len.next_power_of_two().max(1 << 20)])?);
        }
        let arena = self.arena.as_mut().expect("just made");
        let addr = arena.addr;
        let bytes = arena.bytes()?;
        let mut at = 0;
        let mut placed = Vec::with_capacity(pieces.len());
        for p in pieces {
            let mut filtered = Vec::with_capacity(p.filters.len());
            let mut weights = Vec::with_capacity(p.filters.len());
            for f in &p.filters {
                filtered.push((f.image, addr + at as u64));
                if let Some(g) = &f.gpu {
                    Filters::write_weights(&g.blur(), &mut bytes[at + f.pixels()..]);
                }
                weights.push(addr + (at + f.pixels()) as u64);
                at += f.bytes();
            }
            let mut filters = Vec::with_capacity(p.filters.len());
            for ((f, (_, slot)), w) in p.filters.iter().zip(&filtered).zip(weights) {
                filters.push(FilteringAt { tiles: f.content.write(&mut bytes[at..], addr + at as u64, &filtered)?, slot: *slot, weights: w });
                at += f.content.bytes();
            }
            let root = p.root.write(&mut bytes[at..], addr + at as u64, &filtered)?;
            at += p.root.bytes();
            placed.push(PieceAt { filters, root });
        }
        Ok(placed)
    }

    fn filter<H: Host>(&mut self, cx: &mut Context<'d, G, H>, pieces: &[Piece], at: &[PieceAt]) -> Result<(), Error> {
        let jobs: Vec<(&Filtering, &FilteringAt)> = pieces.iter().zip(at).flat_map(|(p, a)| p.filters.iter().zip(&a.filters)).collect();
        let most = jobs.iter().filter(|(f, _)| f.gpu.is_some()).map(|(f, _)| f.pixels()).max().unwrap_or(0);
        if most > 0 && self.scratch.as_ref().is_none_or(|s| s.size < 3 * most as u64) {
            self.scratch = Some(cx.buffer(&vec![0; 3 * most.next_power_of_two()])?);
        }
        let mut last = None;
        let mut i = 0;
        while i < jobs.len() {
            if jobs[i].0.gpu.is_some() {
                let mut r = cx.record();
                let scratch = self.scratch.as_ref().expect("made").addr as u32;
                while let Some((f, a)) = jobs.get(i).filter(|(f, _)| f.gpu.is_some()) {
                    self.fine.record(&mut r, &a.tiles)?;
                    let (w, h) = (u32::from(f.size.0), u32::from(f.size.1));
                    let plane = |k: u32| Plane { addr: scratch + k * f.pixels() as u32, stride: w };
                    let (src, dst) = (Plane { addr: a.tiles.out as u32, stride: w }, Plane { addr: a.slot as u32, stride: w });
                    match f.gpu.as_ref().expect("on the GPU") {
                        OnGpu::Blur(b) => self.filters.blur(&mut r, src, dst, (w, h), b, a.weights as u32, [plane(0), plane(1)])?,
                        OnGpu::DropShadow(d) => self.filters.drop_shadow(&mut r, src, dst, (w, h), d, a.weights as u32, [plane(0), plane(1), plane(2)])?,
                    }
                    i += 1;
                }
                last = Some(cx.submit(r.finish())?);
                continue;
            }
            let (f, a) = jobs[i];
            if let Some(n) = last.take() {
                cx.wait(n)?;
            }
            let mut r = cx.record();
            self.fine.record(&mut r, &a.tiles)?;
            let n = cx.submit(r.finish())?;
            cx.wait(n)?;
            let arena = self.arena.as_mut().expect("uploaded");
            let base = arena.addr;
            let bytes = arena.bytes()?;
            let (w, h) = f.size;
            let len = usize::from(w) * usize::from(h) * 4;
            let mut pixmap = Pixmap::new(w, h);
            pixmap.data_as_u8_slice_mut().copy_from_slice(&bytes[(a.tiles.out - base) as usize..][..len]);
            crate::vello_cpu::detour::filter(&f.filter, &mut pixmap, f.transform);
            bytes[(a.slot - base) as usize..][..len].copy_from_slice(pixmap.data_as_u8_slice());
            i += 1;
        }
        Ok(())
    }

    pub fn draw<H: Host>(&mut self, cx: &mut Context<'d, G, H>, scene: Scene, size: (u16, u16), images: &dyn XeImages) -> Result<Vec<u8>, Error> {
        self.settle(cx)?;
        let piece = self.piece(scene, Affine::IDENTITY, size, images)?;
        let pieces = [piece];
        let at = self.upload(cx, &pieces)?;
        self.filter(cx, &pieces, &at)?;
        let mut r = cx.record();
        self.fine.record(&mut r, &at[0].root)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let arena = self.arena.as_mut().expect("uploaded");
        let from = (at[0].root.out - arena.addr) as usize;
        Ok(arena.bytes()?[from..][..pieces[0].root.pixels()].to_vec())
    }

    pub fn settle<H: Host>(&mut self, cx: &mut Context<'d, G, H>) -> Result<(), Error> {
        match self.inflight.take() {
            Some(n) => cx.wait(n),
            None => Ok(()),
        }
    }

    pub fn compose<C: Content, H: Host>(&mut self, cx: &mut Context<'d, G, H>, canvas: &Canvas, parts: &[Part<C>], targets: Option<&[Px]>, images: &dyn XeImages) -> Result<(), Error> {
        self.submit(cx, canvas, parts, targets, images)?;
        self.settle(cx)
    }

    pub fn submit<C: Content, H: Host>(&mut self, cx: &mut Context<'d, G, H>, canvas: &Canvas, parts: &[Part<C>], targets: Option<&[Px]>, images: &dyn XeImages) -> Result<usize, Error> {
        self.settle(cx)?;
        let full = Px::sized((canvas.size[0], canvas.size[1]));
        let targets = targets.map_or_else(|| vec![full], <[Px]>::to_vec);
        let clips: Vec<[u32; 4]> = targets.iter().map(|t| t.xywh()).collect();
        let mut pieces = Vec::new();
        let mut tiles = Vec::new();
        for (k, part) in parts.iter().enumerate() {
            for (drawn, clip) in part.pieces(&targets, full) {
                let [ox, oy, w, h] = drawn.xywh();
                let scene = part.content.scene_for(drawn);
                if scene.commands.is_empty() {
                    continue;
                }
                let whole = (w.next_multiple_of(4), h.next_multiple_of(4));
                let size = whole.0.try_into().and_then(|tw| Ok((tw, whole.1.try_into()?))).map_err(|_| Error::msg("pages: a part past 65535 pixels"))?;
                tiles.push(self.piece(scene, Affine::translate((-f64::from(ox), -f64::from(oy))), size, images)?);
                pieces.push((k, drawn, clip));
            }
        }
        let at = self.upload(cx, &tiles)?;
        self.filter(cx, &tiles, &at)?;
        let mut r = cx.record();
        let mut f = self.frost.frame(cx.slot());
        clips.iter().try_for_each(|c| f.clear(&mut r, canvas, *c))?;
        let mut next = pieces.iter().zip(&at).peekable();
        for (k, part) in parts.iter().enumerate() {
            if let Some(b) = &part.backdrop {
                f.backdrop(&mut r, canvas, None, b, &clips)?;
            }
            while let Some(((_, drawn, clip), placed)) = next.next_if(|((p, ..), _)| *p == k) {
                let tiles = &placed.root;
                self.fine.record(&mut r, tiles)?;
                let layer = Layer { addr: tiles.out, stride: drawn.xywh()[2].next_multiple_of(4), premultiplied: true };
                for s in &part.shadows {
                    f.shadow(&mut r, canvas, &layer, s, drawn.xywh(), clip.xywh())?;
                }
                match part.effect.is_none() {
                    true => f.place(&mut r, canvas, &layer, [drawn.xywh()[0], drawn.xywh()[1]], clip.xywh())?,
                    false => f.filtered(&mut r, canvas, &layer, &part.effect, drawn.xywh(), clip.xywh())?,
                }
            }
        }
        let n = cx.submit(r.finish())?;
        self.inflight = Some(n);
        Ok(n)
    }
}

pub struct Raster<'d, G: Gpu<Cmd = u64> + 'd, H: Host> {
    pub cx: Context<'d, G, H>,
    pages: Pages<'d, G>,
    canvases: HashMap<usize, Vec<Option<ImportedCanvas<'d, G>>>>,
}

impl<'d, G: Gpu<Cmd = u64> + 'd, H: Host> Raster<'d, G, H> {
    pub fn new(gpu: &'d G, max: [u32; 2]) -> Result<Self, Error> {
        let mut cx = Context::new(gpu)?;
        let pages = Pages::new(&mut cx, &Codegen::default(), max)?;
        Ok(Raster { cx, pages, canvases: HashMap::new() })
    }

    pub fn canvases(&mut self, plane: usize, dmabufs: &[Dmabuf<'_>], (w, h): (u32, u32)) -> Vec<Result<(), String>> {
        let mut results = Vec::new();
        let canvases = dmabufs
            .iter()
            .map(|d| {
                let imported = match d.modifier {
                    Dmabuf::LINEAR => self.cx.import_canvas(d.fd, d.offset, d.stride, [w, h]).map_err(|e| e.to_string()),
                    m => Err(format!("xe3 pages: modifier {m:#x}, linear only (not yet)")),
                };
                results.push(imported.as_ref().map(|_| ()).map_err(Clone::clone));
                imported.ok()
            })
            .collect();
        self.canvases.insert(plane, canvases);
        results
    }

    pub fn release(&mut self, plane: usize) {
        self.canvases.remove(&plane);
    }

    pub fn draw<C: Content>(&mut self, passes: Vec<Pass<C>>, images: &dyn XeImages) -> Result<Submitted, Error> {
        let started = Instant::now();
        let mut last = None;
        let mut waited = Duration::ZERO;
        for pass in &passes {
            let (n, w) = self.submit(pass, images)?;
            last = Some(n);
            waited += w;
        }
        let fence = last.map(|n| self.cx.export_fence(n)).transpose()?;
        Ok(Submitted { fence, waited, recorded: started.elapsed() - waited })
    }

    pub fn submit<C: Content>(&mut self, Pass { plane, size, parts, next, targets }: &Pass<C>, images: &dyn XeImages) -> Result<(usize, Duration), Error> {
        let whole = self.canvases.get(plane).and_then(|c| c.get(*next)?.as_ref()).ok_or_else(|| Error::msg("the canvas is not imported yet"))?.canvas;
        if size.0 > whole.size[0] || size.1 > whole.size[1] {
            return Err(Error::msg(format!("a {size:?} pass into a {:?} canvas", whole.size)));
        }
        let canvas = Canvas { size: [size.0, size.1], ..whole };
        let before = Instant::now();
        self.pages.settle(&mut self.cx)?;
        let waited = before.elapsed();
        Ok((self.pages.submit(&mut self.cx, &canvas, parts, targets.as_deref(), images)?, waited))
    }
}
