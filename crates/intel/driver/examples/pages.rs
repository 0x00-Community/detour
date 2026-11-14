//! acde's pages on Xe3 (R6, `Pages`): an detour `Scene` — what acde's renderer records — drawn by replaying it into
//! vello_cpu (`CpuScene`), its coarse rasterization there and its fine one on the GPU, against vello_cpu rendering the
//! same replay (`OptimizeSpeed`), byte for byte: fills and strokes, a gradient, an image, a clipped translucent
//! layer, glyph runs (plain, emboldened, translucent with a gradient), a box shadow, a blurred layer (a filter: on the CPU, composited on the GPU).
//! Filter layers (a blur, a drop shadow around a blurred layer, a GPU image blurred) drawn on the GPU each alone,
//! filtered as vello_cpu does. A scene painting with a GPU image (no bytes) is refused without them; with them (`XeImages`: texels in Xe buffers,
//! straight and premultiplied, rows padded) it matches vello_cpu drawing the same bytes, every sampler and extend.

use std::sync::Arc;

use detour::base::{ColorMatrix, DropShadow, FilterEffect, GaussianBlur, Glyph, Gpu, Native, PaintScene, Scene};
use detour::xe3::{Codegen, Context, CpuScene, GpuTexels, Pages, XeImages};
use detour::xe::Xe;
use detour::vello_cpu::kurbo::{Affine, Circle, Point, Rect, RoundedRect, Stroke, Vec2};
use detour::vello_cpu::peniko::{Blob, Color, Extend, Fill, FontData, Gradient, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageQuality, Mix};
use detour::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};

const W: u16 = 320;
const H: u16 = 200;

fn scene() -> Scene {
    let mut s = Scene::new();
    let id = Affine::IDENTITY;
    s.fill(Fill::NonZero, id, Color::from_rgba8(240, 240, 235, 255), None, &Rect::new(0.0, 0.0, 320.0, 200.0));
    s.fill(Fill::NonZero, Affine::translate((10.5, 0.0)), Color::from_rgba8(255, 120, 0, 255), None, &Circle::new((60.0, 60.0), 40.3));
    s.stroke(&Stroke::new(3.0), id, Color::from_rgba8(20, 40, 200, 255), None, &RoundedRect::new(120.0, 20.0, 220.0, 90.0, 12.0));
    let g = Gradient::new_linear(Point::new(230.0, 0.0), Point::new(310.0, 0.0)).with_stops([Color::from_rgba8(255, 0, 80, 255), Color::from_rgba8(0, 200, 255, 255)].as_slice());
    s.fill(Fill::NonZero, id, &g, None, &Rect::new(230.0, 20.0, 310.0, 90.0));
    // a 4x3 straight-alpha image, scaled
    let texels: Vec<u8> = (0..12u8).flat_map(|i| [i * 20, 255 - i * 20, 100, 128 + i * 10]).collect();
    let image = ImageBrush::new(ImageData { data: Blob::new(Arc::new(texels)), format: ImageFormat::Rgba8, alpha_type: ImageAlphaType::Alpha, width: 4, height: 3 });
    s.fill(Fill::NonZero, id, image.as_ref(), Some(Affine::translate((20.0, 110.0)) * Affine::scale(15.0)), &Rect::new(20.0, 110.0, 80.0, 155.0));
    // a clipped translucent layer, multiplied
    s.push_layer(Fill::NonZero, Mix::Multiply, 0.7, id, &Circle::new((150.0, 140.0), 45.0), None, None);
    s.fill(Fill::NonZero, id, Color::from_rgba8(0, 180, 80, 255), None, &Rect::new(100.0, 100.0, 210.0, 190.0));
    s.pop_layer();
    // glyph runs (DejaVu Sans, when there): plain, emboldened, translucent with a gradient (an opacity layer)
    if let Ok(bytes) = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") {
        let font = FontData::new(Blob::new(Arc::new(bytes)), 0);
        let glyphs = |y: f32| (0..12u32).map(move |i| Glyph { id: 40 + i, x: 110.0 + 15.0 * i as f32, y });
        s.draw_glyphs(&font, 16.0, false, &[], Vec2::ZERO, Fill::NonZero, Color::from_rgba8(10, 10, 10, 255), 1.0, id, None, glyphs(118.0));
        s.draw_glyphs(&font, 16.0, true, &[], Vec2::new(0.6, 0.6), Fill::NonZero, Color::from_rgba8(160, 20, 20, 255), 0.8, id, None, glyphs(138.0));
        s.draw_glyphs(&font, 18.0, false, &[], Vec2::ZERO, Fill::NonZero, &g, 0.6, id, None, glyphs(160.0));
    }
    // a box shadow
    s.draw_box_shadow(id, Rect::new(30.0, 165.0, 90.0, 190.0), Color::from_rgba8(0, 0, 0, 120), 6.0, 4.0);
    // a blurred layer
    let blur: detour::base::Filter = [FilterEffect::from(GaussianBlur { std_deviation: 4.0 })].into_iter().collect();
    s.push_layer(Fill::NonZero, Mix::Normal, 1.0, id, &Rect::new(220.0, 100.0, 320.0, 200.0), Some(Arc::new(blur)), None);
    s.fill(Fill::NonZero, id, Color::from_rgba8(120, 0, 200, 255), None, &Rect::new(245.0, 125.0, 295.0, 175.0));
    s.pop_layer();
    // nested filter layers: a drop shadow around a blurred layer of a gradient (each drawn alone, inner first)
    let shadow: detour::base::Filter = [FilterEffect::from(DropShadow { dx: 3.0, dy: 4.0, std_deviation: 2.5, color: Color::from_rgba8(0, 0, 60, 160) })].into_iter().collect();
    let soft: detour::base::Filter = [FilterEffect::from(GaussianBlur { std_deviation: 1.5 })].into_iter().collect();
    s.push_layer(Fill::NonZero, Mix::Normal, 1.0, id, &Rect::new(150.0, 150.0, 220.0, 200.0), Some(Arc::new(shadow)), None);
    s.fill(Fill::NonZero, id, Color::from_rgba8(250, 200, 40, 255), None, &Circle::new((172.0, 170.0), 14.0));
    s.push_layer(Fill::NonZero, Mix::Normal, 1.0, id, &Rect::new(180.0, 155.0, 215.0, 195.0), Some(Arc::new(soft)), None);
    s.fill(Fill::NonZero, id, &g, None, &Rect::new(188.0, 162.0, 208.0, 188.0));
    s.pop_layer();
    s.pop_layer();
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // the reference: the same replay, rendered by vello_cpu
    let mut ctx = RenderContext::new(W, H);
    let mut resources = Resources::new();
    let mut cpu = CpuScene::new(&mut ctx, &mut resources);
    cpu.append_scene(scene(), Affine::IDENTITY);
    assert_eq!(cpu.refused(), None);
    let mut reference = Pixmap::new(W, H);
    ctx.render_with(&mut reference, &mut resources, RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });

    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mut pages = Pages::new(&mut cx, &codegen, [u32::from(W), u32::from(H)])?;
    let got = pages.draw(&mut cx, scene(), (W, H), &())?;

    let want = reference.data_as_u8_slice();
    let bad = (0..usize::from(W) * usize::from(H)).filter(|i| got[i * 4..][..4] != want[i * 4..][..4]).count();
    println!("{} pixels, {bad} differ from vello_cpu", usize::from(W) * usize::from(H));
    assert_eq!(bad, 0);

    // a GPU image (no bytes, as the fallback's canvases and WebGPU textures are): refused, not drawn blank
    let mut gpu = Scene::new();
    let empty = ImageBrush::new(ImageData { data: Blob::new(Arc::new(Vec::<u8>::new())), format: ImageFormat::Rgba8, alpha_type: ImageAlphaType::Alpha, width: 8, height: 8 });
    gpu.fill(Fill::NonZero, Affine::IDENTITY, empty.as_ref(), None, &Rect::new(0.0, 0.0, 8.0, 8.0));
    let refused = pages.draw(&mut cx, gpu, (8, 8), &()).err().map(|e| e.to_string());
    println!("a GPU image: {refused:?}");
    assert!(refused.is_some_and(|e| e.contains("GPU image")));

    // a colour matrix filter (vello_cpu does not run one: it would panic): refused
    let mut cm = Scene::new();
    let grey: detour::base::Filter = [FilterEffect::from(ColorMatrix([0.3, 0.6, 0.1, 0.0, 0.0, 0.3, 0.6, 0.1, 0.0, 0.0, 0.3, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))].into_iter().collect();
    cm.push_layer(Fill::NonZero, Mix::Normal, 1.0, Affine::IDENTITY, &Rect::new(0.0, 0.0, 8.0, 8.0), Some(Arc::new(grey)), None);
    cm.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(200, 0, 0, 255), None, &Rect::new(0.0, 0.0, 8.0, 8.0));
    cm.pop_layer();
    let refused = pages.draw(&mut cx, cm, (8, 8), &()).err().map(|e| e.to_string());
    println!("a colour matrix: {refused:?}");
    assert!(refused.is_some_and(|e| e.contains("colour matrix")));
    gpu_images(&mut cx, &mut pages)?;
    println!("ok");
    Ok(())
}

/// GPU images by id (`XeImages`).
struct Table(Vec<(u64, GpuTexels)>);

impl XeImages for Table {
    fn find(&self, id: u64) -> Option<GpuTexels> {
        self.0.iter().find(|(k, _)| *k == id).map(|(_, t)| *t)
    }
}

/// Two images painted every way (nearest, bilinear, bicubic; pad, repeat, reflect; scaled, rotated, tinted alpha):
/// `straight` (13 x 9, straight alpha) and `premul` (7 x 5, premultiplied).
fn images_scene(straight: &ImageBrush, premul: &ImageBrush) -> Scene {
    let mut s = Scene::new();
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(250, 245, 230, 255), None, &Rect::new(0.0, 0.0, 160.0, 120.0));
    let ways = [(ImageQuality::Low, Extend::Pad), (ImageQuality::Medium, Extend::Repeat), (ImageQuality::High, Extend::Reflect), (ImageQuality::Medium, Extend::Pad)];
    for (k, (quality, extend)) in ways.into_iter().enumerate() {
        let x = 4.0 + 39.0 * k as f64;
        for (row, brush) in [straight, premul].into_iter().enumerate() {
            let y = 4.0 + 58.0 * row as f64;
            let b = brush.clone().with_quality(quality).with_extend(extend);
            let place = Affine::translate((x + 3.5, y + 2.25)) * Affine::rotate(0.15 * k as f64) * Affine::scale(2.6);
            s.fill(Fill::NonZero, Affine::IDENTITY, b.as_ref(), Some(place), &Rect::new(x, y, x + 36.0, y + 54.0));
        }
    }
    // a GPU image in a blurred layer (the layer drawn on the GPU, where the image is)
    let blur: detour::base::Filter = [FilterEffect::from(GaussianBlur { std_deviation: 3.0 })].into_iter().collect();
    s.push_layer(Fill::NonZero, Mix::Normal, 0.9, Affine::IDENTITY, &Rect::new(120.0, 60.0, 160.0, 120.0), Some(Arc::new(blur)), None);
    s.fill(Fill::NonZero, Affine::IDENTITY, straight.as_ref(), Some(Affine::translate((126.0, 70.0)) * Affine::scale(2.0)), &Rect::new(126.0, 70.0, 152.0, 88.0));
    s.pop_layer();
    s
}

/// Images on the GPU (texels in an Xe buffer, rows padded; straight and premultiplied) drawn by id, against vello_cpu
/// drawing the same bytes as images.
fn gpu_images<'d>(cx: &mut Context<'d, Xe, Native>, pages: &mut Pages<'d, Xe>) -> Result<(), Box<dyn std::error::Error>> {
    let straight: Vec<[u8; 4]> = (0..13 * 9).map(|i: u32| [(i * 37 % 256) as u8, (i * 91 % 256) as u8, (255 - i * 13 % 256) as u8, [0, 40, 128, 200, 255][(i % 5) as usize]]).collect();
    let premul: Vec<[u8; 4]> = (0..7 * 5)
        .map(|i: u32| {
            let a = (i * 53 % 256) as u8;
            [a / 2, a / 3, a, a]
        })
        .collect();
    let bytes = |px: &[[u8; 4]]| px.concat();
    let data = |px: &[[u8; 4]], w, h, alpha_type| ImageData { data: Blob::new(Arc::new(bytes(px))), format: ImageFormat::Rgba8, alpha_type, width: w, height: h };
    let empty = |w, h, alpha_type| ImageData { data: Blob::new(Arc::new(Vec::<u8>::new())), format: ImageFormat::Rgba8, alpha_type, width: w, height: h };

    // the reference
    let reference = images_scene(&ImageBrush::new(data(&straight, 13, 9, ImageAlphaType::Alpha)), &ImageBrush::new(data(&premul, 7, 5, ImageAlphaType::AlphaPremultiplied)));
    let (w, h) = (160u16, 120u16);
    let mut ctx = RenderContext::new(w, h);
    let mut resources = Resources::new();
    CpuScene::new(&mut ctx, &mut resources).append_scene(reference, Affine::IDENTITY);
    let mut want = Pixmap::new(w, h);
    ctx.render_with(&mut want, &mut resources, RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });

    // the texels on the GPU: the straight image's rows 16 texels apart
    let mut padded = vec![[0xee; 4]; 16 * 9];
    for y in 0..9 {
        padded[y * 16..][..13].copy_from_slice(&straight[y * 13..][..13]);
    }
    let a = cx.buffer(&bytes(&padded))?;
    let b = cx.buffer(&bytes(&premul))?;
    let (sa, sb) = (empty(13, 9, ImageAlphaType::Alpha), empty(7, 5, ImageAlphaType::AlphaPremultiplied));
    let table = Table(vec![
        (sa.data.id(), GpuTexels { addr: a.addr, stride: 16, size: (13, 9), straight: true }),
        (sb.data.id(), GpuTexels { addr: b.addr, stride: 7, size: (7, 5), straight: false }),
    ]);
    let got = pages.draw(cx, images_scene(&ImageBrush::new(sa), &ImageBrush::new(sb)), (w, h), &table)?;
    let want = want.data_as_u8_slice();
    let bad = (0..usize::from(w) * usize::from(h)).filter(|i| got[i * 4..][..4] != want[i * 4..][..4]).count();
    let colours = got.as_chunks::<4>().0.iter().collect::<std::collections::HashSet<_>>().len();
    println!("GPU images: {} pixels ({colours} colours), {bad} differ from vello_cpu drawing their bytes", usize::from(w) * usize::from(h));
    assert!(bad == 0 && colours > 1000);
    Ok(())
}
