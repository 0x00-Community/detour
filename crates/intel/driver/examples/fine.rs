use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context, Fine};
use detour::xe::Xe;
use detour::vello_cpu::color::{AlphaColor, Srgb};
use detour::vello_cpu::kurbo::{Affine, BezPath, Circle, Point, Rect, RoundedRect, Shape};
use detour::vello_cpu::peniko::{BlendMode, Compose, Extend, Gradient, ImageQuality, ImageSampler, Mix};
use detour::vello_cpu::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use detour::vello_cpu::{Image, ImageSource, Mask, Pixmap, Tint, TintMode, RasterizerSettings, RenderContext, RenderMode, Resources};

const W: u16 = 512;
const H: u16 = 256;

fn scene() -> RenderContext {
    let mut ctx = RenderContext::new(W, H);
    let rgba = |r, g, b, a| AlphaColor::<Srgb>::from_rgba8(r, g, b, a);
    ctx.set_paint(rgba(0, 200, 80, 255));
    ctx.fill_path(&Circle::new((300.0, 120.0), 90.3).to_path(0.1));
    ctx.set_paint(rgba(40, 40, 40, 255));
    ctx.fill_rect(&Rect::new(128.0, 40.0, 384.0, 200.0));
    ctx.set_paint(rgba(255, 128, 0, 255));
    ctx.fill_path(&Circle::new((60.5, 70.25), 41.7).to_path(0.1));
    let mut star = BezPath::new();
    for k in 0..10 {
        let r = if k % 2 == 0 { 70.0 } else { 28.0 };
        let t = std::f64::consts::PI * f64::from(k) / 5.0 - std::f64::consts::FRAC_PI_2;
        let p = Point::new(420.0 + r * t.cos(), 150.0 + r * t.sin());
        if k == 0 { star.move_to(p) } else { star.line_to(p) }
    }
    star.close_path();
    ctx.set_paint(rgba(30, 90, 255, 230));
    ctx.fill_path(&star);
    for (i, a) in [100u8, 160, 220].into_iter().enumerate() {
        ctx.set_paint(rgba(255, 40 * i as u8, 200, a));
        let x = 150.0 + 45.3 * i as f64;
        ctx.fill_rect(&Rect::new(x, 90.5 + 10.0 * i as f64, x + 80.0, 230.7));
    }
    ctx.push_clip_layer(&RoundedRect::new(10.3, 140.0, 120.0, 250.0, 17.0).to_path(0.1));
    ctx.set_paint(rgba(250, 230, 20, 255));
    ctx.fill_path(&Circle::new((70.0, 200.0), 60.0).to_path(0.1));
    ctx.pop_layer();
    ctx.push_opacity_layer(0.6);
    ctx.set_paint(rgba(0, 220, 255, 255));
    ctx.fill_rect(&Rect::new(380.0, 10.5, 500.0, 100.0));
    ctx.set_paint(rgba(255, 0, 60, 180));
    ctx.fill_path(&Circle::new((470.0, 60.0), 35.0).to_path(0.1));
    ctx.pop_layer();
    ctx.push_clip_layer(&Rect::new(200.0, 0.0, 330.0, 80.0).to_path(0.1));
    ctx.push_clip_layer(&Circle::new((260.0, 40.0), 45.5).to_path(0.1));
    ctx.push_opacity_layer(0.3);
    ctx.set_paint(rgba(120, 255, 120, 255));
    ctx.fill_rect(&Rect::new(150.0, 0.0, 400.0, 100.0));
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();
    let stops = |a: [u8; 4], b: [u8; 4]| [AlphaColor::<Srgb>::from_rgba8(a[0], a[1], a[2], a[3]), AlphaColor::<Srgb>::from_rgba8(b[0], b[1], b[2], b[3])];
    ctx.set_paint(Gradient::new_linear(Point::new(10.0, 0.0), Point::new(120.0, 0.0)).with_stops(stops([255, 0, 0, 255], [0, 0, 255, 255]).as_slice()));
    ctx.fill_rect(&Rect::new(0.0, 100.0, 130.0, 130.0));
    ctx.set_paint_transform(Affine::rotate(0.6));
    ctx.set_paint(Gradient::new_linear(Point::new(0.0, 0.0), Point::new(23.0, 0.0)).with_extend(Extend::Repeat).with_stops(stops([0, 255, 0, 200], [255, 255, 0, 40]).as_slice()));
    ctx.fill_path(&Circle::new((330.0, 200.0), 50.0).to_path(0.1));
    ctx.reset_paint_transform();
    ctx.push_opacity_layer(0.75);
    ctx.set_paint(Gradient::new_linear(Point::new(400.0, 180.0), Point::new(440.0, 250.0)).with_extend(Extend::Reflect).with_stops(stops([255, 255, 255, 255], [0, 0, 0, 255]).as_slice()));
    ctx.fill_rect(&Rect::new(390.0, 170.0, 510.0, 256.0));
    ctx.pop_layer();
    ctx.set_paint(Gradient::new_linear(Point::new(0.0, 0.0), Point::new(512.0, 0.0)).with_stops(stops([10, 20, 30, 255], [200, 100, 50, 255]).as_slice()));
    ctx.fill_rect(&Rect::new(0.0, 240.0, 512.0, 256.0));
    ctx.set_paint(Gradient::new_radial(Point::new(200.0, 60.0), 45.0).with_stops(stops([255, 255, 255, 255], [255, 0, 128, 90]).as_slice()));
    ctx.fill_rect(&Rect::new(150.0, 10.0, 250.0, 110.0));
    ctx.set_paint(Gradient::new_two_point_radial(Point::new(455.0, 200.0), 5.0, Point::new(440.0, 190.0), 40.0).with_extend(Extend::Repeat).with_stops(stops([0, 0, 255, 255], [255, 200, 0, 255]).as_slice()));
    ctx.fill_path(&Circle::new((440.0, 190.0), 50.0).to_path(0.1));
    ctx.set_paint(Gradient::new_two_point_radial(Point::new(90.0, 40.0), 0.0, Point::new(70.0, 50.0), 30.0).with_extend(Extend::Reflect).with_stops(stops([0, 128, 0, 255], [240, 240, 240, 255]).as_slice()));
    ctx.fill_rect(&Rect::new(20.0, 10.0, 140.0, 95.0));
    ctx.set_paint(Gradient::new_sweep(Point::new(300.0, 140.0), 0.0, std::f32::consts::TAU).with_stops(stops([255, 0, 0, 255], [0, 0, 255, 255]).as_slice()));
    ctx.fill_path(&Circle::new((300.0, 140.0), 38.0).to_path(0.1));
    ctx.set_paint(Gradient::new_sweep(Point::new(230.5, 200.25), 0.3, 1.9).with_extend(Extend::Repeat).with_stops(stops([255, 255, 0, 220], [0, 255, 255, 120]).as_slice()));
    ctx.fill_rect(&Rect::new(190.0, 160.0, 270.0, 240.0));
    let mut pixmap = Pixmap::new(7, 5);
    for (i, px) in pixmap.data_as_u8_slice_mut().as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let a = [255u8, 200, 128, 60][i % 4];
        let c = |k: usize| ((i * 37 + k * 91) % 256) as u8 % (a.saturating_add(1)).max(1);
        px.copy_from_slice(&[c(0), c(1), c(2), a]);
    }
    let pixmap = std::sync::Arc::new(pixmap);
    let image = |x, y| Image { image: ImageSource::Pixmap(pixmap.clone()), sampler: ImageSampler { x_extend: x, y_extend: y, quality: ImageQuality::Low, ..Default::default() } };
    ctx.set_paint_transform(Affine::translate((30.0, 150.0)) * Affine::scale(9.0));
    ctx.set_paint(image(Extend::Pad, Extend::Pad));
    ctx.fill_rect(&Rect::new(10.0, 140.0, 120.0, 210.0));
    ctx.set_paint_transform(Affine::translate((150.0, 30.0)) * Affine::rotate(0.4) * Affine::scale(5.5));
    ctx.set_paint(image(Extend::Repeat, Extend::Repeat));
    ctx.fill_path(&Circle::new((180.0, 70.0), 45.0).to_path(0.1));
    ctx.set_paint_transform(Affine::translate((420.3, 20.7)) * Affine::scale(4.0));
    ctx.set_paint(image(Extend::Reflect, Extend::Repeat));
    ctx.fill_rect(&Rect::new(400.0, 10.0, 500.0, 90.0));
    let smooth = |x, y| Image { image: ImageSource::Pixmap(pixmap.clone()), sampler: ImageSampler { x_extend: x, y_extend: y, quality: ImageQuality::Medium, ..Default::default() } };
    ctx.set_paint_transform(Affine::translate((262.0, 100.0)) * Affine::scale(11.3));
    ctx.set_paint(smooth(Extend::Pad, Extend::Pad));
    ctx.fill_rect(&Rect::new(250.0, 90.0, 360.0, 170.0));
    ctx.set_paint_transform(Affine::translate((90.0, 200.0)) * Affine::rotate(-0.7) * Affine::scale(6.0));
    ctx.set_paint(smooth(Extend::Repeat, Extend::Reflect));
    ctx.fill_path(&Circle::new((90.0, 205.0), 40.0).to_path(0.1));
    ctx.set_paint_transform(Affine::translate((370.0, 150.0)) * Affine::scale(3.7));
    ctx.set_paint(smooth(Extend::Reflect, Extend::Reflect));
    ctx.fill_rect(&Rect::new(360.0, 140.0, 450.0, 230.0));
    let cubic = |x, y| Image { image: ImageSource::Pixmap(pixmap.clone()), sampler: ImageSampler { x_extend: x, y_extend: y, quality: ImageQuality::High, ..Default::default() } };
    ctx.set_paint_transform(Affine::translate((5.0, 5.0)) * Affine::scale(8.6));
    ctx.set_paint(cubic(Extend::Pad, Extend::Pad));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 70.0, 50.0));
    ctx.set_paint_transform(Affine::translate((470.0, 120.0)) * Affine::rotate(1.1) * Affine::scale(4.4));
    ctx.set_paint(cubic(Extend::Repeat, Extend::Repeat));
    ctx.fill_path(&Circle::new((475.0, 120.0), 34.0).to_path(0.1));
    ctx.reset_paint_transform();
    let mixes = [Mix::Multiply, Mix::Screen, Mix::Overlay, Mix::Darken, Mix::Lighten, Mix::HardLight, Mix::Difference, Mix::Exclusion];
    for (i, mix) in mixes.into_iter().enumerate() {
        ctx.set_blend_mode(BlendMode::new(mix, Compose::SrcOver));
        ctx.set_paint(rgba(40 + 25 * i as u8, 200 - 20 * i as u8, 90, 160 + 10 * i as u8));
        let x = 8.0 + 62.5 * i as f64;
        ctx.fill_path(&Circle::new((x + 25.0, 120.0), 28.3).to_path(0.1));
    }
    let fmixes = [Mix::ColorDodge, Mix::ColorBurn, Mix::SoftLight, Mix::Hue, Mix::Saturation, Mix::Color, Mix::Luminosity];
    for (i, mix) in fmixes.into_iter().enumerate() {
        ctx.set_blend_mode(BlendMode::new(mix, Compose::SrcOver));
        ctx.set_paint(rgba(200 - 25 * i as u8, 60 + 25 * i as u8, 150, 190 + 9 * i as u8));
        let x = 12.0 + 71.0 * i as f64;
        ctx.fill_path(&Circle::new((x + 30.0, 236.0), 22.6).to_path(0.1));
    }
    let composes = [Compose::Xor, Compose::SrcAtop, Compose::DestOut, Compose::Plus, Compose::DestOver, Compose::SrcIn];
    for (i, compose) in composes.into_iter().enumerate() {
        ctx.set_blend_mode(BlendMode::new(Mix::Normal, compose));
        ctx.set_paint(rgba(230, 60 + 30 * i as u8, 20, 200));
        let x = 20.0 + 80.0 * i as f64;
        ctx.fill_rect(&Rect::new(x, 172.5, x + 50.7, 199.0));
    }
    ctx.set_blend_mode(BlendMode::default());
    ctx.push_blend_layer(BlendMode::new(Mix::Multiply, Compose::SrcOver));
    ctx.set_paint(rgba(255, 180, 40, 255));
    ctx.fill_path(&Circle::new((256.0, 40.0), 33.0).to_path(0.1));
    ctx.set_paint(rgba(40, 120, 255, 200));
    ctx.fill_rect(&Rect::new(240.0, 20.0, 300.0, 70.0));
    ctx.pop_layer();
    let bytes: Vec<u8> = (0..200u32).flat_map(|y| (0..300u32).map(move |x| if (x / 9 + y / 7) % 3 == 0 { 0 } else { ((x * 255 / 299 + y) % 256) as u8 })).collect();
    let mask = Mask::from_parts(bytes, 300, 200);
    ctx.set_mask(mask.clone());
    ctx.set_paint(rgba(255, 255, 255, 230));
    ctx.fill_path(&Circle::new((280.0, 160.0), 60.0).to_path(0.1));
    ctx.set_blend_mode(BlendMode::new(Mix::Screen, Compose::SrcOver));
    ctx.set_paint(rgba(0, 90, 200, 255));
    ctx.fill_rect(&Rect::new(10.0, 60.0, 120.0, 120.0));
    ctx.set_blend_mode(BlendMode::default());
    ctx.reset_mask();
    ctx.set_paint_transform(Affine::translate((180.0, 205.0)) * Affine::scale(6.0));
    ctx.set_paint(image(Extend::Repeat, Extend::Repeat));
    ctx.set_tint(Some(Tint { color: AlphaColor::<Srgb>::from_rgba8(255, 120, 0, 200), mode: TintMode::AlphaMask }));
    ctx.fill_rect(&Rect::new(170.0, 200.0, 250.0, 250.0));
    ctx.set_paint(smooth(Extend::Pad, Extend::Reflect));
    ctx.set_tint(Some(Tint { color: AlphaColor::<Srgb>::from_rgba8(100, 255, 200, 255), mode: TintMode::Multiply }));
    ctx.fill_rect(&Rect::new(260.0, 200.0, 340.0, 250.0));
    ctx.set_tint(None);
    ctx.reset_paint_transform();
    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur { std_deviation: 3.5, edge_mode: EdgeMode::None }));
    ctx.set_paint(rgba(255, 60, 160, 255));
    ctx.fill_path(&Circle::new((400.0, 220.0), 22.0).to_path(0.1));
    ctx.pop_layer();
    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadow { dx: 4.0, dy: 5.0, std_deviation: 2.5, color: AlphaColor::<Srgb>::from_rgba8(0, 0, 0, 160), edge_mode: EdgeMode::None }));
    ctx.set_paint(rgba(250, 250, 120, 255));
    ctx.fill_rect(&Rect::new(440.0, 200.0, 490.0, 240.0));
    ctx.pop_layer();
    ctx.set_paint(rgba(0, 0, 0, 170));
    ctx.fill_blurred_rounded_rect(&Rect::new(130.0, 200.0, 230.0, 245.0), 12.0, 6.5, false);
    ctx.set_transform(Affine::rotate_about(0.3, Point::new(300.0, 50.0)));
    ctx.set_paint(rgba(30, 80, 220, 255));
    ctx.fill_blurred_rounded_rect(&Rect::new(270.0, 25.0, 330.0, 75.0), 4.0, 1.8, false);
    ctx.reset_transform();
    ctx.set_paint(rgba(200, 40, 40, 140));
    ctx.fill_blurred_rounded_rect(&Rect::new(20.0, 20.0, 80.0, 60.0), 9.0, 3.0, true);
    ctx.push_mask_layer(mask);
    ctx.set_paint(rgba(200, 0, 200, 255));
    ctx.fill_rect(&Rect::new(150.0, 150.0, 400.0, 250.0));
    ctx.pop_layer();
    ctx
}

fn undefined() -> RenderContext {
    let mut ctx = RenderContext::new(256, 128);
    let stops = |a: [u8; 4], b: [u8; 4]| [AlphaColor::<Srgb>::from_rgba8(a[0], a[1], a[2], a[3]), AlphaColor::<Srgb>::from_rgba8(b[0], b[1], b[2], b[3])];
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(30, 30, 40, 255));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 256.0, 128.0));
    let cases = [
        (Point::new(30.0, 40.0), 18.0, Point::new(80.0, 44.0), 18.0, Extend::Pad, Rect::new(0.0, 0.0, 128.0, 64.0)),
        (Point::new(150.0, 30.0), 0.0, Point::new(180.0, 30.0), 30.0, Extend::Repeat, Rect::new(128.0, 0.0, 256.0, 64.0)),
        (Point::new(20.0, 100.0), 0.0, Point::new(70.0, 96.0), 15.0, Extend::Pad, Rect::new(0.0, 64.0, 90.0, 128.0)),
        (Point::new(140.0, 96.0), 25.0, Point::new(170.0, 104.0), 6.0, Extend::Reflect, Rect::new(90.0, 64.0, 200.0, 128.0)),
        (Point::new(230.0, 96.0), 30.0, Point::new(236.0, 100.0), 8.5, Extend::Repeat, Rect::new(200.0, 64.0, 256.0, 128.0)),
    ];
    for (k, (c0, r0, c1, r1, extend, area)) in cases.into_iter().enumerate() {
        let k = k as u8;
        let g = Gradient::new_two_point_radial(c0, r0, c1, r1).with_extend(extend).with_stops(stops([255, 40 * k, 0, 255], [0, 200, 255 - 30 * k, 140 + 20 * k]).as_slice());
        ctx.set_paint(g);
        ctx.fill_rect(&area);
    }
    ctx
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = scene();
    let mut reference = Pixmap::new(W, H);
    ctx.render_with(&mut reference, &mut Resources::new(), RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let coarse = ctx.coarse(&mut Resources::new()).ok_or("multi-threaded")?;
    let depth: usize = coarse.rows.iter().map(|r| r.depth.len()).sum();
    let cmds: usize = coarse.rows.iter().map(|r| r.cmds.len()).sum();
    println!("{} rows, {cmds} render commands, {depth} depth commands, {} paints, {} layers", coarse.rows.len(), coarse.paints.len(), coarse.layers.len());

    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let fine = Fine::new(&mut cx, &codegen)?;
    let t = std::time::Instant::now();
    let got = fine.draw(&mut cx, &coarse)?;
    println!("drawn in {:?}", t.elapsed());

    let want = reference.data_as_u8_slice();
    let bad: Vec<(usize, usize, &[u8], &[u8])> =
        (0..usize::from(W) * usize::from(H)).filter(|i| got[i * 4..][..4] != want[i * 4..][..4]).map(|i| (i % usize::from(W), i / usize::from(W), &got[i * 4..][..4], &want[i * 4..][..4])).collect();
    println!("{} pixels, {} differ from vello_cpu", usize::from(W) * usize::from(H), bad.len());
    assert!(bad.is_empty(), "differing (x, y, got, want) {:?}", &bad[..bad.len().min(8)]);

    let ctx = undefined();
    let mut reference = Pixmap::new(256, 128);
    ctx.render_with(&mut reference, &mut Resources::new(), RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let got = fine.draw(&mut cx, &ctx.coarse(&mut Resources::new()).ok_or("multi-threaded")?)?;
    let want = reference.data_as_u8_slice();
    let bad: Vec<(usize, usize, &[u8], &[u8])> = (0..256 * 128).filter(|i| got[i * 4..][..4] != want[i * 4..][..4]).map(|i| (i % 256, i / 256, &got[i * 4..][..4], &want[i * 4..][..4])).collect();
    let holes = want.as_chunks::<4>().0.iter().filter(|p| **p == [30, 30, 40, 255]).count();
    println!("undefined radial gradients: {} pixels ({holes} where undefined), {} differ from vello_cpu", 256 * 128, bad.len());
    assert!(bad.is_empty(), "differing (x, y, got, want) {:?}", &bad[..bad.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
