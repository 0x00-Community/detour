use detour::base::{Gpu, LUT_WIDTH, MemType, Native, StripPaint, Strips, TextureFormat};
use detour::xe3::{Codegen, Compositor, Context};
use detour::xe::Xe;
use detour::base::vello_common::clip::ClipContext;
use detour::base::vello_common::color::{AlphaColor, Srgb};
use detour::base::vello_common::encode::{EncodeExt, EncodedPaint};
use detour::base::vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, RoundedRect, Shape};
use detour::base::vello_common::paint::{self, PaintType};
use detour::base::vello_common::peniko::{Fill, Gradient};
use detour::base::vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};
use detour::vello_cpu::{Level, Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};

const W: u16 = 512;
const H: u16 = 384;

enum Item {
    Fill(BezPath, PaintType),
    Clip(BezPath),
    Pop,
}

fn scene() -> Vec<Item> {
    let mut star = BezPath::new();
    for i in 0..5 {
        let a = std::f64::consts::PI * (0.5 + 0.8 * f64::from(i));
        let p = Point::new(380.0 + 110.0 * a.cos(), 120.0 + 110.0 * a.sin());
        if i == 0 { star.move_to(p) } else { star.line_to(p) }
    }
    star.close_path();
    let mut sliver = BezPath::new();
    sliver.move_to((20.0, 360.0));
    sliver.line_to((500.0, 200.0));
    sliver.line_to((500.5, 201.3));
    sliver.close_path();
    let solid = |c: [f32; 4]| PaintType::from(AlphaColor::<Srgb>::new(c));
    let linear = |p0: (f64, f64), p1: (f64, f64), stops: &[(f32, [f32; 4])]| {
        let stops: Vec<(f32, AlphaColor<Srgb>)> = stops.iter().map(|(o, c)| (*o, AlphaColor::new(*c))).collect();
        PaintType::from(Gradient::new_linear(p0, p1).with_stops(&stops[..]))
    };
    use Item::{Clip, Fill, Pop};
    vec![
        Fill(Rect::new(0.0, 0.0, 512.0, 384.0).to_path(0.1), solid([0.95, 0.95, 0.9, 1.0])),
        Fill(Circle::new((130.0, 130.0), 100.3).to_path(0.1), solid([0.1, 0.4, 0.9, 1.0])),
        Fill(star, linear((300.0, 20.0), (460.0, 220.0), &[(0.0, [0.9, 0.7, 0.1, 1.0]), (0.5, [0.9, 0.2, 0.1, 1.0]), (1.0, [0.3, 0.1, 0.6, 1.0])])),
        Fill(RoundedRect::new(60.0, 220.0, 300.0, 360.0, 25.0).to_path(0.1), solid([0.2, 0.7, 0.3, 0.6])),
        Clip(Circle::new((250.0, 200.0), 90.0).to_path(0.1)),
        Fill(Rect::new(140.0, 100.0, 380.0, 300.0).to_path(0.1), linear((170.0, 0.0), (330.0, 0.0), &[(0.2, [0.8, 0.1, 0.3, 0.45]), (0.8, [0.1, 0.3, 0.8, 0.9])])),
        Clip(RoundedRect::new(200.0, 150.0, 330.0, 260.0, 30.0).to_path(0.1)),
        Fill(Rect::new(0.0, 0.0, 512.0, 384.0).to_path(0.1), linear((0.0, 384.0), (512.0, 0.0), &[(0.0, [1.0, 1.0, 1.0, 0.0]), (1.0, [0.0, 0.0, 0.0, 0.7])])),
        Pop,
        Pop,
        Fill(sliver, solid([0.0, 0.0, 0.0, 1.0])),
        Fill(Circle::new((420.7, 300.2), 50.5).to_path(0.1), linear((380.0, 260.0), (470.0, 340.0), &[(0.0, [0.5, 0.2, 0.8, 0.3]), (0.3, [0.9, 0.9, 0.2, 0.8]), (0.31, [0.1, 0.6, 0.6, 1.0]), (1.0, [0.5, 0.2, 0.8, 0.3])])),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ctx = RenderContext::new(W, H);
    for item in scene() {
        match item {
            Item::Fill(path, paint) => {
                ctx.set_paint(paint);
                ctx.fill_path(&path);
            }
            Item::Clip(path) => ctx.push_clip_path(&path),
            Item::Pop => ctx.pop_clip(),
        }
    }
    ctx.flush();
    let render = |mode| {
        let mut p = Pixmap::new(W, H);
        ctx.render_with(&mut p, &mut Resources::new(), RasterizerSettings { render_mode: mode, ..Default::default() });
        p.data_as_u8_slice().to_vec()
    };
    let reference = render(RenderMode::OptimizeQuality);
    let fast = render(RenderMode::OptimizeSpeed);

    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mut compositor = Compositor::new(&mut cx, &codegen, 1 << 16, 256, 16)?;
    let clear = cx.clear_pipeline(&codegen, [0.0; 4])?;
    let corners = cx.clear_vertices()?;
    let mut target = cx.target(u32::from(W), u32::from(H), TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let t = std::time::Instant::now();
    let (rows, lut_rows) = compositor.rows();
    let mut strips = Strips::new(u32::from(W), u32::from(H), rows, lut_rows);
    let mut generator = StripGenerator::new(W, H, Level::new());
    let mut storage = StripStorage::new(GenerationMode::Replace);
    let mut clips = ClipContext::new();
    let mut paints = Vec::new();
    for item in scene() {
        match item {
            Item::Fill(path, p) => {
                generator.generate_filled_path(&path, Fill::NonZero, Affine::IDENTITY, None, &mut storage, clips.get().map(|c| c.path));
                let encoded = match p {
                    PaintType::Gradient(g) => g.encode_into(&mut paints, Affine::IDENTITY, None),
                    PaintType::Solid(c) => Some(c.into()),
                    PaintType::Image(_) => None,
                };
                match encoded.ok_or("a paint without area")? {
                    paint::Paint::Solid(c) => strips.push(&storage, StripPaint::Solid(c.as_premul_f32()))?,
                    paint::Paint::Indexed(i) => match &paints[i.index()] {
                        EncodedPaint::Gradient(g) => strips.push(&storage, StripPaint::Gradient(g))?,
                        _ => return Err("not a gradient".into()),
                    },
                }
            }
            Item::Clip(path) => clips.push_clip_path(&path, &mut generator, Fill::NonZero, Affine::IDENTITY, None),
            Item::Pop => clips.pop_clip(),
        }
    }
    let cpu = t.elapsed();
    let slot = cx.slot();
    cx.wait(slot)?;
    compositor.upload(slot, &strips)?;
    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    compositor.draw(&mut r, slot)?;
    let t = std::time::Instant::now();
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    println!("{} instances; strips + instances on the CPU {cpu:?}, GPU submit to done {:?}", strips.instances.len(), t.elapsed());

    let ours = target.bytes()?.to_vec();
    let compare = |name: &str, other: &[u8]| {
        let (mut diff, mut worst) = ([0usize; 3], 0u8);
        for (a, b) in ours.iter().zip(other) {
            let d = a.abs_diff(*b);
            worst = worst.max(d);
            diff[usize::from(d.min(2))] += 1;
        }
        println!("vs {name}: {} channels, {} equal, {} off by one, {} off by more (worst {worst})", ours.len(), diff[0], diff[1], diff[2]);
        diff
    };
    let (w, h) = (usize::from(W), usize::from(H));
    let mut model = vec![0.0f64; w * h * 4];
    let mut tie = vec![1.0f64; w * h * 4];
    let mut edge = vec![false; w * h * 4];
    let near = |v: f64| (v - v.floor() - 0.5).abs();
    let aw = detour::base::ATLAS_WIDTH as f64;
    let ah = f64::from(rows);
    let lw = f64::from(LUT_WIDTH);
    for inst in &strips.instances {
        let x0 = ((f64::from(inst.rect[0]) + 1.0) / 2.0 * w as f64).round() as usize;
        let y0 = ((f64::from(inst.rect[1]) + 1.0) / 2.0 * h as f64).round() as usize;
        let n = (f64::from(inst.rect[2]) / 2.0 * w as f64).round() as usize;
        let (c0, r0) = ((f64::from(inst.uv[0]) * aw).floor() as usize, (f64::from(inst.uv[1]) * ah).floor() as usize);
        let solid = inst.uv[0] == inst.uv[2];
        let [u0, du, dv, pv] = inst.paint.map(f64::from);
        let row = (pv * f64::from(lut_rows)).floor() as usize;
        for dy in 0..4 {
            for dx in 0..n {
                let (x, y) = (x0 + dx, y0 + dy);
                let (c, r) = if solid { (0, 0) } else { (c0 + dx, r0 + dy) };
                let cov = f64::from(strips.atlas[r * detour::base::ATLAS_WIDTH as usize + c]) / 255.0;
                let u = (u0 + du * (dx as f64 + 0.5) / n as f64 + dv * (dy as f64 + 0.5) / 4.0) * lw;
                let at_edge = (du != 0.0 || dv != 0.0) && (u - u.round()).abs() < 0.01;
                let texel = (u.floor().clamp(0.0, lw - 1.0) as usize + row * LUT_WIDTH as usize) * 4;
                let raw: Vec<f64> = (0..4).map(|i| f64::from(strips.lut[texel + i]) / 255.0 * f64::from(inst.color[i]) * cov * 255.0).collect();
                let src: Vec<f64> = raw.iter().map(|v| v.round()).collect();
                for i in 0..4 {
                    let at = (y * w + x) * 4 + i;
                    let blended = src[i] + model[at] * (255.0 - src[3]) / 255.0;
                    tie[at] = tie[at].min(near(raw[i])).min(near(raw[3])).min(near(blended));
                    edge[at] |= at_edge;
                    model[at] = blended.round();
                }
            }
        }
    }
    let model: Vec<u8> = model.iter().map(|v| *v as u8).collect();
    compare("the model (source and result rounded to unorm8 at each layer)", &model);
    let (mut at_edge, mut at_tie, mut wrong) = (0, 0, 0);
    for (i, (a, b)) in ours.iter().zip(&model).enumerate().filter(|(_, (a, b))| a != b) {
        match () {
            _ if edge[i] => at_edge += 1,
            _ if a.abs_diff(*b) == 1 && tie[i] <= 0.01 => at_tie += 1,
            _ => wrong += 1,
        }
    }
    println!("  off the model: {at_edge} at a paint texel's edge, {at_tie} by one at a rounding tie, {wrong} otherwise");
    compare("vello_cpu u8 (OptimizeSpeed)", &fast);
    compare("vello_cpu f32 (OptimizeQuality)", &reference);
    assert_eq!(wrong, 0, "off the model");
    let worst = ours.iter().zip(&reference).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
    assert!(worst <= 2, "off vello_cpu's f32 pipeline by more than the layers' rounding: {worst}");
    println!("ok");
    Ok(())
}
