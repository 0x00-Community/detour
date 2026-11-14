use std::sync::Arc;

use detour::base::{Content, Effect, Gpu, Native, PaintScene, Part, Pass, Px, Scene};
use detour::{PageRaster, SharedImages};
use detour::fallback::{Device, Exported, GpuImage, Offscreen, Overrides, Raster, Shared};
use detour::xe3::{Codegen, Context, CpuScene, GpuTexels, Pages, XeImages};
use detour::xe::Xe;
use detour::vello_cpu::kurbo::{Affine, Circle, Rect, Stroke};
use detour::vello_cpu::peniko::{Blob, Color, Extend, Fill, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageQuality};
use detour::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, RenderMode, Resources};

const IW: u32 = 13;
const IH: u32 = 9;

struct One(u64, GpuTexels);

impl XeImages for One {
    fn find(&self, id: u64) -> Option<GpuTexels> {
        (id == self.0).then_some(self.1)
    }
}

fn scene(image: &ImageBrush) -> Scene {
    let mut s = Scene::new();
    s.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(240, 250, 230, 255), None, &Rect::new(0.0, 0.0, 96.0, 64.0));
    for (k, q) in [ImageQuality::Low, ImageQuality::Medium, ImageQuality::High].into_iter().enumerate() {
        let x = 2.0 + 31.0 * k as f64;
        let b = image.clone().with_quality(q).with_extend(Extend::Repeat);
        s.fill(Fill::NonZero, Affine::IDENTITY, b.as_ref(), Some(Affine::translate((x, 3.0)) * Affine::rotate(0.2) * Affine::scale(2.3)), &Rect::new(x, 2.0, x + 29.0, 62.0));
    }
    s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let texels: Vec<u8> = (0..IW * IH).flat_map(|i| [(i * 37 % 256) as u8, (i * 91 % 256) as u8, (255 - i * 13 % 256) as u8, [0, 60, 128, 200, 255][(i % 5) as usize]]).collect();

    let vk = Device::open()?;
    let (texture, exported) = Exported::texture(&vk.device, (IW, IH))?;
    println!("{}: exported {IW} x {IH}, offset {}, rows of {} bytes", vk.info.name, exported.offset, exported.stride);
    vk.queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &texels,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(IW * 4), rows_per_image: Some(IH) },
        wgpu::Extent3d { width: IW, height: IH, depth_or_array_layers: 1 },
    );
    vk.queue.submit([]);
    vk.device.poll(wgpu::PollType::wait_indefinitely())?;

    let xe = Xe::open_first()?;
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mut pages = Pages::new(&mut cx, &Codegen::default(), [96, 64])?;
    let d = exported.dmabuf();
    let imported = cx.import_canvas(d.fd, d.offset, d.stride, [IW, IH])?;
    let gpu = ImageData { data: Blob::new(Arc::new(Vec::<u8>::new())), format: ImageFormat::Rgba8, alpha_type: ImageAlphaType::Alpha, width: IW, height: IH };
    let table = One(gpu.data.id(), GpuTexels { addr: imported.canvas.addr, stride: imported.canvas.stride, size: (IW as u16, IH as u16), straight: true });
    let got = pages.draw(&mut cx, scene(&ImageBrush::new(gpu)), (96, 64), &table)?;

    let bytes = ImageData { data: Blob::new(Arc::new(texels)), format: ImageFormat::Rgba8, alpha_type: ImageAlphaType::Alpha, width: IW, height: IH };
    let mut ctx = RenderContext::new(96, 64);
    let mut resources = Resources::new();
    CpuScene::new(&mut ctx, &mut resources).append_scene(scene(&ImageBrush::new(bytes)), Affine::IDENTITY);
    let mut want = Pixmap::new(96, 64);
    ctx.render_with(&mut want, &mut resources, RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let want = want.data_as_u8_slice();
    let bad = got.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0).filter(|(g, w)| g != w).count();
    let colours = got.as_chunks::<4>().0.iter().collect::<std::collections::HashSet<_>>().len();
    println!("{} pixels ({colours} colours), {bad} differ from vello_cpu", 96 * 64);
    assert!(bad == 0 && colours > 500);

    offscreen(&vk, &mut cx, &mut pages)?;
    drop(vk);
    page_raster(&xe)?;
    println!("ok");
    Ok(())
}

fn offscreen<'d>(vk: &Device, cx: &mut Context<'d, Xe, Native>, pages: &mut Pages<'d, Xe>) -> Result<(), Box<dyn std::error::Error>> {
    let shared = Shared::exporting(vk.device.clone(), vk.queue.clone());
    let mut off = Offscreen::new(shared.clone())?;
    let mut drawn = Scene::new();
    drawn.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(200, 40, 90, 180), None, &Circle::new((11.0, 8.0), 7.5));
    drawn.stroke(&Stroke::new(1.5), Affine::IDENTITY, Color::from_rgba8(20, 120, 255, 255), None, &Rect::new(2.0, 2.0, 20.0, 14.0));
    let image = off.draw(drawn, (IW + 9, IH + 7), Affine::IDENTITY)?;
    vk.device.poll(wgpu::PollType::wait_indefinitely())?;
    let exported = image.exported().ok_or("the image is not exported")?;
    println!("offscreen: {} x {}, rows of {} bytes", image.texture.width(), image.texture.height(), exported.stride);
    let mut images = SharedImages::new(shared.images.clone());
    let failed = images.refresh(cx);
    assert!(failed.is_empty(), "{failed:?}");
    let got = pages.draw(cx, scene(&image.brush()), (96, 64), &images)?;

    let (w, h) = (image.texture.width(), image.texture.height());
    let bytes = shared.read(&image.texture, [0, 0, w, h])?;
    let data = ImageData { data: Blob::new(Arc::new(bytes)), format: ImageFormat::Rgba8, alpha_type: ImageAlphaType::Alpha, width: w, height: h };
    let mut ctx = RenderContext::new(96, 64);
    let mut resources = Resources::new();
    CpuScene::new(&mut ctx, &mut resources).append_scene(scene(&ImageBrush::new(data)), Affine::IDENTITY);
    let mut want = Pixmap::new(96, 64);
    ctx.render_with(&mut want, &mut resources, RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let bad = got.as_chunks::<4>().0.iter().zip(want.data_as_u8_slice().as_chunks::<4>().0).filter(|(g, w)| g != w).count();
    let colours = got.as_chunks::<4>().0.iter().collect::<std::collections::HashSet<_>>().len();
    println!("offscreen: {} pixels ({colours} colours), {bad} differ from vello_cpu drawing it read back", 96 * 64);
    assert!(bad == 0 && colours > 100);
    Ok(())
}

struct One1(Scene);

impl Content for One1 {
    fn scene_for(&self, _: Px) -> Scene {
        self.0.clone()
    }
}

fn page_raster(xe: &Xe) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = (96u32, 64u32);
    let vulkan = Raster::new(Device::open()?, false)?;
    let (device, queue) = (vulkan.device.device.clone(), vulkan.device.queue.clone());
    let shared = Shared::exporting(device.clone(), queue.clone());
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("interop unexported"),
        size: wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let disc: Vec<u8> = (0..256).flat_map(|i: i32| if (i % 16 - 8).pow(2) + (i / 16 - 8).pow(2) < 36 { [30, 160, 90, 255] } else { [0; 4] }).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &disc,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(64), rows_per_image: Some(16) },
        wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 },
    );
    let image = GpuImage::new(texture, &shared.images);
    let mut raster = PageRaster::new(vulkan, Some(xe), [w, h], shared.images.clone());
    assert!(raster.xe3(), "Xe3's rasterizer did not start");
    let padded = (w + 32, h + 16);
    let canvases: Vec<(wgpu::Texture, Exported)> = (0..2).map(|_| Exported::texture(&device, padded)).collect::<Result<_, _>>()?;
    for (plane, (_, e)) in canvases.iter().enumerate() {
        let results = raster.canvases(plane, &[e.dmabuf()], padded);
        assert!(results.iter().all(Result::is_ok), "{results:?}");
    }
    let plain = scene(&ImageBrush::new(ImageData { data: Blob::new(Arc::new(vec![200; (IW * IH * 4) as usize])), format: ImageFormat::Rgba8, alpha_type: ImageAlphaType::Alpha, width: IW, height: IH }));
    let mut filtered = Scene::new();
    filtered.fill(Fill::NonZero, Affine::IDENTITY, image.brush().as_ref(), Some(Affine::translate((10.0, 10.0))), &Rect::new(10.0, 10.0, 26.0, 26.0));
    let part = |s: &Scene| Part { content: One1(s.clone()), backdrop: None, shadows: Vec::new(), effect: Effect::default(), bounds: None };
    let passes = vec![Pass { plane: 0, size: (w, h), parts: vec![part(&plain)], next: 0, targets: None }, Pass { plane: 1, size: (w, h), parts: vec![part(&filtered)], next: 0, targets: None }];
    let mut overrides = Overrides::from(shared.images.clone());
    let submitted = raster.draw(passes, &mut overrides, Some(&shared))?;
    let refused = raster.refused.clone();
    println!("page raster: the fallback drew what Xe3 refused: {refused:?}");
    assert!(refused.is_some_and(|r| r.contains("not on Xe3")));
    let fence = submitted.fence.ok_or("no sync_file")?;
    let mut fds = [rustix::event::PollFd::new(&fence, rustix::event::PollFlags::IN)];
    let ready = rustix::event::poll(&mut fds, Some(&rustix::event::Timespec { tv_sec: 5, tv_nsec: 0 }))?;
    device.poll(wgpu::PollType::wait_indefinitely())?;

    let got = shared.read(&canvases[0].0, [0, 0, w, h])?;
    let mut ctx = RenderContext::new(w as u16, h as u16);
    let mut resources = Resources::new();
    CpuScene::new(&mut ctx, &mut resources).append_scene(plain, Affine::IDENTITY);
    let mut want = Pixmap::new(w as u16, h as u16);
    ctx.render_with(&mut want, &mut resources, RasterizerSettings { render_mode: RenderMode::OptimizeSpeed, ..Default::default() });
    let bad = got.as_chunks::<4>().0.iter().zip(want.data_as_u8_slice().as_chunks::<4>().0).filter(|(g, w)| g != w).count();
    let other = shared.read(&canvases[1].0, [0, 0, w, h])?;
    let green = other.as_chunks::<4>().0.iter().filter(|p| p[3] > 0 && p[1] > p[0]).count();
    println!("page raster: sync_file signalled {}; plane 0 (Xe3, a {w}x{h} pass in a {}x{} canvas) {bad} of {} pixels differ from vello_cpu; plane 1 (fallback) {green} green pixels", ready == 1, padded.0, padded.1, w * h);
    assert!(ready == 1 && bad == 0 && green > 100);
    Ok(())
}
