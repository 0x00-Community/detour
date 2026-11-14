use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 256;
const H: u32 = 192;
const COLORS: [[u8; 4]; 3] = [[255, 255, 255, 255], [128, 0, 255, 255], [255, 128, 0, 255]];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let src = include_str!("stencil/passes.mlir");
    let [mask, incr, eq0, eq1, eq2, sided] = ["mask", "incr", "eq0", "eq1", "eq2", "sided"].map(|n| cx.pipeline(&codegen, src, n));
    let (mask, incr, eq, sided) = (mask?, incr?, [eq0?, eq1?, eq2?], sided?);
    let clear = cx.clear_depth_stencil_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let far = cx.clear_depth_vertices(1.0)?;
    let all = cx.clear_vertices()?;
    let left = cx.floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
    let top = cx.floats(&[-1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])?;
    let ds = cx.depth_stencil_buffer(W, H)?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.depth(&ds);
    r.stencil_reference(0);
    r.draw(&clear, &[&far], &[], 3, 1)?;
    r.stencil_reference(1);
    r.draw(&mask, &[&left], &[], 4, 1)?;
    r.draw(&incr, &[&top], &[], 4, 1)?;
    for (k, p) in eq.iter().enumerate() {
        r.stencil_reference(k as u8);
        r.draw(p, &[&all], &[], 3, 1)?;
    }
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let mut bad = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let stencil = usize::from(x < W / 2) + usize::from(y < H / 2);
            let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            if got != COLORS[stencil] {
                bad.push((x, y, stencil, got));
            }
        }
    }
    println!("{} pixels, {} wrong", W * H, bad.len());
    assert!(bad.is_empty(), "wrong pixels (x, y, stencil, got) {:?}", &bad[..bad.len().min(8)]);

    let right = cx.floats(&[1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
    let mut r = cx.record();
    r.target(&target);
    r.depth(&ds);
    r.stencil_reference(0);
    r.draw(&clear, &[&far], &[], 3, 1)?;
    r.stencil_reference(1);
    r.draw(&sided, &[&left], &[], 4, 1)?;
    r.draw(&sided, &[&right], &[], 4, 1)?;
    r.draw(&eq[1], &[&all], &[], 3, 1)?;
    r.stencil_reference(255);
    r.draw(&eq[2], &[&all], &[], 3, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let px = target.bytes()?;
    let mut wrong = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let want = if x < W / 2 { COLORS[2] } else { COLORS[1] };
            let got: [u8; 4] = px[((y * W + x) * 4) as usize..][..4].try_into()?;
            if got != want {
                wrong.push((x, y, got));
            }
        }
    }
    println!("double sided: back faces 255 (left), front faces 1 (right): {} pixels, {} wrong", W * H, wrong.len());
    assert!(wrong.is_empty(), "wrong pixels (x, y, got) {:?}", &wrong[..wrong.len().min(8)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
