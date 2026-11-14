use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 256;
const H: u32 = 256;
const COLORS: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let gradient = cx.pipeline(&codegen, include_str!("varying/gradient.mlir"), "gradient")?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 0.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let positions = [[-1.0f32, -1.0], [1.0, -1.0], [-1.0, 1.0]];
    let mut verts = Vec::new();
    for (p, c) in positions.iter().zip(COLORS) {
        verts.extend(p.iter().chain(&c).flat_map(|v| v.to_le_bytes()));
    }
    let vb = cx.buffer(&verts)?;
    let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;

    let mut r = cx.record();
    r.target(&target);
    r.draw(&clear, &[&corners], &[], 3, 1)?;
    r.draw(&gradient, &[&vb], &[], 3, 1)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let px = target.bytes()?;
    let (mut inside, mut bad, mut worst) = (0usize, Vec::new(), 0i32);
    for y in 0..H {
        for x in 0..W {
            let (l1, l2) = ((f64::from(x) + 0.5) / f64::from(W), (f64::from(y) + 0.5) / f64::from(H));
            if l1 + l2 > 1.0 - 1.0 / f64::from(W) {
                continue;
            }
            inside += 1;
            let l0 = 1.0 - l1 - l2;
            let got = &px[((y * W + x) * 4) as usize..][..4];
            for c in 0..4 {
                let want = (l0 * f64::from(COLORS[0][c]) + l1 * f64::from(COLORS[1][c]) + l2 * f64::from(COLORS[2][c])) * 255.0;
                let d = (f64::from(got[c]) - want).abs();
                worst = worst.max(d.ceil() as i32);
                if d > 1.0 {
                    bad.push((x, y, c, got[c], want));
                }
            }
        }
    }
    println!("{inside} pixels inside, worst difference {worst} (unorm8 steps), {} wrong", bad.len());
    let at = |x: u32, y: u32| px[((y * W + x) * 4) as usize..][..4].to_vec();
    println!("(0,0) {:?} (254,0) {:?} (0,254) {:?} (85,85) {:?}", at(0, 0), at(254, 0), at(0, 254), at(85, 85));
    assert!(bad.is_empty(), "wrong: {:?}", &bad[..bad.len().min(8)]);
    println!("ok");
    Ok(())
}
