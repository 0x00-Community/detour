//! Sampler states (`GPUSamplerDescriptor`, a pipeline's `sampler`) on Xe3, pixel by pixel:
//! - address modes: a 4 x 4 texture of distinct texels over a 64 x 64 target, coordinates −1 ..= 3 (nearest): each
//!   pixel's texel as WebGPU defines clamp-to-edge, repeat and mirror-repeat (u and v apart);
//! - LOD clamps: 64 x 64 levels of solid colors, a square at its natural level 0 with lod_min_clamp 2 is level 2, one at
//!   level 3 with lod_max_clamp 1 is level 1;
//! - mag filter: a black | white 2 x 1 texture magnified 64 x: nearest a hard edge, linear a ramp between.

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const N: u32 = 64;

/// A pipeline sampling binding 0 with `sampler` over a quad (x, y, u, v per corner, a strip).
fn pipeline(sampler: &str) -> String {
    include_str!("texture/checker.mlir").replace("filter = \"nearest\",", &format!("sampler = {sampler},"))
}

/// A quad over the whole target, texture coordinates `lo` ..= `hi` across both ways.
fn quad(lo: f32, hi: f32) -> Vec<f32> {
    vec![-1.0, -1.0, lo, lo, 1.0, -1.0, hi, lo, -1.0, 1.0, lo, hi, 1.0, 1.0, hi, hi]
}

/// The texel index along an axis of `size` texels for texel coordinate `t` (floor), by address mode.
fn address(mode: &str, t: f32, size: i32) -> i32 {
    let f = t.floor() as i32;
    match mode {
        "clamp-to-edge" => f.clamp(0, size - 1),
        "repeat" => f.rem_euclid(size),
        "mirror-repeat" => {
            let m = f.rem_euclid(2 * size);
            if m < size { m } else { 2 * size - 1 - m }
        }
        other => panic!("{other}"),
    }
}

fn texel(x: i32, y: i32) -> [u8; 4] {
    [(x * 60 + 10) as u8, (y * 60 + 20) as u8, ((x + 4 * y) * 15) as u8, 255]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let mut ok = true;

    // address modes
    let texels: Vec<u8> = (0..4).flat_map(|y| (0..4).flat_map(move |x| texel(x, y))).collect();
    let tex = cx.texture(4, 4, TextureFormat::Rgba8Unorm, &texels)?;
    let wide = cx.floats(&quad(-1.0, 3.0))?;
    for (u, v) in [("clamp-to-edge", "clamp-to-edge"), ("repeat", "repeat"), ("mirror-repeat", "mirror-repeat"), ("repeat", "mirror-repeat"), ("mirror-repeat", "clamp-to-edge")] {
        let p = cx.pipeline(&codegen, &pipeline(&format!(r#"{{address_mode_u = "{u}", address_mode_v = "{v}"}}"#)), "textured")?;
        let mut target = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&p, &[&wide], &[&tex], 4, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let mut bad = 0;
        for j in 0..N {
            for i in 0..N {
                // u at pixel i's center: −1 + 4 (i + 0.5) / 64, in texels 4 u
                let t = |k: u32| 4.0 * (-1.0 + 4.0 * (k as f32 + 0.5) / N as f32);
                let want = texel(address(u, t(i), 4), address(v, t(j), 4));
                bad += usize::from(px[((j * N + i) * 4) as usize..][..4] != want);
            }
        }
        println!("address {u} / {v}: {bad} of {} pixels wrong", N * N);
        ok &= bad == 0;
    }

    // LOD clamps
    let color = |l: u32| [(40 * l + 20) as u8, (250 - 30 * l) as u8, (50 + 25 * l) as u8, 255];
    let levels: Vec<Vec<u8>> = (0..7).map(|l| color(l).repeat(((N >> l) * (N >> l)) as usize)).collect();
    let refs: Vec<&[u8]> = levels.iter().map(Vec::as_slice).collect();
    let mipped = cx.texture_levels(N, N, TextureFormat::Rgba8Unorm, &refs)?;
    let unit = cx.floats(&quad(0.0, 1.0))?;
    for (sampler, size, want, name) in [
        ("{lod_min_clamp = 2.0}", N, 2, "lod_min_clamp 2 at level 0"),
        ("{lod_max_clamp = 1.0}", N / 8, 1, "lod_max_clamp 1 at level 3"),
        ("{}", N / 8, 3, "unclamped at level 3"),
    ] {
        let p = cx.pipeline(&codegen, &pipeline(sampler), "textured")?;
        let mut target = cx.target(size, size, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&p, &[&unit], &[&mipped], 4, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let bad = (0..(size * size) as usize).filter(|i| px[i * 4..][..4] != color(want)).count();
        println!("{name}: level {want} {}", if bad == 0 { "everywhere" } else { "not everywhere" });
        ok &= bad == 0;
    }

    // mag filter
    let bw = cx.texture(2, 1, TextureFormat::Rgba8Unorm, &[0, 0, 0, 255, 255, 255, 255, 255])?;
    for (mag, name) in [("nearest", "nearest"), ("linear", "linear")] {
        let p = cx.pipeline(&codegen, &pipeline(&format!(r#"{{mag_filter = "{mag}"}}"#)), "textured")?;
        let mut target = cx.target(128, 2, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&p, &[&unit], &[&bw], 4, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let px = target.bytes()?;
        let row: Vec<u8> = (0..128).map(|x| px[x * 4]).collect();
        let hard = row.iter().all(|&v| v == 0 || v == 255) && row[..64].iter().all(|&v| v == 0) && row[64..].iter().all(|&v| v == 255);
        let ramp = row.windows(2).all(|w| w[0] <= w[1]) && row.iter().filter(|&&v| v != 0 && v != 255).count() > 32;
        let good = if mag == "nearest" { hard } else { ramp };
        println!("mag {name}: {}", if good { if hard { "a hard edge" } else { "a ramp" } } else { "wrong" });
        ok &= good;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
