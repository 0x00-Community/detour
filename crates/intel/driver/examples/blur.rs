//! vello_cpu's gaussian blur on Xe3 (R5, `Filters::blur`) against vello_cpu running it (`detour::filter`, the u8
//! kernel), byte for byte: premultiplied noise with shapes, σ from 0.4 (direct convolution) to 40 (four decimations),
//! whole tiles whose decimated sizes are odd (the upsampling's clamped sizes), the source's rows padded. Then drop
//! shadows (offsets fractional, negative, past the size; no blur; translucent colors). `blur K` runs case K alone.

use std::sync::Arc;

use detour::base::vello_common::filter::PreparedFilter;
use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context, Filters, Plane};
use detour::xe::Xe;
use detour::vello_cpu::filter_effects::{EdgeMode, Filter, FilterGraph, FilterPrimitive};
use detour::vello_cpu::kurbo::Affine;
use detour::vello_cpu::Pixmap;

/// A premultiplied image: noise under a few bright discs, transparent corners.
fn image(w: u16, h: u16, seed: u32) -> Vec<[u8; 4]> {
    let mut s = seed.wrapping_mul(2654435761) | 1;
    (0..u32::from(w) * u32::from(h))
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let (x, y) = (i % u32::from(w), i / u32::from(w));
            let disc = (x as i32 - i32::from(w) / 3).pow(2) + (y as i32 - i32::from(h) / 2).pow(2) < (i32::from(h) / 3).pow(2);
            let a = if disc { 255 } else if x < 3 || y < 2 { 0 } else { (s & 0xff) as u8 };
            let c = |k: u32| ((((s >> k) & 0xff) * u32::from(a)) / 255) as u8;
            [c(8), c(16), c(24), a]
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // whole tiles (a filter layer's pixmap is; vello_cpu's in-place upsampling writes past an odd-sized pixmap), the
    // decimated sizes odd: 60 → 30 → 15 → 8 and back clamped
    let cases: [(u16, u16, f32); 9] = [(40, 24, 0.4), (40, 24, 1.0), (36, 20, 2.0), (64, 40, 2.6), (60, 44, 5.0), (100, 68, 9.0), (108, 76, 17.0), (124, 92, 40.0), (8, 8, 3.0)];
    let only: Option<usize> = std::env::args().nth(1).and_then(|a| a.parse().ok());
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let filters = Filters::new(&mut cx, &codegen)?;
    let mut bad_total = 0;
    for (k, &(w, h, sigma)) in cases.iter().enumerate().filter(|(k, _)| only.is_none_or(|o| o == *k)) {
        let px = image(w, h, k as u32 + 1);
        // vello_cpu
        let mut graph = FilterGraph::new();
        let out = graph.add(FilterPrimitive::GaussianBlur { std_deviation: sigma, edge_mode: EdgeMode::None }, None);
        graph.set_output(out);
        let filter = Filter { graph: Arc::new(graph) };
        let mut want = Pixmap::new(w, h);
        want.data_as_u8_slice_mut().copy_from_slice(&px.concat());
        detour::vello_cpu::detour::filter(&filter, &mut want, Affine::IDENTITY);
        let PreparedFilter::GaussianBlur(plan) = PreparedFilter::new(&filter, &Affine::IDENTITY) else { unreachable!("a blur") };

        // Xe3: the source's rows 5 pixels longer
        let (uw, uh) = (u32::from(w), u32::from(h));
        let stride = uw + 5;
        let mut src = vec![[0x77u8; 4]; (stride * uh) as usize];
        for y in 0..usize::from(h) {
            src[y * stride as usize..][..usize::from(w)].copy_from_slice(&px[y * usize::from(w)..][..usize::from(w)]);
        }
        let src = cx.buffer(&src.concat())?;
        let mut weights = vec![0; Filters::weights(&plan)];
        Filters::write_weights(&plan, &mut weights);
        let weights = cx.buffer(&weights)?;
        let mut dst = cx.buffer(&vec![0; (uw * uh * 4) as usize])?;
        let a = cx.buffer(&vec![0; (uw * uh * 4) as usize])?;
        let b = cx.buffer(&vec![0; (uw * uh * 4) as usize])?;
        let plane = |addr: u64, stride| Plane { addr: addr as u32, stride };
        let mut r = cx.record();
        filters.blur(&mut r, plane(src.addr, stride), plane(dst.addr, uw), (uw, uh), &plan, weights.addr as u32, [plane(a.addr, uw), plane(b.addr, uw)])?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let got = dst.bytes()?;
        let want = want.data_as_u8_slice();
        let bad: Vec<(usize, usize, &[u8], &[u8])> = (0..usize::from(w) * usize::from(h)).filter(|i| got[i * 4..][..4] != want[i * 4..][..4]).map(|i| (i % usize::from(w), i / usize::from(w), &got[i * 4..][..4], &want[i * 4..][..4])).collect();
        println!("{k}: {w} x {h}, σ {sigma} ({} decimations, kernel {}): {} differ", plan.n_decimations, plan.kernel_size, bad.len());
        if !bad.is_empty() {
            println!("   first (x, y, got, want): {:?}", &bad[..bad.len().min(6)]);
        }
        bad_total += bad.len();
    }
    // drop shadows (`Filters::drop_shadow`): (w, h, dx, dy, σ, color): fractional, negative, past the size (vello_cpu's
    // u16 wrap: a negative offset as large as the layer clears nothing), no blur, translucent
    let shadows: [(u16, u16, f32, f32, f32, [f32; 4]); 6] = [
        (40, 24, 3.4, 2.6, 2.0, [0.0, 0.0, 0.3, 0.6]),
        (60, 44, -5.5, 4.0, 5.0, [0.9, 0.2, 0.1, 1.0]),
        (36, 20, 0.0, 0.0, 1.0, [0.2, 0.2, 0.2, 0.8]),
        (64, 40, 7.0, -9.0, 0.0, [0.1, 0.5, 0.9, 0.7]),
        (32, 16, -40.0, 3.0, 1.5, [1.0, 1.0, 0.0, 0.5]),
        (100, 68, 12.0, 10.0, 9.0, [0.0, 0.0, 0.0, 0.35]),
    ];
    for (k, &(w, h, dx, dy, sigma, c)) in shadows.iter().enumerate().filter(|(k, _)| only.is_none_or(|o| o == cases.len() + *k)) {
        let px = image(w, h, 100 + k as u32);
        let mut graph = FilterGraph::new();
        let color = detour::vello_cpu::color::AlphaColor::new(c);
        let out = graph.add(FilterPrimitive::DropShadow { dx, dy, std_deviation: sigma, color, edge_mode: EdgeMode::None }, None);
        graph.set_output(out);
        let filter = Filter { graph: Arc::new(graph) };
        let mut want = Pixmap::new(w, h);
        want.data_as_u8_slice_mut().copy_from_slice(&px.concat());
        detour::vello_cpu::detour::filter(&filter, &mut want, Affine::IDENTITY);
        let PreparedFilter::DropShadow(plan) = PreparedFilter::new(&filter, &Affine::IDENTITY) else { unreachable!("a drop shadow") };
        let (uw, uh) = (u32::from(w), u32::from(h));
        let src = cx.buffer(&px.concat())?;
        let blur = Filters::shadow_blur(&plan);
        let mut weights = vec![0; Filters::weights(&blur).max(64)];
        Filters::write_weights(&blur, &mut weights);
        let weights = cx.buffer(&weights)?;
        let mut dst = cx.buffer(&vec![0; (uw * uh * 4) as usize])?;
        let scratch: Vec<_> = (0..3).map(|_| cx.buffer(&vec![0; (uw * uh * 4) as usize])).collect::<Result<_, _>>()?;
        let plane = |addr: u64, stride| Plane { addr: addr as u32, stride };
        let mut r = cx.record();
        filters.drop_shadow(&mut r, plane(src.addr, uw), plane(dst.addr, uw), (uw, uh), &plan, weights.addr as u32, [0, 1, 2].map(|i| plane(scratch[i].addr, uw)))?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let got = dst.bytes()?;
        let want = want.data_as_u8_slice();
        let bad: Vec<(usize, usize, &[u8], &[u8])> = (0..usize::from(w) * usize::from(h)).filter(|i| got[i * 4..][..4] != want[i * 4..][..4]).map(|i| (i % usize::from(w), i / usize::from(w), &got[i * 4..][..4], &want[i * 4..][..4])).collect();
        println!("{}: drop shadow {w} x {h}, ({dx}, {dy}), σ {sigma}, {c:?}: {} differ", cases.len() + k, bad.len());
        if !bad.is_empty() {
            println!("   first (x, y, got, want): {:?}", &bad[..bad.len().min(6)]);
        }
        bad_total += bad.len();
    }
    assert_eq!(bad_total, 0);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
