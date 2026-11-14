use detour::xe3::{Codegen, Context};
use detour::xe::Xe;
use detour::base::{Gpu, Native, TextureFormat};

const N: u32 = 1 << 16;

const KERNELS: &str = r#"
func.func @fill(%dst: i32, %k: i32, %c: i32) attributes {detour.stage = "compute"} {
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %c16 = arith.constant 16 : i32
  %base = arith.muli %g, %c16 : i32
  %i = arith.addi %base, %l : i32
  %ki = arith.muli %i, %k : i32
  %v = arith.addi %ki, %c : i32
  %c4 = arith.constant 4 : i32
  %o = arith.muli %i, %c4 : i32
  %at = arith.addi %dst, %o : i32
  "detour.store"(%at, %v) : (i32, i32) -> ()
  return
}
func.func @twice(%src: i32, %dst: i32) attributes {detour.stage = "compute"} {
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %c16 = arith.constant 16 : i32
  %base = arith.muli %g, %c16 : i32
  %i = arith.addi %base, %l : i32
  %c4 = arith.constant 4 : i32
  %o = arith.muli %i, %c4 : i32
  %from = arith.addi %src, %o : i32
  %v = "detour.load"(%from) : (i32) -> i32
  %d = arith.addi %v, %v : i32
  %to = arith.addi %dst, %o : i32
  "detour.store"(%to, %d) : (i32, i32) -> ()
  return
}
"#;

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().expect("4"))).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let (fill, twice) = (cx.kernel(&codegen, KERNELS, "fill")?, cx.kernel(&codegen, KERNELS, "twice")?);
    let zeros = vec![0u8; (N * 4) as usize];
    let [mut a, mut b, c, mut d] = [cx.buffer(&zeros)?, cx.buffer(&zeros)?, cx.buffer(&zeros)?, cx.buffer(&zeros)?];
    let groups = [N / 16, 1];
    cx.open_compute()?;
    let addr = |x: &detour::xe3::Buffer<'_, Xe>| x.addr as u32;

    let mut f = cx.record_compute()?;
    f.dispatch(&fill, groups, &[addr(&a), 3, 1])?;
    let filled = cx.submit_waiting(f.finish(), &[])?;
    let mut r = cx.record();
    r.dispatch(&twice, groups, &[addr(&a), addr(&b)])?;
    let doubled = cx.submit_waiting(r.finish(), &[filled])?;
    let mut r = cx.record();
    r.dispatch(&fill, groups, &[addr(&c), 7, 5])?;
    let filled_c = cx.submit_waiting(r.finish(), &[doubled])?;
    let mut f = cx.record_compute()?;
    f.dispatch(&twice, groups, &[addr(&c), addr(&d)])?;
    let last = cx.submit_waiting(f.finish(), &[filled_c])?;
    cx.wait_done(last)?;
    cx.wait_done(doubled)?;

    let mut ok = true;
    for (name, buf, k, c0) in [("B (render after compute)", &mut b, 3u32, 1u32), ("D (compute after render)", &mut d, 7, 5)] {
        let got = words(&buf.bytes()?[..(N * 4) as usize]);
        let bad = got.iter().enumerate().filter(|(i, v)| **v != 2 * (k * *i as u32 + c0)).count();
        println!("{name}: {bad} of {N} wrong");
        ok &= bad == 0;
    }

    a.bytes()?[..(N * 4) as usize].fill(0);
    b.bytes()?[..(N * 4) as usize].fill(0);
    let mut f = cx.record_compute()?;
    f.dispatch(&fill, groups, &[addr(&a), 3, 1])?;
    let raced_fill = cx.submit_waiting(f.finish(), &[])?;
    let mut r = cx.record();
    r.dispatch(&twice, groups, &[addr(&a), addr(&b)])?;
    let raced = cx.submit_waiting(r.finish(), &[])?;
    cx.wait_done(raced)?;
    cx.wait_done(raced_fill)?;
    let got = words(&b.bytes()?[..(N * 4) as usize]);
    let early = got.iter().enumerate().filter(|(i, v)| **v != 2 * (3 * *i as u32 + 1)).count();
    println!("without the wait: {early} of {N} read before they were written (racing; not checked)");
    let src = include_str!("storage/kernels.mlir");
    let store = cx.kernel(&codegen, src, "fill")?;
    let (w, h) = (37u32, 23u32);
    let size = |f: TextureFormat| (w * h * f.bytes()) as usize;
    let rgba = cx.texture(w, h, TextureFormat::Rgba32Float, &vec![0; size(TextureFormat::Rgba32Float)])?;
    let mut red = cx.texture(w, h, TextureFormat::R32Float, &vec![0; size(TextureFormat::R32Float)])?;
    let bytes = cx.texture(w, h, TextureFormat::Rgba8Unorm, &vec![0; size(TextureFormat::Rgba8Unorm)])?;
    let mut f = cx.record_compute()?;
    f.dispatch_with(&store, [w.div_ceil(16), h], &[w], &[&rgba, &red, &bytes])?;
    let stored = cx.submit_waiting(f.finish(), &[])?;
    cx.wait_done(stored)?;
    let got = red.bytes()?;
    let bad = (0..w * h).filter(|&i| f32::from_le_bytes(got[(i * 4) as usize..][..4].try_into().expect("4")) != ((i % w) * (i / w)) as f32 + 0.5).count();
    println!("storage texture written from the compute queue: {bad} of {} texels wrong", w * h);
    ok &= bad == 0;
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
