use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const N: usize = 64;

fn reference(i: i32) -> [i32; 5] {
    let r0 = (0..10).map(|k| k * i + 3).sum();
    let r1 = (0..(i & 7)).fold(1i32, |a, k| a.wrapping_mul(3).wrapping_add(k));
    let r2 = if i & 1 != 0 { i * 5 } else { 100 - i };
    let r3 = (0..4).map(|k| if k < (i & 3) { 5 * k } else { i + 10 }).sum();
    let (mut x, mut r4) = (i + 1, 0);
    while x != 1 {
        x = if x & 1 != 0 { 3 * x + 1 } else { x / 2 };
        r4 += 1;
    }
    [r0, r1, r2, r3, r4]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("control/loops.mlir"), "loops")?;
    let mut dst = cx.buffer(&vec![0xa5; N * 32])?;

    let mut r = cx.record();
    r.dispatch(&kernel, [(N / 16) as u32, 1], &[dst.addr as u32])?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;

    let got: Vec<i32> = dst.bytes()?.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    let bad: Vec<_> = (0..N).filter(|i| got[i * 8..][..5] != reference(*i as i32)).map(|i| (i, &got[i * 8..][..5], reference(i as i32))).collect();
    println!("{N} invocations, {} wrong", bad.len());
    assert!(bad.is_empty(), "wrong (i, got, want) {:?}", &bad[..bad.len().min(6)]);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
