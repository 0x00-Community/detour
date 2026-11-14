use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context, Linear};
use detour::xe::Xe;

const N: u32 = 1 << 14;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    cx.open_compute()?;
    cx.open_copy()?;
    let (fill, twice) = (cx.kernel(&codegen, KERNELS, "fill")?, cx.kernel(&codegen, KERNELS, "twice")?);
    let zeros = vec![0u8; (N * 4) as usize];
    let (a, b, mut c) = (cx.buffer(&zeros)?, cx.buffer(&zeros)?, cx.buffer(&zeros)?);
    let events = cx.buffer(&[0u8; 64])?;
    let groups = [N / 16, 1];

    let mut f = cx.record_compute()?;
    f.wait_event(&events, 0, 1)?;
    f.dispatch(&twice, groups, &[a.addr as u32, b.addr as u32])?;
    f.set_event(&events, 8, 2)?;
    let computed = cx.submit_waiting(f.finish(), &[])?;
    let image = |addr: u64| Linear { addr, pitch: 256 * 4, width: 256, height: N / 256, bpp: 4 };
    let mut k = cx.record_copy()?;
    k.wait_event(&events, 8, 2)?;
    k.blit(image(b.addr), [0, 0], image(c.addr), [0, 0], [256, N / 256])?;
    let copied = cx.submit_waiting(k.finish(), &[])?;
    let mut r = cx.record();
    r.dispatch(&fill, groups, &[a.addr as u32, 3, 1])?;
    r.set_event(&events, 0, 1)?;
    let rendered = cx.submit_waiting(r.finish(), &[])?;
    for d in [rendered, computed, copied] {
        cx.wait_done(d)?;
    }
    let got: Vec<u32> = c.bytes()?[..(N * 4) as usize].chunks(4).map(|w| u32::from_le_bytes(w.try_into().expect("4"))).collect();
    let bad = got.iter().enumerate().filter(|(i, v)| **v != 2 * (3 * *i as u32 + 1)).count();
    println!("render fills, compute (waiting for event 1) doubles, copy (waiting for event 2) blits: {bad} of {N} wrong");
    let (a2, mut b2) = (cx.buffer(&zeros)?, cx.buffer(&zeros)?);
    let mut f = cx.record_compute()?;
    f.dispatch(&twice, groups, &[a2.addr as u32, b2.addr as u32])?;
    let raced = cx.submit_waiting(f.finish(), &[])?;
    let mut r = cx.record();
    r.dispatch(&fill, groups, &[a2.addr as u32, 3, 1])?;
    let filled = cx.submit_waiting(r.finish(), &[])?;
    cx.wait_done(raced)?;
    cx.wait_done(filled)?;
    let early = b2.bytes()?[..(N * 4) as usize].chunks(4).enumerate().filter(|(i, w)| u32::from_le_bytes((*w).try_into().expect("4")) != 2 * (3 * *i as u32 + 1)).count();
    println!("without the wait (control): {early} of {N} doubled before A was filled");
    assert_eq!(bad, 0);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
