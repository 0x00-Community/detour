use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const N: usize = 100_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("atomics/atomics.mlir"), "atomics")?;
    let words = |v: &[u32]| v.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    let mut counters = cx.buffer(&[0; 32])?;
    let mut olds = cx.buffer(&vec![0xa5; N * 4])?;
    let mut maxes = cx.buffer(&words(&[i32::MIN as u32; 4]))?;
    let mut umins = cx.buffer(&words(&[u32::MAX; 4]))?;
    const START: u32 = 0xdead_beef;
    let mut swap = cx.buffer(&words(&[START, 0, 0, 0]))?;
    let mut swaps = cx.buffer(&vec![0; N * 4])?;
    let params = cx.buffer(&words(&[counters.addr, olds.addr, maxes.addr, umins.addr, swap.addr, swaps.addr].map(|a| a as u32)))?;
    let mut r = cx.record();
    r.dispatch(&kernel, [N.div_ceil(16) as u32, 1], &[params.addr as u32, N as u32])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let read = |b: &[u8]| b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();

    let cs = read(&counters.bytes()?[..32]);
    let os = read(&olds.bytes()?[..N * 4]);
    let adds_ok = (0..8).all(|k| {
        let mut got: Vec<u32> = (k..N).step_by(8).map(|i| os[i]).collect();
        got.sort_unstable();
        cs[k] as usize == got.len() && got.iter().enumerate().all(|(j, o)| *o as usize == j)
    });
    let want_max: Vec<i32> = (0..4).map(|m| (m..N).step_by(4).map(|i| ((i as u32 * 37) & 1023) as i32 - 512).max().unwrap()).collect();
    let got_max: Vec<i32> = read(&maxes.bytes()?[..16]).into_iter().map(|w| w as i32).collect();
    let want_umin: Vec<u32> = (0..4).map(|u| (0..N).filter(|i| (i >> 4) & 3 == u).map(|i| (i as u32).wrapping_mul(7919) & 65535).min().unwrap()).collect();
    let got_umin = read(&umins.bytes()?[..16]);
    let mut seen: Vec<u32> = read(&swaps.bytes()?[..N * 4]);
    seen.push(read(&swap.bytes()?[..4])[0]);
    seen.sort_unstable();
    let mut want: Vec<u32> = (0..N as u32).chain([START]).collect();
    want.sort_unstable();
    let swap_ok = seen == want;
    println!("adds {}: counters {cs:?}", if adds_ok { "ok" } else { "WRONG" });
    println!("max {got_max:?} (want {want_max:?}); umin {got_umin:?} (want {want_umin:?}); exchange {}", if swap_ok { "ok" } else { "WRONG" });
    let pass = adds_ok && got_max == want_max && got_umin == want_umin && swap_ok;
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
