use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const N: usize = 100_000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("workgroups/workgroups.mlir"), "workgroups")?;
    let src: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let groups = N.div_ceil(64);
    let sbuf = cx.buffer(&src.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let mut dst = cx.buffer(&vec![0xa5; N * 4])?;
    let mut counts = cx.buffer(&vec![0xa5; groups * 16])?;
    let mut r = cx.record();
    r.dispatch(&kernel, [groups as u32, 1], &[sbuf.addr as u32, dst.addr as u32, counts.addr as u32, N as u32])?;
    let s = cx.submit(r.finish())?;
    cx.wait(s)?;
    let read = |b: &[u8]| b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
    let got = read(&dst.bytes()?[..N * 4]);
    let want: Vec<u32> = (0..N).map(|i| src[(i / 64 * 64 + 63 - i % 64).min(N - 1)]).collect();
    let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    let cs = read(&counts.bytes()?[..groups * 16]);
    let counts_ok = cs.iter().all(|c| *c == 16);
    println!("{N} values through shared memory, {bad} wrong; {groups} workgroups' 4 shared counters {}", if counts_ok { "16 each" } else { "WRONG" });
    if !counts_ok {
        println!("  counters: {:?}", &cs[..16]);
    }
    if bad > 0 {
        let i = got.iter().zip(&want).position(|(g, w)| g != w).unwrap();
        println!("  first wrong: {i}: {:#x} want {:#x}", got[i], want[i]);
    }
    let pass = bad == 0 && counts_ok;
    println!("{} (device {:#x})", if pass { "ok" } else { "FAILED" }, xe.info().device_id);
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
