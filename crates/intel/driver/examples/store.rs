//! The smallest job through the xe3 dialect: examples/store.mlir (one MI_STORE_DATA_IMM) laid out, packed, run on the
//! first xe device; the CPU reads the stored dword back.

use std::time::{Duration, Instant};

use detour::xe3::{Codegen, Device};
use detour::xe::Xe;
use detour::base::{Gpu, Native};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let src = include_str!("store.mlir");
    let codegen = Codegen::default();
    let image = codegen.compile(src, "builtin.module(xe3-end, xe3-layout)")?;
    let xe = Xe::open_first()?;
    let device = Device::<Xe, Native>::new(&xe)?;
    let t = Instant::now();
    let mut frame = device.run(&image, Duration::from_secs(5))?;
    let elapsed = t.elapsed();
    let got = u32::from_le_bytes(frame.bytes("result")?[..4].try_into()?);
    println!("stored {got:#010x} in {elapsed:?}");
    assert_eq!(got, 0xacd0_c0de);
    println!("ok");
    Ok(())
}
