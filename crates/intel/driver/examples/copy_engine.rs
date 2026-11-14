//! The copy engine (BCS: `Context::open_copy`, `record_copy`, `Recorder::blit`, XY_BLOCK_COPY_BLT) on Xe3, byte for
//! byte: a 64 x 40 rectangle of a 100 x 60 rgba8unorm texture of noise blitted to (20, 30) of a 128 x 80 one of zeros
//! (the rest left zero); the same with 16-byte texels (rgba32float); and a buffer a render queue dispatch fills, blitted
//! (as a 128 x 64 image of u32s) into another by a copy frame that waits for the dispatch on the GPU.

use detour::base::{Gpu, Native, TextureFormat};
use detour::xe3::{Codegen, Context, Linear};
use detour::xe::Xe;

const FILL: &str = r#"
func.func @fill(%dst: i32) attributes {detour.stage = "compute"} {
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %c16 = arith.constant 16 : i32
  %base = arith.muli %g, %c16 : i32
  %i = arith.addi %base, %l : i32
  %k = arith.constant 2654435761 : i32
  %v = arith.muli %i, %k : i32
  %c4 = arith.constant 4 : i32
  %o = arith.muli %i, %c4 : i32
  %at = arith.addi %dst, %o : i32
  "detour.store"(%at, %v) : (i32, i32) -> ()
  return
}
"#;

fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// The bytes of a `dw` x `dh` image of `bpp`-byte texels after copying `size` texels from (sx, sy) of `src` (`sw` wide)
/// to (dx, dy) of zeros.
fn expected(src: &[u8], sw: u32, (dw, dh): (u32, u32), bpp: u32, [sx, sy]: [u32; 2], [dx, dy]: [u32; 2], [w, h]: [u32; 2]) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * bpp) as usize];
    for row in 0..h {
        let from = (((sy + row) * sw + sx) * bpp) as usize;
        let to = (((dy + row) * dw + dx) * bpp) as usize;
        out[to..to + (w * bpp) as usize].copy_from_slice(&src[from..from + (w * bpp) as usize]);
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    cx.open_copy()?;
    let mut ok = true;
    for (name, format) in [("rgba8unorm", TextureFormat::Rgba8Unorm), ("rgba32float (16-byte texels)", TextureFormat::Rgba32Float)] {
        let bpp = format.bytes();
        let src_bytes = noise((100 * 60 * bpp) as usize, 0x1234_5678 + u64::from(bpp));
        let src = cx.texture(100, 60, format, &src_bytes)?;
        let mut dst = cx.texture(128, 80, format, &vec![0; (128 * 80 * bpp) as usize])?;
        let mut r = cx.record_copy()?;
        r.blit(Linear::from(&src), [10, 5], Linear::from(&dst), [20, 30], [64, 40])?;
        let done = cx.submit_waiting(r.finish(), &[])?;
        cx.wait_done(done)?;
        let want = expected(&src_bytes, 100, (128, 80), bpp, [10, 5], [20, 30], [64, 40]);
        let got = &dst.bytes()?[..want.len()];
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!("{name}: {bad} of {} bytes wrong", want.len());
        ok &= bad == 0;
    }

    let fill = cx.kernel(&codegen, FILL, "fill")?;
    let (w, h) = (128u32, 64u32);
    let zeros = vec![0u8; (w * h * 4) as usize];
    let filled = cx.buffer(&zeros)?;
    let mut copied = cx.buffer(&zeros)?;
    let mut r = cx.record();
    r.dispatch(&fill, [w * h / 16, 1], &[filled.addr as u32])?;
    let written = cx.submit_waiting(r.finish(), &[])?;
    let image = |addr: u64| Linear { addr, pitch: w * 4, width: w, height: h, bpp: 4 };
    let mut c = cx.record_copy()?;
    c.blit(image(filled.addr), [0, 0], image(copied.addr), [0, 0], [w, h])?;
    let done = cx.submit_waiting(c.finish(), &[written])?;
    cx.wait_done(done)?;
    let got = &copied.bytes()?[..(w * h * 4) as usize];
    let bad = got.chunks(4).enumerate().filter(|(i, b)| u32::from_le_bytes((*b).try_into().expect("4")) != (*i as u32).wrapping_mul(2_654_435_761)).count();
    println!("render queue dispatch, then the copy queue's blit after it: {bad} of {} u32s wrong", w * h);
    ok &= bad == 0;
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
