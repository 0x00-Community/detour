use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{BufferTexels, Codegen, Context, Copies, Texels};
use detour::xe::Xe;

fn noise(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

fn rect(src: &[u8], ss: usize, dst: &mut [u8], ds: usize, row: usize, rows: usize) {
    for y in 0..rows {
        dst[y * ds..][..row].copy_from_slice(&src[y * ss..][..row]);
    }
}

fn report(name: &str, got: &[u8], want: &[u8]) -> bool {
    let bad = got.iter().zip(want).filter(|(g, w)| g != w).count();
    println!("{name}: {} bytes, {bad} differ", want.len());
    bad == 0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let copies = Copies::new(&mut cx, &codegen)?;
    let mut ok = true;

    let src = noise(310 * 1024, 7);
    let a = cx.buffer(&src)?;
    let mut b = cx.buffer(&vec![0xee; 310 * 1024])?;
    let mut c = cx.buffer(&vec![0xee; 2048])?;
    let mut r = cx.record();
    copies.buffer_to_buffer(&mut r, (&a, 8), (&b, 16), 300 * 1024)?;
    copies.buffer_to_buffer(&mut r, (&a, 3), (&c, 5), 1001)?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let mut want = vec![0xee; 310 * 1024];
    want[16..16 + 300 * 1024].copy_from_slice(&src[8..8 + 300 * 1024]);
    ok &= report("buffer to buffer, 300 KiB aligned", &b.bytes()?[..want.len()], &want);
    let mut want = vec![0xee; 2048];
    want[5..1006].copy_from_slice(&src[3..1004]);
    ok &= report("buffer to buffer, 1001 bytes from 3 to 5", &c.bytes()?[..2048], &want);

    let (tw, th) = (37u32, 23u32);
    let base = noise((tw * th * 4) as usize, 11);
    let mut tex = cx.texture(tw, th, TextureFormat::Rgba8Unorm, &base)?;
    let staged = noise(256 * 10, 13);
    let stage = cx.buffer(&staged)?;
    let mut out = cx.buffer(&vec![0xee; 160 * 10])?;
    let mut r = cx.record();
    copies.buffer_to_texture(&mut r, BufferTexels { buffer: &stage, offset: 0, bytes_per_row: 256 }, Texels::from(&tex), [5, 3], [20, 10])?;
    copies.texture_to_buffer(&mut r, Texels::from(&tex), [5, 3], BufferTexels { buffer: &out, offset: 0, bytes_per_row: 160 }, [20, 10])?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let mut want = base.clone();
    rect(&staged, 256, &mut want[((3 * tw + 5) * 4) as usize..], (tw * 4) as usize, 80, 10);
    ok &= report("buffer to texture, 20 x 10 at (5, 3)", &tex.bytes()?[..want.len()], &want);
    let mut back = vec![0xee; 1600];
    rect(&staged, 256, &mut back, 160, 80, 10);
    ok &= report("texture to buffer, 160-byte rows", &out.bytes()?[..1600], &back);

    let (rw, rh) = (19u32, 9u32);
    let r8a = noise((rw * rh) as usize, 17);
    let x = cx.texture(rw, rh, TextureFormat::R8Unorm, &r8a)?;
    let mut y = cx.texture(rw, rh, TextureFormat::R8Unorm, &vec![0xee; (rw * rh) as usize])?;
    let mut target = cx.target(tw, th, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
    let mut r = cx.record();
    copies.texture_to_texture(&mut r, (Texels::from(&x), [1, 2]), (Texels::from(&y), [4, 0]), [13, 7])?;
    copies.texture_to_texture(&mut r, (Texels::from(&tex), [0, 0]), (Texels::try_from(&target)?, [0, 0]), [tw, th])?;
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    let mut want = vec![0xee; (rw * rh) as usize];
    rect(&r8a[(2 * rw + 1) as usize..], rw as usize, &mut want[4..], rw as usize, 13, 7);
    ok &= report("texture to texture, r8 13 x 7 from (1, 2) to (4, 0)", &y.bytes()?[..want.len()], &want);
    let whole = tex.bytes()?[..(tw * th * 4) as usize].to_vec();
    ok &= report("texture to texture, rgba8 into a target", &target.bytes()?[..whole.len()], &whole);
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
