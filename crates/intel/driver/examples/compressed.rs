//! Block-compressed textures (WebGPU `texture-compression-bc`: bc1 / bc3 / bc7-rgba-unorm) on Xe3, sampled: 64
//! random blocks a format (32 x 8 texels), drawn texel for texel (nearest, texel centers) into an rgba8unorm target and
//! compared with a decoder written from the formats' definitions (D3D / Khronos Data Format). BC7 here is mode 6 (one
//! subset, 7-bit RGBA endpoints and p-bits, 4-bit indices): its interpolation is exact (weights / 64, + 32), so every
//! texel must match. BC1 / BC3: the endpoints (indices 0, 1) must match; their interpolated colors (D3D allows ±1 as
//! hardware rounds them its own way) are counted against the definitions rounded to nearest and their largest
//! difference reported (on 0xb0a0 about 85% are the same, none off by more than 1).

use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const W: u32 = 32;
const H: u32 = 8;

const SAMPLE: &str = r#"detour.pipeline @p {vertex = @vs, fragment = @fs, topology = "triangle-strip", targets = ["rgba8unorm"], filter = "nearest",
  vertex_buffers = [{stride = 16, attributes = [{location = 0, format = "float32x2", offset = 0}, {location = 1, format = "float32x2", offset = 8}]}]}
func.func @vs(%p: vector<2xf32>, %uv: vector<2xf32>) -> (vector<4xf32>, vector<2xf32>) attributes {detour.stage = "vertex"} {
  %zw = arith.constant dense<[0.0, 1.0]> : vector<2xf32>
  %pos = "vector.shuffle"(%p, %zw) <{mask = array<i64: 0, 1, 2, 3>}> : (vector<2xf32>, vector<2xf32>) -> vector<4xf32>
  return %pos, %uv : vector<4xf32>, vector<2xf32>
}
func.func @fs(%uv: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %out = "detour.sample"(%uv) {binding = 0 : i64} : (vector<2xf32>) -> vector<4xf32>
  return %out : vector<4xf32>
}
"#;

/// xorshift64*
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// A 5:6:5 color as 8 bits a channel (bit replication).
fn rgb565(c: u16) -> [u32; 3] {
    let (r, g, b) = (u32::from(c >> 11), u32::from((c >> 5) & 63), u32::from(c & 31));
    [r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2]
}

/// A BC1 color block's 16 texels (index 0 ..= 3 each, and whether it was an endpoint); `four`: always the 4-color mode
/// (BC3's color block).
fn bc1(block: &[u8], four: bool) -> [([u8; 4], bool); 16] {
    let (c0, c1) = (u16::from_le_bytes([block[0], block[1]]), u16::from_le_bytes([block[2], block[3]]));
    let (a, b) = (rgb565(c0), rgb565(c1));
    let mix = |wa: u32, wb: u32, d: u32| -> [u8; 4] { [0, 1, 2].map(|i| ((wa * a[i] + wb * b[i] + d / 2) / d) as u8).into_iter().chain([255]).collect::<Vec<_>>().try_into().unwrap() };
    let colors: [[u8; 4]; 4] = if four || c0 > c1 {
        [mix(1, 0, 1), mix(0, 1, 1), mix(2, 1, 3), mix(1, 2, 3)]
    } else {
        [mix(1, 0, 1), mix(0, 1, 1), mix(1, 1, 2), [0, 0, 0, 0]]
    };
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    core::array::from_fn(|t| {
        let i = (bits >> (2 * t)) & 3;
        (colors[i as usize], i < 2 || (!four && c0 <= c1 && i == 3))
    })
}

/// A BC3 block: the alpha block (8 bytes: a0, a1, 3-bit indices) then a BC1 color block in the 4-color mode.
fn bc3(block: &[u8]) -> [([u8; 4], bool); 16] {
    let (a0, a1) = (u32::from(block[0]), u32::from(block[1]));
    let alphas: [u32; 8] = if a0 > a1 {
        [a0, a1, (6 * a0 + a1 + 3) / 7, (5 * a0 + 2 * a1 + 3) / 7, (4 * a0 + 3 * a1 + 3) / 7, (3 * a0 + 4 * a1 + 3) / 7, (2 * a0 + 5 * a1 + 3) / 7, (a0 + 6 * a1 + 3) / 7]
    } else {
        [a0, a1, (4 * a0 + a1 + 2) / 5, (3 * a0 + 2 * a1 + 2) / 5, (2 * a0 + 3 * a1 + 2) / 5, (a0 + 4 * a1 + 2) / 5, 0, 255]
    };
    let bits = block[2..8].iter().rev().fold(0u64, |acc, b| acc << 8 | u64::from(*b));
    let colors = bc1(&block[8..], true);
    core::array::from_fn(|t| {
        let i = ((bits >> (3 * t)) & 7) as usize;
        let (mut c, exact) = colors[t];
        c[3] = alphas[i] as u8;
        (c, exact && (i < 2 || (a0 <= a1 && i >= 6)))
    })
}

/// A random BC7 mode 6 block and its 16 texels (exactly defined).
fn bc7_mode6(rng: &mut Rng) -> (Vec<u8>, [[u8; 4]; 16]) {
    const WEIGHTS: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
    let mut bits: u128 = 1 << 6;
    let mut at = 7;
    let mut put = |bits: &mut u128, v: u128, n: u32| {
        *bits |= v << at;
        at += n;
    };
    let ends: [[u32; 2]; 4] = core::array::from_fn(|_| [(rng.next() & 127) as u32, (rng.next() & 127) as u32]);
    for e in &ends {
        put(&mut bits, e[0].into(), 7);
        put(&mut bits, e[1].into(), 7);
    }
    let p = [(rng.next() & 1) as u32, (rng.next() & 1) as u32];
    put(&mut bits, p[0].into(), 1);
    put(&mut bits, p[1].into(), 1);
    let mut index = [0u32; 16];
    for (t, i) in index.iter_mut().enumerate() {
        let n = if t == 0 { 3 } else { 4 };
        *i = (rng.next() as u32) & ((1 << n) - 1);
        put(&mut bits, (*i).into(), n);
    }
    let texels = core::array::from_fn(|t| {
        let w = WEIGHTS[index[t] as usize];
        [0, 1, 2, 3].map(|c| {
            let (e0, e1) = (ends[c][0] << 1 | p[0], ends[c][1] << 1 | p[1]);
            (((64 - w) * e0 + w * e1 + 32) >> 6) as u8
        })
    });
    (bits.to_le_bytes().to_vec(), texels)
}

/// The texels of `blocks` (block k at x = k % (W / 4), y = k / (W / 4)) row by row, and whether each is exact.
fn image(blocks: &[[([u8; 4], bool); 16]]) -> (Vec<u8>, Vec<bool>) {
    let (mut px, mut exact) = (vec![0u8; (W * H * 4) as usize], vec![false; (W * H) as usize]);
    for (k, b) in blocks.iter().enumerate() {
        let (bx, by) = ((k as u32 % (W / 4)) * 4, (k as u32 / (W / 4)) * 4);
        for (t, (c, e)) in b.iter().enumerate() {
            let (x, y) = (bx + t as u32 % 4, by + t as u32 / 4);
            px[((y * W + x) * 4) as usize..][..4].copy_from_slice(c);
            exact[(y * W + x) as usize] = *e;
        }
    }
    (px, exact)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let p = cx.pipeline(&codegen, SAMPLE, "p")?;
    let quad = cx.floats(&[-1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])?;
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let n = (W / 4 * H / 4) as usize;
    let mut cases = Vec::new();
    let bc1_data: Vec<u8> = (0..n * 8).map(|_| rng.next() as u8).collect();
    cases.push(("bc1", TextureFormat::Bc1RgbaUnorm, bc1_data.clone(), image(&bc1_data.chunks(8).map(|b| bc1(b, false)).collect::<Vec<_>>())));
    let bc3_data: Vec<u8> = (0..n * 16).map(|_| rng.next() as u8).collect();
    cases.push(("bc3", TextureFormat::Bc3RgbaUnorm, bc3_data.clone(), image(&bc3_data.chunks(16).map(bc3).collect::<Vec<_>>())));
    let (mut bc7_data, mut bc7_texels) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let (b, t) = bc7_mode6(&mut rng);
        bc7_data.extend(b);
        bc7_texels.push(t.map(|c| (c, true)));
    }
    cases.push(("bc7 mode 6", TextureFormat::Bc7RgbaUnorm, bc7_data, image(&bc7_texels)));
    let mut ok = true;
    for (name, format, data, (want, exact)) in cases {
        let tex = cx.texture(W, H, format, &data)?;
        let mut target = cx.target(W, H, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&p, &[&quad], &[&tex], 4, 1)?;
        let done = cx.submit(r.finish())?;
        cx.wait(done)?;
        let px = target.bytes()?;
        let (mut wrong_exact, mut interpolated, mut same, mut worst) = (0, 0, 0, 0u8);
        for i in 0..(W * H) as usize {
            let (got, w) = (&px[i * 4..][..4], &want[i * 4..][..4]);
            let diff = got.iter().zip(w).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
            if exact[i] {
                wrong_exact += usize::from(diff != 0);
            } else {
                interpolated += 1;
                same += usize::from(diff == 0);
                worst = worst.max(diff);
            }
        }
        println!("{name}: {wrong_exact} of {} exactly defined texels wrong; {same} of {interpolated} interpolated ones the same, the rest within {worst}", (W * H) as usize - interpolated);
        ok &= wrong_exact == 0 && worst <= 1;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
