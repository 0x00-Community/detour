use detour::xe3::{Codegen, Context};
use detour::xe::Xe;
use detour::base::{Gpu, Native};

const N: usize = 1 << 16;
const RESULTS: usize = 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn operands() -> Vec<[i32; 2]> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let special = [0, 1, -1, i32::MIN, i32::MAX, 2, -2, 255, 256, 65535, 65536, 31, 32];
    (0..N)
        .map(|i| {
            let r = rng.next();
            let mut v = [r as i32, (r >> 32) as i32];
            match i % 8 {
                0 => v[1] = v[0],
                1 => v = [(r % 1000) as i32 - 500, ((r >> 32) % 1000) as i32 - 500],
                2 => v = [(r % 65536) as i32, ((r >> 32) % 65536) as i32],
                _ => {}
            }
            if i < special.len() * special.len() {
                v = [special[i % special.len()], special[i / special.len()]];
            }
            v
        })
        .collect()
}

fn reference([a, b]: [i32; 2]) -> [i32; RESULTS] {
    let s = (b & 31) as u32;
    let (ua, ub) = (a as u32, b as u32);
    let fk = a as f32 * 0.37;
    [
        a.wrapping_add(b),
        a.wrapping_sub(b),
        a.wrapping_mul(b),
        a.wrapping_shl(s),
        (ua >> s) as i32,
        a >> s,
        a & b,
        a | b,
        a ^ b,
        a.min(b),
        a.max(b),
        ua.min(ub) as i32,
        ua.max(ub) as i32,
        if a < b { a } else { b },
        i32::from(ua < ub),
        -i32::from(a == b),
        (a as f32).to_bits() as i32,
        (ua as f32).to_bits() as i32,
        fk as i32,
        fk.abs() as u32 as i32,
    ]
}

const NAMES: [&str; RESULTS] = [
    "addi", "subi", "muli", "shli", "shrui", "shrsi", "andi", "ori", "xori", "minsi", "maxsi", "minui", "maxui", "select(slt)", "extui(ult)", "extsi(eq)",
    "sitofp", "uitofp", "fptosi", "fptoui",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let kernel = cx.kernel(&codegen, include_str!("compute/ints.mlir"), "ints")?;
    let ops = operands();
    let src = cx.buffer(&ops.iter().flatten().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
    let mut dst = cx.buffer(&vec![0xa5; N * RESULTS * 4])?;

    let mut r = cx.record();
    r.dispatch(&kernel, [(N / 16) as u32, 1], &[src.addr as u32, dst.addr as u32])?;
    let t = std::time::Instant::now();
    let n = cx.submit(r.finish())?;
    cx.wait(n)?;
    println!("{N} invocations, submit to done {:?}", t.elapsed());

    let got: Vec<i32> = dst.bytes()?.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    let mut wrong = 0;
    for (k, name) in NAMES.iter().enumerate() {
        let bad: Vec<_> = ops.iter().enumerate().filter(|(i, o)| got[i * RESULTS + k] != reference(**o)[k]).map(|(i, o)| (*o, got[i * RESULTS + k], reference(*o)[k])).collect();
        println!("{name}: {} of {N} wrong {:?}", bad.len(), &bad[..bad.len().min(4)]);
        wrong += bad.len();
    }
    assert_eq!(wrong, 0, "results off Rust");
    println!("ok");
    Ok(())
}
