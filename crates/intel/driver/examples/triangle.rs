use std::time::{Duration, Instant};

use detour::xe3::{Codegen, Device};
use detour::xe::Xe;
use detour::base::{Gpu, Native};

const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const STATISTICS: [&str; 6] = ["IA vertices", "IA primitives", "VS invocations", "clipper invocations", "clipper primitives", "PS invocations"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let forms: Vec<String> = std::env::args().skip(1).collect();
    let forms = if forms.is_empty() { vec!["func".into(), "gpu".into(), "spirv".into(), "arith".into()] } else { forms };
    let xe = Xe::open_first()?;
    let device = Device::<Xe, Native>::new(&xe)?;
    let codegen = Codegen::default();
    for form in forms {
        let path = format!("{}/crates/intel/driver/examples/triangle/{form}.mlir", env!("CARGO_MANIFEST_DIR"));
        let src = std::fs::read_to_string(&path)?;
        let t = Instant::now();
        let image = codegen.compile(&src, &Codegen::pipeline())?;
        let compiled = t.elapsed();
        let t = Instant::now();
        let mut frame = device.run(&image, Duration::from_secs(5))?;
        println!("{form}: compiled in {compiled:?}, ran in {:?} ({} memories, device {:#06x})", t.elapsed(), image.segments.len(), xe.info().device_id);

        let stats = frame.bytes("stats")?.to_vec();
        let q = |i: usize| u64::from_le_bytes(stats[i * 8..][..8].try_into().unwrap());
        let delta: Vec<u64> = (0..6).map(|i| q(6 + i).wrapping_sub(q(i))).collect();
        for (name, d) in STATISTICS.iter().zip(&delta) {
            println!("  {name}: {d}");
        }

        let px = frame.bytes("rt")?;
        let at = |x: usize, y: usize| -> [u8; 4] { px[(y * 64 + x) * 4..][..4].try_into().unwrap() };
        for y in (0..64).step_by(4).rev() {
            let row: String = (0..64).step_by(2).map(|x| if at(x, y) == ORANGE { '#' } else if at(x, y) == BLUE { '.' } else { '?' }).collect();
            println!("  {row}");
        }
        let (mut orange, mut bad) = (0, Vec::new());
        for y in 0..64 {
            for x in 0..64 {
                let p = at(x, y);
                orange += usize::from(p == ORANGE);
                let want = match x + y {
                    s if s <= 62 => Some(ORANGE),
                    s if s >= 64 => Some(BLUE),
                    _ => None,
                };
                if want.is_some_and(|w| w != p) || (want.is_none() && p != ORANGE && p != BLUE) {
                    bad.push((x, y, p));
                }
            }
        }
        println!("  {orange} orange pixels, {} wrong", bad.len());
        assert!(bad.is_empty(), "{form}: wrong pixels {:?}", &bad[..bad.len().min(8)]);
        println!("  ok");
    }
    Ok(())
}
