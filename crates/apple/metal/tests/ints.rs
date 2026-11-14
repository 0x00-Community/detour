#![cfg(target_os = "macos")]

use detour::agx_codegen::apple9::Apple9;
use detour_apple_metal::M4;

fn want(a: u32, b: u32) -> [u32; 20] {
    let (sa, sb, s) = (a as i32, b as i32, b & 31);
    let fk = (sa as f32) * 0.37f32;
    [
        a.wrapping_add(b),
        a.wrapping_sub(b),
        a.wrapping_mul(b),
        a << s,
        a >> s,
        (sa >> s) as u32,
        a & b,
        a | b,
        a ^ b,
        sa.min(sb) as u32,
        sa.max(sb) as u32,
        a.min(b),
        a.max(b),
        if sa < sb { a } else { b },
        u32::from(a < b),
        if a == b { u32::MAX } else { 0 },
        (sa as f32).to_bits(),
        (a as f32).to_bits(),
        fk as i32 as u32,
        fk.abs() as u32,
    ]
}

#[test]
fn ints_on_the_m4() {
    let src = include_str!("../../../intel/driver/examples/compute/ints.mlir");
    let codes = Apple9::default().kernels(src).unwrap_or_else(|e| panic!("{e}"));
    let gpu = match M4::open() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("not an Apple9 GPU here: {e}");
            return;
        }
    };
    let kernel = gpu.kernel(&codes[0]).unwrap();
    for groups in [9u32, 64] {
        let pairs: Vec<(u32, u32)> = [(7, 3), (3, 7), (5, 5), (0xfffffffb, 3), (3, 0xfffffffb), (0x80000000, 33), (123456789, 0xffffffff), (0x12345678, 4)].repeat(2 * groups as usize);
        let n = pairs.len();
        let inputs: Vec<u32> = pairs.iter().flat_map(|(a, b)| [*a, *b]).collect();
        let dst = 4 * inputs.len();
        let mut memory = gpu.buffer(dst + 80 * n).unwrap();
        memory.write_words(0, &inputs);
        gpu.dispatch(&kernel, &memory, &[0, dst as u32], [groups, 1, 1]).unwrap();
        let words = memory.words();
        let bad: Vec<String> = pairs
            .iter()
            .enumerate()
            .filter_map(|(t, (a, b))| {
                let got = &words[dst / 4 + 20 * t..dst / 4 + 20 * t + 20];
                (got != want(*a, *b)).then(|| format!("invocation {t} ({a:#x}, {b:#x}): got {got:x?}, want {:x?}", want(*a, *b)))
            })
            .collect();
        assert!(bad.is_empty(), "{groups} workgroups: {} of {n} wrong:\n{}", bad.len(), bad[..bad.len().min(6)].join("\n"));
        eprintln!("{groups} workgroups: {} of {} results right", 20 * n, 20 * n);
    }
}
