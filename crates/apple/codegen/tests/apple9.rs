//! detour's compute/ints shader through the Apple9 backend (mlirpass's Apple9 target): its machine code is
//! tests/apple9/ints.expected (`APPLE9_WRITE=<dir>` writes it there). With `APPLE9_METAL` (mlirpass's
//! scripts/apple9_metal.swift built: `swiftc -O scripts/apple9_metal.swift -o target/apple9-metal`) and `APPLE9_DONOR`
//! (its scripts/apple9_donor.metal) it also runs on this Mac's GPU, an Apple9 (M3 / M4), and every result is checked:
//! 144 invocations (nine workgroups of 16, the last hardware workgroup half full).

use detour::agx_codegen::apple9::Apple9;

fn code() -> (String, Vec<u8>, u32) {
    let src = include_str!("../../../intel/driver/examples/compute/ints.mlir");
    let k = Apple9::default().kernels(src).unwrap_or_else(|e| panic!("{e}"));
    let [k] = &k[..] else { panic!("{} kernels", k.len()) };
    let mut text = format!("{} params={} gprs={} shared={} workgroup={}\n", k.name, k.params, k.gprs, k.shared, k.workgroup);
    text += &k.bytes.chunks(32).map(|c| c.iter().map(|b| format!("{b:02x}")).collect::<String>() + "\n").collect::<String>();
    (text, k.bytes.clone(), k.workgroup)
}

#[test]
fn ints_as_the_m4_runs_it() {
    let (text, _, workgroup) = code();
    if let Ok(dir) = std::env::var("APPLE9_WRITE") {
        std::fs::write(format!("{dir}/ints.expected"), &text).unwrap();
    }
    assert_eq!(workgroup, 16);
    assert_eq!(text, include_str!("apple9/ints.expected"));
}

/// The twenty results of (a, b): the semantics of examples/compute/ints.mlir.
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
fn ints_on_the_gpu() {
    let (Ok(metal), Ok(donor)) = (std::env::var("APPLE9_METAL"), std::env::var("APPLE9_DONOR")) else {
        eprintln!("APPLE9_METAL / APPLE9_DONOR unset: not run on the GPU");
        return;
    };
    let (_, bytes, _) = code();
    let pairs: Vec<(u32, u32)> = [(7, 3), (3, 7), (5, 5), (0xfffffffb, 3), (3, 0xfffffffb), (0x80000000, 33), (123456789, 0xffffffff), (0x12345678, 4)].repeat(18);
    let n = pairs.len();
    let mut words: Vec<u32> = pairs.iter().flat_map(|(a, b)| [*a, *b]).collect();
    let dst = 4 * words.len();
    words.extend(std::iter::repeat_n(0, 20 * n));
    let hex = |w: &[u32]| w.iter().map(|x| format!("{x:x}")).collect::<Vec<_>>().join(",");
    let out = std::process::Command::new(metal)
        .args(["run", &donor, "donor", "--text", &bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(), "--threads", &n.to_string(), "--group", "32"])
        .arg(format!("u32:{}:{}", words.len(), hex(&words)))
        .arg(format!("u32:2:0,{dst:x}"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    let memory: Vec<u32> = stdout.lines().find_map(|l| l.strip_prefix("buf 0 ")).unwrap().split(' ').map(|x| u32::from_str_radix(x, 16).unwrap()).collect();
    let mut bad = Vec::new();
    for (t, (a, b)) in pairs.iter().enumerate() {
        let got = &memory[dst / 4 + 20 * t..dst / 4 + 20 * t + 20];
        if got != want(*a, *b) {
            bad.push(format!("invocation {t} ({a:#x}, {b:#x}): got {got:x?}, want {:x?}", want(*a, *b)));
        }
    }
    assert!(bad.is_empty(), "{} of {n} invocations wrong:\n{}", bad.len(), bad[..bad.len().min(8)].join("\n"));
    eprintln!("{} of {} results right on the GPU", 20 * n, 20 * n);
}
