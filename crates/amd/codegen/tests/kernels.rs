//! detour's compute/ints and copies/copy shaders through the RDNA2 backend on mlirpass's RDNA2 target (after
//! rdna2-precheck and detour-core's bridge), dispatched over 1, 4 and 1024 workgroups: the whole image (the IB's PM4,
//! the kernel and its padding; the buffers' places) is what detour's own RDNA2 backend (crates/amd/{dialect,codegen}
//! before it moved to mlirpass, its last version fbe0c88) made on mlirpass 0.8.3 with the same features, byte for byte
//! (tests/kernels/*.expected).

use detour::rdna2_codegen::Codegen;

/// Built::dump with each segment's zero tail cut to `+N zeros` (the buffers are 64 KiB of zeros).
fn cut(dump: &str) -> String {
    dump.lines()
        .map(|l| match l.strip_prefix("segment ").map(|r| r.split(' ').collect::<Vec<_>>()) {
            Some(f) if f.len() == 3 => {
                let mut t = f[2].trim_end_matches('0').to_owned();
                if t.len() % 2 == 1 {
                    t.push('0');
                }
                format!("segment {} {} {t} +{} zeros\n", f[0], f[1], (f[2].len() - t.len()) / 2)
            }
            _ => format!("{l}\n"),
        })
        .collect()
}

fn check(src: &str, wants: [(u32, &str); 3], kernel: (u64, usize)) {
    let c = Codegen::default();
    for (groups, want) in wants {
        let built = c.image(src, groups).unwrap_or_else(|e| panic!("groups {groups}: {e}"));
        assert_eq!(cut(&built.dump()), want, "groups {groups}\n{}", built.module);
        assert_eq!((built.kernel_addr, built.kernel.len()), kernel);
    }
}

#[test]
fn ints_image_as_detours_own_backend() {
    let wants = [(1, include_str!("kernels/ints-g1.expected")), (4, include_str!("kernels/ints-g4.expected")), (1024, include_str!("kernels/ints-g1024.expected"))];
    check(include_str!("../../../intel/driver/examples/compute/ints.mlir"), wants, (0x100_0100, 392));
}

/// The first kernel of the copy shaders (the only one rdna2-compute dispatched).
#[test]
fn copy_image_as_detours_own_backend() {
    let wants = [(1, include_str!("kernels/copy-g1.expected")), (4, include_str!("kernels/copy-g4.expected")), (1024, include_str!("kernels/copy-g1024.expected"))];
    check(include_str!("../../../intel/driver/src/copies/copy.mlir"), wants, (0x100_0100, 136));
}
