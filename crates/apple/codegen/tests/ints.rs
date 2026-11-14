//! detour's compute/ints shader through the AGX backend on mlirpass's AGX target, dispatched over 1, 4 and 1024
//! workgroups: the whole image (CDM stream, pipeline USC words, parameters, kernel; the buffers' places) is what
//! detour's own AGX backend (crates/apple/{dialect,codegen} before it moved to mlirpass, its last version 4a0ead0)
//! made on the same mlirpass (0.8.3) with the same features, byte for byte (tests/ints/g*.expected).

use detour::agx_codegen::Codegen;

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

#[test]
fn ints_image_as_detours_own_backend() {
    let src = include_str!("../../../intel/driver/examples/compute/ints.mlir");
    let c = Codegen::default();
    for (groups, want) in [(1, include_str!("ints/g1.expected")), (4, include_str!("ints/g4.expected")), (1024, include_str!("ints/g1024.expected"))] {
        let built = c.image(src, groups).unwrap_or_else(|e| panic!("groups {groups}: {e}"));
        assert_eq!(cut(&built.dump()), want, "groups {groups}\n{}", built.module);
        assert_eq!((built.kernel_addr, built.kernel.len()), (0x1_0000_0100, 594));
    }
}
