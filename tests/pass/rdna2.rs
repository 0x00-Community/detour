//! RDNA2 shaders: acdes's four page programs compiled as they were when their code was read through (llvm-mc's
//! disassembly; `DETOUR_BLESS=1` rewrites the expected files after such a reading)

use std::fmt::Write as _;

use detour::{Isa, Rdna2Shader, Shaders, Toolchain};

fn show(out: &mut String, name: &str, s: &Rdna2Shader) {
    writeln!(out, "{name} sgprs={} vgprs={} inputs={} varyings={} ubos={:?} textures={:?} ps_input={:#x}", s.sgprs, s.vgprs, s.inputs, s.varyings, s.ubos, s.textures, s.ps_input).unwrap();
    for line in s.code.chunks(32) {
        writeln!(out, "  {}", line.iter().map(|b| format!("{b:02x}")).collect::<String>()).unwrap();
    }
}

#[test]
fn acdes_programs() {
    let tools = Toolchain::new(Isa::Rdna2);
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pass/rdna2");
    let mut bad = Vec::new();
    for name in ["solid", "ramp", "shadow", "image"] {
        let src = std::fs::read_to_string(dir.join(format!("{name}.mlir"))).unwrap();
        let program = tools.program(&src).unwrap_or_else(|e| panic!("{name}: {}", e.0));
        let Shaders::Rdna2 { vs, fs } = tools.load(&program).unwrap().shaders else { panic!("{name}: not RDNA2") };
        let mut got = String::new();
        show(&mut got, "vs", &vs);
        show(&mut got, "fs", &fs);
        let expected = dir.join(format!("{name}.expected"));
        if std::env::var_os("DETOUR_BLESS").is_some() {
            std::fs::write(&expected, &got).unwrap();
        } else if std::fs::read_to_string(&expected).ok().as_deref() != Some(&got) {
            bad.push(name);
        }
    }
    assert!(bad.is_empty(), "compiled otherwise than read through: {bad:?}");
}
