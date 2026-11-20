//! RDNA2 shaders: acdes's four page programs compiled as they were when their code was read through (llvm-mc's
//! disassembly)

use std::fmt::Write as _;

use detour::{Isa, Rdna2Shader, Shaders, Toolchain};

fn show(out: &mut String, name: &str, s: &Rdna2Shader) {
    writeln!(out, "{name} sgprs={} vgprs={} inputs={} varyings={} ubos={:?} textures={:?} ps_input={:#x}", s.sgprs, s.vgprs, s.inputs, s.varyings, s.ubos, s.textures, s.ps_input).unwrap();
    super::hex(out, &s.code);
}

#[test]
fn acdes_programs() {
    let tools = Toolchain::new(Isa::Rdna2);
    let mut bad = Vec::new();
    for (name, src) in super::programs() {
        let program = tools.program(&src).unwrap_or_else(|e| panic!("{name}: {}", e.0));
        let Shaders::Rdna2 { vs, fs } = tools.load(&program).unwrap().shaders else { panic!("{name}: not RDNA2") };
        let mut got = String::new();
        show(&mut got, "vs", &vs);
        show(&mut got, "fs", &fs);
        if !super::matches(&format!("rdna2/{name}.expected"), &got) {
            bad.push(name);
        }
    }
    assert!(bad.is_empty(), "compiled otherwise than read through: {bad:?}");
}
