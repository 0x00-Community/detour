//! RDNA2: each shader's registers, interface and code (llvm-mc disassembles it)

use std::fmt::Write as _;

use detour::{Isa, Rdna2Shader, Shaders};

fn show(out: &mut String, name: &str, s: &Rdna2Shader) {
    writeln!(out, "{name} sgprs={} vgprs={} inputs={} varyings={} ubos={:?} textures={:?} ps_input={:#x}", s.sgprs, s.vgprs, s.inputs, s.varyings, s.ubos, s.textures, s.ps_input).unwrap();
    super::hex(out, &s.code);
}

#[test]
fn cases() {
    super::check(Isa::Rdna2, |out, shaders| {
        let Shaders::Rdna2 { vs, fs } = shaders else { panic!("not RDNA2") };
        show(out, "vs", vs);
        show(out, "fs", fs);
    });
}
