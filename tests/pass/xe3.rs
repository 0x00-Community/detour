//! Xe3: each shader's interface and code

use std::fmt::Write as _;

use detour::{Isa, Shaders, Xe3Shader};

fn show(out: &mut String, s: &Xe3Shader) {
    writeln!(out, "{:?} inputs={} varyings={} bindings={} buffers={} system={}", s.stage, s.inputs, s.varyings, s.bindings, s.buffers, s.system).unwrap();
    super::hex(out, &s.code);
}

#[test]
fn cases() {
    super::check(Isa::Xe3, |out, shaders| {
        let Shaders::Xe3 { vs, fs } = shaders else { panic!("not Xe3") };
        show(out, vs);
        show(out, fs);
    });
}
