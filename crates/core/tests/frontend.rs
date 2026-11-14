//! The three shader forms of the triangle (func / gpu / spirv) are the same module after the frontend.

use detour::base::{Compiler, FRONTEND};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/crates/intel/driver/examples/triangle/");

#[test]
fn three_forms_one_module() {
    let c = Compiler::default();
    let forms = ["func", "gpu", "spirv"];
    let srcs: Vec<String> = forms.iter().map(|f| std::fs::read_to_string(format!("{DIR}{f}.mlir")).unwrap()).collect();
    let out: Vec<String> = forms
        .iter()
        .zip(&srcs)
        .map(|(f, src)| {
            let mut body = c.parse(src).unwrap_or_else(|e| panic!("{f}: {e}"));
            c.run(&mut body, &format!("builtin.module({FRONTEND})")).unwrap_or_else(|e| panic!("{f}: {e}"));
            c.show(&body)
        })
        .collect();
    println!("{}", out[0]);
    assert_eq!(out[0], out[1], "gpu");
    assert_eq!(out[0], out[2], "spirv");
}
