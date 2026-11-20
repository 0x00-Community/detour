//! detour's compiler: every case, SPIR-V MLIR text with a vertex and a fragment `spirv.module` (`spirv/`: what each
//! covers on its first lines; `programs/`: acdes's four page programs), compiled for each GPU as `Toolchain::compile`
//! compiles it. Its code, or why the GPU refuses it, is held against `<gpu>/<case>.expected`: a change to it is one to
//! read through, then to write with `DETOUR_BLESS=1`. Whether the code draws right is the examples' work, on the GPU.

mod rdna2;
mod xe3;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use detour::{Isa, Shaders, Toolchain};

/// A file under tests/pass
fn path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pass").join(rel)
}

/// Every case: `spirv/NAME` and `programs/NAME`, and its text
fn cases() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for dir in ["spirv", "programs"] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path(dir)).unwrap().map(|e| e.unwrap().path()).collect();
        files.sort();
        for f in files {
            let name = format!("{dir}/{}", f.file_stem().unwrap().to_str().unwrap());
            out.push((name, std::fs::read_to_string(&f).unwrap()));
        }
    }
    out
}

/// `bytes` in lines of 32, hex
fn hex(out: &mut String, bytes: &[u8]) {
    for line in bytes.chunks(32) {
        writeln!(out, "  {}", line.iter().map(|b| format!("{b:02x}")).collect::<String>()).unwrap();
    }
}

/// Every case compiled for `isa`, `show` writing its shaders, held against `<gpu>/<case>.expected` (with
/// `DETOUR_BLESS` set, written there)
fn check(isa: Isa, show: impl Fn(&mut String, &Shaders)) {
    let tools = Toolchain::new(isa);
    let mut bad = Vec::new();
    for (name, text) in cases() {
        let mut got = String::new();
        match tools.compile(&text) {
            Ok(c) => show(&mut got, &c.shaders),
            Err(e) => writeln!(got, "refused: {}", e.0.lines().next().unwrap_or("")).unwrap(),
        }
        let expected = path(&format!("{}/{name}.expected", isa.name()));
        if std::env::var_os("DETOUR_BLESS").is_some() {
            std::fs::create_dir_all(expected.parent().unwrap()).unwrap();
            std::fs::write(&expected, &got).unwrap();
        } else if std::fs::read_to_string(&expected).ok().as_deref() != Some(&*got) {
            bad.push(name);
        }
    }
    assert!(bad.is_empty(), "compiled otherwise than {:?} expects: {bad:?}", isa);
}
