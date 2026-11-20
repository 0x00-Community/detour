//! detour's passes: each GPU's lowering against what Mesa and acgpu ran, and its code against a reading of it
//! (`DETOUR_BLESS=1` rewrites the expected files after such a reading)

mod rdna2;
mod xe3;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mlirpass::Passes;

/// mlirpass's passes and detour's
fn passes() -> Passes {
    let mut p = Passes::upstream();
    p.with(detour::DetourPasses::add);
    p
}

/// A file under tests/pass
fn path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pass").join(rel)
}

/// acdes's four page programs (solid, ramp, shadow, image): their name and MLIR
fn programs() -> Vec<(&'static str, String)> {
    ["solid", "ramp", "shadow", "image"].into_iter().map(|n| (n, std::fs::read_to_string(path(&format!("programs/{n}.mlir"))).unwrap())).collect()
}

/// `bytes` in lines of 32, hex
fn hex(out: &mut String, bytes: &[u8]) {
    for line in bytes.chunks(32) {
        writeln!(out, "  {}", line.iter().map(|b| format!("{b:02x}")).collect::<String>()).unwrap();
    }
}

/// Whether `got` is what `expected` (a file under tests/pass) holds; with DETOUR_BLESS set, it is made to
fn matches(expected: &str, got: &str) -> bool {
    let p = path(expected);
    if std::env::var_os("DETOUR_BLESS").is_some() {
        std::fs::write(&p, got).unwrap();
        return true;
    }
    std::fs::read_to_string(&p).ok().as_deref() == Some(got)
}
