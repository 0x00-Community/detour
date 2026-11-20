//! Xe3 lowering: hand-written SPIR-V shaders and GLSL ones taken whole, their code against a reading of it

use mlirformat::MlirOpt;
use mlirpass::{Body, Ctx, Interfaces};
use detour::Xe3Dialect;

#[test]
fn shaders() {
    use std::fmt::Write as _;

    use mlirpass::{Heads, PassManager, Symbol};
    use detour::{ShaderStage, Xe3Code, Xe3Kernel};
    let mut opt = MlirOpt::upstream();
    opt.load(Xe3Dialect);
    let interfaces = Interfaces::upstream();
    let passes = super::passes();
    let pm = PassManager::parse(&passes, "builtin.module(convert-spirv-entry-points, canonicalize, cse, convert-gpu-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad)").unwrap();
    let cases = ["triangle", "texture", "push", "push_mixed", "uniform", "resolve", "mrt", "math"];

    let mut bad = Vec::new();
    for name in cases {
        let src = std::fs::read_to_string(super::path(&format!("xe3/shaders/{name}.mlir"))).unwrap();
        let mut body = Body::parse(Ctx::new(&opt, &interfaces), &src).unwrap();
        let root = body.root;
        pm.run(&mut body, root).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut got = String::new();
        for k in Xe3Kernel::all(&body, root) {
            let h = body.head::<Xe3Kernel>(k).unwrap();
            let stage = h.stage.map(|s| match s {
                ShaderStage::Vertex => "Vertex",
                ShaderStage::Fragment => "Fragment",
            });
            let stage = stage.map_or("None".to_owned(), |s| format!("Some({s})"));
            writeln!(got, "kernel {} stage={stage} simd={:?} payload={:?} inputs={:?} varyings={:?} targets={:?} omask={:?} bindings={:?} buffers={:?} push={:?} reserve={:?} pad={:?}",
                Symbol::defined(&h.sym_name).unwrap(), h.simd, h.payload, h.inputs, h.varyings, h.targets, h.omask, h.bindings, h.buffers, h.push, h.reserve, h.pad).unwrap();
            super::hex(&mut got, &Xe3Code::padded(&body.operation(k)).unwrap());
        }
        if !super::matches(&format!("xe3/shaders/{name}.expected"), &got) {
            bad.push(name);
        }
    }
    assert!(bad.is_empty(), "compiled otherwise than read through: {bad:?}");
}

/// GLSL shaders that the Xe3 backend takes whole (control flow and locals flattened, promoted and lifted back to scf
/// before convert-gpu-to-xe3), their code as it was read through: uniforms, built-ins, sampling, and per-pixel integer
/// and float arithmetic and loops (`alu`)
#[test]
fn glsl_shaders() {
    use mlirpass::{GlslDialect, GlslReader, GlslStage, PassManager};
    use detour::Xe3Shader;
    let mut opt = MlirOpt::upstream();
    opt.load(GlslDialect);
    opt.load(Xe3Dialect);
    let interfaces = Interfaces::upstream();
    let passes = super::passes();
    let glsl = PassManager::parse(&passes, "builtin.module(glsl-link, glsl-default-block{binding=7}, convert-glsl-to-spirv)").unwrap();
    let xe3 = PassManager::parse(&passes, detour::Xe3Draw::SHADERS).unwrap();
    let cases = [
        ("uniforms", "shader.frag"),
        ("builtins", "bits.frag"),
        ("builtins", "const_cnorm.frag"),
        ("sampling", "offset.frag"),
        ("sampling", "offset_bias.frag"),
        ("alu", "ints.frag"),
        ("alu", "loops.frag"),
        ("alu", "floats.frag"),
        ("alu", "calls.frag"),
    ];
    let mut bad = Vec::new();
    for (dir, frag) in cases {
        let read = |f: &str| std::fs::read_to_string(super::path(&format!("xe3/glsl/{dir}/{f}"))).unwrap();
        let mut body = Body::new(Ctx::new(&opt, &interfaces));
        GlslReader::read(&mut body, &read("shader.vert"), GlslStage::Vertex).unwrap();
        GlslReader::read(&mut body, &read(frag), GlslStage::Fragment).unwrap();
        let r = body.root;
        glsl.run(&mut body, r).unwrap();
        let shown = body.show(&opt, false, |_| true);
        let (_, modules) = shown.split_once('\n').unwrap();
        let text = format!("module attributes {{gpu.container_module}} {{\ngpu.module @shaders [#xevm.target<chip = \"ptl\">] {{\n{modules}\n}}\n");
        let mut gpu = Body::parse(Ctx::new(&opt, &interfaces), &text).unwrap();
        let r = gpu.root;
        if let Err(e) = xe3.run(&mut gpu, r) {
            bad.push(format!("{dir}/{frag}: {e}"));
            continue;
        }
        let mut got = String::new();
        for k in detour::Xe3Kernel::all(&gpu, r) {
            let s = Xe3Shader::of(&gpu, k).unwrap();
            show(&mut got, &s);
        }
        if !super::matches(&format!("xe3/glsl/{dir}/{}.expected", frag.trim_end_matches(".frag")), &got) {
            bad.push(format!("{dir}/{frag}: compiled otherwise than read through"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// acdes's four page programs compiled for Xe3 as they were when their code was read through
#[test]
fn acdes_programs() {
    use detour::{Isa, Shaders, Toolchain};
    let tools = Toolchain::new(Isa::Xe3);
    let mut bad = Vec::new();
    for (name, src) in super::programs() {
        let program = tools.program(&src).unwrap_or_else(|e| panic!("{name}: {}", e.0));
        let Shaders::Xe3 { vs, fs } = tools.load(&program).unwrap().shaders else { panic!("{name}: not Xe3") };
        let mut got = String::new();
        show(&mut got, &vs);
        show(&mut got, &fs);
        if !super::matches(&format!("xe3/programs/{name}.expected"), &got) {
            bad.push(name);
        }
    }
    assert!(bad.is_empty(), "compiled otherwise than read through: {bad:?}");
}

fn show(out: &mut String, s: &detour::Xe3Shader) {
    use std::fmt::Write as _;
    writeln!(out, "{:?} inputs={} varyings={} targets={} omask={} bindings={} buffers={} system={}", s.stage, s.inputs, s.varyings, s.targets, s.omask, s.bindings, s.buffers, s.system).unwrap();
    super::hex(out, &s.code);
}
