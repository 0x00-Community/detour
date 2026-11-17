//! Xe3 lowering: kernels as acgpu compiled them, GLSL shaders taken whole (encoding and genxml
//! packing are mlirbinary's tests)

use mlirformat::MlirOpt;
use mlirpass::{Body, Ctx, Interfaces};
use detour::Xe3Dialect;

#[test]
fn kernels_as_acgpu() {
    use std::fmt::Write as _;

    use mlirformat::Op;
    use mlirpass::{Heads, PassManager, Symbol};
    use detour::{Kernels, ShaderStage, Xe3Code, Xe3Kernel};
    let mut opt = MlirOpt::upstream();
    opt.load(Xe3Dialect);
    let interfaces = Interfaces::upstream();
    let passes = super::passes();
    let pm = PassManager::parse(&passes, "builtin.module(convert-spirv-entry-points, gpu-inline-calls, canonicalize, cse, convert-gpu-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad)").unwrap();
    let cases = [
        ("ints", include_str!("xe3/kernels/ints.mlir"), include_str!("xe3/kernels/ints.expected")),
        ("atomics", include_str!("xe3/kernels/atomics.mlir"), include_str!("xe3/kernels/atomics.expected")),
        ("loops", include_str!("xe3/kernels/loops.mlir"), include_str!("xe3/kernels/loops.expected")),
        ("math64", include_str!("xe3/kernels/math64.mlir"), include_str!("xe3/kernels/math64.expected")),
        ("workgroups", include_str!("xe3/kernels/workgroups.mlir"), include_str!("xe3/kernels/workgroups.expected")),
        ("copy", include_str!("xe3/kernels/copy.mlir"), include_str!("xe3/kernels/copy.expected")),
        ("triangle", include_str!("xe3/kernels/triangle.mlir"), include_str!("xe3/kernels/triangle.expected")),
        ("texture", include_str!("xe3/kernels/texture.mlir"), include_str!("xe3/kernels/texture.expected")),
        ("push", include_str!("xe3/kernels/push.mlir"), include_str!("xe3/kernels/push.expected")),
        ("push_mixed", include_str!("xe3/kernels/push_mixed.mlir"), include_str!("xe3/kernels/push_mixed.expected")),
        ("uniform", include_str!("xe3/kernels/uniform.mlir"), include_str!("xe3/kernels/uniform.expected")),
        ("resolve", include_str!("xe3/kernels/resolve.mlir"), include_str!("xe3/kernels/resolve.expected")),
        ("mrt", include_str!("xe3/kernels/mrt.mlir"), include_str!("xe3/kernels/mrt.expected")),
        ("math", include_str!("xe3/kernels/math.mlir"), include_str!("xe3/kernels/math.expected")),
    ];
    let mut bad = Vec::new();
    for (name, src, want) in cases {
        let mut body = Body::parse(Ctx::new(&opt, &interfaces), src).unwrap();
        let root = body.root;
        pm.run(&mut body, root).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut got = String::new();
        for k in Kernels::all(&body, root, Xe3Kernel::NAME) {
            let h = body.head::<Xe3Kernel>(k).unwrap();
            let stage = h.stage.map(|s| match s {
                ShaderStage::Compute => "Compute",
                ShaderStage::Vertex => "Vertex",
                ShaderStage::Fragment => "Fragment",
            });
            let stage = stage.map_or("None".to_owned(), |s| format!("Some({s})"));
            writeln!(got, "kernel {} stage={stage} simd={:?} payload={:?} inputs={:?} varyings={:?} targets={:?} omask={:?} imask={:?} bindings={:?} buffers={:?} push={:?} reserve={:?} threads={:?} shared={:?} params={:?} pad={:?} align={:?}",
                Symbol::defined(&h.sym_name).unwrap(), h.simd, h.payload, h.inputs, h.varyings, h.targets, h.omask, h.imask, h.bindings, h.buffers, h.push, h.reserve, h.threads, h.shared, h.params, h.pad, h.align).unwrap();
            for line in Xe3Code::padded(&body.operation(k)).unwrap().chunks(32) {
                writeln!(got, "  {}", line.iter().map(|b| format!("{b:02x}")).collect::<String>()).unwrap();
            }
        }
        if got != want {
            bad.push(name);
        }
    }
    assert!(bad.is_empty(), "differ from acgpu's kernels: {bad:?}");
}

/// GLSL shaders from tests/gl that the Xe3 backend takes whole: control flow and locals are
/// flattened, promoted and lifted back to scf before convert-gpu-to-xe3.
#[test]
fn glsl_reaches_xe3() {
    use mlirpass::{GlslDialect, GlslReader, GlslStage, PassManager};
    let mut opt = MlirOpt::upstream();
    opt.load(GlslDialect);
    opt.load(Xe3Dialect);
    let interfaces = Interfaces::upstream();
    let passes = super::passes();
    let glsl = PassManager::parse(&passes, "builtin.module(glsl-link, glsl-default-block{binding=7}, convert-glsl-to-spirv)").unwrap();
    let xe3 = PassManager::parse(&passes, "builtin.module(convert-spirv-entry-points, gpu-inline-calls, convert-spirv-cf-to-cf, sroa, mem2reg, lift-cf-to-scf, canonicalize, cse, convert-gpu-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad)").unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pass/xe3/glsl");
    let cases = [("uniforms", "shader.frag"), ("builtins", "bits.frag"), ("builtins", "const_cnorm.frag"), ("sampling", "offset.frag"), ("sampling", "offset_bias.frag")];
    let mut bad = Vec::new();
    for (dir, frag) in cases {
        let read = |f: &str| std::fs::read_to_string(root.join(dir).join(f)).unwrap();
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
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
