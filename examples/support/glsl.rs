//! How the examples write their shaders: GLSL made SPIR-V MLIR text by mlirpass's front end, the text detour compiles

use mlirformat::MlirOpt;
use mlirpass::{Body, Ctx, GlslDialect, GlslReader, GlslStage, Interfaces, PassManager, Passes};

/// A vertex and a fragment shader in GLSL as SPIR-V MLIR text
pub fn spirv(vert: &str, frag: &str) -> Result<String, String> {
    let mut opt = MlirOpt::upstream();
    opt.load(GlslDialect);
    let interfaces = Interfaces::upstream();
    let mut body = Body::new(Ctx::new(&opt, &interfaces));
    GlslReader::read(&mut body, vert, GlslStage::Vertex).map_err(|e| format!("vertex shader: {}", e.0))?;
    GlslReader::read(&mut body, frag, GlslStage::Fragment).map_err(|e| format!("fragment shader: {}", e.0))?;
    let root = body.root;
    PassManager::parse(&Passes::upstream(), "builtin.module(glsl-link, convert-glsl-to-spirv{client=vulkan})")
        .and_then(|p| p.run(&mut body, root))
        .map_err(|e| format!("to SPIR-V: {}", e.0))?;
    Ok(body.show(&opt, false, |_| true))
}
