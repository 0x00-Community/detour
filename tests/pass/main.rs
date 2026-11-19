//! detour's passes: each GPU's lowering and command streams against what Mesa and acgpu ran

mod rdna2;
mod xe3;

use mlirpass::Passes;

/// mlirpass's passes and detour's
fn passes() -> Passes {
    let mut p = Passes::upstream();
    p.with(detour::DetourPasses::add);
    p
}
