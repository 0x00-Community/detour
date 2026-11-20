//! detour's passes: each GPU's lowering against what Mesa, acgpu and a reading of the code ran

mod rdna2;
mod xe3;

use mlirpass::Passes;

/// mlirpass's passes and detour's
fn passes() -> Passes {
    let mut p = Passes::upstream();
    p.with(detour::DetourPasses::add);
    p
}
