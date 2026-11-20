//! The passes detour adds to mlirpass: each GPU's lowering from `gpu.func` kernels and SPIR-V
//! shaders to its instructions, registers, waits and command streams, and what they share (`gpu`).
//! `Passes::with(DetourPasses::add)` registers them.

mod gpu;
mod rdna2;
mod xe3;

pub use gpu::*;
pub use rdna2::*;
pub use xe3::*;

/// detour's passes, added to mlirpass's (`Passes::with(DetourPasses::add)`)
pub struct DetourPasses;

impl DetourPasses {
    pub fn add(p: &mut mlirpass::Passes) {
        p
            .register::<ConvertGpuToXe3>()
            .register::<GpuInlineCalls>()
            .register::<GpuLocalArrays>()
            .register::<Xe3Branches>()
            .register::<Xe3Control>()
            .register::<Xe3Pad>()
            .register::<Xe3Regalloc>()
            .register::<Xe3Swsb>();
    }
}
