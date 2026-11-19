//! The passes detour adds to mlirpass: Xe3's lowering from `gpu.func` kernels and SPIR-V
//! shaders to its instructions, registers, waits and command streams, and what they share (`gpu`).
//! `Passes::with(DetourPasses::add)` registers them.

mod gpu;
mod xe3;

pub use gpu::*;
pub use xe3::*;

/// detour's passes, added to mlirpass's (`Passes::with(DetourPasses::add)`); its
/// `gpu-module-to-binary` takes the place of mlirpass's, encoding SPIR-V
pub struct DetourPasses;

impl DetourPasses {
    pub fn add(p: &mut mlirpass::Passes) {
        p
            .register::<ConvertGpuToXe3>()
            .register::<GpuInlineCalls>()
            .register::<GpuModuleToBinary>()
            .register::<Xe3Branches>()
            .register::<Xe3Control>()
            .register::<Xe3End>()
            .register::<Xe3Layout>()
            .register::<Xe3Pad>()
            .register::<Xe3Regalloc>()
            .register::<Xe3Swsb>();
    }
}
