use mlirformat::{Attribute, Error, MlirOpt, NamedAttr, Op};
use mlirpass::{Body, Ctx, GlslDialect, Heads, Interfaces, Kernels, OpData, OpId, PassManager, Passes, ShaderStage, Xe3Dialect, Xe3Image, Xe3Kernel, ImageTarget};

/// SPIR-V shaders (from GLSL or a Vulkan shader module) to Xe3: target, entry points, inlining, Xe3 instructions,
/// registers, control flow, scoreboard, branches, padding
const XE3: &str = "builtin.module(xevm-attach-target{chip=ptl}, convert-spirv-entry-points, gpu-inline-calls, canonicalize, cse, convert-gpu-to-xe3, xe3-regalloc, xe3-control, xe3-swsb, xe3-branches, xe3-pad)";

/// The gpu.module the SPIR-V modules move into: convert-spirv-entry-points only takes those inside one
const SHADERS: &str = "shaders";

/// mlirpass's dialects, interfaces and passes: one per device, every program is compiled with it
pub struct Toolchain<'i> {
    opt: MlirOpt,
    interfaces: Interfaces<'i>,
    passes: Passes,
    xe3: PassManager,
}

impl Toolchain<'_> {
    pub fn new() -> Result<Self, Error> {
        let mut opt = MlirOpt::upstream();
        opt.load(GlslDialect);
        opt.load(Xe3Dialect);
        let passes = Passes::upstream();
        let xe3 = PassManager::parse(&passes, XE3)?;
        Ok(Toolchain { opt, interfaces: Interfaces::upstream(), passes, xe3 })
    }

    pub fn body(&self) -> Body<'_> {
        Body::new(Ctx::new(&self.opt, &self.interfaces))
    }

    pub fn pipeline(&self, text: &str) -> Result<PassManager, Error> {
        PassManager::parse(&self.passes, text)
    }

    pub fn show(&self, body: &Body<'_>) -> String {
        body.show(&self.opt, false, |_| true)
    }

    /// The top-level spirv.module ops of `body`, one per stage, compiled to Xe3 kernels
    pub fn xe3<'t>(&'t self, mut body: Body<'t>) -> Result<Xe3Program<'t>, Error> {
        let root = body.root;
        let top = body.top_block(root);
        let modules: Vec<OpId> = body.ops(top).filter(|&m| &*body[m].name == "spirv.module").collect();
        body[root].attributes.push(NamedAttr { name: "gpu.container_module".into(), value: None });
        let shaders = body.create(OpData::build("gpu.module").with("sym_name", Attribute::str(SHADERS)), &[]);
        let region = body.add_region(shaders);
        let block = body.add_block(region, &[]);
        body.insert(shaders, top, None);
        for m in modules {
            body.detach(m);
            body.insert(m, block, None);
        }
        self.xe3.run(&mut body, root).map_err(|e| Error(format!("{e}\n{}", self.show(&body))))?;
        let kernels = Kernels::all(&body, root, Xe3Kernel::NAME);
        Ok(Xe3Program { body, kernels })
    }
}

/// A program's Xe3 kernels, in the body the runtime places them from
pub struct Xe3Program<'t> {
    pub body: Body<'t>,
    pub kernels: Vec<OpId>,
}

impl<'t> Xe3Program<'t> {
    pub fn kernel(&self, stage: ShaderStage) -> Option<OpId> {
        self.kernels.iter().copied().find(|&k| self.body.head::<Xe3Kernel>(k).is_ok_and(|h| h.stage == Some(stage)))
    }

    pub fn head(&self, k: OpId) -> Result<Xe3Kernel<'t>, Error> {
        self.body.head::<Xe3Kernel>(k)
    }

    /// The kernel's bytes as placed in memory (padded for the EU's prefetch)
    pub fn code(&self, k: OpId) -> Result<Vec<u8>, Error> {
        Xe3Image.item(&self.body, k)
    }
}
