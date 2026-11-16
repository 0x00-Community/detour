use mlirformat::{Error, MlirOpt};
use mlirpass::{Body, Ctx, GlslDialect, Interfaces, PassManager, Passes};

use crate::{Xe3Dialect, Xe3Draw, Xe3Passes};

/// mlirpass's dialects, interfaces and passes: one per device, every program is compiled with it
pub struct Toolchain<'i> {
    opt: MlirOpt,
    interfaces: Interfaces<'i>,
    passes: Passes,
}

impl Toolchain<'_> {
    pub fn new() -> Self {
        let mut opt = MlirOpt::upstream();
        opt.load(GlslDialect);
        opt.load(Xe3Dialect);
        let mut passes = Passes::upstream();
        passes.with(Xe3Passes::add);
        Toolchain { opt, interfaces: Interfaces::upstream(), passes }
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

    /// The SPIR-V shaders at the top of `body` compiled by mlirpass for the device's GPU
    pub fn compile<'t>(&'t self, mut body: Body<'t>) -> Result<Compiled<'t>, Error> {
        Xe3Draw::compile(&mut body, &self.passes).map_err(|e| Error(format!("{e}\n{}", self.show(&body))))?;
        Ok(Compiled { body })
    }
}

impl Default for Toolchain<'_> {
    fn default() -> Self {
        Toolchain::new()
    }
}

/// A program's machine code, in the body draws are lowered from
pub struct Compiled<'t> {
    pub body: Body<'t>,
}
