use core::sync::atomic::{AtomicU64, Ordering};

use mlirformat::{Error, MlirOpt};
use mlirpass::{Body, Ctx, GlslDialect, Interfaces, PassManager, Passes};

/// What `Toolchain::program` writes ahead of the compiled program: this, the detour version, the GPU, each ending in
/// a newline
const PROGRAM_MAGIC: &str = "detour program";
/// The GPU programs are compiled for
const PROGRAM_GPU: &str = "xe3";

use crate::{DetourPasses, ImageDialect, Xe3Dialect, Xe3Draw};

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
        opt.load(ImageDialect);
        let mut passes = Passes::upstream();
        passes.with(DetourPasses::add);
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
        Ok(Compiled::new(body))
    }

    /// A program compiled ahead of time (a build script's work): `spirv`, MLIR text with a vertex and a fragment
    /// `spirv.module` at the top, compiled for Xe3, as the bytes `load` takes. The same text and detour give the
    /// same bytes
    pub fn program(&self, spirv: &str) -> Result<Vec<u8>, Error> {
        let body = Body::parse(Ctx::new(&self.opt, &self.interfaces), spirv)?;
        let compiled = self.compile(body)?;
        let text = self.show(&compiled.body);
        Ok(format!("{PROGRAM_MAGIC}\n{}\n{PROGRAM_GPU}\n{text}", env!("CARGO_PKG_VERSION")).into_bytes())
    }

    /// A program `program` compiled, read back without compiling anything: an error for bytes that are not one, or
    /// one compiled by another detour version or for another GPU
    pub fn load<'t>(&'t self, program: &'t [u8]) -> Result<Compiled<'t>, Error> {
        let text = core::str::from_utf8(program).map_err(|_| Error("not a detour program".into()))?;
        let mut lines = text.splitn(4, '\n');
        let (magic, version, gpu, body) = (lines.next(), lines.next(), lines.next(), lines.next());
        if magic != Some(PROGRAM_MAGIC) {
            return Err(Error("not a detour program".into()));
        }
        if version != Some(env!("CARGO_PKG_VERSION")) {
            return Err(Error(format!("a program compiled by detour {}, this is {}", version.unwrap_or("?"), env!("CARGO_PKG_VERSION"))));
        }
        if gpu != Some(PROGRAM_GPU) {
            return Err(Error(format!("a program compiled for {}, not {PROGRAM_GPU}", gpu.unwrap_or("?"))));
        }
        let body = Body::parse(Ctx::new(&self.opt, &self.interfaces), body.unwrap_or_default())?;
        Ok(Compiled::new(body))
    }
}

impl Default for Toolchain<'_> {
    fn default() -> Self {
        Toolchain::new()
    }
}

/// The id the next compiled program takes
static PROGRAMS: AtomicU64 = AtomicU64::new(0);

/// A program's machine code, in the body draws are lowered from
pub struct Compiled<'t> {
    pub body: Body<'t>,
    /// which program it is in the process (two toolchains' programs never share one): a device keeps the images of
    /// its draws by it
    pub id: u64,
}

impl<'t> Compiled<'t> {
    fn new(body: Body<'t>) -> Self {
        Compiled { body, id: PROGRAMS.fetch_add(1, Ordering::Relaxed) }
    }
}
