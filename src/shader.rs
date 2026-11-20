use core::sync::atomic::{AtomicU64, Ordering};

use mlirformat::{Error, MlirOpt, Op};
use mlirpass::{Body, Ctx, GlslDialect, Interfaces, PassManager, Passes};

/// What `Toolchain::program` writes ahead of the compiled program: this, the detour version, the GPU, each ending in
/// a newline
const PROGRAM_MAGIC: &str = "detour program";
/// The GPU programs are compiled for
const PROGRAM_GPU: &str = "xe3";

use crate::{DetourPasses, ImageDialect, Kernels, ShaderStage, Xe3Dialect, Xe3Draw, Xe3Kernel, Xe3Shader};

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

    /// The SPIR-V shaders at the top of `body` compiled by mlirpass for the device's GPU, to machine code
    pub fn compile(&self, mut body: Body<'_>) -> Result<Compiled, Error> {
        Xe3Draw::compile(&mut body, &self.passes).map_err(|e| Error(format!("{e}\n{}", self.show(&body))))?;
        let root = body.root;
        let (mut vs, mut fs) = (None, None);
        for k in Kernels::all(&body, root, Xe3Kernel::NAME) {
            let shader = Xe3Shader::of(&body, k)?;
            let slot = match shader.stage {
                ShaderStage::Vertex => &mut vs,
                ShaderStage::Fragment => &mut fs,
                s => return Err(Error(format!("a program of vertex and fragment shaders, not {s:?}"))),
            };
            if slot.replace(shader).is_some() {
                return Err(Error("a program of one vertex and one fragment shader".into()));
            }
        }
        match (vs, fs) {
            (Some(vs), Some(fs)) => Ok(Compiled::new(vs, fs)),
            _ => Err(Error("a program needs a vertex and a fragment shader".into())),
        }
    }

    /// A program compiled ahead of time (a build script's work): `spirv`, MLIR text with a vertex and a fragment
    /// `spirv.module` at the top, compiled for Xe3, as the bytes `load` takes. The same text and detour give the
    /// same bytes
    pub fn program(&self, spirv: &str) -> Result<Vec<u8>, Error> {
        let body = Body::parse(Ctx::new(&self.opt, &self.interfaces), spirv)?;
        let compiled = self.compile(body)?;
        let mut out = format!("{PROGRAM_MAGIC}\n{}\n{PROGRAM_GPU}\n", env!("CARGO_PKG_VERSION")).into_bytes();
        compiled.vs.write(&mut out);
        compiled.fs.write(&mut out);
        Ok(out)
    }

    /// A program `program` compiled, read back without compiling anything: an error for bytes that are not one, or
    /// one compiled by another detour version or for another GPU
    pub fn load(&self, program: &[u8]) -> Result<Compiled, Error> {
        let not = || Error("not a detour program".into());
        let mut lines = program.splitn(4, |b| *b == b'\n');
        let (magic, version, gpu, mut body) = (lines.next(), lines.next(), lines.next(), lines.next().ok_or_else(not)?);
        if magic != Some(PROGRAM_MAGIC.as_bytes()) {
            return Err(not());
        }
        let text = |b: Option<&[u8]>| b.map_or_else(|| "?".into(), |b| String::from_utf8_lossy(b).into_owned());
        if version != Some(env!("CARGO_PKG_VERSION").as_bytes()) {
            return Err(Error(format!("a program compiled by detour {}, this is {}", text(version), env!("CARGO_PKG_VERSION"))));
        }
        if gpu != Some(PROGRAM_GPU.as_bytes()) {
            return Err(Error(format!("a program compiled for {}, not {PROGRAM_GPU}", text(gpu))));
        }
        let vs = Xe3Shader::read(&mut body)?;
        let fs = Xe3Shader::read(&mut body)?;
        if !body.is_empty() || vs.stage != ShaderStage::Vertex || fs.stage != ShaderStage::Fragment {
            return Err(not());
        }
        Ok(Compiled::new(vs, fs))
    }
}

impl Default for Toolchain<'_> {
    fn default() -> Self {
        Toolchain::new()
    }
}

/// The id the next compiled program takes
static PROGRAMS: AtomicU64 = AtomicU64::new(0);

/// A program: its shaders' machine code and what a draw needs to know of them
#[derive(Clone, Debug)]
pub struct Compiled {
    pub vs: Xe3Shader,
    pub fs: Xe3Shader,
    /// which program it is in the process: a device keeps its code, and the shapes of its draws, by it
    pub id: u64,
}

impl Compiled {
    fn new(vs: Xe3Shader, fs: Xe3Shader) -> Self {
        Compiled { vs, fs, id: PROGRAMS.fetch_add(1, Ordering::Relaxed) }
    }
}
