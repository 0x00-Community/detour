use core::sync::atomic::{AtomicU64, Ordering};

use mlirformat::{Error, MlirOpt};
use mlirpass::{Body, Ctx, GlslDialect, Interfaces, PassManager, Passes};

/// What `Toolchain::program` writes ahead of the compiled program: this, the detour version, the GPU, each ending in
/// a newline
const PROGRAM_MAGIC: &str = "detour program";
use crate::{DetourPasses, Rdna2Graphics, Rdna2Shader, ShaderStage, Xe3Dialect, Xe3Draw, Xe3Kernel, Xe3Shader};

/// The instruction set a program is compiled for: a GPU detour draws on
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Isa {
    /// Intel Xe3 (Panther Lake)
    Xe3,
    /// AMD RDNA2 (gfx10.3: the Radeon 610M)
    Rdna2,
}

impl Isa {
    /// Its name in a program's header
    pub const fn name(self) -> &'static str {
        match self {
            Isa::Xe3 => "xe3",
            Isa::Rdna2 => "rdna2",
        }
    }
}

/// mlirpass's dialects, interfaces and passes, and the instruction set it compiles for: one per device, every
/// program is compiled with it
pub struct Toolchain<'i> {
    opt: MlirOpt,
    interfaces: Interfaces<'i>,
    passes: Passes,
    isa: Isa,
}

impl Toolchain<'_> {
    pub fn new(isa: Isa) -> Self {
        let mut opt = MlirOpt::upstream();
        opt.load(GlslDialect);
        opt.load(Xe3Dialect);
        let mut passes = Passes::upstream();
        passes.with(DetourPasses::add);
        Toolchain { opt, interfaces: Interfaces::upstream(), passes, isa }
    }

    pub fn isa(&self) -> Isa {
        self.isa
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
    pub fn compile(&self, body: Body<'_>) -> Result<Compiled, Error> {
        match self.isa {
            Isa::Xe3 => self.compile_xe3(body),
            Isa::Rdna2 => self.compile_rdna2(body),
        }
    }

    fn compile_rdna2(&self, mut body: Body<'_>) -> Result<Compiled, Error> {
        let (vs, fs) = Rdna2Graphics::compile(&mut body, &self.passes).map_err(|e| Error(format!("{e}\n{}", self.show(&body))))?;
        Ok(Compiled::new(Shaders::Rdna2 { vs, fs }))
    }

    fn compile_xe3(&self, mut body: Body<'_>) -> Result<Compiled, Error> {
        Xe3Draw::compile(&mut body, &self.passes).map_err(|e| Error(format!("{e}\n{}", self.show(&body))))?;
        let root = body.root;
        let (mut vs, mut fs) = (None, None);
        for k in Xe3Kernel::all(&body, root) {
            let shader = Xe3Shader::of(&body, k)?;
            let slot = match shader.stage {
                ShaderStage::Vertex => &mut vs,
                ShaderStage::Fragment => &mut fs,
            };
            if slot.replace(shader).is_some() {
                return Err(Error("a program of one vertex and one fragment shader".into()));
            }
        }
        match (vs, fs) {
            (Some(vs), Some(fs)) => Ok(Compiled::new(Shaders::Xe3 { vs, fs })),
            _ => Err(Error("a program needs a vertex and a fragment shader".into())),
        }
    }

    /// A program compiled ahead of time (a build script's work): `spirv`, MLIR text with a vertex and a fragment
    /// `spirv.module` at the top, compiled for the toolchain's instruction set, as the bytes `load` takes. The same text and detour give the
    /// same bytes
    pub fn program(&self, spirv: &str) -> Result<Vec<u8>, Error> {
        let body = Body::parse(Ctx::new(&self.opt, &self.interfaces), spirv)?;
        let compiled = self.compile(body)?;
        let mut out = format!("{PROGRAM_MAGIC}\n{}\n{}\n", env!("CARGO_PKG_VERSION"), self.isa.name()).into_bytes();
        match &compiled.shaders {
            Shaders::Xe3 { vs, fs } => {
                vs.write(&mut out);
                fs.write(&mut out);
            }
            Shaders::Rdna2 { vs, fs } => {
                vs.write(&mut out);
                fs.write(&mut out);
            }
        }
        Ok(out)
    }

    /// A program `program` compiled, read back without compiling anything: an error for bytes that are not one, or
    /// one compiled by another detour version or for another instruction set
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
        if gpu != Some(self.isa.name().as_bytes()) {
            return Err(Error(format!("a program compiled for {}, not {}", text(gpu), self.isa.name())));
        }
        let (stages, shaders) = match self.isa {
            Isa::Xe3 => {
                let (vs, fs) = (Xe3Shader::read(&mut body)?, Xe3Shader::read(&mut body)?);
                ((vs.stage, fs.stage), Shaders::Xe3 { vs, fs })
            }
            Isa::Rdna2 => {
                let (vs, fs) = (Rdna2Shader::read(&mut body)?, Rdna2Shader::read(&mut body)?);
                ((vs.stage, fs.stage), Shaders::Rdna2 { vs, fs })
            }
        };
        if !body.is_empty() || stages != (ShaderStage::Vertex, ShaderStage::Fragment) {
            return Err(not());
        }
        Ok(Compiled::new(shaders))
    }
}

/// The id the next compiled program takes
static PROGRAMS: AtomicU64 = AtomicU64::new(0);

/// A program's vertex and fragment shader, for one instruction set
#[derive(Clone, Debug)]
pub enum Shaders {
    Xe3 { vs: Xe3Shader, fs: Xe3Shader },
    Rdna2 { vs: Rdna2Shader, fs: Rdna2Shader },
}

/// A program: its shaders' machine code and what a draw needs to know of them
#[derive(Clone, Debug)]
pub struct Compiled {
    pub shaders: Shaders,
    /// which program it is in the process: a device keeps its code, and the shapes of its draws, by it
    pub id: u64,
}

impl Compiled {
    pub fn new(shaders: Shaders) -> Self {
        Compiled { shaders, id: PROGRAMS.fetch_add(1, Ordering::Relaxed) }
    }

    pub fn isa(&self) -> Isa {
        match self.shaders {
            Shaders::Xe3 { .. } => Isa::Xe3,
            Shaders::Rdna2 { .. } => Isa::Rdna2,
        }
    }
}
