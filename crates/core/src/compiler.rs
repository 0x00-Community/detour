mod build;
mod clear;
mod cube;
mod dialect;
mod gpu;
mod sink;
mod spirv;
mod to_gpu;

pub use build::*;
pub use clear::*;
pub use cube::*;
pub use dialect::*;
pub use gpu::*;
pub use sink::*;
pub use spirv::*;
pub use to_gpu::*;

use mlirformat::{Error, MlirOpt};
use mlirpass::{Body, Ctx, Interfaces, PassManager, Passes};

pub struct Compiler<'i> {
    pub opt: MlirOpt,
    pub interfaces: Interfaces<'i>,
    pub passes: Passes,
}

impl Default for Compiler<'_> {
    fn default() -> Self {
        let mut opt = MlirOpt::upstream();
        opt.load(DetourDialect);
        opt.load(crate::base::ImageDialect);
        let mut passes = Passes::upstream();
        passes.with(FrontendPasses::add);
        let mut interfaces = Interfaces::upstream();
        interfaces.register(&DetourInliner);
        Self { opt, interfaces, passes }
    }
}

struct DetourInliner;

impl mlirpass::OpInterfaces for DetourInliner {
    fn namespace(&self) -> &str {
        "detour"
    }

    fn inliner(&self) -> Option<&dyn mlirpass::DialectInliner> {
        Some(&mlirpass::InlineAll { calls: false, regions: true })
    }
}

pub const FRONTEND: &str = "detour-from-spirv, detour-from-gpu, detour-expand-clear, inline, detour-normalize-cube, func.func(canonicalize, cse)";

#[macro_export]
macro_rules! pass_set {
    ($(#[$m:meta])* $name:ident: $($pass:ty),+ $(,)?) => {
        $(#[$m])*
        pub struct $name;

        impl $name {
            pub fn add(p: &mut ::mlirpass::Passes) {
                $(p.register::<$pass>();)+
            }
        }
    };
}

pass_set! {
    FrontendPasses: DetourFromSpirv, DetourFromGpu, DetourExpandClear, DetourNormalizeCube
}

pass_set! {
    GpuPasses: ConvertDetourToGpu
}

impl Compiler<'_> {
    pub fn parse<'a>(&'a self, src: &'a str) -> Result<Body<'a>, Error> {
        Body::parse(Ctx::new(&self.opt, &self.interfaces), src)
    }

    pub fn module(&self) -> Body<'_> {
        Body::new(Ctx::new(&self.opt, &self.interfaces))
    }

    pub fn run(&self, body: &mut Body<'_>, pipeline: &str) -> Result<(), Error> {
        let root = body.root;
        PassManager::parse(&self.passes, pipeline)?.run_on(body, root, &moremath::ThreadSingle)
    }

    pub fn show(&self, body: &Body<'_>) -> String {
        self.opt.show(&mlirformat::Module::from(body)).to_string()
    }
}
