//! detour as MLIR: the `detour` dialect (a queue submission: resources, pipelines, render passes), the shader stages,
//! and the passes every backend runs first. A vendor backend loads its dialect and registers its lowering on the same
//! [`Compiler`].
//!
//! ```text
//!   detour.* + shaders: func.func / spirv.module / gpu.module
//!      │  detour-from-spirv, detour-from-gpu      every shader a func.func with detour.stage
//!      │  detour-expand-clear                    a cleared render pass ⇒ a draw that clears
//!      │  func.func(canonicalize, cse)
//!      ▼
//!   detour.* + func.func (arith / vector)       ⇒ the backend (convert-detour-to-<vendor>)
//! ```
//!
//! A shader is a `func.func` with `detour.stage`. Vertex: the arguments are the vertex attributes by location (`f32`
//! or `vector<Nxf32>`, N ≤ 4), the one result the clip-space position (`vector<4xf32>`), the rest varyings. Fragment:
//! the varyings, the one result the color of render target 0 (`vector<4xf32>`). Compute: the arguments are up to 8
//! `i32` parameters of the dispatch (the same in every invocation), no results; a workgroup is 16 invocations
//! (`"detour.workgroup_id"() {dim} : () -> i32`, dim 0 x (the default) / 1 y / 2 z of the dispatch's grid;
//! `"detour.lane"() : () -> i32`, 0 ..= 15); memory is read and written at a
//! byte address per invocation: `"detour.load"(%addr) {size = 1 | 2 | 4} : (i32) -> T`,
//! `"detour.store"(%addr, %v) {size} : (i32, T) -> ()`, T an `i32` / `f32` or a vector of up to 4 (size 4).
//! `"detour.unpack4xu8"(%w) : (i32) -> vector<4xi32>` (WGSL `unpack4xU8`, low byte first) or `-> vector<4xf32>` (the bytes
//! as f32).

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

/// ⭐ The compiler's context: the dialects `MlirOpt` reads and prints, the op interfaces, the passes by name.
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

/// The detour dialect's op interfaces: its ops inline (a shader's helper that loads or samples), as upstream's
/// `InlineAll`.
struct DetourInliner;

impl mlirpass::OpInterfaces for DetourInliner {
    fn namespace(&self) -> &str {
        "detour"
    }

    fn inliner(&self) -> Option<&dyn mlirpass::DialectInliner> {
        Some(&mlirpass::InlineAll { calls: false, regions: true })
    }
}

/// ⭐ What every module goes through before a backend sees it (passes of a `builtin.module(...)` pipeline).
/// (`inline`: a shader's calls of private helpers, the helpers gone after.)
pub const FRONTEND: &str = "detour-from-spirv, detour-from-gpu, detour-expand-clear, inline, detour-normalize-cube, func.func(canonicalize, cse)";

/// A set of passes registered together (mlirpass 0.5 dropped its own `pass_set!`): `/// doc` `Name: Pass, ..`, a unit
/// struct whose `Name::add` registers them, for `Passes::with(Name::add)`.
#[macro_export]
macro_rules! pass_set {
    ($(#[$m:meta])* $name:ident: $($pass:ty),+ $(,)?) => {
        $(#[$m])*
        pub struct $name;

        impl $name {
            /// Registers the set's passes.
            pub fn add(p: &mut ::mlirpass::Passes) {
                $(p.register::<$pass>();)+
            }
        }
    };
}

pass_set! {
    /// detour's own passes of [`FRONTEND`] (canonicalize and cse are upstream's).
    FrontendPasses: DetourFromSpirv, DetourFromGpu, DetourExpandClear, DetourNormalizeCube
}

pass_set! {
    /// The compute shaders as upstream `gpu.func` kernels for mlirpass's GPU targets (after [`FRONTEND`], before a
    /// target's `convert-gpu-to-<isa>`).
    GpuPasses: ConvertDetourToGpu
}

impl Compiler<'_> {
    /// MLIR text ⇒ IR.
    pub fn parse<'a>(&'a self, src: &'a str) -> Result<Body<'a>, Error> {
        Body::parse(Ctx::new(&self.opt, &self.interfaces), src)
    }

    /// An empty module, to build IR in (no text).
    pub fn module(&self) -> Body<'_> {
        Body::new(Ctx::new(&self.opt, &self.interfaces))
    }

    /// Runs a textual pipeline (`builtin.module(...)`) on the whole module.
    pub fn run(&self, body: &mut Body<'_>, pipeline: &str) -> Result<(), Error> {
        let root = body.root;
        PassManager::parse(&self.passes, pipeline)?.run_on(body, root, &moremath::ThreadSingle)
    }

    /// IR ⇒ MLIR text.
    pub fn show(&self, body: &Body<'_>) -> String {
        self.opt.show(&mlirformat::Module::from(body)).to_string()
    }
}
