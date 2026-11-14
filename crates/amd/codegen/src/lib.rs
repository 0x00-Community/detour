extern crate alloc;

mod precheck;

pub use precheck::*;

use mlirformat::{Error, Op};
use mlirpass::{A, Body, Emit, Heads, Image, ImageData, ImageDialect, ImageMemory, ImageStream, MemType, Quiet, Rdna2Dialect, Rewriter, Symbol};

use crate::base::{Compiler, FRONTEND, GpuPasses};

#[allow(unused_imports, reason = "for mlirpass' macro expansions")]
pub(crate) use mlirpass::{OptionValue, Options, PassDecl, Run};

pub use mlirpass::{RDNA2_IMAGE, Rdna2Code, Rdna2Dispatch as Dispatch, Rdna2Emit, Rdna2Image as Rdna2, Rdna2Kernel};

pub const BACKEND: &str = "rdna2-precheck, convert-detour-to-gpu, convert-gpu-to-rdna2, rdna2-regalloc, rdna2-waitcnt";

pub const END: &str = "rdna2-pm4, rdna2-pad, rdna2-layout";

crate::pass_set! {
    Rdna2Passes: Rdna2Precheck
}

pub struct Codegen<'i>(pub Compiler<'i>);

impl Default for Codegen<'_> {
    fn default() -> Self {
        let mut c = Compiler::default();
        c.opt.load(Rdna2Dialect);
        c.opt.load(ImageDialect);
        c.passes.with(Rdna2Passes::add).with(GpuPasses::add);
        Codegen(c)
    }
}

impl Codegen<'_> {
    pub fn lower<'a>(&'a self, src: &'a str, pipeline: &str) -> Result<Body<'a>, Error> {
        let mut body = self.0.parse(src)?;
        self.0.run(&mut body, pipeline)?;
        Ok(body)
    }

    pub fn pipeline() -> String {
        format!("builtin.module({FRONTEND}, {BACKEND})")
    }

    pub fn image(&self, src: &str, groups: u32) -> Result<Built, Error> {
        let mut body = self.lower(src, &Codegen::pipeline())?;
        let mut kernels = Vec::new();
        body.walk(body.root, &mut |o| {
            if body[o].name == Rdna2Kernel::NAME {
                kernels.push(o);
            }
        });
        let &k = kernels.first().ok_or_else(|| Error("no kernel".into()))?;
        let head = body.head::<Rdna2Kernel>(k)?;
        let vgprs = head.vgprs.ok_or_else(|| Error("a kernel without vgprs".into()))?;
        let d = Dispatch { kernel: Symbol::defined(&head.sym_name)?.to_owned(), vgprs, groups: [groups, 1, 1], params: vec![A::sym("src"), A::sym("dst")], shader_engines: 1 };

        let top = body.top_block(body.root);
        let (code, block) = body.container(&ImageMemory::named("code", MemType::Streaming))?;
        let (ib, ib_block) = body.container(&ImageStream::named("ib"))?;
        let mut quiet = Quiet;
        let mut rw = Rewriter::new(&mut body, &mut quiet);
        rw.set_end(top);
        rw.insert(code);
        rw.set_end(block);
        rw.insert(ib);
        rw.set_end(ib_block);
        for o in d.ops()? {
            rw.create(o, &[]);
        }
        let shaders = rw.body.parent_op(k);
        rw.move_before(k, block, None);
        if let Some(m) = shaders.filter(|m| rw.body[*m].name == "gpu.module" && rw.body.ops(rw.body.top_block(*m)).next().is_none()) {
            rw.erase_op(m);
            let root = rw.body.root;
            rw.modify(root, |b| {
                b[root].remove("gpu.container_module");
            });
        }
        for buffer in ["src", "dst"] {
            let (m, b) = rw.body.container(&ImageMemory::named(buffer, MemType::Coherent))?;
            rw.set_end(top);
            rw.insert(m);
            rw.set_end(b);
            rw.emit(&ImageData::zeros(buffer, 0x1_0000));
        }
        self.0.run(&mut body, &format!("builtin.module({END})"))?;
        let image = RDNA2_IMAGE.emit(&body, body.root)?;
        let kernel = Rdna2Code::of(&body, k)?.bytes;
        let at = match body[k].get("offset") {
            Some(a) => mlirpass::Attrs::as_int(a)? as u64,
            None => return Err(Error("the kernel was not placed".into())),
        };
        let code = image.segment("code").ok_or_else(|| Error("no @code".into()))?;
        if code.bytes.get(at as usize..at as usize + kernel.len()) != Some(&kernel[..]) {
            return Err(Error("@code does not hold the kernel where it was placed".into()));
        }
        let kernel_addr = code.addr + at;
        Ok(Built { module: self.0.show(&body).to_string(), image, kernel, kernel_addr })
    }
}

pub struct Built {
    pub module: String,
    pub image: Image,
    pub kernel: Vec<u8>,
    pub kernel_addr: u64,
}

impl Built {
    pub fn dump(&self) -> String {
        let mut s = format!("stream {:#x} {:#x}\n", self.image.stream, self.image.stream_end);
        for name in ["code", "src", "dst"] {
            if let Some(g) = self.image.segment(name) {
                s += &format!("segment {name} {:#x} {}\n", g.addr, g.bytes.iter().map(|b| format!("{b:02x}")).collect::<String>());
            }
        }
        s
    }
}
