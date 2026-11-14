use mlirformat::{Attribute, Const, Error, FloatKind, FunctionType, Op, Scalar, Type};
use mlirpass::{ApFloat, BlockId, Body, OpData, OpId, Quiet, Rewriter, ValueId, pass};

use crate::base::{A, BlendState, Buffer, DepthState, Heads, Draw, LoadOp, Pipeline, RenderPass, Stage, StepMode, Symbol, Texture, TextureFormat, Topology, VertexAttribute, VertexBufferLayout, VertexFormat};

struct Clear<'r, 'a> {
    rw: Rewriter<'r, 'a>,
    module: OpId,
}

impl<'a> Clear<'_, 'a> {
    fn f32() -> Type<'a> {
        Type::Float(FloatKind::F32)
    }

    fn vec4() -> Type<'a> {
        Type::vector(&[4], &[false], Self::f32())
    }

    fn dense(ty: Type<'a>, values: &[f64]) -> Attribute<'a> {
        let values = values.iter().map(|v| Scalar::Float(ApFloat::from_f64(FloatKind::F32, *v).expect("f32"))).collect();
        Const { ty, values, elements: true }.into()
    }

    fn func(&mut self, name: &str, stage: Stage, args: &[Type<'a>], fill: impl FnOnce(&mut Rewriter<'_, 'a>, BlockId) -> ValueId) {
        let fty = Type::Function(FunctionType { inputs: args.to_vec(), results: vec![Self::vec4()] }.into());
        let top = self.rw.body.top_block(self.module);
        self.rw.set_start(top);
        let data = OpData::build("func.func").with("sym_name", A::str(name).into()).with("function_type", Attribute::Type(fty)).with(Stage::ATTR, stage.into());
        let f = self.rw.create_with(data, &[], |b, f| {
            let region = b.add_region(f);
            b.add_block(region, &args.iter().map(|t| (t.clone(), None)).collect::<Vec<_>>());
        });
        let block = self.rw.body.top_block(f);
        self.rw.set_end(block);
        let v = fill(&mut self.rw, block);
        self.rw.set_end(block);
        self.rw.create(OpData::build("func.return").operands(&[v]), &[]);
    }

    fn expand(&mut self, n: usize, format: TextureFormat, samples: Option<u32>, color: &[f64]) -> Draw<'a> {
        let (pipe, vb) = (format!("__clear{n}"), format!("__clear{n}_vb"));
        ClearPipeline { name: &pipe, format, samples, depth: None, color: Some([color[0], color[1], color[2], color[3]]) }.build(self);
        let top = self.rw.body.top_block(self.module);
        self.rw.set_start(top);
        self.rw.emit(&Buffer::new(A::str(vb.clone()).into(), Some(Self::dense(Type::tensor(&[12], Self::f32()), &ClearPipeline::CORNERS.map(f64::from))), None, false));
        Draw::new(A::sym(pipe).into(), vec![A::sym(vb).into()], None, Some(3), None, None, None, None, None, None, None)
    }
}

pub struct ClearPipeline<'n> {
    pub name: &'n str,
    pub format: TextureFormat,
    pub samples: Option<u32>,
    pub depth: Option<DepthState>,
    pub color: Option<[f64; 4]>,
}

impl ClearPipeline<'_> {
    pub const CORNERS: [f32; 12] = [-1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0];

    pub fn emit<'a>(&self, body: &mut Body<'a>, module: OpId) {
        let mut quiet = Quiet;
        self.build(&mut Clear { rw: Rewriter::new(body, &mut quiet), module });
    }

    fn build(&self, clear: &mut Clear<'_, '_>) {
        let (vs, fs) = (format!("{}_vs", self.name), format!("{}_fs", self.name));
        clear.func(&vs, Stage::Vertex, &[Clear::vec4()], |rw, block| rw.body[block].args[0]);
        let c = self.color.unwrap_or([1.0; 4]);
        clear.func(&fs, Stage::Fragment, &[], move |rw, _| {
            let k = rw.create(OpData::build("arith.constant").with("value", Clear::dense(Clear::vec4(), &c)), &[Clear::vec4()]);
            rw.body[k].results[0]
        });
        let top = clear.rw.body.top_block(clear.module);
        clear.rw.set_start(top);
        let corners = VertexBufferLayout { stride: 16, step_mode: StepMode::Vertex, attributes: vec![VertexAttribute { location: 0, format: VertexFormat::Float32x4, offset: 0 }] };
        let (name, layout, depth) = (A::str(self.name).into(), Attribute::from(&corners), self.depth.as_ref().map(Attribute::from));
        let blends = self.color.is_none().then(|| vec![Attribute::from(&BlendState::CONSTANT)]);
        clear.rw.emit(&Pipeline::new(name, A::sym(vs).into(), A::sym(fs).into(), Some(Topology::TriangleList), vec![self.format], vec![layout], None, None, None, depth, self.samples, None, None, None, None, None, blends, None, None, None));
    }
}

#[pass("detour-expand-clear", on = "builtin.module")]
pub fn detour_expand_clear<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let mut passes = Vec::new();
    body.walk(op, &mut |o| passes.push(o));
    passes.retain(|p| body[*p].name == "detour.render_pass");
    let top = body.top_block(op);
    let mut quiet = Quiet;
    let mut clear = Clear { rw: Rewriter::new(body, &mut quiet), module: op };
    for (n, pass) in passes.into_iter().enumerate() {
        let rp = clear.rw.body.head::<RenderPass>(pass)?;
        if rp.load != Some(LoadOp::Clear) {
            continue;
        }
        let color = rp.clear.unwrap_or_default();
        if color.len() != 4 {
            return Err(Error(format!("render pass: clear is {} values, not 4", color.len())));
        }
        let target = Symbol::name(&rp.color)?;
        let body = &*clear.rw.body;
        let named = |t: OpId| body[t].name == Texture::NAME && body[t].get(Symbol::ATTR).is_some_and(|s| Symbol::defined(s).is_ok_and(|s| s == target));
        let texture = body.ops(top).find(|t| named(*t));
        let texture = body.head::<Texture>(texture.ok_or_else(|| Error(format!("render pass: no texture @{target}")))?)?;
        let draw = clear.expand(n, texture.format, texture.samples, &color);
        let inside = clear.rw.body.top_block(pass);
        clear.rw.set_start(inside);
        clear.rw.emit(&draw);
        clear.rw.modify(pass, |b| {
            b[pass].remove("load");
        });
    }
    Ok(())
}
