use std::collections::BTreeMap;

use mlirformat::spirv_types::SpirvType;
use mlirformat::{Attribute, Error, FunctionType, Type};
use mlirpass::{Body, ConversionConfig, ConversionPatterns, ConversionRewriter, ConversionTarget, Converter, Mode, OpData, OpId, Quiet, Rewriter, ValueId, pass, patterns};

use crate::base::{Attrs, Stage};

type Cr<'r, 'a, 'p> = ConversionRewriter<'r, 'a, 'p>;

#[derive(Clone)]
struct Var<'a> {
    ty: Type<'a>,
    input: bool,
    location: Option<i64>,
    position: bool,
}

impl<'a> Var<'a> {
    fn of(body: &Body<'a>, g: OpId) -> Result<(String, Var<'a>), Error> {
        let name = body[g].get_str("sym_name").ok_or_else(|| Error("spirv.GlobalVariable without a name".into()))?;
        let Some(Attribute::Type(t)) = body[g].get("type") else { return Err(Error(format!("@{name}: no type"))) };
        let Ok(SpirvType::Ptr { pointee, storage }) = SpirvType::try_from(t) else { return Err(Error(format!("@{name}: not a pointer"))) };
        let a = body[g].all_attrs();
        let a = Attrs(&a);
        let position = matches!(a.get("built_in"), Some(Attribute::Str { lit, .. }) if lit == "Position");
        let var = Var { ty: (*pointee).clone(), input: storage == "Input", location: a.int("location")?, position };
        Ok((name.to_owned(), var))
    }
}

pub struct SpirvToArith;

#[patterns]
#[map("spirv.FAdd" => "arith.addf", via = Self::plain)]
#[map("spirv.FSub" => "arith.subf", via = Self::plain)]
#[map("spirv.FMul" => "arith.mulf", via = Self::plain)]
#[map("spirv.FDiv" => "arith.divf", via = Self::plain)]
#[map("spirv.FNegate" => "arith.negf", via = Self::plain)]
#[map("spirv.CompositeConstruct" => "vector.from_elements", via = Self::plain)]
impl SpirvToArith {
    fn plain(cr: &mut Cr<'_, '_, '_>, op: OpId, operands: &[Vec<ValueId>], to: &str) -> bool {
        let Some(ops) = Cr::one_to_one(operands) else { return false };
        let ty = cr.rw.body[cr.rw.body[op].results[0]].ty.clone();
        cr.replace_new(op, OpData::build(to).operands(&ops), &[ty]);
        true
    }

    fn indices<'a>(cr: &Cr<'_, 'a, '_>, op: OpId, name: &str) -> Option<Attribute<'a>> {
        let Attribute::Array(v) = cr.rw.body[op].get(name)? else { return None };
        let v: Vec<i64> = v.iter().map(Attrs::as_int).collect::<Result<_, _>>().ok()?;
        Some(Attribute::dense_i64(&v))
    }

    #[convert("spirv.Constant")]
    fn constant(cr: &mut Cr<'_, '_, '_>, op: OpId, _: &[Vec<ValueId>]) -> bool {
        let ty = cr.rw.body[cr.rw.body[op].results[0]].ty.clone();
        let Some(v) = cr.rw.body[op].get("value").cloned() else { return false };
        cr.replace_new(op, OpData::build("arith.constant").with("value", v), &[ty]);
        true
    }

    #[convert("spirv.CompositeExtract")]
    fn extract(cr: &mut Cr<'_, '_, '_>, op: OpId, operands: &[Vec<ValueId>]) -> bool {
        let (Some(ops), Some(at)) = (Cr::one_to_one(operands), Self::indices(cr, op, "indices")) else { return false };
        let ty = cr.rw.body[cr.rw.body[op].results[0]].ty.clone();
        cr.replace_new(op, OpData::build("vector.extract").operands(&ops).with("static_position", at), &[ty]);
        true
    }

    #[convert("spirv.VectorShuffle")]
    fn shuffle(cr: &mut Cr<'_, '_, '_>, op: OpId, operands: &[Vec<ValueId>]) -> bool {
        let (Some(ops), Some(mask)) = (Cr::one_to_one(operands), Self::indices(cr, op, "components")) else { return false };
        let ty = cr.rw.body[cr.rw.body[op].results[0]].ty.clone();
        cr.replace_new(op, OpData::build("vector.shuffle").operands(&ops).with("mask", mask), &[ty]);
        true
    }
}

struct Entry<'r, 'b, 'a> {
    rw: &'b mut Rewriter<'r, 'a>,
    vars: &'b BTreeMap<String, Var<'a>>,
}

impl<'a> Entry<'_, '_, 'a> {
    fn func(&mut self, f: OpId, stage: Stage, interface: &[String], before: OpId) -> Result<OpId, Error> {
        let var = |n: &String| self.vars.get(n).cloned().map(|v| (n.clone(), v)).ok_or_else(|| Error(format!("no variable @{n}")));
        let vars = interface.iter().map(var).collect::<Result<Vec<_>, _>>()?;
        let (mut inputs, mut outputs): (Vec<_>, Vec<_>) = vars.into_iter().partition(|(_, v)| v.input);
        inputs.sort_by_key(|(_, v)| v.location);
        outputs.sort_by_key(|(_, v)| (!v.position, v.location));
        let fty = FunctionType { inputs: inputs.iter().map(|(_, v)| v.ty.clone()).collect(), results: outputs.iter().map(|(_, v)| v.ty.clone()).collect() };
        let name = self.rw.body[f].get("sym_name").cloned().ok_or_else(|| Error("spirv.func without a name".into()))?;
        let data = OpData::build("func.func").with("sym_name", name).with("function_type", Attribute::Type(Type::Function(fty.clone().into()))).with(Stage::ATTR, stage.into());
        self.rw.set_before(before);
        let new = self.rw.create_with(data, &[], |b, new| {
            let region = b.add_region(new);
            b.add_block(region, &fty.inputs.iter().map(|t| (t.clone(), None)).collect::<Vec<_>>());
        });
        let block = self.rw.body.top_block(new);
        let from = self.rw.body.top_block(f);
        let ops = self.rw.body.block_ops(from);
        self.rw.inline_block_before(from, block, None, &[]);

        let mut ptrs: BTreeMap<ValueId, String> = BTreeMap::new();
        let mut stored: BTreeMap<String, ValueId> = BTreeMap::new();
        for op in ops {
            match &*self.rw.body[op].name {
                "spirv.mlir.addressof" => {
                    let a = self.rw.body[op].all_attrs();
                    let v = Attrs(&a).symbol("variable")?.to_owned();
                    ptrs.insert(self.rw.body[op].results[0], v);
                    self.rw.erase_op(op);
                }
                "spirv.Load" => {
                    let v = ptrs.get(&self.rw.body[op].operands[0]).ok_or_else(|| Error("spirv.Load of an unknown pointer".into()))?;
                    let i = inputs.iter().position(|(n, _)| n == v).ok_or_else(|| Error(format!("spirv.Load of @{v}: not an input")))?;
                    let (arg, r) = (self.rw.body[block].args[i], self.rw.body[op].results[0]);
                    self.rw.replace_all_uses(r, arg);
                    self.rw.erase_op(op);
                }
                "spirv.Store" => {
                    let v = ptrs.get(&self.rw.body[op].operands[0]).ok_or_else(|| Error("spirv.Store to an unknown pointer".into()))?;
                    stored.insert(v.clone(), self.rw.body[op].operands[1]);
                    self.rw.erase_op(op);
                }
                "spirv.Return" => {
                    let values = outputs.iter().map(|(n, _)| stored.get(n).copied().ok_or_else(|| Error(format!("output @{n} never stored")))).collect::<Result<Vec<_>, _>>()?;
                    self.rw.set_before(op);
                    self.rw.create(OpData::build("func.return").operands(&values), &[]);
                    self.rw.erase_op(op);
                }
                _ => {}
            }
        }
        Ok(new)
    }
}

#[pass("detour-from-spirv", on = "builtin.module")]
pub fn detour_from_spirv<'a>(body: &mut Body<'a>, op: OpId) -> Result<(), Error> {
    let top = body.top_block(op);
    let mut funcs = Vec::new();
    let mut quiet = Quiet;
    let mut rw = Rewriter::new(body, &mut quiet);
    let modules: Vec<OpId> = rw.body.ops(top).filter(|m| rw.body[*m].name == "spirv.module").collect();
    for m in modules {
        let inner = rw.body.top_block(m);
        let ops = rw.body.block_ops(inner);
        let vars: BTreeMap<String, Var<'a>> = ops.iter().filter(|g| rw.body[**g].name == "spirv.GlobalVariable").map(|g| Var::of(rw.body, *g)).collect::<Result<_, _>>()?;
        let entries: Vec<OpId> = ops.iter().copied().filter(|e| rw.body[*e].name == "spirv.EntryPoint").collect();
        for e in entries {
            let a = rw.body[e].all_attrs();
            let a = Attrs(&a);
            let model = a.get("execution_model").map(ToString::to_string).unwrap_or_default();
            let stage = [("Vertex", Stage::Vertex), ("Fragment", Stage::Fragment), ("GLCompute", Stage::Compute)]
                .into_iter()
                .find(|(n, _)| model.contains(&format!("<{n}>")))
                .map(|(_, s)| s)
                .ok_or_else(|| Error(format!("entry point: execution model {model}")))?;
            let name = a.symbol("fn")?;
            let f = ops.iter().copied().find(|f| rw.body[*f].name == "spirv.func" && rw.body[*f].get_str("sym_name") == Some(name));
            let f = f.ok_or_else(|| Error(format!("entry point: no spirv.func @{name}")))?;
            let interface: Vec<String> = a.array("interface")?.iter().map(|i| Attrs::as_symbol(i).map(str::to_owned)).collect::<Result<_, _>>()?;
            funcs.push(Entry { rw: &mut rw, vars: &vars }.func(f, stage, &interface, m)?);
        }
        rw.erase_op(m);
    }
    let mut patterns = ConversionPatterns::default();
    SpirvToArith::populate(&mut patterns);
    let mut target = ConversionTarget::default();
    target.legal_dialect("arith").legal_dialect("vector").legal_dialect("func").illegal_dialect("spirv");
    Converter::new(&target, &patterns, ConversionConfig::default()).apply(body, &funcs, Mode::Full)
}
