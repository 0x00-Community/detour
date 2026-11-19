//! detour's binaries: each GPU's machine code encoded as its oracle encodes it, records packed as
//! Mesa packs them

mod xe3;

use mlirformat::{Attribute, Dialect, Item, MlirOpt, Module, NamedAttr, Operation};

/// An op's attributes, inherent and not
fn attrs<'a>(o: &Operation<'a>) -> Vec<NamedAttr<'a>> {
    let mut out = match &o.properties {
        Some(Attribute::Dict(p)) => p.clone(),
        _ => Vec::new(),
    };
    out.extend(o.attributes.iter().cloned());
    out
}

fn ops<'a>(m: Module<'a>) -> Vec<Operation<'a>> {
    m.items.into_iter().filter_map(|i| if let Item::Op(o) = i { Some(o) } else { None }).collect()
}

/// `src`'s ops, and its ops parsed again from how it prints, to `f`
fn twice(dialect: impl Dialect + 'static, src: &str, f: impl FnOnce(&[Operation<'_>], &[Operation<'_>])) {
    let mut opt = MlirOpt::upstream();
    opt.load(dialect);
    let module = opt.parse(src).unwrap();
    let printed = opt.show(&module).to_string();
    f(&ops(module), &ops(opt.parse(&printed).unwrap()));
}
