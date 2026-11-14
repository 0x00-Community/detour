//! Every genxml item packed from MLIR (tests/oracle.mlir: each field set to the value Mesa's C pack functions were
//! given) is what Mesa packed (tests/oracle.expected), dword for dword; reads as its typed op; and printed and read
//! again packs the same.

use detour::base::Compiler;
use detour::xe3_dialect::{Genxml, XE3, Xe3Dialect};
use mlirformat::{NamedAttr, Operation};

#[test]
fn packs_as_mesa() {
    let mut c = Compiler::default();
    c.opt.load(Xe3Dialect);
    let body = c.parse(include_str!("oracle.mlir")).unwrap();
    let printed = c.show(&body);
    let again = c.parse(&printed).unwrap();
    let expected: Vec<&str> = include_str!("oracle.expected").lines().collect();
    // generic ops, as a body has them
    let ops: Vec<Operation<'_>> = body.ops(body.top_block(body.root)).map(|o| body.operation(o)).collect();
    let reread: Vec<Operation<'_>> = again.ops(again.top_block(again.root)).map(|o| again.operation(o)).collect();
    assert_eq!(ops.len(), expected.len());
    let mut bad = Vec::new();
    for ((op, re), want) in ops.iter().zip(&reread).zip(&expected) {
        // the C side wrote the longest form (optional trailing dwords, an empty variable group) as zeros
        let n = want.split(' ').count() - 1;
        let pack = |op: &Operation<'_>| {
            let item = XE3.get(&op.name).unwrap();
            let mut dw = XE3.pack(&op.name, &all(op)).unwrap_or_else(|e| panic!("{e}"));
            dw.resize(dw.len().max(n), 0);
            std::iter::once(item.name.to_owned()).chain(dw.iter().take(n).map(|d| format!("{d:08x}"))).collect::<Vec<_>>().join(" ")
        };
        let (got, got_again) = (pack(op), pack(re));
        let typed = Genxml::verify(op).map(|r| r.map_err(|e| e.to_string()));
        if got != *want || got_again != got || !matches!(typed, Some(Ok(()))) {
            bad.push(format!("want {want}\n got {got}\n printed and read again {got_again}\n typed {typed:?}"));
        }
    }
    assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), expected.len(), bad.join("\n"));
}

/// The properties and the discardable attributes together.
fn all<'a>(op: &Operation<'a>) -> Vec<NamedAttr<'a>> {
    let mut a = match &op.properties {
        Some(mlirformat::Attribute::Dict(p)) => p.clone(),
        _ => Vec::new(),
    };
    a.extend(op.attributes.iter().cloned());
    a
}

