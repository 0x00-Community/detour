//! EU encodings checked against Mesa's assembler: tests/eu.mlir has each instruction as an `xe3.eu` op, with the
//! gentool assembly it is in a comment above it; tests/eu.expected is what `gentool asm -p ptl` assembled that text
//! into (one line of hex per instruction).

use detour::xe3_dialect::{Inst, Xe3Dialect};
use mlirformat::{Item, MlirOpt};

#[test]
fn encodes_like_gentool() {
    let mut opt = MlirOpt::upstream();
    opt.load(Xe3Dialect);
    let module = opt.parse(include_str!("eu.mlir")).unwrap();
    let want: Vec<&str> = include_str!("eu.expected").lines().collect();
    let ops: Vec<_> = module.items.iter().filter_map(|i| if let Item::Op(op) = i { Some(op) } else { None }).collect();
    assert_eq!(ops.len(), want.len());
    let mut bad = Vec::new();
    for (op, want) in ops.iter().zip(want) {
        let inst = Inst::try_from((&*op.name, &op.attributes[..])).unwrap_or_else(|e| panic!("{e}"));
        let got = inst.encode().iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ");
        if got != want {
            bad.push(format!("{}\n  got  {got}\n  want {want}", opt.show(&mlirformat::Module { items: vec![Item::Op((*op).clone())] })));
        }
    }
    assert!(bad.is_empty(), "{} of {} differ from gentool:\n{}", bad.len(), ops.len(), bad.join("\n"));
}
