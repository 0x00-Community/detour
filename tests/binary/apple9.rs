//! Apple9 (M3 / M4) machine code: each instruction encoded as the M4 ran it

use detour::Apple9Isa;
use detour::Apple9Dialect;
use mlirformat::{MlirOpt, Operation};

use super::{attrs, ops};

fn encode(o: &Operation<'_>) -> String {
    match Apple9Isa::encode(&o.name, &attrs(o)) {
        Ok(bytes) => bytes.iter().map(|x| format!("{x:02x}")).collect(),
        Err(e) => e.to_string(),
    }
}

/// `fixtures/isa.mlir`, one instruction a line, encodes as `fixtures/isa.expected` (what the M4 ran),
/// and so does the same text printed and read again
#[test]
fn encodes_as_the_m4_ran_it() {
    let mut opt = MlirOpt::upstream();
    opt.load(Apple9Dialect);
    let src = include_str!("apple9/fixtures/isa.mlir");
    let module = opt.parse(src).unwrap();
    let printed = opt.show(&module).to_string();
    let (ops, reread) = (ops(module), ops(opt.parse(&printed).unwrap()));
    let expected: Vec<&str> = include_str!("apple9/fixtures/isa.expected").lines().collect();
    assert_eq!(ops.len(), expected.len());
    let mut bad = Vec::new();
    for ((op, re), want) in ops.iter().zip(&reread).zip(&expected) {
        let (got, got_again) = (encode(op), encode(re));
        if got != *want || got_again != got {
            bad.push(format!("{}\n want {want}\n  got {got}\n printed and read again {got_again}", op.name));
        }
    }
    assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), expected.len(), bad.join("\n"));
}
