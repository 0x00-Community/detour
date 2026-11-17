//! Intel Xe3 machine code and genxml: EU instructions encoded as Mesa's gentool assembles them,
//! genxml items packed as Mesa's C pack functions pack them

use detour::{Pack, Xe3Isa};
use detour::{Inst, XE3, Xe3Dialect, Xe3Genxml};
use mlirformat::Operation;

use super::{attrs, twice};

/// `fixtures/eu.mlir` encodes as gentool assembles the comment above each op (`fixtures/eu.expected`)
#[test]
fn eu_encodes_like_gentool() {
    twice(Xe3Dialect, include_str!("xe3/fixtures/eu.mlir"), |ops, reread| {
        let want: Vec<&str> = include_str!("xe3/fixtures/eu.expected").lines().collect();
        assert_eq!(ops.len(), want.len());
        let encode = |o: &Operation<'_>| match Inst::try_from((&*o.name, &attrs(o)[..])) {
            Ok(i) => Xe3Isa::encode(&i).iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "),
            Err(e) => e.to_string(),
        };
        let mut bad = Vec::new();
        for ((op, re), want) in ops.iter().zip(reread).zip(&want) {
            let (got, got_again) = (encode(op), encode(re));
            if got != *want || got_again != got {
                bad.push(format!("{}\n  got  {got}\n  want {want}\n  printed and read again {got_again}", op.name));
            }
        }
        assert!(bad.is_empty(), "{} of {} differ from gentool:\n{}", bad.len(), ops.len(), bad.join("\n"));
    });
}

/// `fixtures/oracle.mlir` (every genxml field set) packs as Mesa's C pack functions packed the same
/// values (`fixtures/oracle.expected`)
#[test]
fn genxml_packs_as_mesa() {
    twice(Xe3Dialect, include_str!("xe3/fixtures/oracle.mlir"), |ops, reread| {
        let expected: Vec<&str> = include_str!("xe3/fixtures/oracle.expected").lines().collect();
        assert_eq!(ops.len(), expected.len());
        let mut bad = Vec::new();
        for ((op, re), want) in ops.iter().zip(reread).zip(&expected) {
            let n = want.split(' ').count() - 1;
            let pack = |op: &Operation<'_>| {
                let item = XE3.get(&op.name).unwrap();
                let mut dw = Pack(XE3).pack(&op.name, &attrs(op)).unwrap_or_else(|e| panic!("{e}"));
                dw.resize(dw.len().max(n), 0);
                std::iter::once(item.name.to_owned()).chain(dw.iter().take(n).map(|d| format!("{d:08x}"))).collect::<Vec<_>>().join(" ")
            };
            let (got, got_again) = (pack(op), pack(re));
            let typed = Xe3Genxml::verify(op).map(|r| r.map_err(|e| e.to_string()));
            if got != *want || got_again != got || !matches!(typed, Some(Ok(()))) {
                bad.push(format!("want {want}\n got {got}\n printed and read again {got_again}\n typed {typed:?}"));
            }
        }
        assert!(bad.is_empty(), "{} of {} differ:\n{}", bad.len(), expected.len(), bad.join("\n"));
    });
}
