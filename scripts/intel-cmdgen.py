#!/usr/bin/env python3
"""genxml => the xe3 dialect's command / state / register ops: reads a flattened Mesa genxml file (crates/intel/dialect/genxml/xe3.xml)
and writes `crates/intel/dialect/src/cmd.gen.rs`:

- `ITEMS`: every item's layout (fields: first bit, width, type, default, fixed array dims; the variable group),
  sorted by op name: what `Item::pack` packs an op's attributes by;
- a typed op per item: instructions `xe3.cmd.*`, structs `xe3.state.*`, registers `xe3.reg.*`; each settable field
  an `crate::base::Fill` (a number, a bool or a place), an `f64`, or a struct's typed op (`crate::base::Fields`); the
  variable group `elements`, each an `<Item>Element`; `Genxml::verify`; `INHERENT`, what they declare;
- the enum and field values as constants, like Mesa's `#define`s.

Video-engine items are left out (and the video structs nothing kept refers to).

    python3 scripts/intel-cmdgen.py crates/intel/dialect/genxml/xe3.xml crates/intel/dialect/src/cmd.gen.rs
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Items of the video engines (codecs, scalers): detour has no use for them.
VIDEO = {
    "AVP", "HCP", "HEVC", "HUC", "MFC", "MFD", "MFX", "SFC", "VD", "VD0", "VDENC", "VVCP", "AES", "CHROMA", "LUMA",
    "FRAMEDELTAQP", "FRAMEDELTAQPRANGE", "ROUNDINGPRECISIONTABLE", "VEBOX", "VDAQM", "VDL", "MFX_", "JPEG", "VP9",
}
U64 = (1 << 64) - 1
U32 = (1 << 32) - 1


def rs(s):
    """A string as Rust's `{:?}` spells it."""
    esc = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t", "\0": "\\0"}
    return '"' + "".join(esc.get(c, c if c.isprintable() else f"\\u{{{ord(c):x}}}") for c in s) + '"'


# genxml's spellings => ours

def to_int(s):
    s = s.strip()
    if s[:2] in ("0x", "0X"):
        return int(s[2:], 16)
    return int(s) & U64


def words(s):
    return [w for w in re.split(r"[^A-Za-z0-9]+", s) if w]


def snake(s):
    """An op name: snake case, a leading digit kept (`3dstate_vs`)."""
    return "_".join(words(s)).lower()


def attr(s):
    """An attribute name: a leading digit gets a `_` (`_1x_...`)."""
    i = snake(s)
    return "field" if not i else f"_{i}" if i[0].isdigit() else i


def camel(space, s):
    """The Rust type of an item's op: `Cmd3dstateVs`."""
    return space[:1].upper() + space[1:] + "".join(w[:1].upper() + w[1:].lower() for w in words(s))


def constant(s):
    i = "_".join(words(s)).upper()
    return f"_{i}" if i[0].isdigit() else i


def video(name):
    return re.split(r"[_ ]", name)[0] in VIDEO


@dataclass
class Ty:
    """A field's type: bool uint int float address fixed struct mbo mbz."""
    kind: str
    signed: bool = False
    frac: int = 0
    struct: str = ""

    def rust_table(self):
        return {
            "fixed": f"Ty::{'Sfixed' if self.signed else 'Ufixed'}({self.frac})",
            "struct": f"Ty::Struct({rs(self.struct)})",
        }.get(self.kind, f"Ty::{self.kind.capitalize()}")

    def rust(self, structs):
        """The Rust type of the field in a typed op (`None`: not settable)."""
        if self.kind in ("bool", "uint", "int", "address"):
            return "crate::base::Fill"
        if self.kind in ("float", "fixed"):
            return "crate::base::Real"
        if self.kind == "struct":
            return f"crate::base::Fields<{structs[self.struct]}Fields>"
        return None


@dataclass
class Field:
    name: str
    bit: int  # first bit, from the item's (or the variable group element's) dword 0, for element 0 of every fixed group
    width: int
    ty: Ty
    default: int
    dims: list  # (count, stride in bits) of the enclosing fixed groups, outermost first


@dataclass
class Group:
    """A variable-length group: elements of `size` bits from bit `at`, the caller gives how many."""
    at: int
    size: int
    fields: list


@dataclass
class Item:
    space: str
    orig: str
    length: object
    bias: int
    num: int
    fields: list
    group: object = None
    op: str = field(init=False)

    def __post_init__(self):
        self.op = f"xe3.{self.space}.{snake(self.orig)}"

    def all_fields(self):
        return self.fields + (self.group.fields if self.group else [])

    def fixed_length(self):
        """Dwords of the fixed part: up to the variable group, else the declared length, else the last field."""
        if self.group:
            return self.group.at // 32
        if self.length is not None:
            return self.length
        end = max((f.bit + f.width + sum((c - 1) * s for c, s in f.dims) for f in self.fields), default=0)
        return -(-end // 32)

    def refs(self):
        """The struct op names this item's fields are."""
        return [f.ty.struct for f in self.all_fields() if f.ty.kind == "struct"]


def drop_unreferenced_video(items):
    keep = {i.op for i in items if i.space != "state" or not video(i.orig)}
    while more := {s for i in items if i.op in keep for s in i.refs() if s not in keep}:
        keep |= more
    return [i for i in items if i.op in keep]


class Reader:
    def __init__(self, root):
        named = lambda tag: {c.get("name") for c in root if c.tag == tag}
        self.structs, self.enums, self.values = named("struct"), named("enum"), []

    def ty(self, t):
        if t in ("bool", "uint", "int", "float", "mbo", "mbz"):
            return Ty(t)
        if t in ("address", "offset"):
            return Ty("address")
        if t in self.structs:
            return Ty("struct", struct=f"xe3.state.{snake(t)}")
        if t in self.enums:
            return Ty("uint")
        if t[:1] in ("u", "s") and "." in t[1:]:
            return Ty("fixed", signed=t[0] == "s", frac=int(t[1:].split(".", 1)[1]))
        return Ty("uint")

    def read_values(self, node, prefix):
        for v in (c for c in node if c.tag == "value"):
            name = v.get("name")
            self.values.append((constant(f"{prefix}_{name}" if prefix is not None else name), to_int(v.get("value"))))

    def fields(self, node, base, dims, out, group):
        """The fields of `node` from bit `base`, inside the fixed groups `dims`; a variable group goes to `group[0]`."""
        for c in node:
            if c.tag == "field":
                b = c.get("bits")
                hi, _, lo = b.partition(":")
                hi, lo = int(hi.strip()), int((lo if _ else hi).strip())
                self.read_values(c, c.get("prefix"))
                ty = self.ty(c.get("type"))
                width = hi - lo + 1
                default = (U64 if width >= 64 else (1 << width) - 1) if ty.kind == "mbo" else to_int(c.get("default")) if c.get("default") is not None else 0
                out.append(Field(attr(c.get("name")), base + int(c.get("dword")) * 32 + lo, width, ty, default, list(dims)))
            elif c.tag == "group":
                count, size = int(c.get("count")), int(c.get("size"))
                start = base + int(c.get("dword", "0")) * 32
                if count == 0:
                    assert not dims and group[0] is None, "nested or second variable group"
                    fs = []
                    self.fields(c, 0, [], fs, [None])
                    group[0] = Group(start, size, fs)
                else:
                    self.fields(c, start, dims + [(count, size)], out, group)

    def item(self, c):
        """An instruction, struct or register element as an item (video-engine instructions and registers left out)."""
        space = {"instruction": "cmd", "struct": "state", "register": "reg"}.get(c.tag)
        if space is None:
            return None
        name = c.get("name")
        if space != "state" and (video(name) or c.get("engine") == "video"):
            return None
        fields, group = [], [None]
        before = len(self.values)
        self.fields(c, 0, [], fields, group)
        if space == "reg":
            del self.values[before:]
        for fs in [fields] + ([group[0].fields] if group[0] else []):
            dedup(fs)
        return Item(space, name, int(c.get("length")) if c.get("length") is not None else None, int(c.get("bias", "0")),
                    to_int(c.get("num")) & U32 if c.get("num") is not None else 0, fields, group[0])


def dedup(fields):
    """The same name twice in one item (alternative layouts of the same bits): the later ones numbered."""
    seen = {}
    for f in fields:
        seen[f.name] = n = seen.get(f.name, 0) + 1
        if n > 1:
            f.name = f"{f.name}_{n}"


class Writer:
    def __init__(self):
        self.out = []

    def w(self, s):
        self.out.append(s)

    def fields(self, fields):
        self.w("&[")
        for f in fields:
            if f.ty.kind == "mbz":
                continue
            self.w(f"\n            Field::new({rs(f.name)}, {f.bit}, {f.width}, {f.ty.rust_table()})")
            if f.default:
                self.w(f".with_default({f.default:#x})")
            if f.dims:
                self.w(f".with_dims(&[{', '.join(f'({c}, {s})' for c, s in f.dims)}])")
            self.w(",")
        self.w("\n        ]")

    def item(self, i):
        self.w(f"    Item {{\n        name: {rs(i.orig)},\n        op: {rs(i.op)},\n        length: {i.fixed_length()},\n"
               f"        bias: {i.bias},\n        num: {i.num:#x},\n        fields: ")
        self.fields(i.fields)
        if i.group:
            self.w(f",\n        group: Some(Group {{ at: {i.group.at}, size: {i.group.size}, fields: ")
            self.fields(i.group.fields)
            self.w(" }),\n    },\n")
        else:
            self.w(",\n        group: None,\n    },\n")

    def values(self, values):
        """The enum and field values, like Mesa's `#define`s; a name given two different values is left out."""
        by_name = {}
        for n, v in values:
            by_name.setdefault(n, set()).add(v)
        for n in sorted(by_name, key=lambda n: n.encode()):
            if len(by_name[n]) == 1:
                self.w(f"pub const {n}: u32 = {next(iter(by_name[n])):#x};\n")

    def struct_fields(self, settable):
        for n, t in settable:
            self.w(f'    #[serde(default, skip_serializing_if = "Option::is_none")]\n    pub {n}: Option<{t}>,\n')
        self.w("}\n\n")

    def typed(self, items):
        """The typed half: an op an item (its settable fields; a struct field the struct's typed op; the variable group
        `elements`, each an `<Item>Element`), `Genxml::verify`, the declared attributes for `op_infos`."""
        structs = {i.op: camel(i.space, i.orig) for i in items}

        def settable(fs):
            # a field in fixed groups: an array a dimension (nested, outermost first)
            out = []
            for f in fs:
                t = f.ty.rust(structs)
                if t is not None:
                    for _ in f.dims:
                        t = f"Vec<{t}>"
                    out.append((f.name, t))
            return out

        derive = "#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]"
        # an item another nests (a field of its type): its fields as a plain struct too (an op's own deserializer reads
        # an integer as an i64, which a 64-bit address overflows)
        nested = {s for i in items for s in i.refs()}
        for i in items:
            if i.op in nested:
                self.w(f"/// `{i.op}`'s fields, nested in another item.\n{derive}\npub struct {structs[i.op]}Fields {{\n")
                self.struct_fields(settable(i.fields))
        for i in items:
            ty = structs[i.op]
            if i.group:
                self.w(f"/// An element of `{i.op}`'s variable group.\n{derive}\npub struct {ty}Element {{\n")
                self.struct_fields(settable(i.group.fields))
            self.w(f'/// `{i.op}`: genxml\'s {i.orig}.\n#[mlirformat::mlir::op("{i.op}", format = "attr-dict")]\n#[derive(Default)]\npub struct {ty} {{\n')
            for n, t in settable(i.fields):
                self.w(f"    pub {n}: Option<{t}>,\n")
            if i.group:
                self.w(f"    pub elements: Option<Vec<crate::base::Fields<{ty}Element>>>,\n")
            self.w("}\n\n")
        self.w("/// The items' typed ops.\npub struct Genxml;\n\nimpl Genxml {\n    /// ⭐ Reads `op` (generic, as a module built from a body has it) as its item's typed op (`None`: not an item).\n    pub fn verify(op: &mlirformat::Operation<'_>) -> Option<Result<(), mlirformat::Error>> {\n        Some(match &*op.name {\n")
        for i in items:
            self.w(f"            {rs(i.op)} => op.to_op::<{structs[i.op]}>().map(drop),\n")
        self.w('            _ => return None,\n        })\n    }\n}\n\n#[mlirformat::mlir::dialect("xe3", ops = [')
        self.w(", ".join(structs[i.op] for i in items))
        self.w("])]\npub(crate) struct GenxmlOps;\n\n/// What the typed ops declare (they live in `<{...}>`), for the dialect's `op_infos`.\npub const INHERENT: &[(&str, &[&str])] = &[\n")
        for i in items:
            names = [rs(n) for n, _ in settable(i.fields)] + (['"elements"'] if i.group else [])
            self.w(f"    ({rs(i.op)}, &[{', '.join(names)}]),\n")
        self.w("];\n\n")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: intel-cmdgen.py <flat genxml> <out.rs>")
    _, src, dst = sys.argv
    root = ET.parse(src).getroot()
    r = Reader(root)
    for e in root:
        if e.tag == "enum":
            r.read_values(e, e.get("prefix"))
    items = drop_unreferenced_video([i for c in root if (i := r.item(c))])
    items.sort(key=lambda i: i.op.encode())
    items = [i for k, i in enumerate(items) if k == 0 or items[k - 1].op != i.op]

    w = Writer()
    w.w(f"// @generated by scripts/intel-cmdgen.py from {os.path.basename(src)} ({root.get('name', '?')}). Do not edit.\n//\n"
        "// Field layouts are Mesa's genxml (Copyright Intel Corporation, MIT license).\n\nuse crate::base::{Field, Group, Item, Ty};\n\n")
    w.w("/// Every item's layout, sorted by op name.\npub const ITEMS: &[Item] = &[\n")
    for i in items:
        w.item(i)
    w.w("];\n\n")
    w.typed(items)
    w.values(r.values)
    with open(dst, "w") as f:
        f.write("".join(w.out))
    print(f"{len(items)} items -> {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
