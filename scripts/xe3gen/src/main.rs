use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

/// The commands and state structs detour writes: they, the structs they nest and the values of their fields' enums are
/// generated, nothing else
const KEEP: &[&str] = &[
    "3DPRIMITIVE_EXTENDED", "3DSTATE_3D_MODE", "3DSTATE_AA_LINE_PARAMETERS", "3DSTATE_BINDING_TABLE_POINTERS_DS",
    "3DSTATE_BINDING_TABLE_POINTERS_GS", "3DSTATE_BINDING_TABLE_POINTERS_HS", "3DSTATE_BINDING_TABLE_POINTERS_PS",
    "3DSTATE_BINDING_TABLE_POINTERS_VS", "3DSTATE_BINDING_TABLE_POOL_ALLOC", "3DSTATE_BLEND_STATE_POINTERS",
    "3DSTATE_CC_STATE_POINTERS", "3DSTATE_CLIP", "3DSTATE_CONSTANT_ALL", "3DSTATE_DEPTH_BOUNDS", "3DSTATE_DEPTH_BUFFER",
    "3DSTATE_DRAWING_RECTANGLE_FAST", "3DSTATE_DS", "3DSTATE_GS", "3DSTATE_HIER_DEPTH_BUFFER", "3DSTATE_HS",
    "3DSTATE_INDEX_BUFFER", "3DSTATE_MESH_CONTROL", "3DSTATE_MULTISAMPLE", "3DSTATE_POLY_STIPPLE_OFFSET",
    "3DSTATE_PRIMITIVE_REPLICATION", "3DSTATE_PS", "3DSTATE_PS_BLEND", "3DSTATE_PS_EXTRA",
    "3DSTATE_PUSH_CONSTANT_ALLOC_DS", "3DSTATE_RASTER", "3DSTATE_SAMPLE_MASK", "3DSTATE_SAMPLE_PATTERN",
    "3DSTATE_SAMPLER_STATE_POINTERS_PS", "3DSTATE_SBE", "3DSTATE_SBE_SWIZ", "3DSTATE_SCISSOR_STATE_POINTERS",
    "3DSTATE_SF", "3DSTATE_STENCIL_BUFFER", "3DSTATE_STREAMOUT", "3DSTATE_TASK_CONTROL", "3DSTATE_TE",
    "3DSTATE_URB_ALLOC_DS", "3DSTATE_URB_ALLOC_GS", "3DSTATE_URB_ALLOC_HS", "3DSTATE_URB_ALLOC_VS",
    "3DSTATE_VERTEX_BUFFERS", "3DSTATE_VERTEX_ELEMENTS", "3DSTATE_VF", "3DSTATE_VF_INSTANCING", "3DSTATE_VF_SGVS",
    "3DSTATE_VF_SGVS_2", "3DSTATE_VF_STATISTICS", "3DSTATE_VF_TOPOLOGY", "3DSTATE_VFG",
    "3DSTATE_VIEWPORT_STATE_POINTERS_CC", "3DSTATE_VIEWPORT_STATE_POINTERS_SF_CLIP", "3DSTATE_VS", "3DSTATE_WM",
    "3DSTATE_WM_CHROMAKEY", "3DSTATE_WM_DEPTH_STENCIL", "3DSTATE_WM_HZ_OP", "MI_BATCH_BUFFER_END",
    "MI_LOAD_REGISTER_IMM", "MI_NOOP", "PIPE_CONTROL", "STATE_BASE_ADDRESS", "STATE_COMPUTE_MODE",
    "STATE_SYSTEM_MEM_FENCE_ADDRESS", "BINDING_TABLE_STATE", "BLEND_STATE", "CC_VIEWPORT", "COLOR_CALC_STATE",
    "RENDER_SURFACE_STATE", "SAMPLER_BORDER_COLOR_STATE", "SAMPLER_STATE", "SCISSOR_RECT", "SF_CLIP_VIEWPORT",
];

#[derive(Clone, Debug)]
enum Ty {
    Bool,
    Uint,
    Int,
    Float,
    Address,
    Fixed { signed: bool, frac: u32 },
    Struct(String),
    Mbo,
    Mbz,
}

#[derive(Clone, Debug)]
struct Field {
    name: String,
    bit: u32,
    width: u32,
    ty: Ty,
    default: u64,
    dims: Vec<(u32, u32)>,
    /// the genxml enum its values come from
    of: Option<String>,
}

struct Group {
    at: u32,
    size: u32,
    fields: Vec<Field>,
}

struct Item {
    space: &'static str,
    orig: String,
    length: Option<u32>,
    bias: u32,
    num: u32,
    fields: Vec<Field>,
    group: Option<Group>,
    /// its fields' own values
    values: Vec<(String, u64)>,
}

struct Names;

impl Names {
    fn int(s: &str) -> u64 {
        let s = s.trim();
        match (s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")), s.strip_prefix('-')) {
            (Some(h), _) => u64::from_str_radix(h, 16).unwrap_or_else(|_| panic!("bad hex {s}")),
            (None, Some(n)) => n.parse::<i64>().unwrap_or_else(|_| panic!("bad int {s}")).wrapping_neg() as u64,
            _ => s.parse().unwrap_or_else(|_| panic!("bad int {s}")),
        }
    }

    fn words(s: &str) -> impl Iterator<Item = &str> {
        s.split(|c: char| !c.is_ascii_alphanumeric()).filter(|w| !w.is_empty())
    }

    fn snake(s: &str) -> String {
        Self::words(s).collect::<Vec<_>>().join("_").to_ascii_lowercase()
    }

    fn attr(s: &str) -> String {
        let id = Self::snake(s);
        if id.is_empty() {
            "field".into()
        } else if id.as_bytes()[0].is_ascii_digit() {
            format!("_{id}")
        } else {
            id
        }
    }

    fn camel(space: &str, s: &str) -> String {
        let mut id: String = space[..1].to_ascii_uppercase() + &space[1..];
        for w in Self::words(s) {
            let w = w.to_ascii_lowercase();
            id.push_str(&(w[..1].to_ascii_uppercase() + &w[1..]));
        }
        id
    }

    fn constant(s: &str) -> String {
        let id = Self::words(s).collect::<Vec<_>>().join("_").to_ascii_uppercase();
        if id.as_bytes()[0].is_ascii_digit() { format!("_{id}") } else { id }
    }
}

impl Item {
    fn op(&self) -> String {
        format!("xe3.{}.{}", self.space, Names::snake(&self.orig))
    }

    fn end_bit(fields: &[Field]) -> u32 {
        fields.iter().map(|f| f.bit + f.width + f.dims.iter().map(|(c, s)| (c - 1) * s).sum::<u32>()).max().unwrap_or(0)
    }

    fn fixed_length(&self) -> u32 {
        match (&self.group, self.length) {
            (Some(g), _) => g.at / 32,
            (None, Some(l)) => l,
            (None, None) => Self::end_bit(&self.fields).div_ceil(32),
        }
    }
    fn all_fields(&self) -> impl Iterator<Item = &Field> {
        self.fields.iter().chain(self.group.iter().flat_map(|g| &g.fields))
    }

    fn refs(&self) -> impl Iterator<Item = &String> {
        self.all_fields().filter_map(|f| if let Ty::Struct(s) = &f.ty { Some(s) } else { None })
    }

    /// Only `KEEP` and the structs they nest
    fn keep(items: &mut Vec<Item>) {
        let mut keep: HashSet<String> = items.iter().filter(|i| KEEP.contains(&&*i.orig)).map(Item::op).collect();
        loop {
            let more: Vec<String> = items.iter().filter(|i| keep.contains(&i.op())).flat_map(Item::refs).filter(|s| !keep.contains(*s)).cloned().collect();
            if more.is_empty() {
                break;
            }
            keep.extend(more);
        }
        items.retain(|i| keep.contains(&i.op()));
    }
}

struct Reader<'x> {
    structs: HashSet<&'x str>,
    /// each genxml enum's values
    enums: HashMap<&'x str, Vec<(String, u64)>>,
    values: Vec<(String, u64)>,
}

impl<'x> Reader<'x> {
    fn ty(&self, t: &str) -> Ty {
        match t {
            "bool" => Ty::Bool,
            "uint" => Ty::Uint,
            "int" => Ty::Int,
            "float" => Ty::Float,
            "address" | "offset" => Ty::Address,
            "mbo" => Ty::Mbo,
            "mbz" => Ty::Mbz,
            _ if self.structs.contains(t) => Ty::Struct(format!("xe3.state.{}", Names::snake(t))),
            _ if self.enums.contains_key(t) => Ty::Uint,
            _ if (t.starts_with('u') || t.starts_with('s')) && t[1..].contains('.') => {
                let (_, frac) = t[1..].split_once('.').unwrap();
                Ty::Fixed { signed: t.starts_with('s'), frac: frac.parse().unwrap() }
            }
            _ => Ty::Uint,
        }
    }

    fn values(node: roxmltree::Node, prefix: Option<&str>, out: &mut Vec<(String, u64)>) {
        for v in node.children().filter(|v| v.has_tag_name("value")) {
            let name = v.attribute("name").unwrap();
            let full = prefix.map_or_else(|| name.to_owned(), |p| format!("{p}_{name}"));
            out.push((Names::constant(&full), Names::int(v.attribute("value").unwrap())));
        }
    }

    fn fields(&mut self, node: roxmltree::Node, base: u32, dims: &[(u32, u32)], out: &mut Vec<Field>, group: &mut Option<Group>) {
        for c in node.children().filter(|c| c.is_element()) {
            match c.tag_name().name() {
                "field" => {
                    let b = c.attribute("bits").expect("field without bits");
                    let (hi, lo) = b.split_once(':').unwrap_or((b, b));
                    let (hi, lo): (u32, u32) = (hi.trim().parse().unwrap(), lo.trim().parse().unwrap());
                    let dword: u32 = c.attribute("dword").unwrap().parse().unwrap();
                    Self::values(c, c.attribute("prefix"), &mut self.values);
                    let t = c.attribute("type").unwrap();
                    let ty = self.ty(t);
                    let width = hi - lo + 1;
                    let default = match ty {
                        Ty::Mbo => if width >= 64 { u64::MAX } else { (1 << width) - 1 },
                        _ => c.attribute("default").map_or(0, Names::int),
                    };
                    let of = self.enums.contains_key(t).then(|| t.to_owned());
                    out.push(Field { name: Names::attr(c.attribute("name").unwrap()), bit: base + dword * 32 + lo, width, ty, default, dims: dims.to_vec(), of });
                }
                "group" => {
                    let count: u32 = c.attribute("count").unwrap().parse().unwrap();
                    let dword: u32 = c.attribute("dword").unwrap_or("0").parse().unwrap();
                    let size: u32 = c.attribute("size").unwrap().parse().unwrap();
                    let start = base + dword * 32;
                    if count == 0 {
                        assert!(dims.is_empty() && group.is_none(), "nested or second variable group");
                        let mut fields = Vec::new();
                        self.fields(c, 0, &[], &mut fields, &mut None);
                        *group = Some(Group { at: start, size, fields });
                        continue;
                    }
                    let mut d = dims.to_vec();
                    d.push((count, size));
                    self.fields(c, start, &d, out, group);
                }
                _ => {}
            }
        }
    }

    fn item(&mut self, c: roxmltree::Node) -> Option<Item> {
        let space = match c.tag_name().name() {
            "instruction" => "cmd",
            "struct" => "state",
            "register" => "reg",
            _ => return None,
        };
        let name = c.attribute("name").unwrap();
        let (mut fields, mut group) = (Vec::new(), None);
        self.fields(c, 0, &[], &mut fields, &mut group);
        let values = core::mem::take(&mut self.values);
        Self::dedup(&mut fields);
        if let Some(g) = &mut group {
            Self::dedup(&mut g.fields);
        }
        Some(Item {
            space,
            orig: name.to_owned(),
            length: c.attribute("length").map(|l| l.parse().unwrap()),
            bias: c.attribute("bias").map_or(0, |b| b.parse().unwrap()),
            num: c.attribute("num").map_or(0, |n| Names::int(n) as u32),
            fields,
            group,
            values,
        })
    }

    fn dedup(fields: &mut [Field]) {
        let mut seen: HashMap<String, u32> = HashMap::new();
        for f in fields {
            let n = seen.entry(f.name.clone()).or_default();
            *n += 1;
            if *n > 1 {
                f.name = format!("{}_{}", f.name, n);
            }
        }
    }
}

struct Writer {
    out: String,
}

impl Writer {
    fn ty(t: &Ty) -> String {
        match t {
            Ty::Bool => "Ty::Bool".into(),
            Ty::Uint => "Ty::Uint".into(),
            Ty::Int => "Ty::Int".into(),
            Ty::Float => "Ty::Float".into(),
            Ty::Address => "Ty::Address".into(),
            Ty::Fixed { signed: false, frac } => format!("Ty::Ufixed({frac})"),
            Ty::Fixed { signed: true, frac } => format!("Ty::Sfixed({frac})"),
            Ty::Struct(s) => format!("Ty::Struct({s:?})"),
            Ty::Mbo => "Ty::Mbo".into(),
            Ty::Mbz => unreachable!(),
        }
    }

    fn fields(&mut self, fields: &[Field]) {
        self.out.push_str("&[");
        for f in fields.iter().filter(|f| !matches!(f.ty, Ty::Mbz)) {
            let dims: Vec<String> = f.dims.iter().map(|(c, s)| format!("({c}, {s})")).collect();
            let _ = write!(self.out, "\n            Field::new({:?}, {}, {}, {})", f.name, f.bit, f.width, Self::ty(&f.ty));
            if f.default != 0 {
                let _ = write!(self.out, ".with_default({:#x})", f.default);
            }
            if !dims.is_empty() {
                let _ = write!(self.out, ".with_dims(&[{}])", dims.join(", "));
            }
            self.out.push(',');
        }
        self.out.push_str("\n        ]");
    }

    fn item(&mut self, item: &Item) {
        let _ = write!(
            self.out,
            "    Item {{\n        name: {:?},\n        op: {:?},\n        length: {},\n        bias: {},\n        num: {:#x},\n        fields: ",
            item.orig,
            item.op(),
            item.fixed_length(),
            item.bias,
            item.num
        );
        self.fields(&item.fields);
        match &item.group {
            Some(g) => {
                let _ = write!(self.out, ",\n        group: Some(Group {{ at: {}, size: {}, fields: ", g.at, g.size);
                self.fields(&g.fields);
                self.out.push_str(" }),\n    },\n");
            }
            None => self.out.push_str(",\n        group: None,\n    },\n"),
        }
    }
    /// The kept items' fields' values and their enums' values, each name with one value
    fn values(&mut self, items: &[Item], enums: &HashMap<&str, Vec<(String, u64)>>) {
        let used: HashSet<&str> = items.iter().flat_map(Item::all_fields).filter_map(|f| f.of.as_deref()).collect();
        let values = items.iter().flat_map(|i| &i.values).chain(used.iter().flat_map(|e| &enums[e]));
        let mut by_name: BTreeMap<&str, HashSet<u64>> = BTreeMap::new();
        for (n, v) in values {
            by_name.entry(n).or_default().insert(*v);
        }
        for (n, vs) in &by_name {
            if let [v] = vs.iter().collect::<Vec<_>>()[..] {
                let _ = writeln!(self.out, "pub const XE3_{n}: u32 = {v:#x};");
            }
        }
    }
}

impl Writer {
    fn rust(ty: &Ty, structs: &HashMap<String, String>) -> Option<String> {
        Some(match ty {
            Ty::Bool | Ty::Uint | Ty::Int | Ty::Address => "crate::dialect::layout::Fill".to_owned(),
            Ty::Float | Ty::Fixed { .. } => "crate::dialect::layout::Real".to_owned(),
            Ty::Struct(op) => format!("crate::dialect::layout::Fields<{}Fields>", structs[op]),
            Ty::Mbo | Ty::Mbz => return None,
        })
    }

    fn typed(&mut self, items: &[Item]) {
        let structs: HashMap<String, String> = items.iter().map(|i| (i.op(), Names::camel(i.space, &i.orig))).collect();
        let settable = |fs: &[Field]| fs.iter().filter_map(|f| Some((f.name.clone(), f.dims.iter().fold(Writer::rust(&f.ty, &structs)?, |t, _| format!("Vec<{t}>"))))).collect::<Vec<_>>();
        let nested: HashSet<&String> = items.iter().flat_map(|i| i.fields.iter().chain(i.group.iter().flat_map(|g| &g.fields))).filter_map(|f| if let Ty::Struct(op) = &f.ty { Some(op) } else { None }).collect();
        for i in items.iter().filter(|i| nested.contains(&i.op())) {
            let _ = writeln!(self.out, "/// `{}`'s fields, nested in another item.\n#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]\npub struct {}Fields {{", i.op(), structs[&i.op()]);
            for (n, t) in settable(&i.fields) {
                let _ = writeln!(self.out, "    #[serde(default, skip_serializing_if = \"Option::is_none\")]\n    pub {n}: Option<{t}>,");
            }
            self.out.push_str("}\n\n");
        }
        for i in items {
            let ty = &structs[&i.op()];
            if let Some(g) = &i.group {
                let _ = writeln!(self.out, "/// An element of `{}`'s variable group.\n#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]\npub struct {ty}Element {{", i.op());
                for (n, t) in settable(&g.fields) {
                    let _ = writeln!(self.out, "    #[serde(default, skip_serializing_if = \"Option::is_none\")]\n    pub {n}: Option<{t}>,");
                }
                self.out.push_str("}\n\n");
            }
            let _ = writeln!(self.out, "/// `{}`: genxml's {}.\n#[mlirformat::mlir::op(\"{}\", format = \"attr-dict\")]\n#[derive(Default)]\npub struct {ty} {{", i.op(), i.orig, i.op());
            for (n, t) in settable(&i.fields) {
                let _ = writeln!(self.out, "    pub {n}: Option<{t}>,");
            }
            if i.group.is_some() {
                let _ = writeln!(self.out, "    pub elements: Option<Vec<crate::dialect::layout::Fields<{ty}Element>>>,");
            }
            self.out.push_str("}\n\n");
        }
        self.out.push_str("/// The items' typed ops.\npub struct Xe3Genxml;\n\nimpl Xe3Genxml {\n    /// ⭐ Reads `op` (generic, as a module built from a body has it) as its item's typed op (`None`: not an item).\n    pub fn verify(op: &mlirformat::Operation<'_>) -> Option<Result<(), mlirformat::Error>> {\n        Some(match &*op.name {\n");
        for i in items {
            let _ = writeln!(self.out, "            {:?} => op.to_op::<{}>().map(drop),", i.op(), structs[&i.op()]);
        }
        self.out.push_str("            _ => return None,\n        })\n    }\n}\n\n#[mlirformat::mlir::dialect(\"xe3\", ops = [");
        self.out.push_str(&items.iter().map(|i| structs[&i.op()].clone()).collect::<Vec<_>>().join(", "));
        self.out.push_str("])]\npub struct GenxmlOps;\n\n/// What the typed ops declare (they live in `<{...}>`), for the dialect's `op_infos`.\npub const XE3_INHERENT: &[(&str, &[&str])] = &[\n");
        for i in items {
            let mut names: Vec<String> = settable(&i.fields).into_iter().map(|(n, _)| format!("{n:?}")).collect();
            if i.group.is_some() {
                names.push("\"elements\"".to_owned());
            }
            let _ = writeln!(self.out, "    ({:?}, &[{}]),", i.op(), names.join(", "));
        }
        self.out.push_str("];\n\n");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = &args[..] else { panic!("usage: cmdgen <flat genxml> <out.rs>") };
    let text = std::fs::read_to_string(input).unwrap();
    let doc = roxmltree::Document::parse(&text).unwrap();
    let root = doc.root_element();

    let structs = root.children().filter(|c| c.has_tag_name("struct")).map(|c| c.attribute("name").unwrap()).collect();
    let mut enums: HashMap<&str, Vec<(String, u64)>> = HashMap::new();
    for e in root.children().filter(|c| c.has_tag_name("enum")) {
        Reader::values(e, e.attribute("prefix"), enums.entry(e.attribute("name").unwrap()).or_default());
    }
    let mut r = Reader { structs, enums, values: Vec::new() };
    let mut items: Vec<Item> = root.children().filter_map(|c| r.item(c)).collect();
    Item::keep(&mut items);
    items.sort_by_key(Item::op);
    items.dedup_by_key(|i| i.op());

    let mut w = Writer { out: String::new() };
    let _ = writeln!(
        w.out,
        "// @generated by scripts/xe3gen from {} ({}). Do not edit.\n//\n// Field layouts are Mesa's genxml (Copyright Intel Corporation, MIT license).\n\n#[allow(unused_imports, reason = \"what the items use\")]\nuse alloc::{{vec, vec::Vec}};\n\nuse crate::dialect::layout::{{Field, Group, Item, Ty}};\n",
        std::path::Path::new(input).file_name().unwrap().to_string_lossy(),
        root.attribute("name").unwrap_or("?")
    );
    let _ = writeln!(w.out, "/// Every item's layout, sorted by op name.\npub const XE3_ITEMS: &[Item] = &[");
    for item in &items {
        w.item(item);
    }
    w.out.push_str("];\n\n");
    w.typed(&items);
    w.values(&items, &r.enums);
    std::fs::write(output, &w.out).unwrap();
    eprintln!("{} items -> {output}", items.len());
}
