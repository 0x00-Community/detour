use mlirformat::{Attribute, Content, Error, NamedAttr, Region, Type, Value, mlir};
use mlirpass::{Body, OpId};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextureFormat {
    Rgba8Unorm,
    #[serde(rename = "rgba8unorm-srgb")]
    Rgba8UnormSrgb,
    Bgra8Unorm,
    #[serde(rename = "bgra8unorm-srgb")]
    Bgra8UnormSrgb,
    R8Unorm,
    Rg8Unorm,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,
    #[serde(rename = "bc1-rgba-unorm")]
    Bc1RgbaUnorm,
    #[serde(rename = "bc3-rgba-unorm")]
    Bc3RgbaUnorm,
    #[serde(rename = "bc7-rgba-unorm")]
    Bc7RgbaUnorm,
}

impl TextureFormat {
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Bc1RgbaUnorm => 8,
            Self::Bc3RgbaUnorm | Self::Bc7RgbaUnorm => 16,
            Self::R8Unorm => 1,
            Self::Rg8Unorm | Self::R16Float => 2,
            Self::Rgba8Unorm | Self::Rgba8UnormSrgb | Self::Bgra8Unorm | Self::Bgra8UnormSrgb | Self::Rg16Float | Self::R32Float => 4,
            Self::Rgba16Float | Self::Rg32Float => 8,
            Self::Rgba32Float => 16,
        }
    }

    pub const fn srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    pub const fn block(self) -> (u32, u32) {
        match self {
            Self::Bc1RgbaUnorm | Self::Bc3RgbaUnorm | Self::Bc7RgbaUnorm => (4, 4),
            _ => (1, 1),
        }
    }

    pub const fn compressed(self) -> bool {
        self.block().0 > 1
    }

    pub const fn image_bytes(self, width: u32, height: u32) -> u32 {
        let (bw, bh) = self.block();
        width.div_ceil(bw) * height.div_ceil(bh) * self.bytes()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
}

impl VertexFormat {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float32x2 => "float32x2",
            Self::Float32x3 => "float32x3",
            Self::Float32x4 => "float32x4",
        }
    }

    pub const fn components(self) -> u32 {
        match self {
            Self::Float32 => 1,
            Self::Float32x2 => 2,
            Self::Float32x3 => 3,
            Self::Float32x4 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepMode {
    #[default]
    Vertex,
    Instance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Topology {
    PointList,
    LineList,
    LineStrip,
    #[default]
    TriangleList,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrontFace {
    #[default]
    Ccw,
    Cw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl CompareFunction {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::Less => "less",
            Self::Equal => "equal",
            Self::LessEqual => "less-equal",
            Self::Greater => "greater",
            Self::NotEqual => "not-equal",
            Self::GreaterEqual => "greater-equal",
            Self::Always => "always",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StencilOperation {
    #[default]
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

impl StencilOperation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Zero => "zero",
            Self::Replace => "replace",
            Self::Invert => "invert",
            Self::IncrementClamp => "increment-clamp",
            Self::DecrementClamp => "decrement-clamp",
            Self::IncrementWrap => "increment-wrap",
            Self::DecrementWrap => "decrement-wrap",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoadOp {
    #[default]
    Load,
    Clear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterMode {
    Nearest,
    #[default]
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerState {
    pub address: [AddressMode; 3],
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: Option<MipmapFilter>,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub max_anisotropy: u16,
    pub compare: Option<CompareFunction>,
}

impl SamplerState {
    pub fn of(filter: FilterMode, mipmap: Option<MipmapFilter>) -> Self {
        SamplerState { address: [AddressMode::ClampToEdge; 3], mag_filter: filter, min_filter: filter, mipmap_filter: mipmap, lod_min_clamp: 0.0, lod_max_clamp: 32.0, max_anisotropy: 1, compare: None }
    }
}

impl TryFrom<&Attribute<'_>> for SamplerState {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        let mode = |n: &str| d.get(n).map(Option::unwrap_or_default);
        Ok(SamplerState {
            address: [mode("address_mode_u")?, mode("address_mode_v")?, mode("address_mode_w")?],
            mag_filter: d.get("mag_filter")?.unwrap_or(FilterMode::Nearest),
            min_filter: d.get("min_filter")?.unwrap_or(FilterMode::Nearest),
            mipmap_filter: Some(d.get("mipmap_filter")?.unwrap_or(MipmapFilter::Nearest)),
            lod_min_clamp: d.get("lod_min_clamp")?.unwrap_or(0.0),
            lod_max_clamp: d.get("lod_max_clamp")?.unwrap_or(32.0),
            max_anisotropy: d.get("max_anisotropy")?.unwrap_or(1),
            compare: d.get("compare")?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MipmapFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Blend {
    #[default]
    None,
    Premultiplied,
}

impl From<Blend> for Option<BlendState> {
    fn from(b: Blend) -> Self {
        match b {
            Blend::None => None,
            Blend::Premultiplied => Some(BlendState::PREMULTIPLIED),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
    Src1,
    OneMinusSrc1,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendComponent {
    pub const REPLACE: BlendComponent = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::Zero, operation: BlendOperation::Add };
}

impl TryFrom<&Attribute<'_>> for BlendComponent {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        let r = Self::REPLACE;
        Ok(Self { src_factor: d.get("src_factor")?.unwrap_or(r.src_factor), dst_factor: d.get("dst_factor")?.unwrap_or(r.dst_factor), operation: d.get("operation")?.unwrap_or(r.operation) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const CONSTANT: BlendState = {
        let c = BlendComponent { src_factor: BlendFactor::Constant, dst_factor: BlendFactor::Zero, operation: BlendOperation::Add };
        BlendState { color: c, alpha: c }
    };

    pub const PREMULTIPLIED: BlendState = {
        let c = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add };
        BlendState { color: c, alpha: c }
    };
}

impl From<&BlendComponent> for Attribute<'static> {
    fn from(c: &BlendComponent) -> Self {
        let name = |v: &'static str| Attribute::Str { lit: v.into(), ty: None };
        Attribute::Dict(vec![NamedAttr::of("src_factor", name(c.src_factor.name())), NamedAttr::of("dst_factor", name(c.dst_factor.name())), NamedAttr::of("operation", name(c.operation.name()))])
    }
}

impl From<&BlendState> for Attribute<'static> {
    fn from(b: &BlendState) -> Self {
        Attribute::Dict(vec![NamedAttr::of("color", Attribute::from(&b.color)), NamedAttr::of("alpha", Attribute::from(&b.alpha))])
    }
}

impl BlendFactor {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::One => "one",
            Self::Src => "src",
            Self::OneMinusSrc => "one-minus-src",
            Self::SrcAlpha => "src-alpha",
            Self::OneMinusSrcAlpha => "one-minus-src-alpha",
            Self::Dst => "dst",
            Self::OneMinusDst => "one-minus-dst",
            Self::DstAlpha => "dst-alpha",
            Self::OneMinusDstAlpha => "one-minus-dst-alpha",
            Self::SrcAlphaSaturated => "src-alpha-saturated",
            Self::Constant => "constant",
            Self::OneMinusConstant => "one-minus-constant",
            Self::Src1 => "src1",
            Self::OneMinusSrc1 => "one-minus-src1",
            Self::Src1Alpha => "src1-alpha",
            Self::OneMinusSrc1Alpha => "one-minus-src1-alpha",
        }
    }
}

impl BlendOperation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::ReverseSubtract => "reverse-subtract",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

impl TryFrom<&Attribute<'_>> for BlendState {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        let component = |n: &str| d.raw(n).map_or(Ok(BlendComponent::REPLACE), BlendComponent::try_from);
        Ok(Self { color: component("color")?, alpha: component("alpha")? })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

impl Stage {
    pub const ATTR: &str = "detour.stage";

    pub const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
        }
    }

    pub fn of(body: &Body<'_>, op: OpId) -> Option<Stage> {
        Dict::value(body[op].get(Self::ATTR)?).ok()
    }
}

impl From<Stage> for &'static str {
    fn from(s: Stage) -> Self {
        s.name()
    }
}

impl From<Stage> for Attribute<'_> {
    fn from(s: Stage) -> Self {
        Attribute::Str { lit: s.name().into(), ty: None }
    }
}

pub struct Dict<'r, 'a>(&'r [NamedAttr<'a>]);

impl<'r, 'a> Dict<'r, 'a> {
    pub fn of(a: &'r Attribute<'a>) -> Result<Self, Error> {
        match a {
            Attribute::Dict(d) => Ok(Self(d)),
            a => Err(Error(format!("{a} is not a dictionary"))),
        }
    }

    pub fn value<T: DeserializeOwned>(a: &Attribute<'_>) -> Result<T, Error> {
        Content::Attr(mlirformat::IntoStatic::into_static(a.clone())).to()
    }

    pub fn get<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>, Error> {
        self.0
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| a.value.as_ref())
            .map(|v| Self::value(v).map_err(|e| Error(format!("{name}: {}", e.0))))
            .transpose()
    }

    pub fn raw(&self, name: &str) -> Option<&'r Attribute<'a>> {
        self.0.iter().find(|a| a.name == name).and_then(|a| a.value.as_ref())
    }

    pub fn need<T: DeserializeOwned>(&self, name: &str) -> Result<T, Error> {
        self.get(name)?.ok_or_else(|| Error(format!("no {name}")))
    }

    pub fn each<T>(&self, name: &str, read: impl Fn(&Attribute<'a>) -> Result<T, Error>) -> Result<Vec<T>, Error> {
        match self.0.iter().find(|a| a.name == name).and_then(|a| a.value.as_ref()) {
            None => Ok(Vec::new()),
            Some(Attribute::Array(v)) => v.iter().map(read).collect(),
            Some(a) => Err(Error(format!("{name}: {a} is not an array"))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: VertexFormat,
    pub offset: u32,
}

impl TryFrom<&Attribute<'_>> for VertexAttribute {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        Ok(Self { location: d.need("location")?, format: d.need("format")?, offset: d.get("offset")?.unwrap_or(0) })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexBufferLayout {
    pub stride: u32,
    pub step_mode: StepMode,
    pub attributes: Vec<VertexAttribute>,
}

impl TryFrom<&Attribute<'_>> for VertexBufferLayout {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        Ok(Self { stride: d.need("stride")?, step_mode: d.get("step_mode")?.unwrap_or_default(), attributes: d.each("attributes", |a| VertexAttribute::try_from(a))? })
    }
}

impl From<&VertexAttribute> for Attribute<'static> {
    fn from(v: &VertexAttribute) -> Self {
        let int = |n: u32| Attribute::Int { lit: n.to_string().into(), ty: None };
        Attribute::Dict(vec![
            NamedAttr::of("location", int(v.location)),
            NamedAttr::of("format", Attribute::Str { lit: v.format.name().into(), ty: None }),
            NamedAttr::of("offset", int(v.offset)),
        ])
    }
}

impl From<&VertexBufferLayout> for Attribute<'static> {
    fn from(l: &VertexBufferLayout) -> Self {
        let mut entries = vec![NamedAttr::of("stride", Attribute::Int { lit: l.stride.to_string().into(), ty: None })];
        if l.step_mode == StepMode::Instance {
            entries.push(NamedAttr::of("step_mode", Attribute::Str { lit: "instance".into(), ty: None }));
        }
        entries.push(NamedAttr::of("attributes", Attribute::Array(l.attributes.iter().map(Attribute::from).collect())));
        Attribute::Dict(entries)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthState {
    pub compare: CompareFunction,
    pub write: bool,
    pub stencil: Option<StencilState>,
    pub bias: i32,
    pub bias_slope_scale: f32,
    pub bias_clamp: f32,
    pub bounds: Option<[f32; 2]>,
}

impl DepthState {
    pub fn biased(&self) -> bool {
        self.bias != 0 || self.bias_slope_scale != 0.0
    }
}

impl TryFrom<&Attribute<'_>> for DepthState {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        let stencil = d.raw("stencil").map(StencilState::try_from).transpose()?;
        Ok(Self {
            compare: d.get("compare")?.unwrap_or(CompareFunction::Always),
            write: d.get("write")?.unwrap_or(false),
            stencil,
            bias: d.get("depth_bias")?.unwrap_or(0),
            bias_slope_scale: d.get("depth_bias_slope_scale")?.unwrap_or(0.0),
            bias_clamp: d.get("depth_bias_clamp")?.unwrap_or(0.0),
            bounds: match (d.get::<f32>("depth_bounds_min")?, d.get::<f32>("depth_bounds_max")?) {
                (Some(min), Some(max)) => Some([min, max]),
                (None, None) => None,
                _ => return Err(Error("depth: depth_bounds_min and depth_bounds_max go together".into())),
            },
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilState {
    pub compare: CompareFunction,
    pub fail: StencilOperation,
    pub depth_fail: StencilOperation,
    pub pass: StencilOperation,
    pub read_mask: u8,
    pub write_mask: u8,
    pub back: Option<StencilFace>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    pub compare: CompareFunction,
    pub fail: StencilOperation,
    pub depth_fail: StencilOperation,
    pub pass: StencilOperation,
}

impl TryFrom<&Attribute<'_>> for StencilFace {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        Ok(Self {
            compare: d.get("compare")?.unwrap_or(CompareFunction::Always),
            fail: d.get("fail_op")?.unwrap_or_default(),
            depth_fail: d.get("depth_fail_op")?.unwrap_or_default(),
            pass: d.get("pass_op")?.unwrap_or_default(),
        })
    }
}

impl From<&StencilFace> for Attribute<'static> {
    fn from(f: &StencilFace) -> Self {
        let str = |v: &'static str| Attribute::Str { lit: v.into(), ty: None };
        Attribute::Dict(vec![
            NamedAttr::of("compare", str(f.compare.name())),
            NamedAttr::of("fail_op", str(f.fail.name())),
            NamedAttr::of("depth_fail_op", str(f.depth_fail.name())),
            NamedAttr::of("pass_op", str(f.pass.name())),
        ])
    }
}

impl TryFrom<&Attribute<'_>> for StencilState {
    type Error = Error;

    fn try_from(a: &Attribute<'_>) -> Result<Self, Error> {
        let d = Dict::of(a)?;
        let mask = |name: &str| -> Result<u8, Error> { d.get::<u32>(name)?.map_or(Ok(0xff), |m| u8::try_from(m).map_err(|_| Error(format!("{name}: {m} is not 8 bits")))) };
        Ok(Self {
            compare: d.get("compare")?.unwrap_or(CompareFunction::Always),
            fail: d.get("fail_op")?.unwrap_or_default(),
            depth_fail: d.get("depth_fail_op")?.unwrap_or_default(),
            pass: d.get("pass_op")?.unwrap_or_default(),
            read_mask: mask("read_mask")?,
            write_mask: mask("write_mask")?,
            back: d.raw("back").map(StencilFace::try_from).transpose()?,
        })
    }
}

impl From<&StencilState> for Attribute<'static> {
    fn from(s: &StencilState) -> Self {
        let str = |v: &'static str| Attribute::Str { lit: v.into(), ty: None };
        let int = |n: u8| Attribute::Int { lit: n.to_string().into(), ty: None };
        let mut entries = vec![
            NamedAttr::of("compare", str(s.compare.name())),
            NamedAttr::of("fail_op", str(s.fail.name())),
            NamedAttr::of("depth_fail_op", str(s.depth_fail.name())),
            NamedAttr::of("pass_op", str(s.pass.name())),
            NamedAttr::of("read_mask", int(s.read_mask)),
            NamedAttr::of("write_mask", int(s.write_mask)),
        ];
        entries.extend(s.back.as_ref().map(|b| NamedAttr::of("back", Attribute::from(b))));
        Attribute::Dict(entries)
    }
}

impl From<&DepthState> for Attribute<'static> {
    fn from(d: &DepthState) -> Self {
        let mut entries = vec![NamedAttr::of("compare", Attribute::Str { lit: d.compare.name().into(), ty: None }), NamedAttr::of("write", Attribute::Bool(d.write))];
        entries.extend(d.stencil.as_ref().map(|s| NamedAttr::of("stencil", Attribute::from(s))));
        if d.biased() {
            let float = |v: f32| Attribute::Float { lit: format!("{v:?}").into(), ty: None };
            entries.push(NamedAttr::of("depth_bias", Attribute::Int { lit: d.bias.to_string().into(), ty: None }));
            entries.push(NamedAttr::of("depth_bias_slope_scale", float(d.bias_slope_scale)));
            entries.push(NamedAttr::of("depth_bias_clamp", float(d.bias_clamp)));
        }
        if let Some([min, max]) = d.bounds {
            let float = |v: f32| Attribute::Float { lit: format!("{v:?}").into(), ty: None };
            entries.push(NamedAttr::of("depth_bounds_min", float(min)));
            entries.push(NamedAttr::of("depth_bounds_max", float(max)));
        }
        Attribute::Dict(entries)
    }
}

pub struct Symbol;

impl Symbol {
    pub const ATTR: &str = "sym_name";

    pub fn defined<'r>(a: &'r Attribute<'_>) -> Result<&'r str, Error> {
        match a {
            Attribute::Str { lit, .. } => Ok(lit),
            a => Self::name(a),
        }
    }

    pub fn name<'r>(a: &'r Attribute<'_>) -> Result<&'r str, Error> {
        match a {
            Attribute::SymbolRef(path) if path.len() == 1 => Ok(path[0].trim_matches('"')),
            a => Err(Error(format!("{a} is not a symbol"))),
        }
    }
}

#[mlir::op("detour.buffer", format = "$sym_name attr-dict")]
pub struct Buffer<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub data: Option<Attribute<'a>>,
    pub size: Option<u32>,
    pub readback: bool,
}

#[mlir::op("detour.texture", format = "$sym_name attr-dict")]
pub struct Texture<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub data: Option<Attribute<'a>>,
    pub samples: Option<u32>,
}

#[mlir::op("detour.pipeline", format = "$sym_name attr-dict")]
pub struct Pipeline<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub vertex: Attribute<'a>,
    pub fragment: Attribute<'a>,
    pub topology: Option<Topology>,
    pub targets: Vec<TextureFormat>,
    pub vertex_buffers: Vec<Attribute<'a>>,
    pub blend: Option<Blend>,
    pub filter: Option<FilterMode>,
    pub mipmap: Option<MipmapFilter>,
    pub depth: Option<Attribute<'a>>,
    pub samples: Option<u32>,
    pub blends: Option<Vec<Blend>>,
    pub write_masks: Option<Vec<u32>>,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: Option<FrontFace>,
    pub cull_mode: Option<CullMode>,
    pub color_blends: Option<Vec<Attribute<'a>>>,
    pub sampler: Option<Attribute<'a>>,
    pub alpha_to_coverage: Option<bool>,
    pub sample_shading: Option<bool>,
}

impl Pipeline<'_> {
    pub fn layouts(&self) -> Result<Vec<VertexBufferLayout>, Error> {
        self.vertex_buffers.iter().map(VertexBufferLayout::try_from).collect()
    }

    pub fn blend_states(&self, n: usize) -> Result<Vec<Option<BlendState>>, Error> {
        if let Some(c) = &self.color_blends {
            return c.iter().map(|a| if Dict::of(a)?.0.is_empty() { Ok(None) } else { BlendState::try_from(a).map(Some) }).collect();
        }
        Ok(self.blends.clone().unwrap_or_else(|| vec![self.blend.unwrap_or_default(); n]).into_iter().map(Into::into).collect())
    }

    pub fn sampler_state(&self) -> Result<SamplerState, Error> {
        self.sampler.as_ref().map_or_else(|| Ok(SamplerState::of(self.filter.unwrap_or_default(), self.mipmap)), SamplerState::try_from)
    }

    pub fn depth_state(&self) -> Result<Option<DepthState>, Error> {
        self.depth.as_ref().map(DepthState::try_from).transpose()
    }
}

#[mlir::op("detour.render_pass", format = "attr-dict-with-keyword $body")]
pub struct RenderPass<'a> {
    pub color: Attribute<'a>,
    pub load: Option<LoadOp>,
    pub clear: Option<Vec<f64>>,
    pub statistics: Option<Attribute<'a>>,
    pub body: Region<'a>,
}

#[mlir::op("detour.draw", format = "attr-dict")]
pub struct Draw<'a> {
    pub pipeline: Attribute<'a>,
    pub vertex_buffers: Vec<Attribute<'a>>,
    pub textures: Option<Vec<Attribute<'a>>>,
    pub vertices: Option<u32>,
    pub instances: Option<u32>,
    pub first_vertex: Option<u32>,
    pub index_buffer: Option<Attribute<'a>>,
    pub index_format: Option<IndexFormat>,
    pub indices: Option<u32>,
    pub first_index: Option<u32>,
    pub base_vertex: Option<i32>,
}

#[mlir::op("detour.sample", results(result))]
pub struct Sample<'a> {
    #[slot(uv_ty)]
    pub uv: Value<'a>,
    pub uv_ty: Type<'a>,
    pub binding: u32,
    pub cube: Option<bool>,
    pub result: Type<'a>,
}

#[mlir::op("detour.sample_compare", results(result))]
pub struct SampleCompare<'a> {
    #[slot(uv_ty)]
    pub uv: Value<'a>,
    #[slot(reference_ty)]
    pub reference: Value<'a>,
    pub uv_ty: Type<'a>,
    pub reference_ty: Type<'a>,
    pub binding: u32,
    pub result: Type<'a>,
}

#[mlir::op("detour.load_sample", results(result))]
pub struct LoadSample<'a> {
    #[slot(xy_ty)]
    pub xy: Value<'a>,
    #[slot(sample_ty)]
    pub sample: Value<'a>,
    pub xy_ty: Type<'a>,
    pub sample_ty: Type<'a>,
    pub binding: u32,
    pub result: Type<'a>,
}

#[mlir::op("detour.load", results(result))]
pub struct Load<'a> {
    #[slot(addr_ty)]
    pub addr: Value<'a>,
    pub addr_ty: Type<'a>,
    pub size: Option<u32>,
    pub binding: Option<u32>,
    pub shared: Option<bool>,
    pub result: Type<'a>,
}

#[mlir::op("detour.store")]
pub struct Store<'a> {
    #[slot(addr_ty)]
    pub addr: Value<'a>,
    #[slot(value_ty)]
    pub value: Value<'a>,
    pub addr_ty: Type<'a>,
    pub value_ty: Type<'a>,
    pub size: Option<u32>,
    pub binding: Option<u32>,
    pub shared: Option<bool>,
}

#[mlir::op("detour.barrier")]
pub struct Barrier {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AtomicOp {
    Add,
    Sub,
    Min,
    Max,
    Umin,
    Umax,
    And,
    Or,
    Xor,
    Exchange,
}

#[mlir::op("detour.atomic", results(result))]
pub struct Atomic<'a> {
    #[slot(addr_ty)]
    pub addr: Value<'a>,
    #[slot(value_ty)]
    pub value: Value<'a>,
    pub addr_ty: Type<'a>,
    pub value_ty: Type<'a>,
    pub op: AtomicOp,
    pub shared: Option<bool>,
    pub result: Type<'a>,
}

#[mlir::op("detour.texture_store")]
pub struct TextureStore<'a> {
    #[slot(xy_ty)]
    pub xy: Value<'a>,
    #[slot(value_ty)]
    pub value: Value<'a>,
    pub xy_ty: Type<'a>,
    pub value_ty: Type<'a>,
    pub binding: u32,
}

#[mlir::op("detour.texture_load", results(result))]
pub struct TextureLoad<'a> {
    #[slot(xy_ty)]
    pub xy: Value<'a>,
    pub xy_ty: Type<'a>,
    pub binding: u32,
    pub result: Type<'a>,
}

#[mlir::op("detour.push_constant", results(result))]
pub struct PushConstant<'a> {
    pub offset: u32,
    pub result: Type<'a>,
}

#[mlir::op("detour.sample_index", results(result))]
pub struct SampleIndex<'a> {
    pub result: Type<'a>,
}

#[mlir::op("detour.coverage_mask", results(result))]
pub struct CoverageMask<'a> {
    pub result: Type<'a>,
}

#[mlir::op("detour.lane", results(result))]
pub struct Lane<'a> {
    pub result: Type<'a>,
}

#[mlir::op("detour.workgroup_id", results(result))]
pub struct WorkgroupId<'a> {
    pub dim: Option<u32>,
    pub result: Type<'a>,
}

#[mlir::op("detour.unpack4xu8", results(result))]
pub struct Unpack4xu8<'a> {
    #[slot(word_ty)]
    pub word: Value<'a>,
    pub word_ty: Type<'a>,
    pub result: Type<'a>,
}

#[mlir::dialect("detour", ops = [Buffer, Texture, Pipeline, RenderPass, Draw, Sample, SampleCompare, LoadSample, Load, Store, Atomic, Barrier, TextureStore, TextureLoad, PushConstant, SampleIndex, CoverageMask, Lane, WorkgroupId, Unpack4xu8])]
struct DetourOps;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DetourDialect;

impl mlirformat::Dialect for DetourDialect {
    fn namespace(&self) -> &'static str {
        "detour"
    }

    fn ops(&self) -> Vec<mlirformat::Template> {
        DetourOps.ops()
    }

    fn op_infos(&self) -> Vec<mlirformat::OpInfo> {
        DetourDialect::INHERENT
            .iter()
            .map(|(name, inherent)| {
                let effects = if *name == <Sample as mlirformat::Op>::NAME || *name == <SampleCompare as mlirformat::Op>::NAME || *name == <LoadSample as mlirformat::Op>::NAME { mlirformat::Effects::Known(&[]) } else { mlirformat::Effects::Unknown };
                mlirformat::OpInfo { name, traits: &[], effects, inherent }
            })
            .collect()
    }
}

impl DetourDialect {
    pub const INHERENT: &[(&str, &[&str])] = &[
        ("detour.buffer", &["sym_name", "data", "size", "readback"]),
        ("detour.texture", &["sym_name", "width", "height", "format", "data", "samples"]),
        ("detour.pipeline", &["sym_name", "vertex", "fragment", "topology", "targets", "vertex_buffers", "blend", "filter", "mipmap", "depth", "samples", "blends", "write_masks", "strip_index_format", "front_face", "cull_mode", "color_blends", "sampler", "alpha_to_coverage", "sample_shading"]),
        ("detour.render_pass", &["color", "load", "clear", "statistics"]),
        ("detour.draw", &["pipeline", "vertex_buffers", "textures", "vertices", "instances", "first_vertex", "index_buffer", "index_format", "indices", "first_index", "base_vertex"]),
        ("detour.sample", &["binding", "cube"]),
        ("detour.sample_compare", &["binding"]),
        ("detour.load_sample", &["binding"]),
        ("detour.load", &["size", "binding", "shared"]),
        ("detour.store", &["size", "binding", "shared"]),
        ("detour.atomic", &["op", "shared"]),
        ("detour.barrier", &[]),
        ("detour.texture_store", &["binding"]),
        ("detour.texture_load", &["binding"]),
        ("detour.push_constant", &["offset"]),
        ("detour.sample_index", &[]),
        ("detour.coverage_mask", &[]),
        ("detour.lane", &[]),
        ("detour.workgroup_id", &["dim"]),
        ("detour.unpack4xu8", &[]),
    ];
}

pub struct Attrs<'r, 'a>(pub &'r [NamedAttr<'a>]);

impl<'r, 'a> Attrs<'r, 'a> {
    pub fn get(&self, name: &str) -> Option<&'r Attribute<'a>> {
        self.0.iter().find(|a| a.name == name).and_then(|a| a.value.as_ref())
    }

    fn need(&self, name: &str) -> Result<&'r Attribute<'a>, Error> {
        self.get(name).ok_or_else(|| Error(format!("no {name}")))
    }

    pub fn int(&self, name: &str) -> Result<Option<i64>, Error> {
        self.get(name).map(Self::as_int).transpose()
    }

    pub fn need_int(&self, name: &str) -> Result<i64, Error> {
        Self::as_int(self.need(name)?)
    }

    pub fn bool(&self, name: &str) -> Result<Option<bool>, Error> {
        self.get(name)
            .map(|a| match a {
                Attribute::Bool(b) => Ok(*b),
                Attribute::Unit => Ok(true),
                a => Err(Error(format!("{name}: {a} is not a boolean"))),
            })
            .transpose()
    }

    pub fn str(&self, name: &str) -> Result<&'r str, Error> {
        match self.need(name)? {
            Attribute::Str { lit, .. } => Ok(lit),
            a => Err(Error(format!("{name}: {a} is not a string"))),
        }
    }

    pub fn symbol(&self, name: &str) -> Result<&'r str, Error> {
        Self::as_symbol(self.need(name)?)
    }

    pub fn array(&self, name: &str) -> Result<&'r [Attribute<'a>], Error> {
        match self.get(name) {
            None => Ok(&[]),
            Some(Attribute::Array(v)) => Ok(v),
            Some(a) => Err(Error(format!("{name}: {a} is not an array"))),
        }
    }

    pub fn floats(&self, name: &str) -> Result<Vec<f64>, Error> {
        self.array(name)?.iter().map(Self::as_float).collect()
    }

    pub fn as_int(a: &Attribute<'_>) -> Result<i64, Error> {
        match a {
            Attribute::Int { lit, .. } => match lit.strip_prefix("0x") {
                Some(h) => i64::from_str_radix(h, 16),
                None => lit.parse(),
            }
            .map_err(|_| Error(format!("bad integer {lit}"))),
            a => Err(Error(format!("{a} is not an integer"))),
        }
    }

    fn as_float(a: &Attribute<'_>) -> Result<f64, Error> {
        match a {
            Attribute::Float { lit, .. } => lit.parse().map_err(|_| Error(format!("bad float {lit}"))),
            Attribute::Int { .. } => Self::as_int(a).map(|v| v as f64),
            a => Err(Error(format!("{a} is not a number"))),
        }
    }

    pub fn as_symbol(a: &'r Attribute<'a>) -> Result<&'r str, Error> {
        match a {
            Attribute::SymbolRef(path) if path.len() == 1 => Ok(path[0].trim_matches('"')),
            a => Err(Error(format!("{a} is not a symbol"))),
        }
    }

    pub fn as_dict(a: &'r Attribute<'a>) -> Result<Attrs<'r, 'a>, Error> {
        match a {
            Attribute::Dict(d) => Ok(Attrs(d)),
            a => Err(Error(format!("{a} is not a dictionary"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use mlirformat::{Op, Role};

    use super::*;

    #[test]
    fn inherent() {
        fn props<O: Op>() -> (&'static str, Vec<&'static str>) {
            (O::NAME, O::FIELDS.iter().filter(|f| matches!(f.role, Role::Prop { .. })).map(|f| f.name).collect())
        }
        let typed = [
            props::<Buffer>(),
            props::<Texture>(),
            props::<Pipeline>(),
            props::<RenderPass>(),
            props::<Draw>(),
            props::<Sample>(),
            props::<SampleCompare>(),
            props::<LoadSample>(),
            props::<Load>(),
            props::<Store>(),
            props::<Atomic>(),
            props::<Barrier>(),
            props::<TextureStore>(),
            props::<TextureLoad>(),
            props::<PushConstant>(),
            props::<SampleIndex>(),
            props::<CoverageMask>(),
            props::<Lane>(),
            props::<WorkgroupId>(),
            props::<Unpack4xu8>(),
        ];
        assert_eq!(typed.len(), DetourDialect::INHERENT.len());
        for ((name, mut fields), (listed, inherent)) in typed.into_iter().zip(DetourDialect::INHERENT) {
            let mut inherent = inherent.to_vec();
            fields.sort_unstable();
            inherent.sort_unstable();
            assert_eq!((name, fields), (*listed, inherent));
        }
    }
}
