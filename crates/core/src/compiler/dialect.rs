//! ⭐ The `detour` dialect: what a WebGPU queue submission is, as MLIR.
//!
//! ```mlir
//! detour.buffer @vb {data = dense<[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]> : tensor<6xf32>}
//! detour.texture @rt {width = 64, height = 64, format = "rgba8unorm"}     (`data = dense<...> : tensor<Nxi8>`: its texels)
//! detour.pipeline @tri {vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"],
//!                      vertex_buffers = [{stride = 8, attributes = [{location = 0, format = "float32x2", offset = 0}]}]}
//! detour.render_pass {color = @rt, load = "clear", clear = [0.0, 0.0, 1.0, 1.0]} {
//!   detour.draw {pipeline = @tri, vertex_buffers = [@vb], vertices = 3}
//! }
//! ```
//!
//! Every op is a typed `#[mlir::op]`: its properties are read once, when the op is, into Rust types (the WebGPU
//! enums below, numbers, symbols), and nothing downstream looks at attribute names or strings. Strings become enums
//! there: a misspelled `topology` is a parse error. The nested dictionaries (`vertex_buffers`' layouts, `depth`) are
//! read into their own structs the same way ([`VertexBufferLayout`], [`DepthState`]).
//!
//! The shaders are `func.func`s with `detour.stage` (see [`Stage`]); the ops a shader uses besides `arith` /
//! `vector` / `math` are [`Sample`], [`Load`], [`Store`], [`Lane`], [`WorkgroupId`], [`Unpack4xu8`].

use mlirformat::{Attribute, Content, Error, NamedAttr, Region, Type, Value, mlir};
use mlirpass::{Body, OpId};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

// ---- WebGPU's enums ---------------------------------------------------------------------------------------------

/// `GPUTextureFormat` (the ones detour renders to and samples): WebGPU's names.
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
    /// BC1 (DXT1): 4 x 4 texels in 8 bytes, sampled only
    #[serde(rename = "bc1-rgba-unorm")]
    Bc1RgbaUnorm,
    /// BC3 (DXT5): 4 x 4 texels in 16 bytes, sampled only
    #[serde(rename = "bc3-rgba-unorm")]
    Bc3RgbaUnorm,
    /// BC7: 4 x 4 texels in 16 bytes, sampled only
    #[serde(rename = "bc7-rgba-unorm")]
    Bc7RgbaUnorm,
}

impl TextureFormat {
    /// Bytes an element: a texel, or a block of a compressed format ([`block`](Self::block)).
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

    /// Whether its texels are sRGB-encoded (sampled as linear, rendered and blended in linear and encoded).
    pub const fn srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    /// An element's texels, width and height: 4 x 4 a compressed format's block, else 1 x 1.
    pub const fn block(self) -> (u32, u32) {
        match self {
            Self::Bc1RgbaUnorm | Self::Bc3RgbaUnorm | Self::Bc7RgbaUnorm => (4, 4),
            _ => (1, 1),
        }
    }

    /// Whether it is block-compressed (sampled only: not a render target, not a storage texture).
    pub const fn compressed(self) -> bool {
        self.block().0 > 1
    }

    /// The bytes of `width` × `height` texels, tightly packed (a row of blocks for each 4 rows when compressed).
    pub const fn image_bytes(self, width: u32, height: u32) -> u32 {
        let (bw, bh) = self.block();
        width.div_ceil(bw) * height.div_ceil(bh) * self.bytes()
    }
}

/// `GPUVertexFormat` (the float ones).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
}

impl VertexFormat {
    /// The WebGPU spelling.
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

/// `GPUVertexStepMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepMode {
    #[default]
    Vertex,
    Instance,
}

/// `GPUPrimitiveTopology`.
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

/// `GPUFrontFace`: which winding, as seen on the target, faces the viewer (ccw: counter-clockwise on the screen, as
/// WebGPU's y-up clip space and Vulkan's signed area both mean it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrontFace {
    #[default]
    Ccw,
    Cw,
}

/// `GPUCullMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

/// `GPUIndexFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

/// `GPUCompareFunction`.
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
    /// Its WebGPU name, as the dialect spells it.
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

/// `GPUStencilOperation`.
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
    /// Its WebGPU name, as the dialect spells it.
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

/// `GPULoadOp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoadOp {
    #[default]
    Load,
    Clear,
}

/// `GPUFilterMode`: a pipeline's texture filter (`filter`, linear when absent).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterMode {
    Nearest,
    #[default]
    Linear,
}

/// `GPUAddressMode`: what a coordinate outside 0 ..= 1 samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// `GPUSamplerDescriptor` (without `compare`): `{address_mode_u = "repeat", address_mode_v, address_mode_w, mag_filter
/// = "linear", min_filter, mipmap_filter = "nearest", lod_min_clamp = 0.0, lod_max_clamp = 32.0, max_anisotropy = 1}`,
/// absent entries WebGPU's defaults (clamp-to-edge, nearest, 0, 32, 1); a pipeline's samplers (`sampler`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerState {
    pub address: [AddressMode; 3],
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    /// none: level 0 only (the pipelines' `mipmap` shorthand absent)
    pub mipmap_filter: Option<MipmapFilter>,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub max_anisotropy: u16,
    /// a comparison sampler's function (`detour.sample_compare`: 1 where `ref compare depth` holds, else 0)
    pub compare: Option<CompareFunction>,
}

impl SamplerState {
    /// The pipelines' shorthand: `filter` both ways, `mipmap`, clamped to the edge.
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

/// `GPUMipmapFilterMode`: a pipeline's mip filter (`mipmap`; absent: level 0 only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MipmapFilter {
    Nearest,
    Linear,
}

/// A pipeline's blending (`blend`): none (the output replaces the target), or premultiplied alpha over it
/// (`src + dst * (1 - src.a)`).
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

/// `GPUBlendFactor` (without the dual-source ones).
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
    /// dual-source blending (WebGPU `dual-source-blending`): the fragment shader's second color
    Src1,
    OneMinusSrc1,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

/// `GPUBlendOperation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// `GPUBlendComponent`: `{src_factor = "src-alpha", dst_factor = "one-minus-src-alpha", operation = "add"}`, absent
/// entries WebGPU's defaults (one, zero, add).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// `src · src_factor ⊕ dst · dst_factor`'s defaults: the source replaces the target.
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

/// `GPUBlendState`: `{color = {...}, alpha = {...}}` (each a [`BlendComponent`], absent: replace).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    /// The blend constant replaces the target (with a shader returning 1: `1 · constant + dst · 0`).
    pub const CONSTANT: BlendState = {
        let c = BlendComponent { src_factor: BlendFactor::Constant, dst_factor: BlendFactor::Zero, operation: BlendOperation::Add };
        BlendState { color: c, alpha: c }
    };

    /// Premultiplied alpha over the target: `src + dst · (1 − src.a)`, color and alpha alike.
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
    /// Its WebGPU name, as the dialect spells it.
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
    /// Its WebGPU name, as the dialect spells it.
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

/// A pipeline stage: `detour.stage = "vertex"` on a shader's `func.func` (a discardable attribute of `func`'s op, so
/// read with [`Stage::of`]).
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

    /// The stage of `op`, `None` when it is not a shader.
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

// ---- dictionaries ------------------------------------------------------------------------------------------------

/// ⭐ A dictionary attribute whose entries are read as typed values, through mlirformat's own attribute ⇒ value
/// conversion (`Content`), the one an op's typed fields go through: a string into an enum, an integer into a number.
pub struct Dict<'r, 'a>(&'r [NamedAttr<'a>]);

impl<'r, 'a> Dict<'r, 'a> {
    pub fn of(a: &'r Attribute<'a>) -> Result<Self, Error> {
        match a {
            Attribute::Dict(d) => Ok(Self(d)),
            a => Err(Error(format!("{a} is not a dictionary"))),
        }
    }

    /// An attribute as a `T`.
    pub fn value<T: DeserializeOwned>(a: &Attribute<'_>) -> Result<T, Error> {
        Content::Attr(mlirformat::IntoStatic::into_static(a.clone())).to()
    }

    /// Entry `name` as a `T`, `None` when absent.
    pub fn get<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>, Error> {
        self.0
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| a.value.as_ref())
            .map(|v| Self::value(v).map_err(|e| Error(format!("{name}: {}", e.0))))
            .transpose()
    }

    /// Entry `name` as it is (a nested dictionary), `None` when absent.
    pub fn raw(&self, name: &str) -> Option<&'r Attribute<'a>> {
        self.0.iter().find(|a| a.name == name).and_then(|a| a.value.as_ref())
    }

    pub fn need<T: DeserializeOwned>(&self, name: &str) -> Result<T, Error> {
        self.get(name)?.ok_or_else(|| Error(format!("no {name}")))
    }

    /// Entry `name`'s array, each element read by `read` (empty when absent).
    pub fn each<T>(&self, name: &str, read: impl Fn(&Attribute<'a>) -> Result<T, Error>) -> Result<Vec<T>, Error> {
        match self.0.iter().find(|a| a.name == name).and_then(|a| a.value.as_ref()) {
            None => Ok(Vec::new()),
            Some(Attribute::Array(v)) => v.iter().map(read).collect(),
            Some(a) => Err(Error(format!("{name}: {a} is not an array"))),
        }
    }
}

/// `{location = 0, format = "float32x2", offset = 0}`: a vertex attribute (`GPUVertexAttribute`).
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

/// `{stride = 8, step_mode = "instance", attributes = [...]}`: a vertex buffer's layout (`GPUVertexBufferLayout`).
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

/// `{compare = "less", write = true, stencil = {...}?, depth_bias = 2, depth_bias_slope_scale = 1.0, depth_bias_clamp =
/// 0.0}`: a pipeline's depth test on a depth32float attachment, its stencil test on the attachment's stencil8, and the
/// bias added to its triangles' depth (`GPUDepthStencilState`, depth32float-stencil8; the bias in units of the depth
/// format's resolution at the triangle's depth, plus the slope times its depth slope, clamped to `clamp` when not 0:
/// Vulkan's depthBias*, none when absent); `depth_bounds_min` / `depth_bounds_max` (both): Vulkan's depth bounds test,
/// a fragment kept only where the attachment's depth already there is in [min, max].
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
    /// Whether it biases the depth.
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

/// `{compare = "equal", fail_op = "keep", depth_fail_op = "keep", pass_op = "replace", read_mask = 255, write_mask =
/// 255, back = {compare, fail_op, depth_fail_op, pass_op}?}`: the stencil test (`GPUStencilFaceState` and the masks):
/// the front face's, and the back face's when `back` is given (`stencilBack`; otherwise back faces test as the front);
/// the masks and the reference value (the draw's, `setStencilReference`) are both faces'. Absent entries as WebGPU's
/// defaults.
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

/// One face's stencil test and operations (`GPUStencilFaceState`).
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

/// A flat symbol reference `@name`, or a symbol's own name (`sym_name`): its name.
pub struct Symbol;

impl Symbol {
    /// The attribute naming a symbol (`SymbolTable::getSymbolAttrName`).
    pub const ATTR: &str = "sym_name";

    /// A `sym_name`: its string (or a symbol written `@name`).
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

// ---- the queue's ops ---------------------------------------------------------------------------------------------

/// `detour.buffer @name {data = dense<...>}`: a buffer and its contents; or `{size = n, readback}`: `n` bytes the GPU
/// writes, read back on the CPU after the submission (`readback`: coherent memory).
#[mlir::op("detour.buffer", format = "$sym_name attr-dict")]
pub struct Buffer<'a> {
    #[attr(symbol)]
    pub sym_name: Attribute<'a>,
    pub data: Option<Attribute<'a>>,
    pub size: Option<u32>,
    pub readback: bool,
}

/// `detour.texture @name {width, height, format, data?, samples?}`: a texture (`data`: its texels, `tensor<Nxi8>`;
/// `samples = 4`: multisampled, a render target only).
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

/// `detour.pipeline @name {...}`: a render pipeline (`GPURenderPipelineDescriptor`): its shaders (`@vertex`,
/// `@fragment`), topology (triangle-list when absent), target formats, vertex buffer layouts, blending, the filter and
/// mip filter its samplers use, its depth test (with a depth attachment only), and its targets' sample count
/// (`samples = 4`, `GPUMultisampleState.count`; 1 when absent). The rest of `GPUPrimitiveState`: `strip_index_format`
/// (a strip topology's restart index: all ones of that format ends a strip), `front_face` (ccw when absent),
/// `cull_mode` (none when absent).
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
    /// Each target's blending, overriding `blend` (`GPUColorTargetState.blend`).
    pub blends: Option<Vec<Blend>>,
    /// Each target's channels written, `GPUColorWrite` bits: red 1, green 2, blue 4, alpha 8 (all when absent).
    pub write_masks: Option<Vec<u32>>,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: Option<FrontFace>,
    pub cull_mode: Option<CullMode>,
    /// Each target's `GPUBlendState` (`{color = {...}, alpha = {...}}`; `{}` none), overriding `blend` / `blends`.
    pub color_blends: Option<Vec<Attribute<'a>>>,
    /// Its samplers' [`SamplerState`] (`GPUSamplerDescriptor`), overriding `filter` / `mipmap`.
    pub sampler: Option<Attribute<'a>>,
    /// `GPUMultisampleState.alphaToCoverageEnabled`: target 0's alpha picks how many of a pixel's samples it covers.
    pub alpha_to_coverage: Option<bool>,
    /// The fragment shader once per sample, its varyings interpolated at the sample (Vulkan `sampleShadingEnable` with
    /// `minSampleShading` 1; what WGSL's `@interpolate(..., sample)` asks), not once per pixel.
    pub sample_shading: Option<bool>,
}

impl Pipeline<'_> {
    /// Its vertex buffer layouts, read.
    pub fn layouts(&self) -> Result<Vec<VertexBufferLayout>, Error> {
        self.vertex_buffers.iter().map(VertexBufferLayout::try_from).collect()
    }

    /// Each of its `n` targets' blending: `color_blends`, else `blends`, else `blend` for all (none when absent).
    pub fn blend_states(&self, n: usize) -> Result<Vec<Option<BlendState>>, Error> {
        if let Some(c) = &self.color_blends {
            return c.iter().map(|a| if Dict::of(a)?.0.is_empty() { Ok(None) } else { BlendState::try_from(a).map(Some) }).collect();
        }
        Ok(self.blends.clone().unwrap_or_else(|| vec![self.blend.unwrap_or_default(); n]).into_iter().map(Into::into).collect())
    }

    /// Its samplers' state: `sampler`, else `filter` (linear when absent) and `mipmap`.
    pub fn sampler_state(&self) -> Result<SamplerState, Error> {
        self.sampler.as_ref().map_or_else(|| Ok(SamplerState::of(self.filter.unwrap_or_default(), self.mipmap)), SamplerState::try_from)
    }

    /// Its depth test, read.
    pub fn depth_state(&self) -> Result<Option<DepthState>, Error> {
        self.depth.as_ref().map(DepthState::try_from).transpose()
    }
}

/// `detour.render_pass {color = @t, load = "clear", clear = [r, g, b, a], statistics = @q} { draws }`: a render pass
/// into texture `color` (`load = "clear"` clears it to `clear` first; `statistics`: the pipeline statistics of its
/// draws stored into buffer `@q`).
#[mlir::op("detour.render_pass", format = "attr-dict-with-keyword $body")]
pub struct RenderPass<'a> {
    pub color: Attribute<'a>,
    pub load: Option<LoadOp>,
    pub clear: Option<Vec<f64>>,
    pub statistics: Option<Attribute<'a>>,
    pub body: Region<'a>,
}

/// `detour.draw {pipeline = @p, vertex_buffers = [@vb], textures = [@t], vertices = 3}`: one draw, naming all its state
/// (`setPipeline` / `setVertexBuffer` / bind groups already applied); `instances` defaults to 1, `first_vertex` to 0.
/// Indexed (`drawIndexed`): `index_buffer = @ib, index_format = "uint16", indices = n` (instead of `vertices`),
/// `first_index`, `base_vertex`.
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

// ---- the shaders' ops --------------------------------------------------------------------------------------------

/// `%c = "detour.sample"(%uv) {binding = 0} : (vector<2xf32>) -> vector<4xf32>` (WGSL `textureSample`): texture
/// `binding` of the draw at normalized coordinates, the level from the 2x2 quad's derivatives, with the pipeline's
/// sampler; `vector<3xf32>` (u, v, layer) samples a 2D array (WGSL's `array_index`, rounded and clamped), or with
/// `cube = true` a cube by a direction (any length: `detour-normalize-cube` divides it by its largest magnitude).
#[mlir::op("detour.sample", results(result))]
pub struct Sample<'a> {
    #[slot(uv_ty)]
    pub uv: Value<'a>,
    pub uv_ty: Type<'a>,
    pub binding: u32,
    pub cube: Option<bool>,
    pub result: Type<'a>,
}

/// `%c = "detour.sample_compare"(%uv, %ref) {binding = 0} : (vector<2xf32>, f32) -> f32` (WGSL `textureSampleCompare`
/// of a `texture_depth_2d`): 1 where `ref compare depth` holds at the texel (the sampler's `compare`), else 0; with a
/// linear filter the texels' results filtered (percentage closer).
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

/// `%c = "detour.load_sample"(%xy, %s) {binding = 0} : (vector<2xi32>, i32) -> vector<4xf32>` (WGSL `textureLoad` of a
/// `texture_multisampled_2d`): texel (x, y) of multisampled texture `binding`, sample `s`, unfiltered.
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

/// `%v = "detour.load"(%addr) {size = 1 | 2 | 4, binding?, shared?} : (i32) -> T`: memory at a byte address per
/// invocation (a flat 32-bit address; with `binding = k` an offset into buffer binding k; `shared`: of the
/// workgroup's shared memory, from 0), `size` bytes a component (4 when absent; 1 and 2 zero-extended, scalars only).
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

/// `"detour.store"(%addr, %v) {size, binding?} : (i32, T) -> ()`: as [`Load`].
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

/// `"detour.barrier"() : () -> ()` (WGSL `workgroupBarrier`): every invocation of the workgroup waits here until all
/// have come (their shared memory's writes before it seen after it). Only where all its invocations come.
#[mlir::op("detour.barrier")]
pub struct Barrier {}

/// What `detour.atomic` does to the dword (WGSL's `atomicAdd` .. `atomicExchange`; min / max signed, umin / umax
/// unsigned).
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

/// `%old = "detour.atomic"(%addr, %v) {op = "add"} : (i32, i32) -> i32`: a read-modify-write of the dword at a byte
/// address per invocation (flat, 32-bit; `shared`: of the shared memory; as [`Load`]), atomic with every other
/// invocation's; its value before.
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

/// `"detour.texture_store"(%xy, %v) {binding = 0} : (vector<2xi32>, vector<4xf32>) -> ()` (WGSL `textureStore` of a
/// `texture_storage_2d`): texel (x, y) of storage texture `binding` set to `v` (1 ..= 4 components, f32 or i32 / u32 as the
/// format's channels are read; converted to the format by the data port), unless outside the texture.
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

/// `%v = "detour.texture_load"(%xy) {binding = 0} : (vector<2xi32>) -> vector<4xf32>` (WGSL `textureLoad` of a
/// `texture_storage_2d` with read access): texel (x, y) of storage texture `binding`, its channels as the format reads.
#[mlir::op("detour.texture_load", results(result))]
pub struct TextureLoad<'a> {
    #[slot(xy_ty)]
    pub xy: Value<'a>,
    pub xy_ty: Type<'a>,
    pub binding: u32,
    pub result: Type<'a>,
}

/// `%v = "detour.push_constant"() {offset = 16} : () -> vector<4xf32>` (WGSL `var<immediate>`; Vulkan push constants):
/// the value at byte `offset` (4-aligned) of the draw's push constants (`Recorder::push_constants`), the same in every
/// invocation; 1 ..= 4 dwords.
#[mlir::op("detour.push_constant", results(result))]
pub struct PushConstant<'a> {
    pub offset: u32,
    pub result: Type<'a>,
}

/// `%s = "detour.sample_index"() : () -> i32` (WGSL `@builtin(sample_index)`): the sample a fragment shader with
/// `sample_shading` runs for (only meaningful with it).
#[mlir::op("detour.sample_index", results(result))]
pub struct SampleIndex<'a> {
    pub result: Type<'a>,
}

/// `%m = "detour.coverage_mask"() : () -> i32` (WGSL `@builtin(sample_mask)` in): the samples of the fragment's pixel
/// its primitive covers (bit k: sample k; the whole pixel's even with `sample_shading`: AND it with 1 << sample_index
/// for WGSL's per-sample value).
#[mlir::op("detour.coverage_mask", results(result))]
pub struct CoverageMask<'a> {
    pub result: Type<'a>,
}

/// `%l = "detour.lane"() : () -> i32`: the invocation's lane, 0 ..= 15 (in a workgroup of more threads, `detour.workgroup`:
/// its invocation index, 0 ..= 16 threads - 1).
#[mlir::op("detour.lane", results(result))]
pub struct Lane<'a> {
    pub result: Type<'a>,
}

/// `%w = "detour.workgroup_id"() {dim = 0 | 1 | 2} : () -> i32`: the workgroup's index in the dispatch's grid (x when
/// absent).
#[mlir::op("detour.workgroup_id", results(result))]
pub struct WorkgroupId<'a> {
    pub dim: Option<u32>,
    pub result: Type<'a>,
}

/// `%b = "detour.unpack4xu8"(%w) : (i32) -> vector<4xi32>` (WGSL `unpack4xU8`, low byte first; `vector<4xf32>`: the
/// bytes as floats 0 ..= 255).
#[mlir::op("detour.unpack4xu8", results(result))]
pub struct Unpack4xu8<'a> {
    #[slot(word_ty)]
    pub word: Value<'a>,
    pub word_ty: Type<'a>,
    pub result: Type<'a>,
}

#[mlir::dialect("detour", ops = [Buffer, Texture, Pipeline, RenderPass, Draw, Sample, SampleCompare, LoadSample, Load, Store, Atomic, Barrier, TextureStore, TextureLoad, PushConstant, SampleIndex, CoverageMask, Lane, WorkgroupId, Unpack4xu8])]
struct DetourOps;

/// ⭐ The `detour` dialect: load it with `MlirOpt::load(DetourDialect)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DetourDialect;

impl mlirformat::Dialect for DetourDialect {
    fn namespace(&self) -> &'static str {
        "detour"
    }

    fn ops(&self) -> Vec<mlirformat::Template> {
        DetourOps.ops()
    }

    /// Every op's properties are inherent (so a body keeps them where the typed op reads them, whichever form the
    /// text spelled them in); every op has effects (none may be dropped as dead) but `detour.sample`, a pure read of a
    /// texture (it can be moved, merged, dropped when unused).
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
    /// Each op's properties, by name (checked against the typed ops' fields by the `inherent` test).
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

/// ⭐ An op's attributes, read as the detour dialect spells them.
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

    /// A boolean (`true` / `false`; a unit attribute is `true`).
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

    /// A flat symbol reference `@x`.
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

    /// Floats (or integers) of an array.
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

    /// The inherent lists are the typed ops' properties, exactly.
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
