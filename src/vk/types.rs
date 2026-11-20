use core::fmt;

use super::VkBuffer;
use crate::{ColorFormat, CullMode, DrawPrimitive, VertexFormat};

/// What a Vulkan command fails with: VkResult's error codes; what detour does not do yet is FeatureNotPresent, and
/// what a validation layer would catch is ValidationFailed
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VkError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    MemoryMapFailed,
    FormatNotSupported,
    /// the GPU hung or refused the work
    DeviceLost(String),
    /// what detour does not do yet
    FeatureNotPresent(String),
    /// a valid-usage rule broken (VK_ERROR_VALIDATION_FAILED_EXT)
    ValidationFailed(String),
    /// a shader detour could not compile, and other failures (VK_ERROR_UNKNOWN)
    Unknown(String),
}

impl fmt::Display for VkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VkError::OutOfHostMemory => f.write_str("VK_ERROR_OUT_OF_HOST_MEMORY"),
            VkError::OutOfDeviceMemory => f.write_str("VK_ERROR_OUT_OF_DEVICE_MEMORY"),
            VkError::MemoryMapFailed => f.write_str("VK_ERROR_MEMORY_MAP_FAILED"),
            VkError::FormatNotSupported => f.write_str("VK_ERROR_FORMAT_NOT_SUPPORTED"),
            VkError::DeviceLost(e) => write!(f, "VK_ERROR_DEVICE_LOST: {e}"),
            VkError::FeatureNotPresent(e) => write!(f, "VK_ERROR_FEATURE_NOT_PRESENT: {e}"),
            VkError::ValidationFailed(e) => write!(f, "VK_ERROR_VALIDATION_FAILED_EXT: {e}"),
            VkError::Unknown(e) => write!(f, "VK_ERROR_UNKNOWN: {e}"),
        }
    }
}

impl From<mlirformat::Error> for VkError {
    fn from(e: mlirformat::Error) -> Self {
        VkError::Unknown(e.0)
    }
}

pub type VkResult<T> = Result<T, VkError>;

/// The formats detour takes: color attachments and vertex attributes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkFormat {
    R32Sfloat,
    R32g32Sfloat,
    R32g32b32Sfloat,
    R32g32b32a32Sfloat,
    R8g8b8a8Unorm,
}

impl VkFormat {
    /// The bytes a texel or an attribute takes
    pub const fn bytes(self) -> u32 {
        match self {
            VkFormat::R32Sfloat => 4,
            VkFormat::R32g32Sfloat => 8,
            VkFormat::R32g32b32Sfloat => 12,
            VkFormat::R32g32b32a32Sfloat => 16,
            VkFormat::R8g8b8a8Unorm => 4,
        }
    }

    /// A clear color as the format stores it (UNORM: round to nearest, clamped)
    pub fn texel(self, rgba: [f32; 4]) -> VkResult<Vec<u8>> {
        match self {
            VkFormat::R32g32b32a32Sfloat => Ok(rgba.iter().flat_map(|c| c.to_le_bytes()).collect()),
            VkFormat::R8g8b8a8Unorm => Ok(rgba.iter().map(|c| (c.clamp(0.0, 1.0) * f32::from(u8::MAX)).round() as u8).collect()),
            _ => Err(VkError::FormatNotSupported),
        }
    }
}

impl TryFrom<VkFormat> for ColorFormat {
    type Error = VkError;

    fn try_from(f: VkFormat) -> VkResult<ColorFormat> {
        match f {
            VkFormat::R32g32b32a32Sfloat => Ok(ColorFormat::Rgba32Float),
            VkFormat::R8g8b8a8Unorm => Ok(ColorFormat::Rgba8Unorm),
            _ => Err(VkError::FormatNotSupported),
        }
    }
}

impl TryFrom<VkFormat> for VertexFormat {
    type Error = VkError;

    fn try_from(f: VkFormat) -> VkResult<VertexFormat> {
        match f {
            VkFormat::R32Sfloat => Ok(VertexFormat::Float32),
            VkFormat::R32g32Sfloat => Ok(VertexFormat::Float32x2),
            VkFormat::R32g32b32Sfloat => Ok(VertexFormat::Float32x3),
            VkFormat::R32g32b32a32Sfloat => Ok(VertexFormat::Float32x4),
            VkFormat::R8g8b8a8Unorm => Ok(VertexFormat::Unorm8x4),
        }
    }
}

impl From<VertexFormat> for VkFormat {
    fn from(f: VertexFormat) -> VkFormat {
        match f {
            VertexFormat::Float32 => VkFormat::R32Sfloat,
            VertexFormat::Float32x2 => VkFormat::R32g32Sfloat,
            VertexFormat::Float32x3 => VkFormat::R32g32b32Sfloat,
            VertexFormat::Float32x4 => VkFormat::R32g32b32a32Sfloat,
            VertexFormat::Unorm8x4 => VkFormat::R8g8b8a8Unorm,
        }
    }
}

/// VkMemoryPropertyFlags
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct VkMemoryPropertyFlags {
    pub device_local: bool,
    pub host_visible: bool,
    pub host_coherent: bool,
    pub host_cached: bool,
}

impl VkMemoryPropertyFlags {
    /// Every property `want` asks for is one of these
    pub const fn contains(self, want: VkMemoryPropertyFlags) -> bool {
        (self.device_local || !want.device_local) && (self.host_visible || !want.host_visible) && (self.host_coherent || !want.host_coherent) && (self.host_cached || !want.host_cached)
    }
}

/// VkMemoryType
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkMemoryType {
    pub property_flags: VkMemoryPropertyFlags,
    pub heap_index: u32,
}

/// VkMemoryRequirements: `memory_type_bits` has bit i set when memory type i can back the object
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkMemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    pub memory_type_bits: u32,
}

/// VkExtent2D / VkOffset2D / VkRect2D
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct VkRect2D {
    pub offset: [i32; 2],
    pub extent: [u32; 2],
}

/// VkViewport
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct VkViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

/// VkImageCreateInfo (2D, one level, one layer, one sample)
#[derive(Clone, Copy, Debug)]
pub struct VkImageCreateInfo {
    pub format: VkFormat,
    pub extent: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkAttachmentLoadOp {
    Load,
    Clear,
    DontCare,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkAttachmentStoreOp {
    Store,
    DontCare,
}

/// VkAttachmentDescription (the layouts are not kept: detour's images have one layout)
#[derive(Clone, Copy, Debug)]
pub struct VkAttachmentDescription {
    pub format: VkFormat,
    pub load_op: VkAttachmentLoadOp,
    pub store_op: VkAttachmentStoreOp,
}

/// VkRenderPassCreateInfo with one subpass: its color attachments, by index into `attachments`
#[derive(Clone, Copy, Debug)]
pub struct VkRenderPassCreateInfo<'a> {
    pub attachments: &'a [VkAttachmentDescription],
    pub color_attachments: &'a [u32],
}

/// VkClearValue
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VkClearValue {
    Color([f32; 4]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VkShaderStage {
    Vertex,
    Fragment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkPrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
}

impl TryFrom<VkPrimitiveTopology> for DrawPrimitive {
    type Error = VkError;

    fn try_from(t: VkPrimitiveTopology) -> VkResult<DrawPrimitive> {
        match t {
            VkPrimitiveTopology::PointList => Ok(DrawPrimitive::Points),
            VkPrimitiveTopology::LineList => Ok(DrawPrimitive::Lines),
            VkPrimitiveTopology::LineStrip => Ok(DrawPrimitive::LineStrip),
            VkPrimitiveTopology::TriangleList => Ok(DrawPrimitive::Triangles),
            VkPrimitiveTopology::TriangleStrip => Ok(DrawPrimitive::TriangleStrip),
            VkPrimitiveTopology::TriangleFan => Err(VkError::FeatureNotPresent("triangle fans".into())),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VkCullMode {
    #[default]
    None,
    Front,
    Back,
    FrontAndBack,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VkFrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VkVertexInputRate {
    #[default]
    Vertex,
    Instance,
}

/// VkVertexInputBindingDescription
#[derive(Clone, Copy, Debug)]
pub struct VkVertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: VkVertexInputRate,
}

/// VkVertexInputAttributeDescription
#[derive(Clone, Copy, Debug)]
pub struct VkVertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: VkFormat,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkDescriptorType {
    UniformBuffer,
}

/// VkDescriptorSetLayoutBinding
#[derive(Clone, Copy, Debug)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: VkDescriptorType,
    pub descriptor_count: u32,
}

/// VkRasterizationStateCreateInfo's state detour keeps
#[derive(Clone, Copy, Debug, Default)]
pub struct VkRasterization {
    pub cull_mode: VkCullMode,
    pub front_face: VkFrontFace,
}

impl VkRasterization {
    /// The draw's cull mode, or None when every face is culled (nothing of a triangle is drawn)
    pub fn cull(self, primitive: DrawPrimitive) -> Option<CullMode> {
        let triangles = matches!(primitive, DrawPrimitive::Triangles | DrawPrimitive::TriangleStrip);
        match self.cull_mode {
            VkCullMode::None => Some(CullMode::None),
            VkCullMode::Front => Some(CullMode::Front),
            VkCullMode::Back => Some(CullMode::Back),
            VkCullMode::FrontAndBack if triangles => None,
            VkCullMode::FrontAndBack => Some(CullMode::None),
        }
    }
}

/// VkBufferImageCopy (color, level 0, layer 0): `buffer_row_length` / `buffer_image_height` 0 are the image extent's
#[derive(Clone, Copy, Debug)]
pub struct VkBufferImageCopy {
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_offset: [u32; 2],
    pub image_extent: [u32; 2],
}

/// VkDescriptorBufferInfo: `range` None is VK_WHOLE_SIZE
#[derive(Clone, Copy, Debug)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: u64,
    pub range: Option<u64>,
}
