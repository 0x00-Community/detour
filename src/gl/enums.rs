use super::{
    AttachPoint, BufferTarget, ColorBuffer, Compare, Filter, FramebufferStatus, FramebufferTarget, GlError, GlResult, ImageTarget, InternalFormat, PixelFormat, PixelType, ShaderType, Swizzle, TextureTarget,
    Usage, Wrap,
};

include!("enums.gen.rs");

/// A GL enum parameter's conversions from its `GL` table: a value the table does not list is INVALID_ENUM
macro_rules! gl_enums {
    ($($t:ty),* $(,)?) => {$(
        impl $t {
            pub fn from_gl(v: u32) -> GlResult<Self> {
                Self::GL.iter().find(|(g, _)| *g == v).map(|(_, e)| *e).ok_or(GlError::InvalidEnum)
            }

            pub fn to_gl(self) -> u32 {
                Self::GL.iter().find(|(_, e)| *e == self).map_or(Gl::NONE, |(g, _)| *g)
            }
        }
    )*};
}

gl_enums!(GlError, BufferTarget, Usage, ShaderType, TextureTarget, ImageTarget, InternalFormat, PixelFormat, PixelType, Filter, Wrap, Compare, Swizzle, FramebufferTarget, FramebufferStatus);

impl Gl {
    /// A value by its name (with or without GL_)
    pub fn value(name: &str) -> Option<u32> {
        let name = name.strip_prefix("GL_").unwrap_or(name);
        Gl::NAMES.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }

    /// A value's first name
    pub fn name(v: u32) -> Option<String> {
        Gl::NAMES.iter().find(|(_, x)| *x == v).map(|(n, _)| (*n).to_owned())
    }
}

impl GlError {
    const GL: &[(u32, Self)] = &[
        (Gl::INVALID_ENUM, GlError::InvalidEnum),
        (Gl::INVALID_VALUE, GlError::InvalidValue),
        (Gl::INVALID_OPERATION, GlError::InvalidOperation),
        (Gl::INVALID_FRAMEBUFFER_OPERATION, GlError::InvalidFramebufferOperation),
        (Gl::OUT_OF_MEMORY, GlError::OutOfMemory),
    ];
}

impl BufferTarget {
    const GL: &[(u32, Self)] = &[
        (Gl::ARRAY_BUFFER, BufferTarget::Array),
        (Gl::ELEMENT_ARRAY_BUFFER, BufferTarget::ElementArray),
        (Gl::UNIFORM_BUFFER, BufferTarget::Uniform),
        (Gl::COPY_READ_BUFFER, BufferTarget::CopyRead),
        (Gl::COPY_WRITE_BUFFER, BufferTarget::CopyWrite),
        (Gl::PIXEL_PACK_BUFFER, BufferTarget::PixelPack),
        (Gl::PIXEL_UNPACK_BUFFER, BufferTarget::PixelUnpack),
        (Gl::TRANSFORM_FEEDBACK_BUFFER, BufferTarget::TransformFeedback),
    ];
}

impl Usage {
    const GL: &[(u32, Self)] = &[
        (Gl::STREAM_DRAW, Usage::StreamDraw),
        (Gl::STREAM_READ, Usage::StreamRead),
        (Gl::STREAM_COPY, Usage::StreamCopy),
        (Gl::STATIC_DRAW, Usage::StaticDraw),
        (Gl::STATIC_READ, Usage::StaticRead),
        (Gl::STATIC_COPY, Usage::StaticCopy),
        (Gl::DYNAMIC_DRAW, Usage::DynamicDraw),
        (Gl::DYNAMIC_READ, Usage::DynamicRead),
        (Gl::DYNAMIC_COPY, Usage::DynamicCopy),
    ];
}

impl ShaderType {
    const GL: &[(u32, Self)] = &[(Gl::VERTEX_SHADER, ShaderType::Vertex), (Gl::FRAGMENT_SHADER, ShaderType::Fragment)];
}

impl TextureTarget {
    const GL: &[(u32, Self)] = &[
        (Gl::TEXTURE_2D, TextureTarget::Texture2D),
        (Gl::TEXTURE_3D, TextureTarget::Texture3D),
        (Gl::TEXTURE_2D_ARRAY, TextureTarget::Texture2DArray),
        (Gl::TEXTURE_CUBE_MAP, TextureTarget::CubeMap),
    ];
}

impl ImageTarget {
    const GL: &[(u32, Self)] = &[
        (Gl::TEXTURE_2D, ImageTarget::Texture2D),
        (Gl::TEXTURE_CUBE_MAP_POSITIVE_X, ImageTarget::CubeFace(0)),
        (Gl::TEXTURE_CUBE_MAP_NEGATIVE_X, ImageTarget::CubeFace(1)),
        (Gl::TEXTURE_CUBE_MAP_POSITIVE_Y, ImageTarget::CubeFace(2)),
        (Gl::TEXTURE_CUBE_MAP_NEGATIVE_Y, ImageTarget::CubeFace(3)),
        (Gl::TEXTURE_CUBE_MAP_POSITIVE_Z, ImageTarget::CubeFace(4)),
        (Gl::TEXTURE_CUBE_MAP_NEGATIVE_Z, ImageTarget::CubeFace(5)),
        (Gl::TEXTURE_3D, ImageTarget::Texture3D),
        (Gl::TEXTURE_2D_ARRAY, ImageTarget::Texture2DArray),
    ];
}

impl InternalFormat {
    const GL: &[(u32, Self)] = &[
        (Gl::R8, InternalFormat::R8),
        (Gl::R8_SNORM, InternalFormat::R8Snorm),
        (Gl::R16F, InternalFormat::R16f),
        (Gl::R32F, InternalFormat::R32f),
        (Gl::R8UI, InternalFormat::R8ui),
        (Gl::R8I, InternalFormat::R8i),
        (Gl::R16UI, InternalFormat::R16ui),
        (Gl::R16I, InternalFormat::R16i),
        (Gl::R32UI, InternalFormat::R32ui),
        (Gl::R32I, InternalFormat::R32i),
        (Gl::RG8, InternalFormat::Rg8),
        (Gl::RG8_SNORM, InternalFormat::Rg8Snorm),
        (Gl::RG16F, InternalFormat::Rg16f),
        (Gl::RG32F, InternalFormat::Rg32f),
        (Gl::RG8UI, InternalFormat::Rg8ui),
        (Gl::RG8I, InternalFormat::Rg8i),
        (Gl::RG16UI, InternalFormat::Rg16ui),
        (Gl::RG16I, InternalFormat::Rg16i),
        (Gl::RG32UI, InternalFormat::Rg32ui),
        (Gl::RG32I, InternalFormat::Rg32i),
        (Gl::RGB8, InternalFormat::Rgb8),
        (Gl::SRGB8, InternalFormat::Srgb8),
        (Gl::RGB565, InternalFormat::Rgb565),
        (Gl::RGB8_SNORM, InternalFormat::Rgb8Snorm),
        (Gl::R11F_G11F_B10F, InternalFormat::R11fG11fB10f),
        (Gl::RGB9_E5, InternalFormat::Rgb9E5),
        (Gl::RGB16F, InternalFormat::Rgb16f),
        (Gl::RGB32F, InternalFormat::Rgb32f),
        (Gl::RGB8UI, InternalFormat::Rgb8ui),
        (Gl::RGB8I, InternalFormat::Rgb8i),
        (Gl::RGB16UI, InternalFormat::Rgb16ui),
        (Gl::RGB16I, InternalFormat::Rgb16i),
        (Gl::RGB32UI, InternalFormat::Rgb32ui),
        (Gl::RGB32I, InternalFormat::Rgb32i),
        (Gl::RGBA8, InternalFormat::Rgba8),
        (Gl::SRGB8_ALPHA8, InternalFormat::Srgb8Alpha8),
        (Gl::RGBA8_SNORM, InternalFormat::Rgba8Snorm),
        (Gl::RGB5_A1, InternalFormat::Rgb5A1),
        (Gl::RGBA4, InternalFormat::Rgba4),
        (Gl::RGB10_A2, InternalFormat::Rgb10A2),
        (Gl::RGBA16F, InternalFormat::Rgba16f),
        (Gl::RGBA32F, InternalFormat::Rgba32f),
        (Gl::RGBA8UI, InternalFormat::Rgba8ui),
        (Gl::RGBA8I, InternalFormat::Rgba8i),
        (Gl::RGB10_A2UI, InternalFormat::Rgb10A2ui),
        (Gl::RGBA16UI, InternalFormat::Rgba16ui),
        (Gl::RGBA16I, InternalFormat::Rgba16i),
        (Gl::RGBA32I, InternalFormat::Rgba32i),
        (Gl::RGBA32UI, InternalFormat::Rgba32ui),
        (Gl::DEPTH_COMPONENT16, InternalFormat::DepthComponent16),
        (Gl::DEPTH_COMPONENT24, InternalFormat::DepthComponent24),
        (Gl::DEPTH_COMPONENT32F, InternalFormat::DepthComponent32f),
        (Gl::DEPTH24_STENCIL8, InternalFormat::Depth24Stencil8),
        (Gl::DEPTH32F_STENCIL8, InternalFormat::Depth32fStencil8),
        (Gl::STENCIL_INDEX8, InternalFormat::StencilIndex8),
        (Gl::RGB, InternalFormat::Rgb),
        (Gl::RGBA, InternalFormat::Rgba),
        (Gl::LUMINANCE_ALPHA, InternalFormat::LuminanceAlpha),
        (Gl::LUMINANCE, InternalFormat::Luminance),
        (Gl::ALPHA, InternalFormat::Alpha),
    ];
}

impl PixelFormat {
    const GL: &[(u32, Self)] = &[
        (Gl::RED, PixelFormat::Red),
        (Gl::RED_INTEGER, PixelFormat::RedInteger),
        (Gl::RG, PixelFormat::Rg),
        (Gl::RG_INTEGER, PixelFormat::RgInteger),
        (Gl::RGB, PixelFormat::Rgb),
        (Gl::RGB_INTEGER, PixelFormat::RgbInteger),
        (Gl::RGBA, PixelFormat::Rgba),
        (Gl::RGBA_INTEGER, PixelFormat::RgbaInteger),
        (Gl::DEPTH_COMPONENT, PixelFormat::DepthComponent),
        (Gl::DEPTH_STENCIL, PixelFormat::DepthStencil),
        (Gl::LUMINANCE_ALPHA, PixelFormat::LuminanceAlpha),
        (Gl::LUMINANCE, PixelFormat::Luminance),
        (Gl::ALPHA, PixelFormat::Alpha),
    ];
}

impl PixelType {
    const GL: &[(u32, Self)] = &[
        (Gl::UNSIGNED_BYTE, PixelType::UnsignedByte),
        (Gl::BYTE, PixelType::Byte),
        (Gl::UNSIGNED_SHORT, PixelType::UnsignedShort),
        (Gl::SHORT, PixelType::Short),
        (Gl::UNSIGNED_INT, PixelType::UnsignedInt),
        (Gl::INT, PixelType::Int),
        (Gl::HALF_FLOAT, PixelType::HalfFloat),
        (Gl::FLOAT, PixelType::Float),
        (Gl::UNSIGNED_SHORT_5_6_5, PixelType::UnsignedShort565),
        (Gl::UNSIGNED_SHORT_4_4_4_4, PixelType::UnsignedShort4444),
        (Gl::UNSIGNED_SHORT_5_5_5_1, PixelType::UnsignedShort5551),
        (Gl::UNSIGNED_INT_2_10_10_10_REV, PixelType::UnsignedInt2101010Rev),
        (Gl::UNSIGNED_INT_10F_11F_11F_REV, PixelType::UnsignedInt10f11f11fRev),
        (Gl::UNSIGNED_INT_5_9_9_9_REV, PixelType::UnsignedInt5999Rev),
        (Gl::UNSIGNED_INT_24_8, PixelType::UnsignedInt248),
        (Gl::FLOAT_32_UNSIGNED_INT_24_8_REV, PixelType::Float32UnsignedInt248Rev),
    ];
}

impl Filter {
    const GL: &[(u32, Self)] = &[
        (Gl::NEAREST, Filter::Nearest),
        (Gl::LINEAR, Filter::Linear),
        (Gl::NEAREST_MIPMAP_NEAREST, Filter::NearestMipmapNearest),
        (Gl::LINEAR_MIPMAP_NEAREST, Filter::LinearMipmapNearest),
        (Gl::NEAREST_MIPMAP_LINEAR, Filter::NearestMipmapLinear),
        (Gl::LINEAR_MIPMAP_LINEAR, Filter::LinearMipmapLinear),
    ];
}

impl Wrap {
    const GL: &[(u32, Self)] = &[(Gl::REPEAT, Wrap::Repeat), (Gl::CLAMP_TO_EDGE, Wrap::ClampToEdge), (Gl::MIRRORED_REPEAT, Wrap::MirroredRepeat)];
}

impl Compare {
    const GL: &[(u32, Self)] = &[
        (Gl::NEVER, Compare::Never),
        (Gl::LESS, Compare::Less),
        (Gl::EQUAL, Compare::Equal),
        (Gl::LEQUAL, Compare::LessEqual),
        (Gl::GREATER, Compare::Greater),
        (Gl::NOTEQUAL, Compare::NotEqual),
        (Gl::GEQUAL, Compare::GreaterEqual),
        (Gl::ALWAYS, Compare::Always),
    ];
}

impl Swizzle {
    const GL: &[(u32, Self)] = &[(Gl::RED, Swizzle::Red), (Gl::GREEN, Swizzle::Green), (Gl::BLUE, Swizzle::Blue), (Gl::ALPHA, Swizzle::Alpha), (Gl::ZERO, Swizzle::Zero), (Gl::ONE, Swizzle::One)];
}

impl FramebufferTarget {
    const GL: &[(u32, Self)] = &[(Gl::FRAMEBUFFER, FramebufferTarget::Both), (Gl::DRAW_FRAMEBUFFER, FramebufferTarget::Draw), (Gl::READ_FRAMEBUFFER, FramebufferTarget::Read)];
}

impl FramebufferStatus {
    const GL: &[(u32, Self)] = &[
        (Gl::FRAMEBUFFER_COMPLETE, FramebufferStatus::Complete),
        (Gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT, FramebufferStatus::IncompleteAttachment),
        (Gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT, FramebufferStatus::IncompleteMissingAttachment),
        (Gl::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE, FramebufferStatus::IncompleteMultisample),
        (Gl::FRAMEBUFFER_UNSUPPORTED, FramebufferStatus::Unsupported),
    ];
}

/// The color attachment points GL ES 3.0 names (COLOR_ATTACHMENT0 ..)
const COLOR_ATTACHMENTS: u32 = 16;

impl AttachPoint {
    pub fn from_gl(v: u32) -> GlResult<AttachPoint> {
        match v {
            Gl::DEPTH_ATTACHMENT => Ok(AttachPoint::Depth),
            Gl::STENCIL_ATTACHMENT => Ok(AttachPoint::Stencil),
            Gl::DEPTH_STENCIL_ATTACHMENT => Ok(AttachPoint::DepthStencil),
            v if (Gl::COLOR_ATTACHMENT0..Gl::COLOR_ATTACHMENT0 + COLOR_ATTACHMENTS).contains(&v) => Ok(AttachPoint::Color(v - Gl::COLOR_ATTACHMENT0)),
            _ => Err(GlError::InvalidEnum),
        }
    }
}

impl ColorBuffer {
    pub fn from_gl(v: u32) -> GlResult<ColorBuffer> {
        match v {
            Gl::NONE => Ok(ColorBuffer::None),
            Gl::BACK => Ok(ColorBuffer::Back),
            v if (Gl::COLOR_ATTACHMENT0..Gl::COLOR_ATTACHMENT0 + COLOR_ATTACHMENTS).contains(&v) => Ok(ColorBuffer::Color(v - Gl::COLOR_ATTACHMENT0)),
            _ => Err(GlError::InvalidEnum),
        }
    }
}
