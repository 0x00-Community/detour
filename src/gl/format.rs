/// A texture's or renderbuffer's internal format (GL ES 3.0 tables 3.13 and 3.14, the unsized ones of table 3.3)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum InternalFormat {
    R8,
    R8Snorm,
    R16f,
    R32f,
    R8ui,
    R8i,
    R16ui,
    R16i,
    R32ui,
    R32i,
    Rg8,
    Rg8Snorm,
    Rg16f,
    Rg32f,
    Rg8ui,
    Rg8i,
    Rg16ui,
    Rg16i,
    Rg32ui,
    Rg32i,
    Rgb8,
    Srgb8,
    Rgb565,
    Rgb8Snorm,
    R11fG11fB10f,
    Rgb9E5,
    Rgb16f,
    Rgb32f,
    Rgb8ui,
    Rgb8i,
    Rgb16ui,
    Rgb16i,
    Rgb32ui,
    Rgb32i,
    Rgba8,
    Srgb8Alpha8,
    Rgba8Snorm,
    Rgb5A1,
    Rgba4,
    Rgb10A2,
    Rgba16f,
    Rgba32f,
    Rgba8ui,
    Rgba8i,
    Rgb10A2ui,
    Rgba16ui,
    Rgba16i,
    Rgba32i,
    Rgba32ui,
    DepthComponent16,
    DepthComponent24,
    DepthComponent32f,
    Depth24Stencil8,
    Depth32fStencil8,
    /// renderbuffers only
    StencilIndex8,
    /// unsized: the format of the data is the format of the texture
    Rgb,
    Rgba,
    LuminanceAlpha,
    Luminance,
    Alpha,
}

/// glTexImage*'s `format`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Red,
    RedInteger,
    Rg,
    RgInteger,
    Rgb,
    RgbInteger,
    Rgba,
    RgbaInteger,
    DepthComponent,
    DepthStencil,
    LuminanceAlpha,
    Luminance,
    Alpha,
}

impl PixelFormat {
    pub const fn components(self) -> u32 {
        match self {
            PixelFormat::Red | PixelFormat::RedInteger | PixelFormat::DepthComponent | PixelFormat::Luminance | PixelFormat::Alpha => 1,
            PixelFormat::Rg | PixelFormat::RgInteger | PixelFormat::DepthStencil | PixelFormat::LuminanceAlpha => 2,
            PixelFormat::Rgb | PixelFormat::RgbInteger => 3,
            PixelFormat::Rgba | PixelFormat::RgbaInteger => 4,
        }
    }
}

/// glTexImage*'s `type`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelType {
    UnsignedByte,
    Byte,
    UnsignedShort,
    Short,
    UnsignedInt,
    Int,
    HalfFloat,
    Float,
    UnsignedShort565,
    UnsignedShort4444,
    UnsignedShort5551,
    UnsignedInt2101010Rev,
    UnsignedInt10f11f11fRev,
    UnsignedInt5999Rev,
    UnsignedInt248,
    Float32UnsignedInt248Rev,
}

impl PixelType {
    /// The bytes of one pixel of `format` in this type (a packed type holds the whole pixel)
    pub const fn pixel(self, format: PixelFormat) -> u32 {
        match self {
            PixelType::UnsignedByte | PixelType::Byte => format.components(),
            PixelType::UnsignedShort | PixelType::Short | PixelType::HalfFloat => 2 * format.components(),
            PixelType::UnsignedInt | PixelType::Int | PixelType::Float => 4 * format.components(),
            PixelType::UnsignedShort565 | PixelType::UnsignedShort4444 | PixelType::UnsignedShort5551 => 2,
            PixelType::UnsignedInt2101010Rev | PixelType::UnsignedInt10f11f11fRev | PixelType::UnsignedInt5999Rev | PixelType::UnsignedInt248 => 4,
            PixelType::Float32UnsignedInt248Rev => 8,
        }
    }

    /// The bytes of the type's element: what GL_UNPACK_ALIGNMENT is measured against
    pub const fn element(self) -> u32 {
        match self {
            PixelType::UnsignedByte | PixelType::Byte => 1,
            PixelType::UnsignedShort | PixelType::Short | PixelType::HalfFloat | PixelType::UnsignedShort565 | PixelType::UnsignedShort4444 | PixelType::UnsignedShort5551 => 2,
            PixelType::Float32UnsignedInt248Rev => 8,
            _ => 4,
        }
    }
}

/// What a format can be used for (GL ES 3.0 table 3.13 with EXT_color_buffer_float, which Mesa exposes on ES 3)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Renderable {
    Color,
    Depth,
    Stencil,
    DepthStencil,
    No,
}

use InternalFormat as I;
use PixelFormat as F;
use PixelType as T;

impl InternalFormat {
    /// GL ES 3.0 table 3.2: every internal format with the client formats and types it is specified from. The first
    /// row of a format is how detour stores it.
    const TABLE: &[(InternalFormat, PixelFormat, PixelType)] = &[
        (I::Rgba8, F::Rgba, T::UnsignedByte),
        (I::Rgb5A1, F::Rgba, T::UnsignedByte),
        (I::Rgb5A1, F::Rgba, T::UnsignedShort5551),
        (I::Rgb5A1, F::Rgba, T::UnsignedInt2101010Rev),
        (I::Rgba4, F::Rgba, T::UnsignedByte),
        (I::Rgba4, F::Rgba, T::UnsignedShort4444),
        (I::Srgb8Alpha8, F::Rgba, T::UnsignedByte),
        (I::Rgba8Snorm, F::Rgba, T::Byte),
        (I::Rgb10A2, F::Rgba, T::UnsignedInt2101010Rev),
        (I::Rgba16f, F::Rgba, T::HalfFloat),
        (I::Rgba16f, F::Rgba, T::Float),
        (I::Rgba32f, F::Rgba, T::Float),
        (I::Rgba8ui, F::RgbaInteger, T::UnsignedByte),
        (I::Rgba8i, F::RgbaInteger, T::Byte),
        (I::Rgb10A2ui, F::RgbaInteger, T::UnsignedInt2101010Rev),
        (I::Rgba16ui, F::RgbaInteger, T::UnsignedShort),
        (I::Rgba16i, F::RgbaInteger, T::Short),
        (I::Rgba32i, F::RgbaInteger, T::Int),
        (I::Rgba32ui, F::RgbaInteger, T::UnsignedInt),
        (I::Rgb8, F::Rgb, T::UnsignedByte),
        (I::Rgb565, F::Rgb, T::UnsignedByte),
        (I::Rgb565, F::Rgb, T::UnsignedShort565),
        (I::Srgb8, F::Rgb, T::UnsignedByte),
        (I::Rgb8Snorm, F::Rgb, T::Byte),
        (I::R11fG11fB10f, F::Rgb, T::UnsignedInt10f11f11fRev),
        (I::R11fG11fB10f, F::Rgb, T::HalfFloat),
        (I::R11fG11fB10f, F::Rgb, T::Float),
        (I::Rgb9E5, F::Rgb, T::UnsignedInt5999Rev),
        (I::Rgb9E5, F::Rgb, T::HalfFloat),
        (I::Rgb9E5, F::Rgb, T::Float),
        (I::Rgb16f, F::Rgb, T::HalfFloat),
        (I::Rgb16f, F::Rgb, T::Float),
        (I::Rgb32f, F::Rgb, T::Float),
        (I::Rgb8ui, F::RgbInteger, T::UnsignedByte),
        (I::Rgb8i, F::RgbInteger, T::Byte),
        (I::Rgb16ui, F::RgbInteger, T::UnsignedShort),
        (I::Rgb16i, F::RgbInteger, T::Short),
        (I::Rgb32ui, F::RgbInteger, T::UnsignedInt),
        (I::Rgb32i, F::RgbInteger, T::Int),
        (I::Rg8, F::Rg, T::UnsignedByte),
        (I::Rg8Snorm, F::Rg, T::Byte),
        (I::Rg16f, F::Rg, T::HalfFloat),
        (I::Rg16f, F::Rg, T::Float),
        (I::Rg32f, F::Rg, T::Float),
        (I::Rg8ui, F::RgInteger, T::UnsignedByte),
        (I::Rg8i, F::RgInteger, T::Byte),
        (I::Rg16ui, F::RgInteger, T::UnsignedShort),
        (I::Rg16i, F::RgInteger, T::Short),
        (I::Rg32ui, F::RgInteger, T::UnsignedInt),
        (I::Rg32i, F::RgInteger, T::Int),
        (I::R8, F::Red, T::UnsignedByte),
        (I::R8Snorm, F::Red, T::Byte),
        (I::R16f, F::Red, T::HalfFloat),
        (I::R16f, F::Red, T::Float),
        (I::R32f, F::Red, T::Float),
        (I::R8ui, F::RedInteger, T::UnsignedByte),
        (I::R8i, F::RedInteger, T::Byte),
        (I::R16ui, F::RedInteger, T::UnsignedShort),
        (I::R16i, F::RedInteger, T::Short),
        (I::R32ui, F::RedInteger, T::UnsignedInt),
        (I::R32i, F::RedInteger, T::Int),
        (I::DepthComponent16, F::DepthComponent, T::UnsignedShort),
        (I::DepthComponent16, F::DepthComponent, T::UnsignedInt),
        (I::DepthComponent24, F::DepthComponent, T::UnsignedInt),
        (I::DepthComponent32f, F::DepthComponent, T::Float),
        (I::Depth24Stencil8, F::DepthStencil, T::UnsignedInt248),
        (I::Depth32fStencil8, F::DepthStencil, T::Float32UnsignedInt248Rev),
        (I::Rgba, F::Rgba, T::UnsignedByte),
        (I::Rgba, F::Rgba, T::UnsignedShort4444),
        (I::Rgba, F::Rgba, T::UnsignedShort5551),
        (I::Rgb, F::Rgb, T::UnsignedByte),
        (I::Rgb, F::Rgb, T::UnsignedShort565),
        (I::LuminanceAlpha, F::LuminanceAlpha, T::UnsignedByte),
        (I::Luminance, F::Luminance, T::UnsignedByte),
        (I::Alpha, F::Alpha, T::UnsignedByte),
    ];

    /// Table 3.2 has this combination
    pub fn takes(self, format: PixelFormat, ty: PixelType) -> bool {
        InternalFormat::TABLE.iter().any(|&(i, f, t)| i == self && f == format && t == ty)
    }

    /// The client format and type detour stores the texels in
    pub fn stored(self) -> Option<(PixelFormat, PixelType)> {
        InternalFormat::TABLE.iter().find(|(i, _, _)| *i == self).map(|&(_, f, t)| (f, t))
    }

    /// The bytes of one stored texel
    pub fn bytes(self) -> u32 {
        self.stored().map_or(1, |(f, t)| t.pixel(f))
    }

    /// Unsized (RGBA, RGB, LUMINANCE …): glTexStorage* and renderbuffers take only sized formats
    pub const fn sized(self) -> bool {
        !matches!(self, I::Rgb | I::Rgba | I::LuminanceAlpha | I::Luminance | I::Alpha)
    }

    /// Any client format and type table 3.2 lists for an unsized internal format names that format
    pub fn unsized_of(format: PixelFormat) -> Option<InternalFormat> {
        match format {
            F::Rgba => Some(I::Rgba),
            F::Rgb => Some(I::Rgb),
            F::LuminanceAlpha => Some(I::LuminanceAlpha),
            F::Luminance => Some(I::Luminance),
            F::Alpha => Some(I::Alpha),
            _ => None,
        }
    }

    pub const fn renderable(self) -> Renderable {
        match self {
            I::DepthComponent16 | I::DepthComponent24 | I::DepthComponent32f => Renderable::Depth,
            I::Depth24Stencil8 | I::Depth32fStencil8 => Renderable::DepthStencil,
            I::StencilIndex8 => Renderable::Stencil,
            I::R8 | I::Rg8 | I::Rgb8 | I::Rgb565 | I::Rgba4 | I::Rgb5A1 | I::Rgba8 | I::Rgb10A2 | I::Rgb10A2ui | I::Srgb8Alpha8 => Renderable::Color,
            I::R8ui | I::R8i | I::R16ui | I::R16i | I::R32ui | I::R32i => Renderable::Color,
            I::Rg8ui | I::Rg8i | I::Rg16ui | I::Rg16i | I::Rg32ui | I::Rg32i => Renderable::Color,
            I::Rgba8ui | I::Rgba8i | I::Rgba16ui | I::Rgba16i | I::Rgba32i | I::Rgba32ui => Renderable::Color,
            I::R16f | I::Rg16f | I::Rgba16f | I::R32f | I::Rg32f | I::Rgba32f | I::R11fG11fB10f => Renderable::Color,
            I::Rgb | I::Rgba => Renderable::Color,
            _ => Renderable::No,
        }
    }

    /// Signed or unsigned integer: never filtered, never multisampled in a renderbuffer
    pub const fn integer(self) -> bool {
        matches!(
            self,
            I::R8ui | I::R8i | I::R16ui | I::R16i | I::R32ui | I::R32i | I::Rg8ui | I::Rg8i | I::Rg16ui | I::Rg16i | I::Rg32ui | I::Rg32i | I::Rgb8ui | I::Rgb8i | I::Rgb16ui | I::Rgb16i | I::Rgb32ui | I::Rgb32i | I::Rgba8ui | I::Rgba8i | I::Rgb10A2ui | I::Rgba16ui | I::Rgba16i | I::Rgba32i | I::Rgba32ui
        )
    }
}
