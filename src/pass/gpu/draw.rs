use alloc::vec::Vec;

/// What a draw assembles its vertices into
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawPrimitive {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
}

/// Which faces a draw culls
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

/// A blend factor (glBlendFunc's)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendEquation {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// How a draw blends into its target: RGB's and alpha's factors and equations, and the constant color
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blend {
    pub src: [BlendFactor; 2],
    pub dst: [BlendFactor; 2],
    pub equation: [BlendEquation; 2],
    pub color: [f32; 4],
}

impl Default for Blend {
    fn default() -> Self {
        Blend { src: [BlendFactor::One; 2], dst: [BlendFactor::Zero; 2], equation: [BlendEquation::Add; 2], color: [0.0; 4] }
    }
}

/// A vertex attribute's format in its buffer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    /// four bytes, each 0..255 as 0.0..1.0 (a color)
    Unorm8x4,
}

impl VertexFormat {
    pub const fn components(self) -> u32 {
        match self {
            VertexFormat::Float32 => 1,
            VertexFormat::Float32x2 => 2,
            VertexFormat::Float32x3 => 3,
            VertexFormat::Float32x4 | VertexFormat::Unorm8x4 => 4,
        }
    }
}

impl TryFrom<u32> for VertexFormat {
    type Error = mlirformat::Error;

    /// A vertex attribute of `n` 32-bit floats
    fn try_from(n: u32) -> Result<VertexFormat, mlirformat::Error> {
        match n {
            1 => Ok(VertexFormat::Float32),
            2 => Ok(VertexFormat::Float32x2),
            3 => Ok(VertexFormat::Float32x3),
            4 => Ok(VertexFormat::Float32x4),
            n => Err(mlirformat::Error(alloc::format!("a vertex attribute of {n} floats"))),
        }
    }
}

/// A color target's texel format (sampled as it is: no sRGB decoding)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorFormat {
    Rgba32Float,
    #[default]
    Rgba8Unorm,
    /// one channel, red; sampled as (r, 0, 0, 1)
    R8Unorm,
}

impl ColorFormat {
    pub const fn bytes(self) -> u32 {
        match self {
            ColorFormat::Rgba32Float => 16,
            ColorFormat::Rgba8Unorm => 4,
            ColorFormat::R8Unorm => 1,
        }
    }
}

/// An index buffer's index size
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    U16,
    U32,
}

/// A draw's indices (little-endian, `format`): its `first` and `count` count indices, each the vertex it takes
#[derive(Clone, Copy, Debug)]
pub struct IndexBuffer<'d> {
    pub bytes: &'d [u8],
    pub format: IndexFormat,
}

/// A vertex shader input, in location order: which vertex buffer, its format, where in a vertex
#[derive(Clone, Copy, Debug)]
pub struct VertexElement {
    pub buffer: u32,
    pub format: VertexFormat,
    pub offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexBuffer<'d> {
    pub bytes: &'d [u8],
    pub stride: u32,
}

/// A color target: its size, format and layout in memory, and what it holds before the draw (row 0 is the bottom
/// row of the viewport's y axis; the pixels and clears in `load` are linear)
#[derive(Clone, Debug)]
pub struct ColorTarget<'d> {
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub tiling: crate::Tiling,
    pub load: TargetLoad<'d>,
}

impl Default for TargetLoad<'_> {
    fn default() -> Self {
        TargetLoad::Keep(&[])
    }
}

impl Default for ColorTarget<'_> {
    fn default() -> Self {
        ColorTarget { width: 0, height: 0, format: ColorFormat::default(), tiling: crate::Tiling::Linear, load: TargetLoad::default() }
    }
}

impl ColorTarget<'_> {
    /// A row's bytes of pixels
    pub const fn row(&self) -> u32 {
        self.width * self.format.bytes()
    }

    /// A row's bytes in memory
    pub const fn pitch(&self) -> u32 {
        self.tiling.pitch(self.row())
    }

    /// Its pixels' bytes
    pub const fn len(&self) -> usize {
        (self.row() * self.height) as usize
    }

    /// Whether it has no pixels
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes it takes in memory
    pub const fn size(&self) -> usize {
        self.tiling.size(self.row(), self.height)
    }
}

/// What a color target holds before a draw: what it held after the last one, every pixel one texel, or these
/// pixels; then cleared rectangles over it
#[derive(Clone, Debug)]
pub enum TargetLoad<'d> {
    Keep(&'d [TargetClear]),
    Fill(&'d [u8], &'d [TargetClear]),
    Pixels(&'d [u8], &'d [TargetClear]),
}

/// A rectangle of a target set to one texel, some of its channels (glClear within the scissor, with a color mask)
#[derive(Clone, Debug, PartialEq)]
pub struct TargetClear {
    /// x0, y0, x1, y1 (exclusive), within the target
    pub rect: [u32; 4],
    pub channels: [bool; 4],
    pub texel: Vec<u8>,
}

impl TargetClear {
    /// A channel's bytes: a quarter of the texel's, or all of a one-channel texel's
    fn channel_bytes(&self) -> usize {
        (self.texel.len() / self.channels.len()).max(1)
    }

    /// The clear applied to `memory`, a target `width` pixels wide laid out as `tiling`
    pub fn apply_laid_out(&self, memory: &mut [u8], width: u32, tiling: crate::Tiling) {
        self.apply_pitched(memory, tiling.pitch(width * self.texel.len() as u32), tiling);
    }

    /// The clear applied to `memory`, laid out as `tiling` with rows `pitch` bytes apart
    pub fn apply_pitched(&self, memory: &mut [u8], pitch: u32, tiling: crate::Tiling) {
        let [x0, y0, x1, y1] = self.rect;
        let size = self.texel.len();
        let per = self.channel_bytes();
        for y in y0..y1 {
            for x in x0..x1 {
                for (c, _) in self.channels.iter().enumerate().take(size / per).filter(|(_, on)| **on) {
                    let at = tiling.byte(x * size as u32 + (c * per) as u32, y, pitch);
                    memory[at..at + per].copy_from_slice(&self.texel[c * per..(c + 1) * per]);
                }
            }
        }
    }

    /// The clear applied to `pixels`, a target `width` pixels wide
    pub fn apply(&self, pixels: &mut [u8], width: u32) {
        let [x0, y0, x1, y1] = self.rect;
        let size = self.texel.len();
        let per = self.channel_bytes();
        for y in y0..y1 {
            let row = &mut pixels[(y * width) as usize * size..][(x0 as usize * size)..(x1 as usize * size)];
            for px in row.chunks_exact_mut(size) {
                for (c, _) in self.channels.iter().enumerate().take(size / per).filter(|(_, on)| **on) {
                    px[c * per..(c + 1) * per].copy_from_slice(&self.texel[c * per..(c + 1) * per]);
                }
            }
        }
    }
}

/// Every pixel of `pixels` set to `texel`
pub struct TargetFill;

impl TargetFill {
    pub fn apply(pixels: &mut [u8], texel: &[u8]) {
        match texel.iter().all(|b| *b == texel[0]) {
            true => pixels.fill(texel[0]),
            false => pixels.chunks_exact_mut(texel.len()).for_each(|p| p.copy_from_slice(texel)),
        }
    }
}

/// How a sampled texture filters between texels
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    #[default]
    Nearest,
    Linear,
}

/// What a sampled texture gives for coordinates outside [0, 1]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Wrap {
    #[default]
    ClampToEdge,
    Repeat,
    MirroredRepeat,
}

/// A surface the fragment shader samples, at its sampled image `binding` (one mip level, its own size and format)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampled {
    pub binding: u32,
    pub surface: crate::Surface,
    pub filter: Filter,
    pub wrap: Wrap,
}

/// One draw, described without the GPU: a target's constructor (detour's `Xe3Draw`, say) lowers it, with the program's kernels,
/// into the image the GPU runs
#[derive(Clone, Debug)]
pub struct Draw<'d> {
    pub buffers: Vec<VertexBuffer<'d>>,
    pub elements: Vec<VertexElement>,
    /// uniform blocks, by binding
    pub uniforms: Vec<(u32, &'d [u8])>,
    /// surfaces the fragment shader samples
    pub textures: Vec<Sampled>,
    pub primitive: DrawPrimitive,
    /// None: the vertices in order from `first`; else `first` and `count` count the indices
    pub indices: Option<IndexBuffer<'d>>,
    pub first: u32,
    pub count: u32,
    pub instances: u32,
    /// x, y, width, height in pixels
    pub viewport: [f32; 4],
    pub depth_range: [f32; 2],
    /// x, y, width, height; None: the whole target
    pub scissor: Option<[u32; 4]>,
    pub cull: CullMode,
    pub front_ccw: bool,
    /// None: the fragments replace the target's pixels
    pub blend: Option<Blend>,
    /// red, green, blue, alpha: written or kept
    pub color_mask: [bool; 4],
    pub target: ColorTarget<'d>,
}

impl Default for Draw<'_> {
    /// No vertices, uniforms or textures; one instance of a triangle list, the whole target, depth 0 to 1, no culling,
    /// no blending, every channel written
    fn default() -> Self {
        Draw {
            buffers: Vec::new(),
            elements: Vec::new(),
            uniforms: Vec::new(),
            textures: Vec::new(),
            primitive: DrawPrimitive::Triangles,
            indices: None,
            first: 0,
            count: 0,
            instances: 1,
            viewport: [0.0; 4],
            depth_range: [0.0, 1.0],
            scissor: None,
            cull: CullMode::None,
            front_ccw: true,
            blend: None,
            color_mask: [true; 4],
            target: ColorTarget::default(),
        }
    }
}
