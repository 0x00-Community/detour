use crate::Blend;

/// glEnable / glDisable capabilities (GL ES 3.0)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    Blend,
    CullFace,
    DepthTest,
    Dither,
    PolygonOffsetFill,
    PrimitiveRestartFixedIndex,
    RasterizerDiscard,
    SampleAlphaToCoverage,
    SampleCoverage,
    ScissorTest,
    StencilTest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Depth and stencil comparisons
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Compare {
    Never,
    #[default]
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Face {
    Front,
    #[default]
    Back,
    FrontAndBack,
}

/// The fixed-function state a draw packs into the GPU's state commands
#[derive(Clone, Debug)]
pub struct State {
    pub blend: bool,
    pub cull: bool,
    pub depth_test: bool,
    pub dither: bool,
    pub polygon_offset: bool,
    pub primitive_restart: bool,
    pub rasterizer_discard: bool,
    pub alpha_to_coverage: bool,
    pub sample_coverage: bool,
    pub scissor_test: bool,
    pub stencil_test: bool,
    pub viewport: Rect,
    pub scissor: Rect,
    pub depth_range: [f32; 2],
    pub clear_color: [f32; 4],
    pub clear_depth: f32,
    pub clear_stencil: i32,
    pub color_mask: [bool; 4],
    pub depth_mask: bool,
    pub depth_func: Compare,
    pub cull_face: Face,
    /// glFrontFace: counter-clockwise is front (the default)
    pub front_ccw: bool,
    pub blending: Blend,
    pub line_width: f32,
    pub polygon_offset_factor: [f32; 2],
}

impl State {
    /// GL ES 3.0's initial state for a default framebuffer of `width` × `height`
    pub fn new(width: u32, height: u32) -> Self {
        let full = Rect { x: 0, y: 0, width, height };
        State {
            blend: false,
            cull: false,
            depth_test: false,
            dither: true,
            polygon_offset: false,
            primitive_restart: false,
            rasterizer_discard: false,
            alpha_to_coverage: false,
            sample_coverage: false,
            scissor_test: false,
            stencil_test: false,
            viewport: full,
            scissor: full,
            depth_range: [0.0, 1.0],
            clear_color: [0.0; 4],
            clear_depth: 1.0,
            clear_stencil: 0,
            color_mask: [true; 4],
            depth_mask: true,
            depth_func: Compare::Less,
            cull_face: Face::Back,
            front_ccw: true,
            blending: Blend::default(),
            line_width: 1.0,
            polygon_offset_factor: [0.0; 2],
        }
    }

    pub fn flag(&mut self, cap: Capability) -> &mut bool {
        match cap {
            Capability::Blend => &mut self.blend,
            Capability::CullFace => &mut self.cull,
            Capability::DepthTest => &mut self.depth_test,
            Capability::Dither => &mut self.dither,
            Capability::PolygonOffsetFill => &mut self.polygon_offset,
            Capability::PrimitiveRestartFixedIndex => &mut self.primitive_restart,
            Capability::RasterizerDiscard => &mut self.rasterizer_discard,
            Capability::SampleAlphaToCoverage => &mut self.alpha_to_coverage,
            Capability::SampleCoverage => &mut self.sample_coverage,
            Capability::ScissorTest => &mut self.scissor_test,
            Capability::StencilTest => &mut self.stencil_test,
        }
    }
}
