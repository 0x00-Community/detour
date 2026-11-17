use mlirformat::Error;
use crate::{ColorFormat, ColorTarget, CullMode, Draw, DrawPrimitive, Surface, TargetClear, TargetFill, TargetLoad, VertexBuffer, VertexElement, VertexFormat};

use super::{DEFAULT_UNIFORM_BLOCK, AttribType, Context, Face, GlError, Rect, ShaderType, Storage};
use crate::Device;

/// glDrawArrays' mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawMode {
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

impl TryFrom<DrawMode> for DrawPrimitive {
    type Error = Error;

    fn try_from(m: DrawMode) -> Result<DrawPrimitive, Error> {
        match m {
            DrawMode::Points => Ok(DrawPrimitive::Points),
            DrawMode::Lines => Ok(DrawPrimitive::Lines),
            DrawMode::LineStrip => Ok(DrawPrimitive::LineStrip),
            DrawMode::Triangles => Ok(DrawPrimitive::Triangles),
            DrawMode::TriangleStrip => Ok(DrawPrimitive::TriangleStrip),
            m => Err(Error(format!("{m:?}: not drawn yet"))),
        }
    }
}

/// glClear's buffers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ClearMask {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
}

/// The default framebuffer's color buffer: row 0 is the bottom row (glReadPixels' order). Its pixels live on the GPU
/// once drawn into (in the device's surface); before that, and between a clear and the next draw, as `base` and
/// `clears` say
pub struct DefaultFramebuffer {
    pub width: u32,
    pub height: u32,
    pub format: ColorFormat,
    pub surface: Option<Surface>,
    base: Base,
    /// cleared rectangles over `base`, in order
    clears: Vec<TargetClear>,
}

/// What a default framebuffer holds under its pending clears
#[derive(Clone, Debug)]
enum Base {
    /// these pixels (zero before anything is drawn)
    Pixels(Vec<u8>),
    /// every pixel this texel
    Fill(Vec<u8>),
    /// what its surface holds
    Surface,
}

impl DefaultFramebuffer {
    pub fn new(width: u32, height: u32, format: ColorFormat) -> Self {
        DefaultFramebuffer { width, height, format, surface: None, base: Base::Pixels(vec![0; (width * height * format.bytes()) as usize]), clears: Vec::new() }
    }

    /// Its bytes
    pub fn len(&self) -> usize {
        (self.width * self.height * self.format.bytes()) as usize
    }

    /// One pixel of `rgba` as the format stores it
    fn texel(&self, rgba: [f32; 4]) -> Vec<u8> {
        match self.format {
            ColorFormat::Rgba32Float => rgba.iter().flat_map(|c| c.to_le_bytes()).collect(),
            ColorFormat::Rgba8Unorm => rgba.iter().map(|c| (c * f32::from(u8::MAX)).round() as u8).collect(),
        }
    }

    /// The pixels `r` covers, clipped to the framebuffer: x0, y0, x1, y1
    fn clip(&self, r: Rect) -> [u32; 4] {
        let end = |at: i32, len: u32, max: u32| ((i64::from(at) + i64::from(len)).max(0) as u32).min(max);
        [r.x.max(0) as u32, r.y.max(0) as u32, end(r.x, r.width, self.width), end(r.y, r.height, self.height)]
    }

    /// What the next draw loads the target with
    fn load(&self) -> TargetLoad<'_> {
        match &self.base {
            Base::Pixels(p) => TargetLoad::Pixels(p, &self.clears),
            Base::Fill(t) => TargetLoad::Fill(t, &self.clears),
            Base::Surface => TargetLoad::Keep(&self.clears),
        }
    }

    /// Its pixels (glReadPixels' order), read from the device when they are on the GPU
    pub fn pixels(&self, device: &Device<'_>) -> Result<Vec<u8>, Error> {
        let mut pixels = match (&self.base, self.surface) {
            (Base::Pixels(p), _) => p.clone(),
            (Base::Fill(t), _) => {
                let mut p = vec![0; self.len()];
                TargetFill::apply(&mut p, t);
                p
            }
            (Base::Surface, Some(s)) => device.read(s, self.len())?,
            (Base::Surface, None) => return Err(Error("a framebuffer on the GPU without a surface".into())),
        };
        self.clears.iter().for_each(|c| c.apply(&mut pixels, self.width));
        Ok(pixels)
    }
}

impl Context<'_> {
    /// glClear: the color buffer within the scissor (when on), with the color mask
    pub fn clear(&mut self, mask: ClearMask) {
        if !mask.color {
            return;
        }
        let fb = &mut self.framebuffer;
        let texel = fb.texel(self.state.clear_color);
        let full = Rect { x: 0, y: 0, width: fb.width, height: fb.height };
        let rect = fb.clip(if self.state.scissor_test { self.state.scissor } else { full });
        let channels = self.state.color_mask;
        if rect[0] >= rect[2] || rect[1] >= rect[3] || !channels.iter().any(|c| *c) {
            return;
        }
        if rect == [0, 0, fb.width, fb.height] && channels.iter().all(|c| *c) {
            fb.base = Base::Fill(texel);
            fb.clears.clear();
            return;
        }
        fb.clears.push(TargetClear { rect, channels, texel });
    }

    /// glDrawArrays into the default framebuffer on `device`. A GL error is recorded (nothing drawn); an error
    /// returned is what detour cannot draw yet, or the device's
    pub fn draw_arrays(&mut self, device: &Device<'_>, mode: DrawMode, first: i32, count: i32) -> Result<(), Error> {
        let (Ok(first), Ok(count)) = (u32::try_from(first), u32::try_from(count)) else {
            self.record_error(GlError::InvalidValue);
            return Ok(());
        };
        let program = self.program;
        if self.objects_program(program).is_none() {
            self.record_error(GlError::InvalidOperation);
            return Ok(());
        }
        if count == 0 || self.state.rasterizer_discard {
            return Ok(());
        }
        let cull = match (self.state.cull, self.state.cull_face) {
            (false, _) => CullMode::None,
            (true, Face::Back) => CullMode::Back,
            (true, Face::Front) => CullMode::Front,
            (true, Face::FrontAndBack) if matches!(mode, DrawMode::Triangles | DrawMode::TriangleStrip | DrawMode::TriangleFan) => return Ok(()),
            (true, Face::FrontAndBack) => CullMode::None,
        };
        let vao = self.vertex_array();
        let Some(linked) = self.objects_program(program) else { return Ok(()) };
        let mut inputs: Vec<_> = linked.variables.iter().filter(|v| v.stage == ShaderType::Vertex && v.storage == Storage::Input && !v.builtin).collect();
        inputs.sort_by_key(|v| v.location);
        let mut buffers = Vec::new();
        let mut elements = Vec::new();
        for v in &inputs {
            let location = v.location.ok_or_else(|| Error(format!("attribute {} without a location", v.name)))?;
            let a = vao.attribs[location as usize];
            if !a.enabled || a.ty != AttribType::Float || a.normalized || a.integer {
                return Err(Error(format!("attribute {}: only enabled float arrays are drawn yet", v.name)));
            }
            let data = &self.buffers.get(a.buffer).ok_or_else(|| Error(format!("attribute {}: client-side arrays are not drawn", v.name)))?.data;
            buffers.push(VertexBuffer { bytes: data.get(a.offset..).unwrap_or_default(), stride: a.step() });
            elements.push(VertexElement { buffer: elements.len() as u32, format: VertexFormat::try_from(a.size)?, offset: 0 });
        }
        let (s, fb) = (&self.state, &self.framebuffer);
        let draw = Draw {
            buffers,
            elements,
            uniforms: if linked.block.is_empty() { Vec::new() } else { vec![(DEFAULT_UNIFORM_BLOCK, linked.block.as_slice())] },
            primitive: mode.try_into()?,
            first,
            count,
            instances: 1,
            viewport: [s.viewport.x as f32, s.viewport.y as f32, s.viewport.width as f32, s.viewport.height as f32],
            depth_range: s.depth_range,
            scissor: s.scissor_test.then(|| {
                let [x0, y0, x1, y1] = fb.clip(s.scissor);
                [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
            }),
            cull,
            front_ccw: s.front_ccw,
            target: ColorTarget { width: fb.width, height: fb.height, format: fb.format, load: fb.load() },
        };
        let surface = match fb.surface {
            Some(s) => s,
            None => device.surface(fb.len())?,
        };
        device.draw(&linked.compiled, &draw, surface)?;
        let fb = &mut self.framebuffer;
        fb.surface = Some(surface);
        fb.base = Base::Surface;
        fb.clears.clear();
        Ok(())
    }
}
