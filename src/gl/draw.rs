use mlirformat::Error;

use super::{AttribType, Context, Face, GlError, Storage, ShaderType};
use crate::{Cull, Device, Draw, Primitive, SurfaceFormat, Target, VertexBuffer, VertexElement};

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

impl TryFrom<DrawMode> for Primitive {
    type Error = Error;

    fn try_from(m: DrawMode) -> Result<Primitive, Error> {
        match m {
            DrawMode::Points => Ok(Primitive::Points),
            DrawMode::Lines => Ok(Primitive::Lines),
            DrawMode::LineStrip => Ok(Primitive::LineStrip),
            DrawMode::Triangles => Ok(Primitive::Triangles),
            DrawMode::TriangleStrip => Ok(Primitive::TriangleStrip),
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

/// The default framebuffer's color buffer: row 0 is the bottom row (glReadPixels' order)
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub format: SurfaceFormat,
    pub pixels: Vec<u8>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32, format: SurfaceFormat) -> Self {
        Framebuffer { width, height, format, pixels: vec![0; (width * height * format.bytes()) as usize] }
    }

    /// One pixel of `rgba` as the format stores it
    fn texel(&self, rgba: [f32; 4]) -> Result<Vec<u8>, Error> {
        match self.format {
            SurfaceFormat::R32G32B32A32Float => Ok(rgba.iter().flat_map(|c| c.to_le_bytes()).collect()),
            SurfaceFormat::R8G8B8A8Unorm => Ok(rgba.iter().map(|c| (c * f32::from(u8::MAX)).round() as u8).collect()),
            f => Err(Error(format!("a color buffer of {f:?}"))),
        }
    }
}

impl Context<'_> {
    /// glClear: the color buffer within the scissor (when on), with the color mask
    pub fn clear(&mut self, mask: ClearMask) -> Result<(), Error> {
        if !mask.color {
            return Ok(());
        }
        let fb = &mut self.framebuffer;
        let texel = fb.texel(self.state.clear_color)?;
        let full = super::Rect { x: 0, y: 0, width: fb.width, height: fb.height };
        let r = if self.state.scissor_test { self.state.scissor } else { full };
        let (x0, y0) = (r.x.max(0) as u32, r.y.max(0) as u32);
        let (x1, y1) = (((r.x as i64 + r.width as i64).max(0) as u32).min(fb.width), ((r.y as i64 + r.height as i64).max(0) as u32).min(fb.height));
        let per = texel.len() / 4;
        for y in y0..y1 {
            for x in x0..x1 {
                let at = ((y * fb.width + x) as usize) * texel.len();
                for (c, on) in self.state.color_mask.iter().enumerate() {
                    if *on {
                        fb.pixels[at + c * per..at + (c + 1) * per].copy_from_slice(&texel[c * per..(c + 1) * per]);
                    }
                }
            }
        }
        Ok(())
    }

    /// glDrawArrays into the default framebuffer on `device`. A GL error is recorded (nothing drawn); an error
    /// returned is what detour cannot draw yet, or the device's
    pub fn draw_arrays(&mut self, device: &Device, mode: DrawMode, first: i32, count: i32) -> Result<(), Error> {
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
            (false, _) => Cull::None,
            (true, Face::Back) => Cull::Back,
            (true, Face::Front) => Cull::Front,
            (true, Face::FrontAndBack) if matches!(mode, DrawMode::Triangles | DrawMode::TriangleStrip | DrawMode::TriangleFan) => return Ok(()),
            (true, Face::FrontAndBack) => Cull::None,
        };
        let vao = self.vertex_array().clone();
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
            elements.push(VertexElement { buffer: elements.len() as u32, format: SurfaceFormat::floats(a.size)?, offset: 0 });
        }
        let s = &self.state;
        let draw = Draw {
            program: &linked.xe3,
            buffers,
            elements,
            uniforms: Vec::new(),
            topology: mode.try_into()?,
            first,
            count,
            instances: 1,
            viewport: [s.viewport.x as f32, s.viewport.y as f32, s.viewport.width as f32, s.viewport.height as f32],
            depth_range: s.depth_range,
            scissor: s.scissor_test.then_some([s.scissor.x.max(0) as u32, s.scissor.y.max(0) as u32, s.scissor.width, s.scissor.height]),
            cull,
            front_ccw: s.front_ccw,
        };
        let fb = &self.framebuffer;
        let mut pixels = fb.pixels.clone();
        let mut target = Target { width: fb.width, height: fb.height, format: fb.format, pixels: &mut pixels };
        device.draw(&draw, &mut target)?;
        self.framebuffer.pixels = pixels;
        Ok(())
    }
}
