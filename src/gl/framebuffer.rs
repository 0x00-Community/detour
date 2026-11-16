use std::collections::BTreeMap;

use super::{GlError, GlResult, InternalFormat, Names, Renderable, Texture, TextureTarget};

/// Renderbuffer and draw buffer limits (the values Mesa's iris reports on Xe3)
pub const MAX_RENDERBUFFER_SIZE: u32 = 16384;
pub const MAX_COLOR_ATTACHMENTS: u32 = 8;
pub const MAX_DRAW_BUFFERS: u32 = 8;
/// The sample counts a multisampled renderbuffer takes, the smallest first: a request rounds up to one
pub const SAMPLE_COUNTS: [u32; 4] = [2, 4, 8, 16];

/// A renderbuffer's storage
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Renderbuffer {
    pub format: Option<InternalFormat>,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
}

impl Default for Renderbuffer {
    fn default() -> Self {
        Renderbuffer { format: None, width: 0, height: 0, samples: 0 }
    }
}

impl Renderbuffer {
    /// glRenderbufferStorageMultisample (glRenderbufferStorage is `samples` 0), GL ES 3.0 §4.4.2
    pub fn storage(&mut self, samples: i32, format: InternalFormat, width: i32, height: i32) -> GlResult<()> {
        if !format.sized() || format.renderable() == Renderable::No {
            return Err(GlError::InvalidEnum);
        }
        let (Ok(samples), Ok(width), Ok(height)) = (u32::try_from(samples), u32::try_from(width), u32::try_from(height)) else { return Err(GlError::InvalidValue) };
        if width > MAX_RENDERBUFFER_SIZE || height > MAX_RENDERBUFFER_SIZE {
            return Err(GlError::InvalidValue);
        }
        let samples = match samples {
            0 => 0,
            n => *SAMPLE_COUNTS.iter().find(|&&s| s >= n).ok_or(GlError::InvalidOperation)?,
        };
        *self = Renderbuffer { format: Some(format), width, height, samples };
        Ok(())
    }
}

/// A framebuffer's attachment points
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Attachment {
    Color(u32),
    Depth,
    Stencil,
}

/// glFramebufferTexture2D / glFramebufferTextureLayer / glFramebufferRenderbuffer's point: DEPTH_STENCIL is both
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachPoint {
    Color(u32),
    Depth,
    Stencil,
    DepthStencil,
}

impl AttachPoint {
    pub fn points(self) -> GlResult<Vec<Attachment>> {
        match self {
            AttachPoint::Color(i) if i < MAX_COLOR_ATTACHMENTS => Ok(vec![Attachment::Color(i)]),
            AttachPoint::Color(_) => Err(GlError::InvalidEnum),
            AttachPoint::Depth => Ok(vec![Attachment::Depth]),
            AttachPoint::Stencil => Ok(vec![Attachment::Stencil]),
            AttachPoint::DepthStencil => Ok(vec![Attachment::Depth, Attachment::Stencil]),
        }
    }
}

/// What an attachment point holds: a texture image (`layer`: the cube face, or the 3D / array layer) or a
/// renderbuffer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attached {
    Texture { name: u32, level: u32, layer: u32 },
    Renderbuffer(u32),
}

/// glCheckFramebufferStatus's answers
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramebufferStatus {
    Complete,
    IncompleteAttachment,
    IncompleteMissingAttachment,
    IncompleteMultisample,
    Unsupported,
}

/// glDrawBuffers / glReadBuffer's buffers on a framebuffer object
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorBuffer {
    None,
    Color(u32),
    /// GL_BACK: the default framebuffer's only
    Back,
}

/// A framebuffer object: its attachments, draw buffers and read buffer
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Framebuffer {
    pub attached: BTreeMap<Attachment, Attached>,
    pub draw_buffers: Vec<ColorBuffer>,
    pub read_buffer: ColorBuffer,
}

impl Default for Framebuffer {
    fn default() -> Self {
        Framebuffer { attached: BTreeMap::new(), draw_buffers: vec![ColorBuffer::Color(0)], read_buffer: ColorBuffer::Color(0) }
    }
}

/// An attached image's format, size and samples
struct Image {
    format: InternalFormat,
    size: [u32; 3],
    samples: u32,
}

impl Framebuffer {
    pub fn attach(&mut self, point: AttachPoint, what: Option<Attached>) -> GlResult<()> {
        for a in point.points()? {
            match what {
                Some(w) => self.attached.insert(a, w),
                None => self.attached.remove(&a),
            };
        }
        Ok(())
    }

    /// glDrawBuffers on a framebuffer object: buffer i is NONE or COLOR_ATTACHMENTi (GL ES 3.0 §4.2.1)
    pub fn set_draw_buffers(&mut self, bufs: &[ColorBuffer]) -> GlResult<()> {
        if bufs.len() > MAX_DRAW_BUFFERS as usize {
            return Err(GlError::InvalidValue);
        }
        for (i, b) in bufs.iter().enumerate() {
            match b {
                ColorBuffer::None => {}
                ColorBuffer::Color(c) if *c as usize == i => {}
                ColorBuffer::Color(c) if *c >= MAX_COLOR_ATTACHMENTS => return Err(GlError::InvalidEnum),
                _ => return Err(GlError::InvalidOperation),
            }
        }
        self.draw_buffers = bufs.to_vec();
        Ok(())
    }

    pub fn set_read_buffer(&mut self, b: ColorBuffer) -> GlResult<()> {
        match b {
            ColorBuffer::Back => Err(GlError::InvalidOperation),
            ColorBuffer::Color(c) if c >= MAX_COLOR_ATTACHMENTS => Err(GlError::InvalidOperation),
            b => {
                self.read_buffer = b;
                Ok(())
            }
        }
    }

    fn image(what: Attached, textures: &Names<Texture>, renderbuffers: &Names<Renderbuffer>) -> Option<Image> {
        match what {
            Attached::Texture { name, level, layer } => {
                let t = textures.get(name)?;
                let (face, layer) = if t.target == TextureTarget::CubeMap { (layer as usize, 0) } else { (0, layer) };
                let l = t.level(face, level)?;
                (layer < l.size[2] || t.target == TextureTarget::Texture2D).then_some(Image { format: l.format, size: l.size, samples: 0 })
            }
            Attached::Renderbuffer(name) => {
                let r = renderbuffers.get(name)?;
                Some(Image { format: r.format?, size: [r.width, r.height, 1], samples: r.samples })
            }
        }
    }

    /// GL ES 3.0 §4.4.4: attachment completeness, then framebuffer completeness
    pub fn status(&self, textures: &Names<Texture>, renderbuffers: &Names<Renderbuffer>) -> FramebufferStatus {
        if self.attached.is_empty() {
            return FramebufferStatus::IncompleteMissingAttachment;
        }
        let mut samples = None;
        for (&point, &what) in &self.attached {
            let Some(img) = Framebuffer::image(what, textures, renderbuffers) else { return FramebufferStatus::IncompleteAttachment };
            if img.size[0] == 0 || img.size[1] == 0 {
                return FramebufferStatus::IncompleteAttachment;
            }
            let r = img.format.renderable();
            let fits = match point {
                Attachment::Color(_) => r == Renderable::Color,
                Attachment::Depth => matches!(r, Renderable::Depth | Renderable::DepthStencil),
                Attachment::Stencil => matches!(r, Renderable::Stencil | Renderable::DepthStencil),
            };
            if !fits {
                return FramebufferStatus::IncompleteAttachment;
            }
            if *samples.get_or_insert(img.samples) != img.samples {
                return FramebufferStatus::IncompleteMultisample;
            }
        }
        match (self.attached.get(&Attachment::Depth), self.attached.get(&Attachment::Stencil)) {
            (Some(d), Some(s)) if d != s => FramebufferStatus::Unsupported,
            _ => FramebufferStatus::Complete,
        }
    }
}
