use super::{AttachPoint, Attached, Attachment, Compare, Context, Filter, FramebufferTarget, Gl, GlError, GlResult, InternalFormat, Swizzle, TexParam, TextureTarget, Wrap};

/// glRenderbufferStorage's format before any storage (GL ES 3.0 table 6.14)
const RENDERBUFFER_FORMAT_INITIAL: u32 = Gl::RGBA4;

impl TexParam {
    /// glTexParameteri's `pname` and `param`
    pub fn from_gl(pname: u32, v: i32) -> GlResult<TexParam> {
        let e = || u32::try_from(v).map_err(|_| GlError::InvalidEnum);
        Ok(match pname {
            Gl::TEXTURE_MIN_FILTER => TexParam::MinFilter(Filter::from_gl(e()?)?),
            Gl::TEXTURE_MAG_FILTER => TexParam::MagFilter(Filter::from_gl(e()?)?),
            Gl::TEXTURE_WRAP_S => TexParam::WrapS(Wrap::from_gl(e()?)?),
            Gl::TEXTURE_WRAP_T => TexParam::WrapT(Wrap::from_gl(e()?)?),
            Gl::TEXTURE_WRAP_R => TexParam::WrapR(Wrap::from_gl(e()?)?),
            Gl::TEXTURE_MIN_LOD => TexParam::MinLod(v as f32),
            Gl::TEXTURE_MAX_LOD => TexParam::MaxLod(v as f32),
            Gl::TEXTURE_BASE_LEVEL => TexParam::BaseLevel(v),
            Gl::TEXTURE_MAX_LEVEL => TexParam::MaxLevel(v),
            Gl::TEXTURE_COMPARE_MODE => match e()? {
                Gl::COMPARE_REF_TO_TEXTURE => TexParam::CompareRef(true),
                Gl::NONE => TexParam::CompareRef(false),
                _ => return Err(GlError::InvalidEnum),
            },
            Gl::TEXTURE_COMPARE_FUNC => TexParam::CompareFunc(Compare::from_gl(e()?)?),
            Gl::TEXTURE_SWIZZLE_R => TexParam::Swizzle(0, Swizzle::from_gl(e()?)?),
            Gl::TEXTURE_SWIZZLE_G => TexParam::Swizzle(1, Swizzle::from_gl(e()?)?),
            Gl::TEXTURE_SWIZZLE_B => TexParam::Swizzle(2, Swizzle::from_gl(e()?)?),
            Gl::TEXTURE_SWIZZLE_A => TexParam::Swizzle(3, Swizzle::from_gl(e()?)?),
            _ => return Err(GlError::InvalidEnum),
        })
    }
}

impl Context<'_> {
    /// A GL error the caller found (an argument no enum takes): recorded like any other
    pub fn fail(&mut self, e: GlError) {
        self.record::<()>(Err(e));
    }

    /// glGetTexParameteriv on the bound texture
    pub fn get_tex_parameter(&mut self, target: TextureTarget, pname: u32) -> i32 {
        let name = self.bound_texture(target);
        let r = self.textures.get(name).ok_or(GlError::InvalidOperation).and_then(|t| {
            let s = &t.sampling;
            let g = |v: u32| v as i32;
            Ok(match pname {
                Gl::TEXTURE_MIN_FILTER => g(s.min_filter.to_gl()),
                Gl::TEXTURE_MAG_FILTER => g(s.mag_filter.to_gl()),
                Gl::TEXTURE_WRAP_S => g(s.wrap[0].to_gl()),
                Gl::TEXTURE_WRAP_T => g(s.wrap[1].to_gl()),
                Gl::TEXTURE_WRAP_R => g(s.wrap[2].to_gl()),
                Gl::TEXTURE_MIN_LOD => s.min_lod as i32,
                Gl::TEXTURE_MAX_LOD => s.max_lod as i32,
                Gl::TEXTURE_BASE_LEVEL => s.base_level as i32,
                Gl::TEXTURE_MAX_LEVEL => s.max_level as i32,
                Gl::TEXTURE_COMPARE_MODE => g(if s.compare_ref { Gl::COMPARE_REF_TO_TEXTURE } else { Gl::NONE }),
                Gl::TEXTURE_COMPARE_FUNC => g(s.compare_func.to_gl()),
                Gl::TEXTURE_SWIZZLE_R => g(s.swizzle[0].to_gl()),
                Gl::TEXTURE_SWIZZLE_G => g(s.swizzle[1].to_gl()),
                Gl::TEXTURE_SWIZZLE_B => g(s.swizzle[2].to_gl()),
                Gl::TEXTURE_SWIZZLE_A => g(s.swizzle[3].to_gl()),
                Gl::TEXTURE_IMMUTABLE_FORMAT => i32::from(t.immutable.is_some()),
                Gl::TEXTURE_IMMUTABLE_LEVELS => t.immutable.unwrap_or(0) as i32,
                _ => return Err(GlError::InvalidEnum),
            })
        });
        self.record(r)
    }

    /// glGetRenderbufferParameteriv on the bound renderbuffer
    pub fn get_renderbuffer_parameter(&mut self, pname: u32) -> i32 {
        let name = self.renderbuffer;
        let r = self.renderbuffers.get(name).ok_or(GlError::InvalidOperation).and_then(|rb| {
            Ok(match pname {
                Gl::RENDERBUFFER_WIDTH => rb.width as i32,
                Gl::RENDERBUFFER_HEIGHT => rb.height as i32,
                Gl::RENDERBUFFER_INTERNAL_FORMAT => rb.format.map_or(RENDERBUFFER_FORMAT_INITIAL, InternalFormat::to_gl) as i32,
                Gl::RENDERBUFFER_SAMPLES => rb.samples as i32,
                _ => return Err(GlError::InvalidEnum),
            })
        });
        self.record(r)
    }

    /// glGetFramebufferAttachmentParameteriv on a bound framebuffer object (GL ES 3.0 §6.1.13)
    pub fn get_framebuffer_attachment_parameter(&mut self, target: FramebufferTarget, point: AttachPoint, pname: u32) -> i32 {
        let name = if target == FramebufferTarget::Read { self.read_framebuffer } else { self.draw_framebuffer };
        let r = self.framebuffers.get(name).ok_or(GlError::InvalidOperation).and_then(|f| {
            let at = match point {
                AttachPoint::DepthStencil => {
                    let (d, s) = (f.attached.get(&Attachment::Depth), f.attached.get(&Attachment::Stencil));
                    if d != s {
                        return Err(GlError::InvalidOperation);
                    }
                    d.copied()
                }
                p => f.attached.get(&p.points()?[0]).copied(),
            };
            match (at, pname) {
                (None, Gl::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) => Ok(Gl::NONE as i32),
                (None, Gl::FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) => Ok(0),
                (None, _) => Err(GlError::InvalidOperation),
                (Some(a), Gl::FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) => Ok(match a {
                    Attached::Texture { .. } => Gl::TEXTURE,
                    Attached::Renderbuffer(_) => Gl::RENDERBUFFER,
                } as i32),
                (Some(Attached::Texture { name, .. } | Attached::Renderbuffer(name)), Gl::FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) => Ok(name as i32),
                (Some(Attached::Texture { level, .. }), Gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL) => Ok(level as i32),
                (Some(Attached::Texture { name, layer, .. }), Gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE) => {
                    let cube = self.textures.get(name).is_some_and(|t| t.target == TextureTarget::CubeMap);
                    Ok(if cube { (Gl::TEXTURE_CUBE_MAP_POSITIVE_X + layer) as i32 } else { 0 })
                }
                (Some(Attached::Texture { name, layer, .. }), Gl::FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER) => {
                    let layered = self.textures.get(name).is_some_and(|t| matches!(t.target, TextureTarget::Texture3D | TextureTarget::Texture2DArray));
                    Ok(if layered { layer as i32 } else { 0 })
                }
                _ => Err(GlError::InvalidEnum),
            }
        });
        self.record(r)
    }
}
