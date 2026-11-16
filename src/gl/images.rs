use mlirformat::Error;

use super::{
    AttachPoint, Attached, BufferTarget, ColorBuffer, Context, Framebuffer, FramebufferStatus, GlError, GlResult, ImageTarget, InternalFormat, Level, MAX_COMBINED_TEXTURE_IMAGE_UNITS, PixelFormat,
    PixelStore, PixelType, Pixels, Renderbuffer, TexParam, Texture, TextureTarget,
};

/// Where glTexImage* / glTexSubImage* read texels from: nothing, client memory, or (with a PIXEL_UNPACK_BUFFER
/// bound) an offset into that buffer
#[derive(Clone, Copy, Debug)]
pub enum TexelSource<'d> {
    None,
    Client(&'d [u8]),
    Offset(usize),
}

/// Which of the 2D / 3D entry points a call is: glTexImage2D takes 2D targets, glTexImage3D the others
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dims {
    Two,
    Three,
}

/// glBindFramebuffer's targets
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramebufferTarget {
    /// GL_FRAMEBUFFER: draw and read
    Both,
    Draw,
    Read,
}

/// One glTexImage* / glTexSubImage* region: level, offset, size
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub level: i32,
    pub offset: [i32; 3],
    pub size: [i32; 3],
}

impl Context<'_> {
    pub fn active_texture(&mut self, unit: u32) {
        let r = if unit < MAX_COMBINED_TEXTURE_IMAGE_UNITS { Ok(self.active_unit = unit) } else { Err(GlError::InvalidEnum) };
        self.record(r)
    }

    pub fn gen_textures(&mut self, n: usize) -> Vec<u32> {
        self.textures.reserve_n(n)
    }

    /// glBindTexture: the first bind fixes the texture's target; another target is an error
    pub fn bind_texture(&mut self, target: TextureTarget, name: u32) {
        let r = match name {
            0 => Ok(()),
            n => {
                self.textures.bind_any(n, || Texture::new(target));
                match self.textures.get(n) {
                    Some(t) if t.target != target => Err(GlError::InvalidOperation),
                    _ => Ok(()),
                }
            }
        };
        if self.record(r.map(|()| true)) {
            self.units[self.active_unit as usize].insert(target, name);
        }
    }

    /// The texture bound to `target` on the active unit
    pub fn bound_texture(&self, target: TextureTarget) -> u32 {
        self.units[self.active_unit as usize].get(&target).copied().unwrap_or(0)
    }

    /// glDeleteTextures: unbound from every unit, detached from the bound framebuffers
    pub fn delete_textures(&mut self, names: &[u32]) {
        for &n in names.iter().filter(|&&n| n != 0) {
            if self.textures.delete(n).is_none() {
                continue;
            }
            for unit in &mut self.units {
                unit.values_mut().filter(|b| **b == n).for_each(|b| *b = 0);
            }
            self.detach_bound(|a| matches!(a, Attached::Texture { name, .. } if *name == n));
        }
    }

    pub fn is_texture(&self, name: u32) -> bool {
        self.textures.is(name)
    }

    pub fn pixel_store(&mut self, p: PixelStore) {
        let r = self.unpack.set(&mut self.pack, p);
        self.record(r)
    }

    /// The texels a call reads: client memory, or the PIXEL_UNPACK_BUFFER (GL ES 3.0 §3.7.1)
    fn texels(&self, source: TexelSource<'_>, size: [u32; 3], format: PixelFormat, ty: PixelType) -> GlResult<Option<Vec<u8>>> {
        let unpack = self.bound.get(&BufferTarget::PixelUnpack).copied().unwrap_or(0);
        match (unpack, source) {
            (0, TexelSource::None) => Ok(None),
            (0, TexelSource::Client(d)) => Ok(Some(d.to_vec())),
            (0, TexelSource::Offset(_)) => Err(GlError::InvalidOperation),
            (b, s) => {
                let offset = match s {
                    TexelSource::Offset(o) => o,
                    _ => 0,
                };
                let data = &self.buffers.get(b).ok_or(GlError::InvalidOperation)?.data;
                if offset % ty.element() as usize != 0 || offset as u64 + self.unpack.extent(size, format, ty) > data.len() as u64 {
                    return Err(GlError::InvalidOperation);
                }
                Ok(Some(data[offset..].to_vec()))
            }
        }
    }

    /// The texture a glTexImage* call writes: the one bound to its target
    fn image_texture(&mut self, target: ImageTarget) -> GlResult<&mut Texture> {
        let name = self.bound_texture(target.texture());
        self.textures.get_mut(name).ok_or(GlError::InvalidOperation)
    }

    /// glTexImage2D (`dims` 2, `size[2]` 1) and glTexImage3D (`dims` 3). A GL error is recorded; an error
    /// returned is what detour cannot store yet
    #[allow(clippy::too_many_arguments, reason = "glTexImage3D's arguments")]
    pub fn tex_image(&mut self, dims: Dims, target: ImageTarget, level: i32, internal: InternalFormat, size: [i32; 3], border: i32, format: PixelFormat, ty: PixelType, source: TexelSource<'_>) -> Result<(), Error> {
        if target.two_d() != (dims == Dims::Two) {
            return Ok(self.fail(GlError::InvalidEnum));
        }
        let dims = size.map(|v| u32::try_from(v).unwrap_or(0));
        let texels = match self.texels(source, dims, format, ty) {
            Ok(t) => t,
            Err(e) => return Ok(self.record(Err(e))),
        };
        let unpack = self.unpack;
        let r = self.image_texture(target).and_then(|t| t.image(target, level, internal, size, border, Pixels { format, ty, data: texels.as_deref() }, &unpack));
        self.record(r.map(Some)).unwrap_or(Ok(()))
    }

    /// glTexSubImage2D (`dims` 2, `region.size[2]` 1, `offset[2]` 0) and glTexSubImage3D (`dims` 3)
    pub fn tex_sub_image(&mut self, dims: Dims, target: ImageTarget, region: Region, format: PixelFormat, ty: PixelType, source: TexelSource<'_>) -> Result<(), Error> {
        if target.two_d() != (dims == Dims::Two) {
            return Ok(self.fail(GlError::InvalidEnum));
        }
        let dims = region.size.map(|v| u32::try_from(v).unwrap_or(0));
        let texels = match self.texels(source, dims, format, ty) {
            Ok(t) => t,
            Err(e) => return Ok(self.record(Err(e))),
        };
        let unpack = self.unpack;
        let r = self.image_texture(target).and_then(|t| t.sub_image(target, region.level, region.offset, region.size, Pixels { format, ty, data: texels.as_deref() }, &unpack));
        self.record(r.map(Some)).unwrap_or(Ok(()))
    }

    /// glTexStorage2D (`size[2]` 1) and glTexStorage3D
    pub fn tex_storage(&mut self, target: TextureTarget, levels: i32, internal: InternalFormat, size: [i32; 3]) {
        let name = self.bound_texture(target);
        let r = self.textures.get_mut(name).ok_or(GlError::InvalidOperation).and_then(|t| t.storage(target, levels, internal, size));
        self.record(r)
    }

    pub fn tex_parameter(&mut self, target: TextureTarget, p: TexParam) {
        let name = self.bound_texture(target);
        let r = self.textures.get_mut(name).ok_or(GlError::InvalidOperation).and_then(|t| t.sampling.set(p));
        self.record(r)
    }

    /// glGetTexLevelParameter's answer for the bound texture: the level, if it has one
    pub fn tex_level(&self, target: ImageTarget, level: u32) -> Option<&Level> {
        let face = match target {
            ImageTarget::CubeFace(f) => f as usize,
            _ => 0,
        };
        self.textures.get(self.bound_texture(target.texture()))?.level(face, level)
    }

    pub fn gen_renderbuffers(&mut self, n: usize) -> Vec<u32> {
        self.renderbuffers.reserve_n(n)
    }

    pub fn bind_renderbuffer(&mut self, name: u32) {
        if name != 0 {
            self.renderbuffers.bind_any(name, Renderbuffer::default);
        }
        self.renderbuffer = name;
    }

    pub fn delete_renderbuffers(&mut self, names: &[u32]) {
        for &n in names.iter().filter(|&&n| n != 0) {
            if self.renderbuffers.delete(n).is_none() {
                continue;
            }
            if self.renderbuffer == n {
                self.renderbuffer = 0;
            }
            self.detach_bound(|a| *a == Attached::Renderbuffer(n));
        }
    }

    pub fn is_renderbuffer(&self, name: u32) -> bool {
        self.renderbuffers.is(name)
    }

    /// glRenderbufferStorageMultisample; glRenderbufferStorage with `samples` 0
    pub fn renderbuffer_storage(&mut self, samples: i32, format: InternalFormat, width: i32, height: i32) {
        let name = self.renderbuffer;
        let r = self.renderbuffers.get_mut(name).ok_or(GlError::InvalidOperation).and_then(|rb| rb.storage(samples, format, width, height));
        self.record(r)
    }

    pub fn gen_framebuffers(&mut self, n: usize) -> Vec<u32> {
        self.framebuffers.reserve_n(n)
    }

    pub fn bind_framebuffer(&mut self, target: FramebufferTarget, name: u32) {
        if name != 0 {
            self.framebuffers.bind_any(name, Framebuffer::default);
        }
        if target != FramebufferTarget::Read {
            self.draw_framebuffer = name;
        }
        if target != FramebufferTarget::Draw {
            self.read_framebuffer = name;
        }
    }

    /// glDeleteFramebuffers: a bound one is unbound (back to the default framebuffer)
    pub fn delete_framebuffers(&mut self, names: &[u32]) {
        for &n in names.iter().filter(|&&n| n != 0) {
            if self.framebuffers.delete(n).is_none() {
                continue;
            }
            if self.draw_framebuffer == n {
                self.draw_framebuffer = 0;
            }
            if self.read_framebuffer == n {
                self.read_framebuffer = 0;
            }
        }
    }

    pub fn is_framebuffer(&self, name: u32) -> bool {
        self.framebuffers.is(name)
    }

    /// The framebuffer object a call on `target` changes (GL_FRAMEBUFFER: the draw one); the default one is an error
    fn framebuffer_of(&mut self, target: FramebufferTarget) -> GlResult<&mut Framebuffer> {
        let name = if target == FramebufferTarget::Read { self.read_framebuffer } else { self.draw_framebuffer };
        if name == 0 {
            return Err(GlError::InvalidOperation);
        }
        self.framebuffers.get_mut(name).ok_or(GlError::InvalidOperation)
    }

    /// Images `gone` matches leave the bound draw and read framebuffers (GL ES 3.0 §4.4.2.3)
    fn detach_bound(&mut self, gone: impl Fn(&Attached) -> bool) {
        for fb in [self.draw_framebuffer, self.read_framebuffer] {
            if let Some(f) = self.framebuffers.get_mut(fb) {
                f.attached.retain(|_, a| !gone(a));
            }
        }
    }

    /// glFramebufferTexture2D: a 2D texture's level or a cube face's (texture 0 detaches)
    pub fn framebuffer_texture_2d(&mut self, target: FramebufferTarget, point: AttachPoint, image: ImageTarget, texture: u32, level: i32) {
        if !image.two_d() {
            return self.fail(GlError::InvalidOperation);
        }
        let r = self.texture_attachment(image, texture, level, 0).and_then(|a| self.framebuffer_of(target)?.attach(point, a));
        self.record(r)
    }

    /// glFramebufferTextureLayer: a layer of a 3D or array texture's level (texture 0 detaches)
    pub fn framebuffer_texture_layer(&mut self, target: FramebufferTarget, point: AttachPoint, texture: u32, level: i32, layer: i32) {
        let r = match self.textures.get(texture).map(|t| t.target) {
            Some(TextureTarget::Texture3D) => Ok(ImageTarget::Texture3D),
            Some(TextureTarget::Texture2DArray) => Ok(ImageTarget::Texture2DArray),
            Some(_) => Err(GlError::InvalidOperation),
            None if texture == 0 => Ok(ImageTarget::Texture3D),
            None => Err(GlError::InvalidOperation),
        };
        let r = r.and_then(|image| {
            let layer = u32::try_from(layer).map_err(|_| GlError::InvalidValue)?;
            let a = self.texture_attachment(image, texture, level, layer)?;
            self.framebuffer_of(target)?.attach(point, a)
        });
        self.record(r)
    }

    /// The attachment of `texture`'s `level` / `layer` through `image`: the texture's target must be the image's
    fn texture_attachment(&self, image: ImageTarget, texture: u32, level: i32, layer: u32) -> GlResult<Option<Attached>> {
        if texture == 0 {
            return Ok(None);
        }
        let t = self.textures.get(texture).ok_or(GlError::InvalidOperation)?;
        if t.target != image.texture() {
            return Err(GlError::InvalidOperation);
        }
        let level = u32::try_from(level).map_err(|_| GlError::InvalidValue)?;
        let max = match t.target {
            TextureTarget::Texture3D => super::MAX_3D_TEXTURE_SIZE,
            TextureTarget::CubeMap => super::MAX_CUBE_MAP_TEXTURE_SIZE,
            _ => super::MAX_TEXTURE_SIZE,
        };
        if level > max.ilog2() || t.immutable.is_some_and(|n| level >= n) {
            return Err(GlError::InvalidValue);
        }
        let deepest = match t.target {
            TextureTarget::Texture3D => super::MAX_3D_TEXTURE_SIZE,
            TextureTarget::Texture2DArray => super::MAX_ARRAY_TEXTURE_LAYERS,
            _ => 1,
        };
        if layer >= deepest {
            return Err(GlError::InvalidValue);
        }
        let layer = match image {
            ImageTarget::CubeFace(f) => u32::from(f),
            _ => layer,
        };
        Ok(Some(Attached::Texture { name: texture, level, layer }))
    }

    /// glFramebufferRenderbuffer (renderbuffer 0 detaches)
    pub fn framebuffer_renderbuffer(&mut self, target: FramebufferTarget, point: AttachPoint, renderbuffer: u32) {
        let r = match renderbuffer {
            0 => Ok(None),
            n if self.renderbuffers.is(n) => Ok(Some(Attached::Renderbuffer(n))),
            _ => Err(GlError::InvalidOperation),
        };
        let r = r.and_then(|a| self.framebuffer_of(target)?.attach(point, a));
        self.record(r)
    }

    /// glCheckFramebufferStatus: the default framebuffer is complete
    pub fn check_framebuffer_status(&self, target: FramebufferTarget) -> FramebufferStatus {
        let name = if target == FramebufferTarget::Read { self.read_framebuffer } else { self.draw_framebuffer };
        match self.framebuffers.get(name) {
            None => FramebufferStatus::Complete,
            Some(f) => f.status(&self.textures, &self.renderbuffers),
        }
    }

    /// glDrawBuffers: on the default framebuffer one buffer, BACK or NONE
    pub fn draw_buffers(&mut self, bufs: &[ColorBuffer]) {
        let r = match self.draw_framebuffer {
            0 => match bufs {
                [b @ (ColorBuffer::Back | ColorBuffer::None)] => Ok(self.default_draw = *b),
                [] | [_] => Err(GlError::InvalidOperation),
                _ => Err(GlError::InvalidValue),
            },
            _ => self.framebuffer_of(FramebufferTarget::Draw).and_then(|f| f.set_draw_buffers(bufs)),
        };
        self.record(r)
    }

    /// glReadBuffer: on the default framebuffer BACK or NONE
    pub fn read_buffer(&mut self, b: ColorBuffer) {
        let r = match (self.read_framebuffer, b) {
            (0, ColorBuffer::Back | ColorBuffer::None) => Ok(self.default_read = b),
            (0, _) => Err(GlError::InvalidOperation),
            _ => self.framebuffer_of(FramebufferTarget::Read).and_then(|f| f.set_read_buffer(b)),
        };
        self.record(r)
    }
}
