use mlirformat::Error;

use super::{Compare, GlError, GlResult, InternalFormat, PixelFormat, PixelType};

/// Texture size limits (the values Mesa's iris reports on Xe3)
pub const MAX_TEXTURE_SIZE: u32 = 16384;
pub const MAX_3D_TEXTURE_SIZE: u32 = 2048;
pub const MAX_CUBE_MAP_TEXTURE_SIZE: u32 = 16384;
pub const MAX_ARRAY_TEXTURE_LAYERS: u32 = 2048;
/// GL ES 3.0's minimum; texture units 0 .. this
pub const MAX_COMBINED_TEXTURE_IMAGE_UNITS: u32 = 32;
/// Faces of a cube map
const CUBE_FACES: usize = 6;
/// GL_UNPACK_ALIGNMENT's initial value and the ones it takes
const UNPACK_ALIGNMENT: u32 = 4;
const ALIGNMENTS: [u32; 4] = [1, 2, 4, 8];

/// What glBindTexture binds
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TextureTarget {
    Texture2D,
    Texture3D,
    Texture2DArray,
    CubeMap,
}

impl TextureTarget {
    pub const ALL: [TextureTarget; 4] = [TextureTarget::Texture2D, TextureTarget::Texture3D, TextureTarget::Texture2DArray, TextureTarget::CubeMap];

    const fn max_size(self) -> u32 {
        match self {
            TextureTarget::Texture2D | TextureTarget::Texture2DArray => MAX_TEXTURE_SIZE,
            TextureTarget::Texture3D => MAX_3D_TEXTURE_SIZE,
            TextureTarget::CubeMap => MAX_CUBE_MAP_TEXTURE_SIZE,
        }
    }

    /// The deepest level a texture of this target can have
    fn max_level(self) -> u32 {
        self.max_size().ilog2()
    }

    const fn faces(self) -> usize {
        if matches!(self, TextureTarget::CubeMap) { CUBE_FACES } else { 1 }
    }
}

/// What glTexImage*/glTexSubImage* write: a texture target, or one face of a cube map (0 .. 6: +X −X +Y −Y +Z −Z)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageTarget {
    Texture2D,
    CubeFace(u8),
    Texture3D,
    Texture2DArray,
}

impl ImageTarget {
    pub const fn texture(self) -> TextureTarget {
        match self {
            ImageTarget::Texture2D => TextureTarget::Texture2D,
            ImageTarget::CubeFace(_) => TextureTarget::CubeMap,
            ImageTarget::Texture3D => TextureTarget::Texture3D,
            ImageTarget::Texture2DArray => TextureTarget::Texture2DArray,
        }
    }

    const fn face(self) -> usize {
        match self {
            ImageTarget::CubeFace(f) => f as usize,
            _ => 0,
        }
    }

    /// glTexImage2D / glTexSubImage2D / glTexStorage2D take 2D targets, the 3D calls the others
    pub const fn two_d(self) -> bool {
        matches!(self, ImageTarget::Texture2D | ImageTarget::CubeFace(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrap {
    Repeat,
    ClampToEdge,
    MirroredRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swizzle {
    Red,
    Green,
    Blue,
    Alpha,
    Zero,
    One,
}

/// glTexParameter*: a parameter and its value
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TexParam {
    MinFilter(Filter),
    MagFilter(Filter),
    WrapS(Wrap),
    WrapT(Wrap),
    WrapR(Wrap),
    MinLod(f32),
    MaxLod(f32),
    BaseLevel(i32),
    MaxLevel(i32),
    /// TEXTURE_COMPARE_MODE: COMPARE_REF_TO_TEXTURE (true) or NONE
    CompareRef(bool),
    CompareFunc(Compare),
    Swizzle(usize, Swizzle),
}

/// A texture's sampling state (GL ES 3.0 table 6.10)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub min_filter: Filter,
    pub mag_filter: Filter,
    pub wrap: [Wrap; 3],
    pub min_lod: f32,
    pub max_lod: f32,
    pub base_level: u32,
    pub max_level: u32,
    pub compare_ref: bool,
    pub compare_func: Compare,
    pub swizzle: [Swizzle; 4],
}

/// GL_TEXTURE_MAX_LEVEL's initial value
const MAX_LEVEL_INITIAL: u32 = 1000;
const MIN_LOD_INITIAL: f32 = -1000.0;
const MAX_LOD_INITIAL: f32 = 1000.0;

impl Default for Sampling {
    fn default() -> Self {
        Sampling {
            min_filter: Filter::NearestMipmapLinear,
            mag_filter: Filter::Linear,
            wrap: [Wrap::Repeat; 3],
            min_lod: MIN_LOD_INITIAL,
            max_lod: MAX_LOD_INITIAL,
            base_level: 0,
            max_level: MAX_LEVEL_INITIAL,
            compare_ref: false,
            compare_func: Compare::LessEqual,
            swizzle: [Swizzle::Red, Swizzle::Green, Swizzle::Blue, Swizzle::Alpha],
        }
    }
}

impl Sampling {
    /// GL ES 3.0 §3.8.7: the values each parameter takes
    pub fn set(&mut self, p: TexParam) -> GlResult<()> {
        match p {
            TexParam::MinFilter(f) => self.min_filter = f,
            TexParam::MagFilter(f @ (Filter::Nearest | Filter::Linear)) => self.mag_filter = f,
            TexParam::MagFilter(_) => return Err(GlError::InvalidEnum),
            TexParam::WrapS(w) => self.wrap[0] = w,
            TexParam::WrapT(w) => self.wrap[1] = w,
            TexParam::WrapR(w) => self.wrap[2] = w,
            TexParam::MinLod(v) => self.min_lod = v,
            TexParam::MaxLod(v) => self.max_lod = v,
            TexParam::BaseLevel(v) => self.base_level = u32::try_from(v).map_err(|_| GlError::InvalidValue)?,
            TexParam::MaxLevel(v) => self.max_level = u32::try_from(v).map_err(|_| GlError::InvalidValue)?,
            TexParam::CompareRef(on) => self.compare_ref = on,
            TexParam::CompareFunc(f) => self.compare_func = f,
            TexParam::Swizzle(c, s) => *self.swizzle.get_mut(c).ok_or(GlError::InvalidEnum)? = s,
        }
        Ok(())
    }
}

/// glPixelStorei's unpack state: how client texels are laid out
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unpack {
    pub alignment: u32,
    pub row_length: u32,
    pub image_height: u32,
    pub skip_pixels: u32,
    pub skip_rows: u32,
    pub skip_images: u32,
}

impl Default for Unpack {
    fn default() -> Self {
        Unpack { alignment: UNPACK_ALIGNMENT, row_length: 0, image_height: 0, skip_pixels: 0, skip_rows: 0, skip_images: 0 }
    }
}

/// glPixelStorei's parameters
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelStore {
    PackAlignment(i32),
    PackRowLength(i32),
    PackSkipPixels(i32),
    PackSkipRows(i32),
    UnpackAlignment(i32),
    UnpackRowLength(i32),
    UnpackImageHeight(i32),
    UnpackSkipPixels(i32),
    UnpackSkipRows(i32),
    UnpackSkipImages(i32),
}

impl Unpack {
    /// glPixelStorei: the unpack state is `self`, the pack state `pack` (GL ES 3.0 table 3.1 and 4.5)
    pub fn set(&mut self, pack: &mut Unpack, p: PixelStore) -> GlResult<()> {
        let n = |v: i32| u32::try_from(v).map_err(|_| GlError::InvalidValue);
        match p {
            PixelStore::PackAlignment(v) => {
                let v = n(v)?;
                if !ALIGNMENTS.contains(&v) {
                    return Err(GlError::InvalidValue);
                }
                pack.alignment = v;
            }
            PixelStore::PackRowLength(v) => pack.row_length = n(v)?,
            PixelStore::PackSkipPixels(v) => pack.skip_pixels = n(v)?,
            PixelStore::PackSkipRows(v) => pack.skip_rows = n(v)?,
            PixelStore::UnpackAlignment(v) => {
                let v = n(v)?;
                if !ALIGNMENTS.contains(&v) {
                    return Err(GlError::InvalidValue);
                }
                self.alignment = v;
            }
            PixelStore::UnpackRowLength(v) => self.row_length = n(v)?,
            PixelStore::UnpackImageHeight(v) => self.image_height = n(v)?,
            PixelStore::UnpackSkipPixels(v) => self.skip_pixels = n(v)?,
            PixelStore::UnpackSkipRows(v) => self.skip_rows = n(v)?,
            PixelStore::UnpackSkipImages(v) => self.skip_images = n(v)?,
        }
        Ok(())
    }

    /// GL ES 3.0 §3.7.2: the bytes between rows and between images of a `w` × `h` image of `format` / `ty`
    fn strides(&self, w: u32, h: u32, format: PixelFormat, ty: PixelType) -> (u64, u64) {
        let pixel = u64::from(ty.pixel(format));
        let element = u64::from(ty.element());
        let row = u64::from(if self.row_length > 0 { self.row_length } else { w }) * pixel;
        let a = u64::from(self.alignment);
        let row = if element >= a { row } else { row.div_ceil(a) * a };
        (row, row * u64::from(if self.image_height > 0 { self.image_height } else { h }))
    }

    /// The bytes a `w` × `h` × `d` image reads, from the start of the client data
    pub fn extent(&self, [w, h, d]: [u32; 3], format: PixelFormat, ty: PixelType) -> u64 {
        if w == 0 || h == 0 || d == 0 {
            return 0;
        }
        let (row, image) = self.strides(w, h, format, ty);
        let pixel = u64::from(ty.pixel(format));
        let start = u64::from(self.skip_images) * image + u64::from(self.skip_rows) * row + u64::from(self.skip_pixels) * pixel;
        start + u64::from(d - 1) * image + u64::from(h - 1) * row + u64::from(w) * pixel
    }

    /// The tightly packed texels of a `w` × `h` × `d` image read from `data`
    fn read(&self, data: &[u8], [w, h, d]: [u32; 3], format: PixelFormat, ty: PixelType) -> Vec<u8> {
        let (row, image) = self.strides(w, h, format, ty);
        let pixel = u64::from(ty.pixel(format));
        let line = (u64::from(w) * pixel) as usize;
        let mut out = Vec::with_capacity(line * (h * d) as usize);
        for z in 0..u64::from(d) {
            for y in 0..u64::from(h) {
                let at = ((u64::from(self.skip_images) + z) * image + (u64::from(self.skip_rows) + y) * row + u64::from(self.skip_pixels) * pixel) as usize;
                out.extend_from_slice(&data[at..at + line]);
            }
        }
        out
    }
}

/// One level of one face: its size and texels (stored as its internal format's first table 3.2 row)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    pub size: [u32; 3],
    pub format: InternalFormat,
    pub texels: Vec<u8>,
}

impl Level {
    fn blank(size: [u32; 3], format: InternalFormat) -> Level {
        let n = u64::from(size[0]) * u64::from(size[1]) * u64::from(size[2]) * u64::from(format.bytes());
        Level { size, format, texels: vec![0; n as usize] }
    }
}

/// Client texels for glTexImage* / glTexSubImage*: their format, type and bytes (None: no data, a blank image)
#[derive(Clone, Copy, Debug)]
pub struct Pixels<'d> {
    pub format: PixelFormat,
    pub ty: PixelType,
    pub data: Option<&'d [u8]>,
}

/// A texture object: its target (set by its first bind), its faces' levels, its sampling state
#[derive(Clone, Debug)]
pub struct Texture {
    pub target: TextureTarget,
    pub faces: Vec<Vec<Option<Level>>>,
    /// glTexStorage*: the levels are fixed
    pub immutable: Option<u32>,
    pub sampling: Sampling,
}

impl Texture {
    pub fn new(target: TextureTarget) -> Texture {
        Texture { target, faces: vec![Vec::new(); target.faces()], immutable: None, sampling: Sampling::default() }
    }

    pub fn level(&self, face: usize, level: u32) -> Option<&Level> {
        self.faces.get(face)?.get(level as usize)?.as_ref()
    }

    /// The checks every glTexImage* / glTexStorage* size goes through (GL ES 3.0 §3.8.3)
    fn check_size(target: ImageTarget, level: u32, [w, h, d]: [u32; 3]) -> GlResult<()> {
        let t = target.texture();
        if level > t.max_level() {
            return Err(GlError::InvalidValue);
        }
        let max = t.max_size() >> level;
        let deep = match t {
            TextureTarget::Texture3D => max,
            TextureTarget::Texture2DArray => MAX_ARRAY_TEXTURE_LAYERS,
            _ => 1,
        };
        if w > max || h > max || d > deep {
            return Err(GlError::InvalidValue);
        }
        if t == TextureTarget::CubeMap && w != h {
            return Err(GlError::InvalidValue);
        }
        Ok(())
    }

    /// glTexImage2D / glTexImage3D: one level of one face, replaced (`border` must be 0)
    pub fn image(&mut self, target: ImageTarget, level: i32, internal: InternalFormat, size: [i32; 3], border: i32, pixels: Pixels<'_>, unpack: &Unpack) -> GlResult<Result<(), Error>> {
        let level = u32::try_from(level).map_err(|_| GlError::InvalidValue)?;
        let size = size.map(u32::try_from);
        let [Ok(w), Ok(h), Ok(d)] = size else { return Err(GlError::InvalidValue) };
        Texture::check_size(target, level, [w, h, d])?;
        if border != 0 {
            return Err(GlError::InvalidValue);
        }
        if !internal.takes(pixels.format, pixels.ty) {
            return Err(GlError::InvalidOperation);
        }
        if self.immutable.is_some() {
            return Err(GlError::InvalidOperation);
        }
        let mut lvl = Level::blank([w, h, d], internal);
        if let Some(data) = pixels.data {
            if let Err(e) = Texture::store(&mut lvl, [0; 3], [w, h, d], pixels, data, unpack) {
                return Ok(Err(e));
            }
        }
        let face = &mut self.faces[target.face()];
        if face.len() <= level as usize {
            face.resize(level as usize + 1, None);
        }
        face[level as usize] = Some(lvl);
        Ok(Ok(()))
    }

    /// glTexSubImage2D / glTexSubImage3D: a region of a level that exists
    pub fn sub_image(&mut self, target: ImageTarget, level: i32, offset: [i32; 3], size: [i32; 3], pixels: Pixels<'_>, unpack: &Unpack) -> GlResult<Result<(), Error>> {
        let level = u32::try_from(level).map_err(|_| GlError::InvalidValue)?;
        if level > target.texture().max_level() {
            return Err(GlError::InvalidValue);
        }
        let (Ok(offset), Ok(size)) = (offset.map(u32::try_from).into_iter().collect::<Result<Vec<_>, _>>(), size.map(u32::try_from).into_iter().collect::<Result<Vec<_>, _>>()) else { return Err(GlError::InvalidValue) };
        let lvl = self.faces[target.face()].get_mut(level as usize).and_then(Option::as_mut).ok_or(GlError::InvalidOperation)?;
        let fits = (0..3).all(|i| u64::from(offset[i]) + u64::from(size[i]) <= u64::from(lvl.size[i]));
        if !fits {
            return Err(GlError::InvalidValue);
        }
        let format_ok = match lvl.format.sized() {
            true => lvl.format.takes(pixels.format, pixels.ty),
            false => InternalFormat::unsized_of(pixels.format) == Some(lvl.format) && lvl.format.takes(pixels.format, pixels.ty),
        };
        if !format_ok {
            return Err(GlError::InvalidOperation);
        }
        let Some(data) = pixels.data else { return Ok(Ok(())) };
        Ok(Texture::store(lvl, [offset[0], offset[1], offset[2]], [size[0], size[1], size[2]], pixels, data, unpack))
    }

    /// glTexStorage2D / glTexStorage3D: every level, fixed (GL ES 3.0 §3.8.4)
    pub fn storage(&mut self, target: TextureTarget, levels: i32, internal: InternalFormat, size: [i32; 3]) -> GlResult<()> {
        if !internal.sized() {
            return Err(GlError::InvalidEnum);
        }
        let (Ok(levels), [Ok(w), Ok(h), Ok(d)]) = (u32::try_from(levels), size.map(u32::try_from)) else { return Err(GlError::InvalidValue) };
        if w < 1 || h < 1 || d < 1 || levels < 1 {
            return Err(GlError::InvalidValue);
        }
        let largest = if target == TextureTarget::Texture3D { w.max(h).max(d) } else { w.max(h) };
        if levels > target.max_level() + 1 || levels > largest.ilog2() + 1 {
            return Err(GlError::InvalidOperation);
        }
        if self.immutable.is_some() {
            return Err(GlError::InvalidOperation);
        }
        let image = match target {
            TextureTarget::CubeMap => ImageTarget::CubeFace(0),
            TextureTarget::Texture2D => ImageTarget::Texture2D,
            TextureTarget::Texture3D => ImageTarget::Texture3D,
            TextureTarget::Texture2DArray => ImageTarget::Texture2DArray,
        };
        Texture::check_size(image, 0, [w, h, d])?;
        let layered = target == TextureTarget::Texture2DArray;
        for face in &mut self.faces {
            *face = (0..levels).map(|l| Some(Level::blank([(w >> l).max(1), (h >> l).max(1), if layered { d } else { (d >> l).max(1) }], internal))).collect();
        }
        self.immutable = Some(levels);
        Ok(())
    }

    /// Client texels into a level's region: only in the format and type the level is stored as, for now
    fn store(lvl: &mut Level, [x, y, z]: [u32; 3], size: [u32; 3], pixels: Pixels<'_>, data: &[u8], unpack: &Unpack) -> Result<(), Error> {
        if lvl.format.stored() != Some((pixels.format, pixels.ty)) {
            return Err(Error(format!("texels of {:?} / {:?} into {:?}: detour does not convert texels yet", pixels.format, pixels.ty, lvl.format)));
        }
        if unpack.extent(size, pixels.format, pixels.ty) > data.len() as u64 {
            return Err(Error(format!("{} bytes of texels for an image that reads {}", data.len(), unpack.extent(size, pixels.format, pixels.ty))));
        }
        let packed = unpack.read(data, size, pixels.format, pixels.ty);
        let texel = lvl.format.bytes() as usize;
        let [w, h, _] = lvl.size.map(|v| v as usize);
        let line = size[0] as usize * texel;
        for (k, src) in packed.chunks(line.max(1)).enumerate() {
            let (row, slice) = (k % size[1] as usize, k / size[1] as usize);
            let at = (((z as usize + slice) * h + y as usize + row) * w + x as usize) * texel;
            lvl.texels[at..at + line].copy_from_slice(src);
        }
        Ok(())
    }
}
