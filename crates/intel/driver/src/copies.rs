//! ⭐ WebGPU's copies on Xe3 (`copyBufferToBuffer`, `copyBufferToTexture`, `copyTextureToBuffer`,
//! `copyTextureToTexture`): each a rectangle of rows copied by a compute kernel (`copies/copy.mlir`), a dword an
//! invocation when every address, pitch and row is 4-aligned, else a byte; textures linear (level 0 of a plain 2D one,
//! or a single-sampled target).

use crate::base::{Gpu, Host};

use crate::xe3::{Buffer, Codegen, Context, Error, Image2d, Kernel, Recorder, Target};

const COPY: &str = include_str!("copies/copy.mlir");

/// Where a copy reads or writes texels: level 0 of a linear 2D texture (its address, row pitch in bytes, bytes a texel,
/// size in texels).
#[derive(Clone, Copy, Debug)]
pub struct Texels {
    pub addr: u64,
    pub pitch: u32,
    pub bpp: u32,
    pub size: [u32; 2],
}

impl<'d, G: Gpu<Cmd = u64> + 'd> From<&Image2d<'d, G>> for Texels {
    fn from(i: &Image2d<'d, G>) -> Self {
        let bpp = i.texture.format.bytes();
        Texels { addr: i.addr, pitch: i.texture.width * bpp, bpp, size: [i.texture.width, i.texture.height] }
    }
}

impl<'d, G: Gpu<Cmd = u64> + 'd> TryFrom<&Target<'d, G>> for Texels {
    type Error = Error;

    fn try_from(t: &Target<'d, G>) -> Result<Self, Error> {
        if t.texture.samples > 1 {
            return Err(Error::msg("copies: a multisampled target (resolve it first)"));
        }
        let bpp = t.texture.format.bytes();
        Ok(Texels { addr: t.addr, pitch: t.texture.width * bpp, bpp, size: [t.texture.width, t.texture.height] })
    }
}

impl Texels {
    /// The address of texel `[x, y]` once `[w, h]` from it is checked to be inside.
    fn at(&self, [x, y]: [u32; 2], [w, h]: [u32; 2]) -> Result<u64, Error> {
        if x + w > self.size[0] || y + h > self.size[1] {
            return Err(Error::msg(format!("copies: {w} x {h} at ({x}, {y}) past a {:?} texture", self.size)));
        }
        Ok(self.addr + u64::from(y) * u64::from(self.pitch) + u64::from(x * self.bpp))
    }
}

/// A buffer's part a copy reads or writes as texels (`GPUTexelCopyBufferInfo`): its byte `offset`, `bytes_per_row`.
pub struct BufferTexels<'b, 'd, G: Gpu<Cmd = u64> + 'd> {
    pub buffer: &'b Buffer<'d, G>,
    pub offset: u64,
    pub bytes_per_row: u32,
}

impl<'d, G: Gpu<Cmd = u64> + 'd> BufferTexels<'_, 'd, G> {
    /// Its address once `rows` rows of `row` bytes are checked to be inside.
    fn at(&self, row: u32, rows: u32) -> Result<u64, Error> {
        let need = self.offset + u64::from(self.bytes_per_row) * u64::from(rows.saturating_sub(1)) + u64::from(row);
        if rows > 0 && (self.bytes_per_row < row || need > self.buffer.size) {
            return Err(Error::msg(format!("copies: {rows} rows of {row} bytes ({} a row) from {} of a {}-byte buffer", self.bytes_per_row, self.offset, self.buffer.size)));
        }
        Ok(self.buffer.addr + self.offset)
    }
}

/// ⭐ The copy kernels, compiled once.
pub struct Copies {
    words: Kernel,
    bytes: Kernel,
}

impl Copies {
    pub fn new<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(cx: &mut Context<'d, G, H>, codegen: &Codegen<'_>) -> Result<Self, Error> {
        Ok(Copies { words: cx.kernel(codegen, COPY, "copy_words")?, bytes: cx.kernel(codegen, COPY, "copy_bytes")? })
    }

    /// `rows` rows of `row` bytes from `src` (rows `ss` bytes apart) to `dst` (`ds` apart), recorded into `r`.
    fn rect<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, (src, ss): (u64, u32), (dst, ds): (u64, u32), row: u32, rows: u32) -> Result<(), Error> {
        if row == 0 || rows == 0 {
            return Ok(());
        }
        if row > 0xffff * 4 || rows > 0xffff {
            return Err(Error::msg(format!("copies: {rows} rows of {row} bytes (not yet: split it)")));
        }
        let aligned = [src, dst, u64::from(ss), u64::from(ds), u64::from(row)].iter().all(|v| v % 4 == 0);
        let (kernel, units) = if aligned { (&self.words, row / 4) } else { (&self.bytes, row) };
        if units > 0xffff {
            return Err(Error::msg(format!("copies: a row of {row} bytes unaligned (not yet: split it)")));
        }
        r.dispatch(kernel, [units.div_ceil(16), rows], &[src as u32, ss, dst as u32, ds, units | rows << 16])
    }

    /// ⭐ `copyBufferToBuffer`: `size` bytes from byte `src.1` of `src.0` to byte `dst.1` of `dst.0`.
    pub fn buffer_to_buffer<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, (src, so): (&Buffer<'d, G>, u64), (dst, d): (&Buffer<'d, G>, u64), size: u64) -> Result<(), Error> {
        if so + size > src.size || d + size > dst.size {
            return Err(Error::msg(format!("copies: {size} bytes from {so} of {} to {d} of {}", src.size, dst.size)));
        }
        // rows of at most 64 KiB (a row's units fit 16 bits)
        let row = 1u64 << 16;
        let (whole, rest) = (size / row, size % row);
        let addr = |b: &Buffer<'d, G>, o: u64| b.addr + o;
        if whole > 0 {
            self.rect(r, (addr(src, so), row as u32), (addr(dst, d), row as u32), row as u32, whole as u32)?;
        }
        self.rect(r, (addr(src, so + whole * row), 0), (addr(dst, d + whole * row), 0), rest as u32, 1)
    }

    /// ⭐ `copyBufferToTexture`: `[w, h]` texels from `src` into `dst` at `origin`.
    pub fn buffer_to_texture<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, src: BufferTexels<'_, 'd, G>, dst: Texels, origin: [u32; 2], [w, h]: [u32; 2]) -> Result<(), Error> {
        let row = w * dst.bpp;
        self.rect(r, (src.at(row, h)?, src.bytes_per_row), (dst.at(origin, [w, h])?, dst.pitch), row, h)
    }

    /// ⭐ `copyTextureToBuffer`: `[w, h]` texels of `src` at `origin` into `dst`.
    pub fn texture_to_buffer<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, src: Texels, origin: [u32; 2], dst: BufferTexels<'_, 'd, G>, [w, h]: [u32; 2]) -> Result<(), Error> {
        let row = w * src.bpp;
        self.rect(r, (src.at(origin, [w, h])?, src.pitch), (dst.at(row, h)?, dst.bytes_per_row), row, h)
    }

    /// ⭐ `copyTextureToTexture`: `[w, h]` texels of `src` at `from` into `dst` at `to` (texels of the same size).
    pub fn texture_to_texture<'d, G: Gpu<Cmd = u64> + 'd, H: Host>(&self, r: &mut Recorder<'_, 'd, G, H>, (src, from): (Texels, [u32; 2]), (dst, to): (Texels, [u32; 2]), [w, h]: [u32; 2]) -> Result<(), Error> {
        if src.bpp != dst.bpp {
            return Err(Error::msg(format!("copies: texels of {} bytes into texels of {}", src.bpp, dst.bpp)));
        }
        self.rect(r, (src.at(from, [w, h])?, src.pitch), (dst.at(to, [w, h])?, dst.pitch), w * src.bpp, h)
    }
}
