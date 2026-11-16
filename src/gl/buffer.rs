use super::{GlError, GlResult};

/// glBindBuffer's targets (GL ES 3.0)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BufferTarget {
    Array,
    ElementArray,
    Uniform,
    CopyRead,
    CopyWrite,
    PixelPack,
    PixelUnpack,
    TransformFeedback,
}

/// glBufferData's usage hint: kept for glGetBufferParameteriv, it does not change where detour places the buffer
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Usage {
    StreamDraw,
    StreamRead,
    StreamCopy,
    #[default]
    StaticDraw,
    StaticRead,
    StaticCopy,
    DynamicDraw,
    DynamicRead,
    DynamicCopy,
}

/// A buffer object's store: the bytes the next draw uploads
#[derive(Default)]
pub struct Buffer {
    pub data: Vec<u8>,
    pub usage: Usage,
    /// bumped on every write, so a draw knows what it uploaded is stale
    pub generation: u64,
}

impl Buffer {
    /// glBufferData: a new store of `size` bytes, from `data` when given
    pub fn store(&mut self, size: usize, data: Option<&[u8]>, usage: Usage) -> GlResult<()> {
        if data.is_some_and(|d| d.len() < size) {
            return Err(GlError::InvalidValue);
        }
        self.data = data.map_or_else(|| vec![0; size], |d| d[..size].to_vec());
        self.usage = usage;
        self.generation += 1;
        Ok(())
    }

    /// glBufferSubData
    pub fn write(&mut self, offset: usize, data: &[u8]) -> GlResult<()> {
        let end = offset.checked_add(data.len()).filter(|&e| e <= self.data.len()).ok_or(GlError::InvalidValue)?;
        self.data[offset..end].copy_from_slice(data);
        self.generation += 1;
        Ok(())
    }

    /// glCopyBufferSubData: this buffer's range from `src` (which may be this buffer)
    pub fn copy(&mut self, src: Option<&Buffer>, read: usize, write: usize, size: usize) -> GlResult<()> {
        let from = src.map_or(&self.data, |b| &b.data);
        let in_range = |at: usize, len: usize| at.checked_add(size).is_some_and(|e| e <= len);
        if !in_range(read, from.len()) || !in_range(write, self.data.len()) {
            return Err(GlError::InvalidValue);
        }
        if src.is_none() && read.max(write) - read.min(write) < size {
            return Err(GlError::InvalidValue);
        }
        let bytes = from[read..read + size].to_vec();
        self.data[write..write + size].copy_from_slice(&bytes);
        self.generation += 1;
        Ok(())
    }
}
