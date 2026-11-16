use super::{GlError, GlResult};

/// The GL ES 3.0 minimum (MAX_VERTEX_ATTRIBS) detour advertises
pub const MAX_VERTEX_ATTRIBS: usize = 16;

/// glVertexAttribPointer's component type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AttribType {
    Byte,
    UnsignedByte,
    Short,
    UnsignedShort,
    Int,
    UnsignedInt,
    HalfFloat,
    #[default]
    Float,
    Fixed,
    Int2101010Rev,
    UnsignedInt2101010Rev,
}

impl AttribType {
    pub const fn size(self) -> u32 {
        match self {
            AttribType::Byte | AttribType::UnsignedByte => 1,
            AttribType::Short | AttribType::UnsignedShort | AttribType::HalfFloat => 2,
            _ => 4,
        }
    }

    const fn packed(self) -> bool {
        matches!(self, AttribType::Int2101010Rev | AttribType::UnsignedInt2101010Rev)
    }

    const fn integer(self) -> bool {
        matches!(self, AttribType::Byte | AttribType::UnsignedByte | AttribType::Short | AttribType::UnsignedShort | AttribType::Int | AttribType::UnsignedInt)
    }
}

/// One generic vertex attribute's array: glVertexAttrib{I}Pointer + glVertexAttribDivisor + glEnableVertexAttribArray
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attrib {
    pub enabled: bool,
    pub size: u32,
    pub ty: AttribType,
    pub normalized: bool,
    /// glVertexAttribIPointer: read as integers
    pub integer: bool,
    pub stride: u32,
    /// the bound array buffer's name when the pointer was set, and the offset into it
    pub buffer: u32,
    pub offset: usize,
    pub divisor: u32,
}

impl Default for Attrib {
    fn default() -> Self {
        Attrib { enabled: false, size: 4, ty: AttribType::Float, normalized: false, integer: false, stride: 0, buffer: 0, offset: 0, divisor: 0 }
    }
}

impl Attrib {
    /// The bytes between two vertices: a stride of 0 is tightly packed
    pub const fn step(&self) -> u32 {
        if self.stride == 0 { self.size * self.ty.size() } else { self.stride }
    }
}

/// A vertex array object: the attribute arrays and the element array buffer
#[derive(Clone, Debug, Default)]
pub struct VertexArray {
    pub attribs: [Attrib; MAX_VERTEX_ATTRIBS],
    pub elements: u32,
}

impl VertexArray {
    /// glVertexAttribPointer / glVertexAttribIPointer with the array buffer `buffer` bound (GL ES 3.0 §2.8)
    pub fn pointer(&mut self, index: u32, mut attrib: Attrib, buffer: u32, vao: u32) -> GlResult<()> {
        let at = self.attribs.get_mut(index as usize).ok_or(GlError::InvalidValue)?;
        let sizes = if attrib.ty.packed() { 4..=4 } else { 1..=4 };
        if !sizes.contains(&attrib.size) || attrib.stride > i32::MAX as u32 {
            return Err(GlError::InvalidValue);
        }
        if attrib.integer && !attrib.ty.integer() {
            return Err(GlError::InvalidEnum);
        }
        if buffer == 0 && vao != 0 && attrib.offset != 0 {
            return Err(GlError::InvalidOperation);
        }
        attrib.enabled = at.enabled;
        attrib.divisor = at.divisor;
        attrib.buffer = buffer;
        *at = attrib;
        Ok(())
    }

    pub fn enable(&mut self, index: u32, on: bool) -> GlResult<()> {
        self.attribs.get_mut(index as usize).ok_or(GlError::InvalidValue)?.enabled = on;
        Ok(())
    }

    pub fn divisor(&mut self, index: u32, divisor: u32) -> GlResult<()> {
        self.attribs.get_mut(index as usize).ok_or(GlError::InvalidValue)?.divisor = divisor;
        Ok(())
    }
}
