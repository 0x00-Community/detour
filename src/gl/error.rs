/// GL's error flag: the first error recorded stays until `Context::error` reads it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlError {
    InvalidEnum,
    InvalidValue,
    InvalidOperation,
    InvalidFramebufferOperation,
    OutOfMemory,
}

impl GlError {
    /// The GLenum glGetError returns
    pub const fn code(self) -> u32 {
        match self {
            GlError::InvalidEnum => GL_INVALID_ENUM,
            GlError::InvalidValue => GL_INVALID_VALUE,
            GlError::InvalidOperation => GL_INVALID_OPERATION,
            GlError::InvalidFramebufferOperation => GL_INVALID_FRAMEBUFFER_OPERATION,
            GlError::OutOfMemory => GL_OUT_OF_MEMORY,
        }
    }
}

const GL_INVALID_ENUM: u32 = 0x0500;
const GL_INVALID_VALUE: u32 = 0x0501;
const GL_INVALID_OPERATION: u32 = 0x0502;
const GL_OUT_OF_MEMORY: u32 = 0x0505;
const GL_INVALID_FRAMEBUFFER_OPERATION: u32 = 0x0506;

/// What a GL entry point returns inside detour: an error is recorded on the context, never raised
pub type GlResult<T> = Result<T, GlError>;
