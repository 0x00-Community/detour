use std::collections::BTreeMap;

use mlirformat::Error;

use crate::{BlendEquation, BlendFactor};

use super::{GlslPipelines, Attrib, Buffer, BufferTarget, Capability, GlError, GlResult, Linked, Names, Program, Shader, ShaderType, State, Usage, UniformValue, VertexArray};
use super::{ColorBuffer, DefaultFramebuffer, Framebuffer, MAX_COMBINED_TEXTURE_IMAGE_UNITS, Renderbuffer, Texture, TextureTarget, Unpack};
use crate::ColorFormat;

use crate::Toolchain;

/// Shaders and programs share one name space
enum Object<'t> {
    Shader(Shader),
    Program(Program<'t>),
}

/// A GL ES 3.0 context: the objects, the bindings and the state; draws are recorded from it
pub struct Context<'t> {
    tools: &'t Toolchain<'t>,
    glsl: GlslPipelines,
    error: Option<GlError>,
    pub buffers: Names<Buffer>,
    pub vertex_arrays: Names<VertexArray>,
    objects: Names<Object<'t>>,
    /// glBindBuffer's bindings, by target (0: none)
    pub bound: BTreeMap<BufferTarget, u32>,
    pub vertex_array: u32,
    /// vertex array 0, the default one
    default_vertex_array: VertexArray,
    pub program: u32,
    pub state: State,
    /// the default framebuffer: its pixels live here between draws
    pub framebuffer: DefaultFramebuffer,
    pub textures: Names<Texture>,
    /// each texture unit's bindings, by target
    pub units: Vec<BTreeMap<TextureTarget, u32>>,
    /// glActiveTexture's unit
    pub active_unit: u32,
    pub renderbuffers: Names<Renderbuffer>,
    pub renderbuffer: u32,
    pub framebuffers: Names<Framebuffer>,
    pub draw_framebuffer: u32,
    pub read_framebuffer: u32,
    /// the default framebuffer's draw and read buffers (BACK or NONE)
    pub default_draw: ColorBuffer,
    pub default_read: ColorBuffer,
    pub unpack: Unpack,
    /// glReadPixels' layout (the unpack state's pack fields)
    pub pack: Unpack,
}

impl<'t> Context<'t> {
    /// A context whose default framebuffer is `width` × `height` pixels of `format`
    pub fn new(tools: &'t Toolchain<'t>, width: u32, height: u32, format: ColorFormat) -> Result<Self, Error> {
        Ok(Context {
            tools,
            glsl: GlslPipelines::new(tools)?,
            error: None,
            buffers: Names::default(),
            vertex_arrays: Names::default(),
            objects: Names::default(),
            bound: BTreeMap::new(),
            vertex_array: 0,
            default_vertex_array: VertexArray::default(),
            program: 0,
            state: State::new(width, height),
            framebuffer: DefaultFramebuffer::new(width, height, format),
            textures: Names::default(),
            units: vec![BTreeMap::new(); MAX_COMBINED_TEXTURE_IMAGE_UNITS as usize],
            active_unit: 0,
            renderbuffers: Names::default(),
            renderbuffer: 0,
            framebuffers: Names::default(),
            draw_framebuffer: 0,
            read_framebuffer: 0,
            default_draw: ColorBuffer::Back,
            default_read: ColorBuffer::Back,
            unpack: Unpack::default(),
            pack: Unpack::default(),
        })
    }

    /// glGetError: the first error since the last call, then none
    pub fn error(&mut self) -> Option<GlError> {
        self.error.take()
    }

    pub(super) fn record_error(&mut self, e: GlError) {
        self.error.get_or_insert(e);
    }

    /// A linked program, read-only
    pub(super) fn objects_program(&self, name: u32) -> Option<&Linked<'t>> {
        match self.objects.get(name) {
            Some(Object::Program(p)) => p.linked.as_ref(),
            _ => None,
        }
    }

    /// An entry point's outcome: an error is recorded (the first one stays), the call does nothing
    pub(super) fn record<T: Default>(&mut self, r: GlResult<T>) -> T {
        r.unwrap_or_else(|e| {
            self.error.get_or_insert(e);
            T::default()
        })
    }

    pub fn gen_buffers(&mut self, n: usize) -> Vec<u32> {
        self.buffers.reserve_n(n)
    }

    pub fn bind_buffer(&mut self, target: BufferTarget, name: u32) {
        if name != 0 {
            self.buffers.bind_any(name, Buffer::default);
        }
        {
            self.bound.insert(target, name);
            if target == BufferTarget::ElementArray {
                self.vertex_array_mut().elements = name;
            }
        }
    }

    fn bound_buffer(&mut self, target: BufferTarget) -> GlResult<&mut Buffer> {
        let name = match target {
            BufferTarget::ElementArray => self.vertex_array().elements,
            t => self.bound.get(&t).copied().unwrap_or(0),
        };
        self.buffers.get_mut(name).ok_or(GlError::InvalidOperation)
    }

    pub fn buffer_data(&mut self, target: BufferTarget, size: usize, data: Option<&[u8]>, usage: Usage) {
        let r = self.bound_buffer(target).and_then(|b| b.store(size, data, usage));
        self.record(r)
    }

    pub fn buffer_sub_data(&mut self, target: BufferTarget, offset: usize, data: &[u8]) {
        let r = self.bound_buffer(target).and_then(|b| b.write(offset, data));
        self.record(r)
    }

    /// glDeleteBuffers: a deleted buffer is unbound from the context and the current vertex array
    pub fn delete_buffers(&mut self, names: &[u32]) {
        for &n in names.iter().filter(|&&n| n != 0) {
            self.buffers.delete(n);
            self.bound.values_mut().filter(|b| **b == n).for_each(|b| *b = 0);
            let vao = self.vertex_array_mut();
            if vao.elements == n {
                vao.elements = 0;
            }
            vao.attribs.iter_mut().filter(|a| a.buffer == n).for_each(|a| a.buffer = 0);
        }
    }

    pub fn is_buffer(&self, name: u32) -> bool {
        self.buffers.is(name)
    }

    pub fn gen_vertex_arrays(&mut self, n: usize) -> Vec<u32> {
        self.vertex_arrays.reserve_n(n)
    }

    pub fn bind_vertex_array(&mut self, name: u32) {
        let r = if name == 0 { Ok(()) } else { self.vertex_arrays.bind(name, VertexArray::default) };
        if self.record(r.map(|()| true)) {
            self.vertex_array = name;
        }
    }

    pub fn delete_vertex_arrays(&mut self, names: &[u32]) {
        for &n in names.iter().filter(|&&n| n != 0) {
            self.vertex_arrays.delete(n);
            if self.vertex_array == n {
                self.vertex_array = 0;
            }
        }
    }

    pub fn vertex_array(&self) -> &VertexArray {
        self.vertex_arrays.get(self.vertex_array).unwrap_or(&self.default_vertex_array)
    }

    fn vertex_array_mut(&mut self) -> &mut VertexArray {
        match self.vertex_arrays.get_mut(self.vertex_array) {
            Some(v) => v,
            None => &mut self.default_vertex_array,
        }
    }

    /// glVertexAttribPointer / glVertexAttribIPointer: the array buffer bound now is the attribute's
    pub fn vertex_attrib_pointer(&mut self, index: u32, attrib: Attrib) {
        let (buffer, vao) = (self.bound.get(&BufferTarget::Array).copied().unwrap_or(0), self.vertex_array);
        let r = self.vertex_array_mut().pointer(index, attrib, buffer, vao);
        self.record(r)
    }

    pub fn enable_vertex_attrib_array(&mut self, index: u32, on: bool) {
        let r = self.vertex_array_mut().enable(index, on);
        self.record(r)
    }

    pub fn vertex_attrib_divisor(&mut self, index: u32, divisor: u32) {
        let r = self.vertex_array_mut().divisor(index, divisor);
        self.record(r)
    }

    pub fn create_shader(&mut self, ty: ShaderType) -> u32 {
        self.objects.create(Object::Shader(Shader::new(ty)))
    }

    /// A shader name: a program name is INVALID_OPERATION, an unknown one INVALID_VALUE
    fn shader(&mut self, name: u32) -> GlResult<&mut Shader> {
        match self.objects.get_mut(name) {
            Some(Object::Shader(s)) => Ok(s),
            Some(Object::Program(_)) => Err(GlError::InvalidOperation),
            None => Err(GlError::InvalidValue),
        }
    }

    fn program_of(&mut self, name: u32) -> GlResult<&mut Program<'t>> {
        match self.objects.get_mut(name) {
            Some(Object::Program(p)) => Ok(p),
            Some(Object::Shader(_)) => Err(GlError::InvalidOperation),
            None => Err(GlError::InvalidValue),
        }
    }

    pub fn shader_source(&mut self, name: u32, source: &str) {
        let r = self.shader(name).map(|s| s.source = source.to_owned());
        self.record(r)
    }

    pub fn compile_shader(&mut self, name: u32) {
        let tools = self.tools;
        let r = self.shader(name).map(|s| s.compile(tools));
        self.record(r)
    }

    pub fn shader_compiled(&mut self, name: u32) -> bool {
        let r = self.shader(name).map(|s| s.compiled);
        self.record(r)
    }

    pub fn shader_log(&mut self, name: u32) -> String {
        let r = self.shader(name).map(|s| s.log.clone());
        self.record(r)
    }

    /// glDeleteShader: an attached shader goes when its last program lets it go
    pub fn delete_shader(&mut self, name: u32) {
        if name == 0 {
            return;
        }
        let attached = self.attached(name);
        let r = self.shader(name).map(|s| s.deleted = true);
        self.record(r);
        if !attached && self.error.is_none() {
            self.objects.delete(name);
        }
    }

    fn attached(&self, shader: u32) -> bool {
        self.objects.iter().any(|(_, o)| matches!(o, Object::Program(p) if p.shaders.contains(&shader)))
    }

    pub fn create_program(&mut self) -> u32 {
        self.objects.create(Object::Program(Program::default()))
    }

    pub fn attach_shader(&mut self, program: u32, shader: u32) {
        let r = self.shader(shader).map(|_| ()).and_then(|()| self.program_of(program)).and_then(|p| {
            if p.shaders.contains(&shader) {
                return Err(GlError::InvalidOperation);
            }
            p.shaders.push(shader);
            Ok(())
        });
        self.record(r)
    }

    pub fn detach_shader(&mut self, program: u32, shader: u32) {
        let r = self.program_of(program).and_then(|p| {
            let at = p.shaders.iter().position(|&s| s == shader).ok_or(GlError::InvalidOperation)?;
            p.shaders.remove(at);
            Ok(())
        });
        self.record(r);
        if matches!(self.objects.get(shader), Some(Object::Shader(s)) if s.deleted) && !self.attached(shader) {
            self.objects.delete(shader);
        }
    }

    pub fn link_program(&mut self, name: u32) {
        let attached = match self.program_of(name) {
            Ok(p) => p.shaders.clone(),
            Err(e) => return self.record(Err(e)),
        };
        let shaders: Vec<Shader> = attached.iter().filter_map(|&s| match self.objects.get(s) {
            Some(Object::Shader(shader)) => Some(shader.clone()),
            _ => None,
        }).collect();
        let shaders: Vec<&Shader> = shaders.iter().collect();
        if let Some(Object::Program(p)) = self.objects.get_mut(name) {
            p.link(self.tools, &self.glsl, &shaders);
        }
    }

    pub fn link_status(&mut self, name: u32) -> bool {
        let r = self.program_of(name).map(|p| p.linked.is_some() && p.log.is_empty());
        self.record(r)
    }

    pub fn program_log(&mut self, name: u32) -> String {
        let r = self.program_of(name).map(|p| p.log.clone());
        self.record(r)
    }

    pub fn use_program(&mut self, name: u32) {
        let r = if name == 0 { Ok(()) } else { self.program_of(name).and_then(|p| p.linked.as_ref().map(|_| ()).ok_or(GlError::InvalidOperation)) };
        if self.record(r.map(|()| true)) {
            self.program = name;
        }
    }

    pub fn linked(&mut self, name: u32) -> GlResult<&mut Linked<'t>> {
        self.program_of(name)?.linked.as_mut().ok_or(GlError::InvalidOperation)
    }

    /// glGetAttribLocation: -1 for none
    pub fn attrib_location(&mut self, program: u32, name: &str) -> i32 {
        let r = self.linked(program).map(|l| l.attribute(name).map_or(-1, |at| at as i32));
        self.record(r.map(Some)).unwrap_or(-1)
    }

    /// glGetUniformLocation: -1 for none
    pub fn uniform_location(&mut self, program: u32, name: &str) -> i32 {
        let r = self.linked(program).map(|l| l.uniform(name).map_or(-1, |at| at as i32));
        self.record(r.map(Some)).unwrap_or(-1)
    }

    /// glUniform* on the current program
    pub fn uniform(&mut self, location: i32, value: UniformValue<'_>) {
        let program = self.program;
        let r = if program == 0 { Err(GlError::InvalidOperation) } else { self.linked(program).and_then(|l| l.set(location, value)) };
        self.record(r)
    }

    pub fn enable(&mut self, cap: Capability, on: bool) {
        *self.state.flag(cap) = on;
    }

    pub fn is_enabled(&mut self, cap: Capability) -> bool {
        *self.state.flag(cap)
    }

    pub fn viewport(&mut self, x: i32, y: i32, width: i32, height: i32) {
        let r = match (u32::try_from(width), u32::try_from(height)) {
            (Ok(width), Ok(height)) => Ok(self.state.viewport = super::Rect { x, y, width, height }),
            _ => Err(GlError::InvalidValue),
        };
        self.record(r)
    }

    pub fn scissor(&mut self, x: i32, y: i32, width: i32, height: i32) {
        let r = match (u32::try_from(width), u32::try_from(height)) {
            (Ok(width), Ok(height)) => Ok(self.state.scissor = super::Rect { x, y, width, height }),
            _ => Err(GlError::InvalidValue),
        };
        self.record(r)
    }

    pub fn clear_color(&mut self, rgba: [f32; 4]) {
        self.state.clear_color = rgba.map(|c| c.clamp(0.0, 1.0));
    }

    /// glBlendFunc
    pub fn blend_func(&mut self, src: BlendFactor, dst: BlendFactor) {
        self.blend_func_separate(src, dst, src, dst);
    }

    /// glBlendFuncSeparate
    pub fn blend_func_separate(&mut self, src_rgb: BlendFactor, dst_rgb: BlendFactor, src_alpha: BlendFactor, dst_alpha: BlendFactor) {
        self.state.blending.src = [src_rgb, src_alpha];
        self.state.blending.dst = [dst_rgb, dst_alpha];
    }

    /// glBlendEquation
    pub fn blend_equation(&mut self, mode: BlendEquation) {
        self.blend_equation_separate(mode, mode);
    }

    /// glBlendEquationSeparate
    pub fn blend_equation_separate(&mut self, rgb: BlendEquation, alpha: BlendEquation) {
        self.state.blending.equation = [rgb, alpha];
    }

    /// glBlendColor
    pub fn blend_color(&mut self, rgba: [f32; 4]) {
        self.state.blending.color = rgba.map(|c| c.clamp(0.0, 1.0));
    }

    /// glColorMask
    pub fn color_mask(&mut self, red: bool, green: bool, blue: bool, alpha: bool) {
        self.state.color_mask = [red, green, blue, alpha];
    }
}
