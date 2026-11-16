use std::collections::BTreeMap;

use mlirformat::spirv_types::SpirvType;
use mlirformat::{Attribute, Dim, Error, FloatKind, Op, ShapedKind, Signedness, Type};
use mlirpass::{Body, GlslGlobal, GlslReader, GlslStage, Heads, OpId, PassManager, Storage as GlslStorage, Symbol};

use super::{GlError, GlResult};
use crate::{Compiled, Toolchain};

/// glCreateShader's type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

impl From<ShaderType> for GlslStage {
    fn from(t: ShaderType) -> Self {
        match t {
            ShaderType::Vertex => GlslStage::Vertex,
            ShaderType::Fragment => GlslStage::Fragment,
        }
    }
}

#[derive(Clone)]
pub struct Shader {
    pub ty: ShaderType,
    pub source: String,
    pub compiled: bool,
    pub log: String,
    /// glDeleteShader on an attached shader only flags it
    pub deleted: bool,
}

impl Shader {
    pub fn new(ty: ShaderType) -> Self {
        Shader { ty, source: String::new(), compiled: false, log: String::new(), deleted: false }
    }

    /// glCompileShader: GLSL ES read into the glsl dialect; the program links the stages again from source
    pub fn compile(&mut self, tools: &Toolchain<'_>) {
        let mut body = tools.body();
        let read = GlslReader::read(&mut body, &self.source, self.ty.into());
        self.compiled = read.is_ok();
        self.log = read.err().map(|e| e.0).unwrap_or_default();
    }
}

/// The scalar a uniform's glUniform* call must match
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    Float,
    Int,
    Uint,
    Bool,
    /// a sampler: glUniform1i sets its texture unit
    Sampler,
}

/// A uniform's shape: its scalar, components per element (a vector's size, a matrix's columns × rows) and elements
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformType {
    pub scalar: Scalar,
    pub columns: u32,
    pub rows: u32,
    pub count: u32,
}

impl UniformType {
    pub const fn components(self) -> u32 {
        self.columns * self.rows
    }

    pub const fn matrix(self) -> bool {
        self.columns > 1
    }
}

impl TryFrom<&Type<'_>> for UniformType {
    type Error = Error;

    fn try_from(t: &Type<'_>) -> Result<Self, Error> {
        let scalar = |t: &Type<'_>| match t {
            Type::Float(FloatKind::F32) => Ok(Scalar::Float),
            Type::Integer { width: 1, .. } => Ok(Scalar::Bool),
            Type::Integer { signedness: Signedness::Unsigned, width: 32 } => Ok(Scalar::Uint),
            Type::Integer { width: 32, .. } => Ok(Scalar::Int),
            t => Err(Error(format!("{t:?}: not a GLSL scalar"))),
        };
        let vector = |t: &Type<'_>| match t {
            Type::Shaped(s) if s.kind == ShapedKind::Vector => match s.shape.as_deref() {
                Some([Dim::Static(n)]) => Ok((scalar(&s.elem)?, *n as u32)),
                _ => Err(Error(format!("{t:?}: not a GLSL vector"))),
            },
            t => Ok((scalar(t)?, 1)),
        };
        match SpirvType::try_from(t) {
            Ok(SpirvType::Array { len, elem, .. }) => Ok(UniformType { count: len, ..UniformType::try_from(&*elem)? }),
            Ok(SpirvType::Matrix { cols, column }) => {
                let (scalar, rows) = vector(&column)?;
                Ok(UniformType { scalar, columns: cols, rows, count: 1 })
            }
            Ok(SpirvType::SampledImage(_) | SpirvType::Image { .. } | SpirvType::Sampler) => Ok(UniformType { scalar: Scalar::Sampler, columns: 1, rows: 1, count: 1 }),
            Ok(other) => Err(Error(format!("{other:?}: not a GLSL uniform"))),
            Err(_) => vector(t).map(|(scalar, rows)| UniformType { scalar, columns: 1, rows, count: 1 }),
        }
    }
}

/// Where a SPIR-V global lives, as convert-glsl-to-spirv declares a GLSL global
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Input,
    Output,
    /// a loose uniform or a sampler (the default uniform block)
    UniformConstant,
    /// a uniform block
    Uniform,
}

impl TryFrom<&str> for Storage {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self, Error> {
        match s {
            "Input" => Ok(Storage::Input),
            "Output" => Ok(Storage::Output),
            "UniformConstant" => Ok(Storage::UniformConstant),
            "Uniform" => Ok(Storage::Uniform),
            s => Err(Error(format!("storage class {s}: not one GLSL declares"))),
        }
    }
}

/// A linked program's global, read off the SPIR-V before it is lowered
#[derive(Clone, Debug)]
pub struct Variable<'t> {
    pub name: String,
    pub stage: ShaderType,
    pub storage: Storage,
    pub ty: Type<'t>,
    pub location: Option<u32>,
    pub binding: Option<u32>,
    /// a built-in (gl_Position, gl_FragCoord …): not an active attribute or varying
    pub builtin: bool,
}

impl<'t> Variable<'t> {
    fn of(body: &Body<'t>, stage: ShaderType, g: OpId) -> Result<Variable<'t>, Error> {
        let op = &body[g];
        let name = op.get_str("sym_name").ok_or_else(|| Error("spirv.GlobalVariable without a name".into()))?.to_owned();
        let Some(Attribute::Type(t)) = op.get("type") else { return Err(Error(format!("@{name}: no type"))) };
        let Ok(SpirvType::Ptr { pointee, storage }) = SpirvType::try_from(t) else { return Err(Error(format!("@{name}: not a pointer"))) };
        let int = |k: &str| match op.get(k) {
            Some(Attribute::Int { lit, .. }) => lit.parse().ok(),
            _ => None,
        };
        Ok(Variable { stage, storage: Storage::try_from(&*storage)?, ty: *pointee, location: int("location"), binding: int("binding"), builtin: op.get("built_in").is_some(), name })
    }
}

/// A loose uniform of the default block: its name and shape
#[derive(Clone, Debug)]
pub struct Uniform {
    pub name: String,
    pub ty: UniformType,
}

/// glUniform*'s argument: the scalars and how many per element
#[derive(Clone, Copy, Debug)]
pub enum UniformValue<'v> {
    Float(u32, &'v [f32]),
    Int(u32, &'v [i32]),
    Uint(u32, &'v [u32]),
    /// glUniformMatrix{c}x{r}fv: columns, rows, column-major values
    Matrix(u32, u32, &'v [f32]),
}

impl UniformValue<'_> {
    /// The words the block holds: a bool uniform takes 0 or 1 (GL ES 3.0 §2.12.6: false only for 0)
    fn words(self, scalar: Scalar) -> Vec<u32> {
        let bool_of = |zero: bool| u32::from(!zero);
        match (self, scalar) {
            (UniformValue::Float(_, v), Scalar::Bool) => v.iter().map(|f| bool_of(*f == 0.0)).collect(),
            (UniformValue::Int(_, v), Scalar::Bool) => v.iter().map(|i| bool_of(*i == 0)).collect(),
            (UniformValue::Uint(_, v), Scalar::Bool) => v.iter().map(|u| bool_of(*u == 0)).collect(),
            (UniformValue::Float(_, v) | UniformValue::Matrix(_, _, v), _) => v.iter().copied().map(f32::to_bits).collect(),
            (UniformValue::Int(_, v), _) => v.iter().copied().map(i32::cast_unsigned).collect(),
            (UniformValue::Uint(_, v), _) => v.to_vec(),
        }
    }

    /// GL ES 3.0 §2.12.6: the call's type and size must be the uniform's; bools take any scalar, samplers glUniform1i
    fn fits(self, ty: UniformType) -> bool {
        let shape = match self {
            UniformValue::Float(n, _) | UniformValue::Int(n, _) | UniformValue::Uint(n, _) => (1, n),
            UniformValue::Matrix(c, r, _) => (c, r),
        };
        let scalar = match (self, ty.scalar) {
            (UniformValue::Int(1, _), Scalar::Sampler) => true,
            (UniformValue::Float(..), Scalar::Float) => !ty.matrix(),
            (UniformValue::Matrix(..), Scalar::Float) => ty.matrix(),
            (UniformValue::Int(..), Scalar::Int) | (UniformValue::Uint(..), Scalar::Uint) => true,
            (UniformValue::Float(..) | UniformValue::Int(..) | UniformValue::Uint(..), Scalar::Bool) => true,
            _ => false,
        };
        scalar && (ty.scalar == Scalar::Sampler || shape == (ty.columns, ty.rows))
    }

    fn len(self) -> usize {
        match self {
            UniformValue::Float(_, v) | UniformValue::Matrix(_, _, v) => v.len(),
            UniformValue::Int(_, v) => v.len(),
            UniformValue::Uint(_, v) => v.len(),
        }
    }
}

/// The default uniform block's binding: past the GL ES 3.0 uniform buffer bindings an application can use
pub const DEFAULT_UNIFORM_BLOCK: u32 = MAX_UNIFORM_BUFFER_BINDINGS;
/// GL ES 3.0's minimum MAX_UNIFORM_BUFFER_BINDINGS, the number detour advertises
pub const MAX_UNIFORM_BUFFER_BINDINGS: u32 = 24;
/// std140: a matrix column and the block's size are 16-byte aligned
const STD140_VEC4: u32 = 16;
const WORD: u32 = 4;
/// The global glsl-default-block packs a shader's loose uniforms into, and its location ⇒ byte offset table
const DEFAULT_BLOCK_NAME: &str = "gl_DefaultUniformBlock";
const OFFSETS: &str = "offsets";

/// GLSL to SPIR-V in three runs: linked (the loose uniforms' locations, read here), the loose uniforms packed into
/// one block (its offsets, read here before the block becomes SPIR-V), the SPIR-V as OpenGL computes it
pub struct GlslPipelines {
    link: PassManager,
    block: PassManager,
    spirv: PassManager,
}

impl GlslPipelines {
    pub fn new(tools: &Toolchain<'_>) -> Result<Self, Error> {
        Ok(GlslPipelines {
            link: tools.pipeline("builtin.module(glsl-link)")?,
            block: tools.pipeline(&format!("builtin.module(glsl-default-block{{binding={DEFAULT_UNIFORM_BLOCK}}})"))?,
            spirv: tools.pipeline("builtin.module(convert-glsl-to-spirv{client=opengl})")?,
        })
    }
}

/// What glLinkProgram made: the machine code and the interface GL queries
pub struct Linked<'t> {
    pub compiled: Compiled<'t>,
    pub variables: Vec<Variable<'t>>,
    /// uniform locations: an array uniform takes one location per element (uniform, element)
    pub uniforms: BTreeMap<u32, (usize, u32)>,
    pub values: Vec<Uniform>,
    /// the default block's byte offset of every location
    pub offsets: BTreeMap<u32, u32>,
    /// the default block as glUniform* last set it (std140)
    pub block: Vec<u8>,
    /// sampler uniforms' texture units, by location
    pub units: BTreeMap<u32, i32>,
}

#[derive(Default)]
pub struct Program<'t> {
    pub shaders: Vec<u32>,
    pub linked: Option<Linked<'t>>,
    pub log: String,
    pub deleted: bool,
}

impl<'t> Program<'t> {
    /// glLinkProgram: the attached stages from source, linked, their loose uniforms in one block, the SPIR-V (as
    /// OpenGL computes), the GPU's machine code; a failed link keeps the previous program running (GL ES 3.0 §2.11.3)
    pub fn link(&mut self, tools: &'t Toolchain<'t>, glsl: &GlslPipelines, shaders: &[&Shader]) -> bool {
        match Program::build(tools, glsl, shaders) {
            Ok(linked) => {
                self.linked = Some(linked);
                self.log.clear();
                true
            }
            Err(e) => {
                self.log = e.0;
                false
            }
        }
    }

    fn build(tools: &'t Toolchain<'t>, glsl: &GlslPipelines, shaders: &[&Shader]) -> Result<Linked<'t>, Error> {
        let stages = [ShaderType::Vertex, ShaderType::Fragment];
        if let Some(s) = shaders.iter().find(|s| !s.compiled) {
            return Err(Error(format!("a {:?} shader that did not compile", s.ty)));
        }
        let mut body = tools.body();
        for stage in stages {
            let mut of = shaders.iter().filter(|s| s.ty == stage);
            let (Some(s), None) = (of.next(), of.next()) else { return Err(Error(format!("not one {stage:?} shader"))) };
            GlslReader::read(&mut body, &s.source, stage.into())?;
        }
        let root = body.root;
        glsl.link.run(&mut body, root)?;
        let values = Program::loose(&body)?;
        glsl.block.run(&mut body, root)?;
        let offsets = Program::offsets(&body)?;
        glsl.spirv.run(&mut body, root)?;
        let top = body.top_block(root);
        let modules: Vec<OpId> = body.ops(top).filter(|&m| &*body[m].name == "spirv.module").collect();
        let mut variables = Vec::new();
        for (m, stage) in modules.iter().zip(stages) {
            let globals: Vec<OpId> = body.ops(body.top_block(*m)).filter(|&g| &*body[g].name == "spirv.GlobalVariable").collect();
            for g in globals {
                variables.push(Variable::of(&body, stage, g)?);
            }
        }
        let mut uniforms = BTreeMap::new();
        for (u, (value, at)) in values.iter().enumerate() {
            for e in 0..value.ty.count {
                uniforms.insert(at + e, (u, e));
            }
        }
        let values: Vec<Uniform> = values.into_iter().map(|(v, _)| v).collect();
        let mut ranges: Vec<(u32, u32, u32)> = uniforms.iter().filter_map(|(&at, &(u, _))| offsets.get(&at).map(|&off| (off, off + values[u].ty.span(), at))).collect();
        ranges.sort_unstable();
        if let Some(w) = ranges.windows(2).find(|w| w[1].0 < w[0].1) {
            return Err(Error(format!("the default uniform block puts locations {} and {} on the same bytes ({}..{} and {}..{})", w[0].2, w[1].2, w[0].0, w[0].1, w[1].0, w[1].1)));
        }
        let size = ranges.iter().map(|r| r.1).max().unwrap_or(0);
        let block = vec![0; size.next_multiple_of(STD140_VEC4) as usize];
        Ok(Linked { compiled: tools.compile(body)?, variables, uniforms, values, offsets, block, units: BTreeMap::new() })
    }

    /// The loose uniforms (the default block's and the samplers) after glsl-link assigned their locations; one
    /// used in both stages is one uniform
    fn loose(body: &Body<'t>) -> Result<Vec<(Uniform, u32)>, Error> {
        let mut globals = Vec::new();
        body.walk(body.root, &mut |o| globals.push(o));
        let mut out: Vec<(Uniform, u32)> = Vec::new();
        for g in globals.into_iter().filter(|&g| &*body[g].name == GlslGlobal::NAME) {
            let h = body.head::<GlslGlobal>(g)?;
            if h.storage != GlslStorage::Uniform || h.builtin.is_some() {
                continue;
            }
            let name = Symbol::defined(&h.sym_name)?.to_owned();
            if out.iter().any(|(u, _)| u.name == name) {
                continue;
            }
            if let Ok(SpirvType::Struct { .. }) = SpirvType::try_from(&h.ty) {
                if h.binding.is_some() {
                    continue;
                }
                return Err(Error(format!("uniform {name}: struct uniforms are not reflected yet")));
            }
            let ty = UniformType::try_from(&h.ty)?;
            let at = h.location.ok_or_else(|| Error(format!("uniform {name} without a location after glsl-link")))?;
            out.push((Uniform { name, ty }, at));
        }
        Ok(out)
    }

    /// The location ⇒ byte offset tables glsl-default-block left on each shader's block (the same in every stage)
    fn offsets(body: &Body<'t>) -> Result<BTreeMap<u32, u32>, Error> {
        let mut globals = Vec::new();
        body.walk(body.root, &mut |o| globals.push(o));
        let mut offsets = BTreeMap::new();
        for g in globals {
            let op = &body[g];
            if &*op.name != GlslGlobal::NAME || op.get(Symbol::ATTR).and_then(|s| Symbol::defined(s).ok()) != Some(DEFAULT_BLOCK_NAME) {
                continue;
            }
            let Some(Attribute::Array(pairs)) = op.get(OFFSETS) else { return Err(Error(format!("{DEFAULT_BLOCK_NAME} without its {OFFSETS}"))) };
            for p in pairs {
                let Attribute::Array(lo) = p else { return Err(Error(format!("{OFFSETS}: {p}"))) };
                let [l, o] = lo.as_slice() else { return Err(Error(format!("{OFFSETS}: {p}"))) };
                let int = |a: &Attribute<'_>| a.int().ok().and_then(|v| u32::try_from(v).ok()).ok_or_else(|| Error(format!("{OFFSETS}: {p}")));
                offsets.insert(int(l)?, int(o)?);
            }
        }
        Ok(offsets)
    }
}

impl UniformType {
    /// The bytes one element takes in std140: a matrix's columns 16 bytes apart
    const fn span(self) -> u32 {
        if self.matrix() { (self.columns - 1) * STD140_VEC4 + self.rows * WORD } else { self.rows * WORD }
    }
}

impl Linked<'_> {
    /// glGetAttribLocation
    pub fn attribute(&self, name: &str) -> Option<u32> {
        self.variables.iter().find(|v| v.stage == ShaderType::Vertex && v.storage == Storage::Input && !v.builtin && v.name == name).and_then(|v| v.location)
    }

    /// glGetUniformLocation: `name` or `name[i]`
    pub fn uniform(&self, name: &str) -> Option<u32> {
        let (base, index) = match name.strip_suffix(']').and_then(|n| n.split_once('[')) {
            Some((b, i)) => (b, i.parse().ok()?),
            None => (name, 0),
        };
        let (at, _) = self.uniforms.iter().find(|(_, (u, e))| self.values[*u].name == base && *e == index)?;
        Some(*at)
    }

    /// glUniform*: `location` -1 is ignored; the values fill elements from `location` on, each at its offset
    pub fn set(&mut self, location: i32, value: UniformValue<'_>) -> GlResult<()> {
        if location == -1 {
            return Ok(());
        }
        let location = u32::try_from(location).map_err(|_| GlError::InvalidOperation)?;
        let &(u, element) = self.uniforms.get(&location).ok_or(GlError::InvalidOperation)?;
        let ty = self.values[u].ty;
        let per = ty.components() as usize;
        if !value.fits(ty) || value.len() % per != 0 || (value.len() > per && ty.count == 1) {
            return Err(GlError::InvalidOperation);
        }
        let words = value.words(ty.scalar);
        let elements = (words.len() / per).min((ty.count - element) as usize);
        if ty.scalar == Scalar::Sampler {
            self.units.insert(location, words[0].cast_signed());
            return Ok(());
        }
        for (k, chunk) in words.chunks(per).take(elements).enumerate() {
            let at = *self.offsets.get(&(location + k as u32)).ok_or(GlError::InvalidOperation)?;
            for (i, w) in chunk.iter().enumerate() {
                let (c, r) = (i as u32 / ty.rows, i as u32 % ty.rows);
                let off = (at + c * STD140_VEC4 + r * WORD) as usize;
                self.block[off..off + WORD as usize].copy_from_slice(&w.to_le_bytes());
            }
        }
        Ok(())
    }
}
