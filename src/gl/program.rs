use std::collections::BTreeMap;

use mlirformat::spirv_types::SpirvType;
use mlirformat::{Attribute, Dim, Error, FloatKind, ShapedKind, Signedness, Type};
use mlirpass::{Body, GlslReader, GlslStage, OpId};

use super::{GlError, GlResult};
use crate::{Toolchain, Xe3Program};

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

/// A uniform's value as glUniform* last set it, in 32-bit words
#[derive(Clone, Debug)]
pub struct Uniform {
    pub name: String,
    pub ty: UniformType,
    pub words: Vec<u32>,
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
    fn words(self) -> Vec<u32> {
        match self {
            UniformValue::Float(_, v) | UniformValue::Matrix(_, _, v) => v.iter().copied().map(f32::to_bits).collect(),
            UniformValue::Int(_, v) => v.iter().copied().map(i32::cast_unsigned).collect(),
            UniformValue::Uint(_, v) => v.to_vec(),
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

/// What glLinkProgram made: the Xe3 kernels and the interface GL queries
pub struct Linked<'t> {
    pub xe3: Xe3Program<'t>,
    pub variables: Vec<Variable<'t>>,
    /// uniform locations: an array uniform takes one location per element
    pub uniforms: BTreeMap<u32, (usize, u32)>,
    pub values: Vec<Uniform>,
}

#[derive(Default)]
pub struct Program<'t> {
    pub shaders: Vec<u32>,
    pub linked: Option<Linked<'t>>,
    pub log: String,
    pub deleted: bool,
}

impl<'t> Program<'t> {
    /// glLinkProgram: the attached stages from source, glsl-link and convert-glsl-to-spirv (as OpenGL computes),
    /// then Xe3; a failed link keeps the previous program running (GL ES 3.0 §2.11.3)
    pub fn link(&mut self, tools: &'t Toolchain<'t>, glsl: &mlirpass::PassManager, shaders: &[&Shader]) -> bool {
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

    fn build(tools: &'t Toolchain<'t>, glsl: &mlirpass::PassManager, shaders: &[&Shader]) -> Result<Linked<'t>, Error> {
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
        glsl.run(&mut body, root)?;
        let top = body.top_block(root);
        let modules: Vec<OpId> = body.ops(top).filter(|&m| &*body[m].name == "spirv.module").collect();
        let mut variables = Vec::new();
        for (m, stage) in modules.iter().zip(stages) {
            let globals: Vec<OpId> = body.ops(body.top_block(*m)).filter(|&g| &*body[g].name == "spirv.GlobalVariable").collect();
            for g in globals {
                variables.push(Variable::of(&body, stage, g)?);
            }
        }
        let (uniforms, values) = Program::uniforms(&variables)?;
        Ok(Linked { xe3: tools.xe3(body)?, variables, uniforms, values })
    }

    /// The default block's uniforms, by location; one used in both stages is one uniform
    fn uniforms(variables: &[Variable<'t>]) -> Result<(BTreeMap<u32, (usize, u32)>, Vec<Uniform>), Error> {
        let mut uniforms = BTreeMap::new();
        let mut values: Vec<Uniform> = Vec::new();
        for v in variables.iter().filter(|v| v.storage == Storage::UniformConstant) {
            if values.iter().any(|u| u.name == v.name) {
                continue;
            }
            let ty = UniformType::try_from(&v.ty)?;
            let at = v.location.ok_or_else(|| Error(format!("uniform {} without a location", v.name)))?;
            for e in 0..ty.count {
                uniforms.insert(at + e, (values.len(), e));
            }
            values.push(Uniform { name: v.name.clone(), ty, words: vec![0; (ty.components() * ty.count) as usize] });
        }
        Ok((uniforms, values))
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

    /// glUniform*: `location` -1 is ignored; the values fill elements from `location` on
    pub fn set(&mut self, location: i32, value: UniformValue<'_>) -> GlResult<()> {
        if location == -1 {
            return Ok(());
        }
        let &(u, element) = u32::try_from(location).ok().and_then(|l| self.uniforms.get(&l)).ok_or(GlError::InvalidOperation)?;
        let uniform = &mut self.values[u];
        let per = uniform.ty.components() as usize;
        if !value.fits(uniform.ty) || value.len() % per != 0 || (value.len() > per && uniform.ty.count == 1) {
            return Err(GlError::InvalidOperation);
        }
        let start = element as usize * per;
        let words = value.words();
        let end = (start + words.len()).min(uniform.words.len());
        uniform.words[start..end].copy_from_slice(&words[..end - start]);
        Ok(())
    }
}
