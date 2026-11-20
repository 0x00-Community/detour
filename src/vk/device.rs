use std::collections::BTreeMap;

use mlirformat::Attribute;
use mlirformat::spirv_types::SpirvType;
use mlirpass::{Body, OpId};

use super::{VkAttachmentDescription, VkCommand, VkDescriptorBufferInfo, VkDescriptorSetLayoutBinding, VkDescriptorType, VkError, VkFormat, VkHandle, VkImageCreateInfo, VkMemoryPropertyFlags};
use super::{VkMemoryRequirements, VkMemoryType, VkObjects, VkPrimitiveTopology, VkRasterization, VkRect2D, VkRenderPassCreateInfo, VkResult, VkShaderStage, VkVertexInputAttributeDescription};
use super::{VkVertexInputBindingDescription, VkVertexInputRate, VkViewport};
use crate::{ColorTarget, Compiled, Device, Surface, Tiling, Toolchain};

/// A buffer's memory is placed on this alignment
const BUFFER_ALIGN: u64 = 64;
/// An image's memory is placed on a page
const IMAGE_ALIGN: u64 = 4096;
/// SPIR-V's magic number, its first word
const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_WORD: usize = 4;
/// The SPIR-V op a module's globals are
const GLOBAL_VARIABLE: &str = "spirv.GlobalVariable";
/// A SPIR-V global's storage class, as far as a pipeline's vertex inputs care
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VkStorage {
    Input,
    Other,
}

impl From<&str> for VkStorage {
    fn from(s: &str) -> Self {
        match s {
            "Input" => VkStorage::Input,
            _ => VkStorage::Other,
        }
    }
}

/// The one memory type: the GPU shares the host's memory
const MEMORY_TYPES: [VkMemoryType; 1] = [VkMemoryType {
    property_flags: VkMemoryPropertyFlags { device_local: true, host_visible: true, host_coherent: true, host_cached: false },
    heap_index: 0,
}];
/// memory_type_bits of an object any memory type can back
const ANY_MEMORY_TYPE: u32 = (1 << MEMORY_TYPES.len()) - 1;

/// VkDeviceMemory: host-visible and coherent, so what is written through a mapping is what a submission reads
pub struct VkDeviceMemoryObject {
    pub bytes: Vec<u8>,
}

pub struct VkBufferObject {
    pub size: u64,
    /// the memory it is bound to, and where in it
    pub memory: Option<(VkDeviceMemory, u64)>,
}

/// An image's texels live in a device surface (linear, row 0 first), made when it is bound to memory; the memory
/// it is bound to holds none of them
pub struct VkImageObject {
    pub info: VkImageCreateInfo,
    pub surface: Option<Surface>,
}

pub struct VkImageViewObject {
    pub image: VkImage,
}

pub struct VkRenderPassObject {
    pub attachments: Vec<VkAttachmentDescription>,
    pub color_attachments: Vec<u32>,
}

pub struct VkFramebufferObject {
    pub render_pass: VkRenderPass,
    pub attachments: Vec<VkImageView>,
    pub extent: [u32; 2],
}

pub struct VkShaderModuleObject {
    pub code: Vec<u8>,
}

pub struct VkDescriptorSetLayoutObject {
    pub bindings: Vec<VkDescriptorSetLayoutBinding>,
}

pub struct VkPipelineLayoutObject {
    pub set_layouts: Vec<VkDescriptorSetLayout>,
}

/// A graphics pipeline: its stages' machine code, and the fixed-function state a draw takes from it (viewport and
/// scissor None: set by the command buffer)
pub struct VkPipelineObject {
    pub compiled: Compiled,
    pub bindings: Vec<VkVertexInputBindingDescription>,
    /// the vertex shader's inputs' attributes, in location order
    pub inputs: Vec<VkVertexInputAttributeDescription>,
    pub topology: VkPrimitiveTopology,
    pub viewport: Option<VkViewport>,
    pub scissor: Option<VkRect2D>,
    pub rasterization: VkRasterization,
}

/// The kind of a pipeline's handle (its object borrows the toolchain)
pub enum VkPipelineKind {}

pub struct VkDescriptorPoolObject {
    pub sets: Vec<VkDescriptorSet>,
}

pub struct VkDescriptorSetObject {
    pub layout: VkDescriptorSetLayout,
    /// the uniform buffers written to it, by binding
    pub buffers: BTreeMap<u32, VkDescriptorBufferInfo>,
}

pub struct VkCommandPoolObject {
    pub buffers: Vec<VkCommandBuffer>,
}

/// A command buffer: the commands recorded, run when it is submitted
#[derive(Default)]
pub struct VkCommandBufferObject {
    pub commands: Vec<VkCommand>,
    pub recording: bool,
}

pub struct VkFenceObject {
    pub signaled: bool,
}

pub type VkDeviceMemory = VkHandle<VkDeviceMemoryObject>;
pub type VkBuffer = VkHandle<VkBufferObject>;
pub type VkImage = VkHandle<VkImageObject>;
pub type VkImageView = VkHandle<VkImageViewObject>;
pub type VkRenderPass = VkHandle<VkRenderPassObject>;
pub type VkFramebuffer = VkHandle<VkFramebufferObject>;
pub type VkShaderModule = VkHandle<VkShaderModuleObject>;
pub type VkDescriptorSetLayout = VkHandle<VkDescriptorSetLayoutObject>;
pub type VkPipelineLayout = VkHandle<VkPipelineLayoutObject>;
pub type VkPipeline = VkHandle<VkPipelineKind>;
pub type VkDescriptorPool = VkHandle<VkDescriptorPoolObject>;
pub type VkDescriptorSet = VkHandle<VkDescriptorSetObject>;
pub type VkCommandPool = VkHandle<VkCommandPoolObject>;
pub type VkCommandBuffer = VkHandle<VkCommandBufferObject>;
pub type VkFence = VkHandle<VkFenceObject>;

/// vkGetDeviceQueue's queue: detour has one, on the render engine, that runs a submission to its end
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VkQueue;

/// VkPipelineShaderStageCreateInfo (a module has one entry point: `name` is not looked up)
#[derive(Clone, Copy, Debug)]
pub struct VkPipelineShaderStageCreateInfo<'a> {
    pub stage: VkShaderStage,
    pub module: VkShaderModule,
    pub name: &'a str,
}

/// VkGraphicsPipelineCreateInfo: no viewports or scissors means dynamic ones (vkCmdSetViewport / vkCmdSetScissor)
#[derive(Clone, Copy, Debug)]
pub struct VkGraphicsPipelineCreateInfo<'a> {
    pub stages: &'a [VkPipelineShaderStageCreateInfo<'a>],
    pub vertex_bindings: &'a [VkVertexInputBindingDescription],
    pub vertex_attributes: &'a [VkVertexInputAttributeDescription],
    pub topology: VkPrimitiveTopology,
    pub viewports: &'a [VkViewport],
    pub scissors: &'a [VkRect2D],
    pub rasterization: VkRasterization,
    pub layout: VkPipelineLayout,
    pub render_pass: VkRenderPass,
    pub subpass: u32,
}

/// VkWriteDescriptorSet of uniform buffers
#[derive(Clone, Copy, Debug)]
pub struct VkWriteDescriptorSet<'a> {
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub descriptor_type: VkDescriptorType,
    pub buffers: &'a [VkDescriptorBufferInfo],
}

/// A Vulkan device on detour: its objects by handle, its work run on the device's GPU and its programs compiled by
/// the toolchain
pub struct VkDevice<'t, 'x> {
    pub tools: &'t Toolchain<'t>,
    pub gpu: &'t Device<'x>,
    pub memories: VkObjects<VkDeviceMemoryObject>,
    pub buffers: VkObjects<VkBufferObject>,
    pub images: VkObjects<VkImageObject>,
    pub image_views: VkObjects<VkImageViewObject>,
    pub render_passes: VkObjects<VkRenderPassObject>,
    pub framebuffers: VkObjects<VkFramebufferObject>,
    pub shader_modules: VkObjects<VkShaderModuleObject>,
    pub set_layouts: VkObjects<VkDescriptorSetLayoutObject>,
    pub pipeline_layouts: VkObjects<VkPipelineLayoutObject>,
    pub pipelines: VkObjects<VkPipelineObject, VkPipelineKind>,
    pub descriptor_pools: VkObjects<VkDescriptorPoolObject>,
    pub descriptor_sets: VkObjects<VkDescriptorSetObject>,
    pub command_pools: VkObjects<VkCommandPoolObject>,
    pub command_buffers: VkObjects<VkCommandBufferObject>,
    pub fences: VkObjects<VkFenceObject>,
}

impl<'t, 'x> VkDevice<'t, 'x> {
    /// vkCreateDevice: one graphics queue
    pub fn new(tools: &'t Toolchain<'t>, gpu: &'t Device<'x>) -> Self {
        VkDevice {
            tools,
            gpu,
            memories: VkObjects::default(),
            buffers: VkObjects::default(),
            images: VkObjects::default(),
            image_views: VkObjects::default(),
            render_passes: VkObjects::default(),
            framebuffers: VkObjects::default(),
            shader_modules: VkObjects::default(),
            set_layouts: VkObjects::default(),
            pipeline_layouts: VkObjects::default(),
            pipelines: VkObjects::default(),
            descriptor_pools: VkObjects::default(),
            descriptor_sets: VkObjects::default(),
            command_pools: VkObjects::default(),
            command_buffers: VkObjects::default(),
            fences: VkObjects::default(),
        }
    }

    /// vkGetPhysicalDeviceMemoryProperties' memory types
    pub const fn memory_types(&self) -> &[VkMemoryType] {
        &MEMORY_TYPES
    }

    /// vkGetDeviceQueue: queue family 0, queue 0
    pub fn queue(&self, family: u32, index: u32) -> VkResult<VkQueue> {
        match (family, index) {
            (0, 0) => Ok(VkQueue),
            _ => Err(VkError::ValidationFailed(format!("queue {index} of family {family}: detour has one queue"))),
        }
    }

    pub fn allocate_memory(&mut self, size: u64, memory_type: u32) -> VkResult<VkDeviceMemory> {
        if memory_type as usize >= MEMORY_TYPES.len() {
            return Err(VkError::ValidationFailed(format!("memory type {memory_type}")));
        }
        let len = usize::try_from(size).map_err(|_| VkError::OutOfDeviceMemory)?;
        Ok(self.memories.insert(VkDeviceMemoryObject { bytes: vec![0; len] }))
    }

    /// vkMapMemory: `size` None is VK_WHOLE_SIZE
    pub fn map_memory(&mut self, memory: VkDeviceMemory, offset: u64, size: Option<u64>) -> VkResult<&mut [u8]> {
        let bytes = &mut self.memories.get_mut(memory)?.bytes;
        let start = usize::try_from(offset).map_err(|_| VkError::MemoryMapFailed)?;
        let end = match size {
            Some(s) => start.checked_add(usize::try_from(s).map_err(|_| VkError::MemoryMapFailed)?).ok_or(VkError::MemoryMapFailed)?,
            None => bytes.len(),
        };
        bytes.get_mut(start..end).ok_or(VkError::MemoryMapFailed)
    }

    pub fn create_buffer(&mut self, size: u64) -> VkBuffer {
        self.buffers.insert(VkBufferObject { size, memory: None })
    }

    pub fn buffer_memory_requirements(&self, buffer: VkBuffer) -> VkResult<VkMemoryRequirements> {
        Ok(VkMemoryRequirements { size: self.buffers.get(buffer)?.size.next_multiple_of(BUFFER_ALIGN), alignment: BUFFER_ALIGN, memory_type_bits: ANY_MEMORY_TYPE })
    }

    pub fn bind_buffer_memory(&mut self, buffer: VkBuffer, memory: VkDeviceMemory, offset: u64) -> VkResult<()> {
        let room = self.memories.get(memory)?.bytes.len() as u64;
        let b = self.buffers.get_mut(buffer)?;
        if offset.checked_add(b.size).is_none_or(|end| end > room) {
            return Err(VkError::ValidationFailed(format!("{buffer:?}: {} bytes at {offset} of a memory of {room}", b.size)));
        }
        b.memory = Some((memory, offset));
        Ok(())
    }

    /// `len` bytes of `buffer` from `offset` (None: to its end)
    pub fn buffer_bytes(&self, buffer: VkBuffer, offset: u64, len: Option<u64>) -> VkResult<&[u8]> {
        let b = self.buffers.get(buffer)?;
        let (memory, at) = b.memory.ok_or_else(|| VkError::ValidationFailed(format!("{buffer:?}: not bound to memory")))?;
        let end = len.map_or(b.size, |l| offset + l).min(b.size);
        let bytes = &self.memories.get(memory)?.bytes;
        bytes.get((at + offset) as usize..(at + end) as usize).ok_or_else(|| VkError::ValidationFailed(format!("{buffer:?}: {offset}..{end} past its memory")))
    }

    /// The bytes of `buffer` from `offset` on, to write
    pub fn buffer_bytes_mut(&mut self, buffer: VkBuffer, offset: u64) -> VkResult<&mut [u8]> {
        let b = self.buffers.get(buffer)?;
        let (memory, at) = b.memory.ok_or_else(|| VkError::ValidationFailed(format!("{buffer:?}: not bound to memory")))?;
        let size = b.size;
        let bytes = &mut self.memories.get_mut(memory)?.bytes;
        bytes.get_mut((at + offset) as usize..(at + size) as usize).ok_or_else(|| VkError::ValidationFailed(format!("{buffer:?}: {offset}.. past its memory")))
    }

    pub fn create_image(&mut self, info: VkImageCreateInfo) -> VkResult<VkImage> {
        crate::ColorFormat::try_from(info.format)?;
        Ok(self.images.insert(VkImageObject { info, surface: None }))
    }

    /// The bytes an image's texels take, rows packed
    pub fn image_len(info: &VkImageCreateInfo) -> u64 {
        u64::from(info.extent[0]) * u64::from(info.extent[1]) * u64::from(info.format.bytes())
    }

    pub fn image_memory_requirements(&self, image: VkImage) -> VkResult<VkMemoryRequirements> {
        let len = VkDevice::image_len(&self.images.get(image)?.info);
        Ok(VkMemoryRequirements { size: len.next_multiple_of(IMAGE_ALIGN), alignment: IMAGE_ALIGN, memory_type_bits: ANY_MEMORY_TYPE })
    }

    /// vkBindImageMemory: the image gets its surface on the GPU
    pub fn bind_image_memory(&mut self, image: VkImage, memory: VkDeviceMemory, _offset: u64) -> VkResult<()> {
        self.memories.get(memory)?;
        let i = self.images.get_mut(image)?;
        let target = ColorTarget { width: i.info.extent[0], height: i.info.extent[1], format: i.info.format.try_into()?, tiling: Tiling::Linear, ..ColorTarget::default() };
        i.surface = Some(self.gpu.surface(&target).map_err(|e| VkError::DeviceLost(e.0))?);
        Ok(())
    }

    /// vkCreateImageView of a whole 2D image, its own format
    pub fn create_image_view(&mut self, image: VkImage) -> VkResult<VkImageView> {
        self.images.get(image)?;
        Ok(self.image_views.insert(VkImageViewObject { image }))
    }

    pub fn create_render_pass(&mut self, info: &VkRenderPassCreateInfo<'_>) -> VkResult<VkRenderPass> {
        if let Some(a) = info.color_attachments.iter().find(|&&a| a as usize >= info.attachments.len()) {
            return Err(VkError::ValidationFailed(format!("color attachment {a} of {}", info.attachments.len())));
        }
        Ok(self.render_passes.insert(VkRenderPassObject { attachments: info.attachments.to_vec(), color_attachments: info.color_attachments.to_vec() }))
    }

    pub fn create_framebuffer(&mut self, render_pass: VkRenderPass, attachments: &[VkImageView], extent: [u32; 2]) -> VkResult<VkFramebuffer> {
        let pass = self.render_passes.get(render_pass)?;
        if pass.attachments.len() != attachments.len() {
            return Err(VkError::ValidationFailed(format!("{} attachments for a render pass of {}", attachments.len(), pass.attachments.len())));
        }
        Ok(self.framebuffers.insert(VkFramebufferObject { render_pass, attachments: attachments.to_vec(), extent }))
    }

    /// vkCreateShaderModule: `code` is SPIR-V, read when a pipeline is made of it
    pub fn create_shader_module(&mut self, code: &[u8]) -> VkResult<VkShaderModule> {
        let magic = code.first_chunk::<SPIRV_WORD>().map(|w| u32::from_le_bytes(*w));
        if !code.len().is_multiple_of(SPIRV_WORD) || magic != Some(SPIRV_MAGIC) {
            return Err(VkError::ValidationFailed("a shader module that is not SPIR-V".into()));
        }
        Ok(self.shader_modules.insert(VkShaderModuleObject { code: code.to_vec() }))
    }

    pub fn create_descriptor_set_layout(&mut self, bindings: &[VkDescriptorSetLayoutBinding]) -> VkDescriptorSetLayout {
        self.set_layouts.insert(VkDescriptorSetLayoutObject { bindings: bindings.to_vec() })
    }

    pub fn create_pipeline_layout(&mut self, set_layouts: &[VkDescriptorSetLayout]) -> VkResult<VkPipelineLayout> {
        for &l in set_layouts {
            self.set_layouts.get(l)?;
        }
        Ok(self.pipeline_layouts.insert(VkPipelineLayoutObject { set_layouts: set_layouts.to_vec() }))
    }

    /// vkCreateGraphicsPipelines, one: the stages' SPIR-V compiled for the GPU
    pub fn create_graphics_pipeline(&mut self, info: &VkGraphicsPipelineCreateInfo<'_>) -> VkResult<VkPipeline> {
        self.render_passes.get(info.render_pass)?;
        self.pipeline_layouts.get(info.layout)?;
        if let Some(b) = info.vertex_bindings.iter().find(|b| b.input_rate == VkVertexInputRate::Instance) {
            return Err(VkError::FeatureNotPresent(format!("vertex binding {}: per-instance input", b.binding)));
        }
        let mut stages = info.stages.to_vec();
        stages.sort_by_key(|s| s.stage);
        let [vs, fs] = stages[..] else { return Err(VkError::FeatureNotPresent("a pipeline of other than a vertex and a fragment stage".into())) };
        if (vs.stage, fs.stage) != (VkShaderStage::Vertex, VkShaderStage::Fragment) {
            return Err(VkError::FeatureNotPresent("a pipeline of other than a vertex and a fragment stage".into()));
        }
        let mut body = self.tools.body();
        let vertex = self.tools.read_spirv(&mut body, &self.shader_modules.get(vs.module)?.code)?;
        self.tools.read_spirv(&mut body, &self.shader_modules.get(fs.module)?.code)?;
        let inputs = VkDevice::inputs(&body, vertex)?
            .into_iter()
            .map(|at| info.vertex_attributes.iter().find(|a| a.location == at).copied().ok_or_else(|| VkError::ValidationFailed(format!("the vertex shader's input {at} has no attribute"))))
            .collect::<VkResult<Vec<_>>>()?;
        for a in &inputs {
            if !info.vertex_bindings.iter().any(|b| b.binding == a.binding) {
                return Err(VkError::ValidationFailed(format!("attribute {}: binding {} not described", a.location, a.binding)));
            }
        }
        let compiled = self.tools.compile(body)?;
        Ok(self.pipelines.insert(VkPipelineObject {
            compiled,
            bindings: info.vertex_bindings.to_vec(),
            inputs,
            topology: info.topology,
            viewport: info.viewports.first().copied(),
            scissor: info.scissors.first().copied(),
            rasterization: info.rasterization,
        }))
    }

    /// The vertex shader's inputs' locations, in order (its built-ins left out)
    fn inputs(body: &Body<'_>, module: OpId) -> VkResult<Vec<u32>> {
        let mut at = Vec::new();
        for g in body.ops(body.top_block(module)).filter(|&g| &*body[g].name == GLOBAL_VARIABLE) {
            let op = &body[g];
            let Some(Attribute::Type(t)) = op.get("type") else { continue };
            let Ok(SpirvType::Ptr { storage, .. }) = SpirvType::try_from(t) else { continue };
            if VkStorage::from(&*storage) != VkStorage::Input || op.get("built_in").is_some() {
                continue;
            }
            let location = match op.get("location") {
                Some(Attribute::Int { lit, .. }) => lit.parse().ok(),
                _ => None,
            };
            at.push(location.ok_or_else(|| VkError::ValidationFailed(format!("input {} without a location", op.get_str("sym_name").unwrap_or("?"))))?);
        }
        at.sort_unstable();
        Ok(at)
    }

    pub fn create_descriptor_pool(&mut self) -> VkDescriptorPool {
        self.descriptor_pools.insert(VkDescriptorPoolObject { sets: Vec::new() })
    }

    pub fn allocate_descriptor_sets(&mut self, pool: VkDescriptorPool, layouts: &[VkDescriptorSetLayout]) -> VkResult<Vec<VkDescriptorSet>> {
        self.descriptor_pools.get(pool)?;
        let mut sets = Vec::new();
        for &layout in layouts {
            self.set_layouts.get(layout)?;
            sets.push(self.descriptor_sets.insert(VkDescriptorSetObject { layout, buffers: BTreeMap::new() }));
        }
        self.descriptor_pools.get_mut(pool)?.sets.extend(&sets);
        Ok(sets)
    }

    /// vkUpdateDescriptorSets' writes: buffer i of a write goes to binding dst_binding + i
    pub fn update_descriptor_sets(&mut self, writes: &[VkWriteDescriptorSet<'_>]) -> VkResult<()> {
        for w in writes {
            let layout = self.set_layouts.get(self.descriptor_sets.get(w.dst_set)?.layout)?;
            for (i, b) in w.buffers.iter().enumerate() {
                let binding = w.dst_binding + i as u32;
                if !layout.bindings.iter().any(|l| l.binding == binding && l.descriptor_type == w.descriptor_type) {
                    return Err(VkError::ValidationFailed(format!("{:?}: binding {binding} is not a {:?} of its layout", w.dst_set, w.descriptor_type)));
                }
                self.buffers.get(b.buffer)?;
            }
            let set = self.descriptor_sets.get_mut(w.dst_set)?;
            for (i, b) in w.buffers.iter().enumerate() {
                set.buffers.insert(w.dst_binding + i as u32, *b);
            }
        }
        Ok(())
    }

    pub fn create_command_pool(&mut self, family: u32) -> VkResult<VkCommandPool> {
        self.queue(family, 0)?;
        Ok(self.command_pools.insert(VkCommandPoolObject { buffers: Vec::new() }))
    }

    pub fn allocate_command_buffers(&mut self, pool: VkCommandPool, count: u32) -> VkResult<Vec<VkCommandBuffer>> {
        self.command_pools.get(pool)?;
        let buffers: Vec<_> = (0..count).map(|_| self.command_buffers.insert(VkCommandBufferObject::default())).collect();
        self.command_pools.get_mut(pool)?.buffers.extend(&buffers);
        Ok(buffers)
    }

    pub fn create_fence(&mut self, signaled: bool) -> VkFence {
        self.fences.insert(VkFenceObject { signaled })
    }

    /// vkWaitForFences: a submission runs to its end before vkQueueSubmit returns, so there is nothing to wait for;
    /// true when the fences are signaled (all of them, or one when not `wait_all`), false is VK_TIMEOUT
    pub fn wait_for_fences(&self, fences: &[VkFence], wait_all: bool) -> VkResult<bool> {
        let signaled = fences.iter().map(|&f| self.fences.get(f).map(|f| f.signaled)).collect::<VkResult<Vec<_>>>()?;
        Ok(if wait_all { signaled.iter().all(|s| *s) } else { signaled.iter().any(|s| *s) })
    }

    pub fn reset_fences(&mut self, fences: &[VkFence]) -> VkResult<()> {
        for &f in fences {
            self.fences.get_mut(f)?.signaled = false;
        }
        Ok(())
    }

    /// vkGetFenceStatus: Ok(true) is VK_SUCCESS, Ok(false) VK_NOT_READY
    pub fn fence_status(&self, fence: VkFence) -> VkResult<bool> {
        Ok(self.fences.get(fence)?.signaled)
    }

    /// The format an attachment of `pass` is, checked against the image its framebuffer puts there
    pub fn attachment_format(&self, pass: &VkRenderPassObject, fb: &VkFramebufferObject, a: u32) -> VkResult<(VkFormat, VkImage)> {
        let view = *fb.attachments.get(a as usize).ok_or_else(|| VkError::ValidationFailed(format!("attachment {a}")))?;
        let image = self.image_views.get(view)?.image;
        let format = self.images.get(image)?.info.format;
        let described = pass.attachments[a as usize].format;
        if format != described {
            return Err(VkError::ValidationFailed(format!("attachment {a}: a {format:?} image where the render pass has {described:?}")));
        }
        Ok((format, image))
    }
}
