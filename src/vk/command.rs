use std::collections::BTreeMap;

use super::{VkAttachmentLoadOp, VkBuffer, VkBufferImageCopy, VkClearValue, VkCommandBuffer, VkDescriptorSet, VkDevice, VkError, VkFence, VkFramebuffer, VkImage, VkPipeline, VkPipelineLayout};
use super::{VkQueue, VkRect2D, VkRenderPass, VkResult, VkViewport};
use crate::{ColorFormat, ColorTarget, Draw, Surface, TargetClear, TargetFill, TargetLoad, Tiling, VertexBuffer, VertexElement};

/// The channels a clear writes: all of them
const ALL_CHANNELS: [bool; 4] = [true; 4];

/// VkRenderPassBeginInfo: a clear value per attachment (read for those cleared)
#[derive(Clone, Copy, Debug)]
pub struct VkRenderPassBeginInfo<'a> {
    pub render_pass: VkRenderPass,
    pub framebuffer: VkFramebuffer,
    pub render_area: VkRect2D,
    pub clear_values: &'a [VkClearValue],
}

/// A recorded command
#[derive(Clone, Debug)]
pub enum VkCommand {
    BeginRenderPass { render_pass: VkRenderPass, framebuffer: VkFramebuffer, render_area: VkRect2D, clear_values: Vec<VkClearValue> },
    EndRenderPass,
    BindPipeline(VkPipeline),
    BindVertexBuffers { first: u32, buffers: Vec<(VkBuffer, u64)> },
    BindDescriptorSets { layout: VkPipelineLayout, first: u32, sets: Vec<VkDescriptorSet> },
    SetViewport(VkViewport),
    SetScissor(VkRect2D),
    Draw { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32 },
    CopyImageToBuffer { image: VkImage, buffer: VkBuffer, regions: Vec<VkBufferImageCopy> },
}

/// VkSubmitInfo (no semaphores: a submission has finished when vkQueueSubmit returns)
#[derive(Clone, Copy, Debug)]
pub struct VkSubmitInfo<'a> {
    pub command_buffers: &'a [VkCommandBuffer],
}

/// The render pass instance a command buffer is in: its color attachment, and the clear its first draw (or its
/// end, when nothing is drawn) still owes
struct PassInstance {
    surface: Surface,
    format: ColorFormat,
    extent: [u32; 2],
    area: [u32; 4],
    clear: Option<Vec<u8>>,
}

impl PassInstance {
    /// What the next draw loads its target with; its clear is then paid
    fn load(&mut self, clears: &mut Vec<TargetClear>) -> Option<Vec<u8>> {
        let texel = self.clear.take()?;
        let full = self.area == [0, 0, self.extent[0], self.extent[1]];
        match full {
            true => Some(texel),
            false => {
                clears.push(TargetClear { rect: self.area, channels: ALL_CHANNELS, texel });
                None
            }
        }
    }
}

/// What a command buffer has bound while it runs
#[derive(Default)]
struct Bound {
    pipeline: Option<VkPipeline>,
    vertex: BTreeMap<u32, (VkBuffer, u64)>,
    sets: BTreeMap<u32, VkDescriptorSet>,
    viewport: Option<VkViewport>,
    scissor: Option<VkRect2D>,
    pass: Option<PassInstance>,
}

impl VkDevice<'_, '_> {
    fn recording(&mut self, cb: VkCommandBuffer) -> VkResult<&mut Vec<VkCommand>> {
        let b = self.command_buffers.get_mut(cb)?;
        match b.recording {
            true => Ok(&mut b.commands),
            false => Err(VkError::ValidationFailed(format!("{cb:?} is not recording"))),
        }
    }

    fn record(&mut self, cb: VkCommandBuffer, c: VkCommand) -> VkResult<()> {
        self.recording(cb)?.push(c);
        Ok(())
    }

    /// vkBeginCommandBuffer: what it held is reset
    pub fn begin_command_buffer(&mut self, cb: VkCommandBuffer) -> VkResult<()> {
        let b = self.command_buffers.get_mut(cb)?;
        b.commands.clear();
        b.recording = true;
        Ok(())
    }

    pub fn end_command_buffer(&mut self, cb: VkCommandBuffer) -> VkResult<()> {
        self.recording(cb)?;
        self.command_buffers.get_mut(cb)?.recording = false;
        Ok(())
    }

    pub fn cmd_begin_render_pass(&mut self, cb: VkCommandBuffer, info: &VkRenderPassBeginInfo<'_>) -> VkResult<()> {
        let c = VkCommand::BeginRenderPass { render_pass: info.render_pass, framebuffer: info.framebuffer, render_area: info.render_area, clear_values: info.clear_values.to_vec() };
        self.record(cb, c)
    }

    pub fn cmd_end_render_pass(&mut self, cb: VkCommandBuffer) -> VkResult<()> {
        self.record(cb, VkCommand::EndRenderPass)
    }

    pub fn cmd_bind_pipeline(&mut self, cb: VkCommandBuffer, pipeline: VkPipeline) -> VkResult<()> {
        self.pipelines.get(pipeline)?;
        self.record(cb, VkCommand::BindPipeline(pipeline))
    }

    pub fn cmd_bind_vertex_buffers(&mut self, cb: VkCommandBuffer, first: u32, buffers: &[VkBuffer], offsets: &[u64]) -> VkResult<()> {
        if buffers.len() != offsets.len() {
            return Err(VkError::ValidationFailed(format!("{} vertex buffers, {} offsets", buffers.len(), offsets.len())));
        }
        self.record(cb, VkCommand::BindVertexBuffers { first, buffers: buffers.iter().copied().zip(offsets.iter().copied()).collect() })
    }

    pub fn cmd_bind_descriptor_sets(&mut self, cb: VkCommandBuffer, layout: VkPipelineLayout, first: u32, sets: &[VkDescriptorSet]) -> VkResult<()> {
        self.record(cb, VkCommand::BindDescriptorSets { layout, first, sets: sets.to_vec() })
    }

    /// vkCmdSetViewport of viewport 0
    pub fn cmd_set_viewport(&mut self, cb: VkCommandBuffer, viewport: VkViewport) -> VkResult<()> {
        self.record(cb, VkCommand::SetViewport(viewport))
    }

    /// vkCmdSetScissor of scissor 0
    pub fn cmd_set_scissor(&mut self, cb: VkCommandBuffer, scissor: VkRect2D) -> VkResult<()> {
        self.record(cb, VkCommand::SetScissor(scissor))
    }

    pub fn cmd_draw(&mut self, cb: VkCommandBuffer, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> VkResult<()> {
        self.record(cb, VkCommand::Draw { vertex_count, instance_count, first_vertex, first_instance })
    }

    pub fn cmd_copy_image_to_buffer(&mut self, cb: VkCommandBuffer, image: VkImage, buffer: VkBuffer, regions: &[VkBufferImageCopy]) -> VkResult<()> {
        self.record(cb, VkCommand::CopyImageToBuffer { image, buffer, regions: regions.to_vec() })
    }

    /// vkQueueSubmit: the command buffers run in order, each to its end, then `fence` is signaled
    pub fn queue_submit(&mut self, _queue: VkQueue, submits: &[VkSubmitInfo<'_>], fence: Option<VkFence>) -> VkResult<()> {
        for s in submits {
            for &cb in s.command_buffers {
                let b = self.command_buffers.get(cb)?;
                if b.recording {
                    return Err(VkError::ValidationFailed(format!("{cb:?} submitted while recording")));
                }
                let commands = b.commands.clone();
                let mut bound = Bound::default();
                for c in &commands {
                    self.execute(&mut bound, c)?;
                }
            }
        }
        if let Some(f) = fence {
            self.fences.get_mut(f)?.signaled = true;
        }
        Ok(())
    }

    /// vkQueueWaitIdle / vkDeviceWaitIdle: nothing runs after vkQueueSubmit returns
    pub fn queue_wait_idle(&self, _queue: VkQueue) -> VkResult<()> {
        Ok(())
    }

    fn execute(&mut self, bound: &mut Bound, c: &VkCommand) -> VkResult<()> {
        match c {
            VkCommand::BeginRenderPass { render_pass, framebuffer, render_area, clear_values } => {
                bound.pass = Some(self.begin(*render_pass, *framebuffer, *render_area, clear_values)?);
            }
            VkCommand::EndRenderPass => {
                let mut pass = bound.pass.take().ok_or_else(|| VkError::ValidationFailed("vkCmdEndRenderPass outside a render pass".into()))?;
                self.load_undrawn(&mut pass)?;
            }
            VkCommand::BindPipeline(p) => bound.pipeline = Some(*p),
            VkCommand::BindVertexBuffers { first, buffers } => {
                for (i, b) in buffers.iter().enumerate() {
                    bound.vertex.insert(first + i as u32, *b);
                }
            }
            VkCommand::BindDescriptorSets { layout, first, sets } => {
                let layout = self.pipeline_layouts.get(*layout)?;
                for (i, &s) in sets.iter().enumerate() {
                    let at = first + i as u32;
                    if layout.set_layouts.get(at as usize) != Some(&self.descriptor_sets.get(s)?.layout) {
                        return Err(VkError::ValidationFailed(format!("{s:?} bound at set {at}: not the layout's")));
                    }
                    bound.sets.insert(at, s);
                }
            }
            VkCommand::SetViewport(v) => bound.viewport = Some(*v),
            VkCommand::SetScissor(s) => bound.scissor = Some(*s),
            VkCommand::Draw { vertex_count, instance_count, first_vertex, first_instance } => {
                if *first_instance != 0 {
                    return Err(VkError::FeatureNotPresent("a first instance other than 0".into()));
                }
                if *vertex_count != 0 && *instance_count != 0 {
                    self.draw(bound, *first_vertex, *vertex_count, *instance_count)?;
                }
            }
            VkCommand::CopyImageToBuffer { image, buffer, regions } => self.copy_image_to_buffer(*image, *buffer, regions)?,
        }
        Ok(())
    }

    fn begin(&self, render_pass: VkRenderPass, framebuffer: VkFramebuffer, area: VkRect2D, clear_values: &[VkClearValue]) -> VkResult<PassInstance> {
        let pass = self.render_passes.get(render_pass)?;
        let fb = self.framebuffers.get(framebuffer)?;
        let [a] = pass.color_attachments[..] else { return Err(VkError::FeatureNotPresent(format!("a subpass of {} color attachments (one, for now)", pass.color_attachments.len()))) };
        let (format, image) = self.attachment_format(pass, fb, a)?;
        let i = self.images.get(image)?;
        let surface = i.surface.ok_or_else(|| VkError::ValidationFailed(format!("{image:?}: not bound to memory")))?;
        let extent = i.info.extent;
        let clear = match pass.attachments[a as usize].load_op {
            VkAttachmentLoadOp::Clear => {
                let VkClearValue::Color(rgba) = clear_values.get(a as usize).ok_or_else(|| VkError::ValidationFailed(format!("no clear value for attachment {a}")))?;
                Some(format.texel(*rgba)?)
            }
            VkAttachmentLoadOp::Load | VkAttachmentLoadOp::DontCare => None,
        };
        let area = Rect::clip(area, [0, 0, extent[0].min(fb.extent[0]), extent[1].min(fb.extent[1])]);
        Ok(PassInstance { surface, format: format.try_into()?, extent, area, clear })
    }

    fn draw(&self, bound: &mut Bound, first: u32, count: u32, instances: u32) -> VkResult<()> {
        let pipeline = self.pipelines.get(bound.pipeline.ok_or_else(|| VkError::ValidationFailed("a draw with no pipeline bound".into()))?)?;
        let pass = bound.pass.as_mut().ok_or_else(|| VkError::ValidationFailed("a draw outside a render pass".into()))?;
        let primitive = pipeline.topology.try_into()?;
        let Some(cull) = pipeline.rasterization.cull(primitive) else { return Ok(()) };
        let viewport = pipeline.viewport.or(bound.viewport).ok_or_else(|| VkError::ValidationFailed("a dynamic viewport not set".into()))?;
        let scissor = pipeline.scissor.or(bound.scissor).ok_or_else(|| VkError::ValidationFailed("a dynamic scissor not set".into()))?;
        if viewport.height < 0.0 {
            return Err(VkError::FeatureNotPresent("a viewport of negative height".into()));
        }
        let mut buffers = Vec::new();
        let mut slots = BTreeMap::new();
        for b in &pipeline.bindings {
            let Some(&(buffer, offset)) = bound.vertex.get(&b.binding) else { continue };
            slots.insert(b.binding, buffers.len() as u32);
            buffers.push(VertexBuffer { bytes: self.buffer_bytes(buffer, offset, None)?, stride: b.stride });
        }
        let elements = pipeline
            .inputs
            .iter()
            .map(|a| {
                let buffer = *slots.get(&a.binding).ok_or_else(|| VkError::ValidationFailed(format!("attribute {}: no vertex buffer bound at {}", a.location, a.binding)))?;
                Ok(VertexElement { buffer, format: a.format.try_into()?, offset: a.offset })
            })
            .collect::<VkResult<Vec<_>>>()?;
        let mut uniforms = Vec::new();
        for (&at, &set) in &bound.sets {
            let set = self.descriptor_sets.get(set)?;
            if at != 0 && !set.buffers.is_empty() {
                return Err(VkError::FeatureNotPresent(format!("descriptor set {at} (set 0 only, for now)")));
            }
            for (&binding, b) in &set.buffers {
                uniforms.push((binding, self.buffer_bytes(b.buffer, b.offset, b.range)?));
            }
        }
        let [x0, y0, x1, y1] = Rect::clip(scissor, pass.area);
        let mut clears = Vec::new();
        let fill = pass.load(&mut clears);
        let load = match &fill {
            Some(texel) => TargetLoad::Fill(texel, &[]),
            None => TargetLoad::Keep(&clears),
        };
        let draw = Draw {
            buffers,
            elements,
            uniforms,
            primitive,
            first,
            count,
            instances,
            viewport: [viewport.x, viewport.y, viewport.width, viewport.height],
            depth_range: [viewport.min_depth, viewport.max_depth],
            scissor: Some([x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]),
            cull,
            front_ccw: pipeline.rasterization.front_face == super::VkFrontFace::CounterClockwise,
            target: ColorTarget { width: pass.extent[0], height: pass.extent[1], format: pass.format, tiling: Tiling::Linear, load },
            ..Draw::default()
        };
        self.gpu.draw(&pipeline.compiled, &draw, pass.surface).map_err(|e| VkError::DeviceLost(e.0))
    }

    /// A render pass that drew nothing: its load op (a clear) still applied to the attachment, written over it
    fn load_undrawn(&self, pass: &mut PassInstance) -> VkResult<()> {
        let mut clears = Vec::new();
        let fill = pass.load(&mut clears);
        if fill.is_none() && clears.is_empty() {
            return Ok(());
        }
        let [width, height] = pass.extent;
        let target = ColorTarget { width, height, format: pass.format, tiling: Tiling::Linear, ..ColorTarget::default() };
        let lost = |e: mlirformat::Error| VkError::DeviceLost(e.0);
        let mut pixels = match &fill {
            Some(texel) => {
                let mut p = vec![0; target.len()];
                TargetFill::apply(&mut p, texel);
                p
            }
            None => self.gpu.read(pass.surface, target.len()).map_err(lost)?,
        };
        for c in &clears {
            c.apply(&mut pixels, width);
        }
        self.gpu.write(pass.surface, [0, 0, width, height], &pixels).map_err(lost)
    }

    /// vkCmdCopyImageToBuffer: the image's texels read back from its surface
    fn copy_image_to_buffer(&mut self, image: VkImage, buffer: VkBuffer, regions: &[VkBufferImageCopy]) -> VkResult<()> {
        let i = self.images.get(image)?;
        let info = i.info;
        let surface = i.surface.ok_or_else(|| VkError::ValidationFailed(format!("{image:?}: not bound to memory")))?;
        let len = usize::try_from(VkDevice::image_len(&info)).map_err(|_| VkError::OutOfHostMemory)?;
        let texels = self.gpu.read(surface, len).map_err(|e| VkError::DeviceLost(e.0))?;
        let bpp = info.format.bytes() as usize;
        let width = info.extent[0] as usize;
        for r in regions {
            let ([ox, oy], [w, h]) = (r.image_offset, r.image_extent);
            if ox + w > info.extent[0] || oy + h > info.extent[1] {
                return Err(VkError::ValidationFailed(format!("a copy of {w}×{h} at ({ox}, {oy}) out of a {:?} image", info.extent)));
            }
            let row_len = if r.buffer_row_length == 0 { w } else { r.buffer_row_length } as usize;
            let out = self.buffer_bytes_mut(buffer, r.buffer_offset)?;
            for y in 0..h as usize {
                let src = ((oy as usize + y) * width + ox as usize) * bpp;
                let dst = y * row_len * bpp;
                let n = w as usize * bpp;
                let room = out.len();
                out.get_mut(dst..dst + n).ok_or_else(|| VkError::ValidationFailed(format!("a copy past the buffer's {room} bytes")))?.copy_from_slice(&texels[src..src + n]);
            }
        }
        Ok(())
    }
}

/// Rectangles as x0, y0, x1, y1 (exclusive)
struct Rect;

impl Rect {
    /// `r` within `bounds`
    fn clip(r: VkRect2D, bounds: [u32; 4]) -> [u32; 4] {
        let lo = |at: i32, b: u32| (at.max(0) as u32).clamp(b, u32::MAX);
        let hi = |at: i32, len: u32, b0: u32, b1: u32| ((i64::from(at) + i64::from(len)).max(0) as u32).clamp(b0, b1);
        let (x0, y0) = (lo(r.offset[0], bounds[0]).min(bounds[2]), lo(r.offset[1], bounds[1]).min(bounds[3]));
        [x0, y0, hi(r.offset[0], r.extent[0], x0, bounds[2]), hi(r.offset[1], r.extent[1], y0, bounds[3])]
    }
}
