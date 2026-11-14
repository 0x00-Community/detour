mod common;

use std::time::{Duration, Instant};

use ash::vk;
use common::{Api, Report, Scene};

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const INTEL: u32 = 0x8086;

struct Mem {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut u8,
    size: usize,
}

struct Vk {
    _entry: ash::Entry,
    instance: ash::Instance,
    dev: ash::Device,
    props: vk::PhysicalDeviceMemoryProperties,
    name: String,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    busy: [bool; common::RING],
    next: usize,
    last: usize,
    image: vk::Image,
    image_mem: vk::DeviceMemory,
    view: vk::ImageView,
    layout: vk::PipelineLayout,
    tri: vk::Pipeline,
    fill: vk::Pipeline,
    set_layout: vk::DescriptorSetLayout,
    descriptors: vk::DescriptorPool,
    set: vk::DescriptorSet,
    compute_layout: vk::PipelineLayout,
    compute: vk::Pipeline,
    bufs: Vec<Mem>,
}

#[derive(Clone, Copy)]
enum Buf {
    Triangle,
    Quad,
    Full,
    Out,
    Readback,
}

impl Vk {
    fn new(tiling: vk::ImageTiling) -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let entry = ash::Entry::load()?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?;
            let (pd, name) = instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|pd| (pd, instance.get_physical_device_properties(pd)))
                .find(|(_, p)| p.vendor_id == INTEL && p.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU)
                .map(|(pd, p)| (pd, p.device_name_as_c_str().map_or(String::new(), |n| n.to_string_lossy().into_owned())))
                .ok_or("no Intel GPU in Vulkan")?;
            let family = instance
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE))
                .ok_or("no graphics queue")? as u32;
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&[1.0])];
            let dev = instance.create_device(pd, &vk::DeviceCreateInfo::default().queue_create_infos(&queues).push_next(&mut v13), None)?;
            let props = instance.get_physical_device_memory_properties(pd);
            let queue = dev.get_device_queue(family, 0);
            let pool = dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmds = dev.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(common::RING as u32),
            )?;
            let fences = (0..common::RING).map(|_| dev.create_fence(&vk::FenceCreateInfo::default(), None)).collect::<Result<_, _>>()?;

            let image = dev.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(FORMAT)
                    .extent(vk::Extent3D { width: common::W, height: common::H, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(tiling)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC),
                None,
            )?;
            let req = dev.get_image_memory_requirements(image);
            let image_mem = dev.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(Self::memory_type(&props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?),
                None,
            )?;
            dev.bind_image_memory(image, image_mem, 0)?;
            let view = dev.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(FORMAT)
                    .subresource_range(Self::color_range()),
                None,
            )?;

            let layout = dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;
            let set_layout = dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)]),
                None,
            )?;
            let descriptors = dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&[vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }]),
                None,
            )?;
            let set = dev.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptors).set_layouts(&[set_layout]),
            )?[0];
            let compute_layout = dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None)?;

            let mut vk = Vk {
                _entry: entry,
                instance,
                dev,
                props,
                name: format!("Vulkan {}", if tiling == vk::ImageTiling::LINEAR { "linear" } else { "optimal" }),
                queue,
                pool,
                cmds,
                fences,
                busy: [false; common::RING],
                next: 0,
                last: 0,
                image,
                image_mem,
                view,
                layout,
                tri: vk::Pipeline::null(),
                fill: vk::Pipeline::null(),
                set_layout,
                descriptors,
                set,
                compute_layout,
                compute: vk::Pipeline::null(),
                bufs: Vec::new(),
            };
            let le = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            let out = *common::LANES.iter().max().unwrap() as usize * 4;
            for (bytes, usage) in [
                (le(&common::TRIANGLE), vk::BufferUsageFlags::VERTEX_BUFFER),
                (le(&common::QUAD), vk::BufferUsageFlags::VERTEX_BUFFER),
                (common::instances(), vk::BufferUsageFlags::VERTEX_BUFFER),
                (vec![0; out], vk::BufferUsageFlags::STORAGE_BUFFER),
                (vec![0; (common::W * common::H * 4) as usize], vk::BufferUsageFlags::TRANSFER_DST),
            ] {
                let m = vk.buffer(bytes.len(), usage)?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), m.ptr, bytes.len());
                vk.bufs.push(m);
            }
            vk.dev.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo { buffer: vk.buf(Buf::Out).buf, offset: 0, range: vk::WHOLE_SIZE }])],
                &[],
            );

            let t = Instant::now();
            vk.tri = vk.graphics(common::TRI_VS, common::TRI_FS, false)?;
            vk.fill = vk.graphics(common::FILL_VS, common::FILL_FS, true)?;
            let cs = vk.shader(common::COMPUTE, naga::ShaderStage::Compute)?;
            vk.compute = vk
                .dev
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(cs).name(c"main"))
                        .layout(compute_layout)],
                    None,
                )
                .map_err(|e| e.1)?[0];
            vk.dev.destroy_shader_module(cs, None);
            eprintln!("Vulkan ({name}, {tiling:?}): 3 pipelines (naga + anv) in {:?}", t.elapsed());
            Ok(vk)
        }
    }

    fn color_range() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }
    }

    fn memory_type(props: &vk::PhysicalDeviceMemoryProperties, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, String> {
        (0..props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(flags))
            .ok_or_else(|| format!("no memory type {flags:?}"))
    }

    fn buf(&self, b: Buf) -> &Mem {
        &self.bufs[b as usize]
    }

    unsafe fn buffer(&self, size: usize, usage: vk::BufferUsageFlags) -> Result<Mem, Box<dyn std::error::Error>> {
        unsafe {
            let buf = self.dev.create_buffer(&vk::BufferCreateInfo::default().size(size as u64).usage(usage), None)?;
            let req = self.dev.get_buffer_memory_requirements(buf);
            let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = self.dev.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(Self::memory_type(&self.props, req.memory_type_bits, flags)?),
                None,
            )?;
            self.dev.bind_buffer_memory(buf, mem, 0)?;
            let ptr = self.dev.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?.cast();
            Ok(Mem { buf, mem, ptr, size })
        }
    }

    unsafe fn shader(&self, src: &str, stage: naga::ShaderStage) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
        let module = naga::front::glsl::Frontend::default()
            .parse(&naga::front::glsl::Options::from(stage), src)
            .map_err(|e| format!("{stage:?}: {e:?}"))?;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module)?;
        let mut options = naga::back::spv::Options::default();
        options.flags.remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);
        let words = naga::back::spv::write_vec(
            &module,
            &info,
            &options,
            Some(&naga::back::spv::PipelineOptions { shader_stage: stage, entry_point: "main".into() }),
        )?;
        Ok(unsafe { self.dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)? })
    }

    unsafe fn graphics(&self, vs: &str, fs: &str, fill: bool) -> Result<vk::Pipeline, Box<dyn std::error::Error>> {
        unsafe {
            let (vs, fs) = (self.shader(vs, naga::ShaderStage::Vertex)?, self.shader(fs, naga::ShaderStage::Fragment)?);
            let stages = [
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vs).name(c"main"),
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fs).name(c"main"),
            ];
            let f2 = vk::Format::R32G32_SFLOAT;
            let f4 = vk::Format::R32G32B32A32_SFLOAT;
            let (bindings, attrs): (Vec<_>, Vec<_>) = if fill {
                (
                    vec![
                        vk::VertexInputBindingDescription { binding: 0, stride: 8, input_rate: vk::VertexInputRate::VERTEX },
                        vk::VertexInputBindingDescription { binding: 1, stride: 32, input_rate: vk::VertexInputRate::INSTANCE },
                    ],
                    vec![
                        vk::VertexInputAttributeDescription { location: 0, binding: 0, format: f2, offset: 0 },
                        vk::VertexInputAttributeDescription { location: 1, binding: 1, format: f4, offset: 0 },
                        vk::VertexInputAttributeDescription { location: 2, binding: 1, format: f4, offset: 16 },
                    ],
                )
            } else {
                (
                    vec![vk::VertexInputBindingDescription { binding: 0, stride: 8, input_rate: vk::VertexInputRate::VERTEX }],
                    vec![vk::VertexInputAttributeDescription { location: 0, binding: 0, format: f2, offset: 0 }],
                )
            };
            let input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attrs);
            let topology = if fill { vk::PrimitiveTopology::TRIANGLE_STRIP } else { vk::PrimitiveTopology::TRIANGLE_LIST };
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
            let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
            let raster = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).line_width(1.0);
            let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend = [vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(fill)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
            let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
            let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&[FORMAT]);
            let p = self
                .dev
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&stages)
                        .vertex_input_state(&input)
                        .input_assembly_state(&assembly)
                        .viewport_state(&viewport)
                        .rasterization_state(&raster)
                        .multisample_state(&ms)
                        .color_blend_state(&blend)
                        .dynamic_state(&dynamic)
                        .layout(self.layout)
                        .push_next(&mut rendering)],
                    None,
                )
                .map_err(|e| e.1)?[0];
            self.dev.destroy_shader_module(vs, None);
            self.dev.destroy_shader_module(fs, None);
            Ok(p)
        }
    }

    unsafe fn begin(&mut self) -> usize {
        unsafe {
            let i = self.next % common::RING;
            self.next += 1;
            if self.busy[i] {
                self.dev.wait_for_fences(&[self.fences[i]], true, u64::MAX).expect("fence");
                self.busy[i] = false;
            }
            self.dev.reset_fences(&[self.fences[i]]).expect("reset");
            self.dev.reset_command_buffer(self.cmds[i], vk::CommandBufferResetFlags::empty()).expect("reset");
            let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.dev.begin_command_buffer(self.cmds[i], &begin).expect("begin");
            i
        }
    }

    unsafe fn submit(&mut self, i: usize) {
        unsafe {
            self.dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&self.cmds[i..=i])], self.fences[i]).expect("submit");
            self.busy[i] = true;
            self.last = i;
        }
    }

    unsafe fn barrier(&self, cmd: vk::CommandBuffer, from: (vk::ImageLayout, vk::PipelineStageFlags, vk::AccessFlags), to: (vk::ImageLayout, vk::PipelineStageFlags, vk::AccessFlags)) {
        let b = vk::ImageMemoryBarrier::default()
            .old_layout(from.0)
            .new_layout(to.0)
            .src_access_mask(from.2)
            .dst_access_mask(to.2)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(Self::color_range());
        unsafe { self.dev.cmd_pipeline_barrier(cmd, from.1, to.1, vk::DependencyFlags::empty(), &[], &[], &[b]) };
    }
}

impl Api for Vk {
    type Done = usize;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn frame(&mut self, scene: Scene) -> (usize, Option<Duration>) {
        unsafe {
            let i = self.begin();
            let t = Instant::now();
            let cmd = self.cmds[i];
            let attach = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let write = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
            self.barrier(cmd, (vk::ImageLayout::UNDEFINED, attach, write), (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, attach, write));
            let clear = if scene == Scene::Tri { common::BLUE } else { common::FILL_CLEAR };
            let color = [vk::RenderingAttachmentInfo::default()
                .image_view(self.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: clear } })];
            let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: common::W, height: common::H } };
            self.dev.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color));
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: common::W as f32, height: common::H as f32, min_depth: 0.0, max_depth: 1.0 };
            self.dev.cmd_set_viewport(cmd, 0, &[viewport]);
            self.dev.cmd_set_scissor(cmd, 0, &[area]);
            match scene {
                Scene::Tri => {
                    self.dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.tri);
                    self.dev.cmd_bind_vertex_buffers(cmd, 0, &[self.buf(Buf::Triangle).buf], &[0]);
                    self.dev.cmd_draw(cmd, 3, 1, 0, 0);
                }
                Scene::Fill => {
                    self.dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.fill);
                    self.dev.cmd_bind_vertex_buffers(cmd, 0, &[self.buf(Buf::Quad).buf, self.buf(Buf::Full).buf], &[0, 0]);
                    self.dev.cmd_draw(cmd, 4, common::LAYERS, 0, 0);
                }
            }
            self.dev.cmd_end_rendering(cmd);
            self.dev.end_command_buffer(cmd).expect("end");
            let record = t.elapsed();
            self.submit(i);
            (i, Some(record))
        }
    }

    fn wait(&mut self, done: usize) {
        unsafe { self.dev.wait_for_fences(&[self.fences[done]], true, u64::MAX).expect("fence") };
    }

    fn pixels(&mut self) -> Vec<u8> {
        unsafe {
            let i = self.begin();
            let cmd = self.cmds[i];
            let attach = (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
            let src = (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ);
            self.barrier(cmd, attach, src);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width: common::W, height: common::H, depth: 1 });
            let rb = self.buf(Buf::Readback);
            self.dev.cmd_copy_image_to_buffer(cmd, self.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, rb.buf, &[region]);
            let host = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(rb.buf)
                .size(vk::WHOLE_SIZE);
            self.dev.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[], &[host], &[]);
            self.dev.end_command_buffer(cmd).expect("end");
            self.submit(i);
            self.wait(i);
            std::slice::from_raw_parts(self.buf(Buf::Readback).ptr, self.buf(Buf::Readback).size).to_vec()
        }
    }

    fn dispatch(&mut self, lanes: u32, count: u32) -> Option<Duration> {
        unsafe {
            let i = self.begin();
            let t = Instant::now();
            let cmd = self.cmds[i];
            self.dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.compute);
            self.dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.compute_layout, 0, &[self.set], &[]);
            let shader = vk::PipelineStageFlags::COMPUTE_SHADER;
            for _ in 0..count {
                self.dev.cmd_dispatch(cmd, lanes / 16, 1, 1);
                let m = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::HOST_READ);
                self.dev.cmd_pipeline_barrier(cmd, shader, shader | vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[m], &[], &[]);
            }
            self.dev.end_command_buffer(cmd).expect("end");
            let record = t.elapsed();
            self.submit(i);
            self.wait(i);
            Some(record)
        }
    }

    fn words(&mut self, lanes: u32) -> Vec<u32> {
        let out = self.buf(Buf::Out);
        let b = unsafe { std::slice::from_raw_parts(out.ptr, lanes as usize * 4) };
        b.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect()
    }
}

impl Drop for Vk {
    fn drop(&mut self) {
        unsafe {
            let d = &self.dev;
            d.device_wait_idle().ok();
            for m in &self.bufs {
                d.destroy_buffer(m.buf, None);
                d.free_memory(m.mem, None);
            }
            for p in [self.tri, self.fill, self.compute] {
                d.destroy_pipeline(p, None);
            }
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_pipeline_layout(self.compute_layout, None);
            d.destroy_descriptor_pool(self.descriptors, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_image_view(self.view, None);
            d.destroy_image(self.image, None);
            d.free_memory(self.image_mem, None);
            for f in &self.fences {
                d.destroy_fence(*f, None);
            }
            d.destroy_command_pool(self.pool, None);
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = common::frames_arg();
    let xe = common::Detour::device()?;
    let mut detour = common::Detour::new(&xe)?;
    let a = Report::run(&mut detour, n);
    drop(detour);
    let mut reports = vec![a];
    for tiling in [vk::ImageTiling::OPTIMAL, vk::ImageTiling::LINEAR] {
        let mut vk = Vk::new(tiling)?;
        reports.push(Report::run(&mut vk, n));
    }
    Report::table(&reports, n);
    Ok(())
}
