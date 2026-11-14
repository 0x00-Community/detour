mod common;

use std::time::{Duration, Instant};

use common::{Api, Report, Scene};
use glow::HasContext;
use khronos_egl as egl;

const PLATFORM_SURFACELESS_MESA: egl::Enum = 0x31DD;

const WAIT_NS: i32 = 1_000_000_000;

struct Gl {
    egl: egl::DynamicInstance<egl::EGL1_5>,
    display: egl::Display,
    context: egl::Context,
    gl: glow::Context,
    name: String,
    fences: [Option<glow::Fence>; common::RING],
    next: usize,
    fbo: glow::Framebuffer,
    texture: glow::Texture,
    tri: (glow::Program, glow::VertexArray),
    fill: (glow::Program, glow::VertexArray),
    compute: glow::Program,
    buffers: Vec<glow::Buffer>,
    out: glow::Buffer,
}

impl Gl {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        unsafe {
            let egl = egl::DynamicInstance::<egl::EGL1_5>::load_required()?;
            let display = egl.get_platform_display(PLATFORM_SURFACELESS_MESA, egl::DEFAULT_DISPLAY, &[egl::ATTRIB_NONE])?;
            egl.initialize(display)?;
            egl.bind_api(egl::OPENGL_API)?;
            let config = egl
                .choose_first_config(display, &[egl::SURFACE_TYPE, egl::PBUFFER_BIT, egl::RENDERABLE_TYPE, egl::OPENGL_BIT, egl::NONE])?
                .ok_or("no EGL config")?;
            let attrs = [
                egl::CONTEXT_MAJOR_VERSION, 4,
                egl::CONTEXT_MINOR_VERSION, 6,
                egl::CONTEXT_OPENGL_PROFILE_MASK, egl::CONTEXT_OPENGL_CORE_PROFILE_BIT,
                egl::NONE,
            ];
            let context = egl.create_context(display, config, None, &attrs)?;
            egl.make_current(display, None, None, Some(context))?;
            let gl = glow::Context::from_loader_function(|s| egl.get_proc_address(s).map_or(std::ptr::null(), |f| f as *const _));
            let renderer = gl.get_parameter_string(glow::RENDERER);
            if !renderer.contains("Intel") {
                return Err(format!("GL renderer is {renderer}, not the Intel GPU").into());
            }

            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::RGBA8, common::W as i32, common::H as i32);
            let fbo = gl.create_framebuffer()?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(texture), 0);
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                return Err("framebuffer incomplete".into());
            }
            gl.viewport(0, 0, common::W as i32, common::H as i32);
            gl.blend_func(glow::ONE, glow::ONE_MINUS_SRC_ALPHA);

            let t = Instant::now();
            let tri = Self::program(&gl, &[(glow::VERTEX_SHADER, common::TRI_VS), (glow::FRAGMENT_SHADER, common::TRI_FS)])?;
            let fill = Self::program(&gl, &[(glow::VERTEX_SHADER, common::FILL_VS), (glow::FRAGMENT_SHADER, common::FILL_FS)])?;
            let compute = Self::program(&gl, &[(glow::COMPUTE_SHADER, common::COMPUTE)])?;
            eprintln!("GL: 3 programs (iris) in {:?}", t.elapsed());

            let mut buffers = Vec::new();
            let mut buffer = |target, bytes: &[u8]| -> Result<glow::Buffer, String> {
                let b = gl.create_buffer()?;
                gl.bind_buffer(target, Some(b));
                gl.buffer_data_u8_slice(target, bytes, glow::STATIC_DRAW);
                buffers.push(b);
                Ok(b)
            };
            let le = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            let triangle = buffer(glow::ARRAY_BUFFER, &le(&common::TRIANGLE))?;
            let quad = buffer(glow::ARRAY_BUFFER, &le(&common::QUAD))?;
            let full = buffer(glow::ARRAY_BUFFER, &common::instances())?;
            let out = buffer(glow::SHADER_STORAGE_BUFFER, &vec![0; *common::LANES.iter().max().unwrap() as usize * 4])?;

            let vao = |attrs: &[(glow::Buffer, u32, i32, i32, i32, bool)]| -> Result<glow::VertexArray, String> {
                let v = gl.create_vertex_array()?;
                gl.bind_vertex_array(Some(v));
                for &(b, loc, n, stride, offset, instance) in attrs {
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(b));
                    gl.vertex_attrib_pointer_f32(loc, n, glow::FLOAT, false, stride, offset);
                    gl.enable_vertex_attrib_array(loc);
                    gl.vertex_attrib_divisor(loc, u32::from(instance));
                }
                Ok(v)
            };
            let tri_vao = vao(&[(triangle, 0, 2, 8, 0, false)])?;
            let fill_vao = vao(&[(quad, 0, 2, 8, 0, false), (full, 1, 4, 32, 0, true), (full, 2, 4, 32, 16, true)])?;
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(out));

            Ok(Gl {
                name: format!("OpenGL ({renderer})"),
                egl,
                display,
                context,
                gl,
                fences: [None; common::RING],
                next: 0,
                fbo,
                texture,
                tri: (tri, tri_vao),
                fill: (fill, fill_vao),
                compute,
                buffers,
                out,
            })
        }
    }

    unsafe fn program(gl: &glow::Context, stages: &[(u32, &str)]) -> Result<glow::Program, String> {
        unsafe {
            let p = gl.create_program()?;
            let mut shaders = Vec::new();
            for &(kind, src) in stages {
                let s = gl.create_shader(kind)?;
                gl.shader_source(s, src);
                gl.compile_shader(s);
                if !gl.get_shader_compile_status(s) {
                    return Err(gl.get_shader_info_log(s));
                }
                gl.attach_shader(p, s);
                shaders.push(s);
            }
            gl.link_program(p);
            if !gl.get_program_link_status(p) {
                return Err(gl.get_program_info_log(p));
            }
            for s in shaders {
                gl.delete_shader(s);
            }
            Ok(p)
        }
    }

    unsafe fn fence(&mut self) -> glow::Fence {
        unsafe {
            let f = self.gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0).expect("fence");
            self.gl.flush();
            f
        }
    }

    unsafe fn wait_fence(&self, f: glow::Fence) {
        unsafe {
            while self.gl.client_wait_sync(f, glow::SYNC_FLUSH_COMMANDS_BIT, WAIT_NS) == glow::TIMEOUT_EXPIRED {}
            self.gl.delete_sync(f);
        }
    }
}

impl Api for Gl {
    type Done = usize;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn frame(&mut self, scene: Scene) -> (usize, Option<Duration>) {
        unsafe {
            let i = self.next % common::RING;
            self.next += 1;
            if let Some(f) = self.fences[i].take() {
                self.wait_fence(f);
            }
            let gl = &self.gl;
            let (clear, (program, vao)) = match scene {
                Scene::Tri => (common::BLUE, self.tri),
                Scene::Fill => (common::FILL_CLEAR, self.fill),
            };
            gl.disable(glow::BLEND);
            gl.clear_color(clear[0], clear[1], clear[2], clear[3]);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(vao));
            match scene {
                Scene::Tri => gl.draw_arrays(glow::TRIANGLES, 0, 3),
                Scene::Fill => {
                    gl.enable(glow::BLEND);
                    gl.draw_arrays_instanced(glow::TRIANGLE_STRIP, 0, 4, common::LAYERS as i32);
                }
            }
            self.fences[i] = Some(self.fence());
            (i, None)
        }
    }

    fn wait(&mut self, done: usize) {
        if let Some(f) = self.fences[done].take() {
            unsafe { self.wait_fence(f) };
        }
    }

    fn pixels(&mut self) -> Vec<u8> {
        let mut px = vec![0; (common::W * common::H * 4) as usize];
        unsafe {
            self.gl.read_pixels(0, 0, common::W as i32, common::H as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut px)));
        }
        px
    }

    fn dispatch(&mut self, lanes: u32, count: u32) -> Option<Duration> {
        unsafe {
            self.gl.use_program(Some(self.compute));
            for _ in 0..count {
                self.gl.dispatch_compute(lanes / 16, 1, 1);
                self.gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT | glow::BUFFER_UPDATE_BARRIER_BIT);
            }
            let f = self.fence();
            self.wait_fence(f);
        }
        None
    }

    fn words(&mut self, lanes: u32) -> Vec<u32> {
        let mut b = vec![0; lanes as usize * 4];
        unsafe {
            self.gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.out));
            self.gl.get_buffer_sub_data(glow::SHADER_STORAGE_BUFFER, 0, &mut b);
        }
        b.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect()
    }
}

impl Drop for Gl {
    fn drop(&mut self) {
        unsafe {
            self.gl.finish();
            for f in self.fences.iter_mut().filter_map(Option::take) {
                self.gl.delete_sync(f);
            }
            for b in &self.buffers {
                self.gl.delete_buffer(*b);
            }
            for (p, v) in [self.tri, self.fill] {
                self.gl.delete_program(p);
                self.gl.delete_vertex_array(v);
            }
            self.gl.delete_program(self.compute);
            self.gl.delete_framebuffer(self.fbo);
            self.gl.delete_texture(self.texture);
            self.egl.make_current(self.display, None, None, None).ok();
            self.egl.destroy_context(self.display, self.context).ok();
            self.egl.terminate(self.display).ok();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = common::frames_arg();
    let xe = common::Detour::device()?;
    let mut detour = common::Detour::new(&xe)?;
    let a = Report::run(&mut detour, n);
    drop(detour);
    let mut gl = Gl::new()?;
    let b = Report::run(&mut gl, n);
    Report::table(&[a, b], n);
    Ok(())
}
