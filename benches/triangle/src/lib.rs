//! A window detour draws into: each frame is recorded on Xe3 into a render target the size of the window, read back,
//! and shown through softbuffer (a CPU copy; zero-copy into the compositor is acde's job, not this example's).
//!
//! A [`Scene`] says what is drawn after the clear; [`Shown::run`] opens the window and drives it (`Xe::open_first`
//! gives it the GPU). Esc or closing the
//! window quits.

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;

use detour::base::{MemType, Native, TextureFormat};
use detour::xe3::{Buffer, Codegen, Context, Error, Pipeline, Recorder, Target};
use detour::xe::Xe;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// The clear color, as the target holds it (RGBA8).
pub const BACKGROUND: [u8; 4] = [26, 26, 31, 255];

/// ⭐ What a window shows: made once with the context, recorded every frame.
pub trait Scene<'d>: Sized {
    const TITLE: &'static str;

    fn new(cx: &mut Context<'d, Xe, Native>, codegen: &Codegen<'_>) -> Result<Self, Error>;

    /// Draws after the clear. `seconds` since the first frame (0 in it); `size` the target's.
    fn record(&mut self, r: &mut Recorder<'_, 'd, Xe, Native>, seconds: f32, size: (u32, u32)) -> Result<(), Error>;

    /// Keeps redrawing (an animation) instead of drawing only when the window needs it.
    fn animated(&self) -> bool {
        false
    }

    /// Checks the first frame's pixels (RGBA8 rows), so a wrong picture fails instead of just showing.
    fn check(&self, pixels: &[u8], size: (u32, u32)) -> Result<(), String>;
}

/// The window, its surface and the target behind it.
struct Window2<'d> {
    window: Rc<Window>,
    surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    target: Option<(Target<'d, Xe>, (u32, u32))>,
}

/// ⭐ A scene in a window.
pub struct Shown<'d, S: Scene<'d>> {
    cx: Context<'d, Xe, Native>,
    clear: Pipeline,
    corners: Buffer<'d, Xe>,
    scene: S,
    shown: Option<Window2<'d>>,
    /// when the first frame was drawn: frames count time from it
    start: Option<Instant>,
    checked: bool,
    failed: Option<String>,
}

impl<'d, S: Scene<'d>> Shown<'d, S> {
    /// Compiles the scene on `xe` and runs the window until it closes.
    pub fn run(xe: &'d Xe) -> Result<(), Box<dyn std::error::Error>> {
        let codegen = Codegen::default();
        let mut cx = Context::<Xe, Native>::new(xe)?;
        let rgba = BACKGROUND.map(|c| f32::from(c) / 255.0);
        let clear = cx.clear_pipeline(&codegen, rgba)?;
        let corners = cx.clear_vertices()?;
        let scene = S::new(&mut cx, &codegen)?;
        let mut shown = Shown { cx, clear, corners, scene, shown: None, start: None, checked: false, failed: None };
        EventLoop::new()?.run_app(&mut shown)?;
        match shown.failed {
            Some(e) => Err(e.into()),
            None => Ok(()),
        }
    }

    /// Records, submits and waits for one frame, then copies it into the window.
    fn frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(w) = &mut self.shown else { return Ok(()) };
        let size = w.window.inner_size();
        let (Some(width), Some(height)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) else { return Ok(()) };
        let size = (width.get(), height.get());
        if w.target.as_ref().is_none_or(|(_, s)| *s != size) {
            w.target = Some((self.cx.target(size.0, size.1, TextureFormat::Rgba8Unorm, MemType::Coherent)?, size));
            w.surface.resize(width, height)?;
        }
        let Some((target, _)) = &mut w.target else { return Ok(()) };
        let mut r = self.cx.record();
        r.target(target);
        r.draw(&self.clear, &[&self.corners], &[], 3, 1)?;
        let seconds = self.start.get_or_insert_with(Instant::now).elapsed().as_secs_f32();
        self.scene.record(&mut r, seconds, size)?;
        let frame = r.finish();
        let n = self.cx.submit(frame)?;
        self.cx.wait(n)?;
        let pixels = target.bytes()?;
        if !self.checked {
            self.scene.check(pixels, size)?;
            self.checked = true;
            println!("{}: first frame {}x{} checked", S::TITLE, size.0, size.1);
        }
        let mut buffer = w.surface.buffer_mut()?;
        for (to, p) in buffer.iter_mut().zip(pixels.as_chunks::<4>().0) {
            *to = u32::from_be_bytes([0, p[0], p[1], p[2]]);
        }
        buffer.present()?;
        Ok(())
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, e: impl ToString) {
        self.failed = Some(e.to_string());
        event_loop.exit();
    }
}

impl<'d, S: Scene<'d>> ApplicationHandler for Shown<'d, S> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.shown.is_some() {
            return;
        }
        let opened = event_loop.create_window(Window::default_attributes().with_title(S::TITLE)).map(Rc::new).map_err(|e| e.to_string()).and_then(|window| {
            let context = softbuffer::Context::new(window.clone()).map_err(|e| e.to_string())?;
            let surface = softbuffer::Surface::new(&context, window.clone()).map_err(|e| e.to_string())?;
            Ok(Window2 { window, surface, target: None })
        });
        match opened {
            Ok(w) => self.shown = Some(w),
            Err(e) => self.fail(event_loop, e),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Named(NamedKey::Escape), state: ElementState::Pressed, .. }, .. } => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame() {
                    return self.fail(event_loop, e);
                }
                if self.scene.animated()
                    && let Some(w) = &self.shown
                {
                    w.window.request_redraw();
                }
            }
            WindowEvent::Resized(_) => {
                if let Some(w) = &self.shown {
                    w.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
