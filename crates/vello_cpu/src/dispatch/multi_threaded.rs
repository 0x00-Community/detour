use crate::vello_cpu::RasterizerSettings;
use crate::vello_cpu::coarse::CommandBucketer;
use crate::vello_cpu::coarse::depth::DepthBuffer;
use crate::vello_cpu::dispatch::Dispatcher;
use crate::vello_cpu::dispatch::multi_threaded::cost::{COST_THRESHOLD, estimate_render_task_cost};
use crate::vello_cpu::dispatch::multi_threaded::worker::Worker;
use crate::vello_cpu::filter::context::FilterContext;
use crate::vello_cpu::fine::{Fine, FineKernel, FineRenderParams, FineResources, rasterize_region};
use crate::vello_cpu::kurbo::{Affine, BezPath, PathEl, Rect, Stroke};
use crate::vello_cpu::peniko::{BlendMode, Fill};
use crate::vello_cpu::record::RecordedFill;
use crate::vello_cpu::region::Regions;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use crossbeam_channel::TryRecvError;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::cell::RefCell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Barrier, Mutex};
use thread_local::ThreadLocal;
use vello_common::clip::{ClipContext, ClipShape};
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::{Level, Simd, dispatch};
use vello_common::filter::FilterData;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint, PremulColor};
use vello_common::pixmap::PixmapMut;
use vello_common::record::{CommandRecorder, LayerClip, LayerProps, PoppedLayer};
use vello_common::strip::Strip;
use vello_common::strip_generator::{GenerationMode, StripGenerator, StripStorage};

mod cost;
mod worker;

type RenderTaskSender = crossbeam_channel::Sender<RenderTask>;
type RecordedCommandSender = ordered_channel::Sender<RecordedCommandTask>;
type RecordedCommandReceiver = ordered_channel::Receiver<RecordedCommandTask>;

pub(crate) struct MultiThreadedDispatcher {
    bucketer: Mutex<CommandBucketer>,
    clip_context: ClipContext,
    recorder: CommandRecorder<RecordedFill>,
    strip_storage: StripStorage,
    thread_pool: ThreadPool,
    allocation_group: AllocationGroup,
    batch_cost: f32,
    task_sender: Option<RenderTaskSender>,
    workers: Arc<ThreadLocal<RefCell<Worker>>>,
    recorded_command_receiver: Option<RecordedCommandReceiver>,
    alpha_storage: MaybePresent<Vec<Vec<u8>>>,
    task_idx: u32,
    num_threads: u16,
    strip_generator: StripGenerator,
    level: Level,
    flushed: bool,
    allocations: Allocations,
    layer_depth: usize,
}

impl MultiThreadedDispatcher {
    pub(crate) fn new(width: u16, height: u16, num_threads: u16, level: Level) -> Self {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build()
            .unwrap();
        let alpha_storage = MaybePresent::new(vec![vec![]; usize::from(num_threads)]);
        let workers = Arc::new(ThreadLocal::new());

        {
            let thread_ids = Arc::new(AtomicU8::new(0));
            let workers = workers.clone();

            thread_pool.spawn_broadcast(move |_| {
                let thread_id = thread_ids.fetch_add(1, Ordering::SeqCst);
                let worker = Worker::new(width, height, thread_id, level);

                let _ = workers.get_or(|| RefCell::new(worker));
            });
        }

        let task_idx = 0;
        let batch_cost = 0.0;
        let flushed = true;

        Self {
            bucketer: Mutex::new(CommandBucketer::from_wh(width, height)),
            thread_pool,
            allocations: Allocations::default(),
            allocation_group: AllocationGroup::default(),
            batch_cost,
            task_idx,
            flushed,
            workers,
            clip_context: ClipContext::new(),
            recorder: CommandRecorder::new(width, height),
            task_sender: None,
            recorded_command_receiver: None,
            strip_generator: StripGenerator::new(width, height, level),
            strip_storage: StripStorage::new(GenerationMode::Append),
            level,
            alpha_storage,
            num_threads,
            layer_depth: 0,
        }
    }

    #[cfg(all())]
    fn rasterize_f32(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::vello_cpu::fine::F32Kernel;
        dispatch!(self.level, simd => self.rasterize_with::<_, F32Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    #[cfg(all())]
    fn rasterize_u8(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        use crate::vello_cpu::fine::U8Kernel;
        dispatch!(self.level, simd => self.rasterize_with::<_, U8Kernel>(simd, target, scene_width, scene_height, settings, encoded_paints, image_resolver));
    }

    fn init(&mut self) {
        let (render_task_sender, render_task_receiver) = crossbeam_channel::unbounded();
        let (recorded_command_sender, recorded_command_receiver) = ordered_channel::unbounded();
        let workers = self.workers.clone();
        let alpha_storage = self.alpha_storage.clone();

        self.task_sender = Some(render_task_sender);
        self.recorded_command_receiver = Some(recorded_command_receiver);

        self.thread_pool.spawn_broadcast(move |_| {
            let render_task_receiver = render_task_receiver.clone();
            let mut recorded_command_sender = recorded_command_sender.clone();
            let worker = workers.get().unwrap();
            let mut worker = worker.borrow_mut();
            let thread_id = worker.thread_id();

            alpha_storage
                .with_inner(|alphas| worker.init(std::mem::take(&mut alphas[thread_id as usize])));

            while let Ok(task) = render_task_receiver.recv() {
                worker.run_render_task(task, &mut recorded_command_sender);
            }

            alpha_storage.with_inner(|alphas| {
                alphas[thread_id as usize] = worker.finalize();
            });

            drop(recorded_command_sender);
        });
    }

    fn register_task(&mut self, task: RenderTaskType) {
        self.flushed = false;
        if self.task_sender.is_none() {
            self.init();
        }

        let cost = estimate_render_task_cost(&task, &self.allocation_group.path);
        self.allocation_group.render_tasks.push(task);
        self.batch_cost += cost;

        if self.batch_cost > COST_THRESHOLD {
            self.flush_tasks();
        }
    }

    fn flush_tasks(&mut self) {
        if self.allocation_group.render_tasks.is_empty() {
            return;
        }

        self.send_pending_tasks();

        self.batch_cost = 0.0;
    }

    fn bump_task_idx(&mut self) -> u32 {
        let idx = self.task_idx;
        self.task_idx += 1;
        idx
    }

    fn send_pending_tasks(&mut self) {
        let task_idx = self.bump_task_idx();
        let allocation_group =
            std::mem::replace(&mut self.allocation_group, self.allocations.get());
        let task_sender = self.task_sender.as_mut().unwrap();
        let clip_path = self.clip_context.get().map(|c| OwnedClip {
            strips: c.path.strips.into(),
            alphas: c.path.alphas.into(),
            bbox: c.path.bbox,
            shape: c.shape,
        });
        let task = RenderTask {
            idx: task_idx,
            clip_path,
            allocation_group,
        };
        task_sender.send(task).unwrap();
        self.record_finished_commands(true);
    }

    fn append_strips(&mut self, strips: &[Strip]) -> Range<usize> {
        let start = self.strip_storage.strips.len();
        self.strip_storage.strips.extend_from_slice(strips);
        start..self.strip_storage.strips.len()
    }

    fn record_finished_commands(&mut self, abort_empty: bool) {
        loop {
            match self.recorded_command_receiver.as_mut().unwrap().try_recv() {
                Ok(mut task) => {
                    let num_tasks = task.allocation_group.recorded_commands.len();
                    for cmd in task.allocation_group.recorded_commands.drain(0..num_tasks) {
                        match cmd {
                            RecordedCommand::RenderPath {
                                strips: strip_range,
                                paint,
                                blend_mode,
                                thread_id,
                                mask,
                            } => {
                                let strip_range = self.append_strips(
                                    &task.allocation_group.strips
                                        [strip_range.start as usize..strip_range.end as usize],
                                );
                                let strips =
                                    &self.strip_storage.strips[strip_range.start..strip_range.end];
                                let draw = RecordedFill::new(
                                    thread_id,
                                    strip_range,
                                    paint.clone(),
                                    blend_mode,
                                    mask,
                                );

                                self.recorder.push_draw(draw, strips);
                            }
                            RecordedCommand::PushLayer {
                                thread_id,
                                clip_path,
                                clip_bbox,
                                blend_mode,
                                mask,
                                opacity,
                            } => {
                                let clip_path = clip_path.map(|strip_range| {
                                    let strip_range = self.append_strips(
                                        &task.allocation_group.strips
                                            [strip_range.start as usize..strip_range.end as usize],
                                    );
                                    LayerClip {
                                        strip_range,
                                        thread_idx: thread_id,
                                        bbox: clip_bbox.unwrap(),
                                    }
                                });

                                self.recorder.push_layer(
                                    LayerProps {
                                        blend_mode,
                                        opacity,
                                        mask,
                                        clip_path,
                                    },
                                    None,
                                );
                            }
                            RecordedCommand::PopLayer => match self.recorder.pop_layer() {
                                PoppedLayer::Regular => {}
                                PoppedLayer::Filter => {
                                    unreachable!("filters are not supported by MT")
                                }
                            },
                        }
                    }

                    self.allocations.put(task.allocation_group);
                }
                Err(e) => match e {
                    TryRecvError::Empty => {
                        if abort_empty {
                            return;
                        }
                    }
                    TryRecvError::Disconnected => return,
                },
            }
        }
    }

    fn rasterize_with<S: Simd, F: FineKernel<S>>(
        &self,
        simd: S,
        mut target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        let mut bucketer = self.bucketer.lock().unwrap();
        let filters = FilterContext::new(0);
        bucketer.reset(RectU16::new(0, 0, scene_width, scene_height));
        let target_init = settings.target_init.map(PremulColor::from_alpha_color);
        bucketer.bucket_commands(
            &self.recorder.nodes,
            &self.recorder.draws,
            &self.recorder.layers,
            &self.strip_storage.strips,
            encoded_paints,
            &filters,
        );

        let alpha_slots = self.alpha_storage.take();
        {
            let alpha_buffers = alpha_slots.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let resources = FineResources {
                alpha_buffers: &alpha_buffers,
                encoded_paints,
                filter_paints: &bucketer.filter_paints,
                image_resolver,
            };
            let params = FineRenderParams {
                scene_size: (scene_width, scene_height),
                target_offset: settings.offset,
            };

            let mut regions = Regions::new(
                &mut target,
                params.scene_size,
                params.target_offset,
                bucketer.rows().len(),
            );
            let fines = ThreadLocal::new();
            self.thread_pool.install(|| {
                regions.update_par(|region| {
                    let mut fine = fines
                        .get_or(|| {
                            RefCell::new((
                                Fine::<S, F>::new(simd, bucketer.width()),
                                DepthBuffer::new(bucketer.width()),
                            ))
                        })
                        .borrow_mut();
                    let (fine, depth) = &mut *fine;

                    rasterize_region::<S, F>(
                        fine,
                        depth,
                        region,
                        &bucketer,
                        resources,
                        target_init,
                        self.recorder.root_is_blend_target,
                    );
                });
            });
        }

        self.alpha_storage.init(alpha_slots);
    }
}

impl Dispatcher for MultiThreadedDispatcher {
    fn has_layers(&self) -> bool {
        self.layer_depth != 0
    }

    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let start = self.allocation_group.path.len() as u32;
        self.allocation_group.path.extend(path);
        let end = self.allocation_group.path.len() as u32;
        self.register_task(RenderTaskType::FillPath {
            path_range: start..end,
            transform,
            paint,
            fill_rule,
            blend_mode,
            aliasing_threshold,
            mask,
        });
    }

    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    ) {
        let start = self.allocation_group.path.len() as u32;
        self.allocation_group.path.extend(path);
        let end = self.allocation_group.path.len() as u32;
        self.register_task(RenderTaskType::StrokePath {
            path_range: start..end,
            transform,
            paint,
            stroke: stroke.clone(),
            blend_mode,
            aliasing_threshold,
            mask,
        });
    }

    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) {
        self.register_task(RenderTaskType::FillRect {
            rect: *rect,
            paint,
            blend_mode,
            mask,
        });
    }

    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter_data: Option<FilterData>,
    ) {
        if filter_data.is_some() {
            unimplemented!("Filter effects are not yet supported in multi-threaded rendering");
        }

        let clip_path = clip_path.map(|c| {
            let start = self.allocation_group.path.len() as u32;
            self.allocation_group.path.extend(c);
            let end = self.allocation_group.path.len() as u32;

            (start..end, clip_transform)
        });

        self.register_task(RenderTaskType::PushLayer {
            clip_path,
            blend_mode,
            opacity,
            mask,
            fill_rule,
            aliasing_threshold,
        });
        self.layer_depth += 1;
    }

    fn pop_layer(&mut self) {
        self.register_task(RenderTaskType::PopLayer);
        self.layer_depth = self
            .layer_depth
            .checked_sub(1)
            .expect("layer stack underflow");
    }

    fn reset(&mut self, width: u16, height: u16) {
        self.flush();

        self.clip_context.reset();
        self.recorder.reset(width, height);
        self.strip_storage.clear();
        self.allocation_group.clear();
        self.batch_cost = 0.0;
        self.task_idx = 0;
        self.layer_depth = 0;
        self.task_sender = None;
        self.recorded_command_receiver = None;
        self.strip_generator.reset(width, height);
        self.alpha_storage.with_inner(|alphas| {
            for alpha in alphas {
                alpha.clear();
            }
        });

        let workers = self.workers.clone();
        let barrier = Arc::new(Barrier::new(usize::from(self.num_threads) + 1));
        let t_barrier = barrier.clone();

        self.thread_pool.spawn_broadcast(move |_| {
            let worker = workers.get().unwrap();
            let mut borrowed = worker.borrow_mut();
            borrowed.reset(width, height);
            t_barrier.wait();
        });

        barrier.wait();
    }

    fn flush(&mut self) {
        if self.flushed {
            return;
        }

        self.flush_tasks();
        let sender = core::mem::take(&mut self.task_sender);
        drop(sender);
        self.record_finished_commands(false);

        self.flushed = true;
    }

    fn rasterize(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    ) {
        assert!(self.flushed, "attempted to rasterize before flushing");

        #[cfg(all(all(), not(all())))]
        {
            self.rasterize_u8(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }
        #[cfg(all(all(), not(all())))]
        {
            self.rasterize_f32(
                target,
                scene_width,
                scene_height,
                settings,
                encoded_paints,
                image_resolver,
            );
        }

        #[cfg(all(all(), all()))]
        match settings.render_mode {
            crate::vello_cpu::RenderMode::OptimizeSpeed => {
                self.rasterize_u8(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
            crate::vello_cpu::RenderMode::OptimizeQuality => {
                self.rasterize_f32(
                    target,
                    scene_width,
                    scene_height,
                    settings,
                    encoded_paints,
                    image_resolver,
                );
            }
        }
    }

    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    ) {
        self.flush_tasks();
        self.clip_context.push_clip_path(
            path.iter(),
            &mut self.strip_generator,
            fill_rule,
            transform,
            aliasing_threshold,
        );
    }

    fn push_clip_rect(&mut self, rect: &Rect) {
        self.flush_tasks();
        self.clip_context
            .push_clip_rect(rect, &mut self.strip_generator);
    }

    fn pop_clip(&mut self) {
        self.flush_tasks();
        self.clip_context.pop_clip();
    }

    fn is_multi_threaded(&self) -> bool {
        true
    }
}

impl Debug for MultiThreadedDispatcher {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str("MultiThreadedDispatcher { .. }")
    }
}

impl Drop for MultiThreadedDispatcher {
    fn drop(&mut self) {
        self.flush();
    }
}

#[derive(Debug)]
pub(crate) struct OwnedClip {
    strips: Box<[Strip]>,
    alphas: Box<[u8]>,
    bbox: RectU16,
    shape: ClipShape,
}

struct AllocationManager<T> {
    entries: Vec<Vec<T>>,
}

impl<T> AllocationManager<T> {
    fn get(&mut self) -> Vec<T> {
        self.entries.pop().unwrap_or_default()
    }

    fn put(&mut self, mut allocation: Vec<T>) {
        allocation.clear();
        self.entries.push(allocation);
    }
}

impl<T> Default for AllocationManager<T> {
    fn default() -> Self {
        Self { entries: vec![] }
    }
}

#[derive(Default)]
struct Allocations {
    render_tasks: AllocationManager<RenderTaskType>,
    paths: AllocationManager<PathEl>,
    strips: AllocationManager<Strip>,
    recorded_commands: AllocationManager<RecordedCommand>,
}

impl Allocations {
    fn get(&mut self) -> AllocationGroup {
        let render_tasks = self.render_tasks.get();
        let path = self.paths.get();
        let strips = self.strips.get();
        let recorded_commands = self.recorded_commands.get();

        AllocationGroup {
            path,
            render_tasks,
            recorded_commands,
            strips,
        }
    }

    fn put(&mut self, allocation: AllocationGroup) {
        self.render_tasks.put(allocation.render_tasks);
        self.paths.put(allocation.path);
        self.strips.put(allocation.strips);
        self.recorded_commands.put(allocation.recorded_commands);
    }
}

#[derive(Default, Debug)]
pub(crate) struct AllocationGroup {
    pub(crate) path: Vec<PathEl>,
    pub(crate) render_tasks: Vec<RenderTaskType>,
    pub(crate) strips: Vec<Strip>,
    pub(crate) recorded_commands: Vec<RecordedCommand>,
}

impl AllocationGroup {
    fn clear(&mut self) {
        self.path.clear();
        self.render_tasks.clear();
        self.strips.clear();
        self.recorded_commands.clear();
    }
}

#[derive(Debug)]
pub(crate) struct RenderTask {
    pub(crate) idx: u32,
    pub(crate) clip_path: Option<OwnedClip>,
    pub(crate) allocation_group: AllocationGroup,
}

#[derive(Debug, Clone)]
pub(crate) enum RenderTaskType {
    FillRect {
        rect: Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    },
    FillPath {
        path_range: Range<u32>,
        transform: Affine,
        paint: Paint,
        fill_rule: Fill,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    },
    StrokePath {
        path_range: Range<u32>,
        transform: Affine,
        paint: Paint,
        stroke: Stroke,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    },
    PushLayer {
        clip_path: Option<(Range<u32>, Affine)>,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
    },
    PopLayer,
}

pub(crate) struct RecordedCommandTask {
    allocation_group: AllocationGroup,
}

#[derive(Debug)]
pub(crate) enum RecordedCommand {
    RenderPath {
        thread_id: u8,
        strips: Range<u32>,
        blend_mode: BlendMode,
        paint: Paint,
        mask: Option<Mask>,
    },
    PushLayer {
        thread_id: u8,
        clip_path: Option<Range<u32>>,
        clip_bbox: Option<RectU16>,
        blend_mode: BlendMode,
        mask: Option<Mask>,
        opacity: f32,
    },
    PopLayer,
}

#[derive(Clone)]
pub(crate) struct MaybePresent<T: Default> {
    present: Arc<AtomicBool>,
    value: Arc<Mutex<T>>,
}

impl<T: Default> MaybePresent<T> {
    pub(crate) fn new(val: T) -> Self {
        Self {
            present: Arc::new(AtomicBool::new(true)),
            value: Arc::new(Mutex::new(val)),
        }
    }

    pub(crate) fn init(&self, value: T) {
        let mut locked = self.value.lock().unwrap();
        *locked = value;
        self.present.store(true, Ordering::SeqCst);
    }

    pub(crate) fn with_inner(&self, mut func: impl FnMut(&mut T)) {
        assert!(
            self.present.load(Ordering::SeqCst),
            "Tried to access `MaybePresent` before initialization."
        );

        let mut lock = self.value.lock().unwrap();
        func(&mut lock);
    }

    pub(crate) fn take(&self) -> T {
        assert!(
            self.present.load(Ordering::SeqCst),
            "Tried to access `MaybePresent` before initialization."
        );

        let mut locked = self.value.lock().unwrap();
        self.present.store(false, Ordering::SeqCst);
        std::mem::take(&mut *locked)
    }
}

#[cfg(any())]
mod tests {
    use crate::vello_cpu::Level;
    use crate::vello_cpu::color::palette::css::BLUE;
    use crate::vello_cpu::dispatch::Dispatcher;
    use crate::vello_cpu::dispatch::multi_threaded::{MultiThreadedDispatcher, RenderTaskType};
    use crate::vello_cpu::kurbo::{Affine, Rect, Shape};
    use crate::vello_cpu::peniko::{BlendMode, Fill};
    use vello_common::paint::{Paint, PremulColor};

    #[test]
    fn allocations() {
        let mut dispatcher = MultiThreadedDispatcher::new(100, 100, 4, Level::new());
        for _ in 0..20 {
            dispatcher.fill_path(
                &Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1),
                Fill::NonZero,
                Affine::IDENTITY,
                Paint::Solid(PremulColor::from_alpha_color(BLUE)),
                BlendMode::default(),
                None,
                None,
            );
            dispatcher.flush();
        }

        assert_eq!(dispatcher.allocations.paths.entries.len(), 1);
        assert_eq!(dispatcher.allocations.strips.entries.len(), 1);
        assert_eq!(dispatcher.allocations.render_tasks.entries.len(), 1);
        assert_eq!(dispatcher.allocations.recorded_commands.entries.len(), 1);
    }

    #[test]
    fn fast_rect_is_dispatched_without_path_elements() {
        let mut dispatcher = MultiThreadedDispatcher::new(100, 100, 2, Level::new());
        let rect = Rect::new(10.25, 20.5, 70.75, 80.0);

        dispatcher.fill_rect_fast(
            &rect,
            Paint::Solid(PremulColor::from_alpha_color(BLUE)),
            BlendMode::default(),
            None,
        );

        assert!(dispatcher.allocation_group.path.is_empty());
        assert!(matches!(
            dispatcher.allocation_group.render_tasks.as_slice(),
            [RenderTaskType::FillRect { rect: actual, .. }] if *actual == rect
        ));
    }
}
