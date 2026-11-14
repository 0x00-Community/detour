//! ⭐ detour's Apple M3 / M4 (Apple9) runtime on macOS: mlirpass's Apple9 machine code (`Apple9Code`, made by
//! detour-apple-codegen's `apple9`) run on the GPU through Metal.
//!
//! macOS's GPU driver (AGXG16G, IOKit's IOGPU) takes work only through Metal; its own submission interface is
//! Apple's, undocumented. So the code goes in where Metal looks for a pipeline's: a binary archive. A donor kernel
//! (src/donor.metal: binding 0 the memory, binding 1 the parameter words, many live registers, 16 KiB of threadgroup
//! memory) is compiled into an archive by build.rs (another process: a pipeline compiled in this one would come back
//! from Metal's cache instead of the archive, measured); for each kernel the archive's copy has the donor's main (`_agc.main` in
//! the `applegpu` slice's GPU Mach-O, `__TEXT,__text`) replaced by the code, padded with `halt`, and the pipeline is
//! made from it with `failOnBinaryArchiveMiss`: Metal runs what the archive holds (no signature on it). Everything
//! the GPU runs is detour's; Metal only binds the two buffers and dispatches (mlirpass's Apple9 ABI:
//! targets/apple9.rs).
//!
//! ```text
//!   M4::open()                       the device (an Apple9 GPU), the donor's archive
//!   M4::kernel(&Apple9Code)          a pipeline running that code
//!   M4::buffer(bytes)                memory the CPU and the GPU share (binding 0: byte addresses into it)
//!   M4::dispatch(&kernel, &memory, params, groups)
//!                                    the parameter words, then the grid's size in workgroups (binding 1); groups of
//!                                    the kernel's workgroup (32, or 16: two a hardware workgroup), waited for
//! ```

#![cfg(target_os = "macos")]

use core::ptr::NonNull;
use std::cell::Cell;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTLBinaryArchiveDescriptor, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction, MTLLibrary, MTLPipelineOption, MTLResourceOptions, MTLSize,
};

pub use mlirpass::Apple9Code;

/// What went wrong, said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

fn err(m: impl Into<String>) -> Error {
    Error(m.into())
}

/// The donor kernel the code runs in (as mlirpass's scripts/apple9_donor.metal).
const DONOR: &str = include_str!("donor.metal");
/// Its pipeline's archive, serialized by build.rs (empty without an Apple9 GPU there).
const DONOR_ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/donor.bin"));
/// The donor's name.
const DONOR_NAME: &str = "donor";
/// The fat file's `applegpu` slice (CPU type).
const APPLEGPU: u32 = 0x0100_0013;
/// Threads of a hardware workgroup.
const SIMD: usize = 32;

/// A section of a 64-bit Mach-O: where its bytes are in the file.
struct Section {
    seg: String,
    sect: String,
    offset: usize,
    size: usize,
}

/// The sections and symbols of the 64-bit Mach-O at `base` in `b`.
struct MachO {
    sections: Vec<Section>,
    symbols: Vec<(String, usize)>,
}

impl MachO {
    fn read(b: &[u8], base: usize) -> Option<MachO> {
        let u32_at = |p: usize| -> Option<usize> { Some(u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?) as usize) };
        let u64_at = |p: usize| -> Option<usize> { Some(u64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?) as usize) };
        let str_at = |p: usize, n: usize| -> Option<String> { Some(String::from_utf8_lossy(b.get(p..p + n)?.split(|c| *c == 0).next()?).into_owned()) };
        if u32_at(base)? != 0xfeed_facf {
            return None;
        }
        let (mut sections, mut symbols) = (Vec::new(), Vec::new());
        let mut p = base + 32;
        for _ in 0..u32_at(base + 16)? {
            let (cmd, size) = (u32_at(p)?, u32_at(p + 4)?);
            match cmd {
                // LC_SEGMENT_64
                0x19 => {
                    for s in 0..u32_at(p + 64)? {
                        let q = p + 72 + 80 * s;
                        sections.push(Section { seg: str_at(q + 16, 16)?, sect: str_at(q, 16)?, offset: base + u32_at(q + 48)?, size: u64_at(q + 40)? });
                    }
                }
                // LC_SYMTAB
                0x2 => {
                    let (symoff, nsyms, stroff) = (u32_at(p + 8)?, u32_at(p + 12)?, u32_at(p + 16)?);
                    for k in 0..nsyms {
                        let e = base + symoff + 16 * k;
                        let n = base + stroff + u32_at(e)?;
                        let name = b.get(n..)?.split(|c| *c == 0).next()?;
                        symbols.push((String::from_utf8_lossy(name).into_owned(), u64_at(e + 8)?));
                    }
                }
                _ => {}
            }
            p += size;
        }
        Some(MachO { sections, symbols })
    }

    fn section(&self, seg: &str, sect: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.seg == seg && s.sect == sect)
    }
}

/// Where in a serialized archive the one kernel's main is: its file offset and the bytes up to its `__text`'s end.
fn main_of(b: &[u8]) -> Result<(usize, usize)> {
    let be = |p: usize| -> Option<usize> { Some(u32::from_be_bytes(b.get(p..p + 4)?.try_into().ok()?) as usize) };
    let slices = be(4).ok_or_else(|| err("the archive is not a fat file"))?;
    for k in 0..slices {
        let a = 8 + 20 * k;
        if be(a) != Some(APPLEGPU as usize) {
            continue;
        }
        let Some(slice) = MachO::read(b, be(a + 8).unwrap_or(0)) else { continue };
        for s in slice.sections.iter().filter(|s| s.seg == "__TEXT" && s.sect == "__compute") {
            let Some(inner) = MachO::read(b, s.offset) else { continue };
            let text = inner.section("__TEXT", "__text").ok_or_else(|| err("the GPU Mach-O has no __text"))?;
            let main = inner.symbols.iter().find(|(n, _)| n == "_agc.main").map(|(_, at)| *at).ok_or_else(|| err("no _agc.main"))?;
            return Ok((text.offset + main, text.size - main));
        }
    }
    Err(err("no applegpu kernel in the archive"))
}

/// A temporary file, removed when dropped.
struct Temp(std::path::PathBuf);

impl Temp {
    fn new(tag: &str) -> Temp {
        thread_local!(static N: Cell<u64> = const { Cell::new(0) });
        let n = N.with(|c| {
            c.set(c.get() + 1);
            c.get()
        });
        Temp(std::env::temp_dir().join(format!("detour-apple9-{}-{tag}-{n}.bin", std::process::id())))
    }

    fn url(&self) -> Retained<NSURL> {
        NSURL::fileURLWithPath(&NSString::from_str(&self.0.to_string_lossy()))
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// ⭐ An Apple9 GPU (an M3 / M4) through Metal.
pub struct M4 {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    donor: Retained<ProtocolObject<dyn MTLFunction>>,
    /// The donor's archive serialized, and where its main is (file offset, bytes to the end of `__text`).
    archive: Vec<u8>,
    main: (usize, usize),
}

/// ⭐ A pipeline running one kernel's code.
pub struct Kernel {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Invocations of a workgroup: 32, or 16 (two a hardware workgroup).
    pub workgroup: u32,
    /// Its parameter words.
    pub params: u32,
}

/// ⭐ Memory the CPU and the GPU share.
pub struct Buffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

impl Buffer {
    fn ptr(&self) -> NonNull<u8> {
        self.buffer.contents().cast()
    }

    /// Its bytes (not while the GPU runs: [`M4::dispatch`] waits for it).
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: a shared buffer's contents are `len` bytes, mapped for its life; dispatches finish before returning
        unsafe { core::slice::from_raw_parts(self.ptr().as_ptr(), self.len) }
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `bytes`, and `&mut self` is the one view
        unsafe { core::slice::from_raw_parts_mut(self.ptr().as_ptr(), self.len) }
    }

    /// Its little-endian words.
    pub fn words(&self) -> Vec<u32> {
        self.bytes().chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap_or_default())).collect()
    }

    /// `words` written from word `at` on.
    pub fn write_words(&mut self, at: usize, words: &[u32]) {
        for (k, w) in words.iter().enumerate() {
            self.bytes_mut()[4 * (at + k)..4 * (at + k) + 4].copy_from_slice(&w.to_le_bytes());
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl M4 {
    /// ⭐ The system's GPU, if it is an Apple9 one (Metal's `applegpu_g16*`), and the donor's archive.
    pub fn open() -> Result<M4> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| err("no Metal device"))?;
        let arch = device.architecture().name().to_string();
        if !arch.starts_with("applegpu_g16") {
            return Err(err(format!("{arch}: not an Apple9 GPU (an M3 / M4: applegpu_g16)")));
        }
        let queue = device.newCommandQueue().ok_or_else(|| err("no command queue"))?;
        let library = device.newLibraryWithSource_options_error(&NSString::from_str(DONOR), None).map_err(|e| err(format!("the donor: {e}")))?;
        let donor = library.newFunctionWithName(&NSString::from_str(DONOR_NAME)).ok_or_else(|| err("no donor function"))?;
        if DONOR_ARCHIVE.is_empty() {
            return Err(err("no donor archive: built without an Apple9 GPU (build.rs)"));
        }
        let archive = DONOR_ARCHIVE.to_vec();
        let main = main_of(&archive)?;
        Ok(M4 { device, queue, donor, archive, main })
    }

    /// ⭐ A pipeline running `code` (its main in the donor's place).
    pub fn kernel(&self, code: &Apple9Code) -> Result<Kernel> {
        let (at, room) = self.main;
        if code.bytes.len() > room || code.bytes.len() % 2 != 0 {
            return Err(err(format!("{}: {} bytes of code, the donor's main has {room}", code.name, code.bytes.len())));
        }
        if code.gprs > 64 || code.shared > 16384 {
            return Err(err(format!("{}: {} GPRs, {} bytes of threadgroup memory (the donor's: 64, 16384)", code.name, code.gprs, code.shared)));
        }
        let mut b = self.archive.clone();
        // halt (0e 00 00 00) after the code; a 2-byte tail stays as Metal left it (never reached)
        let mut fill = code.bytes.clone();
        while fill.len() + 4 <= room {
            fill.extend([0x0e, 0, 0, 0]);
        }
        b[at..at + fill.len()].copy_from_slice(&fill);
        let file = Temp::new("kernel");
        std::fs::write(&file.0, &b).map_err(|e| err(format!("{}: {e}", file.0.display())))?;
        let d = MTLBinaryArchiveDescriptor::new();
        d.setUrl(Some(&file.url()));
        let ar = self.device.newBinaryArchiveWithDescriptor_error(&d).map_err(|e| err(format!("the kernel's archive: {e}")))?;
        let cd = MTLComputePipelineDescriptor::new();
        cd.setComputeFunction(Some(&self.donor));
        let archives = NSArray::from_retained_slice(&[ProtocolObject::from_retained(ar)]);
        cd.setBinaryArchives(Some(&archives));
        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(&cd, MTLPipelineOption::FailOnBinaryArchiveMiss, None)
            .map_err(|e| err(format!("{}: the pipeline (a donor archive of another macOS? rebuild): {e}", code.name)))?;
        Ok(Kernel { pipeline, workgroup: code.workgroup, params: code.params })
    }

    /// ⭐ `bytes` of memory the CPU and the GPU share, zeroed.
    pub fn buffer(&self, bytes: usize) -> Result<Buffer> {
        let len = bytes.max(4);
        let buffer = self.device.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared).ok_or_else(|| err(format!("a buffer of {len} bytes")))?;
        let mut b = Buffer { buffer, len };
        b.bytes_mut().fill(0);
        Ok(b)
    }

    /// ⭐ `kernel` over `groups` workgroups (x, y, z) of its own size: binding 0 `memory`, binding 1 `params` then the
    /// grid's size; waited for.
    pub fn dispatch(&self, kernel: &Kernel, memory: &Buffer, params: &[u32], groups: [u32; 3]) -> Result<()> {
        if params.len() != kernel.params as usize {
            return Err(err(format!("{} parameter words, the kernel reads {}", params.len(), kernel.params)));
        }
        let mut words = params.to_vec();
        words.extend(groups);
        let mut p = self.buffer(4 * words.len())?;
        p.write_words(0, &words);
        let size = |w: usize, h: usize, d: usize| MTLSize { width: w, height: h, depth: d };
        let [gx, gy, gz] = groups.map(|g| g as usize);
        let cb = self.queue.commandBuffer().ok_or_else(|| err("no command buffer"))?;
        let e = cb.computeCommandEncoder().ok_or_else(|| err("no compute encoder"))?;
        e.setComputePipelineState(&kernel.pipeline);
        // SAFETY: both buffers live until the dispatch is waited for, below
        unsafe {
            e.setBuffer_offset_atIndex(Some(&memory.buffer), 0, 0);
            e.setBuffer_offset_atIndex(Some(&p.buffer), 0, 1);
        }
        match kernel.workgroup {
            // a hardware workgroup a workgroup
            32 => e.dispatchThreadgroups_threadsPerThreadgroup(size(gx, gy, gz), size(SIMD, 1, 1)),
            // two a hardware workgroup: 16 x invocations a workgroup (the last hardware workgroup of a row half full)
            16 => e.dispatchThreads_threadsPerThreadgroup(size(16 * gx, gy, gz), size(SIMD, 1, 1)),
            w => return Err(err(format!("a workgroup of {w}"))),
        }
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return Err(err(format!("the command buffer: {:?} {:?}", cb.status(), cb.error().map(|e| e.to_string()))));
        }
        Ok(())
    }
}
