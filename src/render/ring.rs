//! A ring of bytes the GPU reads, each device's vertices, indices, uniforms and command buffers

use std::collections::VecDeque;
use std::rc::Rc;

use mlirformat::Error;

use super::{io, wait};
use crate::{Bo, Syncobj};

/// Batches submitted a ring keeps the fences of (the oldest waited for past them: done long ago)
const INFLIGHT: usize = 64;

/// Bytes the GPU reads, each batch's after the last's around one buffer object mapped at `addr`: a place is written
/// once the batches that read what it held are done. Positions count the bytes ever written (a place is a position
/// modulo the size); a batch never laps itself, and a piece that would cross the end starts at the start
pub(crate) struct Ring<'a, B> {
    addr: u64,
    /// where each piece starts, and the least size the ring takes
    align: usize,
    least: usize,
    bo: Option<B>,
    size: usize,
    head: u64,
    /// where the batch being recorded started
    start: u64,
    /// each batch submitted: where it started, its fence
    inflight: VecDeque<(u64, Rc<Syncobj<'a>>)>,
}

impl<'a, B: Bo> Ring<'a, B> {
    /// A ring at `addr`, its pieces on `align`, at least `least` bytes once it has memory (`restart`)
    pub(crate) fn new(addr: u64, align: usize, least: usize) -> Ring<'a, B> {
        Ring { addr, align, least, bo: None, size: 0, head: 0, start: 0, inflight: VecDeque::new() }
    }

    /// Whether `len` more bytes of the batch being recorded fit
    pub(crate) fn fits(&self, len: usize) -> bool {
        self.bo.is_some() && (self.head - self.start) as usize + 2 * (len + self.align) <= self.size
    }

    /// Where it has written to
    pub(crate) fn head(&self) -> u64 {
        self.head
    }

    /// Whether bytes placed after `head` was `since` are the batch being recorded's: there until it is done (a batch
    /// never laps itself)
    pub(crate) fn recorded(&self, since: u64) -> bool {
        since >= self.start
    }

    /// `bytes` placed: their address
    pub(crate) fn put(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.put_with(bytes.len(), |to| to.copy_from_slice(bytes))
    }

    /// `len` bytes placed, `fill` writing them: their address
    pub(crate) fn put_with(&mut self, len: usize, fill: impl FnOnce(&mut [u8])) -> Result<u64, Error> {
        let size = self.size as u64;
        let mut at = self.head.next_multiple_of(self.align as u64);
        if at % size + len as u64 > size {
            at = at.next_multiple_of(size);
        }
        let end = at + len as u64;
        // the batches that read what was here
        let mut last = None;
        while self.inflight.front().is_some_and(|(start, _)| *start + size < end) {
            last = self.inflight.pop_front();
        }
        if let Some((_, f)) = last {
            wait(&f, "a draw")?;
        }
        let place = (at % size) as usize;
        let bo = self.bo.as_mut().ok_or_else(|| Error("a ring with no memory".into()))?;
        fill(&mut bo.bytes().map_err(io)?[place..place + len]);
        self.head = end;
        Ok(self.addr + place as u64)
    }

    /// The batch recorded went to the GPU, `done` once it is
    pub(crate) fn submitted(&mut self, done: Rc<Syncobj<'a>>) -> Result<(), Error> {
        if self.head > self.start {
            self.inflight.push_back((self.start, done));
        }
        self.start = self.head;
        if self.inflight.len() > INFLIGHT
            && let Some((_, f)) = self.inflight.pop_front()
        {
            wait(&f, "a draw")?;
        }
        Ok(())
    }

    /// Room for a batch with a piece of `len` bytes (nothing running): the ring from a place of its own, nothing before
    /// `recorded`; its memory anew from `memory` (given the old memory to let go of, and a size) when
    /// too small
    pub(crate) fn restart(&mut self, len: usize, memory: impl FnOnce(Option<B>, usize) -> Result<B, Error>) -> Result<(), Error> {
        self.inflight.clear();
        if self.bo.is_none() || 2 * (len + self.align) > self.size {
            let size = (4 * (len + self.align)).next_power_of_two().max(self.least);
            self.bo = Some(memory(self.bo.take(), size)?);
            self.size = size;
        }
        self.head = (self.head + self.size as u64).next_multiple_of(self.size as u64);
        self.start = self.head;
        Ok(())
    }
}
