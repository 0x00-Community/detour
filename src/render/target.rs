use std::rc::Rc;

use mlirformat::Error;

use super::io;
use crate::{Bo, ColorFormat, ColorTarget, Syncobj, TargetLoad, Tiling};

/// A surface's memory and what last wrote it: the device's own, or a dma-buf imported
pub(crate) struct SurfaceMem<'x, B> {
    pub(crate) bo: B,
    pub(crate) slot: usize,
    pub(crate) addr: u64,
    /// where its pixels start in `bo`, the bytes mapped from there
    pub(crate) offset: usize,
    pub(crate) range: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: ColorFormat,
    pub(crate) tiling: Tiling,
    /// a row's bytes in memory
    pub(crate) pitch: u32,
    pub(crate) fence: Option<Rc<Syncobj<'x>>>,
}

impl<'x, B: Bo> SurfaceMem<'x, B> {
    pub(crate) fn row(&self) -> u32 {
        self.width * self.format.bytes()
    }

    pub(crate) fn len(&self) -> usize {
        self.pitch as usize * self.tiling.rows(self.height) as usize
    }

    /// Whether a draw describing its target as `t` draws into it
    pub(crate) fn holds(&self, t: &ColorTarget<'_>) -> bool {
        (t.width, t.height, t.format, t.tiling) == (self.width, self.height, self.format, self.tiling)
    }

    /// Its memory as laid out (nothing writing it)
    pub(crate) fn memory(&mut self) -> Result<&mut [u8], Error> {
        let (at, n) = (self.offset, self.len());
        self.bo.bytes().map_err(io)?.get_mut(at..at + n).ok_or_else(|| Error("a surface past its memory's end".into()))
    }

    /// Whether the CPU reads and writes it as its tiling lays rows out
    pub(crate) fn natural(&self) -> Result<(), Error> {
        match self.tiling == Tiling::Linear || self.pitch == self.tiling.pitch(self.row()) {
            true => Ok(()),
            false => Err(Error(format!("a {:?} surface with rows {} bytes apart: not read or written by the CPU", self.tiling, self.pitch))),
        }
    }

    /// Its pixels in rows (nothing writing it)
    pub(crate) fn pixels(&mut self) -> Result<Vec<u8>, Error> {
        self.natural()?;
        let (row, height, pitch, tiling) = (self.row() as usize, self.height, self.pitch as usize, self.tiling);
        let mut out = vec![0; row * height as usize];
        let memory = self.memory()?;
        match tiling {
            Tiling::Linear => out.chunks_exact_mut(row).zip(memory.chunks(pitch)).for_each(|(o, m)| o.copy_from_slice(&m[..row])),
            _ => tiling.untile(memory, row as u32, height, &mut out),
        }
        Ok(out)
    }

    /// `pixels` (rows of `rect`'s width) written into rectangle `rect` of it (nothing using it)
    pub(crate) fn write_rect(&mut self, rect: [u32; 4], pixels: &[u8]) -> Result<(), Error> {
        let [x, y, w, h] = rect;
        if x.checked_add(w).is_none_or(|r| r > self.width) || y.checked_add(h).is_none_or(|b| b > self.height) {
            return Err(Error(format!("a {w}x{h} rectangle at {x}, {y} of a {}x{} surface", self.width, self.height)));
        }
        let bpp = self.format.bytes();
        let row = (w * bpp) as usize;
        if pixels.len() < row * h as usize {
            return Err(Error(format!("{} bytes for a {w}x{h} rectangle of {bpp}-byte texels", pixels.len())));
        }
        let (pitch, tiling) = (self.pitch, self.tiling);
        let memory = self.memory()?;
        for (r, src) in pixels.chunks_exact(row).take(h as usize).enumerate() {
            let yy = y + r as u32;
            let mut done = 0;
            while done < row {
                let bx = x * bpp + done as u32;
                let run = (Tiling::RUN - bx % Tiling::RUN) as usize;
                let n = run.min(row - done);
                let at = tiling.byte(bx, yy, pitch);
                memory[at..at + n].copy_from_slice(&src[done..done + n]);
                done += n;
            }
        }
        Ok(())
    }

    /// `pixels` (rows) written into it (nothing using it)
    pub(crate) fn write(&mut self, pixels: &[u8]) -> Result<(), Error> {
        self.natural()?;
        let (row, height, pitch, tiling) = (self.row() as usize, self.height, self.pitch as usize, self.tiling);
        let pixels = pixels.get(..row * height as usize).ok_or_else(|| Error("fewer pixels than the target holds".into()))?;
        let memory = self.memory()?;
        match tiling {
            Tiling::Linear => memory.chunks_mut(pitch).zip(pixels.chunks_exact(row)).for_each(|(m, p)| m[..row].copy_from_slice(p)),
            _ => tiling.tile(pixels, row as u32, height, memory),
        }
        Ok(())
    }

    /// What it holds set as `load` says: kept, filled with one texel or written with pixels, then the clears
    /// applied (nothing using it)
    pub(crate) fn load(&mut self, load: &TargetLoad<'_>) -> Result<(), Error> {
        let clears = match load {
            TargetLoad::Keep(c) | TargetLoad::Fill(_, c) | TargetLoad::Pixels(_, c) => c,
        };
        match load {
            TargetLoad::Fill(texel, _) => {
                let memory = self.memory()?;
                match texel.iter().all(|b| *b == texel[0]) {
                    true => memory.fill(texel[0]),
                    false => memory.chunks_exact_mut(texel.len()).for_each(|p| p.copy_from_slice(texel)),
                }
            }
            TargetLoad::Pixels(p, _) => self.write(p)?,
            TargetLoad::Keep(_) => {}
        }
        let (pitch, tiling) = (self.pitch, self.tiling);
        let memory = self.memory()?;
        clears.iter().for_each(|c| c.apply_pitched(memory, pitch, tiling));
        Ok(())
    }
}
