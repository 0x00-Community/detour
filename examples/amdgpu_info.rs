//! The amdgpu kernel interface on the first AMD render node: the device's facts; a buffer object in GTT mapped into
//! the GPU's address space; an indirect buffer on the GFX ring writing words into it (PM4 WRITE_DATA), its syncobj
//! waited for; the words read back; the buffer exported as a dma-buf and imported again.
//!
//!   cargo run --release --example amdgpu_info

use std::time::{Duration, Instant};

use detour::{Amdgpu, AmdgpuIb, Bo, Caching, MemType};

/// PM4 type-3 packet header: opcode, then `n` dwords of body
const fn pkt3(op: u32, n: u32) -> u32 {
    3 << 30 | (n - 1) << 16 | op << 8
}
const WRITE_DATA: u32 = 0x37;
/// WRITE_DATA control: destination memory, write confirmed
const DST_MEM: u32 = 5 << 8 | 1 << 20;
/// A one-dword NOP the GFX ring takes as padding
const NOP: u32 = 0xffff_1000;

fn main() {
    let amd = Amdgpu::open_first().expect("an amdgpu render node");
    let i = amd.driver_info();
    println!(
        "{}: device {:#06x} rev {:#x}, family {}, {} CUs, VA {:#x}..{:#x} (high {:#x}..{:#x}), alignment {:#x}",
        amd.path().display(),
        i.device_id,
        i.pci_rev,
        i.family,
        i.cu_active_number,
        i.virtual_address_offset,
        i.virtual_address_max,
        i.high_va_offset,
        i.high_va_max,
        i.virtual_address_alignment
    );
    let vm = amd.vm().unwrap();
    let queue = amd.queue(&vm).unwrap();
    let base = i.virtual_address_offset.max(1 << 32);
    let (data_addr, ib_addr) = (base, base + (1 << 20));
    let mut data = amd.bo((4096, Caching::WriteBack).into()).unwrap();
    data.bytes().unwrap().fill(0);
    vm.bind(&data, 0, data_addr, 4096, MemType::Streaming, &[]).unwrap();
    let words: Vec<u32> = (0..16).map(|k| 0xdead_0000 | k).collect();
    let mut ib = vec![pkt3(WRITE_DATA, 3 + words.len() as u32), DST_MEM, data_addr as u32, (data_addr >> 32) as u32];
    ib.extend(&words);
    while !ib.len().is_multiple_of(8) {
        ib.push(NOP);
    }
    let mut cmd = amd.bo((4096, Caching::WriteCombined).into()).unwrap();
    cmd.bytes().unwrap()[..4 * ib.len()].copy_from_slice(&ib.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>());
    vm.bind(&cmd, 0, ib_addr, 4096, MemType::Streaming, &[]).unwrap();
    let done = amd.fence(false).unwrap();
    let t = Instant::now();
    queue.submit(AmdgpuIb { addr: ib_addr, dwords: ib.len() as u32 }, &[], &[&done]).unwrap();
    assert!(done.wait(Duration::from_secs(5)).unwrap(), "the IB did not finish");
    let took = t.elapsed();
    let got: Vec<u32> = data.bytes().unwrap()[..64].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    assert_eq!(got, words, "the words written");
    println!(
        "ok   WRITE_DATA of {} words on the GFX ring, fenced in {:.3} ms; fault: {:?}",
        words.len(),
        took.as_secs_f64() * 1e3,
        queue.fault().unwrap()
    );
    let fd = data.export().unwrap();
    let mut again = amd.import(std::os::fd::AsFd::as_fd(&fd)).unwrap();
    assert_eq!(again.bytes().unwrap()[..64], data.bytes().unwrap()[..64], "the import's bytes");
    println!("ok   exported and imported again: the same bytes");
    vm.unbind(data_addr, 4096, &[]).unwrap();
    vm.unbind(ib_addr, 4096, &[]).unwrap();
}
