//! Print what the amdgpu device reports: ids, family, CUs, the VA range, the rings and their IB alignment.

use detour::amdgpu::Amdgpu;
use detour::base::Gpu;

fn main() -> detour::base::Result<()> {
    let amd = Amdgpu::open_first()?;
    println!("{}: {:x?}", amd.path().display(), amd.info());
    println!("{:#x?}", amd.driver_info());
    Ok(())
}
