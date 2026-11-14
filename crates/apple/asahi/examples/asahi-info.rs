use detour::asahi::Asahi;
use detour::base::Gpu;

fn main() -> detour::base::Result<()> {
    let asahi = Asahi::open_first()?;
    let info = asahi.driver_info();
    println!("{}: {} {:x?}", asahi.path().display(), info.name(), asahi.info());
    println!("{info:#x?}");
    let vm = asahi.vm()?;
    println!("vm: usc {:#x?} user {:#x?}", vm.usc, vm.user);
    Ok(())
}
