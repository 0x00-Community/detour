//! Print what the xe device reports: ids, engines, GTs, memory, topology, hwconfig.

use detour::base::Gpu;
use detour::xe::Xe;

fn main() -> detour::base::Result<()> {
    let xe = Xe::open_first()?;
    let info = xe.driver_info();
    println!("{}: {:x?}", xe.path().display(), xe.info());
    println!("max_prio {}", info.max_exec_queue_priority);
    for e in &info.engines {
        println!("engine class {} instance {} gt {}", e.engine_class, e.engine_instance, e.gt_id);
    }
    for g in &info.gts {
        println!("{g:?}");
    }
    for r in &info.mem_regions {
        println!("{r:?}");
    }
    for t in &info.topology {
        let bits: u32 = t.mask.iter().map(|b| b.count_ones()).sum();
        println!("topology gt {} kind {} bits {bits} mask {:02x?}", t.gt, t.kind, t.mask);
    }
    for (k, v) in info.hwconfig() {
        println!("hwconfig {k} = {v:?}");
    }
    Ok(())
}
