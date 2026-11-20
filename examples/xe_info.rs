use detour::Xe;

fn main() -> detour::Result<()> {
    let xe = Xe::open_first()?;
    let info = xe.driver_info();
    println!("{}: {:x?}", xe.path().display(), xe.info());
    for e in &info.engines {
        println!("engine class {} instance {} gt {}", e.engine_class, e.engine_instance, e.gt_id);
    }
    for g in &info.gts {
        println!("{g:?}");
    }
    Ok(())
}
