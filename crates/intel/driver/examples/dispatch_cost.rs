use detour::base::{Gpu, Native};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;
use std::time::Instant;

const KERNEL: &str = r#"
func.func @k(%out: i32, %n: i32) attributes {detour.stage = "compute"} {
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %g = "detour.workgroup_id"() : () -> i32
  %l = "detour.lane"() : () -> i32
  %b = arith.muli %g, %c16 : i32
  %i = arith.addi %b, %l : i32
  %in = arith.cmpi slt, %i, %n : i32
  scf.if %in {
    %o = arith.muli %i, %c4 : i32
    %a = arith.addi %out, %o : i32
    "detour.store"(%a, %i) : (i32, i32) -> ()
  }
  return
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let k = cx.kernel(&codegen, KERNEL, "k")?;
    let out = cx.buffer(&vec![0; 1 << 22])?;
    for lanes in [16usize, 4096, 1 << 20] {
        let mut times = Vec::new();
        for count in [1usize, 200, 1, 200] {
            let mut r = cx.record();
            for _ in 0..count {
                r.dispatch(&k, [lanes.div_ceil(16) as u32, 1], &[out.addr as u32, lanes as u32])?;
            }
            let t = Instant::now();
            let s = cx.submit(r.finish())?;
            cx.wait(s)?;
            times.push((count, t.elapsed().as_secs_f64() * 1e3));
        }
        let (one, many) = (times[2].1, times[3].1);
        println!("{lanes} lanes: one dispatch {one:.3} ms, 200 in one submission {many:.2} ms: {:.1} us a dispatch", (many - one) / 199.0 * 1e3);
    }
    Ok(())
}
