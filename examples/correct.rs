//! detour's correctness on Xe3: draws through `Device::draw` (what acdes uses) checked against a CPU reference.
//!
//! - coverage: pixel centres, the fill rule, shared edges (no gaps, no double hits: each hit adds one), thin,
//!   degenerate and huge triangles, strips, first / count, culling, scissors, viewports, instancing, the color mask,
//!   mixed draws in a frame; byte for byte against `correct/reference.rs`
//! - blending: every factor pair with every equation into RGBA8, and float to unorm8 rounding
//! - programs: vertex formats, strides and offsets, interpolation, std140 blocks, and math lowered from SPIR-V
//!   within the Vulkan spec's precision (with NaN, infinities, zeros and denormals)
//!
//!   cargo run --release --example correct [FILTER...]      (cases whose name contains a filter)

#[path = "correct/gpu.rs"]
mod gpu;
#[path = "support/glsl.rs"]
mod glsl;
#[path = "correct/mesa.rs"]
mod mesa;
#[path = "correct/raster.rs"]
mod raster;
#[path = "correct/reference.rs"]
mod reference;
#[path = "correct/shaders.rs"]
mod shaders;

use detour::{Isa, Device, Toolchain};
use gpu::Gpu;
use reference::FillRule;

/// The fill rule Xe3 follows in memory coordinates
const RULE: FillRule = FillRule { top_low: true, left: true };

fn floats(b: &[u8]) -> Vec<[f32; 4]> {
    b.as_chunks::<16>().0.iter().map(|p| core::array::from_fn(|c| f32::from_le_bytes(p[4 * c..4 * c + 4].try_into().unwrap()))).collect()
}

struct Tally {
    filters: Vec<String>,
    failed: Vec<String>,
    passed: usize,
}

impl Tally {
    fn wanted(&self, name: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| name.contains(f.as_str()))
    }

    fn report(&mut self, name: &str, r: Result<String, String>, quiet: bool) {
        match r {
            Ok(s) => {
                self.passed += 1;
                if !quiet {
                    println!("ok   {name}: {s}");
                }
            }
            Err(e) => {
                println!("FAIL {name}: {e}");
                self.failed.push(name.to_owned());
            }
        }
    }
}

fn raster(gpu: &Gpu<'_>, t: &mut Tally, cases: Vec<raster::Case>, quiet: bool) {
    let glref = std::env::var("GLREF").ok();
    for c in cases {
        if !t.wanted(&c.name) {
            continue;
        }
        let ops: Vec<gpu::Op> = c.draws.iter().map(|(op, _)| op.clone()).collect();
        let r = gpu.run(&c.target, &ops).and_then(|got| {
            let want = raster::expected(&c, RULE);
            let d = raster::differences(&c.target, &got, &want, c.tolerance);
            let mesa = match glref.as_deref().and_then(|g| mesa::run(g, &c)) {
                None => None,
                Some(Err(e)) => return Err(format!("glref: {e}")),
                Some(Ok(m)) => Some(m),
            };
            let vs_mesa = mesa.as_ref().map(|m| raster::differences(&c.target, &got, m, 0));
            if d.is_empty() && vs_mesa.as_deref().is_none_or(str::is_empty) {
                let hits = got.chunks_exact(c.target.format.bytes() as usize).filter(|p| p.iter().any(|b| *b != 0)).count();
                let exact = raster::differences(&c.target, &got, &want, 0);
                let off = if exact.is_empty() { String::new() } else { format!(", {} pixels 1 off the f32 reference", exact.split(' ').next().unwrap_or("")) };
                return Ok(format!("{hits} pixels written{}{off}", if mesa.is_some() { ", as Mesa" } else { "" }));
            }
            let mut why = Vec::new();
            if !d.is_empty() {
                let rules: Vec<String> = FillRule::ALL.iter().filter(|r| raster::expected(&c, **r) == got).map(|r| format!("{r:?}")).collect();
                why.push(match rules.is_empty() {
                    true => format!("reference: {d}"),
                    false => format!("reference: {d} (matches {})", rules.join(", ")),
                });
            }
            match vs_mesa {
                Some(m) if m.is_empty() => why.push("Mesa draws the same as detour".into()),
                Some(m) => why.push(format!("Mesa: {m}")),
                None => {}
            }
            Err(why.join("; "))
        });
        t.report(&c.name, r, quiet);
    }
}

fn programs(gpu: &Gpu<'_>, t: &mut Tally, cases: Vec<shaders::Case>) {
    for c in cases {
        if !t.wanted(&c.name) {
            continue;
        }
        let r = gpu.run(&c.target, &[c.op]).and_then(|b| {
            if std::env::var_os("CORRECT_DUMP").is_some() {
                for (i, p) in floats(&b).iter().enumerate().take(64) {
                    println!("  ({}, {}) {p:?}", i as u32 % c.target.width, i as u32 / c.target.width);
                }
            }
            (c.check)(&floats(&b), c.target.width)
        });
        t.report(&c.name, r, false);
    }
}

fn math(gpu: &Gpu<'_>, t: &mut Tally) {
    for f in shaders::funcs() {
        let name = format!("math {}", f.name);
        if !t.wanted(&name) {
            continue;
        }
        let r = gpu.run(&gpu::Target::rgba32f(shaders::MATH_W, shaders::MATH_H), &[shaders::math_op(&f)]).and_then(|b| {
            let px = floats(&b);
            let mut worst = 0.0f64;
            let mut bad = Vec::new();
            let mut specials = Vec::new();
            for (i, p) in px.iter().enumerate() {
                let special = i as u32 / shaders::MATH_W == shaders::MATH_H - 1;
                match shaders::judge(&f, p[1], p[2], p[0]) {
                    Ok(u) => worst = worst.max(u),
                    Err(e) if special => specials.push(format!("x={:e} y={:e}: {e}", p[1], p[2])),
                    Err(e) => bad.push(format!("x={:e} y={:e}: {e}", p[1], p[2])),
                }
            }
            specials.dedup();
            let s = |v: &[String]| v.iter().take(4).cloned().collect::<Vec<_>>().join("; ");
            match (bad.is_empty(), specials.is_empty()) {
                (true, true) => Ok(format!("within bounds, worst {worst:.2} ulp")),
                (false, _) => Err(format!("{} of {} out of bounds: {}", bad.len(), px.len(), s(&bad))),
                (true, false) => Err(format!("worst {worst:.2} ulp; {} special values wrong: {}", specials.len(), s(&specials))),
            }
        });
        t.report(&name, r, false);
    }
}

fn main() {
    let filters: Vec<String> = std::env::args().skip(1).collect();
    let tools = Toolchain::new(Isa::Xe3);
    let xe = detour::Xe::open_first().expect("an Xe3 render node");
    let gpu = Gpu::new(&tools, Device::new(&xe));
    let mut t = Tally { filters, failed: Vec::new(), passed: 0 };
    raster(&gpu, &mut t, raster::coverage(), false);
    raster(&gpu, &mut t, raster::state(), false);
    raster(&gpu, &mut t, raster::rounding(), false);
    let before = t.passed;
    raster(&gpu, &mut t, raster::blends(), true);
    println!("blend combinations: {} ok", t.passed - before);
    programs(&gpu, &mut t, shaders::vertex_fetch());
    programs(&gpu, &mut t, shaders::uniforms());
    programs(&gpu, &mut t, shaders::builtins());
    programs(&gpu, &mut t, shaders::arrays());
    math(&gpu, &mut t);
    println!("{} passed, {} failed", t.passed, t.failed.len());
    std::process::exit(i32::from(!t.failed.is_empty()));
}
