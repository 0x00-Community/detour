use detour::base::{Gpu, IndexFormat, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const ORANGE: [u8; 4] = [255, 128, 64, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const N: u32 = 64;

const SHADERS: &str = r#"
func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {detour.stage = "vertex"} {
  %x = vector.extract %p[0] : f32 from vector<2xf32>
  %y = vector.extract %p[1] : f32 from vector<2xf32>
  %z = arith.constant 0.0 : f32
  %w = arith.constant 1.0 : f32
  %pos = "vector.from_elements"(%x, %y, %z, %w) : (f32, f32, f32, f32) -> vector<4xf32>
  return %pos : vector<4xf32>
}

func.func @fs() -> vector<4xf32> attributes {detour.stage = "fragment"} {
  %c = arith.constant dense<[1.0, 0.5, 0.25, 1.0]> : vector<4xf32>
  return %c : vector<4xf32>
}
"#;

fn pipeline(name: &str, extra: &str) -> String {
    format!(
        "detour.pipeline @{name} {{vertex = @vs, fragment = @fs, {extra}, targets = [\"rgba8unorm\"],\n  vertex_buffers = [{{stride = 8, attributes = [{{location = 0, format = \"float32x2\", offset = 0}}]}}]}}\n{SHADERS}"
    )
}

fn center(x: u32, y: u32) -> [f32; 2] {
    [(x as f32 + 0.5) / N as f32 * 2.0 - 1.0, (y as f32 + 0.5) / N as f32 * 2.0 - 1.0]
}

fn corner(x: u32, y: u32) -> [f32; 2] {
    [x as f32 / N as f32 * 2.0 - 1.0, y as f32 / N as f32 * 2.0 - 1.0]
}

struct Run<'c, 'd> {
    cx: &'c mut Context<'d, Xe, Native>,
    codegen: &'c Codegen<'c>,
    clear: detour::xe3::Pipeline,
    corners: detour::xe3::Buffer<'d, Xe>,
}

impl<'d> Run<'_, 'd> {
    fn orange(&mut self, extra: &str, draw: impl FnOnce(&mut detour::xe3::Recorder<'_, 'd, Xe, Native>, &detour::xe3::Pipeline) -> Result<(), detour::xe3::Error>) -> Result<Vec<(u32, u32)>, Box<dyn std::error::Error>> {
        let p = self.cx.pipeline(self.codegen, &pipeline("p", extra), "p")?;
        let mut target = self.cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = self.cx.record();
        r.target(&target);
        r.draw(&self.clear, &[&self.corners], &[], 3, 1)?;
        draw(&mut r, &p)?;
        let n = self.cx.submit(r.finish())?;
        self.cx.wait(n)?;
        let px = target.bytes()?;
        let mut lit = Vec::new();
        for y in 0..N {
            for x in 0..N {
                let p: [u8; 4] = px[((y * N + x) * 4) as usize..][..4].try_into()?;
                if p == ORANGE {
                    lit.push((x, y));
                } else if p != BLUE {
                    return Err(format!("pixel ({x}, {y}) is {p:?}").into());
                }
            }
        }
        Ok(lit)
    }
}

fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<(u32, u32)> {
    (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).collect()
}

fn check(name: &str, mut got: Vec<(u32, u32)>, mut want: Vec<(u32, u32)>) -> bool {
    got.sort_by_key(|&(x, y)| (y, x));
    want.sort_by_key(|&(x, y)| (y, x));
    let ok = got == want;
    println!("{name}: {} pixels lit, {}", got.len(), if ok { "as expected".to_owned() } else { format!("expected {}: got {:?}", want.len(), &got[..got.len().min(12)]) });
    ok
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 0.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let mut run = Run { cx: &mut cx, codegen: &codegen, clear, corners };
    let mut ok = true;

    let pts = [(3, 4), (10, 10), (40, 7), (63, 63)];
    let vb = run.cx.floats(&pts.iter().flat_map(|&(x, y)| center(x, y)).collect::<Vec<_>>())?;
    let got = run.orange("topology = \"point-list\"", |r, p| r.draw(p, &[&vb], &[], 4, 1))?;
    ok &= check("point-list", got, pts.to_vec());

    let vb = run.cx.floats(&[center(5, 20), center(25, 20), center(30, 2), center(30, 12)].concat())?;
    let lines = run.orange("topology = \"line-list\"", |r, p| r.draw(p, &[&vb], &[], 4, 1))?;
    ok &= check("line-list (diamond exit: first pixel, not the last)", lines, [rect(5, 20, 25, 21), rect(30, 2, 31, 12)].concat());
    let vb = run.cx.floats(&[center(5, 40), center(25, 40), center(25, 55)].concat())?;
    let strip = run.orange("topology = \"line-strip\"", |r, p| r.draw(p, &[&vb], &[], 3, 1))?;
    ok &= check("line-strip", strip, [rect(5, 40, 25, 41), rect(25, 40, 26, 55)].concat());

    let quads = [corner(4, 4), corner(12, 4), corner(4, 12), corner(12, 12), corner(20, 8), corner(28, 8), corner(20, 16), corner(28, 16)].concat();
    let vb = run.cx.floats(&quads)?;
    let want = [rect(4, 4, 12, 12), rect(20, 8, 28, 16)].concat();
    for (format, restart, name) in [(IndexFormat::Uint16, u32::from(u16::MAX), "uint16"), (IndexFormat::Uint32, u32::MAX, "uint32")] {
        let idx = [0, 1, 2, 3, restart, 4, 5, 6, 7];
        let bytes: Vec<u8> = match format {
            IndexFormat::Uint16 => idx.iter().flat_map(|i| (*i as u16).to_le_bytes()).collect(),
            IndexFormat::Uint32 => idx.iter().flat_map(|i| i.to_le_bytes()).collect(),
        };
        let ib = run.cx.buffer(&bytes)?;
        let got = run.orange(&format!("topology = \"triangle-strip\", strip_index_format = \"{name}\""), |r, p| r.draw_indexed(p, &[&vb], (&ib, format), &[], 9, 1, 0, 0))?;
        ok &= check(&format!("triangle-strip, restart {name}"), got, want.clone());
    }

    let ccw = [corner(4, 40), corner(4, 48), corner(12, 48)];
    let cw = [corner(30, 40), corner(38, 40), corner(38, 48)];
    let vb = run.cx.floats(&[ccw.concat(), cw.concat()].concat())?;
    let all = run.orange("topology = \"triangle-list\"", |r, p| r.draw(p, &[&vb], &[], 6, 1))?;
    let left: Vec<(u32, u32)> = all.iter().copied().filter(|p| p.0 < 20).collect();
    let right: Vec<(u32, u32)> = all.iter().copied().filter(|p| p.0 >= 20).collect();
    println!("cull none: {} pixels (ccw {}, cw {})", all.len(), left.len(), right.len());
    for (extra, want, name) in [
        ("cull_mode = \"back\"", left.clone(), "cull back, front ccw: the ccw one"),
        ("cull_mode = \"front\"", right.clone(), "cull front, front ccw: the cw one"),
        ("front_face = \"cw\", cull_mode = \"back\"", right.clone(), "cull back, front cw: the cw one"),
        ("front_face = \"cw\", cull_mode = \"front\"", left.clone(), "cull front, front cw: the ccw one"),
    ] {
        let got = run.orange(&format!("topology = \"triangle-list\", {extra}"), |r, p| r.draw(p, &[&vb], &[], 6, 1))?;
        ok &= check(name, got, want);
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
