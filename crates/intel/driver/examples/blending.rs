use detour::base::{Gpu, MemType, Native, TextureFormat};
use detour::xe3::{Codegen, Context};
use detour::xe::Xe;

const N: u32 = 32;
const SRC: [f64; 4] = [1.0, 0.0, 1.0, 0.2];
const DST: [f64; 4] = [0.0, 1.0, 1.0, 1.0];

fn pipeline(blend: &str) -> String {
    format!(
        r#"detour.pipeline @p {{vertex = @vs, fragment = @fs, topology = "triangle-list", targets = ["rgba8unorm"], color_blends = [{blend}],
  vertex_buffers = [{{stride = 8, attributes = [{{location = 0, format = "float32x2", offset = 0}}]}}]}}

func.func @vs(%p: vector<2xf32>) -> vector<4xf32> attributes {{detour.stage = "vertex"}} {{
  %x = vector.extract %p[0] : f32 from vector<2xf32>
  %y = vector.extract %p[1] : f32 from vector<2xf32>
  %z = arith.constant 0.0 : f32
  %w = arith.constant 1.0 : f32
  %pos = "vector.from_elements"(%x, %y, %z, %w) : (f32, f32, f32, f32) -> vector<4xf32>
  return %pos : vector<4xf32>
}}

func.func @fs() -> vector<4xf32> attributes {{detour.stage = "fragment"}} {{
  %c = arith.constant dense<[1.0, 0.0, 1.0, 0.2]> : vector<4xf32>
  return %c : vector<4xf32>
}}
"#
    )
}

fn factor(name: &str, k: usize, constant: [f64; 4]) -> f64 {
    let (s, d) = (SRC, DST);
    let c = if k == 3 { 3 } else { k };
    match name {
        "zero" => 0.0,
        "one" => 1.0,
        "src" => s[c],
        "one-minus-src" => 1.0 - s[c],
        "src-alpha" => s[3],
        "one-minus-src-alpha" => 1.0 - s[3],
        "dst" => d[c],
        "one-minus-dst" => 1.0 - d[c],
        "dst-alpha" => d[3],
        "one-minus-dst-alpha" => 1.0 - d[3],
        "src-alpha-saturated" if k == 3 => 1.0,
        "src-alpha-saturated" => s[3].min(1.0 - d[3]),
        "constant" => constant[c],
        "one-minus-constant" => 1.0 - constant[c],
        other => panic!("factor {other}"),
    }
}

fn blended(k: usize, (sf, df, op): (&str, &str, &str), constant: [f64; 4]) -> u8 {
    let (s, d) = (SRC[k] * factor(sf, k, constant), DST[k] * factor(df, k, constant));
    let v = match op {
        "add" => s + d,
        "subtract" => s - d,
        "reverse-subtract" => d - s,
        "min" => SRC[k].min(DST[k]),
        "max" => SRC[k].max(DST[k]),
        other => panic!("operation {other}"),
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xe = Xe::open_first()?;
    let codegen = Codegen::default();
    let mut cx = Context::<Xe, Native>::new(&xe)?;
    let clear = cx.clear_pipeline(&codegen, [0.0, 1.0, 1.0, 1.0])?;
    let corners = cx.clear_vertices()?;
    let quad = cx.floats(&[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])?;
    type C<'a> = (&'a str, &'a str, &'a str);
    let cases: [(&str, C, C); 12] = [
        ("over, alpha apart", ("src-alpha", "one-minus-src-alpha", "add"), ("one", "one-minus-src-alpha", "add")),
        ("add", ("one", "one", "add"), ("one", "one", "add")),
        ("subtract", ("one", "one", "subtract"), ("one", "one", "subtract")),
        ("reverse subtract", ("one", "one", "reverse-subtract"), ("one", "one", "reverse-subtract")),
        ("min", ("one", "one", "min"), ("one", "one", "min")),
        ("max", ("one", "one", "max"), ("one", "one", "max")),
        ("source times target", ("dst", "zero", "add"), ("dst", "zero", "add")),
        ("target times source", ("zero", "src", "add"), ("zero", "src", "add")),
        ("src-alpha-saturated", ("src-alpha-saturated", "zero", "add"), ("one", "zero", "add")),
        ("the constant (0)", ("constant", "one-minus-constant", "add"), ("constant", "one-minus-constant", "add")),
        ("the constant (1, 0, 0.2, 1)", ("constant", "one-minus-constant", "add"), ("constant", "one-minus-constant", "add")),
        ("one-minus-dst-alpha", ("one-minus-dst-alpha", "one", "add"), ("one-minus-dst-alpha", "one", "add")),
    ];
    let constant = cx.blend_constant([1.0, 0.0, 0.2, 1.0])?;
    let mut ok = true;
    for (name, color, alpha) in cases {
        let component = |(s, d, o): C| format!(r#"{{src_factor = "{s}", dst_factor = "{d}", operation = "{o}"}}"#);
        let p = cx.pipeline(&codegen, &pipeline(&format!("{{color = {}, alpha = {}}}", component(color), component(alpha))), "p")?;
        let mut target = cx.target(N, N, TextureFormat::Rgba8Unorm, MemType::Coherent)?;
        let mut r = cx.record();
        r.target(&target);
        r.draw(&clear, &[&corners], &[], 3, 1)?;
        let set = name.contains("0.2");
        let k = if set { [1.0, 0.0, 0.2, 1.0] } else { [0.0; 4] };
        if set {
            r.blend_constant(&constant);
        }
        r.draw(&p, &[&quad], &[], 6, 1)?;
        let n = cx.submit(r.finish())?;
        cx.wait(n)?;
        let want = [blended(0, color, k), blended(1, color, k), blended(2, color, k), blended(3, alpha, k)];
        let px = target.bytes()?;
        let bad = (0..(N * N) as usize).filter(|i| px[i * 4..][..4] != want).count();
        println!("{name}: want {want:?}, got {:?}, {bad} differ", &px[..4]);
        ok &= bad == 0;
    }
    assert!(ok);
    println!("ok (device {:#06x})", xe.info().device_id);
    Ok(())
}
