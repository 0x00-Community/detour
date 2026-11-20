//! acdes's four page programs (solid, ramp, shadow, image; their MLIR given as arguments in that order) compiled
//! for the card's instruction set and drawn as acdes draws them, the pixels written to the last argument: the same
//! run on Xe3 and on RDNA2 should give the same bytes (within rounding)

use detour::{Card, ColorFormat, ColorTarget, Device, Draw, Filter, Sampled, TargetLoad, Tiling, Toolchain, VertexBuffer, VertexElement, VertexFormat, Wrap};

const W: u32 = 256;
const H: u32 = 128;

fn block(parts: &[(usize, &[f32])], len: usize) -> Vec<u8> {
    let mut b = vec![0u8; len];
    for (at, vals) in parts {
        for (i, v) in vals.iter().enumerate() {
            b[at + 4 * i..at + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    b
}

/// Two triangles over a rectangle, each vertex its position then `extra`
fn quad(x0: f32, y0: f32, x1: f32, y1: f32, extra: &[&[f32]; 4]) -> Vec<u8> {
    let c = [(x0, y0, 0), (x1, y0, 1), (x1, y1, 2), (x0, y0, 0), (x1, y1, 2), (x0, y1, 3)];
    c.iter().flat_map(|(x, y, k)| [*x, *y].into_iter().chain(extra[*k].iter().copied())).flat_map(f32::to_le_bytes).collect()
}

fn shape<'d>(size: &'d [u8], vertices: &'d [u8], per: u32, extra: Option<VertexFormat>, fs: Option<&'d [u8]>, textures: Vec<Sampled>) -> Draw<'d> {
    let mut elements = vec![VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 }];
    elements.extend(extra.map(|format| VertexElement { buffer: 0, format, offset: 8 }));
    let mut uniforms = vec![(0, size)];
    uniforms.extend(fs.map(|b| (1, b)));
    Draw {
        buffers: vec![VertexBuffer { bytes: vertices, stride: per * 4 }],
        elements,
        uniforms,
        textures,
        count: (vertices.len() / (per as usize * 4)) as u32,
        viewport: [0.0, 0.0, W as f32, H as f32],
        scissor: Some([0, 0, W, H]),
        target: ColorTarget { width: W, height: H, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Linear, load: TargetLoad::Keep(&[]) },
        ..Draw::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let card = Card::open_first().expect("a render node");
    let tools = Toolchain::new(card.isa());
    let device = Device::new(&card);
    let program = |k: usize| {
        let text = std::fs::read_to_string(&args[k]).expect("a program");
        let bytes = tools.program(&text).unwrap_or_else(|e| panic!("{}: {}", args[k], e.0));
        tools.load(&bytes).unwrap_or_else(|e| panic!("{}: {}", args[k], e.0))
    };
    let (solid, ramp, shadow, image) = (program(0), program(1), program(2), program(3));
    let target = |load| ColorTarget { width: W, height: H, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Linear, load };
    let rt = device.surface(&target(TargetLoad::Fill(&[0, 0, 0, 0], &[]))).expect("a target");
    let size = block(&[(0, &[W as f32, H as f32])], 8);
    // solid: a colour per corner
    let v = quad(0.0, 0.0, 64.0, 64.0, &[&[1.0, 0.0, 0.0, 1.0], &[0.0, 1.0, 0.0, 1.0], &[0.0, 0.0, 1.0, 1.0], &[0.5, 0.5, 0.5, 0.5]]);
    device.draw(&solid, &shape(&size, &v, 6, Some(VertexFormat::Float32x4), None, vec![]), rt).expect("solid");

    // ramps: linear pad, radial reflect, conic repeat, linear with 4 stops
    let colors: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.8, 0.0, 0.8], [0.0, 0.0, 1.0, 1.0], [0.3, 0.3, 0.0, 0.6]];
    let ramps: [([f32; 4], [f32; 4], f32, f32, [f32; 4], usize); 4] = [
        ([64.0, 0.0, 128.0, 64.0], [70.0, 10.0, 120.0, 50.0], 0.0, 0.0, [0.0, 0.5, 1.0, 1.0], 3),
        ([128.0, 0.0, 192.0, 64.0], [160.0, 32.0, 14.0, 0.0], 1.0, 2.0, [0.0, 0.3, 1.0, 1.0], 3),
        ([192.0, 0.0, 256.0, 64.0], [224.0, 32.0, 0.5, 3.5], 2.0, 1.0, [0.0, 0.5, 1.0, 1.0], 3),
        ([0.0, 64.0, 64.0, 128.0], [0.0, 64.0, 64.0, 128.0], 0.0, 1.0, [0.1, 0.3, 0.6, 0.9], 4),
    ];
    for (rect, geometry, kind, extend, offsets, count) in ramps {
        let mut parts: Vec<(usize, Vec<f32>)> =
            vec![(0, vec![1.0, 0.0, 0.0]), (16, vec![0.0, 1.0, 0.0]), (32, geometry.to_vec()), (48, vec![kind, extend]), (64, offsets.to_vec())];
        for (i, c) in colors.iter().enumerate() {
            parts.push((80 + 16 * i, if i < count { c.to_vec() } else { colors[count - 1].to_vec() }));
        }
        let parts: Vec<(usize, &[f32])> = parts.iter().map(|(a, v)| (*a, v.as_slice())).collect();
        let fs = block(&parts, 144);
        let v = quad(rect[0], rect[1], rect[2], rect[3], &[&[1.0], &[1.0], &[0.75], &[1.0]]);
        device.draw(&ramp, &shape(&size, &v, 3, Some(VertexFormat::Float32), Some(&fs), vec![]), rt).expect("ramp");
    }

    // a box shadow: rect (80, 72)-(120, 112), radius 6, sigma 4 (acdes's BoxShadow::new for it)
    let fs = block(
        &[
            (0, &[1.0, 0.0, 0.0]),
            (16, &[0.0, 1.0, 0.0]),
            (32, &[100.5, 92.5, 9.4142, 9.4142]),
            (48, &[10.5857, 2.6315, 0.38, 0.4999]),
            (64, &[0.25, 40.0]),
            (80, &[0.0, 0.0, 0.0, 0.6]),
        ],
        96,
    );
    let v = quad(64.0, 64.0, 136.0, 128.0, &[&[], &[], &[], &[]]);
    device.draw(&shadow, &shape(&size, &v, 2, None, Some(&fs), vec![]), rt).expect("shadow");

    // an image: 4x4 texels over (144, 64)-(208, 128), nearest then linear halves
    let texels: Vec<u8> = (0..16u8).flat_map(|i| [i * 16, 255 - i * 16, (i % 4) * 80, 255]).collect();
    let tex = device
        .surface(&ColorTarget { width: 4, height: 4, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Linear, load: TargetLoad::Pixels(&texels, &[]) })
        .expect("a texture");
    for (x0, filter) in [(144.0, Filter::Nearest), (208.0, Filter::Linear)] {
        let fs = block(&[(0, &[1.0 / 16.0, 0.0, -x0 / 16.0]), (16, &[0.0, 1.0 / 16.0, -4.0]), (32, &[4.0, 4.0, 0.9])], 48);
        let v = quad(x0, 64.0, x0 + 48.0, 128.0, &[&[1.0], &[1.0], &[1.0], &[0.5]]);
        let t = vec![Sampled { binding: 2, surface: tex, filter, wrap: Wrap::ClampToEdge }];
        device.draw(&image, &shape(&size, &v, 3, Some(VertexFormat::Float32), Some(&fs), t), rt).expect("image");
    }

    let px = device.read(rt, (W * H * 4) as usize).expect("read");
    std::fs::write(&args[4], &px).expect("written");
    let lit = px.chunks(4).filter(|p| p != &[0; 4]).count();
    println!("{}: {:?}, {lit} pixels lit, written to {}", card.path().display(), card.isa(), args[4]);
}
