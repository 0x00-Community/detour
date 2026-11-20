//! `Device` on the Radeon (RDNA2, gfx10.3) through `Card`, with shaders hand-assembled in detour's RDNA2 ABI
//! (`Rdna2Shader`): ACO's NGG vertex shader moved onto the ABI's user SGPRs, and small pixel shaders (a varying
//! interpolated, a uniform block, a texture sampled). Checks the pixels radeonsi draws, an imported target, blending
//! against the CPU, uniforms, textures (filters, wraps), indices, culling; then times many small draws.
//!
//!   cargo run --release --example amdgpu_device

use std::time::Instant;

use detour::{
    Blend, BlendEquation, BlendFactor, Card, ColorFormat, ColorTarget, Compiled, CullMode, Device, Draw, DrawPrimitive, Filter, IndexBuffer, IndexFormat, Isa, RDNA2_PERSP_CENTER, Rdna2Shader,
    Sampled, ShaderStage, Shaders, Surface, TargetLoad, Tiling, VertexBuffer, VertexElement, VertexFormat, Wrap,
};

/// s_code_end, padding a shader for the instruction prefetch
const CODE_END: u32 = 0xbf9f_0000;
/// s_waitcnt, s_endpgm
const WAIT_VM: u32 = 0xbf8c_3f70;
const WAIT_LGKM: u32 = 0xbf8c_c07f;
const ENDPGM: u32 = 0xbf81_0000;
/// exp mrt0 v0, v0, v1, v1 done compr vm: four fp16 values, RG in v0, BA in v1
const EXPORT: [u32; 2] = [0xf800_1c0f, 0x8080_0100];

/// ACO's NGG passthrough vertex shader (radeonsi's, Mesa 26.2), its user SGPRs as the ABI's for two inputs: the
/// position's V# at s[8:11], the color's at s[12:15], the first vertex in s16; it exports the position and the color
const VS: [u32; 23] = [
    0xbf8f0003, // s_setprio 3
    0x8f008803, // s_lshl_b32 s0, s3, 8
    0x93fe00c1, // s_bfe_u32 exec_lo, -1, s0
    0xbf880002, // s_cbranch_execz
    0xf8000941, 0x80808000, // exp prim, v0 done
    0x92808003, // s_bfm_b64 s[0:1], s3, 0
    0xbf8cff0f, // s_waitcnt expcnt(0)
    0xbefe0300, // s_mov_b32 exec_lo, s0
    0xbf88000c, // s_cbranch_execz
    0x4a0a0a10, // v_add_nc_u32 v5, s16, v5
    0xe00c2000, 0x80030005, // buffer_load_format_xyzw v[0:3], v5, s[12:15], 0 idxen
    0xe0042000, 0x80020405, // buffer_load_format_xy v[4:5], v5, s[8:11], 0 idxen
    0x7e0c0280, // v_mov_b32 v6, 0
    0x7e0e02f2, // v_mov_b32 v7, 1.0
    WAIT_VM,
    0xf80008cf, 0x07060504, // exp pos0, v4, v5, v6, v7 done
    0xf800020f, 0x03020100, // exp param0, v0, v1, v2, v3
    ENDPGM,
];

/// v_interp_p1_f32 / v_interp_p2_f32 `dst`, `src` (i or j), attr0.`chan`
const fn interp(p2: bool, dst: u32, chan: u32, src: u32) -> u32 {
    0xc800_0000 | dst << 18 | (p2 as u32) << 16 | chan << 8 | src
}

/// v_cvt_pkrtz_f16_f32 `dst`, `src0` (an SGPR below 256, a VGPR from 256), v`src1`
const fn pack(dst: u32, src0: u32, src1: u32) -> u32 {
    0x5e00_0000 | dst << 17 | src1 << 9 | src0
}

/// v_mov_b32 v`dst`, `src` (an SGPR, an inline constant, 255: the literal after it)
const fn mov(dst: u32, src: u32) -> u32 {
    0x7e00_0200 | dst << 17 | src
}

fn shader(words: &[u32]) -> Vec<u8> {
    words.iter().chain([CODE_END; 16].iter()).flat_map(|w| w.to_le_bytes()).collect()
}

fn vs() -> Rdna2Shader {
    Rdna2Shader { stage: ShaderStage::Vertex, code: shader(&VS), sgprs: 17, vgprs: 8, inputs: 2, varyings: 1, ubos: vec![], textures: vec![], ps_input: 0 }
}

fn ps(words: &[u32], sgprs: u32, vgprs: u32, inputs: u32, ubos: Vec<u32>, textures: Vec<u32>) -> Rdna2Shader {
    let code = shader(words);
    Rdna2Shader { stage: ShaderStage::Fragment, code, sgprs, vgprs, inputs, varyings: 0, ubos, textures, ps_input: RDNA2_PERSP_CENTER }
}

/// The color varying interpolated (i, j in v0, v1; the primitive mask in s0) and exported
fn color_ps() -> Rdna2Shader {
    let mut w = vec![0xbefc_0300]; // s_mov_b32 m0, s0
    for c in 0..4 {
        w.extend([interp(false, 2 + c, c, 0), interp(true, 2 + c, c, 1)]);
    }
    w.extend([pack(0, 256 + 2, 3), pack(1, 256 + 4, 5)]);
    w.extend(EXPORT);
    w.push(ENDPGM);
    ps(&w, 1, 6, 1, vec![], vec![])
}

/// Uniform block 0's first vec4 (its address in s[0:1]) exported
fn uniform_ps() -> Rdna2Shader {
    let w = [
        0xf408_0100, 0xfa00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
        WAIT_LGKM,
        mov(2, 5),
        mov(3, 7),
        pack(0, 4, 2),
        pack(1, 6, 3),
        EXPORT[0], EXPORT[1],
        ENDPGM,
    ];
    ps(&w, 8, 4, 0, vec![0], vec![])
}

/// Texture 0 (its T# in s[0:7], its S# in s[8:11]) sampled at (`at`, `at`) and exported
fn texture_ps(at: f32) -> Rdna2Shader {
    let w = [
        mov(2, 255), at.to_bits(),
        mov(3, 255), at.to_bits(),
        0xf080_0f08, 0x0040_0402, // image_sample v[4:7], v[2:3], s[0:7], s[8:11] dmask:0xf dim:2D
        WAIT_VM,
        pack(0, 256 + 4, 5),
        pack(1, 256 + 6, 7),
        EXPORT[0], EXPORT[1],
        ENDPGM,
    ];
    ps(&w, 13, 8, 0, vec![], vec![0])
}

fn program(fs: Rdna2Shader) -> Compiled {
    Compiled::new(Shaders::Rdna2 { vs: vs(), fs })
}

const SIDE: u32 = 64;

/// x, y, rgba8 per vertex: radeonsi's oracle triangle pair
fn vertices() -> Vec<u8> {
    let v: [(f32, f32, [u8; 4]); 6] = [
        (-1.0, -1.0, [255, 0, 0, 255]),
        (1.0, -1.0, [0, 255, 0, 255]),
        (1.0, 1.0, [0, 0, 255, 255]),
        (-1.0, -1.0, [255, 0, 0, 255]),
        (1.0, 1.0, [0, 0, 255, 255]),
        (-1.0, 1.0, [255, 255, 255, 128]),
    ];
    v.iter().flat_map(|(x, y, c)| [x.to_le_bytes(), y.to_le_bytes(), *c].concat()).collect()
}

fn target<'d>(load: TargetLoad<'d>) -> ColorTarget<'d> {
    ColorTarget { width: SIDE, height: SIDE, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Linear, load }
}

fn draw<'d>(vertices: &'d [u8], load: TargetLoad<'d>) -> Draw<'d> {
    Draw {
        buffers: vec![VertexBuffer { bytes: vertices, stride: 12 }],
        elements: vec![
            VertexElement { buffer: 0, format: VertexFormat::Float32x2, offset: 0 },
            VertexElement { buffer: 0, format: VertexFormat::Unorm8x4, offset: 8 },
        ],
        count: 6,
        viewport: [0.0, 0.0, SIDE as f32, SIDE as f32],
        scissor: Some([4, 4, 56, 56]),
        target: target(load),
        ..Default::default()
    }
}

fn at(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    px[(4 * (y * SIDE + x)) as usize..][..4].try_into().unwrap()
}

fn check(what: &str, ok: bool, detail: String) {
    println!("{}  {what}: {detail}", if ok { "ok  " } else { "FAIL" });
    assert!(ok, "{what}");
}

fn main() {
    let card = Card::open_first().expect("a render node detour draws on");
    assert_eq!(card.isa(), Isa::Rdna2, "{}: not an RDNA2 card", card.path().display());
    println!("{}: {:?}", card.path().display(), card.isa());
    let device = Device::new(&card);
    let verts = vertices();
    let zero = [0u8; 4];

    // the plain case: radeonsi's pixels
    let plain = program(color_ps());
    let rt = device.surface(&target(TargetLoad::Keep(&[]))).unwrap();
    let t = Instant::now();
    device.draw(&plain, &draw(&verts, TargetLoad::Fill(&zero, &[])), rt).unwrap();
    let reference = device.read(rt, (4 * SIDE * SIDE) as usize).unwrap();
    let first = t.elapsed();
    let (centre, corner) = (at(&reference, 32, 32), at(&reference, 1, 1));
    check("plain", centre == [126, 0, 129, 255] && corner == [0; 4], format!("(32,32) {centre:?} (1,1) {corner:?}, first draw and read {:.3} ms", first.as_secs_f64() * 1e3));

    // the same into a dma-buf: one surface exported, imported as the target
    let owner = device.surface(&target(TargetLoad::Fill(&[9, 9, 9, 9], &[]))).unwrap();
    let e = device.export(owner).unwrap();
    let imported = device.import(&target(TargetLoad::Keep(&[])), std::os::fd::AsFd::as_fd(&e.fd), e.offset, e.pitch).unwrap();
    device.draw(&plain, &draw(&verts, TargetLoad::Fill(&zero, &[])), imported).unwrap();
    let fence = device.fence(imported).unwrap();
    device.finish(imported).unwrap();
    let via_owner = device.read(owner, reference.len()).unwrap();
    check("imported target", via_owner == reference && fence.is_some(), format!("{} bytes as the plain draw's, pitch {}, fenced", via_owner.len(), e.pitch));

    // blend ONE / ONE_MINUS_SRC_ALPHA over a non-zero target, against the CPU
    let under = [40u8, 80, 120, 200];
    let mut blended = draw(&verts, TargetLoad::Fill(&under, &[]));
    blended.blend = Some(Blend { src: [BlendFactor::One; 2], dst: [BlendFactor::OneMinusSrcAlpha; 2], equation: [BlendEquation::Add; 2], color: [0.0; 4] });
    device.draw(&plain, &blended, rt).unwrap();
    let got = device.read(rt, reference.len()).unwrap();
    let mut worst = 0;
    for (i, (g, s)) in got.chunks(4).zip(reference.chunks(4)).enumerate() {
        let (x, y) = (i as u32 % SIDE, i as u32 / SIDE);
        let inside = (4..60).contains(&x) && (4..60).contains(&y);
        for c in 0..4 {
            let want = if inside { (f32::from(s[c]) + f32::from(under[c]) * (1.0 - f32::from(s[3]) / 255.0)).min(255.0) } else { f32::from(under[c]) };
            worst = worst.max((f32::from(g[c]) - want).abs().round() as u32);
        }
    }
    check("blend ONE, ONE_MINUS_SRC_ALPHA", worst <= 2, format!("(32,32) {:?}, (8,56) {:?}, worst channel off by {worst}", at(&got, 32, 32), at(&got, 8, 56)));

    // a uniform block's color
    let uniform = program(uniform_ps());
    let color: Vec<u8> = [0.25f32, 0.5, 0.75, 1.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut d = draw(&verts, TargetLoad::Fill(&zero, &[]));
    d.uniforms = vec![(0, &color)];
    device.draw(&uniform, &d, rt).unwrap();
    let got = device.read(rt, reference.len()).unwrap();
    let c = at(&got, 32, 32);
    check("uniform block", c.iter().zip([64, 128, 191, 255]).all(|(g, w)| g.abs_diff(w) <= 1) && at(&got, 1, 1) == [0; 4], format!("(32,32) {c:?}"));

    // a 4x4 texture: texel (x, y) = (60x, 60y, 10 + x + 4y, 255)
    let texels: Vec<u8> = (0..16u8).flat_map(|i| [60 * (i % 4), 60 * (i / 4), 10 + i, 255]).collect();
    let tex = device.surface(&ColorTarget { width: 4, height: 4, format: ColorFormat::Rgba8Unorm, tiling: Tiling::Linear, load: TargetLoad::Pixels(&texels, &[]) }).unwrap();
    let texel = |x: usize, y: usize| -> [u8; 4] { texels[4 * (4 * y + x)..][..4].try_into().unwrap() };
    let mean = |ts: &[[u8; 4]]| -> [u8; 4] { std::array::from_fn(|c| (ts.iter().map(|t| f32::from(t[c])).sum::<f32>() / ts.len() as f32).round() as u8) };
    let cases = [
        ("nearest at 0.5", 0.5, Filter::Nearest, Wrap::ClampToEdge, texel(2, 2)),
        ("linear at 0.5", 0.5, Filter::Linear, Wrap::ClampToEdge, mean(&[texel(1, 1), texel(2, 1), texel(1, 2), texel(2, 2)])),
        ("repeat at 1.375", 1.375, Filter::Nearest, Wrap::Repeat, texel(1, 1)),
        ("clamp at 1.375", 1.375, Filter::Nearest, Wrap::ClampToEdge, texel(3, 3)),
        ("mirror at 1.375", 1.375, Filter::Nearest, Wrap::MirroredRepeat, texel(2, 2)),
    ];
    for (what, coord, filter, wrap, want) in cases {
        let textured = program(texture_ps(coord));
        let mut d = draw(&verts, TargetLoad::Fill(&zero, &[]));
        d.textures = vec![Sampled { binding: 0, surface: tex, filter, wrap }];
        device.draw(&textured, &d, rt).unwrap();
        let got = at(&device.read(rt, reference.len()).unwrap(), 32, 32);
        check(&format!("texture, {what}"), got.iter().zip(want).all(|(g, w)| g.abs_diff(w) <= 1), format!("{got:?}, want {want:?}"));
    }

    // the same quad from 4 vertices and 6 u16 indices
    let quad: Vec<u8> = [0usize, 1, 2, 5].iter().flat_map(|&v| verts[12 * v..12 * v + 12].to_vec()).collect();
    let idx: Vec<u8> = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
    for format in [IndexFormat::U16, IndexFormat::U32] {
        let idx32: Vec<u8> = [0u32, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut d = draw(&quad, TargetLoad::Fill(&zero, &[]));
        d.indices = Some(IndexBuffer { bytes: if format == IndexFormat::U16 { &idx } else { &idx32 }, format });
        device.draw(&plain, &d, rt).unwrap();
        let got = device.read(rt, reference.len()).unwrap();
        check(&format!("indexed {format:?}"), got == reference, format!("(32,32) {:?}", at(&got, 32, 32)));
    }

    // the first vertex: the second triangle alone
    let mut d = draw(&verts, TargetLoad::Fill(&zero, &[]));
    (d.first, d.count) = (3, 3);
    device.draw(&plain, &d, rt).unwrap();
    let got = device.read(rt, reference.len()).unwrap();
    check("first vertex 3", at(&got, 8, 56) == at(&reference, 8, 56) && at(&got, 56, 8) == [0; 4], format!("(8,56) {:?} (56,8) {:?}", at(&got, 8, 56), at(&got, 56, 8)));

    // vertices past the ring's size: it wraps (waiting for what read it) and grows
    let mut big = verts.clone();
    for (n, len) in [(48, 512 << 10), (2, 9 << 20), (16, 1 << 20)] {
        big.resize(len, 0);
        for _ in 0..n {
            device.draw(&plain, &draw(&big, TargetLoad::Keep(&[])), rt).unwrap();
        }
        device.draw(&plain, &draw(&big, TargetLoad::Fill(&zero, &[])), rt).unwrap();
        let got = device.read(rt, reference.len()).unwrap();
        check(&format!("{n} draws of {} KiB of vertices", len >> 10), got == reference, format!("(32,32) {:?}", at(&got, 32, 32)));
    }

    // culling: the triangles are counter-clockwise, front
    for (cull, lit) in [(CullMode::Back, true), (CullMode::Front, false)] {
        let mut d = draw(&verts, TargetLoad::Fill(&zero, &[]));
        d.cull = cull;
        device.draw(&plain, &d, rt).unwrap();
        let got = device.read(rt, reference.len()).unwrap();
        check(&format!("cull {cull:?}"), (got == reference) == lit, format!("(32,32) {:?}", at(&got, 32, 32)));
    }

    // a line strip around the scissor's inside and points: some pixels lit, nothing hangs
    for primitive in [DrawPrimitive::LineStrip, DrawPrimitive::Lines, DrawPrimitive::Points, DrawPrimitive::TriangleStrip] {
        let mut d = draw(&quad, TargetLoad::Fill(&zero, &[]));
        (d.primitive, d.count, d.scissor) = (primitive, 4, None);
        device.draw(&plain, &d, rt).unwrap();
        let lit = device.read(rt, reference.len()).unwrap().chunks(4).filter(|p| p != &[0; 4]).count();
        check(&format!("{primitive:?}"), lit > 0, format!("{lit} pixels lit"));
    }

    // many small draws: each a quad of its own into one target, recorded, then the GPU's time
    let rt2: Surface = device.surface(&target(TargetLoad::Fill(&zero, &[]))).unwrap();
    for n in [1000usize, 1000] {
        let mut boxes: Vec<Vec<u8>> = Vec::with_capacity(n);
        for i in 0..n {
            let (x, y) = ((i % 32) as f32 / 16.0 - 1.0, (i / 32 % 32) as f32 / 16.0 - 1.0);
            let v: [(f32, f32); 6] = [(x, y), (x + 0.0625, y), (x + 0.0625, y + 0.0625), (x, y), (x + 0.0625, y + 0.0625), (x, y + 0.0625)];
            boxes.push(v.iter().flat_map(|(x, y)| [x.to_le_bytes(), y.to_le_bytes(), [i as u8, 128, 255 - i as u8, 255]].concat()).collect());
        }
        let t = Instant::now();
        for b in &boxes {
            let mut d = draw(b, TargetLoad::Keep(&[]));
            d.scissor = None;
            device.draw(&plain, &d, rt2).unwrap();
        }
        let recorded = t.elapsed();
        device.finish(rt2).unwrap();
        let done = t.elapsed();
        println!(
            "time  {n} draws: {:.2} us each recorded, {:.2} us each to the GPU and done ({:.2} ms in all)",
            recorded.as_secs_f64() * 1e6 / n as f64,
            done.as_secs_f64() * 1e6 / n as f64,
            done.as_secs_f64() * 1e3
        );
    }
    let lit = device.read(rt2, reference.len()).unwrap().chunks(4).filter(|p| p != &[0; 4]).count();
    check("many draws", lit == 1000 * 4, format!("{lit} pixels lit, 4 by each of 1000 boxes"));
    let t = Instant::now();
    for _ in 0..100 {
        let mut d = draw(&verts, TargetLoad::Keep(&[]));
        d.scissor = None;
        device.draw(&plain, &d, rt2).unwrap();
        device.finish(rt2).unwrap();
    }
    println!("time  one draw submitted and waited for: {:.3} ms each", t.elapsed().as_secs_f64() * 1e3 / 100.0);
    println!("{} programs uploaded", device.lowered());
}
