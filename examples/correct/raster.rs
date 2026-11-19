//! Cases of fixed-function state with a solid program: coverage, blending, scissor, viewport, culling, vertex
//! fetch; each drawn on Xe3 and by `reference`, compared byte for byte.

use detour::{Blend, BlendEquation, BlendFactor, ColorFormat, CullMode, DrawPrimitive, Tiling, VertexElement, VertexFormat};

use crate::gpu::{Op, Target, bytes};
use crate::reference::{self, FillRule};

pub const SOLID_VS: &str = "#version 310 es\nlayout(location = 0) in vec4 a_pos;\nvoid main() { gl_Position = a_pos; }\n";
pub const SOLID_FS: &str = "#version 310 es\nprecision highp float;\nlayout(std140, binding = 1) uniform U { vec4 color; } u;\nlayout(location = 0) out vec4 o;\nvoid main() { o = u.color; }\n";
pub const INSTANCED_VS: &str = "#version 310 es\nlayout(location = 0) in vec4 a_pos;\nlayout(std140, binding = 0) uniform V { vec4 step; } v;\nvoid main() { gl_Position = a_pos + v.step * float(gl_InstanceID); }\n";

/// Adds one count a hit: what draws of coverage blend with
pub const COUNT: Blend = Blend { src: [BlendFactor::One; 2], dst: [BlendFactor::One; 2], equation: [BlendEquation::Add; 2], color: [0.0; 4] };
pub const ONE: f32 = 1.0 / 255.0;

/// What the reference needs of a draw beyond `Op`: each vertex's clip position, the source color, the per-instance
/// clip offset
#[derive(Clone)]
pub struct Shade {
    pub positions: Vec<[f32; 4]>,
    pub color: [f32; 4],
    pub step: [f32; 4],
}

pub struct Case {
    pub name: String,
    pub target: Target,
    pub draws: Vec<(Op, Shade)>,
    /// how far a channel may be from the f32 reference: 0 for coverage; 1 where the GPU blends or rounds to unorm8
    /// in less than f32 (Mesa draws the same there)
    pub tolerance: u8,
}

/// A triangle list draw of window-coordinate vertices (as clip positions for a `w`×`h` viewport at 0, 0) in one color
pub fn draw(w: u32, h: u32, window: &[[f32; 2]], color: [f32; 4], blend: Option<Blend>) -> (Op, Shade) {
    let positions: Vec<[f32; 4]> = window.iter().map(|&[x, y]| [x / (w as f32 * 0.5) - 1.0, y / (h as f32 * 0.5) - 1.0, 0.0, 1.0]).collect();
    clip(w, h, positions, color, blend)
}

pub fn clip(w: u32, h: u32, positions: Vec<[f32; 4]>, color: [f32; 4], blend: Option<Blend>) -> (Op, Shade) {
    let flat: Vec<f32> = positions.iter().flatten().copied().collect();
    let op = Op {
        program: (SOLID_VS.into(), SOLID_FS.into()),
        buffers: vec![(bytes(&flat), 16)],
        elements: vec![VertexElement { buffer: 0, format: VertexFormat::Float32x4, offset: 0 }],
        uniforms: vec![(1, bytes(&color))],
        primitive: DrawPrimitive::Triangles,
        first: 0,
        count: positions.len() as u32,
        instances: 1,
        viewport: [0.0, 0.0, w as f32, h as f32],
        scissor: None,
        cull: CullMode::None,
        front_ccw: true,
        blend,
        mask: [true; 4],
    };
    (op, Shade { positions, color, step: [0.0; 4] })
}

fn triangles(primitive: DrawPrimitive, first: u32, count: u32) -> Vec<[usize; 3]> {
    let (f, n) = (first as usize, count as usize);
    match primitive {
        DrawPrimitive::Triangles => (0..n / 3).map(|t| [f + 3 * t, f + 3 * t + 1, f + 3 * t + 2]).collect(),
        DrawPrimitive::TriangleStrip => {
            (0..n.saturating_sub(2)).map(|t| if t % 2 == 0 { [f + t, f + t + 1, f + t + 2] } else { [f + t + 1, f + t, f + t + 2] }).collect()
        }
        _ => Vec::new(),
    }
}

/// What the draws write, by the reference with fill rule `rule`
pub fn expected(case: &Case, rule: FillRule) -> Vec<u8> {
    let t = &case.target;
    let fixed = t.format == ColorFormat::Rgba8Unorm;
    let n = (t.width * t.height) as usize;
    let mut px: Vec<[f32; 4]> = match (&t.pixels, t.format) {
        (Some(p), ColorFormat::Rgba8Unorm) => p.as_chunks::<4>().0.iter().map(|b| core::array::from_fn(|c| f32::from(b[c]) / 255.0)).collect(),
        (Some(p), ColorFormat::Rgba32Float) => p.as_chunks::<16>().0.iter().map(|b| core::array::from_fn(|c| f32::from_le_bytes(b[4 * c..4 * c + 4].try_into().unwrap()))).collect(),
        _ => match t.format {
        ColorFormat::Rgba8Unorm => vec![core::array::from_fn(|c| f32::from(t.clear[c]) / 255.0); n],
        ColorFormat::Rgba32Float => vec![core::array::from_fn(|c| f32::from_le_bytes(t.clear[4 * c..4 * c + 4].try_into().unwrap())); n],
        },
    };
    for (op, shade) in &case.draws {
        for instance in 0..op.instances {
            for tri in triangles(op.primitive, op.first, op.count) {
                let p = tri.map(|i| {
                    let c: [f32; 4] = core::array::from_fn(|k| shade.positions[i][k] + shade.step[k] * instance as f32);
                    reference::window(c, op.viewport).map(reference::snap)
                });
                if reference::culled(p, op.cull, op.front_ccw) {
                    continue;
                }
                let [vx, vy, vw, vh] = op.viewport;
                reference::triangle(p, t.width, t.height, rule, |x, y| {
                    let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
                    if cx < vx || cy < vy || cx >= vx + vw || cy >= vy + vh {
                        return;
                    }
                    if let Some([sx, sy, sw, sh]) = op.scissor
                        && (x < sx || y < sy || x >= sx + sw || y >= sy + sh)
                    {
                        return;
                    }
                    let d = &mut px[(y * t.width + x) as usize];
                    let s = op.blend.as_ref().map_or(shade.color, |b| reference::blend(b, shade.color, *d, fixed));
                    for c in 0..4 {
                        if op.mask[c] {
                            d[c] = if fixed { f32::from(reference::unorm8(s[c])) / 255.0 } else { s[c] };
                        }
                    }
                });
            }
        }
    }
    match t.format {
        ColorFormat::Rgba8Unorm => px.iter().flat_map(|p| p.map(reference::unorm8)).collect(),
        ColorFormat::Rgba32Float => px.iter().flat_map(|p| p.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>()).collect(),
    }
}

/// "" when they match (each RGBA8 channel within `tolerance`), else how they differ
pub fn differences(t: &Target, got: &[u8], want: &[u8], tolerance: u8) -> String {
    let size = t.format.bytes() as usize;
    let close = |a: &[u8], b: &[u8]| match t.format {
        ColorFormat::Rgba8Unorm => a.iter().zip(b).all(|(x, y)| x.abs_diff(*y) <= tolerance),
        ColorFormat::Rgba32Float => a == b,
    };
    let bad: Vec<usize> = (0..(t.width * t.height) as usize).filter(|&i| !close(&got[i * size..(i + 1) * size], &want[i * size..(i + 1) * size])).collect();
    if bad.is_empty() {
        return String::new();
    }
    let show = |b: &[u8]| match t.format {
        ColorFormat::Rgba8Unorm => format!("{b:?}"),
        ColorFormat::Rgba32Float => format!("{:?}", b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<_>>()),
    };
    let first: Vec<String> = bad
        .iter()
        .take(6)
        .map(|&i| format!("({}, {}) {} want {}", i as u32 % t.width, i as u32 / t.width, show(&got[i * size..(i + 1) * size]), show(&want[i * size..(i + 1) * size])))
        .collect();
    format!("{} pixels differ: {}", bad.len(), first.join("; "))
}

/// A tiny deterministic generator
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// In [lo, hi)
    pub fn float(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn counted(name: &str, w: u32, h: u32, tris: Vec<[[f32; 2]; 3]>) -> Case {
    let window: Vec<[f32; 2]> = tris.into_iter().flatten().collect();
    Case { name: name.into(), target: Target::rgba8(w, h, [0; 4]), draws: vec![draw(w, h, &window, [ONE; 4], Some(COUNT))], tolerance: 0 }
}

fn quad(x0: f32, y0: f32, x1: f32, y1: f32) -> [[[f32; 2]; 3]; 2] {
    [[[x0, y0], [x1, y0], [x1, y1]], [[x0, y0], [x1, y1], [x0, y1]]]
}

/// A grid of `n`×`n` cells over [lo, hi)², its inner vertices jittered by up to `jitter` cells, two triangles a cell
fn mesh(rng: &mut Rng, n: usize, lo: f32, hi: f32, jitter: f32) -> Vec<[[f32; 2]; 3]> {
    let cell = (hi - lo) / n as f32;
    let mut v = vec![[0.0f32; 2]; (n + 1) * (n + 1)];
    for j in 0..=n {
        for i in 0..=n {
            let inner = i > 0 && j > 0 && i < n && j < n;
            let (dx, dy) = if inner { (rng.float(-jitter, jitter) * cell, rng.float(-jitter, jitter) * cell) } else { (0.0, 0.0) };
            v[j * (n + 1) + i] = [lo + i as f32 * cell + dx, lo + j as f32 * cell + dy];
        }
    }
    let mut tris = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            let (b, c, d) = (a + 1, a + n + 2, a + n + 1);
            if (i + j) % 2 == 0 {
                tris.extend([[v[a], v[b], v[c]], [v[a], v[c], v[d]]]);
            } else {
                tris.extend([[v[a], v[b], v[d]], [v[b], v[c], v[d]]]);
            }
        }
    }
    tris
}

/// A fan of `n` triangles around `c` out to radius `r`, its angles uneven
fn fan(rng: &mut Rng, c: [f32; 2], r: f32, n: usize) -> Vec<[[f32; 2]; 3]> {
    let mut a: Vec<f32> = (0..n).map(|i| (i as f32 + rng.float(-0.4, 0.4)) / n as f32 * core::f32::consts::TAU).collect();
    a.sort_by(f32::total_cmp);
    let p: Vec<[f32; 2]> = a.iter().map(|t| [c[0] + r * t.cos(), c[1] + r * t.sin()]).collect();
    (0..n).map(|i| [c, p[i], p[(i + 1) % n]]).collect()
}

pub fn coverage() -> Vec<Case> {
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let mut cases = Vec::new();
    let s = 1.0 / 256.0;
    // edges on pixel centres, an eighth of a sub-pixel either side of them, and on pixel corners
    let mut rects = Vec::new();
    for (k, off) in [0.0, s, -s, 0.5, 0.5 + s, 0.5 - s, 0.25, 0.75].iter().enumerate() {
        let (x, y) = (2.0 + 7.0 * k as f32, 2.0 + 7.0 * k as f32);
        rects.extend(quad(x + off, 3.0 + off, x + 5.0 + off, 9.0 + off));
        rects.extend(quad(3.0 + off, y + 20.0 + off, 9.0 + off, y + 24.0 + off));
    }
    cases.push(counted("pixel centres", 64, 128, rects));
    // edges of every slope through pixel centres, both windings
    let mut tris = Vec::new();
    for (k, &(dx, dy)) in [(4.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, -4.0), (4.0, 2.0), (2.0, 4.0), (4.0, 1.0), (-1.0, 4.0)].iter().enumerate() {
        let o = [4.5 + 8.0 * (k % 4) as f32, 4.5 + 8.0 * (k / 4) as f32];
        tris.push([o, [o[0] + dx, o[1] + dy], [o[0] + dy.abs().max(1.0), o[1] - dx.abs().max(1.0) + 6.0]]);
        let o = [o[0], o[1] + 20.0];
        tris.push([o, [o[0] - dy.abs().max(1.0) + 3.0, o[1] + dx.abs().max(1.0)], [o[0] + dx, o[1] + dy]]);
    }
    cases.push(counted("edges through centres", 40, 48, tris));
    // a mesh on dyadic coordinates, then jittered meshes and fans: every pixel hit once
    cases.push(counted("mesh, centres", 64, 64, mesh(&mut rng, 8, 0.5, 64.5, 0.0)));
    for (i, jitter) in [0.3, 0.45].iter().enumerate() {
        cases.push(counted(&format!("mesh, jittered {i}"), 64, 64, mesh(&mut rng, 12, -3.0, 67.0, *jitter)));
    }
    cases.push(counted("fan of 7", 64, 64, fan(&mut rng, [31.7, 30.3], 40.0, 7)));
    cases.push(counted("fan of 64", 64, 64, fan(&mut rng, [32.0, 32.0], 28.0, 64)));
    cases.push(counted("fan at a corner", 64, 64, fan(&mut rng, [16.5, 16.5], 60.0, 33)));
    // thin and degenerate
    let mut thin = Vec::new();
    for (k, w) in [1.0, 0.5, s, 0.1, 2.0 * s].iter().enumerate() {
        let y = 4.0 + 10.0 * k as f32;
        thin.extend(quad(2.0, y + 0.3, 60.0, y + 0.3 + w));
        thin.push([[2.0, y + 4.0], [60.0, y + 4.0 + 3.0], [60.0, y + 4.0 + 3.0 + w]]);
        thin.push([[3.0 + 11.0 * k as f32, 60.0], [3.0 + 11.0 * k as f32 + w, 60.0], [4.0 + 11.0 * k as f32, 112.0]]);
    }
    cases.push(counted("thin", 64, 128, thin));
    let degenerate = vec![
        [[2.0, 2.0], [30.0, 30.0], [60.0, 60.0]],
        [[10.5, 10.5], [10.5, 10.5], [40.5, 20.5]],
        [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
        [[1.5, 40.5], [62.5, 40.5], [30.0, 40.5]],
        [[20.0, 50.0], [20.0 + s * 0.25, 50.0], [20.0, 50.0 + s * 0.25]],
    ];
    cases.push(counted("degenerate", 64, 64, degenerate));
    // large coordinates: the guard band and clipping
    let mut large = Vec::new();
    for &big in &[1e3f32, 1e4, 6e4, 1e6] {
        large.push([[-big, -big * 0.7], [big * 0.9, 13.3], [21.7, big]]);
    }
    cases.push(Case { name: "large".into(), ..counted("", 64, 64, large[..1].to_vec()) });
    for (i, t) in large.iter().enumerate().skip(1) {
        cases.push(Case { name: format!("large {i}"), ..counted("", 64, 64, vec![*t]) });
    }
    let big = 2e5f32;
    let split = vec![[[-big, -big], [big, -big * 0.31], [31.37, 33.11]], [[big, -big * 0.31], [big * 0.7, big], [31.37, 33.11]], [[big * 0.7, big], [-big, -big], [31.37, 33.11]]];
    cases.push(counted("large, shared edges", 64, 64, split));
    cases
}

pub fn state() -> Vec<Case> {
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let mut cases = Vec::new();
    let tris = mesh(&mut rng, 6, 1.0, 63.0, 0.35);
    let window: Vec<[f32; 2]> = tris.iter().flatten().copied().collect();
    // a strip
    let strip: Vec<[f32; 2]> = (0..20).map(|i| [3.0 + 3.1 * i as f32, if i % 2 == 0 { 10.3 } else { 50.7 - i as f32 }]).collect();
    let (mut op, sh) = draw(64, 64, &strip, [ONE; 4], Some(COUNT));
    op.primitive = DrawPrimitive::TriangleStrip;
    cases.push(Case { name: "triangle strip".into(), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(op.clone(), sh.clone())], tolerance: 0 });
    let mut odd = op.clone();
    (odd.first, odd.count) = (3, 11);
    cases.push(Case { name: "strip, first 3 count 11".into(), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(odd, sh)], tolerance: 0 });
    // first and count
    let (op, sh) = draw(64, 64, &window, [ONE; 4], Some(COUNT));
    let mut part = op.clone();
    (part.first, part.count) = (9, 30);
    cases.push(Case { name: "first 9 count 30".into(), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(part, sh.clone())], tolerance: 0 });
    let mut short = op.clone();
    short.count = 32;
    cases.push(Case { name: "count not a multiple of 3".into(), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(short, sh.clone())], tolerance: 0 });
    // culling: half the triangles each way
    let mut flipped = tris.clone();
    for (i, t) in flipped.iter_mut().enumerate() {
        if i % 3 == 0 {
            t.swap(1, 2);
        }
    }
    let fw: Vec<[f32; 2]> = flipped.iter().flatten().copied().collect();
    for cull in [CullMode::Front, CullMode::Back] {
        for ccw in [true, false] {
            let (mut op, sh) = draw(64, 64, &fw, [ONE; 4], Some(COUNT));
            (op.cull, op.front_ccw) = (cull, ccw);
            cases.push(Case { name: format!("cull {cull:?}, front ccw {ccw}"), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(op, sh)], tolerance: 0 });
        }
    }
    // scissors, unaligned, one a draw
    let mut draws = Vec::new();
    for (i, s) in [[0, 0, 64, 64], [3, 5, 7, 9], [17, 1, 1, 1], [63, 63, 1, 1], [0, 60, 64, 4], [31, 0, 2, 64], [5, 33, 50, 3], [40, 40, 100, 100]].iter().enumerate() {
        let (mut op, mut sh) = draw(64, 64, &window, [ONE * (1 + i) as f32; 4], Some(COUNT));
        sh.color = [ONE * (1 + i) as f32; 4];
        op.scissor = Some(*s);
        draws.push((op, sh));
    }
    cases.push(Case { name: "scissors".into(), target: Target::rgba8(64, 64, [0; 4]), draws, tolerance: 0 });
    let mut draws = Vec::new();
    for i in 0..300u32 {
        let (mut op, sh) = draw(64, 64, &window, [ONE; 4], Some(COUNT));
        op.scissor = Some([(i * 7) % 61, (i * 13) % 59, 1 + i % 5, 1 + (i / 5) % 6]);
        draws.push((op, sh));
    }
    cases.push(Case { name: "300 scissors in a frame".into(), target: Target::rgba8(64, 64, [0; 4]), draws, tolerance: 0 });
    // viewports
    for vp in [[8.0, 4.0, 32.0, 32.0], [-16.0, -8.0, 96.0, 80.0], [0.5, 0.25, 48.0, 40.0], [3.0, 7.0, 37.0, 23.0], [10.0, 10.0, 1.0, 1.0]] {
        let (mut op, mut sh) = draw(64, 64, &window, [ONE; 4], Some(COUNT));
        op.viewport = vp;
        sh.positions.iter_mut().for_each(|p| *p = [p[0] * 1.3 - 0.1, p[1] * 1.2 + 0.05, 0.0, 1.0]);
        op.buffers[0].0 = bytes(&sh.positions.iter().flatten().copied().collect::<Vec<_>>());
        cases.push(Case { name: format!("viewport {vp:?}"), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(op, sh)], tolerance: 0 });
    }
    // w other than 1
    let (mut op, mut sh) = draw(64, 64, &window, [ONE; 4], Some(COUNT));
    sh.positions.iter_mut().enumerate().for_each(|(i, p)| {
        let w = 1.0 + (i % 5) as f32 * 0.75;
        *p = [p[0] * w, p[1] * w, 0.0, w];
    });
    op.buffers[0].0 = bytes(&sh.positions.iter().flatten().copied().collect::<Vec<_>>());
    cases.push(Case { name: "w other than 1".into(), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(op, sh)], tolerance: 0 });
    // instancing: the mesh's first row, one instance a step up
    let row: Vec<[f32; 2]> = tris[..12].iter().flatten().copied().collect();
    let (mut op, mut sh) = draw(64, 64, &row, [ONE; 4], Some(COUNT));
    op.program = (INSTANCED_VS.into(), SOLID_FS.into());
    op.instances = 5;
    sh.step = [0.0, 0.3125, 0.0, 0.0];
    op.uniforms.push((0, bytes(&sh.step)));
    cases.push(Case { name: "5 instances".into(), target: Target::rgba8(64, 64, [0; 4]), draws: vec![(op, sh)], tolerance: 0 });
    // the color mask
    let mut draws = Vec::new();
    for m in 0..16u32 {
        let (mut op, sh) = draw(64, 64, &quad(m as f32 * 4.0, 0.0, m as f32 * 4.0 + 4.0, 64.0).concat(), [0.2, 0.4, 0.6, 0.8], None);
        op.mask = core::array::from_fn(|c| m >> c & 1 == 1);
        draws.push((op, sh));
        let (mut op, sh) = draw(64, 64, &quad(m as f32 * 4.0, 32.0, m as f32 * 4.0 + 4.0, 64.0).concat(), [0.1, 0.3, 0.5, 0.5], Some(premultiplied()));
        op.mask = core::array::from_fn(|c| m >> c & 1 == 0);
        draws.push((op, sh));
    }
    cases.push(Case { name: "color masks".into(), target: Target::rgba8(64, 64, [200, 100, 50, 150]), draws, tolerance: 1 });
    // mixed: shapes, scissors, blends and programs interleaved
    let mut draws = Vec::new();
    for i in 0..40u32 {
        let x = (i % 8) as f32 * 8.0;
        let y = (i / 8) as f32 * 12.0;
        let tri = [[x + 0.3, y + 0.7], [x + 9.1, y + 2.2], [x + 3.3, y + 13.9]];
        let color = [0.1 * (i % 7) as f32, 0.05 * (i % 11) as f32, 0.9 - 0.02 * i as f32, 0.25 + 0.015 * i as f32];
        let blend = match i % 3 {
            0 => None,
            1 => Some(premultiplied()),
            _ => Some(Blend { src: [BlendFactor::SrcAlpha, BlendFactor::One], dst: [BlendFactor::OneMinusSrcAlpha; 2], ..COUNT }),
        };
        let (mut op, mut sh) = draw(64, 64, &tri, color, blend);
        if i % 4 == 1 {
            op.scissor = Some([x as u32, y as u32 + 1, 6, 9]);
        }
        if i % 5 == 2 {
            op.program = (INSTANCED_VS.into(), SOLID_FS.into());
            op.instances = 2;
            sh.step = [0.0625, 0.0, 0.0, 0.0];
            op.uniforms.push((0, bytes(&sh.step)));
        }
        draws.push((op, sh));
    }
    cases.push(Case { name: "mixed draws in a frame".into(), target: Target::rgba8(64, 64, [10, 20, 30, 255]), draws, tolerance: 1 });
    let mut linear = cases.last().map(|c| Case { name: "mixed draws, linear target".into(), target: c.target.clone(), draws: c.draws.clone(), tolerance: 1 }).unwrap();
    linear.target.tiling = Tiling::Linear;
    cases.push(linear);
    cases
}

pub fn premultiplied() -> Blend {
    Blend { src: [BlendFactor::One; 2], dst: [BlendFactor::OneMinusSrcAlpha; 2], equation: [BlendEquation::Add; 2], color: [0.0; 4] }
}

/// Every factor pair with every equation, over a target of 256 different texels, for a few source colors
pub fn blends() -> Vec<Case> {
    use BlendFactor as F;
    let factors = [
        F::Zero,
        F::One,
        F::SrcColor,
        F::OneMinusSrcColor,
        F::DstColor,
        F::OneMinusDstColor,
        F::SrcAlpha,
        F::OneMinusSrcAlpha,
        F::DstAlpha,
        F::OneMinusDstAlpha,
        F::ConstantColor,
        F::OneMinusConstantColor,
        F::ConstantAlpha,
        F::OneMinusConstantAlpha,
        F::SrcAlphaSaturate,
    ];
    let sources = [[0.8, 0.3, 0.55, 0.6], [0.0, 1.0, 0.25, 0.0], [0.37, 0.91, 0.13, 1.0]];
    let constant = [0.7, 0.2, 0.45, 0.35];
    let mut cases = Vec::new();
    for eq in [BlendEquation::Add, BlendEquation::Subtract, BlendEquation::ReverseSubtract, BlendEquation::Min, BlendEquation::Max] {
        let pairs: Vec<(F, F)> = if matches!(eq, BlendEquation::Min | BlendEquation::Max) {
            vec![(F::One, F::One), (F::Zero, F::SrcAlpha)]
        } else {
            factors.iter().flat_map(|&s| factors.iter().map(move |&d| (s, d))).collect()
        };
        for (s, d) in pairs {
            let blend = Blend { src: [s, d], dst: [d, s], equation: [eq; 2], color: constant };
            let mut draws = Vec::new();
            for (k, src) in sources.iter().enumerate() {
                let (mut op, sh) = draw(16, 48, &quad(0.0, 16.0 * k as f32, 16.0, 16.0 * (k + 1) as f32).concat(), *src, Some(blend));
                op.viewport = [0.0, 0.0, 16.0, 48.0];
                draws.push((op, sh));
            }
            let pixels = (0..16 * 48u32).flat_map(|i| [i % 256, (i * 7 + 3) % 256, (i * 13 + 5) % 256, (i * 29 + 11) % 256].map(|v| v as u8)).collect();
            cases.push(Case { name: format!("blend {eq:?} {s:?} {d:?}"), target: Target { pixels: Some(pixels), ..Target::rgba8(16, 48, [0; 4]) }, draws, tolerance: 1 });
        }
    }
    cases
}

/// Floats to unorm8 around the halfway points, and out of range: a pixel a draw
pub fn rounding() -> Vec<Case> {
    let mut draws = Vec::new();
    let fracs = [0.5, 0.4999, 0.5001, 0.0];
    let mut put = |x: u32, y: u32, color: [f32; 4]| {
        let (mut op, sh) = draw(64, 8, &quad(x as f32, y as f32, x as f32 + 1.0, y as f32 + 1.0).concat(), color, None);
        op.viewport = [0.0, 0.0, 64.0, 8.0];
        draws.push((op, sh));
    };
    for k in 0..256u32 {
        put(k % 64, k / 64, core::array::from_fn(|c| (k as f32 + fracs[c]) / 255.0));
    }
    let odd = [-0.5, -0.0, 1.5, f32::INFINITY, f32::NEG_INFINITY, 1e-30, 1.0 - 1e-7, 0.5 / 255.0, 1.5 / 255.0, 254.5 / 255.0, 0.1, 1.0 / 3.0];
    for (i, v) in odd.iter().enumerate() {
        put(i as u32, 5, [*v, -*v, *v * 0.5, 1.0 - *v]);
    }
    vec![Case { name: "float to unorm8".into(), target: Target::rgba8(64, 8, [9; 4]), draws, tolerance: 1 }]
}
