//! A raster case run by `examples/support/glref.c` on Mesa's GL ES: the same GPU driven by Mesa

use std::io::Write;
use std::process::{Command, Stdio};

use detour::{BlendEquation, BlendFactor, ColorFormat, CullMode, DrawPrimitive};

use crate::raster::{self, Case};

fn factor(f: BlendFactor) -> u32 {
    match f {
        BlendFactor::Zero => 0,
        BlendFactor::One => 1,
        BlendFactor::SrcColor => 0x300,
        BlendFactor::OneMinusSrcColor => 0x301,
        BlendFactor::SrcAlpha => 0x302,
        BlendFactor::OneMinusSrcAlpha => 0x303,
        BlendFactor::DstAlpha => 0x304,
        BlendFactor::OneMinusDstAlpha => 0x305,
        BlendFactor::DstColor => 0x306,
        BlendFactor::OneMinusDstColor => 0x307,
        BlendFactor::SrcAlphaSaturate => 0x308,
        BlendFactor::ConstantColor => 0x8001,
        BlendFactor::OneMinusConstantColor => 0x8002,
        BlendFactor::ConstantAlpha => 0x8003,
        BlendFactor::OneMinusConstantAlpha => 0x8004,
    }
}

fn equation(e: BlendEquation) -> u32 {
    match e {
        BlendEquation::Add => 0x8006,
        BlendEquation::Min => 0x8007,
        BlendEquation::Max => 0x8008,
        BlendEquation::Subtract => 0x800a,
        BlendEquation::ReverseSubtract => 0x800b,
    }
}

/// The case's script for glref, or None when GL ES can't say it (a fractional viewport, another program)
fn script(c: &Case) -> Option<String> {
    let t = &c.target;
    let mut s = format!("target {} {} {}\n", t.width, t.height, u32::from(t.format == ColorFormat::Rgba32Float));
    let n = (t.width * t.height * t.format.bytes()) as usize;
    let pixels = t.pixels.clone().unwrap_or_else(|| t.clear.repeat(n / t.clear.len()));
    s += &format!("pixels {n} {}\n", pixels.iter().map(u8::to_string).collect::<Vec<_>>().join(" "));
    for (op, sh) in &c.draws {
        if (op.program.0 != raster::SOLID_VS && op.program.0 != raster::INSTANCED_VS) || op.program.1 != raster::SOLID_FS || op.viewport.iter().any(|v| v.fract() != 0.0) {
            return None;
        }
        let v: Vec<String> = sh.positions.iter().flatten().map(|f| format!("{f:e}")).collect();
        s += &format!("draw {} {}\n", sh.positions.len(), v.join(" "));
        let f = |a: &[f32]| a.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" ");
        s += &format!(" color {}\n", f(&sh.color));
        let b = op.blend.unwrap_or_default();
        s += &format!(
            " blend {} {} {} {} {} {} {} {}\n",
            u32::from(op.blend.is_some()),
            factor(b.src[0]),
            factor(b.dst[0]),
            factor(b.src[1]),
            factor(b.dst[1]),
            equation(b.equation[0]),
            equation(b.equation[1]),
            f(&b.color)
        );
        s += &format!(" mask {}\n", op.mask.map(|m| u32::from(m).to_string()).join(" "));
        let sc = op.scissor.unwrap_or([0; 4]);
        s += &format!(" scissor {} {} {} {} {}\n", u32::from(op.scissor.is_some()), sc[0], sc[1], sc[2], sc[3]);
        s += &format!(" viewport {}\n", f(&op.viewport));
        let cull = match op.cull {
            CullMode::None => 0,
            CullMode::Front => 1,
            CullMode::Back => 2,
        };
        s += &format!(" cull {cull} {}\n", u32::from(op.front_ccw));
        let prim = match op.primitive {
            DrawPrimitive::Triangles => 0,
            DrawPrimitive::TriangleStrip => 1,
            _ => return None,
        };
        s += &format!(" prim {prim} {} {} {}\n step {}\n", op.first, op.count, op.instances, f(&sh.step));
    }
    Some(s + "end\n")
}

/// Mesa's pixels for the case (None: not one GL ES can draw)
pub fn run(glref: &str, c: &Case) -> Option<Result<Vec<u8>, String>> {
    let s = script(c)?;
    Some((|| {
        let mut child = Command::new(glref).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
        child.stdin.take().unwrap().write_all(s.as_bytes()).map_err(|e| e.to_string())?;
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        match out.status.success() {
            true => Ok(out.stdout),
            false => Err(String::from_utf8_lossy(&out.stderr).into_owned()),
        }
    })())
}
