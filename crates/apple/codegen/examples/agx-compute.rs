//! A compute shader through the AGX backend into an image with its dispatch: `agx-compute FILE [GROUPS] [--bin OUT]
//! [--image OUT]`. Prints the module at the end; `--bin` writes the kernel's bytes (for Mesa's
//! src/asahi/isa/test/disasm.py), `--image` the image (`Built::dump`). The shader's parameters are the addresses of
//! two 64 KiB buffers, @src and @dst.

use detour::agx_codegen::Codegen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let file = args.get(1).ok_or("usage: agx-compute FILE [GROUPS] [--bin OUT] [--image OUT]")?;
    let groups: u32 = args.get(2).and_then(|g| g.parse().ok()).unwrap_or(4);
    let flag = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1));
    let c = Codegen::default();
    let src = std::fs::read_to_string(file)?;
    let built = c.image(&src, groups)?;
    print!("{}", built.module);
    println!("CDM stream {:#x} .. {:#x}; kernel at {:#x}, {} bytes", built.image.stream, built.image.stream_end, built.kernel_addr, built.kernel.len());
    if let Some(out) = flag("--bin") {
        std::fs::write(out, &built.kernel)?;
    }
    if let Some(out) = flag("--image") {
        std::fs::write(out, built.dump())?;
    }
    Ok(())
}
