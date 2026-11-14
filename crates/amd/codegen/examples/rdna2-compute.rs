//! A compute shader through the RDNA2 backend into an image with its dispatch: `rdna2-compute FILE [GROUPS] [--hex]
//! [--image OUT]`. Prints the module after the whole pipeline and where the kernel is (`--hex`: only the kernel's
//! instruction bytes, as llvm-mc --disassemble reads them; `--image`: the image, `Built::dump`).
//! The shader's parameters are the addresses of two 64 KiB buffers, @src and @dst.

use detour::rdna2_codegen::Codegen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let file = args.get(1).ok_or("usage: rdna2-compute FILE [GROUPS] [--hex] [--image OUT]")?;
    let groups: u32 = args.get(2).and_then(|g| g.parse().ok()).unwrap_or(4);
    let c = Codegen::default();
    let src = std::fs::read_to_string(file)?;
    let built = c.image(&src, groups)?;
    if let Some(out) = args.iter().position(|a| a == "--image").and_then(|i| args.get(i + 1)) {
        std::fs::write(out, built.dump())?;
    }
    if args.iter().any(|a| a == "--hex") {
        println!("{}", built.kernel.iter().map(|b| format!("{b:#04x}")).collect::<Vec<_>>().join(" "));
        return Ok(());
    }
    print!("{}", built.module);
    println!("IB at {:#x} .. {:#x}, kernel at {:#x}, {} bytes of instructions", built.image.stream, built.image.stream_end, built.kernel_addr, built.kernel.len());
    Ok(())
}
