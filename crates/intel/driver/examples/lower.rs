use detour::xe3_codegen::Codegen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let src = std::fs::read_to_string(&args[1])?;
    let pipeline = args.get(2).cloned().unwrap_or_else(Codegen::pipeline);
    let c = Codegen::default();
    let body = c.lower(&src, &pipeline).map_err(|e| e.0)?;
    print!("{}", c.0.show(&body));
    Ok(())
}
