//! `detour`, the debugging tool: one clap subcommand a file under `src/cli/`; this file only puts them together.
//! Built with feature `cli` (and the GPU models' features, to open their devices): `cargo run --features
//! cli,xe3,agx-g13,rdna2 -- devices`.

mod cli;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "detour", about = "detour's debugging tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Devices(cli::Devices),
}

impl Command {
    fn run(&self) -> std::io::Result<()> {
        match self {
            Self::Devices(c) => c.run(),
        }
    }
}

fn main() -> ExitCode {
    match Cli::parse().command.run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("detour: {e}");
            ExitCode::FAILURE
        }
    }
}
