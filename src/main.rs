use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use k4s::contexts::{asm::AssemblyContext, machine::MachineContext};

use crate::k4s::contexts::llvm::LlvmContext;

pub mod debugger;
pub mod k4s;
pub mod tests;

#[derive(Parser)]
#[command(name = "k4s")]
#[command(author = "Connor Statham (@clstatham on github)")]
#[command(version = "0.0.1")]
#[command(about = "Compile and run code for the k4s ISA")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a compiled k4s binary in an emulator
    Run { binary: PathBuf },
    /// Debug a compiled k4s binary with an interactive graphical debugger
    Debug { binary: PathBuf },
    /// Build *.k4sm assembly and LLVM *.bc bitcode into a compiled k4s binary
    Build {
        sources: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
    },
}

fn run(bin_path: PathBuf) -> Result<()> {
    let mut program = vec![];
    let mut file = File::open(bin_path)?;
    file.read_to_end(&mut program)?;
    let mut machine = MachineContext::new(&program, 0x1000000)?;
    machine.run_until_hlt()?;
    Ok(())
}

fn build_llvm(mut src_path: PathBuf) -> Result<Option<PathBuf>> {
    if let Some(ext) = src_path.extension() {
        if &ext.to_string_lossy() == "bc" {
            let mut ctx = LlvmContext::load(src_path.clone());
            log::info!(
                "Lowering {} to k4sm assembly.",
                src_path.file_name().unwrap().to_string_lossy()
            );
            let asm = ctx.lower()?;
            src_path.set_extension("k4sm");
            let mut file = File::create(src_path.clone())?;
            file.write_all(asm.as_bytes())?;
            Ok(Some(src_path))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

fn build_asm(paths: Vec<PathBuf>, mut out_path: PathBuf) -> Result<()> {
    let mut asm = String::new();
    for path in paths {
        let mut file = File::open(path)?;
        file.read_to_string(&mut asm)?;
    }
    let mut assembler = AssemblyContext::new(asm);
    log::info!("Assembling {}.", out_path.as_path().to_string_lossy());
    let program = assembler.assemble()?;
    out_path.set_extension("k4s");
    let mut file = File::create(out_path)?;
    file.write_all(&program)?;
    Ok(())
}

fn main() -> Result<()> {
    env_logger::builder()
        .format_module_path(false)
        .format_target(false)
        .format_timestamp(None)
        .format_level(true)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Run { binary } => {
            run(binary)?;
        }
        Commands::Debug { binary } => {
            debugger::debugger_main(binary)?;
        }
        Commands::Build { sources, output } => {
            let mut asm_files = vec![];
            for source_path in sources {
                if let Ok(paths) = glob::glob(source_path.as_os_str().to_str().ok_or(
                    anyhow::Error::msg(format!("Invalid source path: {:?}", source_path)),
                )?) {
                    for path in paths {
                        let source_path = path?;
                        if let Some(asm_path) = build_llvm(source_path.to_owned())? {
                            asm_files.push(asm_path);
                        } else {
                            asm_files.push(source_path);
                        }
                    }
                } else if let Some(asm_path) = build_llvm(source_path.to_owned())? {
                    asm_files.push(asm_path);
                } else {
                    asm_files.push(source_path);
                }
            }
            build_asm(asm_files, output)?;
        }
    }

    Ok(())
}
