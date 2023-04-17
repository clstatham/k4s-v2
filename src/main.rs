use std::{
    fmt::Write,
    fs::File,
    io::{Read, Write as IoWrite},
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
    Run { bins: Vec<PathBuf> },
    /// Debug a compiled k4s binary with an interactive graphical debugger
    Debug { bins: Vec<PathBuf> },
    /// Build *.k4sm assembly and LLVM *.bc bitcode into a compiled k4s binary
    Build {
        #[arg(short, long, action = clap::ArgAction::Count)]
        release: u8,
        #[arg(short, long)]
        output: PathBuf,
        sources: Vec<PathBuf>,
    },
}

pub const FOUR_GIGS: usize = 1024 * 1024 * 1024 * 4;

fn run(bin_paths: Vec<PathBuf>) -> Result<()> {
    let mut programs = vec![];
    for bin_path in bin_paths {
        let mut program = vec![];
        let mut file = File::open(bin_path)?;
        file.read_to_end(&mut program)?;
        programs.push(program);
    }

    let mut machine = MachineContext::new(programs, FOUR_GIGS)?;
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

fn build_asm(paths: Vec<PathBuf>, mut out_path: PathBuf, include_symbols: bool) -> Result<()> {
    let mut asm = String::new();
    for path in paths {
        let mut file = File::open(path)?;
        file.read_to_string(&mut asm)?;
        writeln!(&mut asm)?;
    }
    let mut assembler = AssemblyContext::new(asm);
    log::info!("Assembling {}.", out_path.as_path().to_string_lossy());
    let program = assembler.assemble(include_symbols)?;
    out_path.set_extension("k4s");
    let mut file = File::create(&out_path)?;
    file.write_all(&program)?;
    let mut kept_lines = Vec::new();
    for kept_block in assembler.kept_blocks.iter() {
        for line in kept_block.lines.iter() {
            kept_lines.push(line.asm.display(&assembler.symbols));
        }
    }
    let mut file = File::create(format!("{}.kept_lines.k4sm", out_path.display()))?;

    write!(&mut file, "{}", kept_lines.join("\n"))?;
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
        Commands::Run { bins } => {
            run(bins)?;
        }
        Commands::Debug { bins } => {
            debugger::v2::debugger_main(bins)?;
        }
        Commands::Build {
            sources,
            output,
            release,
        } => {
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
            // if release > 0 {
            build_asm(asm_files, output, false)?;
            // } else {
            //     build_asm(asm_files, output, true)?;
            // }
        }
    }

    Ok(())
}
