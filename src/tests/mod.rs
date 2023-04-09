use std::{fs::File, io::Write, process::Command};

use anyhow::Result;

use crate::k4s::contexts::{asm::AssemblyContext, llvm::LlvmContext, machine::MachineContext};

// #[test]
pub fn test_assemble_1() -> Result<()> {
    let asm = include_str!("k4sm/test1.k4sm");
    let mut assembler = AssemblyContext::new(asm.to_owned());
    assembler.assemble()?;
    Ok(())
}

// #[test]
pub fn test_run_1() -> Result<()> {
    let asm = include_str!("k4sm/test1.k4sm");
    let mut assembler = AssemblyContext::new(asm.to_owned());
    let program = assembler.assemble()?;
    let mut machine = MachineContext::new(&program, 0x100000)?;
    machine.run_until_hlt()?;
    Ok(())
}

pub fn test_llvm_1() -> Result<()> {
    let status = Command::new("clang")
        .args([
            "-c",
            "src/tests/c/test1.c",
            "-emit-llvm",
            "-o",
            "src/tests/c/test1.bc",
            "-O0",
        ])
        .status()?;
    assert!(status.success());
    let mut ctx = LlvmContext::load("src/tests/c/test1.bc");
    let asm = ctx.lower()?;
    println!("{}", asm);
    {
        let mut file = File::create("src/tests/c/test1.k4sm")?;
        file.write_all(asm.as_bytes())?;
    }

    let asm = include_str!("k4sm/test1.k4sm");
    let mut assembler = AssemblyContext::new(asm.to_owned());
    let program = assembler.assemble()?;
    let mut machine = MachineContext::new(&program, 0x100000)?;
    machine.run_until_hlt()?;
    Ok(())
}
