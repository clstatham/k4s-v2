use std::{fs::File, io::Write, process::Command};

use anyhow::Result;

use crate::k4s::contexts::{asm::AssemblyContext, llvm::LlvmContext, machine::MachineContext};

// #[test]
pub fn test_assemble() -> Result<()> {
    let asm = include_str!("k4sm/test1.k4sm");
    let mut assembler = AssemblyContext::new(asm.to_owned());
    assembler.assemble()?;
    Ok(())
}

// #[test]
pub fn test_run() -> Result<()> {
    let asm = include_str!("k4sm/test1.k4sm");
    let mut assembler = AssemblyContext::new(asm.to_owned());
    let program = assembler.assemble()?;
    let mut machine = MachineContext::new(&program, 0x1000000)?;
    machine.run_until_hlt()?;
    Ok(())
}

// #[test]
pub fn test_llvm() -> Result<()> {
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
    let paths = glob::glob("src/tests/rust/target/*.bc")?;
    for path in paths {
        let path = path?;
        let mut ctx = LlvmContext::load(path.clone());
        println!("Lowering {}.", path.as_path().to_string_lossy());
        let asm = ctx.lower()?;
        // println!("{}", asm);
        let new_path = path.file_name().unwrap().to_string_lossy();
        let new_path = new_path.strip_suffix(".bc").unwrap();
        {
            let mut file = File::create(format!("src/tests/rust/target/{}.k4sm", new_path))?;
            file.write_all(asm.as_bytes())?;
        }
    }

    let asm = include_str!("k4sm/test1.k4sm");
    let mut assembler = AssemblyContext::new(asm.to_owned());
    let program = assembler.assemble()?;
    let mut machine = MachineContext::new(&program, 0x1000000)?;
    machine.run_until_hlt()?;
    Ok(())
}
