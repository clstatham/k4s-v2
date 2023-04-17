use std::{
    fs::File,
    io::{Read, Write},
};

use anyhow::Result;

use crate::k4s::contexts::{asm::AssemblyContext, llvm::LlvmContext};

pub fn test_assemble() -> Result<()> {
    let mut asm = String::new();
    let mut file = File::open("src/tests/k4sm/test1.k4sm")?;
    file.read_to_string(&mut asm)?;
    let paths = glob::glob("src/tests/rust/target/*.k4sm")?;
    for path in paths {
        let mut file = File::open(path?)?;
        file.read_to_string(&mut asm)?;
    }
    let mut assembler = AssemblyContext::new(asm);
    let program = assembler.assemble(true)?;
    let mut file = File::create("target/test1.k4s")?;
    file.write_all(&program)?;
    Ok(())
}

// pub fn test_run() -> Result<()> {
//     let mut program = vec![];
//     let mut file = File::open("target/test1.k4s")?;
//     file.read_to_end(&mut program)?;
//     let mut machine = MachineContext::new(&program, 0x1000000)?;
//     machine.run_until_hlt()?;
//     Ok(())
// }

// #[test]
pub fn test_llvm() -> Result<()> {
    let paths = glob::glob("src/tests/rust/target/k4s-unknown-none/release/deps/*.bc")?;
    for path in paths {
        let path = path?;
        let mut ctx = LlvmContext::load(path.clone());
        log::info!("Lowering {}.", path.as_path().to_string_lossy());
        let asm = ctx.lower()?;
        // println!("{}", asm);
        let new_path = path.file_name().unwrap().to_string_lossy();
        let new_path = new_path.strip_suffix(".bc").unwrap();
        {
            let mut file = File::create(format!("src/tests/rust/target/{}.k4sm", new_path))?;
            file.write_all(asm.as_bytes())?;
        }
    }

    let mut asm = String::new();
    let mut file = File::open("src/tests/k4sm/test1.k4sm")?;
    file.read_to_string(&mut asm)?;
    let paths = glob::glob("src/tests/rust/target/*.k4sm")?;
    for path in paths {
        let mut file = File::open(path?)?;
        file.read_to_string(&mut asm)?;
    }
    let mut assembler = AssemblyContext::new(asm);
    let program = assembler.assemble(true)?;
    let mut file = File::create("target/test1.k4s")?;
    file.write_all(&program)?;
    Ok(())
}
