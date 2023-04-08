use anyhow::Result;

use crate::k4s::contexts::{asm::AssemblyContext, machine::MachineContext};


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