use std::env::args;

use anyhow::Result;

pub mod k4s;
pub mod tests;

fn main() -> Result<()> {
    match args().last().unwrap().as_str() {
        "run" => tests::test_run()?,
        "llvm" => tests::test_llvm()?,
        _ => {}
    };

    Ok(())
}
