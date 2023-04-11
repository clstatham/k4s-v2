use std::env::args;

use anyhow::Result;

pub mod debugger;
pub mod k4s;
pub mod tests;

fn main() -> Result<()> {
    env_logger::builder()
        .format_module_path(false)
        .format_target(false)
        .format_timestamp(None)
        .format_level(true)
        .init();
    match args().last().unwrap().as_str() {
        "run" => tests::test_run()?,
        "llvm" => tests::test_llvm()?,
        "dbg" => debugger::debugger_main()?,
        _ => {
            tests::test_llvm()?;
            tests::test_run()?;
        }
    };

    Ok(())
}
