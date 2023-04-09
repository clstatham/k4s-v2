use anyhow::Result;

pub mod k4s;
pub mod tests;

fn main() -> Result<()> {
    tests::test_run_1()?;
    Ok(())
}
