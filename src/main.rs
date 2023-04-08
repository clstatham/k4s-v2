use anyhow::Result;

pub mod tests;
pub mod k4s;


fn main() -> Result<()> {
    tests::test_run_1()?;
    Ok(())
}
