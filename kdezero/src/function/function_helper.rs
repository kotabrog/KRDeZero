use anyhow::Result;
use crate::error::KDeZeroError;
use crate::Variable;

pub fn check_variable_count(xs: &[Variable], n: usize) -> Result<()> {
    if xs.len() != n {
        Err(KDeZeroError::InvalidVariableCount(
            n,
            xs.len(),
        ).into())
    } else {
        Ok(())
    }
}
