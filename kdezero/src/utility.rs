use anyhow::Result;
use crate::error::KDeZeroError;
use crate::Variable;

pub fn check_variable_count(xs: &Vec<&Variable>, n: usize) -> Result<()> {
    if xs.len() != n {
        Err(KDeZeroError::InvalidVariableCount(
            n,
            xs.len(),
        ).into())
    } else {
        Ok(())
    }
}

pub fn check_variable_count_between(xs: &Vec<&Variable>, n: usize, m: usize) -> Result<usize> {
    let len = xs.len();
    if len < n || m <= len {
        Err(KDeZeroError::OutOfRangeVariableCount(
            len,
            n,
            m,
        ).into())
    } else {
        Ok(len)
    }
}

pub fn check_dimensions(x: &Variable, ndim: usize) -> Result<()> {
    if x.ndim() != ndim {
        Err(KDeZeroError::InvalidDimension(
            ndim,
            x.ndim(),
        ).into())
    } else {
        Ok(())
    }
}
