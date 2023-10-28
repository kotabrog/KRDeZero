use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum KDeZeroError {
    #[error("Error: {0}")]
    Error(String),
    #[error("InvalidVariableCount: expected {0}, actual {1}")]
    InvalidVariableCount(usize, usize),
    #[error("NotImplementedType: {0} is not implemented for {1}")]
    NotImplementedType(String, String),
    #[error("NotCollectType: {0} is not collect type. Expected: {1}")]
    NotCollectType(String, String),
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use super::*;

    fn error() -> Result<()> {
        Err(KDeZeroError::Error("Error".to_string()).into())
    }

    #[test]
    fn kdezero_error() -> Result<()> {
        match error() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>().context("downcast error")?;
                assert_eq!(e.to_string(), "Error: Error");
                Ok(())
            }
        }
    }

    fn error_invalid_variable_count() -> Result<()> {
        Err(KDeZeroError::InvalidVariableCount(1, 2).into())
    }

    #[test]
    fn kdezero_error_invalid_variable_count() -> Result<()> {
        match error_invalid_variable_count() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>().context("downcast error")?;
                assert_eq!(e.to_string(), "InvalidVariableCount: expected 1, actual 2");
                Ok(())
            }
        }
    }

    fn error_not_implemented_type() -> Result<()> {
        Err(KDeZeroError::NotImplementedType("square".to_string(), "bool".to_string()).into())
    }

    #[test]
    fn kdezero_error_not_implemented_type() -> Result<()> {
        match error_not_implemented_type() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<KDeZeroError>().context("downcast error")?;
                assert_eq!(e.to_string(), "NotImplementedType: square is not implemented for bool");
                Ok(())
            }
        }
    }
}
