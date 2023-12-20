use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum TensorError {
    #[error("Error: {0}")]
    Error(String),
    #[error("ShapeError: data: {0:?}, shape: {1:?}")]
    ShapeError(Vec<usize>, Vec<usize>),
    #[error("ShapeSizeError: data: {0}, shape: {1}")]
    ShapeSizeError(usize, usize),
    #[error("ShapeMismatchError: shape: {0:?}, expected: {1:?}")]
    ShapeMismatchError(Vec<usize>, Vec<usize>),
    #[error("DimensionError: dimension: {0}, expected: {1}")]
    DimensionError(usize, usize),
    #[error("DimensionSmallerError: dimension: {0}, expected: >={1}")]
    DimensionSmallerError(usize, usize),
    #[error("DimensionLargerError: dimension: {0}, expected: <={1}")]
    DimensionLargerError(usize, usize),
    #[error("IndexError: shape: {0:?}, index: {1:?}")]
    IndexError(Vec<usize>, Vec<usize>),
    #[error("CastError: type: {0}")]
    CastError(String),
    #[error("NotScalarError: shape: {0:?}")]
    NotScalarError(Vec<usize>),
    #[error("NotVectorError: shape: {0:?}")]
    NotVectorError(Vec<usize>),
    #[error("InvalidArgumentError: {0}")]
    InvalidArgumentError(String),
    #[error("EmptyTensorError: The tensor is empty.")]
    EmptyTensorError(),
    #[error("NewRandomNormalError: Failed to create a normal distribution.")]
    NewRandomNormalError(),
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use super::*;

    fn error() -> Result<()> {
        Err(TensorError::Error("Error".to_string()).into())
    }

    #[test]
    fn tensor_error() -> Result<()> {
        match error() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "Error: Error");
                Ok(())
            }
        }
    }

    fn error_shape() -> Result<()> {
        Err(TensorError::ShapeError(vec![1, 2], vec![3, 4]).into())
    }

    #[test]
    fn tensor_error_shape() -> Result<()> {
        match error_shape() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "ShapeError: data: [1, 2], shape: [3, 4]");
                Ok(())
            }
        }
    }

    fn error_shape_size() -> Result<()> {
        Err(TensorError::ShapeSizeError(1, 2).into())
    }

    #[test]
    fn tensor_error_shape_size() -> Result<()> {
        match error_shape_size() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "ShapeSizeError: data: 1, shape: 2");
                Ok(())
            }
        }
    }

    fn error_shape_mismatch() -> Result<()> {
        Err(TensorError::ShapeMismatchError(vec![1, 2], vec![3, 4]).into())
    }

    #[test]
    fn tensor_error_shape_mismatch() -> Result<()> {
        match error_shape_mismatch() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(
                    e.to_string(),
                    "ShapeMismatchError: shape: [1, 2], expected: [3, 4]"
                );
                Ok(())
            }
        }
    }

    fn error_dimension() -> Result<()> {
        Err(TensorError::DimensionError(1, 2).into())
    }

    #[test]
    fn tensor_error_dimension() -> Result<()> {
        match error_dimension() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "DimensionError: dimension: 1, expected: 2");
                Ok(())
            }
        }
    }

    fn error_dimension_smaller() -> Result<()> {
        Err(TensorError::DimensionSmallerError(1, 2).into())
    }

    #[test]
    fn tensor_error_dimension_smaller() -> Result<()> {
        match error_dimension_smaller() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(
                    e.to_string(),
                    "DimensionSmallerError: dimension: 1, expected: >=2"
                );
                Ok(())
            }
        }
    }

    fn error_dimension_larger() -> Result<()> {
        Err(TensorError::DimensionLargerError(3, 2).into())
    }

    #[test]
    fn tensor_error_dimension_larger() -> Result<()> {
        match error_dimension_larger() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(
                    e.to_string(),
                    "DimensionLargerError: dimension: 3, expected: <=2"
                );
                Ok(())
            }
        }
    }

    fn error_index() -> Result<()> {
        Err(TensorError::IndexError(vec![1, 2], vec![3, 4]).into())
    }

    #[test]
    fn tensor_error_index() -> Result<()> {
        match error_index() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "IndexError: shape: [1, 2], index: [3, 4]");
                Ok(())
            }
        }
    }

    fn error_cast() -> Result<()> {
        Err(TensorError::CastError("Vec<i32>".to_string()).into())
    }

    #[test]
    fn tensor_error_cast() -> Result<()> {
        match error_cast() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "CastError: type: Vec<i32>");
                Ok(())
            }
        }
    }

    fn error_not_scalar() -> Result<()> {
        Err(TensorError::NotScalarError(vec![1, 2]).into())
    }

    #[test]
    fn tensor_error_not_scalar() -> Result<()> {
        match error_not_scalar() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "NotScalarError: shape: [1, 2]");
                Ok(())
            }
        }
    }

    fn error_not_vector() -> Result<()> {
        Err(TensorError::NotVectorError(vec![1, 2]).into())
    }

    #[test]
    fn tensor_error_not_vector() -> Result<()> {
        match error_not_vector() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "NotVectorError: shape: [1, 2]");
                Ok(())
            }
        }
    }

    fn error_invalid_argument() -> Result<()> {
        Err(TensorError::InvalidArgumentError("Error".to_string()).into())
    }

    #[test]
    fn tensor_error_invalid_argument() -> Result<()> {
        match error_invalid_argument() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "InvalidArgumentError: Error");
                Ok(())
            }
        }
    }

    fn error_empty_tensor() -> Result<()> {
        Err(TensorError::EmptyTensorError().into())
    }

    #[test]
    fn tensor_error_empty_tensor() -> Result<()> {
        match error_empty_tensor() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(
                    e.to_string(),
                    "EmptyTensorError: The tensor is empty."
                );
                Ok(())
            }
        }
    }

    fn error_new_random_normal() -> Result<()> {
        Err(TensorError::NewRandomNormalError().into())
    }

    #[test]
    fn tensor_error_new_random_normal() -> Result<()> {
        match error_new_random_normal() {
            Ok(_) => panic!("error"),
            Err(e) => {
                let e = e.downcast::<TensorError>().context("downcast error")?;
                assert_eq!(e.to_string(), "NewRandomNormalError: Failed to create a normal distribution.");
                Ok(())
            }
        }
    }
}
