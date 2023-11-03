mod info;
mod conversion;
mod operater;

use anyhow::Result;
use ktensor::Tensor;
use crate::error::KDeZeroError;

#[derive(Debug, Clone, PartialEq)]
pub enum VariableData {
    None,
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    USIZE(Tensor<usize>),
    Bool(Tensor<bool>),
}

impl VariableData {
    pub fn ones_like(&self) -> Result<Self> {
        Ok(match self {
            Self::F32(x) => Self::F32(Tensor::ones_like(x)),
            Self::F64(x) => Self::F64(Tensor::ones_like(x)),
            Self::I32(x) => Self::I32(Tensor::ones_like(x)),
            Self::I64(x) => Self::I64(Tensor::ones_like(x)),
            Self::USIZE(x) => Self::USIZE(Tensor::ones_like(x)),
            _ => return Err(KDeZeroError::NotImplementedType(
                self.data_type().to_string(),
                "ones_like".to_string(),
            ).into()),
        })
    }
}

impl std::fmt::Display for VariableData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::F32(x) => write!(f, "{}", x),
            Self::F64(x) => write!(f, "{}", x),
            Self::I32(x) => write!(f, "{}", x),
            Self::I64(x) => write!(f, "{}", x),
            Self::USIZE(x) => write!(f, "{}", x),
            Self::Bool(x) => write!(f, "{}", x),
        }
    }
}
